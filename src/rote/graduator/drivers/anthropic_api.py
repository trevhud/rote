"""Anthropic API driver — in-process tool-use loop via the bare SDK.

For users who prefer (or are required to use) Anthropic API-key auth.
Runs the graduator agent in-process using the ``anthropic`` Python
SDK's Messages API with a minimal filesystem tool surface.

# Why the bare SDK and not ``claude-agent-sdk``?

Anthropic's terms of service explicitly forbid third-party agents
built on the Claude Agent SDK from using claude.ai login credentials
without prior approval. That's a policy blocker, not a technical
limitation. Since we already have a separate subscription path via
the ``claude`` driver (which spawns Claude Code directly), depending
on ``claude-agent-sdk`` would only add weight without adding value.

# Optional dependency

The ``anthropic`` package is an optional extra. Users who only use
the subprocess drivers (``claude`` / ``codex``) don't pay the import
cost::

    pip install "rote[api]"

# Filesystem tool surface

The driver implements three minimal tools for the LLM:

* ``read_file(path)`` — scoped to ``skill_dir`` and
  ``graduator_skill_dir`` (read-only).
* ``list_directory(path)`` — read-only listing in the same scope.
* ``write_file(path, content)`` — scoped to ``work_dir`` (write-only).

This is the same effective capability set the subprocess drivers give
their CLIs via ``--add-dir`` and ``--allowedTools``, just implemented
as a small Python shim instead of deferring to a heavier runtime.

Path traversal is blocked at the tool level: every path is resolved
to an absolute form and then checked against the list of allowed
roots via ``Path.relative_to``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from rote.graduator.drivers import DriverError, DriverResult, GraduatorDriver

# Detect the optional dep at import time so is_available() can report
# a specific "pip install rote[api]" message instead of a cryptic
# ImportError.
try:
    import anthropic  # noqa: F401

    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False


# ───────── Defaults ─────────

DEFAULT_MODEL = "claude-sonnet-4-6"
"""Default model for the in-process graduator loop.

Matches the ``ClaudeDriver`` default for consistency: Sonnet 4.6
follows structured rubrics reliably and is ~5× cheaper than Opus.
Users with complex skills can override via
``AnthropicApiDriver(model="claude-opus-4-6")`` or the CLI's
``--model`` flag.
"""

DEFAULT_MAX_ITERATIONS = 60
DEFAULT_MAX_TOKENS_PER_TURN = 8192

#: Hard cap on a single ``read_file`` response. Source skills can have
#: surprisingly large reference files; we don't want a single tool call
#: to blow the context window.
MAX_FILE_READ_BYTES = 200_000


# ───────── Tool schemas ─────────


def _build_tool_schemas() -> list[dict[str, Any]]:
    return [
        {
            "name": "read_file",
            "description": (
                "Read a file from the source skill or the rote-graduate skill. "
                "Returns the file contents as text. Use this to read SKILL.md, "
                "references/*.md, and any other source skill files. The path "
                "must be absolute and inside one of the allowed read roots "
                "(communicated in the system prompt)."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to a file inside an allowed read root.",
                    }
                },
                "required": ["path"],
            },
        },
        {
            "name": "list_directory",
            "description": (
                "List entries in a directory (one per line, with a trailing "
                "slash on directories). Use this to discover the structure of "
                "the source skill (e.g., what's in references/)."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to a directory inside an allowed read root.",
                    }
                },
                "required": ["path"],
            },
        },
        {
            "name": "write_file",
            "description": (
                "Write a file into the work directory. Use this to produce "
                "the final pipeline.yaml deliverable, plus any extracted "
                "Python modules or signature stubs. The path must be absolute "
                "and inside the work directory."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path inside the work directory.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full file contents to write (overwrites existing).",
                    },
                },
                "required": ["path", "content"],
            },
        },
    ]


# ───────── Tool implementations (security-sensitive) ─────────


def _path_within(candidate: Path, root: Path) -> bool:
    """Return True if ``candidate`` resolves inside ``root`` (or equals it).

    Both paths are resolved before comparison so symlinks and ``..``
    segments are followed. Used as the gate for every file tool call.
    """
    try:
        candidate.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _check_read_path(path_str: str, read_roots: list[Path]) -> Path:
    p = Path(path_str)
    for root in read_roots:
        if _path_within(p, root):
            return p
    roots_str = ", ".join(str(r) for r in read_roots)
    raise PermissionError(
        f"Path {path_str!r} is not within allowed read roots: {roots_str}"
    )


def _check_write_path(path_str: str, write_root: Path) -> Path:
    p = Path(path_str)
    if not _path_within(p, write_root):
        raise PermissionError(
            f"Path {path_str!r} is not within allowed write root: {write_root}"
        )
    return p


def _handle_read_file(path_str: str, read_roots: list[Path]) -> str:
    p = _check_read_path(path_str, read_roots)
    if not p.is_file():
        raise FileNotFoundError(f"Not a file: {path_str}")
    data = p.read_bytes()
    if len(data) > MAX_FILE_READ_BYTES:
        raise ValueError(
            f"File too large ({len(data)} bytes > limit {MAX_FILE_READ_BYTES}). "
            f"Consider using a different approach (e.g., grep + targeted reads)."
        )
    return data.decode("utf-8", errors="replace")


def _handle_list_directory(path_str: str, read_roots: list[Path]) -> str:
    p = _check_read_path(path_str, read_roots)
    if not p.is_dir():
        raise NotADirectoryError(f"Not a directory: {path_str}")
    entries = sorted(
        f"{entry.name}{'/' if entry.is_dir() else ''}"
        for entry in p.iterdir()
    )
    return "\n".join(entries) if entries else "(empty)"


def _handle_write_file(path_str: str, content: str, write_root: Path) -> str:
    p = _check_write_path(path_str, write_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} bytes to {path_str}"


# ───────── System prompt assembly ─────────


def _build_system_prompt(
    skill_dir: Path,
    graduator_skill_dir: Path,
    work_dir: Path,
    skill_md_text: str,
) -> str:
    return f"""You are the rote graduator. Your job is to read a source AI skill
bundle and produce a runnable, deterministic pipeline IR (`pipeline.yaml`)
plus any extracted Python modules and typed signature stubs the IR refers to.

Available paths for this run:

- Source skill (read-only):  {skill_dir}
- Rote-graduate rubric (read-only):  {graduator_skill_dir}
- Work directory (write):  {work_dir}

You have three tools:

- `read_file(path)` — read any file in the source skill or rubric directory
- `list_directory(path)` — list entries in an allowed directory
- `write_file(path, content)` — write a file into the work directory only

The rote-graduate skill below tells you the procedure to follow. Reference
files for each phase live at:

- {graduator_skill_dir}/references/node-kinds.md
- {graduator_skill_dir}/references/crystallization-heuristics.md
- {graduator_skill_dir}/references/ir-schema.md
- {graduator_skill_dir}/references/llm-judge-extraction.md

Read each one when the SKILL.md instructs you to.

Your final deliverable is `{work_dir}/pipeline.yaml`. Once you have written
it (and any extracted modules / signatures it references), end your turn.
Do not call any further tools after writing the final pipeline.yaml.

==================== ROTE GRADUATE SKILL ====================
{skill_md_text}
"""


# ───────── The driver ─────────


class AnthropicApiDriver(GraduatorDriver):
    name: str = "api"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        max_tokens_per_turn: int = DEFAULT_MAX_TOKENS_PER_TURN,
    ) -> None:
        self.model = model
        self.max_iterations = max_iterations
        self.max_tokens_per_turn = max_tokens_per_turn

    def is_available(self) -> tuple[bool, str]:
        if not _ANTHROPIC_AVAILABLE:
            return (
                False,
                "The `anthropic` package is not installed. "
                "Install the optional extra with: `pip install rote[api]`.",
            )
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return (
                False,
                "The ANTHROPIC_API_KEY environment variable is not set. "
                "Export it or use the `claude` / `codex` driver to authenticate "
                "via your existing Claude Max/Pro or ChatGPT subscription.",
            )
        return (True, "")

    async def run(
        self,
        skill_dir: Path,
        graduator_skill_dir: Path,
        work_dir: Path,
    ) -> DriverResult:
        if not _ANTHROPIC_AVAILABLE:
            raise DriverError(
                "anthropic package is not installed. "
                "Run: pip install rote[api]"
            )

        skill_dir = skill_dir.resolve()
        graduator_skill_dir = graduator_skill_dir.resolve()
        work_dir = work_dir.resolve()
        work_dir.mkdir(parents=True, exist_ok=True)

        skill_md = graduator_skill_dir / "SKILL.md"
        if not skill_md.is_file():
            raise DriverError(
                f"rote-graduate SKILL.md not found at {skill_md}. "
                f"Pass an explicit graduator_skill_dir to the orchestrator."
            )
        skill_md_text = skill_md.read_text(encoding="utf-8")

        read_roots = [skill_dir, graduator_skill_dir]
        write_root = work_dir

        system_prompt = _build_system_prompt(
            skill_dir, graduator_skill_dir, work_dir, skill_md_text
        )

        client = anthropic.AsyncAnthropic()
        tool_schemas = _build_tool_schemas()

        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": (
                    f"Graduate the skill at {skill_dir}. "
                    f"Begin by reading {skill_dir}/SKILL.md and the reference "
                    f"files in {graduator_skill_dir}/references/, then produce "
                    f"the pipeline.yaml at {work_dir}/pipeline.yaml."
                ),
            }
        ]

        total_input_tokens = 0
        total_output_tokens = 0
        last_text = ""
        completed_iterations = 0

        for iteration in range(1, self.max_iterations + 1):
            completed_iterations = iteration

            response = await client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens_per_turn,
                system=system_prompt,
                tools=tool_schemas,
                messages=messages,
            )

            usage = getattr(response, "usage", None)
            if usage is not None:
                total_input_tokens += getattr(usage, "input_tokens", 0) or 0
                total_output_tokens += getattr(usage, "output_tokens", 0) or 0

            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                # Capture the final text for diagnostics
                for block in response.content:
                    if getattr(block, "type", None) == "text":
                        last_text = getattr(block, "text", "") or last_text
                break

            # Dispatch tool calls
            tool_results: list[dict[str, Any]] = []
            for block in response.content:
                if getattr(block, "type", None) != "tool_use":
                    continue

                tool_name = block.name
                tool_input = block.input or {}
                try:
                    if tool_name == "read_file":
                        result_text = _handle_read_file(
                            tool_input["path"], read_roots
                        )
                    elif tool_name == "list_directory":
                        result_text = _handle_list_directory(
                            tool_input["path"], read_roots
                        )
                    elif tool_name == "write_file":
                        result_text = _handle_write_file(
                            tool_input["path"],
                            tool_input["content"],
                            write_root,
                        )
                    else:
                        raise ValueError(f"Unknown tool: {tool_name}")
                    is_error = False
                except Exception as e:
                    result_text = f"Error: {type(e).__name__}: {e}"
                    is_error = True

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_text,
                        "is_error": is_error,
                    }
                )

            messages.append({"role": "user", "content": tool_results})
        else:
            raise DriverError(
                f"Agent did not complete within {self.max_iterations} iterations.",
                details=f"Last assistant text: {last_text!r}",
            )

        pipeline_yaml = work_dir / "pipeline.yaml"
        if not pipeline_yaml.is_file():
            raise DriverError(
                f"Agent finished but did not produce {pipeline_yaml}.",
                details=f"Last assistant text: {last_text!r}",
            )

        return DriverResult(
            pipeline_yaml_path=pipeline_yaml,
            work_dir=work_dir,
            driver_name=self.name,
            metadata={
                "model": self.model,
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "iterations": completed_iterations,
            },
        )
