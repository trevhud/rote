"""Shared filesystem tool surface for the in-process compiler drivers.

Both :class:`~rote.compiler.drivers.anthropic_api.AnthropicApiDriver`
and :class:`~rote.compiler.drivers.openai_api.OpenAIApiDriver` run the
compiler agent in an in-process tool-use loop over the same three
tools — ``read_file`` / ``list_directory`` / ``write_file`` — with the
same path jailing, read caps, system prompt, and ``progress.ndjson``
phase interception. Only the *wire shape* of the tool declarations and
the message/usage plumbing differ between the Anthropic Messages API
and the OpenAI chat/completions API, so everything format-agnostic lives
here and each driver owns just its own protocol glue.

Path traversal is blocked at the tool level: every path is resolved to
an absolute form and checked against the list of allowed roots via
``Path.relative_to``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rote.compiler.events import (
    PROGRESS_FILENAME,
    EventCallback,
    emit_safely,
    parse_progress_lines,
)

# ───────── Loop limits ─────────

#: Hard cap on a single ``read_file`` response. Source skills can have
#: surprisingly large reference files; we don't want a single tool call
#: to blow the context window.
MAX_FILE_READ_BYTES = 200_000

#: Turn-event message cap. The assistant's text can be long; the live
#: progress line only needs a glimpse of what it's thinking.
TURN_SNIPPET_CHARS = 120


# ───────── Tool schema definitions (wire-shape-neutral) ─────────

#: ``(name, description, json-schema-of-params)`` for each tool. Both
#: shape adapters below derive from this single list so the Anthropic and
#: OpenAI tool declarations can never drift apart.
_TOOL_DEFS: list[tuple[str, str, dict[str, Any]]] = [
    (
        "read_file",
        (
            "Read a file from the source skill or the rote-compile skill. "
            "Returns the file contents as text. Use this to read SKILL.md, "
            "references/*.md, and any other source skill files. The path "
            "must be absolute and inside one of the allowed read roots "
            "(communicated in the system prompt)."
        ),
        {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to a file inside an allowed read root.",
                }
            },
            "required": ["path"],
        },
    ),
    (
        "list_directory",
        (
            "List entries in a directory (one per line, with a trailing "
            "slash on directories). Use this to discover the structure of "
            "the source skill (e.g., what's in references/)."
        ),
        {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to a directory inside an allowed read root.",
                }
            },
            "required": ["path"],
        },
    ),
    (
        "write_file",
        (
            "Write a file into the work directory. Use this to produce "
            "the final pipeline.yaml deliverable, plus any extracted "
            "Python modules or signature stubs. The path must be absolute "
            "and inside the work directory."
        ),
        {
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
    ),
]


def anthropic_tool_schemas() -> list[dict[str, Any]]:
    """The three tools in Anthropic Messages API shape."""
    return [
        {"name": name, "description": desc, "input_schema": schema}
        for name, desc, schema in _TOOL_DEFS
    ]


def openai_tool_schemas() -> list[dict[str, Any]]:
    """The three tools in OpenAI chat/completions function-calling shape."""
    return [
        {"type": "function", "function": {"name": name, "description": desc, "parameters": schema}}
        for name, desc, schema in _TOOL_DEFS
    ]


# ───────── Tool implementations (security-sensitive) ─────────


def path_within(candidate: Path, root: Path) -> bool:
    """Return True if ``candidate`` resolves inside ``root`` (or equals it).

    Both paths are resolved before comparison so symlinks and ``..``
    segments are followed. Used as the gate for every file tool call.
    """
    try:
        candidate.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def check_read_path(path_str: str, read_roots: list[Path]) -> Path:
    p = Path(path_str)
    for root in read_roots:
        if path_within(p, root):
            return p
    roots_str = ", ".join(str(r) for r in read_roots)
    raise PermissionError(f"Path {path_str!r} is not within allowed read roots: {roots_str}")


def check_write_path(path_str: str, write_root: Path) -> Path:
    p = Path(path_str)
    if not path_within(p, write_root):
        raise PermissionError(f"Path {path_str!r} is not within allowed write root: {write_root}")
    return p


def handle_read_file(path_str: str, read_roots: list[Path]) -> str:
    p = check_read_path(path_str, read_roots)
    if not p.is_file():
        raise FileNotFoundError(f"Not a file: {path_str}")
    data = p.read_bytes()
    if len(data) > MAX_FILE_READ_BYTES:
        raise ValueError(
            f"File too large ({len(data)} bytes > limit {MAX_FILE_READ_BYTES}). "
            f"Consider using a different approach (e.g., grep + targeted reads)."
        )
    return data.decode("utf-8", errors="replace")


def handle_list_directory(path_str: str, read_roots: list[Path]) -> str:
    p = check_read_path(path_str, read_roots)
    if not p.is_dir():
        raise NotADirectoryError(f"Not a directory: {path_str}")
    entries = sorted(f"{entry.name}{'/' if entry.is_dir() else ''}" for entry in p.iterdir())
    return "\n".join(entries) if entries else "(empty)"


def handle_write_file(path_str: str, content: str, write_root: Path) -> str:
    p = check_write_path(path_str, write_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} bytes to {path_str}"


def dispatch_tool(
    name: str,
    args: dict[str, Any],
    read_roots: list[Path],
    write_root: Path,
) -> str:
    """Route a decoded tool call to its handler and return the result text.

    Raises on any failure (unknown tool, missing argument, jailed path,
    oversized read) — the caller catches and reports it back to the model
    as an ``is_error`` tool result rather than crashing the loop.
    """
    if name == "read_file":
        return handle_read_file(args["path"], read_roots)
    if name == "list_directory":
        return handle_list_directory(args["path"], read_roots)
    if name == "write_file":
        return handle_write_file(args["path"], args["content"], write_root)
    raise ValueError(f"Unknown tool: {name}")


# ───────── progress.ndjson phase interception ─────────


def emit_progress_phases(
    path_str: str,
    content: str,
    work_dir: Path,
    on_event: EventCallback | None,
    already_emitted: int,
) -> int:
    """Fire phase events for new lines in an agent's progress.ndjson write.

    Returns the running count of phase lines seen, so the caller only
    fires events for lines beyond what a previous rewrite already
    reported (the agent rewrites the whole file each phase). A write to
    any other file, or one that escaped the work dir, is a no-op.
    """
    try:
        target = Path(path_str).resolve()
        target.relative_to(work_dir)
    except ValueError:
        return already_emitted
    if target.name != PROGRESS_FILENAME:
        return already_emitted
    events = parse_progress_lines(content)
    for event in events[already_emitted:]:
        emit_safely(on_event, event)
    return len(events)


# ───────── System prompt assembly ─────────


def build_system_prompt(
    skill_dir: Path,
    compiler_skill_dir: Path,
    work_dir: Path,
    skill_md_text: str,
) -> str:
    return f"""You are the rote compiler. Your job is to read a source AI skill
bundle and produce a runnable, deterministic pipeline IR (`pipeline.yaml`)
plus any extracted Python modules and typed signature stubs the IR refers to.

Available paths for this run:

- Source skill (read-only):  {skill_dir}
- Rote-compile rubric (read-only):  {compiler_skill_dir}
- Work directory (write):  {work_dir}

You have three tools:

- `read_file(path)` — read any file in the source skill or rubric directory
- `list_directory(path)` — list entries in an allowed directory
- `write_file(path, content)` — write a file into the work directory only

The rote-compile skill below tells you the procedure to follow. Reference
files for each phase live at:

- {compiler_skill_dir}/references/node-kinds.md
- {compiler_skill_dir}/references/crystallization-heuristics.md
- {compiler_skill_dir}/references/ir-schema.md
- {compiler_skill_dir}/references/llm-judge-extraction.md

Read each one when the SKILL.md instructs you to.

Your final deliverable is `{work_dir}/pipeline.yaml`. Once you have written
it (and any extracted modules / signatures it references), end your turn.
Do not call any further tools after writing the final pipeline.yaml.

==================== ROTE COMPILE SKILL ====================
{skill_md_text}
"""
