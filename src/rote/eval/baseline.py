"""Pre-compilation baseline: run the raw skill as an agent, measured *and observed*.

``rote eval --run`` measures a skill *after* compilation, to compare both
sides. The baseline runs **before** any compilation: one instrumented
``claude -p`` execution of the raw skill that produces three artifacts at
once —

1. **Measured "before" metrics** — wall clock, turns, tokens, cost — so
   the eventual scorecard's before-side is a measurement, not an estimate.
2. **Observed MCP tool traffic** — every ``mcp__<server>__<tool>`` call
   the agent actually made, with its real input and result payloads. This
   is ground truth for requirement detection (which servers does this
   skill *really* use?) and for schema/type inference downstream.
3. **The full stream-json transcript**, persisted per trial for later
   analysis.

Auth is one world: the rote registry + token store are injected into the
child as ``--mcp-config`` (strict), so ``rote mcp login`` covers the
baseline exactly like it covers emitted pipelines.

Side effects default to a **read-only gate**: only MCP tools whose
server-declared ``readOnlyHint`` annotation is true are allowlisted.
``allow_writes=True`` lifts the gate (``mcp__<server>__*``) for skills
whose writes the user has said are acceptable to fire once.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rote.compiler.drivers.claude import DEFAULT_ALLOWED_TOOLS
from rote.eval.empirical import (
    MCP_CONFIG_FILENAME,
    RESULT_FILENAME,
    EmpiricalError,
    MeasuredRun,
    _reliability_flags,
    _skill_prompt,
    measured_run_record,
)
from rote.inference import build_subscription_env

BASELINE_DIRNAME = "baseline"
METRICS_FILENAME = "metrics.json"
OBSERVED_TOOLS_FILENAME = "observed-tools.json"
DERIVED_INPUT_FILENAME = "derived-input.json"
INFERRED_SCHEMAS_FILENAME = "inferred-schemas.json"

#: Model for input derivation — a cheap, single-shot reading task.
DERIVE_MODEL = "claude-haiku-4-5"


# ───────── Input derivation ─────────


def derive_input_payload(
    skill_dir: str | Path,
    *,
    model: str = DERIVE_MODEL,
    executable: str = "claude",
    timeout_seconds: float = 180.0,
) -> dict[str, Any]:
    """Propose one representative input payload for a skill, from its text.

    A cheap single-shot ``claude -p`` call (no tools, cents) reads the
    SKILL.md and answers with a JSON object. The caller MUST confirm the
    proposal with the user before spending real money on a baseline run —
    a derived input is a guess about how the skill is actually invoked.
    """
    skill_path = Path(skill_dir).resolve()
    skill_md_path = skill_path / "SKILL.md"
    if not skill_md_path.is_file():
        raise EmpiricalError(f"{skill_path} does not contain a SKILL.md")
    if shutil.which(executable) is None:
        raise EmpiricalError(
            f"`{executable}` CLI not found — install Claude Code or pass a different executable"
        )
    skill_md = skill_md_path.read_text(encoding="utf-8")
    prompt = (
        "Read this skill definition and produce ONE representative input "
        "payload for invoking it — the JSON object a user would supply for "
        "a single, typical run. Prefer values the skill's own examples "
        "mention; otherwise invent plausible, obviously-sample values.\n\n"
        "<skill>\n"
        f"{skill_md}\n"
        "</skill>\n\n"
        "Answer with ONLY the JSON object — no prose, no markdown fences."
    )
    try:
        proc = subprocess.run(
            [
                executable,
                "-p",
                prompt,
                "--model",
                model,
                "--allowedTools",
                "",
                "--output-format",
                "json",
                "--max-turns",
                "2",
            ],
            env=build_subscription_env(),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as e:
        raise EmpiricalError(f"input derivation timed out after {timeout_seconds:g}s") from e
    if proc.returncode != 0:
        raise EmpiricalError(
            f"input derivation failed: claude exited {proc.returncode}: "
            f"{(proc.stderr or proc.stdout)[:300]}"
        )
    try:
        meta = json.loads(proc.stdout)
        text = meta.get("result") or ""
    except json.JSONDecodeError as e:
        raise EmpiricalError(f"input derivation: could not parse claude output: {e}") from e
    # Models occasionally fence anyway; strip a ```json wrapper before parsing.
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as e:
        raise EmpiricalError(
            f"input derivation: the model did not answer with a JSON object: {text[:300]!r}"
        ) from e
    if not isinstance(payload, dict):
        raise EmpiricalError(
            f"input derivation: expected a JSON object, got {type(payload).__name__}"
        )
    return payload


# ───────── MCP wiring (whole registry — no pipeline exists yet) ─────────


def mcp_servers_from_registry() -> dict[str, dict[str, Any]]:
    """Every rote-registered server as a ``claude -p`` mcp-config entry.

    Pre-compilation there is no pipeline to name the skill's servers, so
    the baseline wires *all* registered servers and lets the observed
    traffic report which ones the skill actually used — that observation
    is the requirements ground truth, so over-wiring here is a feature,
    not sloppiness. Resolution and auth follow the same rules as
    :func:`rote.eval.empirical.mcp_servers_for_pipeline`: static headers
    verbatim, logged-in servers via a ``headersHelper`` that refreshes
    tokens mid-run.
    """
    import shlex
    import sys

    from rote.eval.empirical import _TRANSPORT_TO_MCP_TYPE
    from rote.mcp import access_token_state, load_registry, read_token_file

    registry = load_registry()
    servers: dict[str, dict[str, Any]] = {}
    for name, entry in sorted(registry.servers.items()):
        config: dict[str, Any] = {
            "type": _TRANSPORT_TO_MCP_TYPE[entry.transport],
            "url": entry.url,
        }
        if entry.headers:
            config["headers"] = dict(entry.headers)
        else:
            doc = read_token_file(name)
            if doc is not None and access_token_state(doc)[0] is not None:
                config["headersHelper"] = (
                    f"{shlex.quote(sys.executable)} -m rote mcp headers {shlex.quote(name)}"
                )
        servers[name] = config
    return servers


async def read_only_allowlist(
    server_names: list[str],
) -> tuple[list[str], dict[str, str]]:
    """Per-server allowlist of tools safe to fire without ``--allow-writes``.

    Connects to each server (same non-interactive auth resolution as
    emitted workflow code) and allows exactly the tools whose
    ``readOnlyHint`` annotation is true. Returns ``(allowed, skipped)``:
    ``allowed`` as ``mcp__<server>__<tool>`` ids, and ``skipped`` mapping
    each server that contributed nothing to the reason (unreachable, not
    authenticated, or simply declaring no read-only tools) — callers
    surface these as non-blocking recommendations.
    """
    from rote.mcp._runtime_helper import RoteMcpAuthNeeded, mcp_client

    allowed: list[str] = []
    skipped: dict[str, str] = {}
    for server in server_names:
        try:
            client = mcp_client(server, None)
            async with client:
                tools = await client.list_tools()
        except RoteMcpAuthNeeded:
            skipped[server] = "not authenticated — rote mcp login " + server
            continue
        except Exception as e:  # noqa: BLE001 — a dead server must not sink the baseline
            skipped[server] = f"unreachable ({type(e).__name__}: {e})"
            continue
        read_only = [
            t.name
            for t in tools
            if t.annotations is not None and t.annotations.readOnlyHint is True
        ]
        if not read_only:
            skipped[server] = (
                f"no tools declare readOnlyHint ({len(tools)} tools total) — "
                "pass --allow-writes to wire this server"
            )
            continue
        allowed.extend(f"mcp__{server}__{name}" for name in sorted(read_only))
    return allowed, skipped


def resolve_mcp_wiring(
    *, allow_writes: bool
) -> tuple[dict[str, dict[str, Any]], list[str], dict[str, str]]:
    """The full MCP wiring decision for one ``claude -p`` skill run.

    Returns ``(servers, allowed_tool_ids, skipped)``: the registry-wide
    mcp-config entries, the tool allowlist (read-only gate applied unless
    ``allow_writes``), and per-server reasons for contributing nothing.
    Shared by the baseline and ``rote run`` so both make identical
    auth/side-effect decisions.
    """
    import asyncio

    servers = mcp_servers_from_registry()
    skipped: dict[str, str] = {}
    mcp_tool_ids: list[str] = []
    if servers:
        if allow_writes:
            mcp_tool_ids = [f"mcp__{name}__*" for name in sorted(servers)]
        else:
            mcp_tool_ids, skipped = asyncio.run(read_only_allowlist(sorted(servers)))
            # A server contributing zero allowed tools stays wired (its
            # tools just aren't callable) — cheaper than rewriting config,
            # and the strictness lives in --allowedTools anyway.
    return servers, mcp_tool_ids, skipped


# ───────── Transcript observation (pure, testable) ─────────


@dataclass(frozen=True)
class ObservedToolCall:
    """One MCP tool invocation the agent actually made during a trial."""

    server: str
    tool: str
    input: dict[str, Any]
    result: Any | None = None
    is_error: bool = False


def extract_observations(lines: list[str]) -> list[ObservedToolCall]:
    """Pull MCP tool calls (with results paired by id) from stream-json.

    Only ``mcp__<server>__<tool>`` invocations are observed — local tools
    (Read/Write/…) are the agent's scaffolding, not the skill's external
    contract. Malformed lines are skipped: the transcript is diagnostic
    input, never a validation target.
    """
    pending: dict[str, tuple[str, str, dict[str, Any]]] = {}  # id -> (server, tool, input)
    observed: list[ObservedToolCall] = []
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        message = event.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if event.get("type") == "assistant" and block.get("type") == "tool_use":
                name = block.get("name") or ""
                if not name.startswith("mcp__"):
                    continue
                parts = name.split("__", 2)
                if len(parts) != 3:
                    continue
                tool_input = block.get("input")
                pending[str(block.get("id"))] = (
                    parts[1],
                    parts[2],
                    tool_input if isinstance(tool_input, dict) else {},
                )
            elif event.get("type") == "user" and block.get("type") == "tool_result":
                key = str(block.get("tool_use_id"))
                if key not in pending:
                    continue
                server, tool, tool_input = pending.pop(key)
                observed.append(
                    ObservedToolCall(
                        server=server,
                        tool=tool,
                        input=tool_input,
                        result=block.get("content"),
                        is_error=bool(block.get("is_error")),
                    )
                )
    # Calls that never got a result line (interrupt, max-turns cut) still
    # prove the skill reaches for the tool — record them without a result.
    for server, tool, tool_input in pending.values():
        observed.append(ObservedToolCall(server=server, tool=tool, input=tool_input))
    return observed


# ───────── One streamed trial ─────────


def run_baseline_trial(
    skill_dir: str | Path,
    input_payload: dict[str, Any],
    *,
    model: str,
    max_turns: int = 60,
    allowed_tools: str = DEFAULT_ALLOWED_TOOLS,
    mcp_tool_ids: list[str] | None = None,
    executable: str = "claude",
    timeout_seconds: float = 1800.0,
    mcp_servers: dict[str, dict[str, Any]] | None = None,
    min_expected_turns: int | None = None,
) -> tuple[MeasuredRun, list[ObservedToolCall], list[str]]:
    """One ``claude -p`` run of the raw skill in stream-json mode.

    Identical billing/metric semantics to
    :func:`rote.eval.empirical.run_skill_trial` (API keys scrubbed, the
    final ``result`` event carries usage/cost/turns) — but streamed, so
    every tool_use/tool_result pair is captured. Returns
    ``(measured_run, observations, transcript_lines)``.

    ``mcp_tool_ids`` is the explicit tool allowlist (read-only gate);
    pass ``["mcp__<server>__*" ...]`` wildcards to lift the gate.
    """
    skill_path = Path(skill_dir).resolve()
    if not (skill_path / "SKILL.md").is_file():
        raise EmpiricalError(f"{skill_path} does not contain a SKILL.md")
    if shutil.which(executable) is None:
        raise EmpiricalError(
            f"`{executable}` CLI not found — install Claude Code or pass a different executable"
        )

    prompt = _skill_prompt(skill_path, input_payload, None)
    with tempfile.TemporaryDirectory(prefix="rote-baseline-") as workdir_str:
        workdir = Path(workdir_str)
        if mcp_tool_ids:
            allowed_tools = ",".join([allowed_tools, *mcp_tool_ids])
        args = [
            executable,
            "-p",
            prompt,
            "--model",
            model,
            "--add-dir",
            str(skill_path),
            "--allowedTools",
            allowed_tools,
            "--output-format",
            "stream-json",
            "--verbose",
            "--max-turns",
            str(max_turns),
        ]
        if mcp_servers:
            mcp_config_path = workdir / MCP_CONFIG_FILENAME
            mcp_config_path.write_text(
                json.dumps({"mcpServers": mcp_servers}, indent=2), encoding="utf-8"
            )
            args += ["--mcp-config", str(mcp_config_path), "--strict-mcp-config"]
        started = time.monotonic()
        try:
            proc = subprocess.run(
                args,
                cwd=workdir,
                env=build_subscription_env(),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as e:
            stdout = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            lines = stdout.splitlines()
            return (
                MeasuredRun(
                    wall_seconds=time.monotonic() - started,
                    output=None,
                    error=f"timed out after {timeout_seconds:g}s",
                    model=model,
                    flags=_reliability_flags(
                        error="timeout",
                        subtype=None,
                        turns=None,
                        min_expected_turns=min_expected_turns,
                        missing_mcp_servers=None,
                    ),
                ),
                extract_observations(lines),
                lines,
            )
        wall = time.monotonic() - started

        result_path = workdir / RESULT_FILENAME
        output: dict[str, Any] | None = None
        parse_error: str | None = f"agent did not write {RESULT_FILENAME}"
        if result_path.is_file():
            try:
                loaded = json.loads(result_path.read_text(encoding="utf-8"))
                output = loaded if isinstance(loaded, dict) else {"result": loaded}
                parse_error = None
            except json.JSONDecodeError as e:
                parse_error = f"{RESULT_FILENAME} was not valid JSON: {e}"

    lines = proc.stdout.splitlines()
    meta: dict[str, Any] = {}
    for line in reversed(lines):
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and candidate.get("type") == "result":
            meta = candidate
            break

    usage = meta.get("usage") or {}
    error: str | None = None
    if proc.returncode != 0 and output is None:
        error = f"claude exited {proc.returncode}: {(proc.stderr or proc.stdout)[:500]}"
    elif output is None:
        error = parse_error
    turns = meta.get("num_turns")
    run = MeasuredRun(
        wall_seconds=wall,
        output=output,
        error=error,
        turns=turns,
        cost_usd=meta.get("total_cost_usd"),
        input_tokens=usage.get("input_tokens"),
        cache_read_tokens=usage.get("cache_read_input_tokens"),
        cache_creation_tokens=usage.get("cache_creation_input_tokens"),
        output_tokens=usage.get("output_tokens"),
        model=model,
        flags=_reliability_flags(
            error=error,
            subtype=meta.get("subtype"),
            turns=turns,
            min_expected_turns=min_expected_turns,
            missing_mcp_servers=None,
        ),
    )
    return run, extract_observations(lines), lines


# ───────── Orchestration + artifacts ─────────


@dataclass(frozen=True)
class BaselineResult:
    """One `rote baseline` invocation: N trials of the raw skill."""

    skill_dir: Path
    input_payload: dict[str, Any]
    model: str
    read_only: bool
    runs: tuple[MeasuredRun, ...]
    observations: tuple[ObservedToolCall, ...] = field(default_factory=tuple)
    servers_wired: tuple[str, ...] = ()
    servers_skipped: dict[str, str] = field(default_factory=dict)

    @property
    def observed_servers(self) -> list[str]:
        """Servers the agent actually called — the requirements ground truth."""
        return sorted({o.server for o in self.observations})


def run_baseline(
    skill_dir: str | Path,
    input_payload: dict[str, Any],
    out_dir: str | Path,
    *,
    model: str,
    trials: int = 1,
    allow_writes: bool = False,
    max_turns: int = 60,
    executable: str = "claude",
    timeout_seconds: float = 1800.0,
) -> BaselineResult:
    """Run the baseline and persist its artifacts under ``out_dir``/baseline.

    Artifacts: ``metrics.json`` (measured runs + wiring report),
    ``observed-tools.json`` (deduplicated by call — full payloads), and
    one ``trial-<n>.transcript.jsonl`` per trial.
    """
    servers, mcp_tool_ids, skipped = resolve_mcp_wiring(allow_writes=allow_writes)

    baseline_dir = Path(out_dir) / BASELINE_DIRNAME
    baseline_dir.mkdir(parents=True, exist_ok=True)

    runs: list[MeasuredRun] = []
    observations: list[ObservedToolCall] = []
    for n in range(1, trials + 1):
        run, observed, lines = run_baseline_trial(
            skill_dir,
            input_payload,
            model=model,
            max_turns=max_turns,
            mcp_tool_ids=mcp_tool_ids,
            executable=executable,
            timeout_seconds=timeout_seconds,
            mcp_servers=servers or None,
        )
        (baseline_dir / f"trial-{n}.transcript.jsonl").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        runs.append(run)
        observations.extend(observed)

    result = BaselineResult(
        skill_dir=Path(skill_dir).resolve(),
        input_payload=input_payload,
        model=model,
        read_only=not allow_writes,
        runs=tuple(runs),
        observations=tuple(observations),
        servers_wired=tuple(sorted(servers)),
        servers_skipped=skipped,
    )

    (baseline_dir / METRICS_FILENAME).write_text(
        json.dumps(
            {
                "skill_dir": str(result.skill_dir),
                "input": result.input_payload,
                "model": result.model,
                "read_only": result.read_only,
                "trials": len(result.runs),
                "runs": [measured_run_record(r) for r in result.runs],
                "servers_wired": list(result.servers_wired),
                "servers_skipped": result.servers_skipped,
                "observed_servers": result.observed_servers,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (baseline_dir / OBSERVED_TOOLS_FILENAME).write_text(
        json.dumps(
            [
                {
                    "server": o.server,
                    "tool": o.tool,
                    "input": o.input,
                    "result": o.result,
                    "is_error": o.is_error,
                }
                for o in result.observations
            ],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if result.observations:
        # Local import: rote.probe imports ObservedToolCall from this
        # module at import time, so the dependency must stay one-way here.
        from rote.probe import infer_tool_schemas

        (baseline_dir / INFERRED_SCHEMAS_FILENAME).write_text(
            json.dumps(
                [s.as_dict() for s in infer_tool_schemas(list(result.observations))],
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return result


def load_observations(baseline_dir: str | Path) -> list[ObservedToolCall]:
    """Read a persisted baseline's observed MCP calls back into objects.

    Returns an empty list when no baseline artifacts exist — callers use
    this to make baseline-informed features opt-in by presence.
    """
    path = Path(baseline_dir) / OBSERVED_TOOLS_FILENAME
    if not path.is_file():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [ObservedToolCall(**entry) for entry in raw]


def render_baseline_markdown(result: BaselineResult) -> str:
    """The scorecard's measured-baseline section (the skill side, measured).

    Appended after the static estimate when a baseline ran alongside a
    compilation — the "before" column stops being a model and becomes data.
    """
    ok = [r for r in result.runs if r.succeeded]
    lines = [
        f"## Measured baseline ({len(ok)}/{len(result.runs)} "
        f"trial{'s' if len(result.runs) != 1 else ''} succeeded)",
        "",
        f"Raw skill executed as an agent (`{result.model}`), "
        + ("read-only MCP gate." if result.read_only else "writes allowed."),
        "",
        "| Metric | Measured |",
        "|---|---|",
    ]

    def _fmt(values: list[float], fmt: str) -> str:
        if not values:
            return "—"
        if len(values) == 1:
            return format(values[0], fmt)
        lo, hi = min(values), max(values)
        return f"{format(lo, fmt)}–{format(hi, fmt)}"

    lines.append(f"| Wall clock (s) | {_fmt([r.wall_seconds for r in ok], '.0f')} |")
    lines.append(f"| Turns | {_fmt([float(r.turns) for r in ok if r.turns], '.0f')} |")
    lines.append(
        f"| Cost (USD) | {_fmt([r.cost_usd for r in ok if r.cost_usd is not None], '.2f')} |"
    )
    if result.observations:
        by_server: dict[str, int] = {}
        for o in result.observations:
            by_server[o.server] = by_server.get(o.server, 0) + 1
        traffic = ", ".join(f"{s} ×{n}" for s, n in sorted(by_server.items()))
        lines.append(f"| Observed MCP calls | {traffic} |")
    flagged = [r for r in result.runs if r.flags]
    if flagged:
        notes = sorted({f for r in flagged for f in r.flags})
        lines.append(f"| Reliability flags | {', '.join(notes)} |")
    lines.append("")
    return "\n".join(lines)
