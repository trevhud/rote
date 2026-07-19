"""Run skills and emitted pipelines locally (``rote run``).

The runners layer is detection + ergonomics over machinery that already
exists and is already tested: the skill side reuses the baseline trial
runner (streamed ``claude -p`` with the rote registry injected as MCP
config and the read-only gate applied), and the pipeline side reuses
the empirical trial runner (plain python script, or DBOS app with
cross-process gate signaling via ``DBOSClient``). Nothing in this
module spawns a subprocess of its own — see :mod:`rote.eval.baseline`
and :mod:`rote.eval.empirical` for the execution primitives.
"""

from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from rote.eval.baseline import (
    ObservedToolCall,
    derive_input_payload,
    resolve_mcp_wiring,
    run_baseline_trial,
)
from rote.eval.empirical import MeasuredRun, run_pipeline_trial

__all__ = [
    "RUNNABLE_RUNTIMES",
    "PipelineRunOutcome",
    "RunTarget",
    "SkillRunOutcome",
    "TargetError",
    "detect_target",
    "native_run_hint",
    "parse_signal_args",
    "resolve_gate_signals",
    "resolve_input",
    "run_pipeline",
    "run_skill",
]


class TargetError(RuntimeError):
    """The path cannot be run as given; the message carries the fix."""


#: Adapter names ``rote run`` can execute in-process today. The
#: remaining runtimes (temporal / cloudflare / inngest / dbos-ts) need
#: a dev-server orchestration layer; until that lands we point at the
#: runtime's own dev command instead of pretending.
RUNNABLE_RUNTIMES = frozenset({"python", "dbos"})

_NATIVE_RUN_HINTS = {
    "temporal": "start a Temporal dev server and the emitted worker (see the emitted README.md)",
    "cloudflare": "npx wrangler dev (see the emitted README.md)",
    "inngest": "npx inngest-cli dev + the emitted serve entrypoint (see the emitted README.md)",
    "dbos-ts": "npm install && npm start against a Postgres (see the emitted README.md)",
}


def native_run_hint(runtime: str) -> str:
    """How to run an emitted runtime ``rote run`` does not orchestrate yet."""
    return _NATIVE_RUN_HINTS.get(runtime, "see the emitted README.md")


# ───────── Target detection ─────────


@dataclass(frozen=True)
class RunTarget:
    """What ``rote run <path>`` resolved to."""

    kind: Literal["skill", "pipeline"]
    path: Path
    """The skill directory, or the emitted runtime directory."""
    runtime: str | None = None
    """Adapter name for pipeline targets (``python`` / ``dbos`` / …)."""
    pipeline_yaml: Path | None = None
    """The graduated IR, when it can be located — enables gate-name
    discovery and input derivation from the recorded source skill."""


def _detect_runtime(d: Path) -> str | None:
    """Classify an emitted runtime directory by its marker files.

    Order matters: cloudflare and dbos-ts both carry a ``package.json``,
    and dbos (python) shares ``dbos-config.yaml`` with dbos-ts — the
    discriminators are ``wrangler.jsonc``, ``src/inngest/``, and
    ``main.py`` respectively.
    """
    if (d / "wrangler.jsonc").is_file():
        return "cloudflare"
    if (d / "src" / "inngest").is_dir():
        return "inngest"
    if (d / "dbos-config.yaml").is_file():
        return "dbos" if (d / "main.py").is_file() else "dbos-ts"
    if (d / "workflow.py").is_file() and (d / "activities.py").is_file():
        return "temporal"
    if (d / "main.py").is_file():
        return "python"
    return None


def _find_pipeline_yaml(runtime_dir: Path) -> Path | None:
    """Locate the graduated IR for an emitted runtime directory.

    A ``graduate --out`` layout keeps it at ``../../graduated/`` relative
    to ``runtime/<target>/``; a bare ``rote emit`` output may sit next to
    the pipeline.yaml it was rendered from.
    """
    candidates = [runtime_dir / "pipeline.yaml"]
    if runtime_dir.parent.name == "runtime":
        candidates.append(runtime_dir.parent.parent / "graduated" / "pipeline.yaml")
    candidates.append(runtime_dir.parent / "pipeline.yaml")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def detect_target(path: str | Path, runtime: str | None = None) -> RunTarget:
    """Resolve ``rote run``'s path argument to something executable.

    Accepts a skill directory (``SKILL.md``), an emitted runtime
    directory (marker files), or a ``rote graduate --out`` directory
    (``runtime/<target>/`` children — ``runtime`` disambiguates when
    several were emitted). Raises :class:`TargetError` with actionable
    guidance for everything else.
    """
    p = Path(path).resolve()
    if not p.is_dir():
        raise TargetError(f"{p} is not a directory")
    if (p / "SKILL.md").is_file():
        return RunTarget(kind="skill", path=p)

    if (p / "runtime").is_dir():
        emitted = sorted(
            d for d in (p / "runtime").iterdir() if d.is_dir() and _detect_runtime(d) is not None
        )
        if not emitted:
            raise TargetError(
                f"{p / 'runtime'} contains no emitted runtime — run `rote emit` first"
            )
        if runtime is not None:
            matches = [d for d in emitted if d.name == runtime]
            if not matches:
                raise TargetError(
                    f"no emitted `{runtime}` runtime under {p / 'runtime'} "
                    f"(found: {', '.join(d.name for d in emitted)})"
                )
            chosen = matches[0]
        elif len(emitted) == 1:
            chosen = emitted[0]
        else:
            raise TargetError(
                f"{p / 'runtime'} contains several emitted runtimes "
                f"({', '.join(d.name for d in emitted)}) — pick one with --runtime"
            )
        return RunTarget(
            kind="pipeline",
            path=chosen,
            runtime=chosen.name,
            pipeline_yaml=_find_pipeline_yaml(chosen),
        )

    detected = _detect_runtime(p)
    if detected is None:
        raise TargetError(
            f"{p} is neither a skill (no SKILL.md) nor an emitted runtime "
            "directory — pass a skill dir, an emitted runtime dir, or a "
            "`rote graduate --out` dir"
        )
    if runtime is not None and runtime != detected:
        raise TargetError(f"{p} looks like an emitted `{detected}` runtime, not `{runtime}`")
    return RunTarget(
        kind="pipeline", path=p, runtime=detected, pipeline_yaml=_find_pipeline_yaml(p)
    )


# ───────── Input resolution ─────────


def resolve_input(
    input_arg: str | None, *, skill_dir: Path | None, assume_yes: bool
) -> dict[str, Any] | None:
    """The run's task payload: inline JSON, a file, or derived + confirmed.

    ``input_arg`` may be an inline JSON object (starts with ``{``) or a
    path to a JSON file. With no ``input_arg``, the payload is derived
    from the skill's text (cheap single-shot call) and must be accepted —
    interactively on a terminal, or via ``assume_yes``. The proposal is
    persisted to a temp file so a declined run can be edited and re-run
    with ``--input``. Returns ``None`` when the user declines (callers
    exit 0); raises :class:`TargetError` for usage errors.
    """
    if input_arg is not None:
        stripped = input_arg.strip()
        if stripped.startswith("{"):
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as e:
                raise TargetError(f"--input is not valid JSON: {e}") from e
        else:
            input_path = Path(input_arg)
            if not input_path.is_file():
                raise TargetError(
                    f"--input is neither an inline JSON object nor a file: {input_path}"
                )
            try:
                payload = json.loads(input_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                raise TargetError(f"--input file is not valid JSON: {e}") from e
        if not isinstance(payload, dict):
            raise TargetError("--input must be a JSON object")
        return payload

    if skill_dir is None:
        raise TargetError(
            "no --input given and no source skill to derive one from — pass "
            "--input with an inline JSON object or a file path"
        )
    print(
        "no --input given — deriving a representative task from SKILL.md (cheap single-shot call)…",
        file=sys.stderr,
    )
    derived = derive_input_payload(skill_dir)
    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", prefix="rote-run-input-", delete=False, encoding="utf-8"
    ) as f:
        f.write(json.dumps(derived, indent=2) + "\n")
        derived_path = Path(f.name)

    print("derived input proposal:", file=sys.stderr)
    print(json.dumps(derived, indent=2), file=sys.stderr)
    if assume_yes:
        print("--yes: accepting derived input", file=sys.stderr)
        return derived
    if not sys.stdin.isatty():
        raise TargetError(
            "not a terminal and no --input given — pass --input, or --yes to "
            f"accept the derived proposal (saved to {derived_path})"
        )
    answer = input(f"Run with this input? (saved to {derived_path}) [y/N] ")
    if answer.strip().lower() not in {"y", "yes"}:
        print(
            f"declined — edit {derived_path} and re-run with --input {derived_path}",
            file=sys.stderr,
        )
        return None
    return derived


# ───────── HITL gate signals ─────────


def parse_signal_args(raw: list[str]) -> dict[str, Any]:
    """``--signal name=JSON`` arguments → ``{name: payload}``."""
    signals: dict[str, Any] = {}
    for item in raw:
        name, sep, value = item.partition("=")
        if not sep or not name:
            raise TargetError(f"--signal must be NAME=JSON, got: {item!r}")
        try:
            signals[name] = json.loads(value)
        except json.JSONDecodeError as e:
            raise TargetError(f"--signal {name}: payload is not valid JSON: {e}") from e
    return signals


def resolve_gate_signals(
    gate_signals: list[str], provided: dict[str, Any], *, interactive: bool
) -> dict[str, Any]:
    """Ensure every HITL gate has a resume payload before the run starts.

    DBOS notifications persist per-topic, so payloads collected up front
    are safe to send before the workflow reaches its ``recv()`` — the
    gate picks them up when it parks. Interactively, missing gates are
    prompted for (empty answer = ``{}``); non-interactively they are a
    hard error listing the exact flags to pass, because a gated run with
    no payload coming would park until its IR timeout.
    """
    unknown = sorted(set(provided) - set(gate_signals))
    if unknown:
        raise TargetError(
            f"--signal names not in the pipeline's gates: {', '.join(unknown)} "
            f"(gates: {', '.join(gate_signals) or 'none'})"
        )
    missing = [g for g in gate_signals if g not in provided]
    if not missing:
        return provided
    if not interactive:
        flags = " ".join(f"--signal {g}='{{...}}'" for g in missing)
        raise TargetError(
            f"this pipeline has HITL gate(s) with no resume payload: "
            f"{', '.join(missing)} — pass {flags} (payloads are delivered "
            "when the workflow parks)"
        )
    resolved = dict(provided)
    for gate in missing:
        answer = input(f"HITL gate '{gate}' — resume payload JSON (empty for {{}}): ").strip()
        if not answer:
            resolved[gate] = {}
            continue
        try:
            resolved[gate] = json.loads(answer)
        except json.JSONDecodeError as e:
            raise TargetError(f"gate '{gate}': payload is not valid JSON: {e}") from e
    return resolved


# ───────── Execution ─────────


@dataclass(frozen=True)
class SkillRunOutcome:
    """One ``rote run`` of a raw skill."""

    run: MeasuredRun
    observations: tuple[ObservedToolCall, ...]
    servers_wired: tuple[str, ...]
    servers_skipped: dict[str, str]
    read_only: bool

    @property
    def observed_servers(self) -> list[str]:
        return sorted({o.server for o in self.observations})


def run_skill(
    skill_dir: Path,
    input_payload: dict[str, Any],
    *,
    model: str,
    allow_writes: bool = False,
    max_turns: int = 60,
    timeout_seconds: float = 1800.0,
    executable: str = "claude",
) -> SkillRunOutcome:
    """Run the raw skill once as an agent, same rules as the baseline.

    Registry-wide MCP injection, read-only gate unless ``allow_writes``,
    subscription billing (API-key env vars scrubbed). The observed MCP
    traffic rides along — it's free and it tells the user what the run
    actually touched.
    """
    servers, mcp_tool_ids, skipped = resolve_mcp_wiring(allow_writes=allow_writes)
    run, observed, _lines = run_baseline_trial(
        skill_dir,
        input_payload,
        model=model,
        max_turns=max_turns,
        mcp_tool_ids=mcp_tool_ids,
        executable=executable,
        timeout_seconds=timeout_seconds,
        mcp_servers=servers or None,
    )
    return SkillRunOutcome(
        run=run,
        observations=tuple(observed),
        servers_wired=tuple(sorted(servers)),
        servers_skipped=skipped,
        read_only=not allow_writes,
    )


@dataclass(frozen=True)
class PipelineRunOutcome:
    """One ``rote run`` of an emitted pipeline."""

    run: MeasuredRun
    runtime: str
    app_dir: Path


def run_pipeline(
    target: RunTarget,
    input_payload: dict[str, Any],
    *,
    signals: dict[str, Any] | None = None,
    timeout_seconds: float = 600.0,
) -> PipelineRunOutcome:
    """Run an emitted pipeline once (python script or DBOS app)."""
    if target.runtime not in RUNNABLE_RUNTIMES:
        raise TargetError(
            f"`rote run` cannot orchestrate the `{target.runtime}` runtime yet — "
            f"{native_run_hint(target.runtime or '')}"
        )
    run = run_pipeline_trial(
        target.path,
        input_payload,
        signals=signals or None,
        timeout_seconds=timeout_seconds,
    )
    return PipelineRunOutcome(run=run, runtime=target.runtime, app_dir=target.path)
