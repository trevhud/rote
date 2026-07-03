"""Empirical eval mode: run both sides for real and measure.

Phase 1 (:mod:`rote.eval.estimate`) predicts; this module verifies.
``rote eval --run`` executes the source skill as raw agent
instructions K times (via ``claude -p``, subscription-billed, same
env rules as :class:`rote.graduator.drivers.claude.ClaudeDriver`) and
the emitted pipeline K times (python-adapter script or DBOS app), then
reports measured wall clock, token usage, notional cost, and output
agreement across trials.

Measurement honesty rules, same spirit as the static side:

* The skill side's ``total_cost_usd`` is Claude Code's list-price
  notional even on subscription auth — labeled as such.
* The pipeline side's LLM usage is *measured*, not estimated: emitted
  signature modules append real per-call token usage to
  ``$ROTE_USAGE_LOG`` (see ``rote.adapters._py_common``).
* Agreement is exact-match on canonicalized JSON — no LLM judging the
  LLM in v1. Divergent runs are reported per-field, not averaged away.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

from rote.eval.pricing import PricingCatalog
from rote.graduator.drivers.claude import (
    DEFAULT_ALLOWED_TOOLS,
    build_subscription_env,
)

RESULT_FILENAME = "result.json"
DEFAULT_CORPUS_PATH = Path.home() / ".local" / "share" / "rote" / "eval-corpus.jsonl"


class EmpiricalError(RuntimeError):
    """A trial could not be started (misconfiguration, missing deps)."""


@dataclass(frozen=True)
class MeasuredRun:
    """One executed trial, either side."""

    wall_seconds: float
    output: dict[str, Any] | None
    error: str | None = None
    # Agent (skill) side:
    turns: int | None = None
    cost_usd: float | None = None
    input_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_creation_tokens: int | None = None
    output_tokens: int | None = None
    model: str | None = None
    # Pipeline side (from the emitted $ROTE_USAGE_LOG hook):
    judge_usage: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    @property
    def succeeded(self) -> bool:
        return self.error is None


# ───────── Skill (before) trials ─────────


def _skill_prompt(
    skill_dir: Path, input_payload: dict[str, Any], output_fields: list[str] | None
) -> str:
    skill_md = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
    fields_clause = (
        " containing at least the fields: " + ", ".join(f"`{f}`" for f in output_fields)
        if output_fields
        else ""
    )
    return (
        "You are executing a skill exactly as written, as a background "
        "job. Do not ask the user anything; make reasonable choices and "
        "proceed.\n\n"
        "The skill definition:\n\n"
        "<skill>\n"
        f"{skill_md}\n"
        "</skill>\n\n"
        f"Reference files live under {skill_dir} — read them with the "
        "Read tool exactly when the skill directs you to.\n\n"
        "Execute the skill now for this input:\n\n"
        f"```json\n{json.dumps(input_payload, indent=2)}\n```\n\n"
        "When the skill is complete, write your final structured result "
        f"as a single JSON object to a file named {RESULT_FILENAME} in "
        f"the current working directory{fields_clause}. Write nothing "
        "else to that file."
    )


def run_skill_trial(
    skill_dir: str | Path,
    input_payload: dict[str, Any],
    *,
    model: str,
    max_turns: int = 60,
    allowed_tools: str = DEFAULT_ALLOWED_TOOLS,
    executable: str = "claude",
    timeout_seconds: float = 1800.0,
    output_fields: list[str] | None = None,
) -> MeasuredRun:
    """Run the raw skill once under ``claude -p`` and measure it.

    Billing follows the ClaudeDriver rules: API-key env vars are
    scrubbed so the run uses the subscription OAuth session, and the
    reported ``total_cost_usd`` is Claude Code's list-price notional.
    """
    skill_path = Path(skill_dir).resolve()
    if not (skill_path / "SKILL.md").is_file():
        raise EmpiricalError(f"{skill_path} does not contain a SKILL.md")
    if shutil.which(executable) is None:
        raise EmpiricalError(
            f"`{executable}` CLI not found — install Claude Code or pass a different executable"
        )

    prompt = _skill_prompt(skill_path, input_payload, output_fields)
    with tempfile.TemporaryDirectory(prefix="rote-eval-skill-") as workdir_str:
        workdir = Path(workdir_str)
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
            "json",
            "--max-turns",
            str(max_turns),
        ]
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
        except subprocess.TimeoutExpired:
            return MeasuredRun(
                wall_seconds=time.monotonic() - started,
                output=None,
                error=f"timed out after {timeout_seconds:g}s",
                model=model,
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

    meta: dict[str, Any] = {}
    # A malformed stdout costs us metadata, not the trial itself —
    # result.json is the deliverable.
    with contextlib.suppress(json.JSONDecodeError):
        meta = json.loads(proc.stdout)

    usage = meta.get("usage") or {}
    error: str | None = None
    if proc.returncode != 0 and output is None:
        error = f"claude exited {proc.returncode}: {(proc.stderr or proc.stdout)[:500]}"
    elif output is None:
        error = parse_error
    return MeasuredRun(
        wall_seconds=wall,
        output=output,
        error=error,
        turns=meta.get("num_turns"),
        cost_usd=meta.get("total_cost_usd"),
        input_tokens=usage.get("input_tokens"),
        cache_read_tokens=usage.get("cache_read_input_tokens"),
        cache_creation_tokens=usage.get("cache_creation_input_tokens"),
        output_tokens=usage.get("output_tokens"),
        model=model,
    )


# ───────── Pipeline (after) trials ─────────


def _dbos_system_database_url(app_dir: Path) -> str:
    """The system database URL the spawned DBOS app will actually use.

    Must mirror the emitted main.py's resolution order exactly —
    ``DBOS_SYSTEM_DATABASE_URL`` env override first, then the default
    SQLite file in the app dir — or signals get delivered to a
    database the app isn't watching.
    """
    override = os.environ.get("DBOS_SYSTEM_DATABASE_URL")
    if override:
        return override
    config = yaml.safe_load((app_dir / "dbos-config.yaml").read_text(encoding="utf-8"))
    name = config["name"] if isinstance(config, dict) and "name" in config else "pipeline"
    return f"sqlite:///{(app_dir / f'{name}.dbos.sqlite').resolve()}"


def _wait_for_workflow_id(stderr_path: Path, proc: subprocess.Popen[Any], deadline: float) -> str:
    """Poll the app's stderr file for the ``workflow started: <id>`` line."""
    marker = "workflow started: "
    while time.monotonic() < deadline:
        text = stderr_path.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            if line.startswith(marker):
                return line[len(marker) :].strip()
        if proc.poll() is not None:
            raise EmpiricalError(
                f"DBOS app exited (code {proc.returncode}) before starting "
                f"a workflow:\n{text[:800]}"
            )
        time.sleep(0.2)
    raise EmpiricalError("timed out waiting for the DBOS app to report a workflow id")


def _send_dbos_signals(system_database_url: str, workflow_id: str, signals: dict[str, Any]) -> None:
    try:
        from dbos import DBOSClient
    except ImportError as e:
        raise EmpiricalError(
            "signaling a gated pipeline requires the dbos extra: pip install 'rote-cli[dbos]'"
        ) from e
    client = DBOSClient(system_database_url=system_database_url)
    # DBOS notifications persist per-topic, so sending before the
    # workflow reaches its recv() is safe — the gate picks the payload
    # up when it parks (empirically validated in tests/test_dbos_e2e.py).
    for signal, payload in signals.items():
        client.send(workflow_id, payload, topic=signal)


def run_pipeline_trial(
    app_dir: str | Path,
    input_payload: dict[str, Any],
    *,
    signals: dict[str, Any] | None = None,
    python_executable: str | None = None,
    timeout_seconds: float = 600.0,
) -> MeasuredRun:
    """Run the emitted pipeline once and measure it.

    ``app_dir`` is an emitted runtime directory: the python adapter's
    plain script or a DBOS app (detected by ``dbos-config.yaml``). For
    gated DBOS pipelines, ``signals`` maps each gate's signal name to
    its resume payload — delivered cross-process via ``DBOSClient``
    against the app's SQLite system database. Real judge usage is
    captured via the emitted ``$ROTE_USAGE_LOG`` hook.
    """
    app_path = Path(app_dir).resolve()
    main_py = app_path / "main.py"
    if not main_py.is_file():
        raise EmpiricalError(f"{app_path} has no main.py — pass an emitted runtime directory")
    python = python_executable or sys.executable
    is_dbos = (app_path / "dbos-config.yaml").is_file()
    if signals and not is_dbos:
        raise EmpiricalError(
            "signals were provided but the app is not a DBOS app — the plain "
            "python runtime cannot park on HITL gates"
        )

    with tempfile.TemporaryDirectory(prefix="rote-eval-pipeline-") as workdir_str:
        workdir = Path(workdir_str)
        usage_log = workdir / "usage.jsonl"
        # Unlike the skill side, the pipeline subprocess keeps the full
        # environment: emitted judges construct anthropic.Anthropic() /
        # openai.OpenAI() directly and NEED their API keys. Only the
        # `claude -p` spawn scrubs keys (subscription billing).
        env = os.environ.copy()
        env["ROTE_USAGE_LOG"] = str(usage_log)
        args = [python, str(main_py), json.dumps(input_payload)]

        stdout_path = workdir / "stdout.txt"
        stderr_path = workdir / "stderr.txt"
        started = time.monotonic()
        with (
            stdout_path.open("w", encoding="utf-8") as out_f,
            stderr_path.open("w", encoding="utf-8") as err_f,
        ):
            proc = subprocess.Popen(
                args, cwd=app_path, env=env, stdout=out_f, stderr=err_f, text=True
            )
            try:
                if is_dbos and signals:
                    workflow_id = _wait_for_workflow_id(
                        stderr_path, proc, deadline=started + min(60.0, timeout_seconds)
                    )
                    _send_dbos_signals(_dbos_system_database_url(app_path), workflow_id, signals)
                proc.wait(timeout=timeout_seconds - (time.monotonic() - started))
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                return MeasuredRun(
                    wall_seconds=time.monotonic() - started,
                    output=None,
                    error=f"timed out after {timeout_seconds:g}s",
                )
            except BaseException:
                # Signal delivery / id-harvest failures must not orphan
                # the child — a gated DBOS app parked on recv() with no
                # signal coming would otherwise live until its IR gate
                # timeout (hours to days).
                proc.kill()
                proc.wait()
                raise
        wall = time.monotonic() - started

        stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
        stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
        usage_rows: list[dict[str, Any]] = []
        if usage_log.is_file():
            for line in usage_log.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                # One torn/malformed line (e.g. two judges racing the
                # append) costs us that row, not the whole trial.
                with contextlib.suppress(json.JSONDecodeError):
                    usage_rows.append(json.loads(line))

    if proc.returncode != 0:
        return MeasuredRun(
            wall_seconds=wall,
            output=None,
            error=f"pipeline exited {proc.returncode}: {(stderr or stdout)[:800]}",
            judge_usage=tuple(usage_rows),
        )
    try:
        parsed = json.loads(stdout)
        output = parsed if isinstance(parsed, dict) else {"result": parsed}
        error = None
    except json.JSONDecodeError:
        output = None
        error = f"pipeline stdout was not JSON: {stdout[:300]}"
    return MeasuredRun(wall_seconds=wall, output=output, error=error, judge_usage=tuple(usage_rows))


# ───────── Agreement ─────────


@dataclass(frozen=True)
class Agreement:
    """Exact-match agreement across a trial set's outputs."""

    total: int
    successful: int
    modal_count: int
    """How many successful runs produced the single most common output."""
    field_agreement: dict[str, float]
    """Per top-level field: fraction of successful runs matching that
    field's most common value (missing counts as its own value)."""

    @property
    def identical_fraction(self) -> float:
        return self.modal_count / self.successful if self.successful else 0.0


def compute_agreement(outputs: list[dict[str, Any] | None]) -> Agreement:
    present = [o for o in outputs if o is not None]
    if not present:
        return Agreement(total=len(outputs), successful=0, modal_count=0, field_agreement={})
    canonical = [json.dumps(o, sort_keys=True, default=str) for o in present]
    modal_count = max(Counter(canonical).values())

    fields_union: set[str] = set()
    for o in present:
        fields_union.update(o.keys())
    field_agreement: dict[str, float] = {}
    for key in sorted(fields_union):
        values = [
            json.dumps(o.get(key, "__rote_missing__"), sort_keys=True, default=str) for o in present
        ]
        field_agreement[key] = max(Counter(values).values()) / len(present)
    return Agreement(
        total=len(outputs),
        successful=len(present),
        modal_count=modal_count,
        field_agreement=field_agreement,
    )


# ───────── Aggregation + rendering ─────────


@dataclass(frozen=True)
class EmpiricalResult:
    trials: int
    skill_runs: tuple[MeasuredRun, ...]
    pipeline_runs: tuple[MeasuredRun, ...]
    skill_model: str | None

    @property
    def skill_agreement(self) -> Agreement:
        return compute_agreement([r.output for r in self.skill_runs])

    @property
    def pipeline_agreement(self) -> Agreement:
        return compute_agreement([r.output for r in self.pipeline_runs])


def measured_pipeline_cost_usd(
    runs: tuple[MeasuredRun, ...], catalog: PricingCatalog
) -> tuple[float, list[str]]:
    """Price the measured judge usage at live rates.

    Returns (mean cost per run in USD, list of model ids that had no
    price in the catalog — their tokens are excluded and flagged).
    """
    unpriced: set[str] = set()
    per_run: list[float] = []
    for run in runs:
        total = 0.0
        for row in run.judge_usage:
            model = str(row.get("model", ""))
            price = catalog.price_for(model)
            if price is None:
                unpriced.add(model)
                continue
            in_rate, out_rate = price
            total += (row.get("input_tokens") or 0) * in_rate / 1_000_000
            total += (row.get("output_tokens") or 0) * out_rate / 1_000_000
        per_run.append(total)
    return (mean(per_run) if per_run else 0.0, sorted(unpriced))


def suggested_priors(skill_runs: tuple[MeasuredRun, ...]) -> dict[str, float]:
    """Prior values re-fitted from measured agent runs (reported, never
    silently applied — the static model's constants stay explicit)."""
    fitted: dict[str, float] = {}
    timed = [r for r in skill_runs if r.turns and r.wall_seconds]
    if timed:
        fitted["seconds_per_turn"] = round(
            mean(r.wall_seconds / r.turns for r in timed if r.turns), 2
        )
    with_output = [r for r in skill_runs if r.turns and r.output_tokens]
    if with_output:
        fitted["output_tokens_per_turn"] = round(
            mean(r.output_tokens / r.turns for r in with_output if r.turns and r.output_tokens),
            1,
        )
    return fitted


def append_corpus(result: EmpiricalResult, *, generated_at: str, path: Path | None = None) -> Path:
    """Append this measurement to the local calibration corpus (JSONL)."""
    corpus_path = path or DEFAULT_CORPUS_PATH
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "generated_at": generated_at,
        "trials": result.trials,
        "skill_model": result.skill_model,
        "skill_runs": [
            {
                "wall_seconds": r.wall_seconds,
                "turns": r.turns,
                "cost_usd": r.cost_usd,
                "input_tokens": r.input_tokens,
                "cache_read_tokens": r.cache_read_tokens,
                "cache_creation_tokens": r.cache_creation_tokens,
                "output_tokens": r.output_tokens,
                "succeeded": r.succeeded,
            }
            for r in result.skill_runs
        ],
        "pipeline_runs": [
            {
                "wall_seconds": r.wall_seconds,
                "judge_usage": list(r.judge_usage),
                "succeeded": r.succeeded,
            }
            for r in result.pipeline_runs
        ],
    }
    with corpus_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return corpus_path


def _mean_range(values: list[float]) -> str:
    if not values:
        return "—"
    if len(values) == 1:
        return f"{values[0]:.1f}"
    return f"{mean(values):.1f} (min {min(values):.1f}, max {max(values):.1f})"


def render_measured_markdown(result: EmpiricalResult, catalog: PricingCatalog) -> str:
    """The scorecard's measured section, appended after the static one."""
    lines = [f"## Measured ({result.trials} trial{'s' if result.trials != 1 else ''} per side)", ""]

    lines.append("| Metric | As agent instructions | As graduated pipeline |")
    lines.append("|---|---|---|")
    skill_ok = [r for r in result.skill_runs if r.succeeded]
    pipe_ok = [r for r in result.pipeline_runs if r.succeeded]
    lines.append(
        f"| Successful runs | {len(skill_ok)}/{len(result.skill_runs)} | "
        f"{len(pipe_ok)}/{len(result.pipeline_runs)} |"
    )
    lines.append(
        "| Wall clock (s) | "
        f"{_mean_range([r.wall_seconds for r in skill_ok])} | "
        f"{_mean_range([r.wall_seconds for r in pipe_ok])} |"
    )
    skill_costs = [r.cost_usd for r in skill_ok if r.cost_usd is not None]
    pipe_cost, unpriced = measured_pipeline_cost_usd(tuple(pipe_ok), catalog)
    lines.append(
        "| Cost per run (USD) | "
        + (f"{mean(skill_costs):.3f} (list-price notional)" if skill_costs else "—")
        + f" | {pipe_cost:.4f} (measured judge usage, live prices) |"
    )
    skill_turns = [float(r.turns) for r in skill_ok if r.turns]
    lines.append(f"| Agent turns | {_mean_range(skill_turns)} | — |")
    sa, pa = result.skill_agreement, result.pipeline_agreement
    lines.append(
        "| Identical outputs | "
        f"{sa.modal_count}/{sa.successful} | {pa.modal_count}/{pa.successful} |"
    )
    lines.append("")

    diverging = {k: v for k, v in sa.field_agreement.items() if v < 1.0}
    if diverging:
        lines.append("Agent-side fields that diverged across runs:")
        for key, frac in sorted(diverging.items(), key=lambda kv: kv[1]):
            lines.append(f"- `{key}`: {frac:.0%} of runs agree")
        lines.append("")
    p_diverging = {k: v for k, v in pa.field_agreement.items() if v < 1.0}
    if p_diverging:
        lines.append("Pipeline-side fields that diverged across runs:")
        for key, frac in sorted(p_diverging.items(), key=lambda kv: kv[1]):
            lines.append(f"- `{key}`: {frac:.0%} of runs agree")
        lines.append("")
    if unpriced:
        lines.append(
            "Unpriced judge models (tokens excluded from measured cost): "
            + ", ".join(f"`{m}`" for m in unpriced)
        )
        lines.append("")

    fitted = suggested_priors(result.skill_runs)
    if fitted:
        lines.append("Suggested prior re-fits from these runs (not auto-applied):")
        for name, value in fitted.items():
            lines.append(f"- `{name}` ≈ {value}")
        lines.append("")

    errors = [r.error for r in (*result.skill_runs, *result.pipeline_runs) if r.error]
    if errors:
        lines.append("Trial errors:")
        for e in errors:
            lines.append(f"- {e}")
        lines.append("")
    return "\n".join(lines)


def measured_to_dict(result: EmpiricalResult, catalog: PricingCatalog) -> dict[str, Any]:
    """JSON form of the measured section."""
    pipe_cost, unpriced = measured_pipeline_cost_usd(
        tuple(r for r in result.pipeline_runs if r.succeeded), catalog
    )
    sa, pa = result.skill_agreement, result.pipeline_agreement
    return {
        "trials": result.trials,
        "skill_model": result.skill_model,
        "skill": {
            "runs": [
                {
                    "wall_seconds": r.wall_seconds,
                    "turns": r.turns,
                    "cost_usd": r.cost_usd,
                    "output_tokens": r.output_tokens,
                    "error": r.error,
                }
                for r in result.skill_runs
            ],
            "identical_fraction": sa.identical_fraction,
            "field_agreement": sa.field_agreement,
        },
        "pipeline": {
            "runs": [
                {
                    "wall_seconds": r.wall_seconds,
                    "judge_usage": list(r.judge_usage),
                    "error": r.error,
                }
                for r in result.pipeline_runs
            ],
            "mean_cost_usd": pipe_cost,
            "unpriced_models": unpriced,
            "identical_fraction": pa.identical_fraction,
            "field_agreement": pa.field_agreement,
        },
        "suggested_priors": suggested_priors(result.skill_runs),
    }
