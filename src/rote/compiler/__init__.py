"""The compiler orchestrator.

The :class:`Compiler` is the high-level entry point used by
``rote compile``. It:

1. Picks a driver (auto-detect or explicit ``--agent``).
2. Locates the rote-compile skill bundle.
3. Creates a temp work directory.
4. Calls ``driver.run()`` to produce ``pipeline.yaml`` (and any
   extracted modules / signature stubs).
5. Validates the produced ``pipeline.yaml`` via
   :func:`rote.ir.load_pipeline`.
6. Moves the work directory contents into the user's output directory.
7. Returns a :class:`CompilationResult`.

This module deliberately knows nothing about specific runtimes
(Temporal, Inngest, etc.) — that's the adapter layer's job. The
compiler's only output is the validated IR + the stub files;
``rote.adapters`` consumes those to emit runnable code.
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
import json
import math
import os
import re
import shutil
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from rote.compiler import api_docs, system1, tool_grounding
from rote.compiler.backend import BACKENDS, write_backend
from rote.compiler.drivers import (
    CompilerDriver,
    DriverError,
    DriverResult,
    Generator,
    auto_detect,
    available_drivers,
    get_driver,
)
from rote.compiler.events import CompilationEvent, EventCallback, emit_safely
from rote.compiler.grounding import (
    GROUNDING_DIRNAME,
    Grounding,
    Sample,
    contract_brief,
    find_sample,
    materialize,
    remove_materialized,
)
from rote.compiler.provenance import REPORT_FILENAME
from rote.compiler.report import ReportExtras, render_compile_report
from rote.compiler.system1 import TIEBREAKS, generate
from rote.compiler.system1.generate import DEFAULT_MAX_PARALLEL
from rote.compiler.system1.structure import DEFAULT as DEFAULT_STRUCTURE
from rote.compiler.system1.structure import MODELS as STRUCTURE_MODELS
from rote.compiler.system1.tiebreak import DEFAULT_TIEBREAK
from rote.compiler.update import (
    UPDATE_CONTEXT_DIRNAME,
    UpdatePlan,
    build_update_plan,
    render_update_brief,
)
from rote.compiler.usage import CompileUsage, compile_usage
from rote.compiler.verify import UNAPPLIED_PROPOSAL, EmitTarget, Verification, verify_deliverable
from rote.eval.sidecar import EVAL_SIDECAR_FILENAME, load_eval_estimates
from rote.extras import install_hint
from rote.ir import Pipeline, load_pipeline, save_pipeline
from rote.skill_source import PROVENANCE_FILENAME, load_provenance, write_provenance
from rote.vendor_specs import SpecStore

#: Work-dir directory the baseline's probe artifacts are materialized
#: into for the agent (mirrors update-context's convention).
PROBE_CONTEXT_DIRNAME = "probe-context"

#: How many times an invalid pipeline.yaml is bounced back to the agent
#: for repair before the run fails with the last validation error. Caps
#: the token cost of the repair loop.
MAX_VALIDATION_REPAIRS = 2
#: Appended to a System-1 IR-validation repair: where the agent's fix belongs.
SYSTEM1_REPAIR_NOTE = (
    "\n\nOn this compile rote restores every decision in pipeline.yaml after each "
    "stop and then applies system1/text.yaml over it: fix the text fields you wrote "
    "(descriptions, a judge's prompt and output_schema) in system1/text.yaml, not in "
    "pipeline.yaml."
)

COMPILE_METRICS_FILENAME = "compile-metrics.json"


class CompilerError(RuntimeError):
    """High-level compiler failure.

    Wraps driver errors and validation errors with a user-facing
    message. The CLI prints the message to stderr and exits non-zero.
    """


@dataclass
class CompilationResult:
    """The output of a successful compilation."""

    pipeline: Pipeline
    output_dir: Path
    driver_name: str
    driver_metadata: dict[str, Any] = field(default_factory=dict)
    #: The final check of the agent's deliverable (``None`` for an update
    #: that changed nothing, which ran no agent). A compile whose
    #: verification is not ``ok`` still returns: the artifacts are the
    #: output of a paid run. The CLI reports it and exits non-zero.
    verification: Verification | None = None
    #: What the compile's model calls are known to have used, stage by
    #: stage (:mod:`rote.compiler.usage`). ``None`` when the compiler that
    #: built this result reported none, which the CLI reports as unknown.
    usage: CompileUsage | None = None


def _write_compile_metrics(result: DriverResult, output_dir: Path) -> CompileUsage:
    """Persist bounded metrics before adapters can fail; never copy driver prose.

    Returns the usage it recorded, which the CLI reports, so the file and
    every other report are one computation."""
    metadata: dict[str, Any] = {}
    for key in (
        "input_tokens",
        "output_tokens",
        "cache_write_tokens",
        "cache_read_tokens",
        "num_turns",
        "iterations",
        "duration_ms",
        "duration_api_ms",
        "repair_rounds",
        "web_search_requests",
        "web_fetch_requests",
    ):
        if key not in result.metadata:
            continue
        value = result.metadata[key]
        if value is None or (isinstance(value, int) and not isinstance(value, bool) and value >= 0):
            metadata[key] = value
    session_id = result.metadata.get("session_id")
    if isinstance(session_id, str) and re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,199}", session_id
    ):
        metadata["session_id"] = session_id
    source = result.metadata.get("usage_source")
    if source in ("stream", "usage", "modelUsage", "generations", "turn.completed"):
        metadata["usage_source"] = source
    complete = result.metadata.get("usage_complete")
    if isinstance(complete, bool):
        metadata["usage_complete"] = complete
    cost = result.metadata.get("cost_usd")
    if (
        isinstance(cost, (int, float))
        and not isinstance(cost, bool)
        and math.isfinite(cost)
        and cost >= 0
    ):
        metadata["cost_usd"] = cost
    search = result.metadata.get("web_search_usd")
    if isinstance(search, (int, float)) and not isinstance(search, bool) and search >= 0:
        metadata["web_search_usd"] = search
    usage = compile_usage(result.driver_name, metadata, output_dir)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "driver": result.driver_name,
        "metadata": metadata,
        # Every stage, so an empty metadata object reads as unmeasured, not
        # as a free run (rote.compiler.usage).
        "usage": usage.as_dict(),
    }
    if "cost_usd" in metadata:
        payload["cost_basis"] = "driver_estimate"

    # Replace atomically: a partial write must not destroy the previous run's
    # metrics. Temporary files are private, and contain only the allowlist.
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_dir,
            prefix=f".{COMPILE_METRICS_FILENAME}.",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_dir / COMPILE_METRICS_FILENAME)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return usage


def _default_compiler_skill_dir() -> Path:
    """Locate the rote-compile skill bundled with the rote source.

    Two layouts exist, checked in order:

    1. **Wheel install.** ``pyproject.toml`` force-includes the
       repo-root ``skills/rote-compile/`` into the wheel at
       ``rote/skills/rote-compile/``, so the bundle sits next to the
       package's ``__file__``.
    2. **Editable install / source checkout.** The skill lives at
       ``<repo>/skills/rote-compile/``; ``rote.__file__`` is
       ``<repo>/src/rote/__init__.py``, so the repo root is three
       parents up.
    """
    import rote

    rote_pkg_dir = Path(rote.__file__).resolve().parent
    candidates = [
        rote_pkg_dir / "skills" / "rote-compile",
        rote_pkg_dir.parent.parent / "skills" / "rote-compile",
    ]
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "SKILL.md").is_file():
            return candidate
    searched = ", ".join(str(c) for c in candidates)
    raise CompilerError(
        f"Could not locate the rote-compile skill (searched: {searched}). "
        "Pass an explicit compiler_skill_dir to Compiler() pointing "
        "at a rote-compile skill bundle."
    )


class Compiler:
    """High-level orchestrator for compiling a skill into a pipeline."""

    def __init__(
        self,
        agent: str | None = None,
        compiler_skill_dir: Path | None = None,
        model: str | None = None,
        on_event: EventCallback | None = None,
        driver_kwargs: dict[str, Any] | None = None,
        run_generated_tests: bool = True,
        trust_skill: bool = False,
        emit_target: EmitTarget | None = None,
        strict_provenance: bool = False,
        system1: bool | None = None,
        system1_decider: str = "auto",
        system1_tiebreak: str = DEFAULT_TIEBREAK,
        system1_structure: str = DEFAULT_STRUCTURE,
        tool_discovery: tool_grounding.Discovery = "named",
        compile_anyway: bool = False,
        max_parallel: int = DEFAULT_MAX_PARALLEL,
    ) -> None:
        """
        Parameters
        ----------
        agent
            Driver name (``"claude"``, ``"codex"``, ``"api"``) or
            ``None`` for auto-detect.
        compiler_skill_dir
            Path to the rote-compile skill bundle. Defaults to the
            bundled one inside the rote source tree.
        model
            Override the LLM model the driver uses for the compiler
            (e.g. ``"claude-opus-4-6"`` to use Opus instead of the
            default Sonnet). ``None`` uses the driver's default.
        on_event
            Optional live-progress sink threaded to the driver and fired
            with orchestrator-level events (``log``, ``artifact``,
            ``complete`` / ``error``). Always invoked through
            :func:`~rote.compiler.events.emit_safely`, so a raising sink
            can't sink a paid run.
        driver_kwargs
            Extra keyword arguments merged into the driver constructor
            call (on top of ``model``). The cloud runner uses this to
            pass the api driver's ``base_url`` / ``default_headers`` for
            AI Gateway routing. Keys collide-last: an explicit
            ``driver_kwargs["model"]`` overrides the ``model`` argument.
        run_generated_tests
            Run the agent-written tests as part of verification (in a
            subprocess with an allowlisted environment and an empty HOME;
            see :mod:`rote.compiler.verify`). ``False`` is
            ``--no-verify-tests``: for a skill you do not trust, nothing
            the agent wrote is executed, and the static checks still run.
        trust_skill
            Run the generated tests without an OS sandbox (``--trust-skill``).
            By default they run under one, and are skipped (the deliverable
            is unverified) when none is available.
        emit_target
            The runtime the caller will emit for. Verification then dry-emits
            the deliverable with that adapter, so an IR it refuses goes back
            to the agent instead of failing after the compile (``None``:
            no emit check, for callers that do not emit). Its
            ``external_backend="api"`` makes this an api-backend compile: the
            driver gets web tools, the agent the brief in
            :mod:`rote.compiler.api_docs`, and every vendor call must cite
            the docs it relied on.
        strict_provenance
            ``--strict-provenance``: every unresolved provenance finding is an
            error, review gaps included. By default only a citation the skill
            does not contain fails; assumptions never do.
        system1
            System-1 (the default): every structural decision is made by
            rules before the agent runs (:mod:`rote.compiler.system1`), and
            the agent, under a smaller rubric, writes only text. rote holds
            the agent to the decisions after every stop. ``None`` (the
            default) means System-1 for a full compile and the full-rubric
            agent for ``update=True``, which System-1 does not support yet.
            ``False`` (``--no-system1``) is the full-rubric agent always.
            ``True`` with ``update=True`` is an error.
        system1_decider
            The decision model on the System-1 path
            (:data:`rote.compiler.system1.DECIDERS`): it checks the structure
            pass, or decides kinds and gates on the rules path. ``"auto"``
            (the default) takes the first of
            :data:`rote.compiler.system1.decision_model.CHECKER_ORDER` with
            credentials, and none means no check; the compile says which and
            why. A model asked for by name with no credentials fails before
            any spend. A model decider with System-1 off is an error, never
            ignored.
        system1_tiebreak
            ``"haiku"``: on the System-1 path, the steps still unsure after
            the decider go to Haiku in one schema-locked call, billed the way
            the compile's driver bills (:mod:`rote.compiler.system1.tiebreak`).
            ``"none"`` (the default): no call.
        max_parallel
            ``--max-parallel``: the most model calls in flight at once when
            System-1 writes its text as parallel calls (drivers with
            ``generate``; :mod:`rote.compiler.system1.generate`).
        """
        self.agent = agent
        self._explicit_compiler_skill_dir = compiler_skill_dir
        self.model = model
        self.on_event = on_event
        self.driver_kwargs = driver_kwargs
        self.run_generated_tests = run_generated_tests
        self.trust_skill = trust_skill
        self.emit_target = emit_target
        #: Compile to the skill runtime even when it would not pay off
        #: (:mod:`rote.compiler.worth`).
        self.compile_anyway = compile_anyway
        self.strict_provenance = strict_provenance
        self.system1 = system1
        self.tool_discovery = tool_discovery
        self.system1_decider = system1_decider
        if system1_tiebreak not in TIEBREAKS:
            raise CompilerError(f"unknown --tiebreak {system1_tiebreak!r}; choose from {TIEBREAKS}")
        self.system1_tiebreak = system1_tiebreak
        if max_parallel < 1:
            raise CompilerError(f"--max-parallel must be at least 1, got {max_parallel}")
        self.max_parallel = max_parallel
        #: The model the selected driver runs, once :meth:`compile` has chosen
        #: it (None before, or for a driver that names none): what live
        #: progress is priced as.
        self.driver_model: str | None = None
        if system1_structure not in ("rules", *STRUCTURE_MODELS):
            raise CompilerError(f"unknown --structure {system1_structure!r}")
        self.system1_structure = system1_structure

    @property
    def api_backend(self) -> bool:
        """Whether this compile writes direct vendor-API calls (``--backend api``)."""
        return self.emit_target is not None and self.emit_target.external_backend == "api"

    @property
    def compiler_skill_dir(self) -> Path:
        if self._explicit_compiler_skill_dir is not None:
            return self._explicit_compiler_skill_dir
        return _default_compiler_skill_dir()

    def _resolve_decider(self) -> tuple[str, str]:
        """``(decider, why)`` for this compile; ``auto`` resolved, and said."""
        if self.system1_decider != "auto":
            return self.system1_decider, f"--decider {self.system1_decider}"
        decider, why = system1.decision_model.default_checker()
        # One line either way: which model runs and why, or what was skipped
        # and the variables that would enable one.
        self._emit("log" if decider != "rules" else "warning", f"system1: {why}")
        return decider, why

    def select_driver(self) -> CompilerDriver:
        """Choose a driver based on the agent setting or auto-detect.

        When a non-default ``model`` was passed to the ``Compiler``
        constructor, it's forwarded to the driver via ``get_driver``
        kwargs. Auto-detect finds an available driver first (using
        a zero-arg probe), then re-constructs it with the kwargs so
        the override is honored.
        """
        driver_kwargs: dict[str, object] = {}
        if self.model is not None:
            driver_kwargs["model"] = self.model
        # Caller-supplied kwargs win over the model shorthand, so an
        # explicit driver_kwargs["model"] (or base_url / default_headers)
        # takes precedence.
        if self.driver_kwargs:
            driver_kwargs.update(self.driver_kwargs)
        if self.api_backend:
            # Current docs beat training-data memory for a vendor call: every
            # driver grants its web search and fetch tools on this flag.
            driver_kwargs["web_tools"] = True

        if self.agent:
            try:
                driver = get_driver(self.agent, **driver_kwargs)
            except KeyError as e:
                raise CompilerError(str(e.args[0])) from None
            available, reason = driver.is_available()
            if not available:
                raise CompilerError(f"Driver {self.agent!r} is not available: {reason}")
            return driver

        probe = auto_detect()
        if probe is None:
            lines = [
                "No compiler driver is available. Tried:",
            ]
            for name, _avail, reason in available_drivers():
                lines.append(f"  - {name}: {reason}")
            lines.append("")
            lines.append(
                "Install one of the supported coding agent CLIs, or set "
                f"ANTHROPIC_API_KEY and `{install_hint('api')}`."
            )
            raise CompilerError("\n".join(lines))

        # Re-construct the detected driver with user-specified kwargs
        # (e.g. model override) instead of returning the probe instance
        # directly.
        return get_driver(probe.name, **driver_kwargs)

    def _refuse_api_backend_on_typescript(self) -> None:
        """``--backend api`` writes Python vendor calls; a TS runtime would stub them all.

        Checked here so every caller (the CLI, the cloud container, the bench)
        fails before the agent runs, not after a paid compile.
        """
        from rote.runners import _TS_RUNTIMES

        if self.api_backend and self.emit_target and self.emit_target.runtime in _TS_RUNTIMES:
            raise CompilerError(
                f"--backend api writes the vendor calls as Python modules, and the "
                f"{self.emit_target.runtime} runtime emits TypeScript, with a stub for "
                f"every external_call. Use the dbos, python or temporal runtime, or "
                f"--backend mcp for {self.emit_target.runtime}"
            )

    def _emit(self, type_: str, message: str, **fields: Any) -> None:
        """Fire an orchestrator-level progress event (no-op without a sink)."""
        emit_safely(
            self.on_event,
            CompilationEvent(type=type_, ts=time.time(), message=message, **fields),  # type: ignore[arg-type]
        )

    async def compile(
        self,
        skill_dir: Path | str,
        output_dir: Path | str,
        update: bool = False,
        extra_instructions: str | None = None,
        probe_dir: Path | str | None = None,
        sample_input: Path | str | None = None,
    ) -> CompilationResult:
        """Run the full compilation flow.

        Parameters
        ----------
        skill_dir
            Path to the source skill bundle (must contain a SKILL.md).
        output_dir
            Where to write the compiled artifacts. Created if missing;
            existing files of the same name are overwritten.
        update
            Incremental mode: require a previous compilation (with its
            provenance sidecar) in ``output_dir``, diff the current
            SKILL.md against the stamped section hashes, and instruct
            the agent to re-derive only nodes whose source material
            changed. A skill with no section changes returns without
            invoking the agent at all.
        extra_instructions
            Additional instructions appended to the agent's prompt —
            typically a hosting runtime's contract (e.g. which
            ``signature_spec.client`` values its isolates can execute).
            Composed with (not replaced by) the update-mode brief.
        probe_dir
            A ``rote baseline`` artifact directory. When it holds
            observed-tools.json / inferred-schemas.json they are
            materialized into the agent's work dir as ground truth
            (observed MCP traffic + inferred schemas). Missing dir or
            artifacts is the normal no-baseline case, never an error.
        sample_input
            A JSON file with a sample input for the skill (``--sample-input``).
            Without one, a sample bundled in the skill dir is used, then the
            input a baseline in ``probe_dir`` ran on. The pipeline's input
            contract is grounded in it and checked against it
            (:mod:`rote.compiler.grounding`).

        Returns
        -------
        CompilationResult
            A validated :class:`Pipeline` plus the output directory and
            driver metadata.

        Raises
        ------
        CompilerError
            For any user-actionable failure: missing skill bundle,
            unavailable driver, driver failure, invalid produced
            pipeline.yaml — or, in update mode, a missing previous run
            or an update that dropped provenance-preserved nodes.
        """
        skill_dir = Path(skill_dir).resolve()
        output_dir = Path(output_dir).resolve()

        resolved_probe_dir = Path(probe_dir).resolve() if probe_dir is not None else None
        try:
            try:
                sample = find_sample(
                    skill_dir,
                    Path(sample_input).resolve() if sample_input is not None else None,
                    resolved_probe_dir,
                )
            except ValueError as e:
                raise CompilerError(str(e)) from e
            return await self._compile_inner(
                skill_dir, output_dir, update, extra_instructions, resolved_probe_dir, sample
            )
        except CompilerError as e:
            self._emit("error", f"compilation failed: {e}")
            raise

    async def _compile_inner(
        self,
        skill_dir: Path,
        output_dir: Path,
        update: bool,
        extra_instructions: str | None = None,
        probe_dir: Path | None = None,
        sample: Sample | None = None,
    ) -> CompilationResult:
        """The compilation body, bracketed by ``compile``'s error event."""
        self._refuse_api_backend_on_typescript()
        if not skill_dir.is_dir():
            raise CompilerError(f"Skill directory does not exist: {skill_dir}")
        if not (skill_dir / "SKILL.md").is_file():
            raise CompilerError(f"Not a skill bundle (no SKILL.md found): {skill_dir}")
        from rote.skill_files import SkillTooLarge, preflight

        try:
            # Before anything reads the skill, and before any driver runs.
            outside = preflight(skill_dir)
        except SkillTooLarge as e:
            raise CompilerError(f"{skill_dir}: {e}") from e
        if outside:
            listed = ", ".join(str(p.relative_to(skill_dir)) for p in outside)
            self._emit(
                "warning",
                f"skipped {len(outside)} reference path(s) that resolve outside the skill "
                f"directory: {listed}",
            )

        skill_md_text = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
        if self.system1 and update:
            raise CompilerError("System-1 does not support --update yet; run a full compile")
        use_system1 = self.system1 if self.system1 is not None else not update
        if not use_system1 and self.system1_decider not in ("rules", "auto"):
            raise CompilerError(
                f"--decider {self.system1_decider} decides System-1's structure, "
                "and this compile does not run System-1"
                + (" (--update uses the full-rubric agent)" if update else "")
            )
        if update and self.system1 is None:
            self._emit(
                "log",
                "--update runs the full-rubric agent: System-1 does not support --update yet",
            )

        plan: UpdatePlan | None = None
        if update:
            prev_pipeline, plan = self._plan_update(output_dir, skill_md_text)
            self._emit(
                "log",
                f"update plan: {len(plan.changed_sections)} changed section(s), "
                f"{len(plan.stale_node_ids)} stale node(s), "
                f"{len(plan.preserved_node_ids)} preserved",
            )
            if plan.is_noop:
                self._emit("log", "no changes — skipping agent")
                self._emit("complete", "up to date: no source sections changed")
                return CompilationResult(
                    pipeline=prev_pipeline,
                    output_dir=output_dir,
                    driver_name="(no-op)",
                    driver_metadata={
                        "update": "no source sections changed; previous pipeline kept"
                    },
                    # No model call ran: a measured zero, not a missing count.
                    usage=CompileUsage([]),
                )

        driver = self.select_driver()
        self.driver_model = getattr(driver, "model", None)
        self._emit("log", f"driver selected: {driver.name}")

        with (
            tempfile.TemporaryDirectory(prefix="rote-compile-") as work_dir_str,
            tempfile.TemporaryDirectory(prefix="rote-system1-rubric-") as rubric_dir_str,
        ):
            work_dir = Path(work_dir_str)
            compiler_skill_dir = self.compiler_skill_dir
            # What the contracts are grounded in, decided before anything
            # else: System-1 makes its input, gate and argument decisions
            # from it, and the agent is handed it.
            # Imported here: rote.eval.empirical imports rote.compiler.drivers,
            # which runs this package's __init__, so a module-level import of
            # rote.eval.baseline would be circular.
            from rote.eval.baseline import load_observations

            observations = tuple(load_observations(probe_dir)) if probe_dir else ()
            if observations and self.tool_discovery != "none":
                from rote.compiler.replay import runtime_shapes

                observations = await runtime_shapes(
                    observations, lambda line: self._emit("log", line)
                )
            try:
                snapshot = tool_grounding.bundled(skill_dir)
            except ValueError as e:
                raise CompilerError(str(e)) from e
            tools, unlisted, candidates = await self._tool_schemas(
                observations, skill_dir, snapshot
            )
            grounding = Grounding(
                sample=sample,
                observations=observations,
                tools=tools,
                unlisted=unlisted,
                bundled=snapshot,
            )
            if candidates and not use_system1:
                # Only the structure pass chooses among offered tools.
                for server in sorted({s for s, _ in candidates}):
                    grounding.unlisted[server] = "the skill names none of its tools"
                    self._emit(
                        "log",
                        f"tool schemas: not using {server!r}: the skill names none of its "
                        "tools, and only System-1 chooses among a logged-in server's tools",
                    )
            s1_plan: system1.Plan | None = None
            if use_system1:
                decider, decider_why = self._resolve_decider()
                structure_runs = self.system1_structure != "rules"
                if structure_runs and self.system1_tiebreak == "haiku":
                    # The structure pass decides every kind and gate, so the
                    # tie-break never runs (the decider double-checks the pass
                    # instead): say so, and resolve no lane for it.
                    self._emit(
                        "warning",
                        "system1: --tiebreak haiku does not run while the structure pass "
                        f"({self.system1_structure}) decides the skeleton; pass "
                        "--structure rules to use it",
                    )
                try:
                    lane = (
                        system1.tiebreak.lane_for(driver)
                        if self.system1_tiebreak == "haiku" and not structure_runs
                        else None
                    )
                    s1_plan = system1.prepare(
                        skill_dir,
                        work_dir,
                        decider,
                        decider_why=decider_why,
                        sample=sample,
                        tools={**grounding.tools, **candidates},
                        offered=sorted({server for server, _ in candidates}),
                        observations=grounding.observations,
                        tiebreak_lane=lane,
                        structure_lane=(
                            system1.structure.lane_for(driver, self.system1_structure)
                            if self.system1_structure != "rules"
                            else None
                        ),
                        # rote's own rubric, whichever compiler skill the agent gets.
                        rubric=_default_compiler_skill_dir() / "references" / "node-kinds.md",
                    )
                except (
                    system1.UnsafeSkillText,
                    system1.DeciderUnavailable,
                    system1.TieBreakUnavailable,
                    system1.structure.StructureUnavailable,
                ) as e:
                    usage = getattr(e, "usage", None) or []
                    spent = system1.tiebreak.spent_summary(usage)
                    if spent:
                        # A failed compile is still a paid one: say what System-1's
                        # calls cost, on the progress feed and in the error.
                        self._emit("log", f"usage before the failure: system1 {spent}")
                        error = CompilerError(f"{e} (spent before the failure: {spent})")
                        error.usage = usage  # type: ignore[attr-defined]
                        raise error from e
                    raise CompilerError(str(e)) from e
                # An offered server the plan binds a step to, or whose tool a
                # loop lists, is now grounding like any other; one it uses
                # nothing of is left unused.
                adopted, unused = tool_grounding.adopt_offered(
                    candidates,
                    (n.mcp["server"] for n in s1_plan.nodes if n.mcp),
                    (t for n in s1_plan.nodes for t in n.tools or ()),
                )
                for server in sorted({s for s, _ in adopted}):
                    self._emit(
                        "log",
                        f"tool schemas: using {server!r} (the structure pass bound a step to "
                        "its tools)",
                    )
                for server in unused:
                    grounding.unlisted[server] = "no step uses its tools"
                    self._emit("log", f"tool schemas: not using {server!r}: no step uses it")
                grounding.tools.update(adopted)
                self._refuse_if_not_worth_it(s1_plan)
                st = s1_plan.structure
                if st is not None:
                    cost = "cost unknown" if st.cost_usd is None else f"${st.cost_usd:.4f}"
                    self._emit(
                        "log",
                        f"system1: structure pass by {st.model_label} via {st.lane}, "
                        f"{st.rounds} round(s), {len(st.resolved)} unresolved ({cost})",
                    )
                tb = s1_plan.tiebreak
                if tb is not None:
                    cost = "cost unknown" if tb.cost_usd is None else f"${tb.cost_usd:.4f}"
                    self._emit(
                        "log",
                        f"system1: tie-break asked {tb.model} about {len(tb.asked)} unsure "
                        f"step(s) via {tb.lane} ({cost})",
                    )
                self._emit(
                    "log",
                    f"system1: decided {len(s1_plan.nodes)} node(s); "
                    f"{len(s1_plan.review)} decision(s) below {system1.REVIEW_BELOW} "
                    "confidence go to the report",
                )
                compiler_skill_dir = system1.materialize_text_rubric(
                    compiler_skill_dir, Path(rubric_dir_str)
                )

            if sample is not None:
                self._emit("log", f"input contract grounded in a sample input: {sample.source}")
            # A --backend api compile keeps the session: its external_call
            # modules are researched vendor calls, written with web tools the
            # parallel calls do not have.
            parallel = (
                s1_plan is not None
                and not self.api_backend
                and callable(getattr(driver, "generate", None))
            )
            research: api_docs.WebResearch | None = None
            specs: SpecStore | None = None
            base_dir = output_dir if plan is not None else None
            run_kwargs: dict[str, Any] = {}
            if not parallel:
                # The session's briefs, context files and repairer. The
                # parallel pieces need none of them: they read the sample, the
                # tool schemas and the recorded calls from `grounding`
                # (generate.Context) and `extra_instructions` as their hosting
                # contract, and generate's own verify loop repairs them.
                instructions: list[str] = []
                if extra_instructions:
                    instructions.append(extra_instructions)
                if self.api_backend:
                    instructions.append(api_docs.brief())
                    research = api_docs.WebResearch()
                    specs = SpecStore()
                if plan is not None:
                    instructions.append(
                        self._materialize_update_context(work_dir, output_dir, plan)
                    )
                if probe_dir is not None:
                    probe_instructions = self._materialize_probe_context(work_dir, probe_dir)
                    if probe_instructions is not None:
                        self._emit(
                            "log", "probe artifacts found — feeding observed traffic to the agent"
                        )
                        instructions.append(probe_instructions)
                ctx_dir = materialize(work_dir, sample) if sample is not None else None
                brief = contract_brief(sample, skill_dir, ctx_dir)
                if brief:
                    instructions.append(brief)
                if grounding.tools:
                    schemas = tool_grounding.materialize(
                        work_dir / GROUNDING_DIRNAME, grounding.tools
                    )
                    instructions.append(
                        tool_grounding.brief(grounding.tools, skill_md_text, schemas)
                    )
                if instructions:
                    run_kwargs["extra_instructions"] = "\n\n".join(instructions)
                # Repair pass: a deliverable that fails verification (an
                # invalid pipeline.yaml, a module that cannot take its payload
                # or parse on 3.11, red generated tests) is bounced back into
                # the SAME agent conversation with the verbatim problems
                # instead of shipping the paid run broken. Only drivers whose
                # run() declares the ``repair`` parameter can resume a
                # conversation (the in-process drivers, claude via
                # `--resume`, and codex via `codex exec resume`).
                if "repair" in inspect.signature(driver.run).parameters:
                    settle_round = (
                        None if s1_plan is None else functools.partial(system1.settle, p=s1_plan)
                    )
                    run_kwargs["repair"] = self._deliverable_repairer(
                        base_dir,
                        skill_dir,
                        before_verify=settle_round,
                        grounding=grounding,
                        portable_judges=s1_plan is not None,
                        web_research=research,
                        spec_store=specs,
                    )

            try:
                if parallel:
                    assert s1_plan is not None
                    result = await self._generate_text(
                        cast("Generator", driver),
                        skill_dir=skill_dir,
                        work_dir=work_dir,
                        s1_plan=s1_plan,
                        grounding=grounding,
                        extra_instructions=extra_instructions,
                    )
                else:
                    result = await driver.run(
                        skill_dir=skill_dir,
                        compiler_skill_dir=compiler_skill_dir,
                        work_dir=work_dir,
                        # An api-backend compile records the agent's web lookups
                        # off the tool events, for the report's docs section.
                        on_event=self.on_event if research is None else research.tee(self.on_event),
                        **run_kwargs,
                    )
            except DriverError as e:
                detail = f"\n\n{e.details}" if getattr(e, "details", None) else ""
                spent = ""
                usage = self._usage_summary(e.metadata) if e.metadata else ""
                if usage:
                    # A failed run is still a paid one: say what it cost, on
                    # the progress feed and in the error the CLI prints.
                    self._emit("log", f"usage before the failure: {usage}")
                    spent = f" after spending {usage}"
                raise CompilerError(f"Driver {driver.name!r} failed{spent}: {e}{detail}") from e

            s1_extras = None
            if s1_plan is not None:
                try:
                    system1.settle(result.work_dir, s1_plan, final=True)
                    system1.record_structure(result.work_dir, s1_plan)
                    s1_extras = system1.report_extras(result.work_dir, s1_plan)
                except Exception as e:  # noqa: BLE001 (keep the paid run, report why)
                    kept = self._keep_rejected_work(result.work_dir, output_dir)
                    raise CompilerError(
                        f"System-1 could not settle the agent's deliverable: "
                        f"{type(e).__name__}: {e}. The agent's work is kept at {kept}."
                    ) from e

            try:
                pipeline = load_pipeline(result.pipeline_yaml_path)
            except Exception as e:
                kept = self._keep_rejected_work(result.work_dir, output_dir)
                raise CompilerError(
                    f"Driver {driver.name!r} produced an invalid pipeline.yaml: {e}\n\n"
                    f"Any previous output in {output_dir} is untouched. The agent's "
                    f"work is kept at {kept}. Fix {kept / 'pipeline.yaml'} "
                    f"where the errors point, then emit it without another agent run: "
                    f"rote emit {kept / 'pipeline.yaml'} --runtime <runtime> --out <dir>"
                ) from e

            if plan is not None:
                produced_ids = {n.id for n in pipeline.nodes}
                missing = [nid for nid in plan.preserved_node_ids if nid not in produced_ids]
                if missing:
                    kept = self._keep_rejected_work(result.work_dir, output_dir)
                    raise CompilerError(
                        f"Update run dropped or renamed nodes whose source sections "
                        f"did not change: {missing}. Node ids version the emitted "
                        f"workflow, so this would orphan in-flight runs. The previous "
                        f"output was left untouched and the rejected update is kept "
                        f"at {kept}. Re-run, or run a full compilation if the "
                        f"restructure is intentional."
                    )

            pipeline = self._pin_loop_tools(
                Path(result.pipeline_yaml_path), pipeline, grounding.tools
            )

            # The final verdict, after any repair rounds: what the CLI
            # reports, and what decides its exit code. Run before the move
            # so a verification crash can't strand the work in the temp dir
            # half-moved; verification itself never raises on bad code.
            verification = verify_deliverable(
                result.work_dir,
                base_dir=base_dir,
                run_tests=self.run_generated_tests,
                trust_skill=self.trust_skill,
                skill_dir=skill_dir,
                emit_target=self.emit_target,
                strict_provenance=self.strict_provenance,
                grounding=grounding,
                portable_judges=s1_plan is not None,
                web_research=research,
                spec_store=specs,
            )
            unresolved = result.metadata.get("unresolved")
            if isinstance(unresolved, dict) and unresolved:
                # A parallel-path piece that never passed its own checks
                # (rote.compiler.system1.generate) is an error, not a pass.
                verification = dataclasses.replace(
                    verification,
                    findings=[
                        *verification.findings,
                        *generate.unresolved_findings(unresolved),
                    ],
                )
            self._emit_verification(verification)
            # The sample was reference material for the agent, not output.
            remove_materialized(Path(result.pipeline_yaml_path).parent)

            # Stamp provenance: hash every SKILL.md section so a later
            # `compile --update` can tell exactly which nodes' source
            # material changed. The agent wrote `source.section` per
            # node; the hashes are ours to compute.
            write_provenance(
                Path(result.pipeline_yaml_path).parent / PROVENANCE_FILENAME,
                pipeline,
                skill_md_text,
            )

            if plan is not None:
                # The context dir was reference material for the agent,
                # not output.
                shutil.rmtree(work_dir / UPDATE_CONTEXT_DIRNAME, ignore_errors=True)
                self._merge_work_dir_to_output(result.work_dir, output_dir)
            else:
                self._move_work_dir_to_output(result.work_dir, output_dir)

            usage = _write_compile_metrics(result, output_dir)
            tool_grounding.write_declared(output_dir, grounding.tools)
            if self.emit_target is not None and self.emit_target.external_backend in BACKENDS:
                # Beside the IR, never in it: a later `rote emit` must not
                # silently swap the researched vendor calls for MCP calls.
                write_backend(output_dir, self.emit_target.external_backend)
            if self.emit_target is not None and self.emit_target.external_backend == "api":
                # rote's own record of the pages fetched, written after the
                # agent is done and over any file the agent wrote by that name
                # (an empty record when the lane reported nothing): the live
                # check vouches for a docs spec's host only from it.
                api_docs.write_research(output_dir, research or api_docs.WebResearch())

            # The report is rendered, not written by the agent: everything
            # in it is already in pipeline.yaml and the extracted code (see
            # rote.compiler.report). Rendered from output_dir, after an
            # update's merge, so it describes the deliverable as kept.
            # Contracts nothing grounded are rote's open questions, not the
            # agent's: listed for the reviewer, never a failure.
            contract_questions = [
                (f"Contract: {q.node_id}", q.text) for q in verification.questions
            ]
            mode = s1_extras or ReportExtras()
            extras = dataclasses.replace(
                mode,
                questions=[*mode.questions, *contract_questions],
                api_backend=self.api_backend,
                runtime=self.emit_target.runtime if self.emit_target is not None else "dbos",
                web_research=research,
                spec_checks=verification.spec_checks,
            )
            (output_dir / REPORT_FILENAME).write_text(
                render_compile_report(pipeline, output_dir, extras), encoding="utf-8"
            )

            # The agent records source_skill relative to its own temp
            # work dir, which is deleted the moment this context manager
            # exits — a dead pointer that silently costs `rote eval` its
            # entire before-side baseline. We know both real paths, so
            # re-point it to resolve from the pipeline.yaml's final home.
            pipeline = self._repoint_source_skill(
                output_dir / "pipeline.yaml", pipeline, skill_dir, output_dir
            )
            self._repoint_sidecar_source_skill(
                output_dir / EVAL_SIDECAR_FILENAME, skill_dir, output_dir
            )

            # The two durable deliverables now live in output_dir; announce
            # them so a consumer knows the artifacts are on disk before the
            # completion summary lands.
            self._emit(
                "artifact",
                "wrote pipeline.yaml",
                path=str((output_dir / "pipeline.yaml").relative_to(output_dir)),
            )
            self._emit(
                "artifact",
                "wrote provenance.json",
                path=str(PROVENANCE_FILENAME),
            )
            self._emit("complete", self._completion_message(result))

            # Re-point the result's path to the moved location for the
            # caller's benefit.
            # System-1's own calls (decider, tie-break, structure pass) are
            # outside the driver's usage; the CLI adds them to the compile's
            # cost so the figure bench and agents read is the whole spend.
            metadata = (
                result.metadata
                if s1_plan is None
                else {**result.metadata, "system1_cost_usd": system1.calls_cost_usd(output_dir)}
            )
            return CompilationResult(
                pipeline=pipeline,
                output_dir=output_dir,
                driver_name=result.driver_name,
                driver_metadata=metadata,
                verification=verification,
                usage=usage,
            )

    async def _generate_text(
        self,
        driver: Generator,
        *,
        skill_dir: Path,
        work_dir: Path,
        s1_plan: system1.Plan,
        grounding: Grounding,
        extra_instructions: str | None,
    ) -> DriverResult:
        """System-1's text as parallel schema-locked calls (:mod:`.system1.generate`).

        Verification between repair rounds is the same check the
        single-session path runs at each stop, on the settled work dir.
        """

        def verify(artifact_dir: Path) -> Verification:
            system1.settle(artifact_dir, s1_plan)
            return verify_deliverable(
                artifact_dir,
                run_tests=self.run_generated_tests,
                trust_skill=self.trust_skill,
                skill_dir=skill_dir,
                emit_target=self.emit_target,
                strict_provenance=self.strict_provenance,
                grounding=grounding,
                portable_judges=True,
            )

        settings = generate.Settings(
            models=generate.Models.every(self.model) if self.model else generate.DEFAULT_MODELS,
            max_parallel=self.max_parallel,
        )
        self._emit(
            "log",
            f"system1: writing the text as parallel calls (at most {settings.max_parallel} "
            f"at once; interface {settings.models.interface}, modules {settings.models.module})",
        )
        return await generate.write_text(
            driver,
            work_dir=work_dir,
            skill_dir=skill_dir,
            compiler_skill_dir=self.compiler_skill_dir,
            plan=s1_plan,
            grounding=grounding,
            verify=verify,
            settings=settings,
            extra_instructions=extra_instructions,
            on_event=self.on_event,
            run_tests=self.run_generated_tests,
            trust_skill=self.trust_skill,
            redecide=self._redecide(driver, skill_dir, work_dir, s1_plan, grounding),
        )

    def _redecide(
        self,
        driver: Any,
        skill_dir: Path,
        work_dir: Path,
        s1_plan: system1.Plan,
        grounding: Grounding,
    ) -> generate.Redecide | None:
        """The structure pass's re-decide for MCP bindings, when it decided the skeleton."""
        record = s1_plan.structure
        if record is None:
            return None
        from dataclasses import replace as _replace

        from rote.compiler.system1 import structure
        from rote.compiler.system1.parse import parse_skill

        lane = structure.lane_for(driver, self.system1_structure)
        skill = _replace(parse_skill(skill_dir), units=structure.section_units(skill_dir))
        skill_md = structure.skill_text(skill_dir)

        def run(findings: dict[str, list[str]]) -> list[str]:
            try:
                changes, usage = structure.redecide(
                    work_dir,
                    findings,
                    skill,
                    skill_md,
                    lane,
                    tools=grounding.tools,
                    sample=grounding.sample,
                )
            except structure.StructureUnavailable as e:
                # The text is already paid for: a failed re-decide leaves the
                # findings standing for verification to report, and its calls
                # still count.
                record.usage += e.usage
                return [f"re-deciding MCP bindings failed, the findings stand: {e}"]
            record.usage += usage
            record.redecided += changes
            return changes

        return run

    def _refuse_if_not_worth_it(self, plan: system1.Plan) -> None:
        """Stop a skill-runtime compile whose plan stops for the model too
        often to pay off, before generation spends anything more. Only the
        structure pass (cents) has run; its cost is in the error."""
        from rote.compiler.worth import worth

        target = self.emit_target
        if target is None or target.runtime != "skill" or self.compile_anyway:
            return
        verdict = worth(plan.nodes, target.judges)
        if verdict.pays_off:
            return
        st = plan.structure
        spent = "" if st is None or st.cost_usd is None else f" (spent so far: ${st.cost_usd:.4f})"
        raise CompilerError(verdict.explain() + spent)

    async def _tool_schemas(
        self,
        observations: tuple[Any, ...],
        skill_dir: Path,
        snapshot: tool_grounding.ToolSchemas | None = None,
    ) -> tuple[tool_grounding.ToolSchemas, dict[str, str], tool_grounding.ToolSchemas]:
        """Every MCP tool's input schema rote can get, before the agent runs.

        Returns the schemas, the candidates (the tools of logged-in servers
        the skill names none of, for the structure pass to choose from), and
        ``{server: why}`` for each registered server
        it did not list (:attr:`Grounding.unlisted`), so the compile summary
        can say what to fix.

        Declared schemas from a live ``tools/list`` on the registered
        servers ``tool_discovery`` allows (by default only the ones the
        skill names: a third-party skill must not trigger authenticated
        requests to the user's other servers), then schemas inferred from a
        baseline's recorded calls for any tool nothing declared. A server
        that cannot be listed is logged and skipped. The skill's bundled
        tools/list ``snapshot`` grounds every server not listed live: a
        live listing always wins (:func:`tool_grounding.prefer_live`).
        """
        from rote.mcp import load_registry

        configs = load_registry().servers
        registered = sorted(configs)
        skill_text = "\n".join(
            p.read_text(encoding="utf-8", errors="replace") for p in skill_dir.rglob("*.md")
        )
        contact = tool_grounding.servers_to_contact(
            registered,
            skill_text,
            self.tool_discovery,
            tool_grounding.skill_tool_table_servers(skill_dir),
        )
        # A server the user logged in to may be listed too, and is kept only
        # if the skill calls one of its tools: the skill names tools, not
        # always the server (Atlassian's say "Jira", never "atlassian").
        logged_in = (
            [s for s in tool_grounding.logged_in_servers(configs) if s not in contact]
            if self.tool_discovery == "named"
            else []
        )
        declared: tool_grounding.ToolSchemas = {}
        candidates: tool_grounding.ToolSchemas = {}
        for server, why in contact.items():
            self._emit("log", f"tool schemas: listing {server!r} (the skill says: {why!r})")
        for server in logged_in:
            self._emit(
                "log",
                f"tool schemas: listing {server!r} (you logged in to it; kept if the skill "
                "calls one of its tools)",
            )
        left = [s for s in registered if s not in contact and s not in logged_in]
        unlisted = {
            s: f"the skill does not name it (--tool-discovery {self.tool_discovery})" for s in left
        }
        if left:
            self._emit(
                "log",
                f"tool schemas: not contacting {', '.join(left)} "
                f"(not named in the skill and not logged in; --tool-discovery "
                f"{self.tool_discovery})",
            )
        if contact or logged_in:

            def _skip(server: str, reason: str) -> None:
                unlisted[server] = reason
                self._emit("log", f"tool schemas: skipped server {server!r}: {reason}")

            declared = await tool_grounding.discover([*contact, *logged_in], on_skip=_skip)
            if logged_in:
                listed = {k: v for k, v in declared.items() if k[0] in logged_in}
                calls = tool_grounding.servers_whose_tools_the_skill_calls(listed, skill_text)
                for server in logged_in:
                    if server in calls:
                        self._emit("log", f"tool schemas: using {server!r} ({calls[server]})")
                    elif server not in unlisted:
                        self._emit(
                            "log",
                            f"tool schemas: offering {server!r}'s tools to the structure pass: "
                            "the skill names none of them, so it decides whether a step uses one",
                        )
                # The rest are candidates: the structure pass sees their
                # names and descriptions and may bind a step to one, which
                # code then checks; a server it binds none of stays unused.
                candidates = {
                    k: v for k, v in declared.items() if k[0] in logged_in and k[0] not in calls
                }
                declared = {k: v for k, v in declared.items() if k not in candidates}
        if snapshot:
            servers = sorted({s for s, _ in snapshot} - {s for s, _ in declared})
            if servers:
                self._emit(
                    "log",
                    f"tool schemas: {', '.join(servers)} from the skill's bundled tools/list",
                )
            declared = tool_grounding.prefer_live(declared, snapshot)
        tools = tool_grounding.merge(declared, tool_grounding.recorded(observations))
        if tools:
            servers = sorted({s for s, _ in tools})
            self._emit(
                "log",
                f"tool arguments grounded in {len(tools)} tool schema(s) from {', '.join(servers)}",
            )
        return tools, unlisted, candidates

    def _emit_verification(self, verification: Verification) -> None:
        """One log line for the per-node implementation tally, one warning per problem."""
        impls = verification.implementations
        if impls:
            done = sum(1 for i in impls if i.status == "implemented")
            self._emit("log", f"implemented {done} of {len(impls)} impl node(s)")
        for finding in verification.errors:
            self._emit("warning", f"unresolved: {finding.node_id}: {finding.message}")
        for finding in verification.review_gaps:
            if finding in verification.provenance_gaps:
                self._emit("warning", f"provenance review: {finding.message}")
            else:
                self._emit("warning", f"review: {finding.node_id}: {finding.message}")
        if verification.assumptions:
            self._emit(
                "log",
                f"{len(verification.assumptions)} assumption(s) the skill does not state; "
                f"listed for review in the compile output and compile-report.md",
            )
        if verification.prefix is not None:
            self._emit("log", f"deterministic prefix on the sample: {verification.prefix.summary}")
        if verification.tests is not None and not verification.tests.passed:
            self._emit(
                "warning",
                f"generated tests {verification.tests.status}: {verification.tests.summary}",
            )

    def _deliverable_repairer(
        self,
        base_dir: Path | None,
        skill_dir: Path,
        before_verify: Callable[[Path], object] | None = None,
        grounding: Grounding | None = None,
        portable_judges: bool = False,
        web_research: api_docs.WebResearch | None = None,
        spec_store: SpecStore | None = None,
    ) -> Callable[[Path], str | None]:
        """A repair callback for drivers that can resume their conversation.

        The driver calls it with the produced pipeline.yaml path at each
        natural stop. A ``None`` return accepts the deliverable; a string
        is appended to the agent's conversation as one user turn and the
        loop continues. What gets bounced is everything
        :func:`~rote.compiler.verify.verify_deliverable` rejects: invalid
        yaml with the verbatim pydantic text (seen in real compiles:
        ``.ts`` module refs in ``impl:``, dict-typed input/output entries),
        then contract errors and red generated tests. Up to
        :data:`MAX_VALIDATION_REPAIRS` rounds of each kind, counted apart (an
        IR that a contract fix broke still goes back); after that the callback
        accepts the deliverable as-is, the orchestrator fails an invalid
        IR with its last error, and anything else is reported.

        ``before_verify`` runs on the artifact dir first: ``--system1``
        restores its decisions there, so what is verified (and repaired)
        is the pipeline rote will actually ship.
        """
        attempts = 0
        # Its own budget: an invalid IR is fatal once the agent stops, so a
        # fix that breaks pipeline.yaml in the last contract round must
        # still go back rather than end the paid run on a validation error.
        invalid_attempts = 0
        # Each proposal goes back once. The agent may rightly keep one (its
        # code cannot meet the skill without the change), and asking again
        # buys a resumed turn whose only right answer is to change nothing.
        sent_proposals: set[tuple[str, str]] = set()

        def _repair(pipeline_yaml_path: Path) -> str | None:
            nonlocal attempts, invalid_attempts
            artifact_dir = pipeline_yaml_path.parent
            if before_verify is not None:
                try:
                    before_verify(artifact_dir)
                except Exception as e:  # noqa: BLE001 (a crash here must not cost the paid run)
                    self._emit("warning", f"preparing the deliverable for verification failed: {e}")
            verification = verify_deliverable(
                artifact_dir,
                base_dir=base_dir,
                run_tests=self.run_generated_tests,
                trust_skill=self.trust_skill,
                skill_dir=skill_dir,
                emit_target=self.emit_target,
                strict_provenance=self.strict_provenance,
                grounding=grounding,
                portable_judges=portable_judges,
                web_research=web_research,
                spec_store=spec_store,
            )
            fresh = [
                f
                for f in verification.findings
                if f.kind != UNAPPLIED_PROPOSAL or (f.node_id, f.message) not in sent_proposals
            ]
            verification = dataclasses.replace(verification, findings=fresh)
            instruction = verification.repair_instruction(artifact_dir)
            if instruction is None:
                return None
            sent_proposals.update(
                (f.node_id, f.message) for f in fresh if f.kind == UNAPPLIED_PROPOSAL
            )
            invalid = verification.pipeline_error is not None
            if (invalid_attempts if invalid else attempts) >= MAX_VALIDATION_REPAIRS:
                return None
            if portable_judges and verification.pipeline_error is not None:
                # System-1 restores pipeline.yaml's decisions after every stop
                # and lays text.yaml over it, so an edit to pipeline.yaml alone
                # is undone and the same error comes back.
                instruction += SYSTEM1_REPAIR_NOTE
            if portable_judges:
                # A refused input field is not a verification finding: say why,
                # or the agent sees only the field still missing.
                instruction += system1.added_inputs.refusal_note(artifact_dir) or ""
            if invalid:
                invalid_attempts += 1
            else:
                attempts += 1
            if invalid:
                what = "pipeline.yaml failed validation"
            else:
                problems = len(verification.errors)
                gaps = len(verification.provenance_gaps)
                proposed = sum(1 for f in verification.review_gaps if f.kind == UNAPPLIED_PROPOSAL)
                dead = len(verification.review_gaps) - gaps - proposed
                tests = verification.tests
                red = tests is not None and tests.status == "failed"
                what = (
                    f"the deliverable failed verification ({problems} contract "
                    f"problem(s){f', {gaps} provenance gap(s)' if gaps else ''}"
                    f"{f', {dead} review gap(s)' if dead else ''}"
                    f"{f', {proposed} unapplied proposal(s)' if proposed else ''}"
                    f"{', generated tests failing' if red else ''})"
                )
            attempt = invalid_attempts if invalid else attempts
            self._emit("warning", f"{what}; asking the compiler to repair (attempt {attempt})")
            return instruction

        return _repair

    @staticmethod
    def _completion_message(result: DriverResult) -> str:
        """One-line completion summary carrying token / cost figures.

        Reads whatever the driver reported: the api driver stamps
        ``input_tokens`` / ``output_tokens``; the subprocess drivers stamp
        ``cost_usd`` / ``num_turns``. Absent fields are simply omitted.

        ``in=`` is the whole input volume, cache writes and cache reads
        included. A prompt-cached driver puts almost nothing in the plain
        ``input_tokens`` field, so reporting that alone announced a
        22-minute compilation as ``in=113``.
        """
        usage = Compiler._usage_summary(result.metadata)
        return "; ".join([f"compiled via {result.driver_name}", *([usage] if usage else [])])

    @staticmethod
    def _usage_summary(meta: dict[str, Any]) -> str:
        """``tokens ...; turns=N; cost=$X`` from what the driver reported ("" if nothing)."""
        parts: list[str] = []
        in_tok = meta.get("input_tokens")
        out_tok = meta.get("output_tokens")
        cached = int(meta.get("cache_write_tokens") or 0) + int(meta.get("cache_read_tokens") or 0)
        if in_tok is not None or out_tok is not None or cached:
            total_in = int(in_tok or 0) + cached
            cached_note = f" ({cached} cached)" if cached else ""
            output = "pending" if meta.get("usage_complete") is False else str(out_tok or 0)
            parts.append(f"tokens in={total_in}{cached_note} out={output}")
        # claude reports num_turns; the in-process drivers report the same
        # count (assistant turns) as iterations.
        turns = meta.get("num_turns", meta.get("iterations"))
        if turns is not None:
            parts.append(f"turns={turns}")
        if meta.get("cost_usd") is not None:
            parts.append(f"cost=${meta['cost_usd']}")
        return "; ".join(parts)

    def _plan_update(self, output_dir: Path, skill_md_text: str) -> tuple[Pipeline, UpdatePlan]:
        """Load the previous run and diff the skill against its provenance."""
        prev_yaml = output_dir / "pipeline.yaml"
        prev_prov_path = output_dir / PROVENANCE_FILENAME
        if not prev_yaml.is_file():
            raise CompilerError(
                f"--update requires a previous compilation in {output_dir} "
                f"(no pipeline.yaml found). Run a full compilation first."
            )
        if not prev_prov_path.is_file():
            raise CompilerError(
                f"--update requires the previous run's provenance sidecar, but "
                f"{prev_prov_path} is missing (the pipeline predates provenance "
                f"stamping). Run one full compilation to establish it."
            )
        try:
            prev_pipeline = load_pipeline(prev_yaml)
        except Exception as e:
            raise CompilerError(f"Previous pipeline at {prev_yaml} is invalid: {e}") from e
        try:
            prev_prov = load_provenance(prev_prov_path)
        except Exception as e:
            raise CompilerError(f"Provenance sidecar {prev_prov_path} is invalid: {e}") from e
        return prev_pipeline, build_update_plan(prev_pipeline, prev_prov, skill_md_text)

    @staticmethod
    def _materialize_update_context(work_dir: Path, output_dir: Path, plan: UpdatePlan) -> str:
        """Copy previous artifacts + the brief into the work dir.

        Drivers only expose the skill, rubric, and work directories to
        the agent, so the previous pipeline and stubs must travel into
        the work dir to be readable. Returns the extra prompt
        instructions pointing the agent at the brief.
        """
        ctx_dir = work_dir / UPDATE_CONTEXT_DIRNAME
        ctx_dir.mkdir(parents=True)
        shutil.copy2(output_dir / "pipeline.yaml", ctx_dir / "pipeline.yaml")
        for sub in ("extracted", "signatures"):
            prev_sub = output_dir / sub
            if prev_sub.is_dir():
                shutil.copytree(prev_sub, ctx_dir / sub)
        (ctx_dir / "UPDATE.md").write_text(render_update_brief(plan), encoding="utf-8")
        return (
            f"IMPORTANT: This is an incremental UPDATE run, not a from-scratch "
            f"compilation. Before starting the procedure, read "
            f"{ctx_dir}/UPDATE.md — it lists exactly which source sections "
            f"changed and which existing nodes must be preserved verbatim. "
            f"Start from the previous pipeline at {ctx_dir}/pipeline.yaml and "
            f"make the minimal changes the brief describes."
        )

    @staticmethod
    def _materialize_probe_context(work_dir: Path, probe_dir: Path) -> str | None:
        """Copy baseline probe artifacts into the work dir, if any exist.

        A prior ``rote baseline`` run recorded the skill's *actual* MCP
        traffic (real inputs and result payloads) and the schemas
        inferred from it. Those artifacts are ground truth the agent
        must not have to guess at, so they travel into the work dir the
        same way update context does. Returns the extra prompt
        instructions, or ``None`` when no artifacts are present (the
        common no-baseline case — never an error).
        """
        artifact_names = ("observed-tools.json", "inferred-schemas.json")
        present = [name for name in artifact_names if (probe_dir / name).is_file()]
        if not present:
            return None
        ctx_dir = work_dir / PROBE_CONTEXT_DIRNAME
        ctx_dir.mkdir(parents=True)
        for name in present:
            shutil.copy2(probe_dir / name, ctx_dir / name)
        return (
            f"GROUND TRUTH from an instrumented run of this exact skill is "
            f"available in {ctx_dir}/ — read it before classifying nodes:\n"
            f"- observed-tools.json: every MCP tool the skill actually "
            f"called (server, tool, real input, real result payload).\n"
            f"- inferred-schemas.json: per-tool input/output JSON Schemas "
            f"inferred from those payloads.\n"
            f"Use it three ways: (1) every observed (server, tool) that "
            f"corresponds to a pipeline step MUST appear as that node's "
            f"mcp: binding — observed traffic with no binding is a "
            f"compilation bug; (2) prefer the observed payload shapes over "
            f"guesses when designing signature_spec schemas and data-flow "
            f"inputs; (3) when you implement extracted/ modules, use the "
            f"real payloads as worked examples in docstrings and, if you "
            f"write tests, as golden fixtures. Observation beats "
            f"inference: if the skill prose and the observed traffic "
            f"disagree, trust the traffic and note the discrepancy in the "
            f"compilation report."
        )

    def _pin_loop_tools(
        self, pipeline_yaml: Path, pipeline: Pipeline, tools: tool_grounding.ToolSchemas
    ) -> Pipeline:
        """Pin each agent-loop tool to the one grounded server that offers it.

        One place for every path (legacy, System-1, update): after the agent's
        last stop and System-1's settle, before the final verification, so
        what is verified is what ships. A pin already in the IR wins.
        """
        pins = tool_grounding.loop_tool_pins(pipeline, tools)
        if not pins:
            return pipeline
        for node_id, pinned in sorted(pins.items()):
            servers = ", ".join(f"{t} -> {s}" for t, s in sorted(pinned.items()))
            self._emit("log", f"tool schemas: pinned {node_id}'s tools: {servers}")
        if not tool_grounding.write_loop_pins(pipeline_yaml, pipeline, pins):
            pinned_pipeline = tool_grounding.with_loop_pins(pipeline, pins)
            save_pipeline(pinned_pipeline, pipeline_yaml)
            self._emit(
                "warning",
                "pipeline.yaml was rewritten whole to add tool_servers (its layout did not "
                "allow a line edit), so its YAML comments were not kept",
            )
        return load_pipeline(pipeline_yaml)

    @staticmethod
    def _repoint_source_skill(
        pipeline_yaml: Path, pipeline: Pipeline, skill_dir: Path, output_dir: Path
    ) -> Pipeline:
        """Rewrite ``source_skill`` to resolve from the pipeline.yaml's home.

        Relative when possible (keeps checked-in examples portable),
        absolute when the two paths share no usable root. The rewrite is
        a surgical single-line substitution so the agent's YAML
        formatting survives; the result is re-validated and the original
        text restored if the substitution somehow broke the file (e.g. a
        multiline ``source_skill`` scalar), so this can never turn a
        good compilation into a broken one.
        """
        try:
            source_ref = os.path.relpath(skill_dir, output_dir)
        except ValueError:  # e.g. different drives on Windows
            source_ref = str(skill_dir)

        original = pipeline_yaml.read_text(encoding="utf-8")
        replacement = f"source_skill: {source_ref}"
        text, n = re.subn(r"(?m)^source_skill:[^\n]*$", replacement, original, count=1)
        if n == 0:
            # The agent omitted the field; add it after the top-level
            # name: line (guaranteed present — the IR requires name).
            text, n = re.subn(r"(?m)^(name:[^\n]*)$", rf"\1\n{replacement}", original, count=1)
        if n == 0:
            return pipeline  # nowhere safe to write; keep the agent's value
        pipeline_yaml.write_text(text, encoding="utf-8")
        try:
            return load_pipeline(pipeline_yaml)
        except Exception:
            pipeline_yaml.write_text(original, encoding="utf-8")
            return pipeline

    @staticmethod
    def _repoint_sidecar_source_skill(
        sidecar_path: Path, skill_dir: Path, output_dir: Path
    ) -> None:
        """Rewrite the eval sidecar's ``source_skill`` the same way.

        The sidecar suffers the identical dead-pointer failure as
        pipeline.yaml (the agent records the path relative to its temp
        work dir), just with a quieter blast radius: its field is
        documentary today, but a stale path checked into an example is a
        bug report waiting to happen. Same surgical substitution, same
        restore-on-breakage guarantee, validated with the sidecar's own
        loader.
        """
        if not sidecar_path.is_file():
            return
        try:
            source_ref = os.path.relpath(skill_dir, output_dir)
        except ValueError:
            source_ref = str(skill_dir)

        original = sidecar_path.read_text(encoding="utf-8")
        replacement = f"source_skill: {source_ref}"
        text, n = re.subn(r"(?m)^source_skill:[^\n]*$", replacement, original, count=1)
        if n == 0:
            text, n = re.subn(r"(?m)^(version:[^\n]*)$", rf"\1\n{replacement}", original, count=1)
        if n == 0:
            return
        sidecar_path.write_text(text, encoding="utf-8")
        try:
            load_eval_estimates(sidecar_path)
        except Exception:
            sidecar_path.write_text(original, encoding="utf-8")

    @classmethod
    def _keep_rejected_work(cls, work_dir: Path, output_dir: Path) -> Path:
        """Move a paid run's work out of the temp dir before it is deleted.

        A rejected deliverable is still the output of a real-money agent
        run, and usually a small fix away from usable. It goes to a
        ``<output_dir>.rejected`` sibling, never into ``output_dir``: a
        re-run into an existing ``--out`` must not replace the last good
        pipeline.yaml and extracted/ (possibly with stubs the user filled in
        since) with work that failed validation. Only the previous rejected
        copy is replaced.
        """
        target = output_dir.with_name(output_dir.name + ".rejected")
        if target.exists():
            shutil.rmtree(target)
        cls._move_work_dir_to_output(work_dir, target)
        return target

    @staticmethod
    def _merge_work_dir_to_output(work_dir: Path, output_dir: Path) -> None:
        """File-level merge of ``work_dir`` into ``output_dir``.

        Update runs produce only what the agent rewrote; everything it
        left alone (unchanged extracted modules — possibly filled in by
        the user since compilation) must survive in the output. That
        rules out :meth:`_move_work_dir_to_output`, whose top-level
        directory replacement would delete the untouched siblings.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        for item in sorted(work_dir.rglob("*")):
            if item.is_dir():
                continue
            target = output_dir / item.relative_to(work_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                target.unlink()
            shutil.move(str(item), str(target))

    @staticmethod
    def _move_work_dir_to_output(work_dir: Path, output_dir: Path) -> None:
        """Move every entry in ``work_dir`` into ``output_dir``.

        Existing entries with the same name in ``output_dir`` are
        replaced. We use ``shutil.move`` per-entry rather than moving
        the whole directory because ``output_dir`` may already exist
        and may contain unrelated files.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        for item in work_dir.iterdir():
            target = output_dir / item.name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            shutil.move(str(item), str(target))


__all__ = [
    "Compiler",
    "CompilationResult",
    "CompilerError",
]
