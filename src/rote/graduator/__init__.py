"""The graduator orchestrator.

The :class:`Graduator` is the high-level entry point used by
``rote graduate``. It:

1. Picks a driver (auto-detect or explicit ``--agent``).
2. Locates the rote-graduate skill bundle.
3. Creates a temp work directory.
4. Calls ``driver.run()`` to produce ``pipeline.yaml`` (and any
   extracted modules / signature stubs).
5. Validates the produced ``pipeline.yaml`` via
   :func:`rote.ir.load_pipeline`.
6. Moves the work directory contents into the user's output directory.
7. Returns a :class:`GraduationResult`.

This module deliberately knows nothing about specific runtimes
(Temporal, Inngest, etc.) — that's the adapter layer's job. The
graduator's only output is the validated IR + the stub files;
``rote.adapters`` consumes those to emit runnable code.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rote.eval.sidecar import EVAL_SIDECAR_FILENAME, load_eval_estimates
from rote.graduator.drivers import (
    DriverError,
    DriverResult,
    GraduatorDriver,
    auto_detect,
    available_drivers,
    get_driver,
)
from rote.graduator.events import EventCallback, GraduationEvent, emit_safely
from rote.graduator.update import (
    UPDATE_CONTEXT_DIRNAME,
    UpdatePlan,
    build_update_plan,
    render_update_brief,
)
from rote.ir import Pipeline, load_pipeline
from rote.skill_source import PROVENANCE_FILENAME, load_provenance, write_provenance


class GraduatorError(RuntimeError):
    """High-level graduator failure.

    Wraps driver errors and validation errors with a user-facing
    message. The CLI prints the message to stderr and exits non-zero.
    """


@dataclass
class GraduationResult:
    """The output of a successful graduation."""

    pipeline: Pipeline
    output_dir: Path
    driver_name: str
    driver_metadata: dict[str, Any] = field(default_factory=dict)


def _default_graduator_skill_dir() -> Path:
    """Locate the rote-graduate skill bundled with the rote source.

    Two layouts exist, checked in order:

    1. **Wheel install.** ``pyproject.toml`` force-includes the
       repo-root ``skills/rote-graduate/`` into the wheel at
       ``rote/skills/rote-graduate/``, so the bundle sits next to the
       package's ``__file__``.
    2. **Editable install / source checkout.** The skill lives at
       ``<repo>/skills/rote-graduate/``; ``rote.__file__`` is
       ``<repo>/src/rote/__init__.py``, so the repo root is three
       parents up.
    """
    import rote

    rote_pkg_dir = Path(rote.__file__).resolve().parent
    candidates = [
        rote_pkg_dir / "skills" / "rote-graduate",
        rote_pkg_dir.parent.parent / "skills" / "rote-graduate",
    ]
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "SKILL.md").is_file():
            return candidate
    searched = ", ".join(str(c) for c in candidates)
    raise GraduatorError(
        f"Could not locate the rote-graduate skill (searched: {searched}). "
        "Pass an explicit graduator_skill_dir to Graduator() pointing "
        "at a rote-graduate skill bundle."
    )


class Graduator:
    """High-level orchestrator for graduating a skill into a pipeline."""

    def __init__(
        self,
        agent: str | None = None,
        graduator_skill_dir: Path | None = None,
        model: str | None = None,
        on_event: EventCallback | None = None,
        driver_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        agent
            Driver name (``"claude"``, ``"codex"``, ``"api"``) or
            ``None`` for auto-detect.
        graduator_skill_dir
            Path to the rote-graduate skill bundle. Defaults to the
            bundled one inside the rote source tree.
        model
            Override the LLM model the driver uses for the graduator
            (e.g. ``"claude-opus-4-6"`` to use Opus instead of the
            default Sonnet). ``None`` uses the driver's default.
        on_event
            Optional live-progress sink threaded to the driver and fired
            with orchestrator-level events (``log``, ``artifact``,
            ``complete`` / ``error``). Always invoked through
            :func:`~rote.graduator.events.emit_safely`, so a raising sink
            can't sink a paid run.
        driver_kwargs
            Extra keyword arguments merged into the driver constructor
            call (on top of ``model``). The cloud runner uses this to
            pass the api driver's ``base_url`` / ``default_headers`` for
            AI Gateway routing. Keys collide-last: an explicit
            ``driver_kwargs["model"]`` overrides the ``model`` argument.
        """
        self.agent = agent
        self._explicit_graduator_skill_dir = graduator_skill_dir
        self.model = model
        self.on_event = on_event
        self.driver_kwargs = driver_kwargs

    @property
    def graduator_skill_dir(self) -> Path:
        if self._explicit_graduator_skill_dir is not None:
            return self._explicit_graduator_skill_dir
        return _default_graduator_skill_dir()

    def select_driver(self) -> GraduatorDriver:
        """Choose a driver based on the agent setting or auto-detect.

        When a non-default ``model`` was passed to the ``Graduator``
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

        if self.agent:
            try:
                driver = get_driver(self.agent, **driver_kwargs)
            except KeyError as e:
                raise GraduatorError(str(e.args[0])) from None
            available, reason = driver.is_available()
            if not available:
                raise GraduatorError(f"Driver {self.agent!r} is not available: {reason}")
            return driver

        probe = auto_detect()
        if probe is None:
            lines = [
                "No graduator driver is available. Tried:",
            ]
            for name, _avail, reason in available_drivers():
                lines.append(f"  - {name}: {reason}")
            lines.append("")
            lines.append(
                "Install one of the supported coding agent CLIs, or set "
                "ANTHROPIC_API_KEY and `pip install rote[api]`."
            )
            raise GraduatorError("\n".join(lines))

        # Re-construct the detected driver with user-specified kwargs
        # (e.g. model override) instead of returning the probe instance
        # directly.
        return get_driver(probe.name, **driver_kwargs)

    def _emit(self, type_: str, message: str, **fields: Any) -> None:
        """Fire an orchestrator-level progress event (no-op without a sink)."""
        emit_safely(
            self.on_event,
            GraduationEvent(type=type_, ts=time.time(), message=message, **fields),  # type: ignore[arg-type]
        )

    async def graduate(
        self,
        skill_dir: Path | str,
        output_dir: Path | str,
        update: bool = False,
        extra_instructions: str | None = None,
    ) -> GraduationResult:
        """Run the full graduation flow.

        Parameters
        ----------
        skill_dir
            Path to the source skill bundle (must contain a SKILL.md).
        output_dir
            Where to write the graduated artifacts. Created if missing;
            existing files of the same name are overwritten.
        update
            Incremental mode: require a previous graduation (with its
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

        Returns
        -------
        GraduationResult
            A validated :class:`Pipeline` plus the output directory and
            driver metadata.

        Raises
        ------
        GraduatorError
            For any user-actionable failure: missing skill bundle,
            unavailable driver, driver failure, invalid produced
            pipeline.yaml — or, in update mode, a missing previous run
            or an update that dropped provenance-preserved nodes.
        """
        skill_dir = Path(skill_dir).resolve()
        output_dir = Path(output_dir).resolve()

        try:
            return await self._graduate_inner(skill_dir, output_dir, update, extra_instructions)
        except GraduatorError as e:
            self._emit("error", f"graduation failed: {e}")
            raise

    async def _graduate_inner(
        self,
        skill_dir: Path,
        output_dir: Path,
        update: bool,
        extra_instructions: str | None = None,
    ) -> GraduationResult:
        """The graduation body, bracketed by ``graduate``'s error event."""
        if not skill_dir.is_dir():
            raise GraduatorError(f"Skill directory does not exist: {skill_dir}")
        if not (skill_dir / "SKILL.md").is_file():
            raise GraduatorError(f"Not a skill bundle (no SKILL.md found): {skill_dir}")

        skill_md_text = (skill_dir / "SKILL.md").read_text(encoding="utf-8")

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
                return GraduationResult(
                    pipeline=prev_pipeline,
                    output_dir=output_dir,
                    driver_name="(no-op)",
                    driver_metadata={
                        "update": "no source sections changed; previous pipeline kept"
                    },
                )

        driver = self.select_driver()
        self._emit("log", f"driver selected: {driver.name}")

        with tempfile.TemporaryDirectory(prefix="rote-graduate-") as work_dir_str:
            work_dir = Path(work_dir_str)

            instructions: list[str] = []
            if extra_instructions:
                instructions.append(extra_instructions)
            if plan is not None:
                instructions.append(self._materialize_update_context(work_dir, output_dir, plan))
            run_kwargs: dict[str, Any] = {}
            if instructions:
                run_kwargs["extra_instructions"] = "\n\n".join(instructions)

            try:
                result = await driver.run(
                    skill_dir=skill_dir,
                    graduator_skill_dir=self.graduator_skill_dir,
                    work_dir=work_dir,
                    on_event=self.on_event,
                    **run_kwargs,
                )
            except DriverError as e:
                detail = f"\n\n{e.details}" if getattr(e, "details", None) else ""
                raise GraduatorError(f"Driver {driver.name!r} failed: {e}{detail}") from e

            try:
                pipeline = load_pipeline(result.pipeline_yaml_path)
            except Exception as e:
                raise GraduatorError(
                    f"Driver {driver.name!r} produced an invalid "
                    f"pipeline.yaml at {result.pipeline_yaml_path}: {e}"
                ) from e

            if plan is not None:
                produced_ids = {n.id for n in pipeline.nodes}
                missing = [nid for nid in plan.preserved_node_ids if nid not in produced_ids]
                if missing:
                    raise GraduatorError(
                        f"Update run dropped or renamed nodes whose source sections "
                        f"did not change: {missing}. Node ids version the emitted "
                        f"workflow, so this would orphan in-flight runs. The previous "
                        f"output was left untouched — re-run, or run a full graduation "
                        f"if the restructure is intentional."
                    )

            # Stamp provenance: hash every SKILL.md section so a later
            # `graduate --update` can tell exactly which nodes' source
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
            return GraduationResult(
                pipeline=pipeline,
                output_dir=output_dir,
                driver_name=result.driver_name,
                driver_metadata=result.metadata,
            )

    @staticmethod
    def _completion_message(result: DriverResult) -> str:
        """One-line completion summary carrying token / cost figures.

        Reads whatever the driver reported: the api driver stamps
        ``input_tokens`` / ``output_tokens``; the subprocess drivers stamp
        ``cost_usd`` / ``num_turns``. Absent fields are simply omitted.
        """
        meta = result.metadata
        parts = [f"graduated via {result.driver_name}"]
        in_tok = meta.get("input_tokens")
        out_tok = meta.get("output_tokens")
        if in_tok is not None or out_tok is not None:
            parts.append(f"tokens in={in_tok or 0} out={out_tok or 0}")
        if meta.get("num_turns") is not None:
            parts.append(f"turns={meta['num_turns']}")
        if meta.get("cost_usd") is not None:
            parts.append(f"cost=${meta['cost_usd']}")
        return "; ".join(parts)

    def _plan_update(self, output_dir: Path, skill_md_text: str) -> tuple[Pipeline, UpdatePlan]:
        """Load the previous run and diff the skill against its provenance."""
        prev_yaml = output_dir / "pipeline.yaml"
        prev_prov_path = output_dir / PROVENANCE_FILENAME
        if not prev_yaml.is_file():
            raise GraduatorError(
                f"--update requires a previous graduation in {output_dir} "
                f"(no pipeline.yaml found). Run a full graduation first."
            )
        if not prev_prov_path.is_file():
            raise GraduatorError(
                f"--update requires the previous run's provenance sidecar, but "
                f"{prev_prov_path} is missing (the pipeline predates provenance "
                f"stamping). Run one full graduation to establish it."
            )
        try:
            prev_pipeline = load_pipeline(prev_yaml)
        except Exception as e:
            raise GraduatorError(f"Previous pipeline at {prev_yaml} is invalid: {e}") from e
        try:
            prev_prov = load_provenance(prev_prov_path)
        except Exception as e:
            raise GraduatorError(f"Provenance sidecar {prev_prov_path} is invalid: {e}") from e
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
            f"graduation. Before starting the procedure, read "
            f"{ctx_dir}/UPDATE.md — it lists exactly which source sections "
            f"changed and which existing nodes must be preserved verbatim. "
            f"Start from the previous pipeline at {ctx_dir}/pipeline.yaml and "
            f"make the minimal changes the brief describes."
        )

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
        good graduation into a broken one.
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

    @staticmethod
    def _merge_work_dir_to_output(work_dir: Path, output_dir: Path) -> None:
        """File-level merge of ``work_dir`` into ``output_dir``.

        Update runs produce only what the agent rewrote; everything it
        left alone (unchanged extracted modules — possibly filled in by
        the user since graduation) must survive in the output. That
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
    "Graduator",
    "GraduationResult",
    "GraduatorError",
]
