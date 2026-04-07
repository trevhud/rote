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

import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rote.graduator.drivers import (
    DriverError,
    GraduatorDriver,
    auto_detect,
    available_drivers,
    get_driver,
)
from rote.ir import Pipeline, load_pipeline


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

    For development installs (``pip install -e .``), the skill lives at
    ``<repo>/skills/rote-graduate/``. We resolve it relative to the
    rote package's ``__file__``:

    * ``rote.__file__`` → ``<repo>/src/rote/__init__.py``
    * ``.parent.parent.parent`` → ``<repo>``
    * ``+ "skills/rote-graduate"`` → the skill bundle

    For wheel installs the package layout differs and we'd need
    package_data — that's a v0.1 issue. For now we error out with a
    pointer to the workaround (passing ``graduator_skill_dir`` explicitly).
    """
    import rote

    rote_pkg_dir = Path(rote.__file__).resolve().parent
    repo_root = rote_pkg_dir.parent.parent
    candidate = repo_root / "skills" / "rote-graduate"
    if candidate.is_dir() and (candidate / "SKILL.md").is_file():
        return candidate
    raise GraduatorError(
        f"Could not locate the rote-graduate skill at {candidate}. "
        "If you installed rote from a wheel rather than editable mode, "
        "pass an explicit graduator_skill_dir to Graduator() pointing "
        "at a checkout of the rote source."
    )


class Graduator:
    """High-level orchestrator for graduating a skill into a pipeline."""

    def __init__(
        self,
        agent: str | None = None,
        graduator_skill_dir: Path | None = None,
        model: str | None = None,
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
        """
        self.agent = agent
        self._explicit_graduator_skill_dir = graduator_skill_dir
        self.model = model

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

        if self.agent:
            try:
                driver = get_driver(self.agent, **driver_kwargs)
            except KeyError as e:
                raise GraduatorError(str(e.args[0])) from None
            available, reason = driver.is_available()
            if not available:
                raise GraduatorError(
                    f"Driver {self.agent!r} is not available: {reason}"
                )
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

    async def graduate(
        self,
        skill_dir: Path | str,
        output_dir: Path | str,
    ) -> GraduationResult:
        """Run the full graduation flow.

        Parameters
        ----------
        skill_dir
            Path to the source skill bundle (must contain a SKILL.md).
        output_dir
            Where to write the graduated artifacts. Created if missing;
            existing files of the same name are overwritten.

        Returns
        -------
        GraduationResult
            A validated :class:`Pipeline` plus the output directory and
            driver metadata.

        Raises
        ------
        GraduatorError
            For any user-actionable failure: missing skill bundle,
            unavailable driver, driver failure, or invalid produced
            pipeline.yaml.
        """
        skill_dir = Path(skill_dir).resolve()
        output_dir = Path(output_dir).resolve()

        if not skill_dir.is_dir():
            raise GraduatorError(f"Skill directory does not exist: {skill_dir}")
        if not (skill_dir / "SKILL.md").is_file():
            raise GraduatorError(
                f"Not a skill bundle (no SKILL.md found): {skill_dir}"
            )

        driver = self.select_driver()

        with tempfile.TemporaryDirectory(prefix="rote-graduate-") as work_dir_str:
            work_dir = Path(work_dir_str)

            try:
                result = await driver.run(
                    skill_dir=skill_dir,
                    graduator_skill_dir=self.graduator_skill_dir,
                    work_dir=work_dir,
                )
            except DriverError as e:
                detail = f"\n\n{e.details}" if getattr(e, "details", None) else ""
                raise GraduatorError(
                    f"Driver {driver.name!r} failed: {e}{detail}"
                ) from e

            try:
                pipeline = load_pipeline(result.pipeline_yaml_path)
            except Exception as e:
                raise GraduatorError(
                    f"Driver {driver.name!r} produced an invalid "
                    f"pipeline.yaml at {result.pipeline_yaml_path}: {e}"
                ) from e

            self._move_work_dir_to_output(result.work_dir, output_dir)

            # Re-point the result's path to the moved location for the
            # caller's benefit.
            return GraduationResult(
                pipeline=pipeline,
                output_dir=output_dir,
                driver_name=result.driver_name,
                driver_metadata=result.metadata,
            )

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
