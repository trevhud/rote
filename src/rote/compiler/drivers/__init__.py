"""Compiler drivers — pluggable backends for running the compiler agent.

Every driver implements the :class:`CompilerDriver` Protocol and is
registered in :data:`DRIVERS`. The CLI dispatches to a driver by name
(``rote compile --agent claude``) or by auto-detection
(:func:`auto_detect`).

See ``docs/agent-runtime.md`` for the full design record, including the
rationale for the three-driver lineup (Claude Code, Codex CLI,
Anthropic SDK) and the explicit non-use of ``claude-agent-sdk``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from rote.compiler.events import EventCallback

# ───────── Shared types ─────────


class DriverError(RuntimeError):
    """Raised when a driver fails to produce a compiled pipeline.yaml.

    The message is user-facing and should explain what went wrong and
    what the user can do about it. Driver implementations should attach
    any captured stdout / stderr via the ``details`` attribute.
    """

    def __init__(self, message: str, *, details: str | None = None) -> None:
        super().__init__(message)
        self.details = details


@dataclass
class DriverResult:
    """The output of a successful driver run.

    Attributes
    ----------
    pipeline_yaml_path
        Path to the produced ``pipeline.yaml`` inside ``work_dir``.
    work_dir
        Scratch directory the driver wrote into. May contain additional
        artifacts (``extracted/*.py``, ``signatures/*.py``, etc.) that
        the compiler orchestrator will move into the user's output
        directory.
    driver_name
        The name of the driver that produced this result (e.g.
        ``"claude"``, ``"codex"``, ``"api"``).
    metadata
        Free-form dict of driver-specific metadata: token counts,
        estimated cost, duration, model name, session id, etc. Used for
        the Phase 7 compilation report but not the IR itself.
    """

    pipeline_yaml_path: Path
    work_dir: Path
    driver_name: str
    metadata: dict[str, Any] = field(default_factory=dict)


class CompilerDriver(Protocol):
    """Protocol every compiler driver implements.

    The driver's only responsibility is: **given a source skill and the
    rote-compile rubric, run the agent loop and make sure
    ``work_dir/pipeline.yaml`` exists when it returns.** How it gets
    there — subprocess, in-process, remote — is the driver's business.
    """

    name: str
    """Short identifier used in the CLI (``--agent <name>``)."""

    def is_available(self) -> tuple[bool, str]:
        """Check whether this driver can run right now.

        Returns
        -------
        tuple[bool, str]
            ``(available, reason)``. When ``available`` is ``True``, the
            reason may be an empty string. When ``False``, ``reason`` is
            a user-facing message explaining what needs to happen
            (install the CLI, run login, set an env var).
        """
        ...

    async def run(
        self,
        skill_dir: Path,
        compiler_skill_dir: Path,
        work_dir: Path,
        extra_instructions: str | None = None,
        on_event: EventCallback | None = None,
    ) -> DriverResult:
        """Run the compiler agent against ``skill_dir``.

        Parameters
        ----------
        skill_dir
            The source skill bundle to compile (read-only to the agent).
        compiler_skill_dir
            The ``rote-compile`` skill with SKILL.md and references/.
        work_dir
            A scratch directory where the agent writes ``pipeline.yaml``
            and any extracted/signature stubs. The caller ensures this
            directory exists and is empty — except in incremental-update
            runs, where the orchestrator pre-materializes read-only
            context the instructions point at.
        extra_instructions
            Optional run-specific instructions appended to the agent's
            task prompt (e.g. the incremental-update pointer). The
            orchestrator only passes this when it has something to say,
            so drivers that predate the parameter keep working on full
            runs.
        on_event
            Optional live-progress sink. Drivers fire
            :class:`~rote.compiler.events.CompilationEvent`\\ s through it
            as the run proceeds (``turn``, ``tool``, ``phase``). Same
            backward-compat precedent as ``extra_instructions``: the
            orchestrator only passes it when a consumer wants progress,
            so drivers that predate the parameter keep working. Fire
            events via :func:`~rote.compiler.events.emit_safely` so a
            raising sink can't kill the run.

        Drivers that can resume their conversation may additionally
        accept a keyword-only ``repair`` callback (the in-process
        drivers do): it is called with the produced ``pipeline.yaml``
        path at each natural stop — a returned string is appended to
        the agent's conversation as one user turn and the loop
        continues; ``None`` accepts the deliverable. The orchestrator
        passes it only to drivers whose ``run`` signature declares it,
        so drivers that predate the parameter keep working.

        Returns
        -------
        DriverResult
            A result pointing at ``work_dir/pipeline.yaml``.

        Raises
        ------
        DriverError
            If the driver cannot produce a valid ``pipeline.yaml``.
        """
        ...


# ───────── Driver registry ─────────


def _claude_driver_factory(**kwargs: Any) -> CompilerDriver:
    from rote.compiler.drivers.claude import ClaudeDriver

    return ClaudeDriver(**kwargs)


def _codex_driver_factory(**kwargs: Any) -> CompilerDriver:
    from rote.compiler.drivers.codex import CodexDriver

    return CodexDriver(**kwargs)


def _anthropic_driver_factory(**kwargs: Any) -> CompilerDriver:
    from rote.compiler.drivers.anthropic_api import AnthropicApiDriver

    return AnthropicApiDriver(**kwargs)


def _openai_driver_factory(**kwargs: Any) -> CompilerDriver:
    from rote.compiler.drivers.openai_api import OpenAIApiDriver

    return OpenAIApiDriver(**kwargs)


#: Name → factory. Keep the values lazy so we don't pay the import cost
#: of the anthropic SDK on every CLI invocation. Factories accept
#: ``**kwargs`` which are forwarded to the driver constructor (e.g.
#: ``model``, ``max_turns``).
DRIVERS: dict[str, Callable[..., CompilerDriver]] = {
    "claude": _claude_driver_factory,
    "codex": _codex_driver_factory,
    "api": _anthropic_driver_factory,
    "openai-api": _openai_driver_factory,
}

#: Order used by :func:`auto_detect`. Claude first because Claude Max/Pro
#: is the most likely subscription rote's audience already has.
#: ``openai-api`` is deliberately excluded — it has no default model, so
#: it can't be probed with a zero-arg construction; it is explicit-only
#: (``--agent openai-api --model <id>``).
AUTO_DETECT_ORDER: tuple[str, ...] = ("claude", "codex", "api")


def get_driver(name: str, **kwargs: Any) -> CompilerDriver:
    """Return a driver instance by name, configured with ``kwargs``.

    ``kwargs`` are forwarded to the driver's constructor (e.g. ``model``
    for ``ClaudeDriver`` and ``AnthropicApiDriver``). Drivers ignore
    kwargs they don't understand via ``**kwargs`` absorption in their
    factories.

    Raises :class:`KeyError` with a helpful message when the name is
    unknown.
    """
    try:
        factory = DRIVERS[name]
    except KeyError:
        available = ", ".join(sorted(DRIVERS))
        raise KeyError(f"Unknown agent driver {name!r}. Available: {available}") from None
    return factory(**kwargs)


def auto_detect() -> CompilerDriver | None:
    """Return the first available driver in :data:`AUTO_DETECT_ORDER`.

    Returns ``None`` when no driver is available. The CLI translates
    that into a helpful error listing setup instructions for each
    driver.
    """
    for name in AUTO_DETECT_ORDER:
        try:
            driver = DRIVERS[name]()
        except Exception:
            # A factory raising at import time (e.g. a missing optional
            # dep that isn't gracefully handled) should never block
            # auto-detect from trying the next driver.
            continue
        try:
            available, _reason = driver.is_available()
        except Exception:
            continue
        if available:
            return driver
    return None


def available_drivers() -> list[tuple[str, bool, str]]:
    """Return a diagnostic list of ``(name, available, reason)`` triples.

    Used by the CLI's ``--agent auto`` failure path to produce a clear
    message telling the user what each driver needs.
    """
    out: list[tuple[str, bool, str]] = []
    for name in AUTO_DETECT_ORDER:
        try:
            driver = DRIVERS[name]()
            available, reason = driver.is_available()
            out.append((name, available, reason))
        except Exception as e:
            out.append((name, False, f"driver failed to load: {e}"))
    return out


__all__ = [
    "DriverError",
    "DriverResult",
    "CompilerDriver",
    "DRIVERS",
    "AUTO_DETECT_ORDER",
    "get_driver",
    "auto_detect",
    "available_drivers",
]
