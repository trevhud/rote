"""Runtime adapters for emitted pipeline code.

Each adapter takes a :class:`rote.ir.Pipeline` and emits runnable code
for a specific durable execution engine. The IR is the source of truth;
adapters are template substitution.

The "two adapters minimum" rule applies: until at least two adapters
work end-to-end, assume the IR shape is secretly leaking the first
runtime's mental model. Inngest is the planned second target after
Temporal.

Adapters are registered in :data:`ADAPTERS` and dispatched by name from
the CLI (``rote emit --runtime <name>``).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from rote.ir import Pipeline


class Adapter(Protocol):
    """Structural protocol every adapter must satisfy."""

    def emit(self, pipeline: Pipeline, output_dir: str | Path) -> dict[str, Path]: ...


def _temporal_adapter_factory() -> Adapter:
    # Lazy import so users who don't use Temporal don't pay the
    # temporalio import cost just for launching the CLI.
    from rote.adapters.temporal import TemporalAdapter

    return TemporalAdapter()


def _cloudflare_adapter_factory() -> Adapter:
    from rote.adapters.cloudflare import CloudflareAdapter

    return CloudflareAdapter()


def _dbos_adapter_factory() -> Adapter:
    from rote.adapters.dbos import DbosAdapter

    return DbosAdapter()


def _dbos_ts_adapter_factory() -> Adapter:
    from rote.adapters.dbos_ts import DbosTsAdapter

    return DbosTsAdapter()


#: Name → factory. Keep the values as zero-arg callables so adapters can
#: lazy-import their heavy dependencies.
ADAPTERS: dict[str, Callable[[], Adapter]] = {
    "temporal": _temporal_adapter_factory,
    "cloudflare": _cloudflare_adapter_factory,
    "dbos": _dbos_adapter_factory,
    "dbos-ts": _dbos_ts_adapter_factory,
}


def get_adapter(name: str) -> Adapter:
    """Return an adapter instance for the given runtime name.

    Raises ``KeyError`` with a helpful message if the runtime is unknown.
    """
    try:
        factory = ADAPTERS[name]
    except KeyError:
        available = ", ".join(sorted(ADAPTERS))
        raise KeyError(f"Unknown runtime {name!r}. Available: {available}") from None
    return factory()
