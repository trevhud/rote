"""Runtime adapters for emitted pipeline code.

Each adapter takes a :class:`rote.ir.Pipeline` and emits runnable code
for a specific durable execution engine. The IR is the source of truth;
adapters are template substitution.

The "two adapters minimum" rule applies: until at least two adapters
work end-to-end, assume the IR shape is secretly leaking the first
runtime's mental model. (Satisfied — five adapters share the IR today:
Temporal, Cloudflare, DBOS Python, DBOS TypeScript, and Inngest.)

Adapters are registered in :data:`ADAPTERS` and dispatched by name from
the CLI (``rote emit --runtime <name>``).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from rote.ir import Pipeline


class Adapter(Protocol):
    """Structural protocol every adapter must satisfy."""

    def emit(self, pipeline: Pipeline, output_dir: str | Path) -> dict[str, Path]: ...


#: Options a factory may receive (forwarded from ``get_adapter``). Factories
#: swallow unknown keys via ``**options`` — mirrors the driver-registry
#: convention — so a runtime that doesn't support an option just ignores it.
#: ``external_backend`` ("mcp" | "api") is understood by the DBOS, DBOS-TS,
#: and Inngest adapters.


def _validated_backend(options: dict[str, Any]) -> Literal["mcp", "api"] | None:
    backend = options.get("external_backend")
    if backend is None:
        return None
    if backend not in ("mcp", "api"):
        raise ValueError(f"external_backend must be 'mcp' or 'api', got {backend!r}")
    return cast(Literal["mcp", "api"], backend)


def _temporal_adapter_factory(**options: Any) -> Adapter:
    # Lazy import so users who don't use Temporal don't pay the
    # temporalio import cost just for launching the CLI.
    from rote.adapters.temporal import TemporalAdapter

    return TemporalAdapter()


def _cloudflare_adapter_factory(**options: Any) -> Adapter:
    from rote.adapters.cloudflare import CloudflareAdapter, CloudflareAdapterConfig

    backend = _validated_backend(options)
    mcp_client = options.get("mcp_client")
    if mcp_client not in (None, "direct", "binding"):
        raise ValueError(f"mcp_client must be 'direct' or 'binding', got {mcp_client!r}")
    if backend is None and mcp_client is None:
        return CloudflareAdapter()
    return CloudflareAdapter(
        CloudflareAdapterConfig(
            external_backend=backend or "mcp",
            mcp_client=mcp_client or "direct",
        )
    )


def _source_dir(options: dict[str, Any]) -> Path | None:
    """The ``extracted_source_dir`` option, normalized to a Path.

    Directory of the pipeline.yaml being emitted; Python-emitting
    adapters use the agent-written ``extracted/`` modules found there
    verbatim instead of IR-derived stubs. TS adapters ignore it — a
    Python module can't back a TS step (the compiler would have to
    emit TS implementations; a known, documented gap).
    """
    value = options.get("extracted_source_dir")
    return Path(value) if value is not None else None


def _dbos_adapter_factory(**options: Any) -> Adapter:
    from rote.adapters.dbos import DbosAdapter, DbosAdapterConfig

    backend = _validated_backend(options)
    return DbosAdapter(
        DbosAdapterConfig(
            external_backend=backend or "mcp",
            extracted_source_dir=_source_dir(options),
        )
    )


def _dbos_ts_adapter_factory(**options: Any) -> Adapter:
    from rote.adapters.dbos_ts import DbosTsAdapter, DbosTsAdapterConfig

    backend = _validated_backend(options)
    if backend is None:
        return DbosTsAdapter()
    return DbosTsAdapter(DbosTsAdapterConfig(external_backend=backend))


def _inngest_adapter_factory(**options: Any) -> Adapter:
    from rote.adapters.inngest import InngestAdapter, InngestAdapterConfig

    backend = _validated_backend(options)
    if backend is None:
        return InngestAdapter()
    return InngestAdapter(InngestAdapterConfig(external_backend=backend))


def _python_adapter_factory(**options: Any) -> Adapter:
    from rote.adapters.python import PythonAdapter, PythonAdapterConfig

    return PythonAdapter(PythonAdapterConfig(extracted_source_dir=_source_dir(options)))


#: Name → factory. Factories accept ``**options`` (forwarded from
#: ``get_adapter``) and lazy-import their heavy dependencies.
ADAPTERS: dict[str, Callable[..., Adapter]] = {
    "temporal": _temporal_adapter_factory,
    "cloudflare": _cloudflare_adapter_factory,
    "dbos": _dbos_adapter_factory,
    "dbos-ts": _dbos_ts_adapter_factory,
    "inngest": _inngest_adapter_factory,
    "python": _python_adapter_factory,
}


def get_adapter(name: str, **options: Any) -> Adapter:
    """Return an adapter instance for the given runtime name.

    ``options`` are forwarded to the adapter's factory (e.g.
    ``external_backend="api"`` for the DBOS adapter). A factory ignores
    options it doesn't understand. Raises ``KeyError`` with a helpful
    message if the runtime is unknown.
    """
    try:
        factory = ADAPTERS[name]
    except KeyError:
        available = ", ".join(sorted(ADAPTERS))
        raise KeyError(f"Unknown runtime {name!r}. Available: {available}") from None
    return factory(**options)
