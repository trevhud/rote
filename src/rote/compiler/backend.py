"""The ``--backend`` a pipeline was compiled for, recorded beside its IR.

A compile for ``--backend api`` writes vendor-API implementations that
only the api backend emits. The backend is a deployment fact, not
behavior, so it stays out of ``pipeline.yaml`` (invariant #1). It lives in
``backend.json`` next to it instead, and ``rote emit`` on that IR uses it
unless told otherwise. Without the record, a later ``rote emit`` with no
``--backend`` silently emitted MCP calls in place of the researched code.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, cast

Backend = Literal["mcp", "api"]
BACKENDS: tuple[Backend, ...] = ("mcp", "api")

BACKEND_FILENAME = "backend.json"

#: Runtimes with no MCP client: every external_call runs its ``impl:``
#: whatever the backend (see ``rote.adapters._common.refuse_mcp_only_nodes``).
IMPL_ONLY_RUNTIMES = frozenset({"python", "temporal"})


class BackendRecordError(ValueError):
    """``backend.json`` exists but does not say a known backend."""


def write_backend(compiled_dir: Path, backend: Backend) -> Path:
    path = compiled_dir / BACKEND_FILENAME
    path.write_text(
        json.dumps({"schema_version": 1, "external_backend": backend}, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def read_backend(compiled_dir: Path) -> Backend | None:
    """The recorded backend, or ``None`` when the IR has no record (hand-written, older)."""
    path = compiled_dir / BACKEND_FILENAME
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8")).get("external_backend")
    except (json.JSONDecodeError, AttributeError, UnicodeDecodeError) as e:
        raise BackendRecordError(f"{path} is not a backend record: {e}") from e
    if value not in BACKENDS:
        raise BackendRecordError(
            f"{path} records external_backend {value!r}; expected one of {', '.join(BACKENDS)}"
        )
    return cast(Backend, value)


def bindings_called(runtime: str, backend: str) -> bool:
    """Whether emitted code for ``runtime`` calls a node's ``mcp:`` binding."""
    return backend == "mcp" and runtime not in IMPL_ONLY_RUNTIMES


__all__ = [
    "BACKENDS",
    "BACKEND_FILENAME",
    "IMPL_ONLY_RUNTIMES",
    "Backend",
    "BackendRecordError",
    "bindings_called",
    "read_backend",
    "write_backend",
]
