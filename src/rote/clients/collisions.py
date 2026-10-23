"""Where two servers want one rote name: found while planning, never at the write.

Two shapes, both from the first-run design review:

- **same identifier**: different servers map to one rote name
  (``acme-crm`` and ``acme_crm``; a plugin's ``linear`` and the user's own
  ``linear`` pointing elsewhere).
- **registered elsewhere**: the name is already in rote's registry with a
  different URL.

:func:`find_collisions` only finds them. ``rote mcp import`` decides what to
do with each one inside its plan (skip with a note, or a ``--rename``), so
the plan the user confirms already says what happens, and the write never
meets a surprise. Entries the client itself does not use (shadowed,
disabled) and entries that cannot be imported take part in no collision.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from rote.clients.base import ImportCandidate

Kind = Literal["same-identifier", "registered-elsewhere"]


@dataclass(frozen=True)
class Collision:
    name: str
    kind: Kind
    #: The candidates that want ``name``.
    candidates: tuple[ImportCandidate, ...]
    #: For ``registered-elsewhere``: the host already registered under it.
    registered_host: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "candidates": [
                {
                    "client": c.client,
                    "original_name": c.original_name,
                    "scope": c.scope,
                    "host": c.host,
                }
                for c in self.candidates
            ],
            "registered_host": self.registered_host,
        }


def find_collisions(
    candidates: Iterable[ImportCandidate],
    registered: Mapping[str, str],
) -> tuple[Collision, ...]:
    """Collisions among ``candidates`` and against ``registered`` (rote name
    to URL, from ``rote.mcp.registry``). Two candidates with the same URL
    are one server found twice, not a collision, and URLs compare in the
    form two spellings of one endpoint share (``normalize_server_url``: case,
    default port, trailing slash)."""
    from urllib.parse import urlsplit

    from rote.mcp._runtime_helper import normalize_server_url

    by_name: dict[str, list[ImportCandidate]] = {}
    for candidate in candidates:
        if candidate.importable and candidate.inactive is None:
            by_name.setdefault(candidate.name, []).append(candidate)
    found: list[Collision] = []
    for name, group in sorted(by_name.items()):
        urls = {normalize_server_url(c.url) for c in group if c.url is not None}
        if len(urls) > 1:
            found.append(Collision(name, "same-identifier", tuple(group)))
        existing = registered.get(name)
        if existing is not None and normalize_server_url(existing) not in urls:
            host = urlsplit(existing).hostname
            found.append(Collision(name, "registered-elsewhere", tuple(group), host))
    return tuple(found)
