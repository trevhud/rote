"""Name collisions are found while planning (rote.clients.collisions)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rote.clients import ImportCandidate, find_collisions
from rote.clients.normalize import server_identifier


def _c(original: str, url: str, **kw: Any) -> ImportCandidate:
    return ImportCandidate(
        client="claude-code",
        name=server_identifier(original),
        original_name=original,
        scope="user",
        origin=Path("/x"),
        transport="streamable-http",
        url=url,
        **kw,
    )


def test_two_names_that_map_to_one_identifier_collide() -> None:
    [collision] = find_collisions(
        [_c("acme-crm", "https://a.example/mcp"), _c("acme_crm", "https://b.example/mcp")], {}
    )
    assert collision.kind == "same-identifier"
    assert collision.name == "acme_crm"
    assert [c.original_name for c in collision.candidates] == ["acme-crm", "acme_crm"]


def test_one_server_found_twice_is_not_a_collision() -> None:
    same = "https://a.example/mcp"
    assert find_collisions([_c("acme-crm", same), _c("acme_crm", same)], {}) == ()


def test_a_name_registered_with_another_url_collides() -> None:
    figma = _c("figma", "https://mcp.figma.com/mcp")
    [collision] = find_collisions([figma], {"figma": "https://old.example/mcp"})
    assert (collision.kind, collision.registered_host) == ("registered-elsewhere", "old.example")
    assert find_collisions([figma], {"figma": "https://mcp.figma.com/mcp"}) == ()


def test_unused_or_blocked_entries_take_no_part() -> None:
    candidates = [
        _c("acme-crm", "https://a.example/mcp"),
        _c("acme_crm", "https://b.example/mcp", inactive="shadowed"),
        _c("acme.crm", "https://c.example/mcp", blocked="stdio"),
    ]
    assert find_collisions(candidates, {}) == ()
