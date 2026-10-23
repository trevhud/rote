"""Shared pieces of the client-source tests: fixture files and a home that refuses.

Every value written here is made up. Sources reach the machine only through
:class:`~rote.clients.ClientContext`, and :func:`refuse_real_home` makes
``Path.home`` raise, so a source that reached for the real home fails the
test instead of reading the developer's configs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rote.clients import ImportCandidate, SourceReading

#: Made-up secrets planted in fixtures. No rendering of a reading may
#: contain one.
SECRET = "sk-literal-NOT-A-REAL-KEY"
QUERY_SECRET = "qk-NOT-REAL"
CLIENT_SECRET = "cs-NOT-REAL-client-secret"
CLIENT_ID = "issued-to-the-client-NOT-REAL"
#: A Zapier-shaped key in a URL path (url_secret_risk).
KEYED_URL = "https://actions.zapier.com/mcp/sk-ak-9fQ2xLm4Rt7Vb8Np3Kd6Wz1Hc5Jy0Ts/sse"


@pytest.fixture
def refuse_real_home(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse() -> Path:
        raise AssertionError("a client source read Path.home(); use ClientContext")

    monkeypatch.setattr(Path, "home", staticmethod(refuse))


def write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def write_json(path: Path, data: Any) -> Path:
    return write_text(path, json.dumps(data, indent=2))


def by_name(reading: SourceReading) -> dict[tuple[str, str], ImportCandidate]:
    """Candidates keyed by ``(scope, original_name)``."""
    return {(c.scope, c.original_name): c for c in reading.candidates}


def assert_trust_unknown(candidate: ImportCandidate, client_label: str) -> None:
    """Listed as the client's, but never picked without the user's word."""
    assert candidate.trust == "unknown", candidate.original_name
    assert candidate.trust_note is not None
    assert candidate.trust_note.startswith(f"trust in {client_label} could not be checked: ")
    assert candidate.inactive is None  # the client may well use it
    assert not candidate.preselectable
    assert candidate.trust_note in candidate.needs_opt_in
    shown = candidate.to_dict()
    assert shown["trust"] == "unknown" and shown["trust_note"] == candidate.trust_note
    assert shown["needs_opt_in"] == list(candidate.needs_opt_in)


def assert_display_safe(reading: SourceReading, *secrets: str) -> None:
    """No planted secret, and no URL, in any rendering of ``reading``."""
    rendered = json.dumps(reading.to_dict()) + repr(reading)
    for secret in (SECRET, QUERY_SECRET, CLIENT_SECRET, CLIENT_ID, *secrets):
        assert secret not in rendered, secret
    assert "https://" not in json.dumps(reading.to_dict())
    assert "9fQ2xLm4" not in rendered
