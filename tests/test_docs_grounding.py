"""A docs spec may not say more than the pages it cites (rote.vendor_specs.grounding).

The fixture page is the real Slack conversations.history reference (as text),
and the spec is the one a real deal-monitor --backend api compile wrote.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rote.vendor_specs.grounding import (
    declared_names,
    docs_grounding_findings,
    occurs,
    page_text,
    to_text,
)

FIXTURES = Path(__file__).parent / "fixtures" / "vendor_specs"
SLACK_PAGE = "https://docs.slack.dev/reference/methods/conversations.history"
SLACK_TEXT = (FIXTURES / "slack_conversations_history.txt").read_text(encoding="utf-8")
DEAL_MONITOR_SPEC = json.loads((FIXTURES / "deal_monitor_slack.web.v1.json").read_text())


def _deliverable(tmp_path: Path, spec: dict[str, Any]) -> Path:
    root = tmp_path / "d"
    (root / "vendor_specs").mkdir(parents=True)
    (root / "vendor_specs" / "slack.web.v1.json").write_text(json.dumps(spec), encoding="utf-8")
    return root


def _pages(url: str) -> str:
    if url == SLACK_PAGE:
        return SLACK_TEXT
    from rote.vendor_specs.grounding import PageUnavailable

    raise PageUnavailable(url)


def _findings(tmp_path: Path, spec: dict[str, Any], skill_dir: Path | None = None) -> list[Any]:
    return docs_grounding_findings(
        _deliverable(tmp_path, spec), skill_dir, tmp_path / "pages", fetch=_pages
    )


def test_the_real_deal_monitor_spec_is_grounded_in_the_real_page(tmp_path: Path) -> None:
    assert _findings(tmp_path, DEAL_MONITOR_SPEC) == []


@pytest.mark.parametrize(
    ("change", "what", "name"),
    [
        # An invented response field, the circular case: the agent adds what
        # its own code reads.
        (
            lambda op: op["response_example"]["messages"][0].update(deal_value=1),
            "response field",
            "deal_value",
        ),
        (
            lambda op: op["params"].update(workspace_id={"location": "query"}),
            "parameter",
            "workspace_id",
        ),
        (
            lambda op: op.update(url_template="https://slack.com/api/conversations.digest"),
            "path segment",
            "conversations.digest",
        ),
        (
            lambda op: op.update(request_example={"filter_mode": "x"}),
            "request field",
            "filter_mode",
        ),
    ],
)
def test_an_invented_name_is_unsupported(tmp_path: Path, change: Any, what: str, name: str) -> None:
    spec = json.loads(json.dumps(DEAL_MONITOR_SPEC))
    change(spec["operations"][0])
    [finding] = _findings(tmp_path, spec)
    assert (finding.kind, finding.severity) == ("docs_spec_unsupported", "error")
    assert (
        f"declares the {what} {name!r}, which none of its cited sources contains" in finding.message
    )


def test_a_skill_source_is_the_skills_own_text(tmp_path: Path) -> None:
    skill = tmp_path / "skill"
    (skill / "references").mkdir(parents=True)
    (skill / "SKILL.md").write_text("Call GET https://intra.example/v1/orders with `status`.\n")
    (skill / "references" / "api.md").write_text("The response has `orders`, each with `id`.\n")
    spec = {
        "vendor": "intra",
        "api": "orders",
        "version": "v1",
        "sources": ["skill"],
        "operations": [
            {
                "id": "orders.list",
                "method": "GET",
                "url_template": "https://intra.example/v1/orders",
                "params": {"status": {"location": "query"}},
                "response_example": {"orders": [{"id": "o1"}]},
            }
        ],
    }
    assert _findings(tmp_path, spec, skill) == []
    spec["operations"][0]["response_example"]["orders"][0]["total"] = 1
    [finding] = _findings(tmp_path / "b", spec, skill)
    assert "'total'" in finding.message


def test_a_spec_none_of_whose_sources_can_be_read_is_ungrounded_not_failed(
    tmp_path: Path,
) -> None:
    spec = dict(DEAL_MONITOR_SPEC, sources=["https://docs.example/unreachable"])
    [finding] = _findings(tmp_path, spec)
    assert (finding.kind, finding.severity) == ("docs_spec_ungrounded", "warning")


def test_one_readable_source_is_enough_to_ground_against(tmp_path: Path) -> None:
    spec = dict(DEAL_MONITOR_SPEC, sources=[SLACK_PAGE, "https://docs.example/unreachable"])
    assert _findings(tmp_path, spec) == []


def test_html_becomes_its_text() -> None:
    page = (
        "<html><head><style>.has_more{}</style><script>var ok = 1;</script></head>"
        "<body><p>Returns <code>has_more</code> &amp; <b>next_cursor</b></p></body></html>"
    )
    text = to_text(page)
    assert occurs("has_more", text) and occurs("next_cursor", text) and "&" in text
    assert not occurs("ok", text), "script text is not documentation"


@pytest.mark.parametrize(
    ("name", "text", "found"),
    [
        ("has_more", "it returns has_more when", True),
        ("has_more", "has_more_items", False),
        ("ts", "the ts field", True),
        ("ts", "timestamps", False),
        ("Ts", "the ts field", False),
        ("response_metadata.next_cursor", "response_metadata.next_cursor", True),
    ],
)
def test_names_match_whole_and_case_sensitive(name: str, text: str, found: bool) -> None:
    assert occurs(name, text) is found


def test_declared_names_cover_the_path_params_and_nested_fields() -> None:
    from rote.vendor_specs.docs import DocsSpecFile

    op = DocsSpecFile.model_validate(DEAL_MONITOR_SPEC).operations[0]
    names = set(declared_names(op))
    assert ("path segment", "conversations.history") in names
    assert ("parameter", "include_all_metadata") in names
    assert ("response field", "next_cursor") in names
    assert ("response field", "user") in names


def test_a_page_is_fetched_once_a_day(tmp_path: Path) -> None:
    calls: list[str] = []

    def fetch(url: str) -> str:
        calls.append(url)
        return "<p>channel</p>"

    assert page_text(SLACK_PAGE, tmp_path, fetch) == " channel "
    assert page_text(SLACK_PAGE, tmp_path, fetch) == " channel "
    assert calls == [SLACK_PAGE]


def test_verification_bounces_an_unsupported_name_into_the_repair_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rote.compiler.verify import EmitTarget, verify_deliverable
    from rote.vendor_specs import SpecStore

    monkeypatch.setattr("rote.vendor_specs.grounding.http_get_text", _pages)
    spec = json.loads(json.dumps(DEAL_MONITOR_SPEC))
    spec["operations"][0]["response_example"]["messages"][0]["deal_value"] = 1
    root = _deliverable(tmp_path, spec)
    (root / "pipeline.yaml").write_text(
        "name: d\nversion: '0.1.0'\ninput: {type: In}\nnodes:\n"
        "  - {id: n, kind: pure_function, description: d, impl: extracted/n.py:n}\nedges: []\n"
    )
    (root / "extracted").mkdir()
    (root / "extracted" / "n.py").write_text("def n():\n    return {}\n")
    verification = verify_deliverable(
        root,
        run_tests=False,
        emit_target=EmitTarget("dbos", "api"),
        spec_store=SpecStore(root=tmp_path / "specs"),
    )
    kinds = {f.kind for f in verification.errors}
    assert "docs_spec_unsupported" in kinds
    # The driver here reported no fetches: said, not skipped in silence.
    assert any(f.kind == "docs_spec_fetches_unreported" for f in verification.findings)
