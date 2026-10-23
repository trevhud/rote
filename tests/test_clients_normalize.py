"""The normalization rules every client source shares (rote.clients.normalize)."""

from __future__ import annotations

import pytest

from rote.clients.normalize import (
    CLAUDE_CODE_SYNTAX,
    header_template,
    is_work_host,
    rewrite_placeholders,
    server_identifier,
    url_problem,
)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("figma", "figma"),
        ("acme-crm-prod", "acme_crm_prod"),
        ("plugin:linear:linear", "plugin_linear_linear"),
        ("9lives", "_9lives"),
        ("a.b c", "a_b_c"),
    ],
)
def test_server_identifier(name: str, expected: str) -> None:
    assert server_identifier(name) == expected


@pytest.mark.parametrize("name", ["", "---", "  "])
def test_server_identifier_refuses_names_with_nothing_to_keep(name: str) -> None:
    with pytest.raises(ValueError):
        server_identifier(name)


def test_rewrite_keeps_references_and_drops_defaults() -> None:
    r = rewrite_placeholders("Bearer ${A} ${B:-x} ${A}", CLAUDE_CODE_SYNTAX)
    assert r.template == "Bearer ${A} ${B} ${A}"
    assert r.env_vars == ("A", "B")
    assert r.dropped_defaults == ("B",)
    assert r.unsupported == ()


def test_rewrite_reports_client_internal_variables() -> None:
    r = rewrite_placeholders("${CLAUDE_PLUGIN_ROOT}/x", CLAUDE_CODE_SYNTAX)
    assert r.unsupported == ("${CLAUDE_PLUGIN_ROOT}",)
    assert r.env_vars == ()


@pytest.mark.parametrize(
    ("value", "copied"),
    [
        ("Bearer ${T}", True),
        ("bearer ${T}", True),
        ("Basic ${T}", True),
        ("${T}", True),
        ("Token ${T}", True),
        ("sk-123", False),  # a literal
        ("Bearer sk-123", False),  # a literal token after the scheme
        ("Bearer sk-1${T}", False),  # half literal
        ("", False),
        ("${CLAUDE_PLUGIN_ROOT}", False),
    ],
)
def test_header_template_copies_only_references(value: str, copied: bool) -> None:
    verdict = header_template("srv", "Authorization", value, CLAUDE_CODE_SYNTAX)
    assert (verdict.header is not None) is copied
    assert (verdict.withheld is None) is copied
    if verdict.withheld is not None:
        assert "sk-" not in verdict.withheld  # a reason never quotes the value


@pytest.mark.parametrize(
    ("url", "problem"),
    [
        ("https://mcp.figma.com/mcp", None),
        ("http://localhost:8080/mcp", None),
        ("https://u:p@x.com/mcp", "credentials"),
        ("https://x.com/mcp?k=v", "query string"),
        ("ftp://x.com", "scheme is ftp"),
        ("https:///nohost", "no host"),
        ("https://x.com:notaport/", "does not parse"),
        ("https://${H}/mcp", "depends on H"),
    ],
)
def test_url_problem(url: str, problem: str | None) -> None:
    found = url_problem(url, CLAUDE_CODE_SYNTAX)
    if problem is None:
        assert found is None
    else:
        assert found is not None and problem in found
        assert url not in found


@pytest.mark.parametrize(
    ("host", "work"),
    [
        ("oms.acme.com", True),
        ("acme.com", True),
        ("acme.com:443", True),
        ("notacme.com", False),
        ("acme.com.evil.io", False),
        (None, False),
    ],
)
def test_is_work_host_matches_the_domain_and_its_subdomains(host: str | None, work: bool) -> None:
    assert is_work_host(host, ("acme.com",)) is work
    assert is_work_host(host, ()) is False
