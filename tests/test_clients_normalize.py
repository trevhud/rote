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
    url_secret_risk,
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


ZAPIER_SHAPED = "https://actions.zapier.com/mcp/sk-ak-9fQ2xLm4Rt7Vb8Np3Kd6Wz1Hc5Jy0Ts/sse"

# Fake key shapes, assembled so the repo's secret scanner (which matches the
# real prefixes) never sees one whole. Every body is made up.
_BODY = "9fQ2xLm4Rt7Vb8Np3Kd6Wz1Hc5Jy0TsAa"
OPENAI_SHAPED = "https://x.example.com/mcp/" + "sk-" + "proj-" + _BODY + "/sse"
GITHUB_SHAPED = "https://x.example.com/mcp/" + "gh" + "p_" + _BODY[:36].ljust(36, "Q") + "/mcp"
SLACK_SHAPED = (
    "https://x.example.com/mcp/"
    + ("xo" + "xb-")
    + "1234567890123-1234567890123-"
    + _BODY[:24]
    + "/mcp"
)
HEX40_SHAPED = "https://x.example.com/k/" + "3fa9c0d1e2b4" * 3 + "a1b2c3d4" + "/mcp"
BASE64URL22_SHAPED = "https://x.example.com/s/k3Jd9sLq2xPw8Zr4TnVb_Q/mcp"


@pytest.mark.parametrize(
    ("url", "risky"),
    [
        (ZAPIER_SHAPED, True),
        (OPENAI_SHAPED, True),
        (GITHUB_SHAPED, True),
        (SLACK_SHAPED, True),
        (HEX40_SHAPED, True),
        (BASE64URL22_SHAPED, True),
        ("https://x.example.com/w/550e8400-e29b-41d4-a716-446655440000/mcp", True),
        ("https://x.example.com/t/" + "ab12" * 8, True),  # a 32-char hex run
        ("https://mcp.figma.com/mcp", False),
        ("https://x.example.com/api/mcp/streamable-http-transport", False),
        ("https://x.example.com/servers/github-enterprise-cloud/mcp", False),
        ("https://x.example.com/v1/2025-03-26/mcp", False),
        ("https://x.example.com/a/abc123def456ghi789/mcp", False),  # 18 chars: an id
        # Long and mixing letters and digits, but far from random (2.27 and
        # 2.54 bits per character): the entropy floor keeps these importable.
        ("https://x.example.com/p/project-00000000000001/mcp", False),
        ("https://x.example.com/v2-2025-03-26-2025-03-26/mcp", False),
        # review6's false positives: long, varied slugs whose letters and
        # digits mix only across words. No single part is a key's body.
        ("https://x.example.com/servers/my-awesome-server-2024/", False),
        ("https://x.example.com/salesforce-opportunity-management-v2/", False),
        ("https://x.example.com/mcp/release.2026.10.build-123456/", False),
    ],
)
def test_url_secret_risk_reads_the_form_of_each_path_segment(url: str, risky: bool) -> None:
    risk = url_secret_risk(url)
    assert (risk is not None) is risky
    if risk is not None:
        assert "may be an API key" in risk
        assert "9fQ2" not in risk and "550e" not in risk  # never quotes the segment


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
