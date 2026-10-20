"""MCP arguments come from the step, and an unnamed write is the pipeline's output.

Two post-merge --system1 defects:
- ops-report's Gmail step had no `mcp.args`, so the emitted step sent the
  whole pipeline input as the tool's arguments, while the skill states the
  query ("Search query: label:dock-activity newer_than:1d").
- expense-audit's "save it to the finance shared folder" became an
  external_call stub raising "target system not specified" instead of the
  report the pipeline returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

from rote.adapters import get_adapter
from rote.compiler import system1
from rote.compiler.system1.parse import stated_parameters
from rote.ir import load_pipeline

REPO = Path(__file__).resolve().parent.parent
EXAMPLES = REPO / "examples"


def _prepare(tmp_path: Path, skill: Path, **kw: Any) -> tuple[system1.Plan, dict[str, Any]]:
    work = tmp_path / "work"
    work.mkdir(parents=True)
    plan = system1.prepare(skill, work, **kw)
    return plan, yaml.safe_load((work / "pipeline.yaml").read_text())


def _tiny(tmp_path: Path, body: str) -> Path:
    skill = tmp_path / "skill"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(f"---\nname: tiny\ndescription: x\n---\n\n# Tiny\n\n{body}\n")
    return skill


def _gmail(doc: dict[str, Any]) -> dict[str, Any]:
    return next(n for n in doc["nodes"] if (n.get("mcp") or {}).get("tool") == "search_threads")


def test_the_stated_query_is_the_tools_argument(tmp_path: Path) -> None:
    _, doc = _prepare(tmp_path, EXAMPLES / "ops-report" / "skill")
    node = _gmail(doc)
    assert node["constants"] == {"search_query": "label:dock-activity newer_than:1d"}
    assert node["mcp"]["args"] == {"search_query": "search_query"}


def test_every_mcp_step_names_its_arguments(tmp_path: Path) -> None:
    """Never the whole input: every bound node carries args, even an empty map,
    and the emitted step never forwards its payload as the arguments."""
    for skill in ("ops-report", "deal-monitor", "bdr-outreach", "expense-audit", "invoice-push"):
        sub = tmp_path / skill
        _, doc = _prepare(sub, EXAMPLES / skill / "skill")
        for n in doc["nodes"]:
            if n.get("mcp"):
                assert isinstance(n["mcp"].get("args"), dict), (skill, n["id"])
        out = sub / "emit"
        get_adapter("dbos", extracted_source_dir=sub / "work").emit(
            load_pipeline(sub / "work" / "pipeline.yaml"), out
        )
        assert "arguments = payload" not in (out / "main.py").read_text(), skill


@dataclass(frozen=True)
class _Schema:
    input_schema: dict[str, Any]


def test_the_tools_schema_names_the_argument(tmp_path: Path) -> None:
    """With grounding's tool schema, "Search query" binds the tool's `query`."""
    schema = _Schema(
        {"type": "object", "properties": {"query": {"type": "string"}, "maxResults": {}}}
    )
    plan, doc = _prepare(
        tmp_path, EXAMPLES / "ops-report" / "skill", tools={("gmail", "search_threads"): schema}
    )
    assert _gmail(doc)["mcp"]["args"] == {"query": "search_query"}
    (d,) = [
        d
        for n in plan.nodes
        for d in n.decisions
        if d.field == "mcp.args" and d.value == {"query": "search_query"}
    ]
    assert d.confidence == 0.9


def test_a_stated_value_with_no_parameter_is_kept_out_of_the_arguments(tmp_path: Path) -> None:
    schema = _Schema({"type": "object", "properties": {"q": {"type": "string"}}})
    plan, doc = _prepare(
        tmp_path, EXAMPLES / "ops-report" / "skill", tools={("gmail", "search_threads"): schema}
    )
    assert _gmail(doc)["mcp"]["args"] == {}
    assert any("no parameter for ['search_query']" in r for r in plan.review)


@pytest.mark.parametrize(
    ("text", "params"),
    [
        (
            "Search query: label:dock-activity newer_than:1d",
            {"search_query": "label:dock-activity newer_than:1d"},
        ),
        ("**Archive folder ID:** `1ExampleFolder`", {"archive_folder_id": "1ExampleFolder"}),
        ("Tab: Dwell Log - Form Responses", {"tab": "Dwell Log - Form Responses"}),
        ("Include: all opps from the UK", {}),
        ("Note: run it weekly", {}),
    ],
)
def test_stated_parameters(text: str, params: dict[str, str]) -> None:
    assert {c.name: c.value for c in stated_parameters(text, 1)} == params


def test_a_write_with_no_named_system_is_the_output(tmp_path: Path) -> None:
    plan, doc = _prepare(tmp_path, EXAMPLES / "expense-audit" / "skill")
    step7 = [n for n in doc["nodes"] if n["source"]["section"].startswith("Step 7")]
    assert [n["kind"] for n in step7] == ["pure_function"]
    assert any("writes to no named system" in r for r in plan.review)


@pytest.mark.parametrize(
    ("text", "kinds"),
    [
        ("Total the batch and save it to HubSpot.", ["pure_function", "external_call"]),
        ("Total the batch and save it to the shared folder.", ["pure_function"]),
        ("Save the HTML file to `./outputs/report.html`.", ["pure_function"]),
        ("Total the batch and post it to #finance-reports.", ["pure_function", "external_call"]),
    ],
)
def test_only_a_named_destination_is_an_external_write(
    tmp_path: Path, text: str, kinds: list[str]
) -> None:
    _, doc = _prepare(tmp_path, _tiny(tmp_path, f"## Step 1: Do it\n\n{text}\n"))
    assert [n["kind"] for n in doc["nodes"]] == kinds
