"""The bundled first-run demo: docs-digest and its recorded reference.

No network and no inference: everything here reads the committed files.
The strongest check is the offline replay, which feeds the tool results
the recorded pipeline run received through the compiled extracted
functions and requires the recorded digest back, so the reference and the
committed code cannot disagree.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

from rote import demo
from rote.adapters import get_adapter
from rote.ir import Pipeline

REPO = Path(__file__).resolve().parent.parent
DEMO = REPO / "examples" / "docs-digest"


def _pipeline() -> Pipeline:
    return Pipeline.model_validate(yaml.safe_load(demo.pipeline_path().read_text(encoding="utf-8")))


def _extracted(name: str, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Import a compiled module by path without writing a .pyc into examples/."""
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    path = DEMO / "compiled" / "extracted" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_docs_digest_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_demo_resolves_to_the_repo_copy_in_a_checkout() -> None:
    assert demo.docs_digest_dir() == DEMO
    assert demo.skill_dir() == DEMO / "skill"
    assert (demo.skill_dir() / "SKILL.md").is_file()
    assert demo.pipeline_path() == DEMO / "compiled" / "pipeline.yaml"


def test_the_reference_matches_the_skill_and_the_pipeline() -> None:
    ref = demo.load_reference()
    assert ref.raw["format"] == demo.REFERENCE_FORMAT == "rote-demo-reference/1"
    assert ref.raw["skill"] == demo.DEMO_NAME
    # Both sides ran on the input compile grounds itself in.
    skill_input = json.loads((DEMO / "skill" / "input.json").read_text(encoding="utf-8"))
    assert ref.input == skill_input
    # The servers to register are exactly the pipeline's MCP bindings.
    bindings = {n.mcp.server: n.mcp.tool for n in _pipeline().nodes if n.mcp is not None}
    assert {name: s["tool"] for name, s in ref.servers.items()} == bindings
    assert ref.servers["mslearn"]["url"] == "https://learn.microsoft.com/api/mcp"
    assert ref.servers["deepwiki"]["url"] == "https://mcp.deepwiki.com/mcp"
    # The pipeline returns its exit node's result under the node id.
    assert list(ref.pipeline_output) == _pipeline().exit_nodes
    for key in ("rote_version", "claude_cli_version", "recorded_on"):
        assert ref.raw[key], key
    assert ref.raw["pipeline_run"]["model_calls"] == 0


def test_every_field_resolves_and_drifting_fields_say_why() -> None:
    ref = demo.load_reference()
    names = [f.name for f in ref.fields]
    assert names == ["reading_list", "repo_outline"]
    for field in ref.fields:
        pipeline_value = demo.value_at(ref.pipeline_output, field.pipeline, field.section)
        skill_value = demo.value_at(ref.skill_output, field.skill, field.section)
        assert pipeline_value and skill_value
        if field.expect == "drifts":
            assert len(field.why) > 20, field.name
    # The recorded runs agreed: the skill's digest and the pipeline's are
    # the same text but for the pipeline's trailing newline.
    skill_digest = demo.value_at(ref.skill_output, "/digest")
    pipeline_digest = demo.value_at(ref.pipeline_output, "/write_digest/digest")
    assert pipeline_digest == skill_digest + "\n"


def test_the_recorded_tool_results_replay_to_the_recorded_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ref = demo.load_reference()
    tools: dict[str, Any] = ref.raw["pipeline_run"]["tool_results"]
    pipeline = _pipeline()
    nodes = {n.id: n for n in pipeline.nodes}
    assert set(tools) == {n.id for n in pipeline.nodes if n.mcp is not None}

    results: dict[str, Any] = dict(tools)
    for node_id in ("pick_reading_list", "write_digest"):
        node = nodes[node_id]
        payload: dict[str, Any] = {}
        for param, source in node.inputs.items():
            assert isinstance(source, str)
            if source == "pipeline.input":
                payload[param] = ref.input
            else:
                upstream = source.removesuffix(".output")
                payload[param] = results[upstream]
        fn = getattr(_extracted(node_id, monkeypatch), node_id)
        results[node_id] = fn(**{**payload, **(node.constants or {})})
    assert results["write_digest"] == ref.pipeline_output["write_digest"]


def test_dbos_runs_the_tool_steps_over_mcp_and_python_would_hit_stubs(tmp_path: Path) -> None:
    """dbos is the recorded runtime: Python, SQLite, no Node or Docker. The
    python runtime has no MCP backend, so it calls the two tool steps'
    direct-API modules, which are stubs (rote does not write vendor calls
    from memory), and `rote run` refuses it up front (README, "Run it")."""
    from rote.adapters._common import unimplemented_stubs

    pipeline = _pipeline()
    get_adapter("dbos", extracted_source_dir=DEMO / "compiled").emit(pipeline, tmp_path / "dbos")
    assert unimplemented_stubs(tmp_path / "dbos") == []
    main_py = (tmp_path / "dbos" / "main.py").read_text(encoding="utf-8")
    assert "Source skill: ../skill" in main_py

    get_adapter("python", extracted_source_dir=DEMO / "compiled").emit(
        pipeline, tmp_path / "python"
    )
    stubs = {p.stem for p in unimplemented_stubs(tmp_path / "python")}
    assert stubs == {"search_microsoft_learn", "read_repo_wiki_outline"}


def test_nothing_in_the_demo_names_a_machine() -> None:
    """A recorded run must not carry the recorder's paths or account."""
    for path in sorted(DEMO.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        for marker in ("/Users/", "/home/", "/private/", "/tmp/claude", "C:\\", "@gmail.com"):
            assert marker not in text, f"{path.relative_to(REPO)} contains {marker!r}"


# ───────── the reference reader ─────────


def test_value_at_reads_pointers_and_markdown_sections() -> None:
    doc = {"a": {"b/c": ["x", "## One\n\nfirst\n\n## Two\nsecond\n"]}}
    assert demo.value_at(doc, "/a/b~1c/0") == "x"
    assert demo.value_at(doc, "/a/b~1c/1", "One") == "first"
    assert demo.value_at(doc, "/a/b~1c/1", "Two") == "second"
    assert demo.value_at(doc, "") == doc
    with pytest.raises(demo.DemoError, match="missing 'z'"):
        demo.value_at(doc, "/a/z")
    with pytest.raises(demo.DemoError, match="not a JSON pointer"):
        demo.value_at(doc, "a")
    with pytest.raises(demo.DemoError, match="no '## Three' heading"):
        demo.value_at(doc, "/a/b~1c/1", "Three")
    with pytest.raises(demo.DemoError, match="not markdown text"):
        demo.value_at(doc, "/a", "One")


def _copy_reference(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    path = tmp_path / "reference.json"
    shutil.copy(demo.reference_path(), path)
    return path, json.loads(path.read_text(encoding="utf-8"))


def test_a_reference_in_another_format_is_refused(tmp_path: Path) -> None:
    path, raw = _copy_reference(tmp_path)
    raw["format"] = "rote-demo-reference/0"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(demo.DemoError, match="expected format 'rote-demo-reference/1'"):
        demo.load_reference(path)


def test_a_field_that_does_not_resolve_is_refused(tmp_path: Path) -> None:
    path, raw = _copy_reference(tmp_path)
    raw["fields"][0]["section"] = "Not a section"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(demo.DemoError, match="Not a section"):
        demo.load_reference(path)


def test_an_unknown_expectation_is_refused(tmp_path: Path) -> None:
    path, raw = _copy_reference(tmp_path)
    raw["fields"][0]["expect"] = "close enough"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(demo.DemoError, match="expect must be equal or drifts"):
        demo.load_reference(path)
