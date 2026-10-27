"""A step reads only keys its producer declares, and an MCP call sends its constants (#291).

A real compile passed 38 generated tests while every MCP call went out with
no arguments, two gate consumers read keys their gates never declared, and a
renderer required an input field the contract did not declare. The tests
built fixtures from the module's own keys, so nothing crossed the real
boundary. These checks read the deliverable itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from rote.compiler.boundaries import boundary_findings, literal_keys_read
from rote.compiler.grounding import Grounding
from rote.compiler.tool_grounding import ToolSchema, unsent_constant_findings
from rote.compiler.verify import verify_deliverable
from rote.contracts import implementation_status
from rote.ir import Pipeline


def _pipeline(nodes: list[dict[str, Any]], input_: dict[str, Any] | None = None) -> Pipeline:
    ids = [n["id"] for n in nodes]
    return Pipeline.model_validate(
        {
            "name": "boundaries",
            "version": "0.1.0",
            "input": input_ or {"type": "In", "required": ["region"], "optional": []},
            "nodes": nodes,
            "edges": [{"from": a, "to": b} for a, b in zip(ids, ids[1:], strict=False)],
            "entry_nodes": ids[:1],
            "exit_nodes": ids[-1:],
        }
    )


def _write(root: Path, pipeline: Pipeline, modules: dict[str, str]) -> Path:
    (root / "extracted").mkdir(parents=True, exist_ok=True)
    (root / "pipeline.yaml").write_text(
        yaml.safe_dump(pipeline.model_dump(mode="json", by_alias=True, exclude_none=True))
    )
    for name, text in modules.items():
        (root / "extracted" / name).write_text(text)
    return root


_GATE = {
    "id": "ask_completion",
    "kind": "hitl_gate",
    "description": "Ask for the appointment completion figures.",
    "signal": "completion",
    "output": {"appointment_completion": "dict"},
}


def _consumer(inputs: dict[str, str]) -> dict[str, Any]:
    return {
        "id": "score",
        "kind": "pure_function",
        "description": "Score it.",
        "impl": "extracted/score.py:score",
        "inputs": inputs,
    }


# ───────── gate keys ─────────


def test_code_that_reads_a_key_the_gate_never_declares_is_an_error(tmp_path: Path) -> None:
    pipeline = _pipeline([_GATE, _consumer({"answer": "ask_completion.output"})])
    wrong = "def score(answer):\n    return {'n': answer['opendock_appointment_completion']}\n"
    root = _write(tmp_path, pipeline, {"score.py": wrong})
    [finding] = boundary_findings(pipeline, root)
    assert (finding.node_id, finding.kind, finding.severity) == (
        "score",
        "gate_key_undeclared",
        "error",
    )
    assert "reads `opendock_appointment_completion` from gate ask_completion's" in finding.message
    assert "declares only `appointment_completion`" in finding.message
    right = "def score(answer):\n    return {'n': answer.get('appointment_completion')}\n"
    assert boundary_findings(pipeline, _write(tmp_path / "ok", pipeline, {"score.py": right})) == []


def test_an_inputs_reference_to_an_undeclared_gate_field_is_refused_at_load() -> None:
    """The code check covers what the IR cannot see; this half the IR
    already refuses, so the two do not overlap."""
    import pytest

    with pytest.raises(ValueError, match="declares only output keys"):
        _pipeline([_GATE, _consumer({"metrics": "ask_completion.output.datadog_metrics"})])


# ───────── pipeline-input fields ─────────


def test_code_that_reads_an_undeclared_input_field_is_an_error(tmp_path: Path) -> None:
    render = {
        "id": "render",
        "kind": "pure_function",
        "description": "Render the report.",
        "impl": "extracted/render.py:render",
        "inputs": {"request": "pipeline.input"},
    }
    pipeline = _pipeline([render])
    code = (
        "def render(request):\n"
        "    url = request.get('dashboard_url')\n"
        "    if not url:\n"
        "        raise ValueError('need a dashboard url')\n"
        "    return {'region': request['region'], 'url': url}\n"
    )
    root = _write(tmp_path, pipeline, {"render.py": code})
    [finding] = boundary_findings(pipeline, root)
    assert (finding.kind, finding.node_id) == ("input_field_undeclared", "render")
    assert "reads `dashboard_url` from the pipeline input" in finding.message
    assert "declares `region`" in finding.message  # `region` is declared, so not flagged
    declared = _pipeline(
        [render], {"type": "In", "required": ["region"], "optional": ["dashboard_url"]}
    )
    clean = _write(tmp_path / "ok", declared, {"render.py": code})
    assert boundary_findings(declared, clean) == []


def test_only_literal_reads_of_the_parameter_itself_count() -> None:
    import ast

    tree = ast.parse(
        "def f(request, other):\n"
        "    a = request['a']\n"
        "    b = request.get('b', 1)\n"
        "    c = request.pop('c')\n"
        "    d = other['d']\n"
        "    e = request[name]\n"
        "    request['set'] = 1\n"
    )
    fn = tree.body[0]
    assert isinstance(fn, ast.FunctionDef)
    assert literal_keys_read(fn, "request") == {"a", "b", "c"}
    # A parameter rebound in the body holds something else: nothing is read off it.
    source = "def f(request):\n    request, n = {}, 1\n    return request['x']\n"
    rebound = ast.parse(source).body[0]
    assert isinstance(rebound, ast.FunctionDef)
    assert literal_keys_read(rebound, "request") == set()


# ───────── constants an MCP call never sends ─────────

_FETCH = {
    "id": "fetch_sheet",
    "kind": "external_call",
    "description": "Read the containers sheet.",
    "impl": "extracted/fetch.py:fetch_sheet",
    "mcp": {"server": "drive", "tool": "read_file", "args": {}},
    "constants": {"file_id": "FILE-123", "tab": "Containers"},
}
_SCHEMA = {
    ("drive", "read_file"): ToolSchema(
        "drive",
        "read_file",
        {"type": "object", "properties": {"fileId": {"type": "string"}, "sheet": {}}},
        "tools/list",
        True,
    )
}


def test_constants_a_call_never_sends_are_an_error_when_the_schema_says_where() -> None:
    [finding] = unsent_constant_findings(_pipeline([_FETCH]), _SCHEMA)
    assert (finding.kind, finding.severity) == ("mcp_constant_unsent", "error")
    assert "`file_id`, `tab`" in finding.message
    assert "drive.read_file takes `fileId`, `sheet`" in finding.message


def test_without_a_schema_the_call_is_unresolved_never_guessed() -> None:
    [finding] = unsent_constant_findings(_pipeline([_FETCH]), {})
    assert (finding.kind, finding.severity) == ("mcp_args_ungrounded", "warning")
    assert "rote mcp add drive <url>" in finding.message


def test_a_bound_constant_and_a_zero_argument_call_are_left_alone() -> None:
    bound = {**_FETCH, "mcp": {**_FETCH["mcp"], "args": {"fileId": "file_id", "sheet": "tab"}}}
    assert unsent_constant_findings(_pipeline([bound]), _SCHEMA) == []
    zero = {k: v for k, v in _FETCH.items() if k != "constants"}
    assert unsent_constant_findings(_pipeline([zero]), {}) == []


def test_a_call_that_never_sends_its_constants_blocks_the_run(tmp_path: Path) -> None:
    """The summary used to say such nodes "run via their mcp binding"."""
    pipeline = _pipeline([_FETCH])
    stub = "def fetch_sheet(**_):\n    raise NotImplementedError('vendor call')\n"
    root = _write(tmp_path, pipeline, {"fetch.py": stub})
    [impl] = implementation_status(pipeline, root)
    assert impl.mcp_backed and impl.blocks_run
    assert impl.unsent_constants == ("file_id", "tab")
    assert impl.reason == (
        "its file_id, tab never reach drive.read_file: mcp.args names none of them"
    )
    bound = {**_FETCH, "mcp": {**_FETCH["mcp"], "args": {"fileId": "file_id", "sheet": "tab"}}}
    [ok] = implementation_status(_pipeline([bound]), root)
    assert ok.mcp_backed and not ok.blocks_run


def test_verification_sends_the_boundary_errors_back_to_repair(tmp_path: Path) -> None:
    pipeline = _pipeline([_FETCH, _GATE, _consumer({"answer": "ask_completion.output"})])
    root = _write(
        tmp_path,
        pipeline,
        {
            "fetch.py": "def fetch_sheet(**_):\n    raise NotImplementedError('vendor call')\n",
            "score.py": "def score(answer):\n    return {'n': answer['completion']}\n",
        },
    )
    v = verify_deliverable(root, grounding=Grounding(tools=_SCHEMA), run_tests=False)
    kinds = {f.kind for f in v.errors}
    assert {"mcp_constant_unsent", "gate_key_undeclared"} <= kinds and not v.ok
    instruction = v.repair_instruction(root) or ""
    assert "drive.read_file takes `fileId`, `sheet`" in instruction
    assert "reads `completion` from gate ask_completion's" in instruction
