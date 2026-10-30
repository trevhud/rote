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

import pytest
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
            "edges": [
                {"from": a, "to": b, **({"fan_out": True} if fanned.get("fan_out") else {})}
                for a, b, fanned in zip(ids, ids[1:], nodes[1:], strict=False)
            ],
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


def test_a_key_read_through_a_helper_in_the_module_counts(tmp_path: Path) -> None:
    """The 0.18.0 recompile of #291's skill read all three wrong gate keys
    through one shared helper, ``payload(value, name)``, so the literal-key
    check saw none of them, and every run failed with ``Missing payload``."""
    pipeline = _pipeline([_GATE, _consumer({"answer": "ask_completion.output"})])
    helper = (
        "def payload(value, name):\n"
        "    if name not in value:\n"
        "        raise ValueError(f'Missing payload: {name}')\n"
        "    return value[name]\n\n"
    )
    wrong = helper + (
        "def score(answer):\n    return {'n': payload(answer, 'opendock_appointment_completion')}\n"
    )
    root = _write(tmp_path, pipeline, {"score.py": wrong})
    [finding] = boundary_findings(pipeline, root)
    assert (finding.node_id, finding.kind) == ("score", "gate_key_undeclared")
    assert "reads `opendock_appointment_completion`" in finding.message
    right = (
        helper + "def score(answer):\n    return {'n': payload(answer, 'appointment_completion')}\n"
    )
    assert boundary_findings(pipeline, _write(tmp_path / "ok", pipeline, {"score.py": right})) == []


def test_helper_reads_follow_keywords_nested_helpers_and_get() -> None:
    import ast

    module = ast.parse(
        "def require(mapping, key):\n"
        "    return mapping.get(key)\n\n"
        "def payload(value, name):\n"
        "    return require(value, name)\n\n"
        "def fixed(value):\n"
        "    return value['always']\n\n"
        "def f(request, other):\n"
        "    a = payload(request, 'a')\n"
        "    b = payload(name='b', value=request)\n"
        "    c = fixed(request)\n"
        "    d = payload(other, 'd')\n"
        "    e = payload(request, computed)\n"
        "    g = payload(*args)\n"
    )
    fn = module.body[-1]
    assert isinstance(fn, ast.FunctionDef)
    assert literal_keys_read(fn, "request", module) == {"a", "b", "always"}
    # Without the module, helpers are not followed.
    assert literal_keys_read(fn, "request") == set()


def test_a_recursive_helper_does_not_loop() -> None:
    import ast

    module = ast.parse(
        "def walk(value, name):\n"
        "    return walk(value, name) if name else value[name]\n\n"
        "def f(request):\n"
        "    return walk(request, 'k')\n"
    )
    fn = module.body[-1]
    assert isinstance(fn, ast.FunctionDef)
    assert literal_keys_read(fn, "request", module) == {"k"}


@pytest.mark.parametrize(
    "source",
    [
        # A comprehension variable shadows the key parameter.
        "def opt(value, key):\n    return [value[key] for key in ('a', 'b') if key in value]\n",
        # A lambda parameter shadows it.
        "def opt(value, key):\n    return sorted(value, key=lambda key: value[key])\n",
        # An except capture rebinds it.
        "def opt(value, key):\n    try:\n        return value[key]\n"
        "    except KeyError as key:\n        return key\n",
        # The module rebinds the helper's name after the def.
        "def opt(value, key):\n    return value[key]\nfrom other import opt\n",
        "def opt(value, key):\n    return value[key]\nopt = dict.get\n",
    ],
)
def test_a_helper_read_through_a_rebound_name_is_not_a_finding(source: str) -> None:
    """A key substituted into a name that something rebinds is a key nothing
    reads, and a false error makes the agent rewrite correct code (PR #311
    review)."""
    import ast

    module = ast.parse(source + "\ndef step(answer):\n    return opt(answer, 'zzz')\n")
    fn = module.body[-1]
    assert isinstance(fn, ast.FunctionDef)
    assert literal_keys_read(fn, "answer", module) == set()


@pytest.mark.parametrize(
    "caller",
    [
        "def step(answer):\n    def opt(v, k):\n        return 1\n    return opt(answer, 'zzz')\n",
        "def step(answer, opt):\n    return opt(answer, 'zzz')\n",
        "def step(answer):\n    from other import opt\n    return opt(answer, 'zzz')\n",
    ],
)
def test_a_helper_the_caller_shadows_is_not_followed(caller: str) -> None:
    import ast

    module = ast.parse("def opt(value, key):\n    return value[key]\n\n" + caller)
    fn = module.body[-1]
    assert isinstance(fn, ast.FunctionDef)
    assert literal_keys_read(fn, "answer", module) == set()


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


# ───────── another step's result (#329) ─────────

_TICKET = {
    "type": "object",
    "properties": {"title": {"type": "string"}, "repo": {"type": "string"}},
    "required": ["title", "repo"],
}


def _orphan_nodes(create_inputs: dict[str, str], passthrough: bool = True) -> list[dict[str, Any]]:
    """#329's shape: an identity call, a judge that copies the signed-in
    account id to the top of its output, and a creation fan-out over its items."""
    output = {
        "type": "object",
        "properties": {
            "items": {"type": "array", "items": _TICKET},
            **({"assignee_account_id": {"type": "string"}} if passthrough else {}),
        },
        "required": ["items", *(["assignee_account_id"] if passthrough else [])],
    }
    return [
        {
            "id": "get_identity",
            "kind": "external_call",
            "description": "Read the signed-in account.",
            "impl": "extracted/identity.py:get_identity",
            "output": {"account_id": "str"},
        },
        {
            "id": "prepare",
            "kind": "llm_judge",
            "description": "Prepare the tickets.",
            "inputs": {"me": "get_identity.output"},
            "signature_spec": {
                "prompt": "Prepare tickets for {{ me }}",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "me": {
                            "type": "object",
                            "properties": {"account_id": {"type": "string"}},
                            "required": ["account_id"],
                        }
                    },
                    "required": ["me"],
                },
                "output_schema": output,
                **(
                    {"passthrough": {"assignee_account_id": "me.account_id"}} if passthrough else {}
                ),
            },
        },
        {
            "id": "create",
            "kind": "external_call",
            "description": "Create one ticket per item.",
            "impl": "extracted/create.py:create",
            "fan_out": True,
            "inputs": create_inputs,
        },
    ]


_CREATE_READS_ITEM_ASSIGNEE = (
    "def create(ticket):\n"
    "    return {'key': ticket['title'], 'assignee': ticket['assignee_account_id']}\n"
)


def test_code_that_reads_a_key_a_judges_item_cannot_hold_is_an_error(tmp_path: Path) -> None:
    pipeline = _pipeline(_orphan_nodes({"ticket": "prepare.output.items"}))
    root = _write(tmp_path, pipeline, {"create.py": _CREATE_READS_ITEM_ASSIGNEE})
    by_kind = {f.kind: f for f in boundary_findings(pipeline, root)}
    finding = by_kind["result_key_undelivered"]
    assert (finding.node_id, finding.severity) == ("create", "error")
    assert "reads `assignee_account_id` from `ticket`" in finding.message
    assert "can hold only `repo`, `title`" in finding.message
    # Where the value is: the judge's top-level copy.
    assert "`prepare.output.assignee_account_id` holds it" in finding.message


def test_a_key_the_item_holds_is_not_a_finding(tmp_path: Path) -> None:
    pipeline = _pipeline(
        _orphan_nodes(
            {"ticket": "prepare.output.items", "who": "prepare.output.assignee_account_id"}
        )
    )
    ok = "def create(ticket, who):\n    return {'key': ticket['title'], 'assignee': who}\n"
    root = _write(tmp_path, pipeline, {"create.py": ok})
    assert boundary_findings(pipeline, root) == []


def test_an_open_schema_is_never_checked(tmp_path: Path) -> None:
    """A pure function's result may carry keys its schema does not list."""
    producer = {
        "id": "load",
        "kind": "pure_function",
        "description": "Load it.",
        "impl": "extracted/load.py:load",
        "output_schema": {"type": "object", "properties": {"rows": {"type": "array"}}},
    }
    pipeline = _pipeline([producer, _consumer({"loaded": "load.output"})])
    reads = "def score(loaded):\n    return {'n': loaded['extra']}\n"
    root = _write(tmp_path, pipeline, {"score.py": reads})
    assert boundary_findings(pipeline, root) == []
    closed = {
        **producer,
        "output_schema": {**producer["output_schema"], "additionalProperties": False},
    }
    pipeline = _pipeline([closed, _consumer({"loaded": "load.output"})])
    [finding] = boundary_findings(pipeline, root)
    assert finding.kind == "result_key_undelivered"


def test_a_copied_value_no_step_reads_is_a_review_gap(tmp_path: Path) -> None:
    """#329: the account id reaches the judge's output and never the fan-out."""
    pipeline = _pipeline(_orphan_nodes({"ticket": "prepare.output.items"}))
    ok_item = "def create(ticket):\n    return {'key': ticket['title']}\n"
    root = _write(tmp_path, pipeline, {"create.py": ok_item})
    [finding] = boundary_findings(pipeline, root)
    assert (finding.node_id, finding.kind, finding.severity) == (
        "prepare",
        "passthrough_unread",
        "warning",
    )
    assert "no later step reads `prepare.output.assignee_account_id`" in finding.message
    assert "`create` read(s) prepare's other fields" in finding.message
    assert "or `get_identity.output.account_id` where it came from" in finding.message
    assert "Never let the step supply a default" in finding.message


@pytest.mark.parametrize(
    "create_inputs",
    [
        {"ticket": "prepare.output.items", "who": "prepare.output.assignee_account_id"},
        # A whole-output reader whose code reads the field.
        {"ticket": "prepare.output.items", "prepared": "prepare.output"},
    ],
)
def test_a_copied_value_a_step_reads_is_fine(tmp_path: Path, create_inputs: dict[str, str]) -> None:
    pipeline = _pipeline(_orphan_nodes(create_inputs))
    params = ", ".join(create_inputs)
    body = (
        "    return {'who': prepared['assignee_account_id']}\n"
        if "prepared" in create_inputs
        else "    return {'who': who}\n"
    )
    root = _write(tmp_path, pipeline, {"create.py": f"def create({params}):\n{body}"})
    assert boundary_findings(pipeline, root) == []


def test_a_whole_output_reader_with_no_readable_code_counts_as_reading_it(
    tmp_path: Path,
) -> None:
    """A stub or MCP call takes the whole value: rote cannot say it drops a field."""
    pipeline = _pipeline(_orphan_nodes({"ticket": "prepare.output.items", "p": "prepare.output"}))
    root = _write(tmp_path, pipeline, {})
    assert boundary_findings(pipeline, root) == []
