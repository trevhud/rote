"""Judge passthrough: input data a judge returns unchanged is copied in code.

``signature_spec.passthrough`` maps output fields to input paths. The
model's request schema leaves those fields out, and after the answer
arrives code copies them from the validated input, before the whole output
schema is checked. So the judge's declared output is unchanged, the copy
cannot drift, and the model writes fewer tokens.

One implementation per language: ``passthrough_answer_schema`` /
``with_passthrough`` in the Python helper (every Python runtime, and the
compiled skill's driver), ``withPassthrough`` in ``_roteSchema.ts`` (every
TS runtime), held equal by the parity test below. No inference: every
model answer comes from a local Messages stub or from the test.
"""

from __future__ import annotations

import copy
import json
import os
import re
import shutil
import subprocess
import threading
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pydantic
import pytest
import yaml

from rote.adapters import get_adapter
from rote.adapters._ts_common import SCHEMA_ERRORS_TS
from rote.eval.empirical import run_pipeline_trial
from rote.inference import _runtime_helper as helper
from rote.inference import structured_output_schema
from rote.ir import LLMSignature, Pipeline, load_pipeline, passthrough_candidates
from rote.plan import build_plan
from rote.skill_target import write_scripts
from tests import _node_trees as node_trees
from tests._messages_api import CURRENT_SONNET, reject_like_the_api, send_json, structured_message
from tests._skill_driver import drive, header, journal, make_skill, result_of

REPO = Path(__file__).resolve().parent.parent
DEAL_MONITOR = REPO / "examples" / "deal-monitor" / "expected" / "pipeline.yaml"

TICKET = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "title": {"type": "string"},
        "body": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "reporter": {"anyOf": [{"type": "string"}, {"type": "null"}]},
    },
    "required": ["id", "title", "body", "tags"],
    "additionalProperties": False,
}

INPUT_SCHEMA = {
    "type": "object",
    "properties": {"ticket": {"$ref": "#/$defs/Ticket"}},
    "required": ["ticket"],
    "additionalProperties": False,
    "$defs": {"Ticket": TICKET},
}

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "ticket": {"$ref": "#/$defs/Ticket"},
        "ticket_id": {"type": "string"},
        "reporter": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "urgency": {"enum": ["low", "high"]},
        "reason": {"type": "string"},
    },
    "required": ["ticket", "ticket_id", "urgency", "reason"],
    "additionalProperties": False,
    "$defs": {"Ticket": TICKET},
}

PASSTHROUGH = {"ticket": "ticket", "ticket_id": "ticket.id", "reporter": "ticket.reporter"}


def _pipeline(
    client: str = "anthropic", passthrough: dict[str, str] | None = None, *, fan: bool = True
) -> Pipeline:
    source = "pipeline.input.tickets" if fan else "pipeline.input.ticket"
    return Pipeline.model_validate(
        {
            "name": "triage",
            "version": "0.1.0",
            "source_skill": "tests/fixtures/triage",
            "description": "Triage each ticket.",
            "input": {
                "type": "Tickets",
                "required": ["tickets" if fan else "ticket"],
                "optional": [],
            },
            "nodes": [
                {
                    "id": "triage",
                    "kind": "llm_judge",
                    "description": "How urgent is each ticket.",
                    "fan_out": fan,
                    "inputs": {"ticket": source},
                    "signature_spec": {
                        "input_schema": INPUT_SCHEMA,
                        "output_schema": OUTPUT_SCHEMA,
                        "passthrough": PASSTHROUGH if passthrough is None else passthrough,
                        "prompt": "Ticket {{ ticket.title }}: {{ ticket.body }}\nHow urgent?",
                        "client": client,
                    },
                }
            ],
            "edges": [],
            "entry_nodes": ["triage"],
            "exit_nodes": ["triage"],
        }
    )


TICKETS = [
    {"id": "T-1", "title": "Site down", "body": "Checkout is down.", "tags": ["prod"]},
    {
        "id": "T-2",
        "title": "Typo",
        "body": "Footer typo.",
        "tags": [],
        "reporter": "Dana",
    },
]


def _decide(prompt: str) -> dict[str, Any]:
    """What a model would answer: only the fields it is asked for."""
    high = "down" in prompt
    return {"urgency": "high" if high else "low", "reason": "outage" if high else "cosmetic"}


def _expected(ticket: dict[str, Any]) -> dict[str, Any]:
    """The judge's whole output for ``ticket``, the same on every language:
    the copies plus the decision. A field the ticket left out (T-1's
    ``reporter``) is absent from the copy, at the top and nested."""
    out: dict[str, Any] = {
        "ticket": dict(ticket),
        "ticket_id": ticket["id"],
        **_decide(ticket["body"]),
    }
    if "reporter" in ticket:
        out["reporter"] = ticket["reporter"]
    return out


def _dumped(output: dict[str, Any]) -> Any:
    """``output`` after a Python runtime's ``Output.model_dump()``, which writes
    every omitted optional field as None, a model-written one included. The
    driver projects through the same rule (``project``). That dump is not
    passthrough's: the copy it dumps is :func:`_expected`'s."""
    return helper.project(OUTPUT_SCHEMA, output)


# ───────── the IR ─────────


def _spec(**changes: Any) -> dict[str, Any]:
    spec = {
        "input_schema": copy.deepcopy(INPUT_SCHEMA),
        "output_schema": copy.deepcopy(OUTPUT_SCHEMA),
        "passthrough": dict(PASSTHROUGH),
        "prompt": "{{ ticket.body }}",
    }
    spec.update(changes)
    return spec


def test_deal_monitor_passes_its_copied_fields_through() -> None:
    nodes = {n.id: n for n in load_pipeline(DEAL_MONITOR).nodes}
    classify = nodes["classify_thread_step"].signature_spec
    score = nodes["score_new_opportunities"].signature_spec
    assert classify is not None and score is not None
    assert classify.passthrough == {
        "account_name": "thread.account_name",
        "thread_id": "thread.thread_id",
        "thread_url": "thread.thread_url",
    }
    assert score.passthrough == {"opportunity": "opportunity"}
    # #264 patched the prompt to render the whole opportunity so the model
    # could copy it; code copies it now, so the prompt no longer does.
    assert "{{ opportunity }}" not in score.prompt
    assert "unchanged" not in score.prompt
    assert "Copy thread" not in classify.prompt


def test_a_valid_passthrough_loads() -> None:
    assert LLMSignature.model_validate(_spec()).passthrough == PASSTHROUGH


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"passthrough": {"ticket_ref": "ticket"}}, "targets ['ticket_ref'] are not output_schema"),
        ({"passthrough": {"ticket_id": "ticket.number"}}, "declares no property 'number'"),
        ({"passthrough": {"ticket_id": "ticket.id.x"}}, "ticket.id is not an object"),
        ({"passthrough": {"ticket_id": "ticket.title.x"}}, "is not an object"),
        # A string target filled from an array: a different schema.
        ({"passthrough": {"ticket_id": "ticket.tags"}}, "has a different schema"),
        # Nullable source into a non-nullable target: also different.
        ({"passthrough": {"ticket_id": "ticket.reporter"}}, "has a different schema"),
        (
            {
                "passthrough": {
                    "ticket": "ticket",
                    "ticket_id": "ticket.id",
                    "reporter": "ticket.reporter",
                    "urgency": "ticket.title",
                    "reason": "ticket.body",
                }
            },
            "different schema",
        ),
        ({"passthrough": {"ticket_id": "ticket.id; import os"}}, "must be a dotted path"),
        ({"passthrough": {'x"]; evil': "ticket.id"}}, "must be an identifier"),
        ({"passthrough": {"ticket_id": "ticket..id"}}, "must be a dotted path"),
    ],
)
def test_an_unsound_passthrough_is_refused_at_load(change: dict[str, Any], message: str) -> None:
    with pytest.raises(pydantic.ValidationError, match=re.escape(message)):
        LLMSignature.model_validate(_spec(**change))


def _with_meta(input_meta: dict[str, Any], output_meta: dict[str, Any]) -> dict[str, Any]:
    spec = _spec(passthrough={"meta": "ticket.meta"})
    spec["input_schema"]["$defs"]["Ticket"]["properties"]["meta"] = input_meta
    spec["output_schema"]["properties"]["meta"] = output_meta
    return spec


def test_a_property_named_like_an_annotation_is_compared() -> None:
    """``title`` / ``description`` are annotations only as keywords; as
    property names they are data, and their schemas must match."""
    for name in ("title", "description", "examples", "$comment"):
        source = {"type": "object", "properties": {name: {"type": "string"}}}
        target = {"type": "object", "properties": {name: {"type": "integer"}}}
        with pytest.raises(pydantic.ValidationError, match="has a different schema"):
            LLMSignature.model_validate(_with_meta(source, target))
        # Negative half: the same property schema on both sides loads.
        LLMSignature.model_validate(_with_meta(source, copy.deepcopy(source)))
    # A const is a literal: a dict inside it keeps its "title" key.
    with pytest.raises(pydantic.ValidationError, match="has a different schema"):
        LLMSignature.model_validate(_with_meta({"const": {"title": 1}}, {"const": {"title": 2}}))


def test_order_of_required_and_enum_does_not_matter() -> None:
    source = {
        "type": "object",
        "properties": {"a": {"enum": ["x", "y"]}, "b": {"type": "string"}},
        "required": ["a", "b"],
        "description": "annotations are ignored",
    }
    target = {
        "type": "object",
        "properties": {"a": {"enum": ["y", "x"]}, "b": {"type": "string"}},
        "required": ["b", "a"],
    }
    LLMSignature.model_validate(_with_meta(source, target))
    with pytest.raises(pydantic.ValidationError, match="has a different schema"):
        LLMSignature.model_validate(_with_meta(source, {**target, "required": ["b"]}))


def test_a_passthrough_that_leaves_the_model_nothing_to_decide_is_refused() -> None:
    output = {
        "type": "object",
        "properties": {"ticket_id": {"type": "string"}, "title": {"type": "string"}},
        "required": ["ticket_id", "title"],
    }
    with pytest.raises(pydantic.ValidationError, match="the model decides nothing"):
        LLMSignature.model_validate(
            _spec(
                output_schema=output,
                passthrough={"ticket_id": "ticket.id", "title": "ticket.title"},
            )
        )
    # Negative half: leave one field to decide and it loads.
    LLMSignature.model_validate(_spec(output_schema=output, passthrough={"ticket_id": "ticket.id"}))


def test_a_root_ref_output_cannot_pass_through() -> None:
    output = {"$ref": "#/$defs/Out", "$defs": {"Out": copy.deepcopy(OUTPUT_SCHEMA)}}
    with pytest.raises(pydantic.ValidationError, match="no root \\$ref"):
        LLMSignature.model_validate(_spec(output_schema=output))


def test_a_source_must_start_at_a_bound_input() -> None:
    data = _pipeline().model_dump(mode="json", by_alias=True, exclude_none=True)
    spec = data["nodes"][0]["signature_spec"]
    spec["input_schema"]["properties"]["context"] = {"$ref": "#/$defs/Ticket"}
    spec["passthrough"] = {"ticket": "context"}
    with pytest.raises(pydantic.ValidationError, match="reads an input `inputs:` does not bind"):
        Pipeline.model_validate(data)
    # Negative half: the same source, bound, loads.
    spec["passthrough"] = {"ticket": "ticket"}
    Pipeline.model_validate(data)


def test_candidates_are_the_input_paths_of_the_same_schema() -> None:
    found = passthrough_candidates(INPUT_SCHEMA, OUTPUT_SCHEMA)
    assert found["ticket"] == ["ticket"]
    assert found["ticket_id"] == ["ticket.id", "ticket.title", "ticket.body"]
    assert found["reporter"] == ["ticket.reporter"]
    assert "urgency" not in found  # no input enum has its schema


# ───────── the Python helper ─────────


def test_the_answer_schema_drops_the_copied_fields_and_their_defs() -> None:
    answer = helper.passthrough_answer_schema(OUTPUT_SCHEMA, PASSTHROUGH)
    assert answer == {
        "type": "object",
        "properties": {"urgency": {"enum": ["low", "high"]}, "reason": {"type": "string"}},
        "required": ["urgency", "reason"],
        "additionalProperties": False,
    }
    # Only the copied fields leave: a $def another field still uses stays.
    kept = helper.passthrough_answer_schema(OUTPUT_SCHEMA, {"ticket_id": "ticket.id"})
    assert set(kept["properties"]) == {"ticket", "reporter", "urgency", "reason"}
    assert kept["$defs"] == {"Ticket": TICKET}
    assert helper.passthrough_answer_schema(OUTPUT_SCHEMA, None) == OUTPUT_SCHEMA
    assert "ticket" in OUTPUT_SCHEMA["properties"]  # the IR's schema is not mutated


def test_the_copy_is_the_inputs_and_wins_over_the_model() -> None:
    inputs = {"ticket": copy.deepcopy(TICKETS[1])}
    merged = helper.with_passthrough(
        {"urgency": "low", "reason": "r", "ticket_id": "FORGED"}, inputs, PASSTHROUGH
    )
    assert merged == {
        "urgency": "low",
        "reason": "r",
        "ticket_id": "T-2",
        "ticket": TICKETS[1],
        "reporter": "Dana",
    }
    merged["ticket"]["tags"].append("mutated")
    assert inputs["ticket"]["tags"] == []  # a copy, not the input's own list
    # A path that reaches no value leaves the field absent; null is a value.
    assert helper.passthrough_values({"ticket": {"id": "T-1"}}, PASSTHROUGH) == {
        "ticket": {"id": "T-1"},
        "ticket_id": "T-1",
    }
    assert helper.passthrough_values({"ticket": None}, {"x": "ticket.id"}) == {}
    assert helper.passthrough_values({"ticket": {"reporter": None}}, {"r": "ticket.reporter"}) == {
        "r": None
    }
    assert helper.with_passthrough(["not", "an", "object"], inputs, PASSTHROUGH) == [
        "not",
        "an",
        "object",
    ]


def test_a_copied_field_the_input_lacks_drops_what_the_model_wrote() -> None:
    merged = helper.with_passthrough(
        {"urgency": "low", "ticket_id": "FORGED", "reporter": "Mallory"},
        {"ticket": {"title": "t"}},
        PASSTHROUGH,
    )
    assert merged == {"urgency": "low", "ticket": {"title": "t"}}


class _JudgeStub(BaseHTTPRequestHandler):
    """Refuses what the real API refuses, then answers from the prompt."""

    requests: list[dict[str, Any]] = []
    answer: Callable[[str], dict[str, Any]] = staticmethod(_decide)

    def do_POST(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        payload = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
        type(self).requests.append(payload)
        if reject_like_the_api(self, payload):
            return
        prompt = payload["messages"][0]["content"]
        send_json(self, structured_message(payload, type(self).answer(prompt)))

    def log_message(self, *args: object) -> None:
        return


@pytest.fixture()
def judge_stub() -> Iterator[str]:
    _JudgeStub.requests = []
    _JudgeStub.answer = staticmethod(_decide)
    server = HTTPServer(("127.0.0.1", 0), _JudgeStub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


def _sent_schema() -> dict[str, Any]:
    return structured_output_schema(
        helper.passthrough_answer_schema(OUTPUT_SCHEMA, PASSTHROUGH), node_id="triage"
    )


def test_call_judge_asks_only_for_the_decision_and_merges_the_copies(
    judge_stub: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ROTE_INFERENCE", "api")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    _JudgeStub.answer = staticmethod(lambda prompt: {**_decide(prompt), "ticket_id": "FORGED"})
    inputs = {"ticket": {**TICKETS[0], "reporter": None}}
    out = helper.call_judge(
        node_id="triage",
        client="anthropic",
        model=CURRENT_SONNET,
        base_url=judge_stub,
        prompt="Checkout is down.",
        output_schema=OUTPUT_SCHEMA,
        passthrough=PASSTHROUGH,
        inputs=inputs,
    )
    assert out == {
        **_decide("down"),
        "ticket": inputs["ticket"],
        "ticket_id": "T-1",
        "reporter": None,
    }
    (request,) = _JudgeStub.requests
    sent = request["output_config"]["format"]["schema"]
    assert sent == _sent_schema()
    assert set(sent["properties"]) == {"urgency", "reason"}


def test_a_copy_the_input_lacks_fails_by_name(
    judge_stub: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A required target whose source a run's input left out is not invented."""
    monkeypatch.setenv("ROTE_INFERENCE", "api")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    with pytest.raises(RuntimeError, match="output: missing key 'ticket_id'"):
        helper.call_judge(
            node_id="triage",
            client="anthropic",
            model=CURRENT_SONNET,
            base_url=judge_stub,
            prompt="Checkout is down.",
            output_schema=OUTPUT_SCHEMA,
            passthrough={"ticket_id": "ticket.id"},
            inputs={"ticket": {}},
        )
    with pytest.raises(RuntimeError, match="was given no inputs"):
        helper.call_judge(
            node_id="triage",
            client="anthropic",
            model=CURRENT_SONNET,
            base_url=judge_stub,
            prompt="p",
            output_schema=OUTPUT_SCHEMA,
            passthrough=PASSTHROUGH,
        )


# ───────── the Python runtimes ─────────


def _stub_env(stub: str) -> dict[str, str]:
    return {
        "ROTE_INFERENCE": "api",
        "ANTHROPIC_API_KEY": "test-key-not-real",
        "ROTE_MODEL_TRIAGE": CURRENT_SONNET,
        "ROTE_BASE_URL_TRIAGE": stub,
        # Anything that ignored the override would reach a closed port.
        "ANTHROPIC_BASE_URL": "http://127.0.0.1:9",
    }


@pytest.mark.parametrize("runtime", ["python", "dbos"])
def test_an_emitted_python_judge_copies_in_code(
    runtime: str, tmp_path: Path, judge_stub: str
) -> None:
    app = tmp_path / runtime
    get_adapter(runtime).emit(_pipeline(), app)
    run = run_pipeline_trial(
        app, {"tickets": TICKETS}, timeout_seconds=120, extra_env=_stub_env(judge_stub)
    )
    assert run.error is None, run.error
    assert run.output == {"triage": [_dumped(_expected(t)) for t in TICKETS]}
    assert len(_JudgeStub.requests) == len(TICKETS)
    for request in _JudgeStub.requests:
        assert request["output_config"]["format"]["schema"] == _sent_schema()


def test_without_passthrough_the_model_is_asked_for_every_field(
    tmp_path: Path, judge_stub: str
) -> None:
    """The negative half: the same judge, no passthrough, sends the whole schema."""
    pipeline = _pipeline()
    spec = pipeline.nodes[0].signature_spec
    assert spec is not None
    spec.passthrough = None
    app = tmp_path / "python"
    get_adapter("python").emit(pipeline, app)
    run_pipeline_trial(
        app, {"tickets": TICKETS}, timeout_seconds=60, extra_env=_stub_env(judge_stub)
    )
    sent = _JudgeStub.requests[0]["output_config"]["format"]["schema"]
    assert sent == structured_output_schema(OUTPUT_SCHEMA, node_id="triage")
    assert "ticket_id" in sent["properties"]


def test_the_emitted_python_judge_copies_from_the_input_as_given() -> None:
    """``exclude_unset``: T-1 has no reporter, so its copy has none, nested or
    at the top, as on TS. A plain model_dump would hand call_judge None."""
    import types

    from rote.adapters._py_common import emit_signature_module

    seen: dict[str, Any] = {}

    def call_judge(**kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        merged = helper.with_passthrough(_decide("down"), kwargs["inputs"], kwargs["passthrough"])
        seen["merged"] = merged
        return merged

    fake = types.ModuleType("signatures._rote_inference")
    fake.call_judge = call_judge  # type: ignore[attr-defined]
    package = types.ModuleType("signatures")
    package._rote_inference = fake  # type: ignore[attr-defined]
    source = emit_signature_module(
        _pipeline().nodes[0],
        anthropic_default_model="m",
        openai_default_model="m",
        generated_by="t",
        regen_command="t",
        context_note="t",
    )
    module = types.ModuleType("_passthrough_judge")
    import sys

    saved = {k: sys.modules.get(k) for k in ("signatures", "signatures._rote_inference")}
    sys.modules.update({"signatures": package, "signatures._rote_inference": fake})
    sys.modules["_passthrough_judge"] = module
    try:
        exec(compile(source, "_passthrough_judge", "exec"), module.__dict__)
        for name in ("Ticket", "TriageInput", "TriageOutput"):
            module.__dict__[name].model_rebuild(_types_namespace=module.__dict__)
        module.Triage().forward(module.TriageInput.model_validate({"ticket": TICKETS[0]}))
    finally:
        del sys.modules["_passthrough_judge"]
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value
    assert seen["inputs"] == {"ticket": TICKETS[0]}
    assert seen["merged"] == _expected(TICKETS[0])


# ───────── the compiled skill's driver ─────────


def _schema_of(block: str) -> dict[str, Any]:
    marker = "--- ANSWER SCHEMA (json, one per item) ---\n"
    loaded: dict[str, Any] = json.loads(block.split(marker, 1)[1].split("\n", 1)[0])
    return loaded


def test_decide_asks_only_for_the_decision_and_done_has_the_copies(tmp_path: Path) -> None:
    scripts = make_skill(tmp_path, _pipeline())
    status, out = drive(scripts, "start", json.dumps({"tickets": TICKETS}))
    assert status == 0 and header(out)["kind"] == "DECIDE", out
    assert _schema_of(out) == helper.passthrough_answer_schema(OUTPUT_SCHEMA, PASSTHROUGH)
    # The prompt reads only title and body; the copied ticket is not shown.
    assert "T-1" not in out and "prod" not in out
    answers = {
        "0": _decide(TICKETS[0]["body"]),
        # An agent that writes a copied field anyway is overruled by the input.
        "1": {**_decide(TICKETS[1]["body"]), "ticket_id": "FORGED"},
    }
    status, done = drive(scripts, "answer", header(out)["run"], "triage", json.dumps(answers))
    assert status == 0 and header(done)["kind"] == "DONE", done
    assert result_of(done)["triage"] == [_dumped(_expected(t)) for t in TICKETS]


def test_the_driver_keeps_copies_of_the_input_as_given(tmp_path: Path) -> None:
    scripts = make_skill(tmp_path, _pipeline())
    _, out = drive(scripts, "start", json.dumps({"tickets": TICKETS}))
    copied = journal(out)["pending"]["copied"]
    assert copied == [{"ticket": t} for t in TICKETS]  # T-1: no reporter key at all


def test_a_single_decide_merges_the_copies_too(tmp_path: Path) -> None:
    scripts = make_skill(tmp_path, _pipeline(fan=False))
    status, out = drive(scripts, "start", json.dumps({"ticket": TICKETS[1]}))
    assert status == 0 and header(out)["kind"] == "DECIDE", out
    marker = "--- ANSWER SCHEMA (json) ---\n"
    shown = json.loads(out.split(marker, 1)[1].split("\n", 1)[0])
    assert shown == helper.passthrough_answer_schema(OUTPUT_SCHEMA, PASSTHROUGH)
    answer = {**_decide(TICKETS[1]["body"]), "ticket_id": "FORGED"}
    status, done = drive(scripts, "answer", header(out)["run"], "triage", json.dumps(answer))
    assert status == 0 and header(done)["kind"] == "DONE", done
    assert result_of(done)["triage"] == _dumped(_expected(TICKETS[1]))


def test_a_do_step_answers_its_whole_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    """Passthrough is a judge's alone. A ``do`` step (an external_call with no
    code) has no signature_spec, so its plan step carries none, and the
    driver ignores one even if a step claimed it."""
    import sys

    from rote.skill_target._driver import answer_schema

    # The driver reads the helper under its emitted name.
    monkeypatch.setitem(sys.modules, "_rote_inference", helper)
    data = _pipeline().model_dump(mode="json", by_alias=True, exclude_none=True)
    data["nodes"].append(
        {
            "id": "file_ticket",
            "kind": "external_call",
            "description": "File each triaged ticket.",
            "impl": "extracted/filing.py:file_ticket",
            "inputs": {"triaged": "triage.output"},
            "output_schema": copy.deepcopy(OUTPUT_SCHEMA),
        }
    )
    data["edges"] = [{"from": "triage", "to": "file_ticket"}]
    data["exit_nodes"] = ["file_ticket"]
    plan = build_plan(
        Pipeline.model_validate(data), mcp="live", implementations={"file_ticket": "stub"}
    )
    steps = {s["id"]: s for s in plan["steps"]}
    do, judge = steps["file_ticket"], steps["triage"]
    assert do["kind"] == "do" and "passthrough" not in do
    assert judge["passthrough"] == PASSTHROUGH
    assert answer_schema(judge) == helper.passthrough_answer_schema(OUTPUT_SCHEMA, PASSTHROUGH)
    claimed = {**do, "passthrough": PASSTHROUGH}
    assert answer_schema(claimed) == do["output_schema"]


def test_helper_judges_in_the_driver_copy_in_code(tmp_path: Path, judge_stub: str) -> None:
    scripts = tmp_path / "scripts"
    write_scripts(scripts, build_plan(_pipeline(), mcp="live", judges="helper"))
    saved = dict(os.environ)
    try:
        os.environ.update(_stub_env(judge_stub))
        status, done = drive(scripts, "start", json.dumps({"tickets": TICKETS}))
    finally:
        os.environ.clear()
        os.environ.update(saved)
    assert status == 0 and header(done)["kind"] == "DONE", done
    assert result_of(done)["triage"] == [_dumped(_expected(t)) for t in TICKETS]
    assert [r["output_config"]["format"]["schema"] for r in _JudgeStub.requests] == [
        _sent_schema()
    ] * len(TICKETS)


# ───────── TypeScript: the same merge, and the emitted judges ─────────

#: (answer, input, passthrough) cases both languages must merge the same way.
MERGE_CASES: list[tuple[Any, Any, dict[str, str]]] = [
    ({"urgency": "low"}, {"ticket": TICKETS[0]}, PASSTHROUGH),
    ({"urgency": "low"}, {"ticket": TICKETS[1]}, PASSTHROUGH),
    ({"urgency": "low", "ticket_id": "FORGED"}, {"ticket": TICKETS[1]}, PASSTHROUGH),
    ({"a": 1}, {"t": None}, {"x": "t.id"}),
    ({"a": 1}, {"t": {"id": None}}, {"x": "t.id"}),
    ({"a": 1}, {"t": {"deep": {"er": [1, {"k": "v"}]}}}, {"x": "t.deep", "y": "t.deep.er"}),
    ({"a": 1}, {"t": [1, 2]}, {"x": "t.0"}),
    ({"a": 1, "x": "FORGED"}, {"t": {}}, {"x": "t.id"}),
    ({"a": 1, "x": "FORGED"}, {}, {"x": "t.id"}),
    (["not", "an", "object"], {"t": 1}, {"x": "t"}),
    (None, {"t": 1}, {"x": "t"}),
    ({}, {}, {}),
]

_TS_MERGE_MAIN = """
import { readFileSync } from "node:fs";

const cases = JSON.parse(readFileSync(0, "utf8")) as [unknown, unknown, Record<string, string>][];
const merged = cases.map(([answer, input, passthrough]) =>
    withPassthrough(answer, input, passthrough),
);
process.stdout.write(JSON.stringify(merged));
"""

_needs_node = pytest.mark.skipif(
    shutil.which("node") is None or shutil.which("npm") is None, reason="requires node + npm"
)


@pytest.mark.slow
@_needs_node
def test_python_and_ts_merge_the_same_way(tmp_path: Path) -> None:
    script = tmp_path / "merge.mts"
    script.write_text(SCHEMA_ERRORS_TS + _TS_MERGE_MAIN, encoding="utf-8")
    cases = json.loads(json.dumps(MERGE_CASES))
    done = subprocess.run(
        ["node", str(script)], input=json.dumps(cases), capture_output=True, text=True, timeout=60
    )
    assert done.returncode == 0, done.stderr
    ours = [helper.with_passthrough(a, i, p) for a, i, p in cases]
    assert json.loads(done.stdout) == ours
    # Not degenerate: the cases copy something, overrule the model, and skip.
    assert ours[2]["ticket_id"] == "T-2" and "x" not in ours[3] and ours[4]["x"] is None
    assert ours[7] == {"a": 1} and ours[8] == {"a": 1}  # a forged copy never survives


DRIVER_TS = """\
import { triage } from "./signatures/triage";

async function main(): Promise<void> {
    const tickets = JSON.parse(process.env.TICKETS ?? "[]") as unknown[];
    const out: unknown[] = [];
    for (const ticket of tickets) {
        out.push(
            await triage({ ticket }, {
                ANTHROPIC_API_KEY: "local-stub-key",
                ROTE_MODEL_TRIAGE: process.env.ROTE_MODEL_TRIAGE,
                ROTE_BASE_URL_TRIAGE: process.env.ROTE_BASE_URL_TRIAGE,
            }),
        );
    }
    console.log(JSON.stringify(out));
}

void main();
"""


@pytest.mark.slow
@_needs_node
def test_an_emitted_ts_judge_copies_in_code(tmp_path: Path, judge_stub: str) -> None:
    app = tmp_path / "app"
    get_adapter("dbos-ts").emit(_pipeline(), app)
    (app / "src" / "driver.ts").write_text(DRIVER_TS, encoding="utf-8")
    node_trees.install(app)
    build = subprocess.run(["npx", "tsc"], cwd=app, capture_output=True, text=True, timeout=300)
    assert build.returncode == 0, f"tsc failed:\n{build.stdout}\n{build.stderr}"
    proc = subprocess.run(
        ["node", "dist/driver.js"],
        cwd=app,
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, **_stub_env(judge_stub), "TICKETS": json.dumps(TICKETS)},
    )
    assert proc.returncode == 0, f"driver failed:\n{proc.stdout}\n{proc.stderr}"
    assert json.loads(proc.stdout.strip().splitlines()[-1]) == [_expected(t) for t in TICKETS]
    for request in _JudgeStub.requests:
        assert request["output_config"]["format"]["schema"] == _sent_schema()


@pytest.mark.slow
@_needs_node
def test_a_fanned_ts_judge_copies_in_a_real_inngest_run(
    tmp_path: Path, judge_stub: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole TS path as a workflow: one durable step per ticket under the
    Inngest dev server, each judge merging its own element's copies."""
    from rote.runners.inngest import run_inngest

    for key, value in _stub_env(judge_stub).items():
        monkeypatch.setenv(key, value)
    pipeline = _pipeline()
    app = tmp_path / "inngest"
    get_adapter("inngest").emit(pipeline, app)
    run = run_inngest(
        app, {"tickets": TICKETS}, pipeline=pipeline, signals={}, gate_order=[], timeout_seconds=240
    )
    assert run.error is None, run.error
    assert run.output == {"triage": [_expected(t) for t in TICKETS]}
    assert len(_JudgeStub.requests) == len(TICKETS)
    for request in _JudgeStub.requests:
        assert request["output_config"]["format"]["schema"] == _sent_schema()


@pytest.mark.slow
@_needs_node
@pytest.mark.parametrize(
    ("runtime", "client"), [("cloudflare", "workers-ai"), ("inngest", "openai")]
)
def test_every_ts_judge_lane_typechecks_with_passthrough(
    tmp_path: Path, runtime: str, client: str
) -> None:
    app = tmp_path / runtime
    get_adapter(runtime).emit(_pipeline(client=client), app)
    judge = (app / "src" / "signatures" / "triage.ts").read_text(encoding="utf-8")
    assert "withPassthrough(" in judge and '"ticket_id": "ticket.id"' in judge
    node_trees.install(app)
    build = subprocess.run(
        ["npx", "tsc", "--noEmit"], cwd=app, capture_output=True, text=True, timeout=300
    )
    assert build.returncode == 0, f"tsc failed:\n{build.stdout}\n{build.stderr}"


def test_a_judge_without_passthrough_emits_no_merge() -> None:
    """Every other judge's emitted module is byte-for-byte what it was."""
    from rote.adapters._py_common import emit_signature_module
    from rote.adapters._ts_common import emit_ts_signature_module

    plain = _pipeline().nodes[0].model_copy(deep=True)
    assert plain.signature_spec is not None
    plain.signature_spec.passthrough = None
    ts = emit_ts_signature_module(
        plain, anthropic_default_model="m", openai_default_model="m", generated_by="t"
    )
    py = emit_signature_module(
        plain,
        anthropic_default_model="m",
        openai_default_model="m",
        generated_by="t",
        regen_command="t",
        context_note="t",
    )
    assert "withPassthrough" not in ts and "PASSTHROUGH" not in ts
    assert "passthrough" not in py and "PASSTHROUGH" not in py


def test_every_committed_passthrough_names_real_fields() -> None:
    """Every example IR that passes fields through loads, holdout included."""
    found = 0
    for path in sorted((REPO / "examples").rglob("pipeline.yaml")):
        if "runtime" in path.parts:
            continue
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        pipeline = Pipeline.model_validate(data)
        found += sum(1 for n in pipeline.nodes if n.signature_spec and n.signature_spec.passthrough)
    assert found >= 2


def test_the_static_estimate_counts_only_the_fields_the_model_writes() -> None:
    from rote.eval.estimate import Range, _model_written_tokens
    from rote.eval.priors import Priors

    priors = Priors()
    per_field = priors.tokens_per_output_field
    with_copy = _pipeline().nodes[0]
    without = with_copy.model_copy(deep=True)
    assert without.signature_spec is not None
    without.signature_spec.passthrough = None
    # Copied: the model writes `urgency` and `reason` only.
    assert _model_written_tokens(with_copy, priors) == Range.exact(2 * per_field)
    # Not copied: the whole ticket too (id, title, body, reporter, and a
    # list of tags of one to `elements_per_list` values), then ticket_id
    # and reporter.
    ticket = Range(4 * per_field + per_field, 4 * per_field + per_field * priors.elements_per_list)
    assert _model_written_tokens(without, priors) == ticket + Range.exact(4 * per_field)
