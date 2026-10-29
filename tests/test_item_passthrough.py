"""Per-item passthrough: a judge's list items get their source fields from the
input record their key names, never from the model's recitation (#320).

A real compile's judge chose which pull requests needed a ticket, drafted a
summary, and copied each one's repo, number and original body into its
output; the step after it appended a ticket reference to that copy and wrote
it back. Nothing compared the copy with the fetched record, so a model that
changed one character changed the pull request. ``items[].body:
prs[number].body`` makes the model write only the number and the summary;
code fills the body from the ``prs`` record with that number, and a number
that matches no record, several records, or another item's fails the judge
before anything downstream runs.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from rote.adapters._ts_common import SCHEMA_ERRORS_TS
from rote.inference import _runtime_helper as helper
from rote.ir import LLMSignature, Pipeline
from tests._skill_driver import drive, header, make_skill, result_of

PR = {
    "type": "object",
    "properties": {
        "number": {"type": "integer"},
        "repo": {"type": "string"},
        "body": {"type": "string"},
    },
    "required": ["number", "repo", "body"],
    "additionalProperties": False,
}
ITEM = {
    "type": "object",
    "properties": {
        "number": {"type": "integer"},
        "repo": {"type": "string"},
        "body": {"type": "string"},
        "summary": {"type": "string"},
    },
    "required": ["number", "repo", "body", "summary"],
    "additionalProperties": False,
}
PASSTHROUGH = {"items[].body": "prs[number].body", "items[].repo": "prs[number].repo"}


def _spec(**changes: Any) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "prompt": "Pick the pull requests that need a ticket: {{ prs }}",
        "input_schema": {
            "type": "object",
            "properties": {"prs": {"type": "array", "items": PR}},
            "required": ["prs"],
        },
        "output_schema": {
            "type": "object",
            "properties": {"items": {"type": "array", "items": ITEM}},
            "required": ["items"],
            "additionalProperties": False,
        },
        "passthrough": dict(PASSTHROUGH),
    }
    spec.update(changes)
    return spec


PRS = [
    {"number": 7, "repo": "acme/api", "body": "ORIGINAL\r\ntext\n"},
    {"number": 9, "repo": "acme/web", "body": "second"},
]


def test_a_per_item_passthrough_loads() -> None:
    assert LLMSignature.model_validate(_spec()).passthrough == PASSTHROUGH


@pytest.mark.parametrize(
    ("passthrough", "message"),
    [
        ({"items[].body": "prs.body"}, "must name the input list, its key and the field"),
        ({"items[].bod": "prs[number].body"}, "output 'items' items declare no property 'bod'"),
        ({"items[].body": "prs[id].body"}, "declare no property 'id'"),
        ({"items[].body": "prs[number].repo", "items[].repo": "prs[number].repo"}, None),
        ({"items[].body": "prs[repo].body"}, None),
        ({"items[].number": "prs[number].number"}, "is what the model writes to match"),
        ({"rows[].body": "prs[number].body"}, "'rows' is not an output_schema property"),
        ({"items[].body": "missing[number].body"}, "input 'missing'"),
        (
            {"items[].body": "prs[number].body", "items[].repo": "prs[repo].repo"},
            "more than one input list or key",
        ),
    ],
)
def test_an_unsound_per_item_passthrough_is_refused(
    passthrough: dict[str, str], message: str | None
) -> None:
    if message is None:
        # Same schema (string to string) and a string key: both are fine.
        LLMSignature.model_validate(_spec(passthrough=passthrough))
        return
    with pytest.raises(ValueError, match=message):
        LLMSignature.model_validate(_spec(passthrough=passthrough))


def test_a_key_that_is_not_a_string_or_integer_is_refused() -> None:
    pr = {**PR, "properties": {**PR["properties"], "number": {"type": "number"}}}
    item = {**ITEM, "properties": {**ITEM["properties"], "number": {"type": "number"}}}
    spec = _spec(
        input_schema={"type": "object", "properties": {"prs": {"type": "array", "items": pr}}},
        output_schema={"type": "object", "properties": {"items": {"type": "array", "items": item}}},
    )
    with pytest.raises(ValueError, match="must be a string or an integer"):
        LLMSignature.model_validate(spec)


def test_the_model_is_never_asked_for_the_copied_fields() -> None:
    asked = helper.passthrough_answer_schema(_spec()["output_schema"], PASSTHROUGH)
    item = asked["properties"]["items"]["items"]
    assert set(item["properties"]) == {"number", "summary"}
    assert item["required"] == ["number", "summary"]
    # The declared schema is untouched: the merged answer is checked against it.
    assert set(_spec()["output_schema"]["properties"]["items"]["items"]["properties"]) == {
        "number",
        "repo",
        "body",
        "summary",
    }


def test_the_copies_are_the_fetched_records_bytes_never_the_models() -> None:
    answer = {
        "items": [
            {"number": 9, "summary": "b", "body": "ALTERED BY MODEL"},
            {"number": 7, "summary": "a"},
        ]
    }
    merged = helper.with_passthrough(answer, {"prs": PRS}, PASSTHROUGH)
    assert merged == {
        "items": [
            {"number": 9, "summary": "b", "repo": "acme/web", "body": "second"},
            {"number": 7, "summary": "a", "repo": "acme/api", "body": "ORIGINAL\r\ntext\n"},
        ]
    }
    assert helper.schema_errors(_spec()["output_schema"], merged, "output") == []


@pytest.mark.parametrize(
    ("items", "prs", "message"),
    [
        ([{"number": 8, "summary": "x"}], PRS, "number 8 matches no record(s) of the input's prs"),
        ([{"summary": "x"}], PRS, "the item names no number"),
        (
            [{"number": 7, "summary": "x"}, {"number": 7, "summary": "y"}],
            PRS,
            "number 7 names a record another item already names",
        ),
        ([{"number": 7, "summary": "x"}], [*PRS, PRS[0]], "matches 2 record(s)"),
        ([{"number": True, "summary": "x"}], [{"number": 1, "body": "b"}], "matches no record"),
    ],
)
def test_an_item_that_names_no_single_record_fails(
    items: list[Any], prs: list[Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message.replace("(", r"\(").replace(")", r"\)")):
        helper.with_passthrough({"items": items}, {"prs": prs}, PASSTHROUGH)


def test_call_judge_fails_the_judge_before_anything_downstream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A judge has no writes, so its step's retry asks again; the write after it
    never sees a mismatched record."""
    asked: list[dict[str, Any]] = []
    monkeypatch.setattr(helper, "select_provider", lambda *a, **k: "api")

    def answer(**kw: Any) -> tuple[dict[str, Any], list[Any]]:
        asked.append(kw["output_schema"])
        return {"items": [{"number": 8, "summary": "x"}]}, []

    monkeypatch.setattr(helper, "structured_call", answer)
    with pytest.raises(RuntimeError, match="judge prepare: passthrough into items"):
        helper.call_judge(
            node_id="prepare",
            client="anthropic",
            model="m",
            prompt="p",
            output_schema=_spec()["output_schema"],
            passthrough=PASSTHROUGH,
            inputs={"prs": PRS},
        )
    # The request never asked for the copied fields.
    assert set(asked[0]["properties"]["items"]["items"]["properties"]) == {"number", "summary"}


_TS_MAIN = """
import { readFileSync } from "node:fs";

const cases = JSON.parse(readFileSync(0, "utf8")) as [unknown, unknown, Record<string, string>][];
const out = cases.map(([answer, input, passthrough]) => {
    try {
        return { ok: withPassthrough(answer, input, passthrough) };
    } catch (e) {
        return { error: (e as Error).message };
    }
});
process.stdout.write(JSON.stringify(out));
"""

CASES: list[tuple[Any, Any, dict[str, str]]] = [
    ({"items": [{"number": 9, "summary": "b", "body": "X"}]}, {"prs": PRS}, PASSTHROUGH),
    ({"items": [{"number": 8, "summary": "x"}]}, {"prs": PRS}, PASSTHROUGH),
    ({"items": [{"summary": "x"}]}, {"prs": PRS}, PASSTHROUGH),
    ({"items": [{"number": 7}, {"number": 7}]}, {"prs": PRS}, PASSTHROUGH),
    ({"items": [{"number": 7}]}, {"prs": [*PRS, PRS[0]]}, PASSTHROUGH),
    ({"items": [{"number": 7}]}, {"prs": "nope"}, PASSTHROUGH),
    ({"items": "not a list"}, {"prs": PRS}, PASSTHROUGH),
]


def _python(answer: Any, inputs: Any, passthrough: dict[str, str]) -> dict[str, Any]:
    try:
        return {"ok": helper.with_passthrough(answer, inputs, passthrough)}
    except ValueError as e:
        return {"error": str(e)}


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("node") is None, reason="requires node")
def test_python_and_ts_copy_and_refuse_the_same_way(tmp_path: Path) -> None:
    script = tmp_path / "merge.mts"
    script.write_text(SCHEMA_ERRORS_TS + _TS_MAIN, encoding="utf-8")
    done = subprocess.run(
        ["node", str(script)], input=json.dumps(CASES), capture_output=True, text=True, timeout=60
    )
    assert done.returncode == 0, done.stderr
    ours = [_python(*case) for case in json.loads(json.dumps(CASES))]
    assert json.loads(done.stdout) == ours
    # Not degenerate: one copies, five refuse with a message, one passes through.
    assert ours[0]["ok"]["items"][0]["body"] == "second"
    assert sum("error" in r for r in ours) == 5


def _pipeline() -> Pipeline:
    return Pipeline.model_validate(
        {
            "name": "repairs",
            "version": "0.1.0",
            "source_skill": "tests/fixtures/repairs",
            "description": "Pick the pull requests that need a ticket.",
            "input": {"type": "In", "required": ["prs"], "optional": []},
            "nodes": [
                {
                    "id": "prepare",
                    "kind": "llm_judge",
                    "description": "Pick and summarize.",
                    "inputs": {"prs": "pipeline.input.prs"},
                    "signature_spec": {**_spec(), "client": "anthropic"},
                }
            ],
            "edges": [],
            "entry_nodes": ["prepare"],
            "exit_nodes": ["prepare"],
        }
    )


def test_the_compiled_skill_fills_the_records_and_reasks_a_bad_key(tmp_path: Path) -> None:
    """The agent answers in a DECIDE block; the driver copies the record's
    fields in, and an item naming no record is asked again, never written."""
    scripts = make_skill(tmp_path, _pipeline())
    status, out = drive(scripts, "start", json.dumps({"prs": PRS}))
    assert status == 0 and header(out)["kind"] == "DECIDE", out
    run = header(out)["run"]
    bad = {"items": [{"number": 8, "summary": "x"}]}
    _, again = drive(scripts, "answer", run, "prepare", json.dumps(bad))
    assert header(again)["kind"] == "DECIDE" and header(again)["attempt"] == "2", again
    assert "number 8 matches no record" in again
    good = {"items": [{"number": 7, "summary": "a", "body": "ALTERED BY MODEL"}]}
    status, done = drive(scripts, "answer", run, "prepare", json.dumps(good))
    assert status == 0 and header(done)["kind"] == "DONE", done
    assert result_of(done)["prepare"] == {
        "items": [{"number": 7, "summary": "a", "repo": "acme/api", "body": "ORIGINAL\r\ntext\n"}]
    }
