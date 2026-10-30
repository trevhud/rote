"""A replay case from a recorded run of a stored skill (#335).

`rote replay prepare` turns a `rote baseline` recording into a case the
paired A/B replays: the recorded input, each recorded read as a fixture for
that exact call, the tools and whether each writes, and the fields to
compare. These tests build the recording by hand; no inference runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from rote.replay import CASE_ID, ReplayError, prepare

PIPELINE = {
    "name": "notes",
    "version": "0.1.0",
    "input": {"type": "In", "required": ["name"]},
    "nodes": [
        {
            "id": "read",
            "kind": "external_call",
            "description": "Read the note.",
            "mcp": {"server": "notes", "tool": "read_note"},
            "inputs": {"name": "pipeline.input.name"},
            "output": {"text": "str"},
        },
        {
            "id": "summarize",
            "kind": "pure_function",
            "description": "Summarize it.",
            "impl": "extracted/s.py:summarize",
            "inputs": {"note": "read.output"},
            "output": {"summary": "str"},
        },
    ],
    "edges": [{"from": "read", "to": "summarize"}],
    "entry_nodes": ["read"],
    "exit_nodes": ["summarize"],
}

TOOLS = {
    "servers": {
        "notes": {
            "tools": [
                {
                    "name": "read_note",
                    "description": "Read a note.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"],
                    },
                    "annotations": {"readOnlyHint": True},
                },
                {
                    "name": "write_note",
                    "description": "Write a note.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}, "text": {"type": "string"}},
                    },
                },
            ]
        }
    }
}


def _recording(tmp_path: Path, observed: list[dict[str, Any]], ok: bool = True) -> Path:
    base = tmp_path / "baseline-out" / "baseline"
    base.mkdir(parents=True)
    (base / "metrics.json").write_text(
        json.dumps({"input": {"name": "alpha"}, "runs": [{"succeeded": ok}]})
    )
    (base / "observed-tools.json").write_text(json.dumps(observed))
    return tmp_path / "baseline-out"


def _skill(tmp_path: Path) -> Path:
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: notes\ndescription: d\n---\n# Notes\n")
    (skill / "mcp-tools.json").write_text(json.dumps(TOOLS))
    return skill


def _compiled(tmp_path: Path, pipeline: dict[str, Any] = PIPELINE) -> Path:
    path = tmp_path / "compiled" / "pipeline.yaml"
    path.parent.mkdir()
    path.write_text(yaml.safe_dump(pipeline))
    return path


def _call(args: dict[str, Any], text: str, error: bool = False) -> dict[str, Any]:
    return {
        "server": "notes",
        "tool": "read_note",
        "input": args,
        "result": [{"type": "text", "text": text}],
        "is_error": error,
    }


def test_a_recording_becomes_a_reviewable_case(tmp_path: Path) -> None:
    observed = [
        _call({"name": "alpha"}, '{"text": "hello"}'),
        _call({"name": "alpha"}, '{"text": "hello"}'),  # the same call again: one fixture
        _call({"name": "beta"}, "plain text, not JSON"),
        _call({"name": "gamma"}, "boom", error=True),  # an error is not a recorded read
    ]
    done = prepare(
        _skill(tmp_path), _recording(tmp_path, observed), _compiled(tmp_path), tmp_path / "replay"
    )
    assert done.reads == 2
    assert done.writes == ("notes.write_note",)
    assert done.notes == ()
    doc = yaml.safe_load((tmp_path / "replay" / "fidelity" / "cases.yaml").read_text())
    [case] = doc["cases"]
    assert case["id"] == CASE_ID and case["input"] == {"name": "alpha"}
    # Each fixture answers one exact call, with the text the server sent.
    assert case["fixtures"] == {
        "notes": {
            "read_note": [
                {"when": {"name": "alpha"}, "result": '{"text": "hello"}'},
                {"when": {"name": "beta"}, "result": "plain text, not JSON"},
            ]
        }
    }
    # A write is declared with no response: nothing is invented.
    assert doc["servers"]["notes"]["tools"] == {
        "read_note": {"side_effect": False},
        "write_note": {"side_effect": True},
    }
    assert set(doc["output_fields"]) == {"summary"}


def test_the_case_loads_with_its_own_tools_and_may_hold_writes(tmp_path: Path) -> None:
    from rote.bench.fidelity import load_cases
    from rote.eval.skill_ab import SkillAbError, bench_case

    skill = _skill(tmp_path)
    prepare(
        skill,
        _recording(tmp_path, [_call({"name": "alpha"}, "x")]),
        _compiled(tmp_path),
        tmp_path / "replay",
    )
    casefile = load_cases(tmp_path / "replay")
    tools = casefile.servers["notes"]["tools"]
    assert tools["write_note"]["side_effect"] is True
    assert tools["read_note"]["parameters"]["required"] == ["name"]
    _, case = bench_case(skill, CASE_ID, tmp_path / "replay")
    assert case.fixtures["notes"]["read_note"][0]["when"] == {"name": "alpha"}
    # An example's case stays read-only: the same file there is refused.
    example = tmp_path / "example"
    (example / "fidelity").mkdir(parents=True)
    text = (tmp_path / "replay" / "fidelity" / "cases.yaml").read_text()
    (example / "fidelity" / "cases.yaml").write_text(
        text.replace("tools_dir: ..", "tools_dir: ../../replay")
    )
    with pytest.raises(SkillAbError, match="write, and the eval is read-only"):
        bench_case(example / "skill", CASE_ID)


def test_different_answers_to_one_call_and_missing_gate_answers_are_noted(tmp_path: Path) -> None:
    gated = {
        **PIPELINE,
        "nodes": [
            *PIPELINE["nodes"][:1],
            {
                "id": "approve",
                "kind": "hitl_gate",
                "description": "Approve.",
                "signal": "ok",
                "output": {"approved": "bool"},
            },
            PIPELINE["nodes"][1],
        ],
        "edges": [{"from": "read", "to": "approve"}, {"from": "approve", "to": "summarize"}],
    }
    observed = [_call({"name": "alpha"}, "first"), _call({"name": "alpha"}, "second")]
    done = prepare(
        _skill(tmp_path), _recording(tmp_path, observed), _compiled(tmp_path, gated), tmp_path / "r"
    )
    assert any("answered differently: the replay answers the first" in n for n in done.notes)
    assert any("1 gate(s) and 0 answer(s)" in n for n in done.notes)


def test_a_recording_with_no_successful_run_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="no recorded trial succeeded"):
        prepare(
            _skill(tmp_path),
            _recording(tmp_path, [], ok=False),
            _compiled(tmp_path),
            tmp_path / "r",
        )


def test_the_report_compares_proposed_writes_and_labels_replay_timing(tmp_path: Path) -> None:
    from rote.eval import skill_ab

    def trial(
        side: str, writes: tuple[dict[str, Any], ...], incomplete: tuple[str, ...] = ()
    ) -> Any:
        usage, _ = skill_ab.parse_stream([])
        return skill_ab.Trial(
            side=side,  # type: ignore[arg-type]
            index=1,
            control=False,
            wall_seconds=1.0,
            usage=usage,
            output={},
            error=None,
            denied_tools=(),
            calls=(),
            portable_calls=(),
            driver=None,
            work_dir=tmp_path,
            writes=writes,
            incomplete=incomplete,
        )

    w = {"server": "notes", "tool": "write_note", "arguments": {"name": "b"}, "answered": False}
    other = {**w, "arguments": {"name": "c"}}
    subject = type("S", (), {"label": "notes/recorded", "replay": True})()
    sides = {
        "original": type("Sum", (), {"trials": (trial("original", (w,)),)})(),
        "compiled": type("Sum", (), {"trials": (trial("compiled", (other,), ("x",)),)})(),
    }
    report = type("R", (), {"subject": subject, "sides": sides})()
    lines = skill_ab._replay_lines(report)  # type: ignore[arg-type]
    assert "not live service latency" in lines[0]
    assert "proposed writes differ (original 1, compiled 1; none sent)" in lines[1]
    assert lines[2] == '  - original: notes.write_note {"name": "b"}'
    assert sides["compiled"].trials[0].problems == ("incomplete: x",)


def test_a_live_listing_keeps_the_wire_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Found live: dumped by field name, `readOnlyHint` became `read_only_hint`,
    so every tool read as a write, and `inputSchema` became `input_schema`, so
    the replay refused its own snapshot."""
    import mcp.types as mt

    listed = [mt.Tool.model_validate(t) for t in TOOLS["servers"]["notes"]["tools"]]

    class FakeClient:
        async def __aenter__(self) -> FakeClient:
            return self

        async def __aexit__(self, *exc: object) -> None:
            return None

        async def list_tools(self) -> list[Any]:
            return listed

    import rote.mcp._runtime_helper as helper

    monkeypatch.setattr(helper, "mcp_client", lambda server, _url: FakeClient())
    skill = _skill(tmp_path)
    (skill / "mcp-tools.json").unlink()  # no bundled snapshot: list live
    done = prepare(
        skill,
        _recording(tmp_path, [_call({"name": "alpha"}, "x")]),
        _compiled(tmp_path),
        tmp_path / "replay",
    )
    assert done.writes == ("notes.write_note",)
    from rote.bench.fidelity import load_cases

    tools = load_cases(tmp_path / "replay").servers["notes"]["tools"]
    assert tools["read_note"]["side_effect"] is False
    assert tools["read_note"]["parameters"]["required"] == ["name"]


def test_a_retried_write_agrees_on_what_but_not_how_often(tmp_path: Path) -> None:
    """Found live: the compiled step retried a failed write three times."""
    from rote.eval import skill_ab

    usage, _ = skill_ab.parse_stream([])

    def trial(side: str, writes: tuple[dict[str, Any], ...]) -> Any:
        return skill_ab.Trial(
            side=side,  # type: ignore[arg-type]
            index=1,
            control=False,
            wall_seconds=1.0,
            usage=usage,
            output={},
            error=None,
            denied_tools=(),
            calls=(),
            portable_calls=(),
            driver=None,
            work_dir=tmp_path,
            writes=writes,
        )

    w = {"server": "notes", "tool": "write_note", "arguments": {"name": "d"}, "answered": False}
    subject = type("S", (), {"label": "x", "replay": True})()
    sides = {
        "original": type("Sum", (), {"trials": (trial("original", (w,)),)})(),
        "compiled": type("Sum", (), {"trials": (trial("compiled", (w, w, w, w)),)})(),
    }
    lines = skill_ab._replay_lines(type("R", (), {"subject": subject, "sides": sides})())  # type: ignore[arg-type]
    assert (
        "proposed writes agree on what to write, not how often (original 1, compiled 4" in lines[1]
    )
    assert lines[3] == '  - compiled: notes.write_note {"name": "d"} x4'


@pytest.mark.asyncio
async def test_a_fixture_answers_as_the_real_tool_did_under_its_output_schema(
    tmp_path: Path,
) -> None:
    """Found live: a fastmcp server declares `{"result": str}` with its wrap
    flag; a fixture without the schema handed the client the wrapper, and the
    compiled step failed reading a dict where the note's text belonged."""
    from fastmcp import Client

    from rote.bench.fixture_mcp import build_server
    from rote.mcp._runtime_helper import returned_value

    schema = {
        "type": "object",
        "properties": {"result": {"type": "string"}},
        "required": ["result"],
        "x-fastmcp-wrap-result": True,
    }
    spec = {
        "server": "notes",
        "tools": {
            "read_note": {
                "parameters": {"type": "object", "properties": {"name": {"type": "string"}}},
                "output_schema": schema,
                "responses": [{"when": {"name": "a"}, "result": '{"result": "hello"}'}],
            },
            "plain": {
                "parameters": {"type": "object", "properties": {}},
                "responses": [{"result": "just text"}],
            },
        },
    }
    async with Client(build_server(spec, tmp_path / "log.jsonl")) as client:
        wrapped = await client.call_tool("read_note", {"name": "a"})
        plain = await client.call_tool("plain", {})
    assert wrapped.structured_content == {"result": "hello"}
    assert returned_value(wrapped.structured_content, schema) == "hello"
    # No declared schema: text only, as before.
    assert plain.structured_content is None
    assert plain.content[0].text == "just text"
