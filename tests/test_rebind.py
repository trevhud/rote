"""Data-flow findings go back to the structure pass, which decides bindings (#329).

A judge copied the signed-in account id into its output and the creation
fan-out read only the items, so every ticket lost its assignee. The agent
proposed the binding; rote recorded it and shipped without it, because the
text agent may not change a structure decision. Now the structure pass
re-decides: one schema-locked call proposes bindings, and code applies one
only when it is an addition that resolves and the pipeline still validates.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from rote.compiler.system1 import structure
from rote.compiler.system1.tiebreak import Lane
from rote.ir import Pipeline
from tests.test_compile_boundaries import _orphan_nodes, _pipeline, _write

_PROBLEM = "prepare copies `me.account_id` into its output as `assignee_account_id` ..."


class _Call:
    def __init__(self, *answers: dict[str, Any]) -> None:
        self.answers = list(answers)
        self.prompts: list[str] = []

    def __call__(self, **kw: Any) -> tuple[dict[str, Any], list[tuple[str, dict[str, Any]]]]:
        assert kw["output_schema"] is structure.REBIND_SCHEMA
        self.prompts.append(kw["prompt"])
        return copy.deepcopy(self.answers.pop(0)), [(kw["model"], {"input_tokens": 9})]


def _work(tmp_path: Path) -> Path:
    pipeline = _pipeline(_orphan_nodes({"ticket": "prepare.output.items"}))
    work = _write(tmp_path, pipeline, {})
    (work / "system1").mkdir()
    (work / "system1" / "skeleton.yaml").write_text((work / "pipeline.yaml").read_text())
    return work


def _skeleton(work: Path) -> dict[str, Any]:
    doc = yaml.safe_load((work / "system1" / "skeleton.yaml").read_text())
    return {n["id"]: n for n in doc["nodes"]}


def _bindings(*entries: tuple[str, str, str]) -> dict[str, Any]:
    return {
        "bindings": [
            {"node": n, "param": p, "ref": r, "why": "the skill assigns it"} for n, p, r in entries
        ]
    }


def test_a_binding_that_resolves_is_written_to_the_skeleton(tmp_path: Path) -> None:
    work = _work(tmp_path)
    call = _Call(_bindings(("create", "who", "prepare.output.assignee_account_id")))
    changes, usage = structure.rebind(
        work, {"prepare": [_PROBLEM]}, "SKILL", Lane("claude-cli"), call=call
    )
    assert changes == [
        "`create`.who <- `prepare.output.assignee_account_id`: bound (the skill assigns it)"
    ]
    assert [row["input_tokens"] for row in usage] == [9]
    create = _skeleton(work)["create"]
    assert create["inputs"] == {
        "ticket": "prepare.output.items",
        "who": "prepare.output.assignee_account_id",
    }
    # The skeleton still loads, and fans over the list with the field shared.
    doc = yaml.safe_load((work / "system1" / "skeleton.yaml").read_text())
    Pipeline.model_validate(doc)
    # The prompt carries the problem and what each step returns.
    assert _PROBLEM in call.prompts[0]
    assert '"assignee_account_id"' in call.prompts[0]


def test_an_edge_is_added_when_nothing_orders_the_source_first(tmp_path: Path) -> None:
    work = _work(tmp_path)
    call = _Call(
        _bindings(
            ("get_identity", "x", "pipeline.input.region"),
            ("create", "me", "get_identity.output.account_id"),
        )
    )
    changes, _ = structure.rebind(work, {"create": ["p"]}, "S", Lane("claude-cli"), call=call)
    assert all(": bound (" in c for c in changes), changes
    doc = yaml.safe_load((work / "system1" / "skeleton.yaml").read_text())
    # get_identity -> prepare -> create already orders it: no new edge.
    assert {"from": "get_identity", "to": "create"} not in doc["edges"]


@pytest.mark.parametrize(
    ("entry", "why"),
    [
        (("create", "who", "prepare.output.assignee"), "`prepare` declares no field `assignee`"),
        (("create", "ticket", "prepare.output.assignee_account_id"), "already has that parameter"),
        (("create", "who", "create.output.x"), "cannot read its own result"),
        (("create", "Bad Name", "prepare.output.assignee_account_id"), "not an identifier"),
        (("nope", "who", "prepare.output.assignee_account_id"), "no such step"),
        (("create", "who", "ghost.output.x"), "no step `ghost`"),
        # A reference that makes a cycle: get_identity would read create.
        (("get_identity", "back", "create.output"), "would not validate"),
    ],
)
def test_a_binding_that_does_not_resolve_is_refused(
    tmp_path: Path, entry: tuple[str, str, str], why: str
) -> None:
    work = _work(tmp_path)
    before = (work / "system1" / "skeleton.yaml").read_text()
    [change] = structure.rebind(
        work, {"prepare": ["p"]}, "S", Lane("claude-cli"), call=_Call(_bindings(entry))
    )[0]
    assert ": not applied: " in change and why in change, change
    assert (work / "system1" / "skeleton.yaml").read_text() == before


def test_a_failed_call_raises_structure_unavailable(tmp_path: Path) -> None:
    def fails(**kw: Any) -> Any:
        raise RuntimeError("boom")

    with pytest.raises(structure.StructureUnavailable, match="re-deciding data flow"):
        structure.rebind(_work(tmp_path), {"prepare": ["p"]}, "S", Lane("claude-cli"), call=fails)


def test_an_edge_is_added_when_the_source_runs_on_another_branch(tmp_path: Path) -> None:
    """Two entry branches: binding across them needs the ordering edge."""

    def fn(id_: str, **extra: Any) -> dict[str, Any]:
        return {
            "id": id_,
            "kind": "pure_function",
            "description": id_,
            "impl": f"extracted/{id_}.py:{id_}",
            **extra,
        }

    pipeline = Pipeline.model_validate(
        {
            "name": "branches",
            "version": "0.1.0",
            "input": {"type": "In", "required": []},
            "nodes": [
                fn("whoami", output={"account_id": "str"}),
                fn("load", output={"rows": "list"}),
                fn("write", inputs={"rows": "load.output.rows"}),
            ],
            "edges": [{"from": "load", "to": "write"}],
            "entry_nodes": ["whoami", "load"],
            "exit_nodes": ["write"],
        }
    )
    work = _write(tmp_path, pipeline, {})
    (work / "system1").mkdir()
    (work / "system1" / "skeleton.yaml").write_text((work / "pipeline.yaml").read_text())
    call = _Call(_bindings(("write", "owner", "whoami.output.account_id")))
    [change], _ = structure.rebind(work, {"write": ["p"]}, "S", Lane("claude-cli"), call=call)
    assert ": bound (" in change
    doc = yaml.safe_load((work / "system1" / "skeleton.yaml").read_text())
    assert {"from": "whoami", "to": "write"} in doc["edges"]
    assert Pipeline.model_validate(doc).node_by_id("write").inputs == {
        "rows": "load.output.rows",
        "owner": "whoami.output.account_id",
    }


# ───────── routing ─────────


def test_the_single_session_repair_sends_data_flow_to_the_structure_pass_once(
    tmp_path: Path,
) -> None:
    from rote.compiler import Compiler

    pipeline = _pipeline(_orphan_nodes({"ticket": "prepare.output.items"}))
    work = _write(tmp_path, pipeline, {"create.py": "def create(ticket):\n    return ticket\n"})
    (work / "system1").mkdir()
    (work / "system1" / "proposals.yaml").write_text(
        yaml.safe_dump(
            [
                {
                    "node": "create",
                    "field": "inputs",
                    "value": {"who": "prepare.output.assignee_account_id"},
                    "reason": "tickets go to the signed-in user",
                }
            ]
        )
    )
    asked: list[dict[str, list[str]]] = []

    def rebind(findings: dict[str, list[str]]) -> list[str]:
        asked.append(findings)
        return ["`create`.who <- `prepare.output.assignee_account_id`: bound (assignee)"]

    repair = Compiler(run_generated_tests=False)._deliverable_repairer(
        None, tmp_path, rebind=rebind
    )
    instruction = repair(work / "pipeline.yaml")
    [first] = asked
    assert "no later step reads `prepare.output.assignee_account_id`" in first["prepare"][0]
    assert first["create"] == [
        "the agent proposes reads {'who': 'prepare.output.assignee_account_id'} "
        "(because: tickets go to the signed-in user)"
    ]
    assert instruction is not None
    assert "rote's structure pass, which decides what each step reads, added" in instruction
    assert "`create`.who <- `prepare.output.assignee_account_id`" in instruction
    # The same findings are not sent again: what the pass keeps is its decision.
    repair(work / "pipeline.yaml")
    assert len(asked) == 1


@pytest.mark.asyncio
async def test_the_parallel_repair_sends_data_flow_to_the_structure_pass(tmp_path: Path) -> None:
    from rote.compiler.system1.generate import Settings, TextWriter
    from rote.compiler.verify import Verification
    from rote.contracts import ContractFinding
    from tests.test_system1_generate import _context

    ctx, _ = _context(tmp_path, "ops-report")
    unread = ContractFinding(
        node_id="some_judge", kind="passthrough_unread", severity="warning", message="dropped"
    )
    calls: list[dict[str, list[str]]] = []
    verdicts = iter([Verification(pipeline=None, findings=[unread]), Verification(pipeline=None)])

    def rebind(findings: dict[str, list[str]]) -> list[str]:
        calls.append(findings)
        return ["`x`.y <- `some_judge.output.z`: bound (needed)"]

    writer = TextWriter(None, ctx, "", Settings(), None, rebind=rebind)  # type: ignore[arg-type]
    verification, rounds = await writer.repair(lambda _: next(verdicts))
    assert calls == [{"some_judge": ["dropped"]}]
    assert verification.ok and rounds == 1
