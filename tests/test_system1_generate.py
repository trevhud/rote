"""Parallel per-node generation of System-1's text (rote.compiler.system1.generate).

No inference: every model call goes to a fake generator that answers from
the request, so the orchestration (interface, pieces, code checks, repairs,
the shared answer schema, cache priming, concurrency) is what is tested.
"""

from __future__ import annotations

import ast
import asyncio
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from rote.compiler import Compiler, system1
from rote.compiler.drivers import DriverError, Generation, GenerationRequest
from rote.compiler.grounding import Grounding
from rote.compiler.system1 import generate
from rote.compiler.system1.generate import (
    ANSWER_SCHEMA,
    Context,
    Ledger,
    Settings,
    apply_answer,
    attribute,
    def_line,
    interface_problems,
    interface_request,
    judge_problems,
    judge_request,
    module_problems,
    module_request,
)
from rote.contracts import ContractFinding
from rote.ir import NodeKind, load_pipeline

REPO = Path(__file__).resolve().parent.parent
EXAMPLES = REPO / "examples"


def _context(tmp_path: Path, skill: str = "expense-audit") -> tuple[Context, system1.Plan]:
    work = tmp_path / "work"
    work.mkdir()
    plan = system1.prepare(EXAMPLES / skill / "skill", work)
    return Context.load(work, plan, Grounding()), plan


def _field(path: str, kind: str, items: str = "", required: bool = True, **extra: Any) -> Any:
    return {
        "path": path,
        "type": kind,
        "items": items,
        "required": required,
        "nullable": extra.get("nullable", False),
        "enum": extra.get("enum", []),
        "description": extra.get("description", ""),
    }


def _fields_for(node: Any) -> list[dict[str, Any]]:
    """What the fake model answers for a node's return value."""
    if isinstance(node.output, dict) and "items" in node.output:
        return [_field("items", "array", "object")]
    return [_field("total", "number")]


def _schema_for(node: Any) -> dict[str, Any]:
    """The JSON Schema :func:`_fields_for` describes, written out by hand."""
    if isinstance(node.output, dict) and "items" in node.output:
        return {
            "type": "object",
            "properties": {
                "items": {"type": "array", "items": {"type": "object", "properties": {}}}
            },
            "required": ["items"],
        }
    return {"type": "object", "properties": {"total": {"type": "number"}}, "required": ["total"]}


def _blank() -> dict[str, Any]:
    return {
        "description": "",
        "output_fields": [],
        "termination": {"condition": "", "max_iterations": 0},
        "module": "",
        "test": "",
        "prompt": "",
        "client": "anthropic",
        "passthrough": [],
        "edits": [],
    }


def _module_for(prompt: str, node_id: str, *, read_params: bool = True) -> tuple[str, str]:
    """A module that uses the request's def line verbatim, and a test of it."""
    line = next(ln for ln in prompt.splitlines() if ln.startswith(f"def {node_id}("))
    # Read every parameter, constants included: a module that ignores one
    # is dead config, which verification sends back for repair.
    fn = ast.parse(f"{line}\n    pass\n").body[0]
    assert isinstance(fn, ast.FunctionDef)
    names = [a.arg for a in [*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs]]
    reads = f"    _ = ({', '.join(names)},)\n" if names and read_params else ""
    module = f"{line}\n{reads}    return {{'total': 1.0, 'items': []}}\n"
    test = (
        f"from extracted.{node_id} import {node_id}\n\n\n"
        f"def test_{node_id}():\n    assert callable({node_id})\n"
    )
    return module, test


class FakeGenerator:
    """Answers every request from what it asks, and records what it was asked."""

    name = "fake-generate"

    def __init__(self, ctx_nodes: dict[str, Any]) -> None:
        self.nodes = ctx_nodes
        self.requests: list[GenerationRequest] = []
        self.in_flight = 0
        self.max_in_flight = 0
        self.started: list[float] = []
        self.ended: list[float] = []
        #: node id -> answers to give, in order, before the default one.
        self.scripted: dict[str, list[dict[str, Any]]] = {}
        self.fail: set[str] = set()
        #: labels that fail on their first call only (a transient error).
        self.fail_once: set[str] = set()
        #: labels whose first module ignores its parameters (dead constants).
        self.dead_once: set[str] = set()
        #: seconds a label's call takes (default 0.01).
        self.delay: dict[str, float] = {}

    def is_available(self) -> tuple[bool, str]:
        return (True, "")

    async def run(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise AssertionError("the parallel path must not run an agent session")

    async def generate(self, request: GenerationRequest) -> Generation:
        self.requests.append(request)
        self.started.append(asyncio.get_running_loop().time())
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self.delay.get(request.label, 0.01))
            if request.label in self.fail_once:
                self.fail_once.discard(request.label)
                raise DriverError(f"claude -p failed for {request.label}: overloaded")
            if request.label in self.fail:
                raise DriverError(
                    f"claude -p failed for {request.label}: structured_output_retry_exhausted",
                    metadata={"input_tokens": 1, "output_tokens": 2, "cost_usd": 0.5},
                )
            answer = _blank()
            if self.scripted.get(request.label):
                answer.update(self.scripted[request.label].pop(0))
            elif request.label.startswith("interface:"):
                node = self.nodes[request.label.split(":", 1)[1]]
                answer["description"] = f"Does {node.id}."
                answer["output_fields"] = _fields_for(node)
                if node.kind is NodeKind.AGENT_LOOP and node.termination is None:
                    answer["termination"] = {"condition": "all done", "max_iterations": 3}
            else:
                node = self.nodes[request.label]
                if node.kind is NodeKind.LLM_JUDGE:
                    answer["prompt"] = (
                        "Decide. Return your decision via the structured output tool."
                    )
                else:
                    dead = node.id in self.dead_once
                    self.dead_once.discard(node.id)
                    answer["module"], answer["test"] = _module_for(
                        request.prompt, node.id, read_params=not dead
                    )
            return Generation(
                payload=answer,
                model="claude-sonnet-5-5",
                input_tokens=3,
                output_tokens=100,
                cache_write_tokens=50,
                cache_read_tokens=1000,
                cost_usd=0.01,
                duration_ms=10,
            )
        finally:
            self.in_flight -= 1
            self.ended.append(asyncio.get_running_loop().time())


async def _write(
    ctx: Context,
    plan: system1.Plan,
    fake: FakeGenerator,
    inject: list[list[ContractFinding]] | None = None,
    **settings: Any,
) -> Any:
    """``inject``: findings to add to each successive verification round."""
    rounds = list(inject or [])

    def verify(work: Path) -> Any:
        from dataclasses import replace as _replace

        from rote.compiler.verify import verify_deliverable

        system1.settle(work, plan)
        v = verify_deliverable(work, run_tests=False, portable_judges=True)
        extra = rounds.pop(0) if rounds else []
        return _replace(v, findings=[*v.findings, *extra])

    return await generate.write_text(
        fake,  # type: ignore[arg-type]
        work_dir=ctx.work_dir,
        skill_dir=EXAMPLES / "expense-audit" / "skill",
        compiler_skill_dir=REPO / "skills" / "rote-compile",
        plan=plan,
        grounding=ctx.grounding,
        verify=verify,
        settings=Settings(prime_seconds=0.05, **settings),
    )


# ───────── the answer schema and the cache ─────────


def test_every_call_shares_one_answer_schema_and_one_system_prompt(tmp_path: Path) -> None:
    """The response schema is a tool, and tools lead the cache prefix: one schema
    for every call is what lets calls after the first read the prefix from cache."""
    ctx, _ = _context(tmp_path, "deal-monitor")
    system = "SYSTEM"
    requests = [interface_request(ctx, n, system, "m") for n in generate.interface_targets(ctx)]
    ctx.interface = {n.id: {"output_schema": _schema_for(n)} for n in ctx.pipeline.nodes}
    for n in ctx.pipeline.nodes:
        if n.kind is NodeKind.PURE_FUNCTION:
            requests.append(module_request(ctx, n, system, "m"))
        if n.kind is NodeKind.LLM_JUDGE:
            requests.append(judge_request(ctx, n, system, "m"))
    kinds = {r.label.split(":")[0] for r in requests}
    assert {"interface"} < kinds and len(requests) > 5
    assert all(r.schema is ANSWER_SCHEMA for r in requests)
    assert all(r.system == system for r in requests)


@pytest.mark.asyncio
async def test_first_call_runs_alone_then_the_rest_in_parallel_under_the_bound(
    tmp_path: Path,
) -> None:
    ctx, plan = _context(tmp_path)
    fake = FakeGenerator({n.id: n for n in ctx.pipeline.nodes})
    result = await _write(ctx, plan, fake, max_parallel=3)

    # Nothing else starts until the first call has written the shared prefix to
    # the cache: it finished, or ran alone for prime_seconds.
    first, *rest = fake.started
    assert rest and min(rest) >= min(fake.ended[0], first + 0.05)
    assert fake.max_in_flight == 3  # bounded, and actually parallel
    assert len(fake.requests) == len(generate.interface_targets(ctx)) + len(
        [n for n in ctx.pipeline.nodes if n.kind is NodeKind.PURE_FUNCTION]
    )
    # Every interface call ran before any piece: pieces are written against it.
    labels = [r.label for r in fake.requests]
    last_iface = max(i for i, x in enumerate(labels) if x.startswith("interface:"))
    first_piece = min(i for i, x in enumerate(labels) if not x.startswith("interface:"))
    assert last_iface < first_piece
    meta = result.metadata
    assert meta["num_turns"] == len(fake.requests)
    assert meta["output_tokens"] == 100 * len(fake.requests)
    assert meta["cache_read_tokens"] == 1000 * len(fake.requests)
    assert meta["cost_usd"] == pytest.approx(0.01 * len(fake.requests))


@pytest.mark.asyncio
async def test_merge_writes_modules_tests_and_text_and_the_pipeline_settles(
    tmp_path: Path,
) -> None:
    ctx, plan = _context(tmp_path)
    fake = FakeGenerator({n.id: n for n in ctx.pipeline.nodes})
    await _write(ctx, plan, fake)
    work = ctx.work_dir
    system1.settle(work, plan, final=True)
    pipeline = load_pipeline(work / "pipeline.yaml")
    for node in pipeline.nodes:
        if node.kind is NodeKind.PURE_FUNCTION:
            assert (work / "extracted" / f"{node.id}.py").is_file()
            assert (work / "tests" / f"test_{node.id}.py").is_file()
            assert node.output_schema == _schema_for(node)
            assert node.description == f"Does {node.id}."
    interface = json.loads((work / "system1" / "interface.json").read_text())
    assert set(interface) == {n.id for n in generate.interface_targets(ctx)}


# ───────── code checks on each answer ─────────


def _node(ctx: Context, kind: NodeKind, constants: bool = False) -> Any:
    return next(n for n in ctx.pipeline.nodes if n.kind is kind and (not constants or n.constants))


def test_module_checks_name_each_contract_break(tmp_path: Path) -> None:
    ctx, _ = _context(tmp_path)
    ctx.interface = {n.id: {"output_schema": _schema_for(n)} for n in ctx.pipeline.nodes}
    node = _node(ctx, NodeKind.PURE_FUNCTION, constants=True)
    prompt = module_request(ctx, node, "S", "m").prompt
    module, test = _module_for(prompt, node.id)
    assert module_problems(ctx, node, {"module": module, "test": test}) == []

    wrong_sig = f"def {node.id}(request):\n    return {{}}\n"
    problems = module_problems(ctx, node, {"module": wrong_sig, "test": test})
    assert any("it must take exactly" in p for p in problems)

    third_party = "import requests\n" + module
    problems = module_problems(ctx, node, {"module": third_party, "test": test})
    assert problems == [f"extracted/{node.id}.py imports `requests`: stdlib only"]

    phantom = f"from extracted.{node.id} import {node.id}, LIMIT\n" + test
    problems = module_problems(ctx, node, {"module": module, "test": phantom})
    assert any("imports LIMIT" in p and "does not define" in p for p in problems)

    by_package = test.replace(f"from extracted.{node.id} import", "from extracted import")
    assert module_problems(ctx, node, {"module": module, "test": by_package}) == []

    # Both are 3.12+ only. On 3.11 the parse itself refuses them; on a newer
    # interpreter rote.contracts.newer_syntax_problems does (PEP 701 f-strings
    # are a tokenizer change that feature_version cannot see).
    for newer in ("\ntype Alias = int\n", "\nX = f\"{'\\n'.join(['a'])}\"\n"):
        problems = module_problems(ctx, node, {"module": module + newer, "test": test})
        assert problems and all(f"extracted/{node.id}.py" in p for p in problems), newer


def test_a_judge_interface_must_name_its_decision_fields(tmp_path: Path) -> None:
    ctx, _ = _context(tmp_path, "deal-monitor")
    node = _node(ctx, NodeKind.LLM_JUDGE)
    bare = {"description": "x", "output_schema": {"type": "object"}}
    assert any("names no properties" in p for p in interface_problems(ctx, node, bare))
    named = {"description": "x", "output_schema": _schema_for(node)}
    assert interface_problems(ctx, node, named) == []


def test_judge_prompt_variables_must_exist_in_the_input_schema(tmp_path: Path) -> None:
    ctx, _ = _context(tmp_path, "deal-monitor")
    node = _node(ctx, NodeKind.LLM_JUDGE)
    param = next(iter(node.inputs))
    assert judge_problems(ctx, node, {"prompt": f"Judge {{{{ {param} }}}}."}) == []
    problems = judge_problems(ctx, node, {"prompt": "Judge {{ nothing_here }}."})
    assert problems and "nothing_here" in problems[0]


def test_a_judge_may_pass_through_only_an_input_field_of_the_same_schema(tmp_path: Path) -> None:
    """Code offers the candidates (same schema, by shape); the model picks
    which the skill returns unchanged; code checks the pick and writes it."""
    ctx, _ = _context(tmp_path, "deal-monitor")
    node = _node(ctx, NodeKind.LLM_JUDGE)
    param, schema = next(iter(ctx.judge_input_schema(node)["properties"].items()))
    ctx.interface = {
        node.id: {
            "description": "x",
            "output_schema": {
                "type": "object",
                "properties": {"copied": schema, "verdict": {"enum": ["a", "b"]}},
                "required": ["copied", "verdict"],
            },
        }
    }
    prompt = judge_request(ctx, node, "S", "m").prompt
    assert f"- `copied` <- `{param}`" in prompt
    assert "`verdict` <-" not in prompt  # no input has its schema
    ok = {"prompt": f"Judge {{{{ {param} }}}}.", "passthrough": {"copied": param}}
    assert judge_problems(ctx, node, ok) == []
    wrong_path = {**ok, "passthrough": {"copied": f"{param}.nope"}}
    assert any("is not a candidate" in p for p in judge_problems(ctx, node, wrong_path))
    wrong_field = {**ok, "passthrough": {"verdict": param}}
    assert any("offered for `verdict`: none" in p for p in judge_problems(ctx, node, wrong_field))
    # Naming every output field leaves nothing to decide: code drops it, as
    # a re-ask cannot (an empty passthrough in a repair keeps the old one).
    assert generate.covers_every_output(ctx, node, {"copied": param, "verdict": param})
    assert not generate.covers_every_output(ctx, node, {"copied": param})
    assert not generate.covers_every_output(ctx, node, {})
    answered = generate.judge_passthrough(
        {"passthrough": [{"output_field": "copied", "input_path": param}]}
    )
    assert answered == {"copied": param}
    entries = generate.text_entries(
        ctx, {node.id: {"prompt": "p", "client": "anthropic", "passthrough": answered}}
    )
    assert entries[node.id]["signature_spec"]["passthrough"] == {"copied": param}
    # Negative half: no passthrough writes none.
    plain = generate.text_entries(ctx, {node.id: {"prompt": "p", "client": "anthropic"}})
    assert "passthrough" not in plain[node.id]["signature_spec"]


def test_interface_checks_fan_out_sources_and_consumer_fields(tmp_path: Path) -> None:
    ctx, _ = _context(tmp_path)
    source = next(
        n for n in ctx.pipeline.nodes if isinstance(n.output, dict) and "items" in n.output
    )
    good = {"description": "x", "output_schema": _schema_for(source)}
    assert interface_problems(ctx, source, good) == []
    flat = {"description": "x", "output_schema": {"type": "object", "properties": {}}}
    assert any("`items` array" in p for p in interface_problems(ctx, source, flat))
    listy = {"description": "x", "output_schema": {"type": "array"}}
    assert any("must return an object" in p for p in interface_problems(ctx, source, listy))
    from rote.compiler.system1.generate import _interface_entry

    parsed = _interface_entry(
        {"description": "x", "output_fields": _fields_for(source)}, ctx, source
    )
    assert parsed["output_schema"] == _schema_for(source)
    assert interface_problems(ctx, source, parsed) == []
    orphan = _interface_entry(
        {"description": "x", "output_fields": [*_fields_for(source), _field("meta.n", "integer")]},
        ctx,
        source,
    )
    assert any("needs its parent `meta`" in p for p in interface_problems(ctx, source, orphan))
    assert "description is empty" in interface_problems(
        ctx, source, {"description": " ", "output_schema": _schema_for(source)}
    )


# ───────── applying an answer ─────────


def test_edits_apply_once_and_an_empty_field_keeps_the_file() -> None:
    prior = {"module": "a = 1\nb = 2\n", "test": "assert True\n"}
    answer, problems = apply_answer(
        prior,
        {"module": "", "test": "", "edits": [{"file": "module", "old": "b = 2", "new": "b = 3"}]},
        ("module", "test"),
    )
    assert problems == []
    assert answer == {"module": "a = 1\nb = 3\n", "test": "assert True\n"}

    _, problems = apply_answer(
        {"module": "x\nx\n", "test": "t"},
        {"edits": [{"file": "module", "old": "x", "new": "y"}]},
        ("module", "test"),
    )
    assert problems and "appears 2 times" in problems[0]

    replaced, _ = apply_answer(prior, {"module": "new = 1\n", "edits": []}, ("module", "test"))
    assert replaced["module"] == "new = 1\n" and replaced["test"] == prior["test"]


@pytest.mark.asyncio
async def test_a_failed_check_is_re_asked_alone_with_the_problem(tmp_path: Path) -> None:
    ctx, plan = _context(tmp_path)
    fake = FakeGenerator({n.id: n for n in ctx.pipeline.nodes})
    node = _node(ctx, NodeKind.PURE_FUNCTION)
    fake.scripted[node.id] = [{"module": "def wrong():\n    pass\n", "test": "x = 1\n"}]
    result = await _write(ctx, plan, fake, models=generate.Models("i", "module-m", "j", "repair-m"))
    # Fixed by its own re-ask, before any verification round.
    assert result.metadata["repair_rounds"] == 0
    assert result.metadata["unresolved"] == {}
    asked = [r for r in fake.requests if r.label == node.id]
    assert len(asked) == 2
    assert [r.model for r in asked] == ["module-m", "repair-m"]
    assert f"defines no top-level `def {node.id}`" in asked[1].prompt
    assert "def wrong():" in asked[1].prompt  # the earlier answer, as text, for edits
    module = (ctx.work_dir / "extracted" / f"{node.id}.py").read_text()
    assert f"def {node.id}(" in module


@pytest.mark.asyncio
async def test_a_constant_the_module_ignores_goes_back_to_that_piece(tmp_path: Path) -> None:
    """Dead config is a review gap: the verification round re-asks the node
    that owns it, with the finding, and the fixed module reads the constant."""
    ctx, plan = _context(tmp_path)
    fake = FakeGenerator({n.id: n for n in ctx.pipeline.nodes})
    node = _node(ctx, NodeKind.PURE_FUNCTION, constants=True)
    fake.dead_once.add(node.id)
    result = await _write(ctx, plan, fake)
    assert result.metadata["repair_rounds"] == 1
    asked = [r for r in fake.requests if r.label == node.id]
    assert len(asked) == 2
    key = next(iter(node.constants))
    assert f"takes the constant `{key}`" in asked[1].prompt
    assert "never reads it" in asked[1].prompt
    module = (ctx.work_dir / "extracted" / f"{node.id}.py").read_text()
    assert key in module.split("\n", 1)[1]  # read in the body, not only the def line
    # Only the owner was re-asked.
    others = [n.id for n in ctx.pipeline.nodes if n.kind is NodeKind.PURE_FUNCTION]
    assert all(
        [r.label for r in fake.requests].count(other) == 1 for other in others if other != node.id
    )


@pytest.mark.asyncio
async def test_a_piece_whose_calls_fail_leaves_the_others_written(tmp_path: Path) -> None:
    ctx, plan = _context(tmp_path)
    fake = FakeGenerator({n.id: n for n in ctx.pipeline.nodes})
    node = _node(ctx, NodeKind.PURE_FUNCTION)
    fake.fail.add(node.id)
    result = await _write(ctx, plan, fake)
    # Asked again in each verification round, since the module is still missing.
    tries = generate.CALL_ATTEMPTS * (1 + generate.MAX_REPAIR_ROUNDS)
    assert [r.label for r in fake.requests].count(node.id) == tries
    assert not (ctx.work_dir / "extracted" / f"{node.id}.py").exists()
    others = [n for n in ctx.pipeline.nodes if n.kind is NodeKind.PURE_FUNCTION and n.id != node.id]
    assert all((ctx.work_dir / "extracted" / f"{n.id}.py").is_file() for n in others)
    # The failed calls were paid for and are counted.
    assert result.metadata["cost_usd"] == pytest.approx(
        0.5 * tries + 0.01 * (len(fake.requests) - tries)
    )
    assert result.metadata["repair_rounds"] == generate.MAX_REPAIR_ROUNDS


@pytest.mark.asyncio
async def test_a_call_that_fails_once_is_asked_again(tmp_path: Path) -> None:
    ctx, plan = _context(tmp_path)
    fake = FakeGenerator({n.id: n for n in ctx.pipeline.nodes})
    node = _node(ctx, NodeKind.PURE_FUNCTION)
    fake.fail_once.add(node.id)
    result = await _write(ctx, plan, fake)
    assert [r.label for r in fake.requests].count(node.id) == 2
    assert (ctx.work_dir / "extracted" / f"{node.id}.py").is_file()
    assert result.metadata["repair_rounds"] == 0


@pytest.mark.asyncio
async def test_a_failed_interface_fails_the_compile_with_its_usage(tmp_path: Path) -> None:
    ctx, plan = _context(tmp_path)
    fake = FakeGenerator({n.id: n for n in ctx.pipeline.nodes})
    target = generate.interface_targets(ctx)[0]
    fake.fail.add(f"interface:{target.id}")
    with pytest.raises(DriverError) as err:
        await _write(ctx, plan, fake)
    assert err.value.metadata is not None
    assert err.value.metadata["cost_usd"] >= 0.5 * generate.CALL_ATTEMPTS


# ───────── attribution and usage ─────────


def test_findings_are_attributed_to_the_node_whose_files_they_name() -> None:
    class V:
        pipeline_error = None
        ok = False
        errors = [
            ContractFinding(
                "(provenance)", "bad_citation", "error", "quote not found", "extracted/a.py"
            ),
            ContractFinding("b", "untested_implementation", "error", "no test calls b()", None),
            ContractFinding("(pipeline)", "emit_failed", "error", "cloudflare refused", None),
        ]
        review_gaps: list[ContractFinding] = []

        class tests:  # noqa: N801
            status = "failed"
            summary = "1 failed"
            output = "FAILED tests/test_b.py::test_x - AssertionError"

    per, loose = attribute(V(), ["a", "b"])
    assert per["a"] == ["quote not found (extracted/a.py)"]
    assert per["b"][0] == "no test calls b()"
    assert "tests/test_b.py" in per["b"][1]
    assert loose == ["cloudflare refused"]


def test_ledger_counts_all_four_buckets_and_refuses_a_partial_cost() -> None:
    ledger = Ledger()
    g = Generation({}, "m", 3, 100, 50, 1000, 0.25, 10)
    ledger.add("a", g)
    ledger.add("b", g)
    meta = ledger.metadata(wall_ms=5, repair_rounds=1)
    assert (meta["input_tokens"], meta["output_tokens"]) == (6, 200)
    assert (meta["cache_write_tokens"], meta["cache_read_tokens"]) == (100, 2000)
    assert meta["cost_usd"] == 0.5
    ledger.add("c", Generation({}, "m", 1, 1, 1, 1, None, 1))
    assert ledger.metadata(wall_ms=5, repair_rounds=1)["cost_usd"] is None


# ───────── the compiler picks the path by the driver ─────────


@pytest.mark.asyncio
async def test_compile_with_a_generating_driver_writes_text_in_parallel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    skill = tmp_path / "skill"
    shutil.copytree(EXAMPLES / "expense-audit" / "skill", skill)
    skeleton = tmp_path / "skeleton"
    skeleton.mkdir()
    nodes = {
        n.id: n
        for n in load_pipeline(
            (system1.prepare(skill, skeleton), skeleton / "pipeline.yaml")[1]
        ).nodes
    }
    fake = FakeGenerator(nodes)
    monkeypatch.setattr(Compiler, "select_driver", lambda self: fake)
    monkeypatch.setattr(generate.Settings, "prime_seconds", 0.0)
    events: list[str] = []
    result = await Compiler(
        run_generated_tests=False,
        system1_tiebreak="none",
        system1_structure="rules",
        max_parallel=2,
        on_event=lambda e: events.append(e.message),
    ).compile(skill, tmp_path / "out")
    assert fake.requests and fake.max_in_flight <= 2
    assert result.verification is not None and result.verification.ok
    assert any("parallel calls (at most 2 at once" in m for m in events)
    text = yaml.safe_load((tmp_path / "out" / "system1" / "text.yaml").read_text())
    assert all("output_schema" in v for v in text.values() if "description" in v)


def test_max_parallel_below_one_is_refused() -> None:
    from rote.compiler import CompilerError

    with pytest.raises(CompilerError, match="at least 1"):
        Compiler(max_parallel=0)


def test_cli_passes_max_parallel(monkeypatch: pytest.MonkeyPatch) -> None:
    from rote.cli import _build_parser

    args = _build_parser().parse_args(["compile", "skill", "--out", "o", "--max-parallel", "3"])
    assert args.max_parallel == 3
    default = _build_parser().parse_args(["compile", "skill", "--out", "o"])
    assert default.max_parallel == generate.DEFAULT_MAX_PARALLEL


def test_unverified_tool_results_tell_consumers_to_accept_every_shape(tmp_path: Path) -> None:
    ctx, _ = _context(tmp_path, "ops-report")
    ctx.interface = {n.id: {"output_schema": _schema_for(n)} for n in ctx.pipeline.nodes}

    def sources(node: Any) -> set[str]:
        refs = [r.split(".", 1)[0] for r in node.inputs.values()]
        return {r for r in refs if r != "pipeline"}

    pure = [n for n in ctx.pipeline.nodes if n.kind is NodeKind.PURE_FUNCTION]
    reads_tool = [n for n in pure if any(ctx.node(s).mcp is not None for s in sources(n))]
    reads_code = [
        n for n in pure if sources(n) and all(ctx.node(s).mcp is None for s in sources(n))
    ]
    assert reads_tool and reads_code
    for n in reads_tool:
        assert generate.UNVERIFIED_TOOL_RESULT in module_request(ctx, n, "S", "m").prompt
    for n in reads_code:
        assert generate.UNVERIFIED_TOOL_RESULT not in module_request(ctx, n, "S", "m").prompt


@pytest.mark.asyncio
async def test_a_backend_api_compile_keeps_the_researching_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--backend api modules are researched vendor calls: the parallel calls have
    no web tools, so the session writes them."""
    from rote.compiler import CompilerError
    from rote.compiler.verify import EmitTarget

    class Session(FakeGenerator):
        ran = False

        async def run(self, *args: Any, **kwargs: Any) -> Any:
            Session.ran = True
            raise DriverError("the session ran")

    skill = tmp_path / "skill"
    shutil.copytree(EXAMPLES / "expense-audit" / "skill", skill)
    fake = Session({})
    monkeypatch.setattr(Compiler, "select_driver", lambda self: fake)
    with pytest.raises(CompilerError, match="the session ran"):
        await Compiler(
            run_generated_tests=False,
            system1_tiebreak="none",
            system1_structure="rules",
            emit_target=EmitTarget("dbos", "api"),
        ).compile(skill, tmp_path / "out")
    assert Session.ran and fake.requests == []


# ───────── review findings: never ship a failing piece, cancel on failure ─────────


def _bad(node: Any, ctx: Context) -> dict[str, Any]:
    """An answer with the right signature that imports a third-party package."""
    saved = ctx.interface
    ctx.interface = {n.id: {"output_schema": _schema_for(n)} for n in ctx.pipeline.nodes}
    module, test = _module_for(module_request(ctx, node, "S", "m").prompt, node.id)
    ctx.interface = saved
    return {"module": "import yaml\n" + module, "test": test}


@pytest.mark.asyncio
async def test_a_piece_that_never_passes_its_checks_is_an_error_not_a_pass(
    tmp_path: Path,
) -> None:
    ctx, plan = _context(tmp_path)
    fake = FakeGenerator({n.id: n for n in ctx.pipeline.nodes})
    node = _node(ctx, NodeKind.PURE_FUNCTION)
    total = (generate.MAX_CHECK_RETRIES + 1) * (1 + generate.MAX_REPAIR_ROUNDS)
    fake.scripted[node.id] = [_bad(node, ctx) for _ in range(total)]
    result = await _write(ctx, plan, fake)
    assert [r.label for r in fake.requests].count(node.id) == total
    assert result.metadata["unresolved"][node.id] == [
        f"extracted/{node.id}.py imports `yaml`: stdlib only"
    ]
    # The files are kept for review, and the verdict says they fail.
    assert "import yaml" in (ctx.work_dir / "extracted" / f"{node.id}.py").read_text()
    (finding,) = generate.unresolved_findings(result.metadata["unresolved"])
    assert finding.kind == "generation_check_failed" and finding.severity == "error"


@pytest.mark.asyncio
async def test_a_repair_that_fails_its_checks_never_replaces_passing_files(
    tmp_path: Path,
) -> None:
    ctx, plan = _context(tmp_path)
    fake = FakeGenerator({n.id: n for n in ctx.pipeline.nodes})
    node, other = [n for n in ctx.pipeline.nodes if n.kind is NodeKind.PURE_FUNCTION][:2]
    finding = ContractFinding(node.id, "prefix_run_failed", "error", "raised KeyError", None)
    attempts = generate.MAX_CHECK_RETRIES + 1
    plain = fake.generate

    async def generate_call(request: GenerationRequest) -> Generation:
        if request.label == node.id and "This is a repair" in request.prompt:
            fake.scripted[node.id] = [_bad(node, ctx)]
        return await plain(request)

    fake.generate = generate_call  # type: ignore[method-assign]
    await _write(
        ctx,
        plan,
        fake,
        inject=[[finding]],
        models=generate.Models("iface-m", "module-m", "judge-m", "repair-m"),
    )
    module = (ctx.work_dir / "extracted" / f"{node.id}.py").read_text()
    assert "import yaml" not in module  # the earlier, passing files stayed
    asked = [r for r in fake.requests if r.label == node.id]
    assert len(asked) == 1 + attempts
    # Only the named node is repaired, with the repair model, and every re-ask
    # still carries the verification finding it is for.
    assert [r.label for r in fake.requests].count(other.id) == 1
    assert asked[0].model == "module-m" and all(r.model == "repair-m" for r in asked[1:])
    assert all("raised KeyError" in r.prompt for r in asked[1:])


@pytest.mark.asyncio
async def test_a_failed_interface_cancels_its_siblings_and_counts_them(tmp_path: Path) -> None:
    ctx, plan = _context(tmp_path)
    fake = FakeGenerator({n.id: n for n in ctx.pipeline.nodes})
    _, failing, *slow = generate.interface_targets(ctx)
    fake.fail.add(f"interface:{failing.id}")
    for n in slow:
        fake.delay[f"interface:{n.id}"] = 5.0  # still running when the failure lands
    with pytest.raises(DriverError) as err:
        await asyncio.wait_for(_write(ctx, plan, fake), timeout=3)
    assert fake.in_flight == 0  # cancelled and awaited, none left running
    meta = err.value.metadata or {}
    assert meta["usage_complete"] is False
    assert f"{len(slow)} call(s) were cancelled" in meta["usage_note"]


@pytest.mark.asyncio
async def test_an_interface_entry_that_never_passes_fails_the_compile(tmp_path: Path) -> None:
    ctx, plan = _context(tmp_path)
    fake = FakeGenerator({n.id: n for n in ctx.pipeline.nodes})
    target = generate.interface_targets(ctx)[0]
    empty = {"description": "", "output_fields": _fields_for(target)}
    fake.scripted[f"interface:{target.id}"] = [empty] * (generate.MAX_CHECK_RETRIES + 1)
    with pytest.raises(generate.GenerationFailed, match="description is empty"):
        await _write(ctx, plan, fake)


def test_a_consumer_field_the_producer_does_not_return_is_an_interface_problem(
    tmp_path: Path,
) -> None:
    ctx, _ = _context(tmp_path)
    source = next(
        n for n in ctx.pipeline.nodes if isinstance(n.output, dict) and "items" in n.output
    )
    no_items = {
        "description": "x",
        "output_schema": {"type": "object", "properties": {"rows": {"type": "array"}}},
    }
    problems = interface_problems(ctx, source, no_items)
    assert any(f"reads `{source.id}.output.items`" in p for p in problems)


def test_a_piece_is_held_to_its_own_tests_and_citations(tmp_path: Path) -> None:
    """run_tests=True: the piece's tests run alone, as verification would run them."""
    ctx, _ = _context(tmp_path)
    ctx.interface = {n.id: {"output_schema": _schema_for(n)} for n in ctx.pipeline.nodes}
    node = _node(ctx, NodeKind.PURE_FUNCTION)
    module, test = _module_for(module_request(ctx, node, "S", "m").prompt, node.id)
    skill = EXAMPLES / "expense-audit" / "skill"

    def check(answer: dict[str, Any]) -> list[str]:
        return generate.piece_findings(
            ctx, node, answer, skill_dir=skill, run_tests=True, trust_skill=True
        )

    assert check({"module": module, "test": test}) == []
    red = test + "\n\ndef test_red():\n    assert False\n"
    assert any("its tests fail" in p for p in check({"module": module, "test": red}))
    cited = module.replace(
        "    return", '    # skill: "this sentence is nowhere in the skill text"\n    return'
    )
    assert any("contains no such text" in p for p in check({"module": cited, "test": test}))


def test_only_named_test_files_are_attributed() -> None:
    class V:
        pipeline_error = None
        ok = False
        errors: list[ContractFinding] = []
        review_gaps: list[ContractFinding] = []

        class tests:  # noqa: N801
            status = "failed"
            summary = "1 failed"
            output = "FAILED tests/test_rule_tables.py::test_x - AssertionError"

    per, loose = attribute(V(), ["a"])
    assert per == {} and loose == ["generated tests fail: 1 failed"]


@pytest.mark.asyncio
async def test_the_compile_verdict_fails_on_a_piece_that_never_passed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    skill = tmp_path / "skill"
    shutil.copytree(EXAMPLES / "expense-audit" / "skill", skill)
    skeleton = tmp_path / "skeleton"
    skeleton.mkdir()
    plan = system1.prepare(skill, skeleton)
    ctx = Context.load(skeleton, plan, Grounding())
    fake = FakeGenerator({n.id: n for n in ctx.pipeline.nodes})
    node = _node(ctx, NodeKind.PURE_FUNCTION)
    fake.scripted[node.id] = [_bad(node, ctx) for _ in range(50)]
    monkeypatch.setattr(Compiler, "select_driver", lambda self: fake)
    monkeypatch.setattr(generate.Settings, "prime_seconds", 0.0)
    result = await Compiler(
        run_generated_tests=False, system1_tiebreak="none", system1_structure="rules"
    ).compile(skill, tmp_path / "out")
    assert result.verification is not None and not result.verification.ok
    kinds = {(f.node_id, f.kind) for f in result.verification.errors}
    assert (node.id, "generation_check_failed") in kinds


@pytest.mark.asyncio
async def test_typed_judge_input_schemas_survive_settle(tmp_path: Path) -> None:
    """enforce keeps a written input_schema only when its names are the node's
    inputs keys: the producer-typed schemas the generator writes must pass."""
    ctx, plan = _context(tmp_path, "deal-monitor")
    fake = FakeGenerator({n.id: n for n in ctx.pipeline.nodes})
    await _write(ctx, plan, fake)
    system1.settle(ctx.work_dir, plan, final=True)
    pipeline = load_pipeline(ctx.work_dir / "pipeline.yaml")
    judges = [n for n in pipeline.nodes if n.kind is NodeKind.LLM_JUDGE]
    assert judges
    written = yaml.safe_load((ctx.work_dir / "system1" / "text.yaml").read_text())
    typed = 0
    for node in judges:
        assert node.signature_spec is not None
        spec = written[node.id]["signature_spec"]
        assert node.signature_spec.input_schema == spec["input_schema"]
        assert node.signature_spec.output_schema == _schema_for(node)
        typed += any(
            v.get("properties") or (v.get("items") or {}).get("properties")
            for v in spec["input_schema"]["properties"].values()
        )
    assert typed  # at least one judge reads a producer typed by the interface
    record = json.loads((ctx.work_dir / "system1" / "enforcement.json").read_text())
    assert not [c for changes in record for c in changes if "input_schema" in c]


def test_a_fanned_producers_whole_output_is_a_list_to_its_consumer(tmp_path: Path) -> None:
    """A fan_out node runs once per item; whoever reads its whole output gets the
    list of runs, so that is the type the consumer is written (and decided) against."""
    ctx, _ = _context(tmp_path)
    ctx.interface = {n.id: {"output_schema": _schema_for(n)} for n in ctx.pipeline.nodes}
    fanned = next(n for n in ctx.pipeline.nodes if n.fan_out)
    consumer, param = ctx.consumers(fanned.id)[0]
    node = ctx.node(consumer)
    assert ctx.param_schema(node, param) == {"type": "array", "items": _schema_for(fanned)}
    assert f"`{param}: list" in def_line(ctx, node).replace("(", "`").replace(", ", " `")
    assert "this is every run's value" in module_request(ctx, node, "S", "m").prompt


# ───────── the return value is a field list, built into a schema by code ─────────


def test_the_answer_schema_asks_for_fields_never_a_schema_string() -> None:
    """The JSON-text output_schema failed live: Sonnet 4.6 closed a nested schema
    with one brace too many, through every re-ask. Nothing asks for schema text."""
    props = ANSWER_SCHEMA["properties"]
    assert "output_schema" not in props and "output_schema" not in ANSWER_SCHEMA["required"]
    assert props["output_fields"]["items"] is generate.FIELD_SCHEMA

    # Strict structured outputs lock every object: each one closes its properties.
    def objects(node: Any) -> list[dict[str, Any]]:
        found = []
        if isinstance(node, dict):
            if node.get("type") == "object":
                found.append(node)
            for v in node.values():
                found += objects(v)
        elif isinstance(node, list):
            for v in node:
                found += objects(v)
        return found

    assert all(o.get("additionalProperties") is False for o in objects(ANSWER_SCHEMA))
    from rote.inference import structured_output_schema

    assert structured_output_schema(ANSWER_SCHEMA, node_id="x") == ANSWER_SCHEMA


def test_a_schema_string_answer_is_refused(tmp_path: Path) -> None:
    from rote.compiler.system1.generate import _interface_entry

    ctx, _ = _context(tmp_path)
    source = generate.interface_targets(ctx)[0]
    old = _interface_entry(
        {"description": "x", "output_schema": '{"type": "object"}}'}, ctx, source
    )
    assert "output_fields is empty" in interface_problems(ctx, source, old)


def test_fields_build_nested_objects_arrays_of_objects_and_enums() -> None:
    schema, problems = generate.fields_to_schema(
        [
            _field("findings[].severity", "string", enum=["block", "review"]),
            _field("findings", "array", "object", description="one row per finding"),
            _field("findings[].expense_id", "string"),
            _field("totals", "object"),
            _field("totals.usd", "number"),
            _field("totals.by_currency", "array", "string", required=False),
            _field("note", "string", required=False),
        ]
    )
    assert problems == []
    assert schema == {
        "type": "object",
        "properties": {
            "findings": {
                "type": "array",
                "description": "one row per finding",
                "items": {
                    "type": "object",
                    "properties": {
                        "severity": {"type": "string", "enum": ["block", "review"]},
                        "expense_id": {"type": "string"},
                    },
                    "required": ["severity", "expense_id"],
                },
            },
            "totals": {
                "type": "object",
                "properties": {
                    "usd": {"type": "number"},
                    "by_currency": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["usd"],
            },
            "note": {"type": "string"},
        },
        "required": ["findings", "totals"],
    }


def test_fields_describe_a_value_that_is_not_an_object() -> None:
    # An MCP call returning the list itself, or its text.
    rows, problems = generate.fields_to_schema(
        [_field("", "array", "object"), _field("[].id", "string")]
    )
    assert problems == []
    assert rows == {
        "type": "array",
        "items": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]},
    }
    text, _ = generate.fields_to_schema([_field("", "string")])
    assert text == {"type": "string"}


def test_a_faulty_field_list_names_each_fault() -> None:
    _, problems = generate.fields_to_schema(
        [
            _field("a", "string"),
            _field("a", "number"),
            _field("b", "string", "object"),
            _field("c", "integer", enum=["1"]),
            _field("d.x", "string"),
            _field("a.y", "string"),
            _field("e[0]", "string"),
        ]
    )
    joined = " | ".join(problems)
    assert "`a` is given twice" in joined
    assert "`b` sets items but has type string" in joined
    assert "`c` lists enum values but has type integer" in joined
    assert "`d.x` needs its parent `d`" in joined
    assert "`a.y` needs its parent `a`" in joined
    assert "`e[0]`: a name may not contain" in joined


def test_nullable_fields_accept_null_nested_and_as_array_items() -> None:
    """Code that sets a not-found value to None is correct; its schema must say so,
    or the prefix's check on the sample turns it into a repair."""
    from rote.compiler.grounding import schema_mismatches

    schema, problems = generate.fields_to_schema(
        [
            _field("owner", "object", nullable=True),
            _field("owner.email", "string", nullable=True),
            _field("dates", "array", "string"),
            _field("rows", "array", "object"),
            _field("rows[].closed_at", "string", required=False, nullable=True),
        ]
    )
    assert problems == []
    assert schema["properties"]["owner"]["type"] == ["object", "null"]
    assert schema["properties"]["owner"]["properties"]["email"]["type"] == ["string", "null"]
    item = schema["properties"]["rows"]["items"]["properties"]["closed_at"]
    assert item["type"] == ["string", "null"]
    value = {
        "owner": {"email": None},
        "dates": [],
        "rows": [{"closed_at": None}, {"closed_at": "2026-10-01"}],
    }
    assert schema_mismatches(schema, value) == []
    assert schema_mismatches(schema, {**value, "owner": None}) == []
    # Not nullable: null is still a mismatch the prefix reports.
    assert schema_mismatches(schema, {**value, "dates": None})


def test_a_judges_output_refuses_nullable(tmp_path: Path) -> None:
    from rote.compiler.system1.generate import _interface_entry

    ctx, _ = _context(tmp_path, "deal-monitor")
    judge = _node(ctx, NodeKind.LLM_JUDGE)
    entry = _interface_entry(
        {"description": "x", "output_fields": [_field("tier", "string", nullable=True)]},
        ctx,
        judge,
    )
    assert any("judge's output cannot be" in p for p in interface_problems(ctx, judge, entry))
    pure = _node(ctx, NodeKind.PURE_FUNCTION)
    entry = _interface_entry(
        {"description": "x", "output_fields": [_field("total", "number", nullable=True)]},
        ctx,
        pure,
    )
    assert entry["output_schema"]["properties"]["total"]["type"] == ["number", "null"]


def test_a_file_path_line_in_module_text_is_named() -> None:
    from rote.compiler.system1.generate import _parse

    problems: list[str] = []
    assert _parse("extracted/draft_brief.py\n\ndef f():\n    return 1\n", "m.py", problems) is None
    assert problems == [
        "m.py line 1 is the bare expression `extracted/draft_brief.py`, which fails on "
        "import. The text must be only Python code: no file path or heading line"
    ]
    ok: list[str] = []
    assert _parse('"""Doc."""\n\nmain()\n', "m.py", ok) is not None and ok == []


def test_interface_keeps_the_output_keys_the_skeleton_names(tmp_path: Path) -> None:
    """The MCP demo lost `missing_limits_page`: no node's result was required to have it."""
    ctx, _ = _context(tmp_path)
    node = _node(ctx, NodeKind.PURE_FUNCTION).model_copy(
        update={"output": {"products": "Any", "missing_limits_page": "Any"}}
    )
    only_products = {
        "description": "x",
        "output_schema": {"type": "object", "properties": {"products": {"type": "array"}}},
    }
    assert interface_problems(ctx, node, only_products) == [
        "output_schema lacks the keys the skeleton names for this node: `missing_limits_page`"
    ]
    both = {
        "description": "x",
        "output_schema": {
            "type": "object",
            "properties": {"products": {"type": "array"}, "missing_limits_page": {"type": "array"}},
        },
    }
    assert interface_problems(ctx, node, both) == []


@pytest.mark.asyncio
async def test_repair_sends_a_binding_finding_to_the_structure_pass(tmp_path: Path) -> None:
    """No text piece can change an MCP binding; the repair loop re-decides it instead."""
    from rote.compiler.system1.generate import Settings, TextWriter
    from rote.compiler.verify import Verification
    from rote.contracts import ContractFinding

    ctx, _ = _context(tmp_path, "ops-report")
    mcp_node = next(n.id for n in ctx.pipeline.nodes if n.mcp is not None)
    bad = ContractFinding(
        node_id=mcp_node,
        kind="tool_arguments_invalid",
        severity="error",
        message="missing required argument(s): query",
    )
    calls: list[dict[str, list[str]]] = []
    verdicts = iter([Verification(pipeline=None, findings=[bad]), Verification(pipeline=None)])

    def redecide(findings: dict[str, list[str]]) -> list[str]:
        calls.append(findings)
        return [f"`{mcp_node}`: re-bound to gmail.search_threads with args {{'query': 'q'}}"]

    writer = TextWriter(None, ctx, "", Settings(), None, redecide=redecide)  # type: ignore[arg-type]
    verification, rounds = await writer.repair(lambda _: next(verdicts))
    assert calls == [{mcp_node: ["missing required argument(s): query"]}]
    assert verification.ok and rounds == 1


def test_a_guessed_mcp_result_reaches_its_reader_untyped(tmp_path: Path) -> None:
    """A judge typed to the interface's guess failed every live run on the real shape."""
    ctx, _ = _context(tmp_path, "ops-report")
    mcp = next(n for n in ctx.pipeline.nodes if n.mcp is not None)
    reader, param = next(
        (n, p)
        for n in ctx.pipeline.nodes
        for p, ref in (n.inputs or {}).items()
        if ref == f"{mcp.id}.output"
    )
    ctx.interface[mcp.id] = {
        "description": "x",
        "output_schema": {"type": "object", "properties": {"threads": {"type": "array"}}},
    }
    assert ctx.unverified_tool_result(mcp.id)
    assert ctx.param_schema(reader, param) == {}
    assert ctx.judge_input_schema(reader)["properties"][param] == {}
    # A non-MCP producer's interface still types its reader.
    plain = next(
        (n, p)
        for n in ctx.pipeline.nodes
        for p, ref in (n.inputs or {}).items()
        if (src := ref.split(".", 1)[0]) != "pipeline" and ctx.node(src).mcp is None
    )
    ctx.interface[plain[0].inputs[plain[1]].split(".", 1)[0]] = {
        "description": "x",
        "output_schema": {"type": "object", "properties": {"n": {"type": "integer"}}},
    }
    assert ctx.param_schema(*plain) != {}


def test_a_skeleton_output_schema_is_fixed_and_named_keys_are_required(tmp_path: Path) -> None:
    from rote.compiler.system1.generate import interface_request

    ctx, _ = _context(tmp_path, "ops-report")
    base = _node(ctx, NodeKind.PURE_FUNCTION)
    fixed = {"type": "object", "properties": {"items": {"type": "array"}}, "required": ["items"]}
    node = base.model_copy(update={"output_schema": fixed})
    assert ctx.known_result_schema(node) == (fixed, "fixed by the skeleton")
    keyed = base.model_copy(update={"output": {"missing_limits_page": "Any"}})
    prompt = interface_request(ctx, keyed, "system", "m").prompt
    assert (
        "Its output MUST have these keys, named by the skeleton: `missing_limits_page`." in prompt
    )
    plain = interface_request(ctx, base, "system", "m").prompt
    assert "named by the skeleton" not in plain


def test_a_known_mcp_result_keeps_its_reader_typed(tmp_path: Path) -> None:
    """The guard's other half: a declared or fixed shape still types the reader."""
    from dataclasses import replace

    from rote.compiler.tool_grounding import ToolSchema

    ctx, plan = _context(tmp_path, "ops-report")
    mcp = next(n for n in ctx.pipeline.nodes if n.mcp is not None)
    reader, param = next(
        (n, p)
        for n in ctx.pipeline.nodes
        for p, ref in (n.inputs or {}).items()
        if ref == f"{mcp.id}.output"
    )
    assert mcp.mcp is not None
    listed = {
        "type": "array",
        "items": {"type": "object", "properties": {"id": {"type": "string"}}},
    }
    declared = ToolSchema(
        mcp.mcp.server, mcp.mcp.tool, {"type": "object"}, "tools/list", True, listed
    )
    ctx.grounding = replace(ctx.grounding, tools={(mcp.mcp.server, mcp.mcp.tool): declared})
    assert not ctx.unverified_tool_result(mcp.id)
    assert ctx.param_schema(reader, param) == listed
    assert ctx.judge_input_schema(reader)["properties"][param] == listed
    # A shape the skeleton fixed types the reader too, with nothing declared.
    ctx.grounding = replace(ctx.grounding, tools={})
    fixed = {"type": "object", "properties": {"items": {"type": "array"}}}
    ctx.pipeline = ctx.pipeline.model_copy(
        update={
            "nodes": [
                n.model_copy(update={"output_schema": fixed}) if n.id == mcp.id else n
                for n in ctx.pipeline.nodes
            ]
        }
    )
    assert not ctx.unverified_tool_result(mcp.id)
    assert ctx.param_schema(reader, param) == fixed


@pytest.mark.asyncio
async def test_the_parallel_path_builds_none_of_the_sessions_briefs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pieces read grounding through generate.Context and are repaired by
    generate's own loop: the session's briefs, context files and repairer were
    built and thrown away, which hid what actually runs."""
    import rote.compiler as compiler_module

    def never(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the parallel path built a session-only piece")

    monkeypatch.setattr(compiler_module, "contract_brief", never)
    monkeypatch.setattr(compiler_module, "materialize", never)
    monkeypatch.setattr(compiler_module.tool_grounding, "brief", never)
    monkeypatch.setattr(Compiler, "_deliverable_repairer", never)
    monkeypatch.setattr(Compiler, "_materialize_probe_context", never)
    skill = tmp_path / "skill"
    shutil.copytree(EXAMPLES / "expense-audit" / "skill", skill)
    skeleton = tmp_path / "skeleton"
    skeleton.mkdir()
    nodes = {
        n.id: n
        for n in load_pipeline(
            (system1.prepare(skill, skeleton), skeleton / "pipeline.yaml")[1]
        ).nodes
    }
    fake = FakeGenerator(nodes)
    monkeypatch.setattr(Compiler, "select_driver", lambda self: fake)
    monkeypatch.setattr(generate.Settings, "prime_seconds", 0.0)
    result = await Compiler(
        run_generated_tests=False, system1_tiebreak="none", system1_structure="rules"
    ).compile(skill, tmp_path / "out", probe_dir=tmp_path)
    assert fake.requests and result.verification is not None and result.verification.ok


def test_a_failed_call_is_recorded_with_its_model_for_pricing() -> None:
    from rote.compiler.drivers import DriverError

    ledger = generate.Ledger()
    ledger.add_failure("interface:x", DriverError("boom", metadata={"input_tokens": 9}), "m-1")
    meta = ledger.metadata(wall_ms=1, repair_rounds=0)
    assert meta["failed_calls"] == [{"label": "interface:x", "model": "m-1", "input_tokens": 9}]
    assert meta["input_tokens"] == 9


@pytest.mark.asyncio
async def test_a_system1_compile_reports_system1s_spend_in_its_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI adds it to the compile's cost; the structure entry is refreshed first."""
    refreshed: list[Path] = []
    monkeypatch.setattr(system1, "record_structure", lambda work, p: refreshed.append(work))
    monkeypatch.setattr(system1, "calls_cost_usd", lambda d: 0.123)
    skill = tmp_path / "skill"
    shutil.copytree(EXAMPLES / "expense-audit" / "skill", skill)
    skeleton = tmp_path / "skeleton"
    skeleton.mkdir()
    nodes = {
        n.id: n
        for n in load_pipeline(
            (system1.prepare(skill, skeleton), skeleton / "pipeline.yaml")[1]
        ).nodes
    }
    monkeypatch.setattr(Compiler, "select_driver", lambda self: FakeGenerator(nodes))
    monkeypatch.setattr(generate.Settings, "prime_seconds", 0.0)
    result = await Compiler(
        run_generated_tests=False, system1_tiebreak="none", system1_structure="rules"
    ).compile(skill, tmp_path / "out")
    assert result.driver_metadata["system1_cost_usd"] == 0.123
    assert len(refreshed) == 1


# ───────── findings no piece names: route by node, or cost no call ─────────


def _ops_writer(tmp_path: Path) -> tuple[Any, Any, FakeGenerator]:
    from rote.compiler.system1.generate import Settings, TextWriter

    ctx, _ = _context(tmp_path, "ops-report")
    fake = FakeGenerator({n.id: n for n in ctx.pipeline.nodes})
    ctx.interface = {n.id: {"description": "d"} for n in generate.interface_targets(ctx)}
    return ctx, TextWriter(fake, ctx, "", Settings(prime_seconds=0.0), None), fake


def _external(ctx: Any) -> Any:
    return next(n for n in ctx.pipeline.nodes if n.kind is NodeKind.EXTERNAL_CALL)


def test_a_validation_error_goes_to_the_node_it_names_by_place_or_id(tmp_path: Path) -> None:
    from rote.compiler.verify import Verification

    ctx, _, _ = _ops_writer(tmp_path)
    ids = generate._yaml_node_ids(ctx.work_dir / "pipeline.yaml")
    ext, pure = _external(ctx), _node(ctx, NodeKind.PURE_FUNCTION)
    writable = [n.id for n in ctx.pipeline.nodes if n.kind is NodeKind.PURE_FUNCTION]
    entries = [n.id for n in generate.interface_targets(ctx)]
    by_place = Verification(
        pipeline=None,
        pipeline_error=f"1 validation error for Pipeline\nnodes.{ids.index(ext.id)}.description",
    )
    per, loose = attribute(by_place, writable, entries, ids)
    assert list(per) == [ext.id] and loose == []
    by_id = Verification(pipeline=None, pipeline_error=f"node {pure.id!r}: output is not JSON")
    per, loose = attribute(by_id, writable, entries, ids)
    assert list(per) == [pure.id] and loose == []
    nobody = Verification(pipeline=None, pipeline_error="edges: a cycle")
    per, loose = attribute(nobody, writable, entries, ids)
    assert per == {} and loose == ["pipeline.yaml: edges: a cycle"]


def test_a_finding_on_rotes_own_file_is_never_sent_to_a_model(tmp_path: Path) -> None:
    """An external_call stub and rule_tables.py are rote's code: an interface
    entry cannot fix them, even though the stub's file carries the node's id."""
    from rote.compiler.verify import Verification

    ctx, _, _ = _ops_writer(tmp_path)
    ext = _external(ctx)
    entries = [n.id for n in generate.interface_targets(ctx)]
    findings = [
        ContractFinding("(provenance)", "uncited_outcome", "error", "x", f"extracted/{ext.id}.py"),
        ContractFinding(
            "(provenance)", "uncited_outcome", "error", "y", "extracted/rule_tables.py"
        ),
    ]
    per, loose = attribute(Verification(pipeline=None, findings=findings), [], entries, [])
    assert per == {} and len(loose) == 2
    own = [ContractFinding(ext.id, "incomplete_signature_spec", "error", "no description", None)]
    per, _ = attribute(Verification(pipeline=None, findings=own), [], entries, [])
    assert per == {ext.id: ["no description"]}


@pytest.mark.asyncio
async def test_the_repair_loop_re_asks_an_interface_entry_a_validation_error_names(
    tmp_path: Path,
) -> None:
    from rote.compiler.verify import Verification

    ctx, writer, fake = _ops_writer(tmp_path)
    ext = _external(ctx)
    ids = generate._yaml_node_ids(ctx.work_dir / "pipeline.yaml")
    error = f"1 validation error for Pipeline\nnodes.{ids.index(ext.id)}.output_schema: bad"
    verdicts = iter(
        [Verification(pipeline=None, pipeline_error=error), Verification(pipeline=None)]
    )
    verification, rounds = await writer.repair(lambda _: next(verdicts))
    assert verification.ok and rounds == 1
    [request] = fake.requests
    assert request.label == f"interface:{ext.id}" and error in request.prompt
    assert json.loads((ctx.work_dir / "system1" / "interface.json").read_text())[ext.id]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "verification_kwargs",
    [
        # pytest itself died: no test file is named.
        {"tests": "no-file"},
        # rote's own stub file.
        {"stub": True},
    ],
)
async def test_an_unrepairable_finding_costs_no_call(
    verification_kwargs: dict[str, Any], tmp_path: Path
) -> None:
    from rote.compiler.verify import Verification

    ctx, writer, fake = _ops_writer(tmp_path)

    class Tests:
        status = "failed"
        summary = "1 error"
        output = "INTERNALERROR> sandbox refused the interpreter"

    if "tests" in verification_kwargs:
        verdict = Verification(pipeline=None, tests=Tests())  # type: ignore[arg-type]
    else:
        stub = ContractFinding(
            "(provenance)", "uncited_outcome", "error", "x", f"extracted/{_external(ctx).id}.py"
        )
        verdict = Verification(pipeline=None, findings=[stub])
    verification, rounds = await writer.repair(lambda _: verdict)
    assert not verification.ok and rounds == 0 and fake.requests == []


def _validation(error: str) -> Any:
    from rote.compiler.verify import Verification

    return Verification(pipeline=None, pipeline_error=error)


def test_a_node_only_read_in_a_validation_error_gets_no_repair() -> None:
    """review6's real shape: `fetch` is referenced, `summarize` is at fault."""
    error = (
        "1 validation error for Pipeline\n  Value error, node summarize reads "
        "fetch.output.rows [type=value_error, input_value={'nodes': [...]}, "
        "input_type=dict]"
    )
    per, _ = attribute(_validation(error), ["summarize", "fetch"], [], [])
    assert list(per) == ["summarize"]


def test_pydantics_type_tail_names_no_node() -> None:
    """A node id `type` must not match `[type=value_error, ...]`."""
    error = (
        "1 validation error for Pipeline\n  Value error, summarize: bad output "
        "[type=value_error, input_value={'nodes': [{'id': 'type'}]}, input_type=dict]"
    )
    per, _ = attribute(_validation(error), ["summarize", "type"], [], [])
    assert list(per) == ["summarize"]


def test_a_node_feeding_a_tool_sees_the_argument_values_a_recorded_run_sent(
    tmp_path: Path,
) -> None:
    """Atlassian's search returns a page's resource id; getConfluencePage
    takes the bare number. Only the recorded call shows the form, so the
    node producing the argument is shown it."""
    from dataclasses import replace as _replace

    from rote.eval.baseline import ObservedToolCall
    from rote.ir import MCPBinding

    ctx, _ = _context(tmp_path, "deal-monitor")
    node, reader = next((n, c) for n in ctx.pipeline.nodes for c, _ in ctx.consumers(n.id))
    ctx.node(reader).mcp = MCPBinding(server="atlassian", tool="getConfluencePage")
    mcp = ctx.node(reader).mcp
    calls = (
        ObservedToolCall(mcp.server, mcp.tool, {"pageId": "5270208513", "n": 1}),
        ObservedToolCall(mcp.server, mcp.tool, {"pageId": "5208211471", "n": 1}),
        ObservedToolCall(mcp.server, mcp.tool, {"pageId": "failed"}, is_error=True),
        ObservedToolCall(mcp.server, "some_other_tool", {"pageId": "elsewhere"}),
    )
    ctx.grounding = _replace(ctx.grounding, observations=calls)
    card = "\n".join(generate._node_card(ctx, node))
    assert f"`{reader}` calls `{mcp.server}.{mcp.tool}`. A recorded run" in card
    assert '- `pageId`: "5270208513", "5208211471"' in card
    assert "- `n`: 1" in card  # each distinct value once
    assert "failed" not in card and "elsewhere" not in card
    # With no recording, the card says nothing about it.
    ctx.grounding = _replace(ctx.grounding, observations=())
    assert "A recorded run" not in "\n".join(generate._node_card(ctx, node))
