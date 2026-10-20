"""Parallel per-node generation of System-1's text (rote.compiler.system1.generate).

No inference: every model call goes to a fake generator that answers from
the request, so the orchestration (interface, pieces, code checks, repairs,
the shared answer schema, cache priming, concurrency) is what is tested.
"""

from __future__ import annotations

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


def _schema_for(node: Any) -> dict[str, Any]:
    if isinstance(node.output, dict) and "items" in node.output:
        return {
            "type": "object",
            "properties": {"items": {"type": "array", "items": {"type": "object"}}},
            "required": ["items"],
        }
    return {"type": "object", "properties": {"total": {"type": "number"}}, "required": ["total"]}


def _blank() -> dict[str, Any]:
    return {
        "description": "",
        "output_schema": "",
        "termination": {"condition": "", "max_iterations": 0},
        "module": "",
        "test": "",
        "prompt": "",
        "client": "anthropic",
        "edits": [],
    }


def _module_for(prompt: str, node_id: str) -> tuple[str, str]:
    """A module that uses the request's def line verbatim, and a test of it."""
    line = next(ln for ln in prompt.splitlines() if ln.startswith(f"def {node_id}("))
    module = f"{line}\n    return {{'total': 1.0, 'items': []}}\n"
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
                answer["output_schema"] = json.dumps(_schema_for(node))
                if node.kind is NodeKind.AGENT_LOOP and node.termination is None:
                    answer["termination"] = {"condition": "all done", "max_iterations": 3}
            else:
                node = self.nodes[request.label]
                if node.kind is NodeKind.LLM_JUDGE:
                    answer["prompt"] = (
                        "Decide. Return your decision via the structured output tool."
                    )
                else:
                    answer["module"], answer["test"] = _module_for(request.prompt, node.id)
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

    broken = _interface_entry({"description": "x", "output_schema": "{not json"}, ctx, source)
    assert any("not valid JSON text" in p for p in interface_problems(ctx, source, broken))
    parsed = _interface_entry(
        {"description": "x", "output_schema": json.dumps(_schema_for(source))}, ctx, source
    )
    assert parsed["output_schema"] == _schema_for(source)
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
    empty = {"description": "", "output_schema": json.dumps(_schema_for(target))}
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
    result = await Compiler(run_generated_tests=False, system1_tiebreak="none").compile(
        skill, tmp_path / "out"
    )
    assert result.verification is not None and not result.verification.ok
    kinds = {(f.node_id, f.kind) for f in result.verification.errors}
    assert (node.id, "generation_check_failed") in kinds
