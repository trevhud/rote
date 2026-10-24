"""A compiled skill computes what its pipeline computes: each example, end to end.

Each committed example that compiles to the skill runtime is emitted with
``rote emit --runtime skill``, installed with ``rote skill install`` into
the test's isolated state, and driven through ``rote skill run`` the way an
agent drives it: ``start``, DECIDE answered with fixed answers, HUMAN gates
answered with ``--said``, until DONE. Its tools are real fastmcp fixture
servers on localhost. The same IR is then emitted for DBOS and run on the
same fixtures, with the same gate answers and its judges answered the same
way by a local Messages stub, and the two results must agree field for
field. DBOS is the other side because it is an independent emitter: the
prefix runner shares :mod:`rote.plan` with the driver, so a plan bug would
hit both and agree with itself.

Then the run's lifecycle, through the CLI with a real driver process: a
SIGKILL mid-wave and a resume, a real 401 that becomes ``HUMAN kind=login``
and a resume once a token is stored, and a plan re-emitted between calls.

No inference. Slow because each parity case runs a DBOS app.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

from rote.adapters import get_adapter
from rote.cli import main
from rote.ir import load_pipeline
from rote.mcp.registry import McpRegistry, McpServerConfig, save_registry
from tests._messages_api import reject_like_the_api, send_json, structured_message
from tests._skill_driver import FixtureServer, fixture_server, header, result_of, run_dir

pytestmark = pytest.mark.slow

REPO = Path(__file__).resolve().parent.parent
EXAMPLES = REPO / "examples"

#: ``ask(block) -> answer`` for a DECIDE step: the agent's own model work.
Decide = Callable[[str], Any]


# ───────── the agent: rote skill run, until DONE ─────────


def _cli(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    status = main(list(argv))
    captured = capsys.readouterr()
    return status, captured.out, captured.err


def items(block: str) -> dict[str, Any]:
    """A fanned DECIDE block's items, by id."""
    loaded: dict[str, Any] = json.loads(
        block.split("--- ITEMS (json, by id) ---\n")[1].split("\n")[0]
    )
    return loaded


def drive_to_done(
    capsys: pytest.CaptureFixture[str],
    name: str,
    payload: Any,
    *,
    decide: dict[str, Decide] | None = None,
    gates: dict[str, tuple[Any, str]] | None = None,
) -> tuple[str, list[str]]:
    """Run ``name`` the way an agent does: ``(the DONE block, the steps it was asked)``.

    Every block is checked to name its own next command, so a block the agent
    could not act on fails here rather than as a hang."""
    decide, gates = decide or {}, gates or {}
    status, out, err = _cli(capsys, "skill", "run", name, "start", json.dumps(payload))
    asked: list[str] = []
    for _ in range(50):
        assert status == 0, out + err
        head = header(out)
        if head["kind"] == "DONE":
            return out, asked
        run, step = head["run"], head["step"]
        asked.append(f"{head['kind']}:{step}")
        assert f"rote skill run {name} " in out.split("--- NEXT ---\n")[1], out
        if head.get("attempt", "1") != "1":
            pytest.fail(f"a fixed answer did not fit:\n{out}")
        if head["kind"] == "DECIDE":
            answer = decide[step](out)
            status, out, err = _cli(
                capsys, "skill", "run", name, "answer", run, step, json.dumps(answer)
            )
        elif head["kind"] == "HUMAN" and head.get("of") == "gate":
            reply, said = gates[step]
            status, out, err = _cli(
                capsys,
                "skill",
                "run",
                name,
                "answer",
                run,
                step,
                json.dumps(reply),
                "--said",
                said,
            )
        else:
            pytest.fail(f"the agent cannot act on this block:\n{out}")
    pytest.fail("the run did not reach DONE in 50 calls")


def install(
    capsys: pytest.CaptureFixture[str],
    compiled: Path,
    out: Path,
    tmp_path: Path,
    *,
    skill: Path | None = None,
    person: pytest.MonkeyPatch | None = None,
) -> Path:
    """Emit ``compiled`` as a skill into ``out`` and install it.

    With ``person``, a person at the terminal confirms the install (a skill
    that relies on a server registered read-only needs one), else ``--yes``."""
    argv = ["emit", str(compiled / "pipeline.yaml"), "--runtime", "skill", "--out", str(out)]
    if skill is not None:
        argv += ["--skill", str(skill)]
    status, _, err = _cli(capsys, *argv)
    assert status == 0, err
    command = ["skill", "install", str(out), "--claude-project", str(tmp_path / "proj")]
    if person is None:
        command.append("--yes")
    else:
        from rote.interact.prompter import Answer
        from tests.test_skill_commands import _person

        _person(
            person, Answer("confirm", "--yes", True), Answer("confirm", "--save-allow-rule", False)
        )
    status, _, err = _cli(capsys, *command)
    assert status == 0, err
    return out


# ───────── the pipeline: the same IR on DBOS, the same fixtures ─────────


class JudgeStub:
    """A local Messages endpoint: each judge call is answered by
    ``answer(node id, prompt)``, the node read from the path its
    ``ROTE_BASE_URL_<ID>`` names. ``calls`` lists the node of every call."""

    def __init__(self) -> None:
        self.answer: Callable[[str, str], Any] = lambda node, prompt: pytest.fail(node)
        self.calls: list[str] = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
                payload = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
                if reject_like_the_api(self, payload):
                    return
                node = self.path.strip("/").split("/")[0]
                prompt = payload["messages"][0]["content"]
                text = prompt if isinstance(prompt, str) else json.dumps(prompt)
                stub.calls.append(node)
                send_json(self, structured_message(payload, stub.answer(node, text)))

            def log_message(self, *args: object) -> None:
                return

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


@pytest.fixture
def judge_stub() -> Iterator[JudgeStub]:
    stub = JudgeStub()
    try:
        yield stub
    finally:
        stub.server.shutdown()
        stub.server.server_close()


def pipeline_result(
    compiled: Path,
    payload: Any,
    work: Path,
    *,
    env: dict[str, str],
    gates: dict[str, Any] | None = None,
    judges: JudgeStub | None = None,
) -> Any:
    """``compiled`` emitted for DBOS and run once: its exit nodes' results."""
    from rote.eval.empirical import run_pipeline_trial

    pipeline = load_pipeline(compiled / "pipeline.yaml")
    app = work / "dbos-app"
    get_adapter("dbos", extracted_source_dir=compiled).emit(pipeline, app)
    signals = {
        node.signal: gates[node.id]
        for node in pipeline.nodes
        if node.signal is not None and gates is not None
    }
    extra = dict(env)
    if judges is not None:
        extra |= {"ROTE_INFERENCE": "api", "ANTHROPIC_API_KEY": "test-key-not-real"}
        for node in pipeline.nodes:
            if node.kind.value == "llm_judge":
                extra[f"ROTE_BASE_URL_{node.id.upper()}"] = f"{judges.url}/{node.id}"
    run = run_pipeline_trial(
        app, payload, signals=signals or None, extra_env=extra, timeout_seconds=300
    )
    assert run.error is None, run.error
    return run.output


def differing_fields(skill: Any, pipeline: Any, at: str = "result") -> list[str]:
    """Every leaf path where the two results differ (empty: they agree)."""
    if isinstance(skill, dict) and isinstance(pipeline, dict):
        return [
            p
            for key in sorted(set(skill) | set(pipeline))
            for p in (
                differing_fields(skill[key], pipeline[key], f"{at}.{key}")
                if key in skill and key in pipeline
                else [f"{at}.{key} (only on the {'skill' if key in skill else 'pipeline'} side)"]
            )
        ]
    if isinstance(skill, list) and isinstance(pipeline, list) and len(skill) == len(pipeline):
        return [
            p
            for i, (a, b) in enumerate(zip(skill, pipeline, strict=True))
            for p in differing_fields(a, b, f"{at}[{i}]")
        ]
    return [] if skill == pipeline else [f"{at}: skill {skill!r} != pipeline {pipeline!r}"]


def _register(servers: dict[str, str], *, read_only: bool = False) -> None:
    save_registry(
        McpRegistry(
            servers={
                name: McpServerConfig(url=url, read_only=read_only or None)
                for name, url in servers.items()
            }
        )
    )


@pytest.fixture(autouse=True)
def _no_proxy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The developer's own proxy or CA variables would be recorded at install."""
    from rote.skill_target.trust import CA_VARS, PROXY_VARS

    for var in (*PROXY_VARS, *CA_VARS):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def fx(tmp_path: Path) -> Iterator[FixtureServer]:
    with fixture_server(tmp_path / "hits.jsonl") as server:
        yield server


# ───────── parity, example by example ─────────


def test_docs_digest_computes_what_its_pipeline_computes(
    tmp_path: Path, fx: FixtureServer, capsys: pytest.CaptureFixture[str]
) -> None:
    from rote import demo

    compiled = EXAMPLES / "docs-digest" / "compiled"
    ref = demo.load_reference()
    servers = {"mslearn": fx.url, "deepwiki": fx.url}
    _register(servers)
    install(capsys, compiled, tmp_path / "docs-digest", tmp_path)
    done, asked = drive_to_done(capsys, "docs-digest", ref.input)
    assert asked == []  # code and tools only: nothing for the agent to do
    skill_side = result_of(done)
    skill_calls = fx.calls()
    fx.reset()
    env = {f"ROTE_MCP_{s.upper()}_URL": url for s, url in servers.items()}
    pipeline_side = pipeline_result(compiled, ref.input, tmp_path / "p", env=env)
    assert differing_fields(skill_side, pipeline_side) == []
    assert skill_side == ref.pipeline_output  # and both are the recorded digest
    # The same tool calls, with the same arguments, on both sides.
    assert sorted(map(json.dumps, skill_calls)) == sorted(map(json.dumps, fx.calls()))
    assert len(skill_calls) == len(ref.tool_calls)


def test_expense_audit_with_a_controllers_answer_matches_its_pipeline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from rote.bench.fidelity import load_cases

    compiled = EXAMPLES / "expense-audit" / "expected"
    (case,) = [c for c in load_cases(EXAMPLES / "expense-audit").cases if c.id == "second_batch"]
    (reply,) = case.gates
    install(capsys, compiled, tmp_path / "expense-audit", tmp_path, skill=compiled.parent / "skill")
    # The report lands under the working directory: the driver runs in the agent's.
    # EXPENSE_AUDIT_REPORT_DIR is set only on the DBOS side below, because
    # `rote skill run` passes the driver an allowlisted environment
    # (rote.skill_target.launch.child_env), so a module's own env var never
    # reaches it there. That gap is trevhud/rote#267.
    monkeypatch.chdir(tmp_path)
    said = "Approve 2410 and 2440, reject 2403 and 2413. Priya"
    done, asked = drive_to_done(
        capsys, "expense-audit", case.input, gates={"controller_review": (reply, said)}
    )
    assert asked == ["HUMAN:controller_review"]
    gate = json.loads((run_dir(done) / "nodes" / "controller_review.json").read_text())
    assert (gate["answered_by"], gate["said"]) == ("human", said)
    skill_side = result_of(done)
    pipeline_side = pipeline_result(
        compiled,
        case.input,
        tmp_path / "p",
        env={"EXPENSE_AUDIT_REPORT_DIR": str(tmp_path / "pipeline-reports")},
        gates={"controller_review": reply},
    )
    # Each side wrote its own report file: the same name and the same bytes.
    paths = [Path(r["file_report"].pop("report_path")) for r in (skill_side, pipeline_side)]
    assert paths[0].parent == tmp_path / "reports"
    assert paths[1].parent == tmp_path / "pipeline-reports"
    assert paths[0].name == paths[1].name
    assert paths[0].read_text(encoding="utf-8") == paths[1].read_text(encoding="utf-8")
    assert differing_fields(skill_side, pipeline_side) == []
    assert skill_side["file_report"]["reviewer"] == "Priya Shah"


def test_ops_report_with_a_duty_managers_answer_matches_its_pipeline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from rote.bench.fidelity import fixture_servers, load_cases, map_servers, read_calls

    compiled = EXAMPLES / "ops-report" / "runs" / "2026-10-23-system1"
    casefile = load_cases(EXAMPLES / "ops-report")
    (case,) = [c for c in casefile.cases if c.id == "busy_day"]
    (reply,) = case.gates
    mapped, unknown = map_servers(load_pipeline(compiled / "pipeline.yaml"), casefile)
    assert unknown == []
    with fixture_servers(casefile, case, tmp_path / "fixtures") as (urls, log):
        servers = {server: urls[target] for server, target in mapped.items()}
        # The bench fixtures mark no tool read-only: the user's own read-only
        # registration is what lets a read-only skill call them.
        _register(servers, read_only=True)
        install(capsys, compiled, tmp_path / "ops-report", tmp_path, person=monkeypatch)
        said = "18 green, 2 red, 1 grey, 1 other; sorters and dwell as in the sheet"
        done, asked = drive_to_done(
            capsys, "ops-report", case.input, gates={"manual_data_gate": (reply, said)}
        )
        skill_calls = read_calls(log)
        log.write_text("", encoding="utf-8")
        env = {f"ROTE_MCP_{server.upper()}_URL": url for server, url in servers.items()}
        pipeline_side = pipeline_result(
            compiled, case.input, tmp_path / "p", env=env, gates={"manual_data_gate": reply}
        )
        pipeline_calls = read_calls(log)
    assert asked == ["HUMAN:manual_data_gate"]
    skill_side = result_of(done)
    assert differing_fields(skill_side, pipeline_side) == []
    assert set(skill_side) == {"executive_report"}
    # Four reads, one per mcp-bound node, with the same arguments on both sides.
    assert sorted(c["tool"] for c in skill_calls) == ["read_file_content"] * 3 + ["search_threads"]
    key = lambda c: json.dumps([c["tool"], c["arguments"]], sort_keys=True)  # noqa: E731
    assert sorted(map(key, skill_calls)) == sorted(map(key, pipeline_calls))


# deal-monitor's committed IR leaves four modules as stubs. These fill them
# from the fixtures' shapes (tests/_skill_mcp_server.py), and use every
# constant the IR passes, so a driver that drops one cannot agree.
DEAL_MODULES = {
    "filtering.py": """\
FIELDS = (
    "account_name", "submitter", "arr", "monthly_dtc_orders", "sku_count",
    "sku_count_unknown", "requested_location", "sales_channels", "current_situation",
    "pallets", "sq_ft", "international_shipping", "go_live_date", "product_item_context",
)


def filter_opportunities(opportunities, na_sku_threshold, always_uk_submitters):
    kept, excluded = [], []
    for raw in opportunities:
        opp = {k: raw.get(k) for k in FIELDS}
        opp["sku_count_unknown"] = bool(raw.get("sku_count_unknown"))
        opp["is_eu_uk"] = (
            raw["requested_location"] in ("UK", "EU") or raw["submitter"] in always_uk_submitters
        )
        opp["sku_flagged_as_risk"] = not opp["is_eu_uk"] and opp["sku_count_unknown"]
        too_big = not opp["is_eu_uk"] and (opp["sku_count"] or 0) >= na_sku_threshold
        (excluded if too_big else kept).append(opp)
    return {"filtered_opps": kept, "excluded_opps": excluded}


def dedup_email_threads(standard_threads, account_threads):
    seen, threads = set(), []
    for thread in [*standard_threads, *account_threads]:
        if thread["thread_id"] not in seen:
            seen.add(thread["thread_id"])
            threads.append(thread)
    return {"threads": threads}
""",
    "matching.py": """\
def match_threads_to_opps(opps, email_threads):
    matched, unmatched = [], []
    for opp in opps:
        hits = [t for t in email_threads if opp["account_name"].lower() in t["subject"].lower()]
        matched += [{"account_name": opp["account_name"], **t} for t in hits]
        if not hits:
            unmatched.append(opp)
    return {"matched_threads": matched, "unmatched_opps": unmatched}
""",
    "dashboard.py": """\
def render_dashboard(scored_opps, classified_threads, all_filtered_opps, unmatched_opps):
    quoting = [t for t in classified_threads if t["step"] != 6]
    closed = [t for t in classified_threads if t["step"] == 6]
    rows = [f"<li>{s['opportunity']['account_name']}: {s['score_tier']}</li>" for s in scored_opps]
    rows += [f"<li>{t['account_name']}: {t['step_label']}</li>" for t in quoting + closed]
    summary = {
        "new": len(scored_opps),
        "quoting": len(quoting),
        "attention": sum(t["needs_attention"] for t in quoting),
        "closed": len(closed),
        "tracked": len(all_filtered_opps),
        "eu_uk": sum(o["is_eu_uk"] for o in all_filtered_opps),
        "unmatched": len(unmatched_opps),
    }
    return {"html": "<ul>" + "".join(rows) + "</ul>", "summary": summary}
""",
    "output.py": """\
from pathlib import Path


def save_dashboard(html, summary, output_path="outputs/deal-monitor.html"):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    text = (
        f"{summary['tracked']} tracked ({summary['eu_uk']} EU/UK), {summary['new']} new, "
        f"{summary['quoting']} quoting ({summary['attention']} need attention), "
        f"{summary['closed']} closed"
    )
    return {"saved_path": str(path), "summary_text": text}
""",
}

#: The opportunities in the fixture's six posts, as a reader extracts them:
#: the answer to parse_intake_messages, on both sides.
DEAL_OPPS = [
    {
        "account_name": name,
        "submitter": rep,
        "sku_count": skus,
        "sku_count_unknown": skus is None,
        "requested_location": where,
        "raw_message": text,
    }
    for name, skus, where, rep, text in [
        ("Acme Corp", 40, "NA", "Dana", "New deal: Acme Corp, 40 SKUs, NA, rep Dana"),
        ("Brightline", 12, "UK", "Sam", "New deal: Brightline, 12 SKUs, UK, rep Sam"),
        ("Corvid", 300, "EU", "Lee", "New deal: Corvid, 300 SKUs, EU, rep Lee"),
        ("Dynamo", 5000, "NA", "Kai", "New deal: Dynamo, 5000 SKUs, NA, rep Kai"),
        (
            "Echo Labs",
            None,
            "Nashville",
            "Sam Patel",
            "New deal: Echo Labs, SKUs unknown, Nashville, rep Sam Patel",
        ),
        ("Fjord Supply", 80, "EU", "Lee", "New deal: Fjord Supply, 80 SKUs, EU, rep Lee"),
    ]
]

#: Each matched thread's classification, by account: one needs attention,
#: one is declined (the Closed tab).
CLASSIFIED = {
    "Acme Corp": ("t1", 4, "Pricing Returned", False),
    "Brightline": ("t2", 2, "Response Needed", True),
    "Echo Labs": ("t3", 6, "Declined", False),
}
#: Each new opportunity's score tier, by account.
TIERS = {"Corvid": "medium", "Fjord Supply": "high"}


def classify(account: str) -> dict[str, Any]:
    thread, step, label, attention = CLASSIFIED[account]
    return {
        "step": step,
        "step_label": label,
        "warehouses_emailed": [
            {"warehouse_domain": "wh.example.com", "sent_date": "2026-10-01", "replied": step > 1}
        ],
        "needs_attention": attention,
        "classification_rationale": f"{account}: {label}",
        "account_name": account,
        "thread_id": thread,
        "thread_url": f"https://mail.example.com/{thread}",
    }


def score(opportunity: dict[str, Any]) -> dict[str, Any]:
    tier = TIERS[opportunity["account_name"]]
    return {
        "opportunity": opportunity,
        "score_tier": tier,
        "score_value": {"high": 4.5, "medium": 3.0}[tier],
        "risk_flags": [],
        "opportunity_summary": f"{opportunity['account_name']} scores {tier}.",
        "scoring_rationale": "Per the rubric.",
    }


def test_deal_monitor_with_the_agents_judgments_matches_its_pipeline(
    tmp_path: Path,
    fx: FixtureServer,
    judge_stub: JudgeStub,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiled = tmp_path / "compiled"
    shutil.copytree(EXAMPLES / "deal-monitor" / "expected", compiled)
    for name, source in DEAL_MODULES.items():
        (compiled / "extracted" / name).write_text(source, encoding="utf-8")
    servers = {"slack": fx.url, "gmail": fx.url}
    _register(servers)
    install(
        capsys,
        compiled,
        tmp_path / "deal-monitor",
        tmp_path,
        skill=EXAMPLES / "deal-monitor" / "skill",
    )
    payload = {"user_email": "ops@example.com", "scoring_rubric": "Score 1 to 5."}
    (tmp_path / "skill-cwd").mkdir()
    monkeypatch.chdir(tmp_path / "skill-cwd")
    seen: dict[str, list[Any]] = {}

    def remember(step: str, block: str) -> dict[str, Any]:
        found = items(block)
        seen.setdefault(step, []).append(found)
        return found

    done, asked = drive_to_done(
        capsys,
        "deal-monitor",
        payload,
        decide={
            "parse_intake_messages": lambda block: {"opportunities": DEAL_OPPS},
            "classify_thread_step": lambda block: {
                i: classify(item["account_name"]) for i, item in remember("classify", block).items()
            },
            "score_new_opportunities": lambda block: {
                i: score(item) for i, item in remember("score", block).items()
            },
        },
    )
    skill_side = result_of(done)
    saved = Path("outputs/deal-monitor.html").read_text(encoding="utf-8")
    # The batches held every element: three matched threads, two new opportunities.
    assert [sorted(i["account_name"] for i in batch.values()) for batch in seen["classify"]] == [
        ["Acme Corp", "Brightline", "Echo Labs"]
    ]
    assert [sorted(i["account_name"] for i in batch.values()) for batch in seen["score"]] == [
        ["Corvid", "Fjord Supply"]
    ]
    assert asked == [
        "DECIDE:parse_intake_messages",
        "DECIDE:classify_thread_step",
        "DECIDE:score_new_opportunities",
    ]

    def answer(node: str, prompt: str) -> Any:
        if node == "parse_intake_messages":
            return {"opportunities": DEAL_OPPS}
        if node == "classify_thread_step":
            return classify(prompt.split("Account name: ")[1].split("\n")[0])
        assert node == "score_new_opportunities", node
        return score(json.loads(prompt.split("unchanged in your output:\n")[1].split("\n")[0]))

    judge_stub.answer = answer
    skill_calls = fx.calls()
    fx.reset()
    env = {f"ROTE_MCP_{s.upper()}_URL": url for s, url in servers.items()}
    pipeline_side = pipeline_result(compiled, payload, tmp_path / "p", env=env, judges=judge_stub)
    assert differing_fields(skill_side, pipeline_side) == []
    assert skill_side["save_dashboard"]["summary_text"] == (
        "5 tracked (4 EU/UK), 2 new, 2 quoting (1 need attention), 1 closed"
    )
    # The app runs in its own dir, so the same relative path is a second file.
    html = tmp_path / "p" / "dbos-app" / "outputs/deal-monitor.html"
    assert html.read_text(encoding="utf-8") == saved
    # One judge call per element on the pipeline side: 1 + 3 + 2.
    assert sorted(judge_stub.calls) == sorted(
        ["parse_intake_messages"] + ["classify_thread_step"] * 3 + ["score_new_opportunities"] * 2
    )
    key = lambda c: json.dumps(c, sort_keys=True)  # noqa: E731
    assert sorted(map(key, skill_calls)) == sorted(map(key, fx.calls()))


# ───────── the run's lifecycle, through the CLI ─────────

#: Two tool steps in one wave, a gate, then code over all three.
WAVE = {
    "name": "waver",
    "version": "0.1.0",
    "input": {"type": "In", "required": ["q"]},
    "nodes": [
        {
            "id": "found",
            "kind": "external_call",
            "description": "Search.",
            "mcp": {"server": "fx", "tool": "search", "args": {"query": "q"}},
            "inputs": {"q": "pipeline.input.q"},
        },
        {
            "id": "held",
            "kind": "external_call",
            "description": "A slow read.",
            "mcp": {"server": "fx", "tool": "hold", "args": {"key": "q"}},
            "inputs": {"q": "pipeline.input.q"},
        },
        {
            "id": "approve",
            "kind": "hitl_gate",
            "description": "Approve the search results.",
            "signal": "approved",
            "timeout": "1h",
            "output": {"approved": "bool", "approver": "str"},
            "inputs": {"found": "found.output"},
        },
        {
            "id": "finished",
            "kind": "pure_function",
            "description": "Finish.",
            "impl": "extracted/finish.py:finish",
            "inputs": {
                "found": "found.output",
                "held": "held.output",
                "approval": "approve.output",
            },
        },
    ],
    "edges": [
        {"from": "found", "to": "approve"},
        {"from": "held", "to": "approve"},
        {"from": "approve", "to": "finished"},
    ],
    "entry_nodes": ["found", "held"],
    "exit_nodes": ["finished"],
}
FINISH = """\
def finish(found, held, approval):
    return {"titles": [r["title"] for r in found], "held": held["held"], "by": approval["approver"]}
"""
APPROVAL = {"approved": True, "approver": "Dana"}


def install_waver(
    root: Path, capsys: pytest.CaptureFixture[str], pipeline: dict[str, Any] = WAVE, **person: Any
) -> Path:
    import yaml

    source = root / "source"
    source.mkdir(exist_ok=True)
    (source / "SKILL.md").write_text(
        "---\nname: waver\ndescription: Search, then ask.\n---\n# Waver\n", encoding="utf-8"
    )
    compiled = root / "compiled"
    (compiled / "extracted").mkdir(parents=True, exist_ok=True)
    (compiled / "extracted" / "finish.py").write_text(FINISH, encoding="utf-8")
    (compiled / "pipeline.yaml").write_text(yaml.safe_dump(pipeline), encoding="utf-8")
    return install(capsys, compiled, root / "waver", root, skill=source, **person)


def _rote(*argv: str, **popen: Any) -> subprocess.Popen[str]:
    """``rote`` as its own process, the way an agent's shell starts it."""
    return subprocess.Popen(
        [sys.executable, "-c", "import sys; from rote.cli import main; sys.exit(main())", *argv],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        **popen,
    )


def _wait_for(condition: Callable[[], bool], what: str, proc: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if condition():
            return
        if proc.poll() is not None:
            pytest.fail(f"rote exited before {what}: {proc.communicate()}")
        time.sleep(0.05)
    pytest.fail(f"timed out waiting for {what}")


def test_a_kill_mid_wave_resumes_without_repeating_a_finished_call(
    tmp_path: Path, fx: FixtureServer, capsys: pytest.CaptureFixture[str]
) -> None:
    _register({"fx": fx.url})
    install_waver(tmp_path, capsys)
    runs = Path(os.environ["ROTE_SKILL_RUN_DIR"]) / "waver"
    gate = fx.hits.with_name("hold")
    gate.write_text("", encoding="utf-8")
    proc = _rote("skill", "run", "waver", "start", '{"q": "owl"}', start_new_session=True)
    # Stop it once the wave's search is journaled and its slow read is in flight.
    _wait_for(lambda: len(fx.calls("hold")) == 1, "the slow read", proc)
    _wait_for(lambda: any(runs.glob("*/nodes/found.json")), "the search result", proc)
    os.killpg(proc.pid, signal.SIGKILL)
    proc.wait(timeout=10)
    (folder,) = runs.iterdir()
    assert not (folder / "nodes" / "held.json").exists()
    assert json.loads((folder / "run.json").read_text())["wave"]["index"] == 0  # claimed, dead
    gate.unlink()
    status, out, err = _cli(capsys, "skill", "run", "waver", "resume", folder.name)
    assert status == 0, out + err
    assert (header(out)["kind"], header(out)["step"]) == ("HUMAN", "approve")
    assert "taking over from pid" in err
    # The finished call is not repeated; the one in flight at the kill is.
    assert [c["arguments"] for c in fx.calls("search")] == [{"query": "owl"}]
    assert [c["arguments"] for c in fx.calls("hold")] == [{"key": "owl"}] * 2
    status, done, err = _cli(
        capsys,
        "skill",
        "run",
        "waver",
        "answer",
        folder.name,
        "approve",
        json.dumps(APPROVAL),
        "--said",
        "yes, Dana",
    )
    assert status == 0, done + err
    assert result_of(done) == {
        "finished": {"titles": ["owl 0", "owl 1", "owl 2"], "held": True, "by": "Dana"}
    }


def test_a_real_401_asks_for_a_login_and_resume_runs_once_a_token_is_stored(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from rote.mcp import write_token_file

    with fixture_server(tmp_path / "hits.jsonl", bearer="fixture-token") as fx:
        _register({"fx": fx.url})
        install_waver(tmp_path, capsys)
        status, out, err = _cli(capsys, "skill", "run", "waver", "start", '{"q": "owl"}')
        assert status == 0, out + err
        head = header(out)
        assert (head["kind"], head["of"], head["server"]) == ("HUMAN", "login", "fx"), out
        assert "rote mcp login fx" in out
        assert fx.calls() == []  # the 401 came before any tool ran

        def token(value: str) -> None:
            write_token_file(
                "fx",
                {
                    "server_url": fx.url,
                    "tokens": {"access_token": value, "token_type": "Bearer"},
                    "expires_at": time.time() + 3600,
                    "client_info": None,
                    "token_endpoint": None,
                },
            )

        # Negative half: a token the server refuses is still a login.
        token("wrong-token")
        status, out, err = _cli(capsys, "skill", "run", "waver", "resume", head["run"])
        assert (status, header(out).get("of")) == (0, "login"), out + err
        assert fx.calls() == []
        token("fixture-token")
        status, out, err = _cli(capsys, "skill", "run", "waver", "resume", head["run"])
        assert status == 0, out + err
        assert (header(out)["kind"], header(out)["step"]) == ("HUMAN", "approve"), out
        assert sorted(c["tool"] for c in fx.calls()) == ["hold", "search"]


def test_a_plan_re_emitted_between_calls_ends_the_run_and_a_new_one_starts(
    tmp_path: Path,
    fx: FixtureServer,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _register({"fx": fx.url})
    install_waver(tmp_path, capsys)
    status, out, err = _cli(capsys, "skill", "run", "waver", "start", '{"q": "owl"}')
    assert header(out)["kind"] == "HUMAN", out + err
    run = header(out)["run"]
    changed = json.loads(json.dumps(WAVE))
    changed["nodes"][2]["description"] = "Approve the results before they are sent."
    # The person re-emits and re-installs between two of the agent's calls.
    install_waver(tmp_path, capsys, changed, person=monkeypatch)
    status, out, err = _cli(
        capsys,
        "skill",
        "run",
        "waver",
        "answer",
        run,
        "approve",
        json.dumps(APPROVAL),
        "--said",
        "yes",
    )
    assert status == 1 and header(out)["kind"] == "ERROR", out + err
    assert "re-emitted after the run started" in out
    folder = Path(os.environ["ROTE_SKILL_RUN_DIR"]) / "waver" / run
    assert not (folder / "nodes" / "approve.json").exists()
    assert not (folder / "result.json").exists()
    # A new run asks the new plan's question.
    status, out, err = _cli(capsys, "skill", "run", "waver", "start", '{"q": "owl"}')
    assert status == 0 and header(out)["run"] != run, out + err
    assert "Approve the results before they are sent." in out
