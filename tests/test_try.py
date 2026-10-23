"""`rote try`: the plan it shows, what runs, and how outputs compare.

No inference and no network. The fast tests swap the three runners
(`rote.proof.Runners`: skill run, compile, pipeline run) for fakes that
record their calls, so "nothing paid runs" is asserted on the calls
themselves. The slow test runs the real emitted DBOS app against local
fixture MCP servers answering with the reference's recorded tool calls.
"""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from rote import demo, proof
from rote.cli import main as cli_main
from rote.eval.empirical import MeasuredRun
from rote.ir import NodeKind, Pipeline, load_pipeline

REPO = Path(__file__).resolve().parent.parent


class FakeRunners:
    """Records every call; returns what the test set up."""

    def __init__(
        self,
        *,
        pipeline_output: dict[str, Any] | None = None,
        skill_output: dict[str, Any] | None = None,
        compiled_from: Path | None = None,
    ) -> None:
        ref = demo.load_reference()
        self.pipeline_output = pipeline_output or copy.deepcopy(ref.pipeline_output)
        self.skill_output = skill_output or copy.deepcopy(ref.skill_output)
        self.compiled_from = compiled_from or demo.pipeline_path().parent
        self.calls: list[str] = []

    def run_skill(self, skill_dir: Path, payload: dict[str, Any]) -> MeasuredRun:
        self.calls.append("skill")
        return MeasuredRun(wall_seconds=12.0, output=self.skill_output, turns=5, cost_usd=0.091)

    def compile(self, skill_dir: Path, input_path: Path, out_dir: Path) -> proof.CompileRun:
        self.calls.append("compile")
        shutil.copytree(self.compiled_from, out_dir / "compiled")
        (out_dir / "runtime" / "dbos").mkdir(parents=True)
        return proof.CompileRun(out_dir, 30.0, 6, 0.07)

    def run_pipeline(
        self, app_dir: Path, payload: dict[str, Any], pipeline: Pipeline
    ) -> MeasuredRun:
        self.calls.append("pipeline")
        assert payload == demo.load_reference().input
        return MeasuredRun(wall_seconds=6.0, output=self.pipeline_output)

    def install(self, monkeypatch: pytest.MonkeyPatch) -> FakeRunners:
        runners = proof.Runners(
            run_skill=self.run_skill, compile=self.compile, run_pipeline=self.run_pipeline
        )
        monkeypatch.setattr(proof.Runners, "default", classmethod(lambda cls: runners))
        return self


def _try(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict[str, Any]]:
    capsys.readouterr()
    code = cli_main(["try", *argv, "--json"])
    return code, json.loads(capsys.readouterr().out)


def _statuses(payload: dict[str, Any]) -> dict[str, str]:
    return {f["name"]: f["status"] for f in payload["comparison"]["fields"]}


def _digest_with(section: str, body: str) -> dict[str, Any]:
    """The recorded pipeline output with one section's body replaced."""
    out = copy.deepcopy(demo.load_reference().pipeline_output)
    digest: str = out["write_digest"]["digest"]
    old = demo.markdown_section(digest, section)
    out["write_digest"]["digest"] = digest.replace(old, body)
    return out


# ───────── the free path ─────────


def test_the_free_path_registers_the_servers_runs_the_pipeline_and_nothing_paid(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from rote.mcp.learned import learned_auth
    from rote.mcp.registry import load_registry

    fakes = FakeRunners().install(monkeypatch)
    code, payload = _try(capsys, "--yes")
    assert code == 0
    assert fakes.calls == ["pipeline"]
    assert payload["mode"] == "precompiled"
    assert payload["headline"] == "reproduced 2 of 2 fields"
    assert payload["same_result"] is True
    assert _statuses(payload) == {
        "sections": "matched",
        "reading_list_size": "matched",
        "reading_list": "matched",
        "repo_outline": "matched",
    }
    # Before is the recorded skill run, after is this run.
    recorded = demo.load_reference().raw["skill_run"]
    assert payload["before"]["cost_usd"] == recorded["cost_usd"] == 0.088
    assert payload["before"]["turns"] == recorded["turns"] == 4
    assert payload["after"] == {
        "label": "compiled pipeline (this run)",
        "wall_seconds": 6.0,
        "turns": None,
        "model_calls": 0,
        "cost_usd": 0.0,
    }
    # The plan has no spend at all, and registered exactly what `rote mcp
    # add <name> <url> --auth none --read-only` would.
    effects = [e for a in payload["plan"]["actions"] for e in a["effects"]]
    assert [e for e in effects if e["kind"] == "spend"] == []
    assert payload["plan"]["needs"] == ["--yes"]
    servers = load_registry().servers
    assert {n: (c.url, c.read_only) for n, c in servers.items()} == {
        "mslearn": ("https://learn.microsoft.com/api/mcp", True),
        "deepwiki": ("https://mcp.deepwiki.com/mcp", True),
    }
    for s in demo.load_reference().servers:
        learned = learned_auth(s.name, s.url)
        assert learned is not None and learned.auth == "none"
    register = payload["plan"]["actions"][0]
    assert register["command"] == (
        "rote mcp add mslearn https://learn.microsoft.com/api/mcp --auth none --read-only && "
        "rote mcp add deepwiki https://mcp.deepwiki.com/mcp --auth none --read-only"
    )
    hosts = [e["host"] for e in effects if e["kind"] == "contact"]
    assert hosts == ["learn.microsoft.com", "mcp.deepwiki.com"]


def test_a_second_run_does_not_register_again(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    FakeRunners().install(monkeypatch)
    assert _try(capsys, "--yes")[0] == 0
    code, payload = _try(capsys, "--yes")
    assert code == 0
    assert [a["id"] for a in payload["plan"]["actions"]] == ["run-pipeline"]
    assert "mslearn: already registered read-only, no login" in payload["plan"]["notes"]


def test_a_deterministic_field_that_differs_is_differs_not_drifted(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The reading list's length is fixed by the skill (5): one item short
    is a real difference, while the list's own text is live data."""
    four = "\n".join(
        demo.markdown_section(
            demo.load_reference().pipeline_output["write_digest"]["digest"], "Reading list"
        ).splitlines()[:4]
    )
    FakeRunners(pipeline_output=_digest_with("Reading list", four)).install(monkeypatch)
    code, payload = _try(capsys, "--yes")
    assert code == 1
    assert _statuses(payload) == {
        "sections": "matched",
        "reading_list_size": "differs",
        "reading_list": "drifted",
        "repo_outline": "matched",
    }
    assert payload["headline"] == "reproduced 1 of 2 fields"
    assert payload["same_result"] is False
    size = next(f for f in payload["comparison"]["fields"] if f["name"] == "reading_list_size")
    assert (size["skill"], size["pipeline"]) == (5, 4)


def test_live_data_that_changed_is_drifted_and_still_the_same_result(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    outline = "Available pages for Azure/azure-functions-durable-python:\n\n- 1 Overview"
    FakeRunners(pipeline_output=_digest_with("Repo outline", outline)).install(monkeypatch)
    code, payload = _try(capsys, "--yes")
    assert code == 0
    assert _statuses(payload)["repo_outline"] == "drifted"
    assert payload["headline"] == "reproduced 2 of 2 fields"
    assert payload["same_result"] is True


def test_a_missing_section_is_differs_even_on_a_drift_field(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    broken = {"write_digest": {"digest": "## Reading list\n\n- [a](b)\n"}}
    FakeRunners(pipeline_output=broken).install(monkeypatch)
    code, payload = _try(capsys, "--yes")
    assert code == 1
    outline = next(f for f in payload["comparison"]["fields"] if f["name"] == "repo_outline")
    assert outline["status"] == "differs"
    assert "no '## Repo outline' heading" in outline["problem"]


def test_without_yes_nothing_runs_or_is_written(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from rote.mcp.registry import registry_path

    fakes = FakeRunners().install(monkeypatch)
    code, payload = _try(capsys)
    assert code == 2
    assert payload["reason"] == "confirmation_required"
    assert payload["missing"] == ["--yes"]
    assert fakes.calls == []
    assert not registry_path().exists()


def test_the_text_report_reads_side_by_side(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    FakeRunners().install(monkeypatch)
    capsys.readouterr()
    assert cli_main(["try", "--yes"]) == 0
    err = capsys.readouterr().err.splitlines()
    assert "rote try docs-digest: reproduced 2 of 2 fields" in err
    [cost] = [line for line in err if line.lstrip().startswith("cost ")]
    assert cost.split() == ["cost", "$0.088", "$0.000"]
    assert "Same result on all 2 fields expected to match exactly." in err


# ───────── the live path ─────────


def test_live_without_the_live_flag_spends_nothing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """A skill directory is always the live path. --yes alone must not
    start it: the plan names --live, and no runner is called."""
    fakes = FakeRunners().install(monkeypatch)
    skill = tmp_path / "skill"
    shutil.copytree(demo.skill_dir(), skill)
    code, payload = _try(capsys, str(skill), "--yes", "--out", str(tmp_path / "out"))
    assert code == 2
    assert payload["missing"] == ["--live"]
    assert fakes.calls == []
    assert not (tmp_path / "out").exists()
    # Negative half: the same command with --live runs all three, in order.
    code, payload = _try(capsys, str(skill), "--yes", "--live", "--out", str(tmp_path / "out"))
    assert code == 0, payload
    assert fakes.calls == ["skill", "compile", "pipeline"]


def test_live_demo_plan_carries_the_recorded_costs_as_its_spend(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    fakes = FakeRunners().install(monkeypatch)
    code, payload = _try(capsys, "--live", "--yes", "--out", str(tmp_path / "out"))
    assert code == 0, payload
    assert fakes.calls == ["skill", "compile", "pipeline"]
    spends = {
        a["id"]: [e for e in a["effects"] if e["kind"] == "spend"]
        for a in payload["plan"]["actions"]
    }
    raw = demo.load_reference().raw
    assert spends["run-skill"] == [
        {
            "kind": "spend",
            "low_usd": raw["skill_run"]["cost_usd"],
            "high_usd": raw["skill_run"]["cost_usd"],
            "payer": proof.SUBSCRIPTION,
            "flag": "--live",
        }
    ]
    assert [s["low_usd"] for s in spends["compile"]] == [raw["compile"]["cost_usd"]]
    # The recorded pipeline made no model call, so its run spends nothing.
    assert spends["run-pipeline"] == []
    assert (
        "total estimate: about $0.16, billed to your Claude subscription (claude -p)"
        in (payload["plan"]["notes"])
    )
    # Both sides are this run's.
    assert payload["before"]["label"] == "skill, as an agent (this run)"
    assert payload["before"]["cost_usd"] == 0.091
    assert payload["compile"]["cost_usd"] == 0.07
    assert payload["headline"] == "reproduced 2 of 2 fields"


def test_a_compiled_pipeline_that_calls_a_model_does_not_run_unapproved(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """The demo's plan has no spend for the pipeline run (the recorded one
    made no model call). A fresh compile that added a judge must not run it."""
    bdr = REPO / "examples" / "bdr-outreach" / "expected"
    fakes = FakeRunners(compiled_from=bdr).install(monkeypatch)
    code, payload = _try(capsys, "--live", "--yes", "--out", str(tmp_path / "out"))
    assert code == 1
    assert fakes.calls == ["skill", "compile"]
    [failed] = [o for o in payload["outcomes"] if o["status"] == "failed"]
    assert failed["id"] == "run-pipeline"
    assert "calls a model" in failed["error"]


def test_an_existing_compile_output_needs_force(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    fakes = FakeRunners().install(monkeypatch)
    out = tmp_path / "out"
    (out / "compiled").mkdir(parents=True)
    (out / "compiled" / "pipeline.yaml").write_text("kept", encoding="utf-8")
    code, payload = _try(capsys, "--live", "--yes", "--out", str(out))
    assert code == 2
    assert payload["missing"] == ["--force"]
    assert fakes.calls == []
    assert (out / "compiled" / "pipeline.yaml").read_text(encoding="utf-8") == "kept"


# ───────── fields from the IR, and estimates ─────────


def _demo_pipeline() -> Pipeline:
    return load_pipeline(demo.pipeline_path())


def test_fields_from_the_ir_are_equal_for_code_and_wording_downstream_of_a_judge() -> None:
    pipeline = _demo_pipeline()
    [digest] = proof.fields_from_ir(pipeline)
    assert (digest.name, digest.skill, digest.pipeline, digest.expect) == (
        "digest",
        "/digest",
        "/write_digest/digest",
        "equal",
    )
    judged = pipeline.model_copy(
        update={
            "nodes": [
                n.model_copy(update={"kind": NodeKind.LLM_JUDGE})
                if n.id == "pick_reading_list"
                else n
                for n in pipeline.nodes
            ]
        }
    )
    [wording] = proof.fields_from_ir(judged)
    assert wording.expect == "wording"
    assert "pick_reading_list" in wording.why
    results = proof.compare_outputs(
        [wording, digest], {"digest": "one wording"}, {"write_digest": {"digest": "another"}}
    )
    assert [r.status for r in results] == ["differs-judge", "differs"]


def test_a_user_skill_estimate_scales_the_recorded_costs_by_rotes_estimate() -> None:
    """The demo skill estimated as if it were a user's: the compile scales
    by 1 exactly, and the agent run's range brackets the recorded cost."""
    ref = demo.load_reference()
    est = proof.estimate_spend(ref, demo.skill_dir())
    assert est.compile == (ref.raw["compile"]["cost_usd"], ref.raw["compile"]["cost_usd"])
    low, high = est.skill_run
    assert low < ref.raw["skill_run"]["cost_usd"] < high
    assert est.pipeline_run == (0.0, high)
    # The demo itself uses the recorded numbers as they are.
    own = proof.estimate_spend(ref, None)
    assert own.skill_run == (0.088, 0.088)
    assert own.pipeline_run is None


# ───────── the real pipeline, against fixture servers ─────────


@pytest.mark.slow
def test_the_free_path_runs_the_real_pipeline_and_reproduces_the_recording(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The emitted DBOS app, live, against local MCP servers answering with
    the recorded tool calls: the comparison must be the recording's own."""
    import subprocess
    import sys
    import time

    from rote.bench.fidelity import _free_port, _wait_for_port

    ref = demo.load_reference()
    snapshot = json.loads((demo.skill_dir() / "mcp-tools.json").read_text(encoding="utf-8"))
    log = tmp_path / "calls.jsonl"
    procs = []
    try:
        for call in ref.tool_calls:
            [tool] = [
                t for t in snapshot["servers"][call.server]["tools"] if t["name"] == call.tool
            ]
            spec = tmp_path / f"{call.server}.json"
            spec.write_text(
                json.dumps(
                    {
                        "server": call.server,
                        "tools": {
                            call.tool: {
                                "description": tool["description"],
                                "parameters": tool["inputSchema"],
                                "read_only_hint": True,
                                "responses": [{"when": call.arguments, "result": call.result}],
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            port = _free_port()
            stderr = tmp_path / f"{call.server}.stderr"
            with stderr.open("w") as err:
                proc = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "rote.bench.fixture_mcp",
                        str(spec),
                        str(port),
                        str(log),
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=err,
                )
            procs.append(proc)
            _wait_for_port(port, proc, time.monotonic() + 30, stderr)
            monkeypatch.setenv(
                f"ROTE_MCP_{call.server.upper()}_URL", f"http://127.0.0.1:{port}/mcp"
            )
        code, payload = _try(capsys, "--yes")
    finally:
        for proc in procs:
            proc.terminate()
            proc.wait(timeout=10)
    assert code == 0, payload
    assert payload["headline"] == "reproduced 2 of 2 fields"
    assert set(_statuses(payload).values()) == {"matched"}
    calls = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert sorted((c["server"], c["tool"], c["matched"]) for c in calls) == [
        ("deepwiki", "read_wiki_structure", True),
        ("mslearn", "microsoft_docs_search", True),
    ]
