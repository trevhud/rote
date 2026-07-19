"""Tests for the runners layer behind ``rote run``.

Detection and input/signal resolution are pure filesystem/string logic
and are tested directly. Execution wrappers are tested with the
underlying trial primitives monkeypatched — the primitives themselves
are covered by test_baseline.py / the empirical eval tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rote.eval.empirical import MeasuredRun
from rote.runners import (
    RUNNABLE_RUNTIMES,
    RunTarget,
    TargetError,
    detect_target,
    parse_signal_args,
    resolve_gate_signals,
    resolve_input,
    run_pipeline,
    run_skill,
)

# ───────── Target detection ─────────


def _mk(d: Path, *names: str) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    for name in names:
        p = d / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("", encoding="utf-8")
    return d


def test_detect_skill_dir(tmp_path: Path) -> None:
    _mk(tmp_path / "skill", "SKILL.md")
    target = detect_target(tmp_path / "skill")
    assert target.kind == "skill"
    assert target.runtime is None


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        (["main.py"], "python"),
        (["main.py", "dbos-config.yaml"], "dbos"),
        (["dbos-config.yaml", "package.json", "src/main.ts"], "dbos-ts"),
        (["wrangler.jsonc", "package.json", "src/workflow.ts"], "cloudflare"),
        (["package.json", "src/inngest/pipeline.ts"], "inngest"),
        (["workflow.py", "activities.py"], "temporal"),
    ],
)
def test_detect_emitted_runtime_dirs(tmp_path: Path, files: list[str], expected: str) -> None:
    _mk(tmp_path / "app", *files)
    target = detect_target(tmp_path / "app")
    assert target.kind == "pipeline"
    assert target.runtime == expected


def test_detect_graduate_out_layout_single_runtime(tmp_path: Path) -> None:
    out = tmp_path / "out"
    _mk(out / "runtime" / "dbos", "main.py", "dbos-config.yaml")
    _mk(out / "graduated", "pipeline.yaml")
    target = detect_target(out)
    assert target.kind == "pipeline"
    assert target.runtime == "dbos"
    assert target.path == (out / "runtime" / "dbos").resolve()
    assert target.pipeline_yaml == (out / "graduated" / "pipeline.yaml").resolve()


def test_detect_graduate_out_layout_needs_runtime_flag_when_ambiguous(tmp_path: Path) -> None:
    out = tmp_path / "out"
    _mk(out / "runtime" / "dbos", "main.py", "dbos-config.yaml")
    _mk(out / "runtime" / "python", "main.py")
    with pytest.raises(TargetError, match="--runtime"):
        detect_target(out)
    target = detect_target(out, runtime="python")
    assert target.runtime == "python"


def test_detect_graduate_out_layout_unknown_runtime_flag(tmp_path: Path) -> None:
    out = tmp_path / "out"
    _mk(out / "runtime" / "dbos", "main.py", "dbos-config.yaml")
    with pytest.raises(TargetError, match="no emitted `temporal` runtime"):
        detect_target(out, runtime="temporal")


def test_detect_runtime_flag_mismatch_on_direct_dir(tmp_path: Path) -> None:
    _mk(tmp_path / "app", "main.py")
    with pytest.raises(TargetError, match="looks like an emitted `python` runtime"):
        detect_target(tmp_path / "app", runtime="dbos")


def test_detect_rejects_unrecognizable_dir(tmp_path: Path) -> None:
    _mk(tmp_path / "junk", "notes.txt")
    with pytest.raises(TargetError, match="neither a skill"):
        detect_target(tmp_path / "junk")


def test_detect_rejects_missing_path(tmp_path: Path) -> None:
    with pytest.raises(TargetError, match="not a directory"):
        detect_target(tmp_path / "nope")


def test_detect_finds_sibling_pipeline_yaml_for_bare_emit(tmp_path: Path) -> None:
    _mk(tmp_path, "pipeline.yaml")
    _mk(tmp_path / "app", "main.py")
    target = detect_target(tmp_path / "app")
    assert target.pipeline_yaml == (tmp_path / "pipeline.yaml").resolve()


# ───────── Input resolution ─────────


def test_resolve_input_inline_json() -> None:
    assert resolve_input('{"a": 1}', skill_dir=None, assume_yes=False) == {"a": 1}


def test_resolve_input_inline_invalid_json() -> None:
    with pytest.raises(TargetError, match="not valid JSON"):
        resolve_input("{broken", skill_dir=None, assume_yes=False)


def test_resolve_input_file(tmp_path: Path) -> None:
    p = tmp_path / "input.json"
    p.write_text('{"b": 2}', encoding="utf-8")
    assert resolve_input(str(p), skill_dir=None, assume_yes=False) == {"b": 2}


def test_resolve_input_missing_file(tmp_path: Path) -> None:
    with pytest.raises(TargetError, match="neither an inline JSON object nor a file"):
        resolve_input(str(tmp_path / "nope.json"), skill_dir=None, assume_yes=False)


def test_resolve_input_rejects_non_object() -> None:
    with pytest.raises(TargetError, match="JSON object"):
        resolve_input("[1, 2]", skill_dir=None, assume_yes=False)


def test_resolve_input_no_input_no_skill_errors() -> None:
    with pytest.raises(TargetError, match="no source skill to derive"):
        resolve_input(None, skill_dir=None, assume_yes=False)


def test_resolve_input_derives_with_assume_yes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("rote.runners.derive_input_payload", lambda skill_dir: {"derived": True})
    payload = resolve_input(None, skill_dir=tmp_path, assume_yes=True)
    assert payload == {"derived": True}


# ───────── Signal parsing + gate resolution ─────────


def test_parse_signal_args() -> None:
    signals = parse_signal_args(['approve={"ok": true}', 'review={"tier": 1}'])
    assert signals == {"approve": {"ok": True}, "review": {"tier": 1}}


def test_parse_signal_args_requires_name_and_json() -> None:
    with pytest.raises(TargetError, match="NAME=JSON"):
        parse_signal_args(["no-equals-sign"])
    with pytest.raises(TargetError, match="not valid JSON"):
        parse_signal_args(["approve={broken"])


def test_resolve_gate_signals_all_provided() -> None:
    provided = {"g1": {"ok": True}}
    assert resolve_gate_signals(["g1"], provided, interactive=False) == provided


def test_resolve_gate_signals_unknown_name() -> None:
    with pytest.raises(TargetError, match="not in the pipeline's gates"):
        resolve_gate_signals(["g1"], {"nope": {}}, interactive=False)


def test_resolve_gate_signals_missing_non_interactive_errors() -> None:
    with pytest.raises(TargetError, match="--signal g2"):
        resolve_gate_signals(["g1", "g2"], {"g1": {}}, interactive=False)


def test_resolve_gate_signals_prompts_interactively(monkeypatch: pytest.MonkeyPatch) -> None:
    answers = iter(['{"approved": true}', ""])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    resolved = resolve_gate_signals(["g1", "g2"], {}, interactive=True)
    assert resolved == {"g1": {"approved": True}, "g2": {}}


def test_resolve_gate_signals_interactive_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("builtins.input", lambda prompt: "{broken")
    with pytest.raises(TargetError, match="not valid JSON"):
        resolve_gate_signals(["g1"], {}, interactive=True)


# ───────── Execution wrappers ─────────


def test_run_skill_applies_wiring(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_wiring(*, allow_writes: bool) -> tuple[dict[str, Any], list[str], dict[str, str]]:
        captured["allow_writes"] = allow_writes
        return (
            {"srv": {"type": "http", "url": "http://x"}},
            ["mcp__srv__lookup"],
            {"other": "unreachable"},
        )

    def fake_trial(skill_dir: Any, payload: Any, **kwargs: Any) -> tuple[Any, list[Any], list[str]]:
        captured["kwargs"] = kwargs
        return MeasuredRun(wall_seconds=1.0, output={"ok": True}), [], []

    monkeypatch.setattr("rote.runners.resolve_mcp_wiring", fake_wiring)
    monkeypatch.setattr("rote.runners.run_baseline_trial", fake_trial)

    outcome = run_skill(tmp_path, {"q": 1}, model="m")
    assert captured["allow_writes"] is False
    assert captured["kwargs"]["mcp_tool_ids"] == ["mcp__srv__lookup"]
    assert captured["kwargs"]["mcp_servers"] == {"srv": {"type": "http", "url": "http://x"}}
    assert outcome.read_only is True
    assert outcome.servers_wired == ("srv",)
    assert outcome.servers_skipped == {"other": "unreachable"}
    assert outcome.run.output == {"ok": True}


def test_run_pipeline_rejects_unorchestrated_runtimes(tmp_path: Path) -> None:
    target = RunTarget(kind="pipeline", path=tmp_path, runtime="temporal")
    with pytest.raises(TargetError, match="cannot orchestrate"):
        run_pipeline(target, {})


def test_run_pipeline_dispatches_to_trial(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_trial(app_dir: Any, payload: Any, **kwargs: Any) -> MeasuredRun:
        captured["app_dir"] = app_dir
        captured["payload"] = payload
        captured["kwargs"] = kwargs
        return MeasuredRun(wall_seconds=0.5, output={"done": 1})

    monkeypatch.setattr("rote.runners.run_pipeline_trial", fake_trial)
    target = RunTarget(kind="pipeline", path=tmp_path, runtime="dbos")
    outcome = run_pipeline(target, {"x": 1}, signals={"g": {"ok": True}}, timeout_seconds=30.0)
    assert captured["app_dir"] == tmp_path
    assert captured["payload"] == {"x": 1}
    assert captured["kwargs"] == {"signals": {"g": {"ok": True}}, "timeout_seconds": 30.0}
    assert outcome.runtime == "dbos"
    assert outcome.run.output == {"done": 1}


def test_runnable_runtimes_is_the_current_set() -> None:
    assert set(RUNNABLE_RUNTIMES) == {"python", "dbos", "cloudflare", "inngest"}


def test_resolve_input_declined_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("rote.runners.derive_input_payload", lambda skill_dir: {"d": 1})
    monkeypatch.setattr("rote.runners.sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    assert resolve_input(None, skill_dir=tmp_path, assume_yes=False) is None


def test_resolve_input_derived_proposal_is_persisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("rote.runners.derive_input_payload", lambda skill_dir: {"d": 1})
    payload = resolve_input(None, skill_dir=tmp_path, assume_yes=True)
    assert payload == {"d": 1}
    err = capsys.readouterr().err
    assert "derived input proposal" in err
    assert json.dumps({"d": 1}, indent=2) in err


# ───────── Cloudflare runner (unit level — live path in test_cloudflare_e2e) ─────────


def test_run_pipeline_dispatches_cloudflare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import rote.runners.cloudflare as cf_runner

    captured: dict[str, Any] = {}

    def fake_run_cloudflare(app_dir: Any, payload: Any, **kwargs: Any) -> MeasuredRun:
        captured["app_dir"] = app_dir
        captured["kwargs"] = kwargs
        return MeasuredRun(wall_seconds=1.0, output={"cf": True})

    monkeypatch.setattr(cf_runner, "run_cloudflare", fake_run_cloudflare)
    target = RunTarget(kind="pipeline", path=tmp_path, runtime="cloudflare")
    outcome = run_pipeline(
        target,
        {"x": 1},
        signals={"g1": {"ok": True}},
        gate_order=["g1"],
        timeout_seconds=42.0,
    )
    assert outcome.run.output == {"cf": True}
    assert captured["kwargs"] == {
        "signals": {"g1": {"ok": True}},
        "gate_order": ["g1"],
        "timeout_seconds": 42.0,
    }


def test_cf_wait_returns_on_terminal_status(monkeypatch: pytest.MonkeyPatch) -> None:
    from rote.runners import cloudflare as cf

    states = iter(
        [
            {"status": "running", "__LOCAL_DEV_STEP_OUTPUTS": [1]},
            {"status": "complete", "output": {"done": 1}},
        ]
    )
    monkeypatch.setattr(cf, "http_get_json", lambda url, timeout=10.0: next(states))
    monkeypatch.setattr(cf.time, "sleep", lambda s: None)
    state = cf._wait_until_parked_or_complete(1, "id", deadline=cf.time.monotonic() + 30)
    assert state["status"] == "complete"


def test_cf_wait_infers_parking_from_step_stability(monkeypatch: pytest.MonkeyPatch) -> None:
    from rote.runners import cloudflare as cf

    stable_state = {"status": "running", "__LOCAL_DEV_STEP_OUTPUTS": [1, 2, 3]}
    monkeypatch.setattr(cf, "http_get_json", lambda url, timeout=10.0: dict(stable_state))
    monkeypatch.setattr(cf.time, "sleep", lambda s: None)
    state = cf._wait_until_parked_or_complete(1, "id", deadline=cf.time.monotonic() + 30)
    assert state["status"] == "running"  # parked, inferred


def test_cf_wait_times_out_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    from rote.eval.empirical import EmpiricalError
    from rote.runners import cloudflare as cf

    growing: list[int] = []

    def _grow(url: str, timeout: float = 10.0) -> dict[str, Any]:
        growing.append(1)
        return {"status": "running", "__LOCAL_DEV_STEP_OUTPUTS": list(growing)}

    monkeypatch.setattr(cf, "http_get_json", _grow)
    monkeypatch.setattr(cf.time, "sleep", lambda s: None)
    with pytest.raises(EmpiricalError, match="timed out"):
        cf._wait_until_parked_or_complete(1, "id", deadline=cf.time.monotonic() + 0.2)


# ───────── Inngest runner (unit level — live path in test_inngest_e2e) ─────────


def test_run_pipeline_inngest_requires_pipeline(tmp_path: Path) -> None:
    target = RunTarget(kind="pipeline", path=tmp_path, runtime="inngest")
    with pytest.raises(TargetError, match="pipeline.yaml"):
        run_pipeline(target, {}, pipeline=None)


def test_run_pipeline_dispatches_inngest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import rote.runners.inngest as inngest_runner

    captured: dict[str, Any] = {}

    def fake_run_inngest(app_dir: Any, payload: Any, **kwargs: Any) -> MeasuredRun:
        captured["kwargs"] = kwargs
        return MeasuredRun(wall_seconds=2.0, output={"inngest": True})

    monkeypatch.setattr(inngest_runner, "run_inngest", fake_run_inngest)
    target = RunTarget(kind="pipeline", path=tmp_path, runtime="inngest")
    sentinel = object()
    outcome = run_pipeline(
        target,
        {"x": 1},
        signals={"g": {}},
        gate_order=["g"],
        pipeline=sentinel,
        timeout_seconds=9.0,
    )
    assert outcome.run.output == {"inngest": True}
    assert captured["kwargs"]["pipeline"] is sentinel
    assert captured["kwargs"]["gate_order"] == ["g"]


def test_inngest_run_output_parses_run_complete(monkeypatch: pytest.MonkeyPatch) -> None:
    from rote.runners import inngest as ig

    ops = [
        {"op": "Step", "id": "a"},
        {"op": "RunComplete", "data": {"final": {"ok": 1}}},
    ]
    response = {"data": {"run": {"status": "Completed", "output": json.dumps(ops)}}}
    monkeypatch.setattr(ig, "http_json", lambda *a, **k: response)
    assert ig._run_output("http://x", "r1") == {"final": {"ok": 1}}


def test_inngest_run_output_errors_without_run_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rote.eval.empirical import EmpiricalError
    from rote.runners import inngest as ig

    response = {"data": {"run": {"status": "Completed", "output": json.dumps([])}}}
    monkeypatch.setattr(ig, "http_json", lambda *a, **k: response)
    with pytest.raises(EmpiricalError, match="RunComplete"):
        ig._run_output("http://x", "r1")
