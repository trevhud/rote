"""Tests for the rote CLI.

Two flavors:

* **In-process tests** call ``rote.cli.main`` directly with fake argv.
  Fast and lets us assert return codes + captured stdout/stderr. These
  cover argument parsing, error handling, and subcommand dispatch.
* **Subprocess tests** invoke the installed ``rote`` entry point via
  ``python -m rote.cli`` so we catch packaging / import-time issues the
  in-process tests would miss.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from rote.cli import main as cli_main

REPO_ROOT = Path(__file__).resolve().parent.parent
BDR_PIPELINE_YAML = REPO_ROOT / "examples" / "bdr-outreach" / "expected" / "pipeline.yaml"


# ───────── In-process tests ─────────


def test_no_args_prints_help_and_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli_main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "rote" in out
    assert "graduate" in out  # subcommand should be in help
    assert "emit" in out


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    from rote import __version__

    with pytest.raises(SystemExit) as excinfo:
        cli_main(["--version"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert __version__ in out


def test_emit_missing_pipeline_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli_main(
        [
            "emit",
            str(tmp_path / "does-not-exist.yaml"),
            "--runtime",
            "temporal",
            "--out",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "not found" in err


def test_emit_bdr_pipeline_in_process(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Run `rote emit` against the real BDR pipeline.yaml.

    Verifies: the CLI loads the pipeline, dispatches to the temporal
    adapter, writes the expected files, and the emitted files parse as
    valid Python.
    """
    out_dir = tmp_path / "emitted"

    rc = cli_main(
        [
            "emit",
            str(BDR_PIPELINE_YAML),
            "--runtime",
            "temporal",
            "--out",
            str(out_dir),
        ]
    )
    assert rc == 0

    captured = capsys.readouterr().out
    assert "bdr-campaign" in captured
    assert str(out_dir) in captured

    # Files exist
    activities_path = out_dir / "activities.py"
    workflow_path = out_dir / "workflow.py"
    init_path = out_dir / "__init__.py"
    assert activities_path.exists()
    assert workflow_path.exists()
    assert init_path.exists()

    # And parse as valid Python
    ast.parse(activities_path.read_text(encoding="utf-8"))
    ast.parse(workflow_path.read_text(encoding="utf-8"))


def test_emit_json_mode_is_machine_readable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`rote emit --json` writes ONE JSON object to stdout (no human prose)
    so a piping agent can locate the output dir and the stubs to fill in."""
    import json

    out_dir = tmp_path / "emitted"
    rc = cli_main(
        [
            "emit",
            str(BDR_PIPELINE_YAML),
            "--runtime",
            "dbos",
            "--out",
            str(out_dir),
            "--json",
        ]
    )
    assert rc == 0

    captured = capsys.readouterr()
    # stdout is ONLY the JSON object — the human "rote: emitted …" line is gone.
    assert "rote: emitted" not in captured.out
    payload = json.loads(captured.out)
    assert payload["pipeline"] == {"name": "bdr-campaign", "version": "0.1.0"}
    assert payload["runtime"] == "dbos"
    assert payload["out_dir"] == str(out_dir.resolve())
    assert "main" in payload["written"]
    assert payload["preserved_new_files"] == []
    # The extracted stubs are enumerated as the agent's fill-in TODO list.
    stubs = payload["unimplemented_stubs"]
    assert stubs and all(s.endswith(".py") for s in stubs)
    assert any("extracted/hubspot.py" in s for s in stubs)


def test_emit_python_refuses_gated_pipeline_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """BDR has HITL gates: `--runtime python` must exit 1 with the
    adapter's actionable message on stderr — not a traceback."""
    rc = cli_main(
        [
            "emit",
            str(BDR_PIPELINE_YAML),
            "--runtime",
            "python",
            "--out",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "cannot durably park" in err
    assert "--runtime dbos" in err
    assert not (tmp_path / "out").exists()


def test_emit_rejects_unknown_runtime(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # argparse rejects unknown choices with exit code 2
    with pytest.raises(SystemExit) as excinfo:
        cli_main(
            [
                "emit",
                str(BDR_PIPELINE_YAML),
                "--runtime",
                "nonexistent",
                "--out",
                str(tmp_path / "out"),
            ]
        )
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "nonexistent" in err or "invalid choice" in err.lower()


def test_graduate_rejects_nonexistent_skill_dir(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = cli_main(
        [
            "graduate",
            str(tmp_path / "nonexistent"),
            "--runtime",
            "temporal",
            "--out",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "not a directory" in err


def _install_mock_graduator(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch ``rote.cli.Graduator`` with a fake that writes the BDR IR.

    Shared by the graduate and analyze tests — the real agent loop never
    runs; the fake writes a copy of the BDR pipeline.yaml (known-valid,
    proven by the IR tests) into the output dir and returns a
    GraduationResult wrapping the loaded IR.

    When a live-progress sink was wired (``on_event``, as ``rote graduate``
    always does), the fake fires a representative phase event and a
    token-carrying turn event through it before returning — enough to
    exercise the JSONL progress sink without a real agent. The returned
    metadata carries the same cumulative token totals a real driver stamps.
    """
    from rote.graduator import GraduationResult
    from rote.graduator.events import GraduationEvent
    from rote.ir import load_pipeline

    real_pipeline = load_pipeline(BDR_PIPELINE_YAML)

    class _MockGraduator:
        def __init__(self, agent: str | None = None, **kwargs: object) -> None:
            self.agent = agent
            self.on_event = kwargs.get("on_event")

        async def graduate(self, skill_path, output_dir, update=False, **_kwargs):  # noqa: ANN001
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "pipeline.yaml").write_text(
                BDR_PIPELINE_YAML.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            on_event = self.on_event
            if callable(on_event):
                on_event(
                    GraduationEvent(
                        type="phase", ts=0.0, phase=1, phase_name="Intake", message="phase 1"
                    )
                )
                on_event(
                    GraduationEvent(
                        type="turn",
                        ts=0.0,
                        turn=1,
                        tokens={"input": 1000, "output": 500},
                        message="turn 1: working",
                    )
                )
            return GraduationResult(
                pipeline=real_pipeline,
                output_dir=output_dir,
                driver_name="mock",
                driver_metadata={"num_turns": 7, "input_tokens": 1000, "output_tokens": 500},
            )

    monkeypatch.setattr("rote.cli.Graduator", _MockGraduator)


def test_analyze_rejects_nonexistent_skill_dir(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = cli_main(["analyze", str(tmp_path / "nonexistent")])
    assert rc == 2
    err = capsys.readouterr().err
    assert "not a directory" in err


def test_analyze_reports_pipeline_shape_without_emitting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`analyze` runs the graduator, prints a structural report, and emits
    no runtime code. Without --out the graduated artifacts are discarded."""
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: t\n---\n")

    _install_mock_graduator(monkeypatch)

    rc = cli_main(["analyze", str(skill_dir)])
    assert rc == 0

    out = capsys.readouterr().out
    assert "bdr-campaign" in out
    assert "Roteness: 77%" in out  # 10 of 13 top-level steps deterministic
    assert "HITL gates:" in out
    # python must be flagged unavailable (BDR has hitl gates)
    assert "python: unavailable" in out
    # report-only: no runtime code written anywhere under tmp_path
    assert not list(tmp_path.rglob("main.py"))
    assert not list(tmp_path.rglob("workflow.py"))
    # and the hint to keep artifacts is shown
    assert "--out" in out


def test_analyze_json_mode_is_machine_readable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import json

    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: t\n---\n")

    _install_mock_graduator(monkeypatch)

    rc = cli_main(["analyze", str(skill_dir), "--json"])
    assert rc == 0

    report = json.loads(capsys.readouterr().out)
    assert report["pipeline"] == "bdr-campaign"
    assert report["nodes"]["total"] == 13
    assert report["untargetable_runtimes"]["python"]
    assert "temporal" in report["targetable_runtimes"]


def test_analyze_out_keeps_pipeline_yaml(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """With --out, the graduated IR survives for a later `rote emit`."""
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: t\n---\n")
    out_dir = tmp_path / "analysis"

    _install_mock_graduator(monkeypatch)

    rc = cli_main(["analyze", str(skill_dir), "--out", str(out_dir)])
    assert rc == 0
    assert (out_dir / "pipeline.yaml").is_file()
    out = capsys.readouterr().out
    assert "graduated IR kept at" in out
    assert "rote emit" in out


def test_analyze_surfaces_graduator_error_with_exit_1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: t\n---\n")

    from rote.cli import GraduatorError

    class _ExplodingGraduator:
        def __init__(self, agent: str | None = None, **kwargs: object) -> None:
            pass

        async def graduate(self, skill_path, output_dir, update=False, **_kwargs):  # noqa: ANN001
            raise GraduatorError("simulated failure: no agent driver available")

    monkeypatch.setattr("rote.cli.Graduator", _ExplodingGraduator)

    rc = cli_main(["analyze", str(skill_dir)])
    assert rc == 1
    assert "simulated failure" in capsys.readouterr().err


def test_graduate_happy_path_with_mocked_graduator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI should run the graduator, emit the adapter output, and
    print a clean summary. We mock ``Graduator`` so the real agent
    loop never runs.
    """
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: t\n---\n")

    out_dir = tmp_path / "out"

    # The fake graduator will be called by the CLI; have it write the
    # BDR pipeline.yaml as a stand-in (we know it's valid because the
    # IR tests prove it loads). Then the real adapter runs against
    # that valid IR and emits real Temporal code.
    from rote.graduator import GraduationResult
    from rote.ir import load_pipeline

    real_pipeline = load_pipeline(BDR_PIPELINE_YAML)

    class _MockGraduator:
        def __init__(self, agent: str | None = None, **kwargs: object) -> None:
            self.agent = agent

        async def graduate(self, skill_path, output_dir, update=False, **_kwargs):  # noqa: ANN001
            # Write a placeholder pipeline.yaml in the output dir so
            # downstream debugging is possible (mimics what a real run
            # produces). Use a copy of the BDR yaml for realism.
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "pipeline.yaml").write_text(
                BDR_PIPELINE_YAML.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            return GraduationResult(
                pipeline=real_pipeline,
                output_dir=output_dir,
                driver_name="mock",
                driver_metadata={"tokens": 1234, "iterations": 7},
            )

    monkeypatch.setattr("rote.cli.Graduator", _MockGraduator)

    rc = cli_main(
        [
            "graduate",
            str(skill_dir),
            "--runtime",
            "temporal",
            "--out",
            str(out_dir),
        ]
    )
    assert rc == 0

    captured = capsys.readouterr().out
    assert "bdr-campaign" in captured
    assert "mock" in captured  # driver name
    assert "tokens=1234" in captured

    # Graduator output (just the pipeline.yaml the mock wrote)
    assert (out_dir / "graduated" / "pipeline.yaml").exists()

    # Emitted Temporal runtime code
    assert (out_dir / "runtime" / "temporal" / "workflow.py").exists()
    assert (out_dir / "runtime" / "temporal" / "activities.py").exists()


def test_graduate_json_mode_is_machine_readable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`rote graduate --json` writes ONE JSON superset object to stdout; the
    progress log and summary go to stderr so stdout stays parseable."""
    import json

    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: t\n---\n")

    _install_mock_graduator(monkeypatch)

    out_dir = tmp_path / "out"
    rc = cli_main(
        [
            "graduate",
            str(skill_dir),
            "--runtime",
            "dbos",
            "--out",
            str(out_dir),
            "--no-eval",  # keep the test hermetic (no live price fetch)
            "--json",
        ]
    )
    assert rc == 0

    captured = capsys.readouterr()
    # The human summary went to stderr; stdout is the JSON object only.
    assert "rote graduate: ✓" not in captured.out
    assert "rote graduate: ✓" in captured.err
    payload = json.loads(captured.out)
    assert payload["pipeline"]["name"] == "bdr-campaign"
    assert payload["runtime"] == "dbos"
    assert payload["out_dir"] == str(out_dir.resolve())
    assert payload["graduated_dir"] == str((out_dir / "graduated").resolve())
    assert payload["runtime_dir"] == str((out_dir / "runtime" / "dbos").resolve())
    assert payload["scorecard"] is None  # --no-eval
    assert payload["driver"] == "mock"
    assert "main" in payload["written"]
    # dbos emits extracted/* stubs, so the TODO list is populated.
    assert any("extracted/" in s for s in payload["unimplemented_stubs"])


# ───────── graduate --progress-file (JSONL sink) ─────────


def _patch_pricing(monkeypatch: pytest.MonkeyPatch, prices: tuple[float, float] | None) -> None:
    """Patch ``fetch_catalog`` so the sink prices offline and hermetically.

    ``prices`` = ``(input_per_mtok, output_per_mtok)`` makes ``price_for``
    return that pair; ``None`` makes ``fetch_catalog`` raise ``PricingError``
    (the offline path — the sink must then omit ``cost_usd`` and keep going).
    """
    from rote.eval.pricing import PricingError

    if prices is None:

        def _raise(*_a: object, **_k: object) -> object:
            raise PricingError("offline (test)")

        monkeypatch.setattr("rote.eval.pricing.fetch_catalog", _raise)
        return

    class _FakeCatalog:
        def price_for(self, _model_id: str) -> tuple[float, float]:
            return prices

    monkeypatch.setattr("rote.eval.pricing.fetch_catalog", lambda *a, **k: _FakeCatalog())


def _read_ndjson(path: Path) -> list[dict[str, object]]:
    """Parse a JSONL file, asserting exactly one JSON object per line."""
    import json

    lines = path.read_text(encoding="utf-8").splitlines()
    objs: list[dict[str, object]] = []
    for line in lines:
        assert line.strip(), "progress file must not contain blank lines"
        obj = json.loads(line)  # raises if any line is not valid JSON
        assert isinstance(obj, dict)
        objs.append(obj)
    return objs


def test_graduate_progress_file_writes_ndjson_with_cost_and_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--progress-file streams one JSON object per event, prices the
    token-carrying lines, and ends with a type:summary digest."""
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: t\n---\n")

    _install_mock_graduator(monkeypatch)
    _patch_pricing(monkeypatch, prices=(3.0, 15.0))  # $3 / $15 per Mtok

    out_dir = tmp_path / "out"
    progress = tmp_path / "progress.ndjson"
    rc = cli_main(
        [
            "graduate",
            str(skill_dir),
            "--runtime",
            "dbos",
            "--out",
            str(out_dir),
            "--no-eval",
            "--progress-file",
            str(progress),
        ]
    )
    assert rc == 0

    objs = _read_ndjson(progress)
    types = [o["type"] for o in objs]
    assert "phase" in types
    assert "turn" in types
    # The summary is the LAST line, exactly once.
    assert types[-1] == "summary"
    assert types.count("summary") == 1

    # The token-carrying turn line was priced: 1000/1e6*3 + 500/1e6*15.
    turn = next(o for o in objs if o["type"] == "turn")
    assert turn["tokens"] == {"input": 1000, "output": 500}
    assert turn["cost_usd"] == pytest.approx(0.0105)
    # None-valued fields are dropped for compactness.
    assert "tool_name" not in turn

    summary = objs[-1]
    assert summary["roteness"] == pytest.approx(10 / 13)
    assert summary["nodes"] == 13
    assert isinstance(summary["node_kinds"], dict)
    assert summary["node_kinds"]["hitl_gate"] >= 1
    assert summary["total_tokens"] == {"input": 1000, "output": 500}
    assert summary["cost_usd"] == pytest.approx(0.0105)
    assert any("extracted/" in s for s in summary["unimplemented_stubs"])  # type: ignore[union-attr]
    assert summary["graduated_dir"] == str((out_dir / "graduated").resolve())
    assert summary["runtime_dir"] == str((out_dir / "runtime" / "dbos").resolve())


def test_graduate_progress_file_survives_offline_pricing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An offline price fetch omits cost_usd but never fails the run."""
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: t\n---\n")

    _install_mock_graduator(monkeypatch)
    _patch_pricing(monkeypatch, prices=None)  # fetch_catalog raises PricingError

    out_dir = tmp_path / "out"
    progress = tmp_path / "progress.ndjson"
    rc = cli_main(
        [
            "graduate",
            str(skill_dir),
            "--runtime",
            "dbos",
            "--out",
            str(out_dir),
            "--no-eval",
            "--progress-file",
            str(progress),
        ]
    )
    assert rc == 0

    objs = _read_ndjson(progress)
    assert objs[-1]["type"] == "summary"
    # Not a single line carries a price when the catalog is unreachable.
    assert all("cost_usd" not in o for o in objs)
    # …yet the token figures are still present (pricing is the only casualty).
    turn = next(o for o in objs if o["type"] == "turn")
    assert turn["tokens"] == {"input": 1000, "output": 500}


def test_graduate_progress_file_composes_with_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--progress-file and --json coexist: stdout stays the single JSON
    result object; the NDJSON stream goes only to the file."""
    import json

    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: t\n---\n")

    _install_mock_graduator(monkeypatch)
    _patch_pricing(monkeypatch, prices=(3.0, 15.0))

    out_dir = tmp_path / "out"
    progress = tmp_path / "progress.ndjson"
    rc = cli_main(
        [
            "graduate",
            str(skill_dir),
            "--runtime",
            "dbos",
            "--out",
            str(out_dir),
            "--no-eval",
            "--json",
            "--progress-file",
            str(progress),
        ]
    )
    assert rc == 0

    captured = capsys.readouterr()
    # stdout is still exactly the one --json result object (no NDJSON leaked).
    payload = json.loads(captured.out)
    assert payload["pipeline"]["name"] == "bdr-campaign"
    assert payload["runtime"] == "dbos"

    # The NDJSON stream landed in the file, terminated by the summary.
    objs = _read_ndjson(progress)
    assert objs[-1]["type"] == "summary"


def test_graduate_surfaces_graduator_error_with_exit_1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A GraduatorError from the orchestrator should print a clean
    error and exit 1, not crash."""
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: t\n---\n")

    class _ExplodingGraduator:
        def __init__(self, agent: str | None = None, **kwargs: object) -> None:
            pass

        async def graduate(self, skill_path, output_dir, update=False, **_kwargs):  # noqa: ANN001
            raise GraduatorError("simulated failure: no agent driver available")

    from rote.cli import GraduatorError  # re-imported here for typing visibility

    monkeypatch.setattr("rote.cli.Graduator", _ExplodingGraduator)

    rc = cli_main(
        [
            "graduate",
            str(skill_dir),
            "--runtime",
            "temporal",
            "--out",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "simulated failure" in err


# ───────── doctor ─────────


def test_doctor_json_mode_is_machine_readable(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import json

    monkeypatch.setattr(
        "rote.cli.available_drivers",
        lambda: [("claude", True, ""), ("codex", False, "codex not installed: brew install codex")],
    )

    rc = cli_main(["doctor", "--json"])
    assert rc == 0

    report = json.loads(capsys.readouterr().out)
    assert set(report) == {
        "version",
        "python",
        "drivers",
        "runtimes",
        "mcp_servers",
        "apps",
        "ok",
    }
    assert report["ok"] is True
    assert report["drivers"] == [
        {"name": "claude", "available": True, "reason": ""},
        {"name": "codex", "available": False, "reason": "codex not installed: brew install codex"},
    ]
    # runtimes report install status for each optional extra
    assert {r["name"] for r in report["runtimes"]} == {
        "temporal",
        "api",
        "openai-api",
        "dbos",
        "serve",
        "mcp",
    }
    assert all(isinstance(r["installed"], bool) for r in report["runtimes"])


def test_doctor_exits_1_when_no_driver_available(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "rote.cli.available_drivers",
        lambda: [
            ("claude", False, "claude CLI not found"),
            ("codex", False, "codex not installed"),
            ("api", False, "ANTHROPIC_API_KEY not set"),
        ],
    )

    rc = cli_main(["doctor", "--json"])
    assert rc == 1

    import json

    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is False


def test_doctor_reports_mcp_servers_and_apps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The mcp_servers section covers every auth state and apps flags stale dirs.

    The autouse conftest fixture already points ROTE_MCP_CONFIG /
    ROTE_MCP_TOKEN_DIR / ROTE_APPS_PATH at per-test tmp dirs, so populating
    them via the real stores is hermetic.
    """
    import json
    import time

    from rote.app_registry import record_app
    from rote.mcp import save_registry
    from rote.mcp.registry import McpRegistry, McpServerConfig
    from rote.mcp.tokens import write_token_file

    def _token(*, expires_at: float | None, refresh: bool) -> dict[str, object]:
        tokens: dict[str, object] = {"access_token": "tok", "token_type": "Bearer"}
        if refresh:
            tokens["refresh_token"] = "ref"
        return {"server_url": "https://x.example/mcp", "tokens": tokens, "expires_at": expires_at}

    save_registry(
        McpRegistry(
            servers={
                "keyd": McpServerConfig(url="https://k.example/mcp", headers={"X-K": "v"}),
                "fresh": McpServerConfig(url="https://f.example/mcp"),
                "noauth": McpServerConfig(url="https://n.example/mcp"),
                "refreshable": McpServerConfig(url="https://r.example/mcp"),
                "dead": McpServerConfig(url="https://d.example/mcp"),
            }
        )
    )
    write_token_file("fresh", _token(expires_at=None, refresh=False))
    write_token_file("refreshable", _token(expires_at=time.time() - 10, refresh=True))
    write_token_file("dead", _token(expires_at=time.time() - 10, refresh=False))

    live_app = tmp_path / "live-app"
    live_app.mkdir()
    record_app(live_app, "dbos", "live_pipeline")
    record_app(tmp_path / "gone-app", "temporal", "stale_pipeline")

    # Pin driver availability so rc is deterministic regardless of whether a
    # claude/codex CLI happens to be on PATH — this test exercises the
    # mcp_servers/apps sections, not driver detection.
    monkeypatch.setattr("rote.cli.available_drivers", lambda: [("claude", True, "")])

    rc = cli_main(["doctor", "--json"])
    assert rc == 0

    report = json.loads(capsys.readouterr().out)
    by_name = {s["name"]: s["auth"] for s in report["mcp_servers"]}
    assert by_name == {
        "keyd": "static headers",
        "fresh": "authenticated",
        "noauth": "not authenticated",
        "refreshable": "expired (refreshable)",
        "dead": "expired",
    }

    apps = {a["pipeline"]: a["exists"] for a in report["apps"]}
    assert apps == {"live_pipeline": True, "stale_pipeline": False}


def test_doctor_human_mode_renders_checklist_with_driver_reason(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "rote.cli.available_drivers",
        lambda: [("claude", True, ""), ("codex", False, "codex not installed: brew install codex")],
    )

    rc = cli_main(["doctor"])
    assert rc == 0

    out = capsys.readouterr().out
    assert "✓ claude" in out
    assert "✗ codex" in out
    assert "codex not installed: brew install codex" in out  # reason shown inline
    assert "ready to graduate" in out
    assert "only verified at run time" in out  # the CLI-subscription caveat


def test_doctor_degrades_gracefully_with_no_registry_or_apps(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No registry, no token dir, no apps file (the conftest tmp paths don't
    exist yet) — the sections are empty lists, never a traceback."""
    import json

    rc = cli_main(["doctor", "--json"])
    assert rc in (0, 1)  # depends on the real drivers; either way no crash
    report = json.loads(capsys.readouterr().out)
    assert report["mcp_servers"] == []
    assert report["apps"] == []


# ───────── Subprocess tests ─────────


@pytest.fixture(scope="module")
def python_executable() -> str:
    return sys.executable


def test_subprocess_python_m_rote_cli(python_executable: str) -> None:
    """Invoke `python -m rote.cli --version` and verify it runs.

    This catches import-time errors, missing entry points, and other
    packaging issues that in-process tests miss.
    """
    result = subprocess.run(
        [python_executable, "-m", "rote.cli", "--version"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"exit={result.returncode}\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "rote" in result.stdout


def test_subprocess_emit_bdr(python_executable: str, tmp_path: Path) -> None:
    """End-to-end subprocess test: `python -m rote.cli emit ...` on BDR."""
    out_dir = tmp_path / "emitted"
    result = subprocess.run(
        [
            python_executable,
            "-m",
            "rote.cli",
            "emit",
            str(BDR_PIPELINE_YAML),
            "--runtime",
            "temporal",
            "--out",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"exit={result.returncode}\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    assert (out_dir / "workflow.py").exists()
    assert (out_dir / "activities.py").exists()


# ───────── graduate live-progress printer ─────────


def test_graduate_progress_printer_renders_expected_lines(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from rote.cli import _graduate_progress_printer
    from rote.graduator.events import GraduationEvent

    printer = _graduate_progress_printer()
    for event in [
        GraduationEvent(type="log", ts=0.0, message="driver selected: api"),
        GraduationEvent(type="phase", ts=0.0, phase=4, phase_name="LLM-Judge Extraction"),
        GraduationEvent(type="turn", ts=0.0, turn=12, message="turn 12: thinking…"),
        GraduationEvent(
            type="tool",
            ts=0.0,
            turn=12,
            tool_name="write_file",
            path="signatures/qualify.ts",
        ),
        GraduationEvent(type="warning", ts=0.0, message="a warning"),
        GraduationEvent(type="complete", ts=0.0, message="graduated via api; tokens in=1 out=2"),
    ]:
        printer(event)

    err = capsys.readouterr().err
    assert "[phase 4/7] LLM-Judge Extraction" in err
    assert "[turn 12] write_file signatures/qualify.ts" in err
    assert "warning: a warning" in err
    assert "graduated via api; tokens in=1 out=2" in err
    # Bare turn events are suppressed to keep the stream readable.
    assert "thinking" not in err
    # log lines print verbatim.
    assert "driver selected: api" in err


def test_graduate_progress_printer_annotates_tool_lines_with_tokens(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A turn event's cumulative tokens ride onto the tool lines that follow
    it — the token figures live on turn events, but the tool lines are what
    the printer actually shows."""
    from rote.cli import _graduate_progress_printer
    from rote.graduator.events import GraduationEvent

    printer = _graduate_progress_printer()
    for event in [
        GraduationEvent(type="turn", ts=0.0, turn=7, tokens={"input": 40234, "output": 8100}),
        GraduationEvent(type="tool", ts=0.0, turn=7, tool_name="Write", path="signatures/foo.py"),
    ]:
        printer(event)

    err = capsys.readouterr().err
    assert "[turn 7] Write signatures/foo.py (in 40.2k / out 8.1k tok)" in err


# ───────── MCP requirements surfacing (analyze / graduate / emit) ─────────

DEAL_MONITOR_PIPELINE_YAML = REPO_ROOT / "examples" / "deal-monitor" / "expected" / "pipeline.yaml"


def _isolate_mcp_stores(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the MCP registry + token store at hermetic tmp locations.

    Without this, requirement-surfacing tests would read the developer's
    real ~/.config/rote/mcp.json and report whatever they have logged in.
    Returns the registry path (absent until a test writes it).
    """
    registry = tmp_path / "mcp.json"
    monkeypatch.setenv("ROTE_MCP_CONFIG", str(registry))
    monkeypatch.setenv("ROTE_MCP_TOKEN_DIR", str(tmp_path / "tokens"))
    return registry


def test_mcp_requirements_classify_registered_and_unregistered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """slack registered but tokenless → not authenticated; gmail unknown →
    not registered. Node ids and tool names ride along for the report."""
    import json

    from rote.cli import _mcp_requirements
    from rote.ir import load_pipeline

    registry = _isolate_mcp_stores(monkeypatch, tmp_path)
    registry.write_text(
        json.dumps({"version": 1, "servers": {"slack": {"url": "https://slack.example/mcp"}}}),
        encoding="utf-8",
    )

    report = _mcp_requirements(load_pipeline(DEAL_MONITOR_PIPELINE_YAML))
    by_server = {entry["server"]: entry for entry in report}
    assert set(by_server) == {"gmail", "slack"}
    assert by_server["slack"]["auth"] == "not authenticated"
    assert by_server["gmail"]["auth"] == "not registered"
    assert by_server["slack"]["nodes"] == ["fetch_intake_messages"]
    assert by_server["slack"]["tools"] == ["slack_read_channel"]
    assert by_server["gmail"]["tools"] == ["get_thread", "search_threads"]


def test_mcp_requirements_empty_pipeline_skips_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No mcp: bindings → empty manifest, registry never consulted."""
    from rote.cli import _mcp_requirements
    from rote.ir import load_pipeline

    monkeypatch.setattr(
        "rote.mcp.load_registry",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("registry read")),
    )
    assert _mcp_requirements(load_pipeline(BDR_PIPELINE_YAML)) == []


def test_emit_surfaces_mcp_requirements_with_login_nudge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Human emit output lists required servers and the non-blocking
    `rote mcp login` recommendation for servers that would park a run."""
    _isolate_mcp_stores(monkeypatch, tmp_path)

    rc = cli_main(
        [
            "emit",
            str(DEAL_MONITOR_PIPELINE_YAML),
            "--runtime",
            "dbos",
            "--out",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "required MCP servers:" in out
    assert "gmail [not registered]" in out
    assert "slack [not registered]" in out
    assert "rote mcp add gmail" in out
    assert "rote mcp login gmail" in out


def test_emit_json_includes_mcp_servers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    _isolate_mcp_stores(monkeypatch, tmp_path)

    rc = cli_main(
        [
            "emit",
            str(DEAL_MONITOR_PIPELINE_YAML),
            "--runtime",
            "dbos",
            "--out",
            str(tmp_path / "out"),
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    servers = {entry["server"]: entry for entry in payload["mcp_servers"]}
    assert servers["gmail"]["auth"] == "not registered"
    assert servers["gmail"]["nodes"] == [
        "fetch_email_thread_content",
        "search_gmail_by_accounts",
        "search_gmail_standard",
    ]


def test_analyze_json_includes_empty_mcp_manifest_for_unbound_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """BDR's expected IR has no mcp: bindings → manifest present but empty,
    so agents can distinguish 'no requirements' from 'field missing'."""
    import json

    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: t\n---\n")

    _install_mock_graduator(monkeypatch)

    rc = cli_main(["analyze", str(skill_dir), "--json"])
    assert rc == 0
    report = json.loads(capsys.readouterr().out)
    assert report["mcp_servers"] == []
