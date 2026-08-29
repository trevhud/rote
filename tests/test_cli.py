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
import json
import subprocess
import sys
from pathlib import Path

import pytest

from rote.cli import _JsonlProgressSink as _SinkForCapture
from rote.cli import main as cli_main

REPO_ROOT = Path(__file__).resolve().parent.parent
BDR_PIPELINE_YAML = REPO_ROOT / "examples" / "bdr-outreach" / "expected" / "pipeline.yaml"


# ───────── In-process tests ─────────


def test_no_args_prints_help_and_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli_main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "rote" in out
    assert "compile" in out  # subcommand should be in help
    assert "emit" in out


def test_help_does_not_advertise_the_retired_verb(capsys: pytest.CaptureFixture[str]) -> None:
    """`graduate` still runs but must not be offered as a second spelling."""
    rc = cli_main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "graduate" not in out.lower()


def test_retired_graduate_alias_dispatches_to_compile(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`rote graduate` keeps working for scripts written against 0.10.x.

    Asserted through the same "no such directory" error the real command
    gives, which proves the alias reached `_cmd_compile`'s validation rather
    than dying in argparse as an unknown subcommand (that would exit 2 with a
    usage message instead).
    """
    missing = tmp_path / "nope"
    rc = cli_main(["graduate", str(missing), "--out", str(tmp_path / "o")])
    captured = capsys.readouterr()
    assert rc == 2
    assert "is now `compile`" in captured.err
    assert str(missing) in captured.err


def test_retired_alias_ignores_a_skill_directory_named_graduate(tmp_path: Path) -> None:
    """Only the subcommand slot is rewritten, never a positional argument.

    A skill directory called `graduate` is a plausible thing to have lying
    around after the rename, and rewriting it would send the compiler at a
    path the user never named.
    """
    from rote.cli import _resolve_retired_command

    assert _resolve_retired_command(["compile", "./graduate", "--out", "o"]) == [
        "compile",
        "./graduate",
        "--out",
        "o",
    ]


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


def test_compile_rejects_nonexistent_skill_dir(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = cli_main(
        [
            "compile",
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


_MCP_FREE_PIPELINE = """\
name: mcp-free
version: 0.1.0
source_skill: tests/fixtures/mcp-free
description: A pipeline that needs no MCP server at all.
input:
  type: In
  required: [topic]
  optional: []
nodes:
  - id: normalize
    kind: pure_function
    description: Normalize the topic.
    impl: extracted/brief.py:normalize
    inputs:
      topic: pipeline.input.topic
edges: []
entry_nodes: [normalize]
exit_nodes: [normalize]
"""


def _write_mcp_free_pipeline(tmp_path: Path) -> Path:
    """A pipeline with no MCP requirement of ANY kind.

    BDR used to serve this role, but its agent loops now resolve their
    tools to servers — so BDR requires five. Reusing it here would have
    quietly re-asserted the bug this field exists to fix.
    """
    path = tmp_path / "mcp-free.yaml"
    path.write_text(_MCP_FREE_PIPELINE, encoding="utf-8")
    return path


def _install_mock_compiler(
    monkeypatch: pytest.MonkeyPatch, pipeline_yaml: Path | None = None
) -> None:
    """Patch ``rote.cli.Compiler`` with a fake that writes the BDR IR.

    Shared by the compile and analyze tests — the real agent loop never
    runs; the fake writes a copy of the BDR pipeline.yaml (known-valid,
    proven by the IR tests) into the output dir and returns a
    CompilationResult wrapping the loaded IR.

    When a live-progress sink was wired (``on_event``, as ``rote compile``
    always does), the fake fires a representative phase event and a
    token-carrying turn event through it before returning — enough to
    exercise the JSONL progress sink without a real agent. The returned
    metadata carries the same cumulative token totals a real driver stamps.
    """
    from rote.compiler import CompilationResult
    from rote.compiler.events import CompilationEvent
    from rote.ir import load_pipeline

    real_pipeline = load_pipeline(pipeline_yaml or BDR_PIPELINE_YAML)

    class _MockCompiler:
        def __init__(self, agent: str | None = None, **kwargs: object) -> None:
            self.agent = agent
            self.on_event = kwargs.get("on_event")

        async def compile(self, skill_path, output_dir, update=False, **_kwargs):  # noqa: ANN001
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "pipeline.yaml").write_text(
                BDR_PIPELINE_YAML.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            on_event = self.on_event
            if callable(on_event):
                on_event(
                    CompilationEvent(
                        type="phase", ts=0.0, phase=1, phase_name="Intake", message="phase 1"
                    )
                )
                on_event(
                    CompilationEvent(
                        type="turn",
                        ts=0.0,
                        turn=1,
                        tokens={"input": 1000, "output": 500},
                        message="turn 1: working",
                    )
                )
            return CompilationResult(
                pipeline=real_pipeline,
                output_dir=output_dir,
                driver_name="mock",
                driver_metadata={"num_turns": 7, "input_tokens": 1000, "output_tokens": 500},
            )

    monkeypatch.setattr("rote.cli.Compiler", _MockCompiler)


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
    """`analyze` runs the compiler, prints a structural report, and emits
    no runtime code. Without --out the compiled artifacts are discarded."""
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: t\n---\n")

    _install_mock_compiler(monkeypatch)

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

    _install_mock_compiler(monkeypatch)

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
    """With --out, the compiled IR survives for a later `rote emit`."""
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: t\n---\n")
    out_dir = tmp_path / "analysis"

    _install_mock_compiler(monkeypatch)

    rc = cli_main(["analyze", str(skill_dir), "--out", str(out_dir)])
    assert rc == 0
    assert (out_dir / "pipeline.yaml").is_file()
    out = capsys.readouterr().out
    assert "compiled IR kept at" in out
    assert "rote emit" in out


def test_analyze_surfaces_compiler_error_with_exit_1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: t\n---\n")

    from rote.cli import CompilerError

    class _ExplodingCompiler:
        def __init__(self, agent: str | None = None, **kwargs: object) -> None:
            pass

        async def compile(self, skill_path, output_dir, update=False, **_kwargs):  # noqa: ANN001
            raise CompilerError("simulated failure: no agent driver available")

    monkeypatch.setattr("rote.cli.Compiler", _ExplodingCompiler)

    rc = cli_main(["analyze", str(skill_dir)])
    assert rc == 1
    assert "simulated failure" in capsys.readouterr().err


def test_compile_happy_path_with_mocked_compiler(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI should run the compiler, emit the adapter output, and
    print a clean summary. We mock ``Compiler`` so the real agent
    loop never runs.
    """
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: t\n---\n")

    out_dir = tmp_path / "out"

    # The fake compiler will be called by the CLI; have it write the
    # BDR pipeline.yaml as a stand-in (we know it's valid because the
    # IR tests prove it loads). Then the real adapter runs against
    # that valid IR and emits real Temporal code.
    from rote.compiler import CompilationResult
    from rote.ir import load_pipeline

    real_pipeline = load_pipeline(BDR_PIPELINE_YAML)

    class _MockCompiler:
        def __init__(self, agent: str | None = None, **kwargs: object) -> None:
            self.agent = agent

        async def compile(self, skill_path, output_dir, update=False, **_kwargs):  # noqa: ANN001
            # Write a placeholder pipeline.yaml in the output dir so
            # downstream debugging is possible (mimics what a real run
            # produces). Use a copy of the BDR yaml for realism.
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "pipeline.yaml").write_text(
                BDR_PIPELINE_YAML.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            return CompilationResult(
                pipeline=real_pipeline,
                output_dir=output_dir,
                driver_name="mock",
                driver_metadata={"tokens": 1234, "iterations": 7},
            )

    monkeypatch.setattr("rote.cli.Compiler", _MockCompiler)

    rc = cli_main(
        [
            "compile",
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

    # Compiler output (just the pipeline.yaml the mock wrote)
    assert (out_dir / "compiled" / "pipeline.yaml").exists()

    # Emitted Temporal runtime code
    assert (out_dir / "runtime" / "temporal" / "workflow.py").exists()
    assert (out_dir / "runtime" / "temporal" / "activities.py").exists()


def test_compile_json_mode_is_machine_readable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`rote compile --json` writes ONE JSON superset object to stdout; the
    progress log and summary go to stderr so stdout stays parseable."""
    import json

    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: t\n---\n")

    _install_mock_compiler(monkeypatch)

    out_dir = tmp_path / "out"
    rc = cli_main(
        [
            "compile",
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
    assert "rote compile: ✓" not in captured.out
    assert "rote compile: ✓" in captured.err
    payload = json.loads(captured.out)
    assert payload["pipeline"]["name"] == "bdr-campaign"
    assert payload["runtime"] == "dbos"
    assert payload["out_dir"] == str(out_dir.resolve())
    assert payload["compiled_dir"] == str((out_dir / "compiled").resolve())
    assert payload["runtime_dir"] == str((out_dir / "runtime" / "dbos").resolve())
    assert payload["scorecard"] is None  # --no-eval
    assert payload["driver"] == "mock"
    assert "main" in payload["written"]
    # dbos emits extracted/* stubs, so the TODO list is populated.
    assert any("extracted/" in s for s in payload["unimplemented_stubs"])


# ───────── compile × rote-cloud login default ─────────


def _cloud_logged_in() -> None:
    from rote.cloud_auth import CloudCredential, save_credential

    save_credential(CloudCredential(url="http://p", token="rote_k", user="t@x"))


def _fake_cloud_deploy(monkeypatch: pytest.MonkeyPatch, *, fail: bool = False) -> list:
    from rote.deploy import DeployError, DeployReport

    calls: list = []

    def fake(target, *, url=None, token=None, input_example=None):  # noqa: ANN001
        calls.append(target)
        if fail:
            raise DeployError("upload failed: HTTP 500")
        return DeployReport(
            target="rote-cloud",
            runtime="cloudflare",
            app_dir=target.path,
            ok=True,
            action="deployed",
            detail='{"pipeline_id": "p1"}',
        )

    monkeypatch.setattr("rote.deploy_rote_cloud.deploy_rote_cloud", fake)
    return calls


def _compile(skill_dir: Path, out_dir: Path, *extra: str) -> int:
    return cli_main(["compile", str(skill_dir), "--out", str(out_dir), "--no-eval", *extra])


def test_compile_logged_in_defaults_to_cloudflare_and_deploys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: t\n---\n")
    _install_mock_compiler(monkeypatch)
    _cloud_logged_in()
    calls = _fake_cloud_deploy(monkeypatch)

    # --local keeps the compilation on this machine; logged in, it still
    # emits cloudflare and uploads (the local auto-deploy leg). The no-flag
    # logged-in default now runs the whole compilation on rote cloud instead
    # (see tests/test_cloud_compile.py).
    rc = _compile(skill_dir, tmp_path / "out", "--local")
    assert rc == 0
    assert (tmp_path / "out" / "runtime" / "cloudflare" / "src" / "workflow.ts").exists()
    assert len(calls) == 1
    assert calls[0].path == tmp_path / "out" / "runtime" / "cloudflare"
    assert "hosted on rote cloud: http://p" in capsys.readouterr().out


def test_compile_no_deploy_still_emits_cloudflare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: t\n---\n")
    _install_mock_compiler(monkeypatch)
    _cloud_logged_in()
    calls = _fake_cloud_deploy(monkeypatch)

    rc = _compile(skill_dir, tmp_path / "out", "--no-deploy")
    assert rc == 0
    assert (tmp_path / "out" / "runtime" / "cloudflare").is_dir()
    assert calls == []


def test_compile_logged_out_falls_back_to_local_with_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The silent local fallback: no prompt, dbos default, one tip line."""
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: t\n---\n")
    _install_mock_compiler(monkeypatch)
    calls = _fake_cloud_deploy(monkeypatch)

    rc = _compile(skill_dir, tmp_path / "out")
    assert rc == 0
    assert (tmp_path / "out" / "runtime" / "dbos" / "main.py").exists()
    assert calls == []
    assert "`rote login` makes rote cloud the default host" in capsys.readouterr().err


def test_compile_explicit_runtime_wins_over_login(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: t\n---\n")
    _install_mock_compiler(monkeypatch)
    _cloud_logged_in()
    calls = _fake_cloud_deploy(monkeypatch)

    # An explicit non-cloudflare runtime is itself a local opt-out (no
    # --local needed): temporal isn't cloud-runnable, so the run stays local.
    rc = _compile(skill_dir, tmp_path / "out", "--runtime", "temporal")
    assert rc == 0
    assert (tmp_path / "out" / "runtime" / "temporal" / "workflow.py").exists()
    assert calls == []  # temporal isn't cloud-runnable; no upload attempted


def test_compile_deploy_failure_keeps_artifacts_and_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: t\n---\n")
    _install_mock_compiler(monkeypatch)
    _cloud_logged_in()
    _fake_cloud_deploy(monkeypatch, fail=True)

    out_dir = tmp_path / "out"
    rc = _compile(skill_dir, out_dir, "--local")
    assert rc == 1
    err = capsys.readouterr().err
    assert "deploy to rote cloud failed" in err
    assert "rote deploy" in err  # retry instructions
    # the paid-for artifacts survived the failed upload
    assert (out_dir / "compiled" / "pipeline.yaml").exists()
    assert (out_dir / "runtime" / "cloudflare").is_dir()


# ───────── compile --progress-file (JSONL sink) ─────────


def _patch_pricing(
    monkeypatch: pytest.MonkeyPatch,
    prices: tuple[float, float] | None,
    cache_read_per_mtok: float | None = None,
    cache_write_per_mtok: float | None = None,
) -> None:
    """Patch the sink's price resolution so it runs hermetically.

    ``prices`` = ``(input_per_mtok, output_per_mtok)`` is what the sink
    resolves; ``None`` is the offline path, where the sink must omit
    ``cost_usd`` and keep going. Patches ``_resolve_prices`` (not
    ``fetch_catalog``) because the suite-wide conftest fixture already
    stubs ``_resolve_prices`` to None; a test overriding pricing must
    patch the same seam to win.

    The cache rates default to ``None`` because that is the common real
    case: ``reference_prices`` covers every model but carries only the
    input/output pair, so only a tier representative has cache rates.
    """
    from rote.cli import _Rates

    rates = (
        None
        if prices is None
        else _Rates(
            input_per_mtok=prices[0],
            output_per_mtok=prices[1],
            cache_read_per_mtok=cache_read_per_mtok,
            cache_write_per_mtok=cache_write_per_mtok,
        )
    )
    monkeypatch.setattr(
        "rote.cli._JsonlProgressSink._resolve_prices",
        staticmethod(lambda _model_id: rates),
    )


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


def test_compile_progress_file_writes_ndjson_with_cost_and_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--progress-file streams one JSON object per event, prices the
    token-carrying lines, and ends with a type:summary digest."""
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: t\n---\n")

    _install_mock_compiler(monkeypatch)
    _patch_pricing(monkeypatch, prices=(3.0, 15.0))  # $3 / $15 per Mtok

    out_dir = tmp_path / "out"
    progress = tmp_path / "progress.ndjson"
    rc = cli_main(
        [
            "compile",
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
    assert summary["total_tokens"] == {
        "input": 1000,
        "output": 500,
        "cache_write": 0,
        "cache_read": 0,
    }
    assert summary["cost_usd"] == pytest.approx(0.0105)
    assert any("extracted/" in s for s in summary["unimplemented_stubs"])  # type: ignore[union-attr]
    assert summary["compiled_dir"] == str((out_dir / "compiled").resolve())
    assert summary["runtime_dir"] == str((out_dir / "runtime" / "dbos").resolve())


def test_compile_writes_progress_sidecar_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without --progress-file the sink still runs, writing
    <out>/progress.jsonl, and the watch hint is printed first thing —
    that's how a user observes a compilation an agent is driving."""
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: t\n---\n")

    _install_mock_compiler(monkeypatch)

    out_dir = tmp_path / "out"
    rc = cli_main(
        ["compile", str(skill_dir), "--runtime", "dbos", "--out", str(out_dir), "--no-eval"]
    )
    assert rc == 0

    sidecar = out_dir / "progress.jsonl"
    objs = _read_ndjson(sidecar)
    assert objs[-1]["type"] == "summary"
    assert f"watch progress with: tail -f {sidecar}" in capsys.readouterr().err


def test_compile_progress_file_survives_offline_pricing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An offline price fetch omits cost_usd but never fails the run."""
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: t\n---\n")

    _install_mock_compiler(monkeypatch)
    _patch_pricing(monkeypatch, prices=None)  # fetch_catalog raises PricingError

    out_dir = tmp_path / "out"
    progress = tmp_path / "progress.ndjson"
    rc = cli_main(
        [
            "compile",
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


def test_compile_progress_file_composes_with_json(
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

    _install_mock_compiler(monkeypatch)
    _patch_pricing(monkeypatch, prices=(3.0, 15.0))

    out_dir = tmp_path / "out"
    progress = tmp_path / "progress.ndjson"
    rc = cli_main(
        [
            "compile",
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


def test_compile_surfaces_compiler_error_with_exit_1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A CompilerError from the orchestrator should print a clean
    error and exit 1, not crash."""
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: t\n---\n")

    class _ExplodingCompiler:
        def __init__(self, agent: str | None = None, **kwargs: object) -> None:
            pass

        async def compile(self, skill_path, output_dir, update=False, **_kwargs):  # noqa: ANN001
            raise CompilerError("simulated failure: no agent driver available")

    from rote.cli import CompilerError  # re-imported here for typing visibility

    monkeypatch.setattr("rote.cli.Compiler", _ExplodingCompiler)

    rc = cli_main(
        [
            "compile",
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
    assert "ready to compile" in out
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


# ───────── compile live-progress printer ─────────


def test_compile_progress_printer_renders_expected_lines(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from rote.cli import _compile_progress_printer
    from rote.compiler.events import CompilationEvent

    printer = _compile_progress_printer()
    for event in [
        CompilationEvent(type="log", ts=0.0, message="driver selected: api"),
        CompilationEvent(type="phase", ts=0.0, phase=4, phase_name="LLM-Judge Extraction"),
        CompilationEvent(type="turn", ts=0.0, turn=12, message="turn 12: thinking…"),
        CompilationEvent(
            type="tool",
            ts=0.0,
            turn=12,
            tool_name="write_file",
            path="signatures/qualify.ts",
        ),
        CompilationEvent(type="warning", ts=0.0, message="a warning"),
        CompilationEvent(type="complete", ts=0.0, message="compiled via api; tokens in=1 out=2"),
    ]:
        printer(event)

    err = capsys.readouterr().err
    assert "[phase 4/7] LLM-Judge Extraction" in err
    assert "[turn 12] write_file signatures/qualify.ts" in err
    assert "warning: a warning" in err
    assert "compiled via api; tokens in=1 out=2" in err
    # Bare turn events are suppressed to keep the stream readable.
    assert "thinking" not in err
    # log lines print verbatim.
    assert "driver selected: api" in err


def test_compile_progress_printer_annotates_tool_lines_with_tokens(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A turn event's cumulative tokens ride onto the tool lines that follow
    it — the token figures live on turn events, but the tool lines are what
    the printer actually shows."""
    from rote.cli import _compile_progress_printer
    from rote.compiler.events import CompilationEvent

    printer = _compile_progress_printer()
    for event in [
        CompilationEvent(type="turn", ts=0.0, turn=7, tokens={"input": 40234, "output": 8100}),
        CompilationEvent(type="tool", ts=0.0, turn=7, tool_name="Write", path="signatures/foo.py"),
    ]:
        printer(event)

    err = capsys.readouterr().err
    assert "[turn 7] Write signatures/foo.py (in 40.2k / out 8.1k tok)" in err


# ───────── MCP requirements surfacing (analyze / compile / emit) ─────────

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
    """No MCP requirement of any kind → empty manifest, registry unread."""
    from rote.cli import _mcp_requirements
    from rote.ir import load_pipeline

    monkeypatch.setattr(
        "rote.mcp.load_registry",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("registry read")),
    )
    assert _mcp_requirements(load_pipeline(_write_mcp_free_pipeline(tmp_path))) == []


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
    """A pipeline needing no MCP server → manifest present but empty, so
    agents can distinguish 'no requirements' from 'field missing'."""
    import json

    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: t\n---\n")

    _install_mock_compiler(monkeypatch, _write_mcp_free_pipeline(tmp_path))

    rc = cli_main(["analyze", str(skill_dir), "--json"])
    assert rc == 0
    report = json.loads(capsys.readouterr().out)
    assert report["mcp_servers"] == []


# ───────── Contract findings surface through the CLI ─────────


def test_emit_reports_contract_breaks_in_json(python_executable: str, tmp_path: Path) -> None:
    """`rote emit` re-checks the IR against the modules beside the
    pipeline.yaml, because that is where the fix-and-re-emit loop runs.

    The newest committed BDR compile carries two real breaks (see
    tests/test_contracts.py), so this doubles as proof the findings reach
    the machine-readable surface rather than only the human one.
    """
    result = subprocess.run(
        [
            python_executable,
            "-m",
            "rote.cli",
            "emit",
            str(REPO_ROOT / "examples/bdr-outreach/runs/2026-07-18-mcp-bindings/pipeline.yaml"),
            "--runtime",
            "dbos",
            "--out",
            str(tmp_path / "emitted"),
            "--json",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    findings = payload["contract_findings"]
    assert {f["node"] for f in findings} == {"hubspot_create_list", "exclusion_check_dnc"}
    assert all(f["severity"] == "error" for f in findings)


def test_emit_stays_silent_on_a_clean_pipeline(python_executable: str, tmp_path: Path) -> None:
    """A checker that reports on healthy compilations gets ignored."""
    result = subprocess.run(
        [
            python_executable,
            "-m",
            "rote.cli",
            "emit",
            str(REPO_ROOT / "examples/ops-report/runs/2026-07-18-rubric-v2/pipeline.yaml"),
            "--runtime",
            "dbos",
            "--out",
            str(tmp_path / "emitted"),
            "--json",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["contract_findings"] == []


# ───────── Prompt-cache cost accounting ─────────


def _sink(tmp_path: Path, rates: object) -> object:
    """A progress sink with its price resolution pinned to ``rates``."""
    from rote.cli import _JsonlProgressSink

    sink = _JsonlProgressSink.__new__(_JsonlProgressSink)
    sink._path = tmp_path / "p.ndjson"  # type: ignore[attr-defined]
    sink._model_id = "claude-sonnet-4-6"  # type: ignore[attr-defined]
    sink._prices = rates  # type: ignore[attr-defined]
    sink._file = None  # type: ignore[attr-defined]
    return sink


def test_cache_buckets_are_billed_at_their_own_rates(tmp_path: Path) -> None:
    """Each bucket bills at its own published rate.

    The rates are deliberately far apart so no two can be swapped without
    moving the total: a cache read is a tenth of plain input and a cache
    write is 1.25x it, so charging either at the input rate is visible.
    """
    from rote.cli import _Rates

    sink = _sink(
        tmp_path,
        _Rates(
            input_per_mtok=3.0,
            output_per_mtok=15.0,
            cache_read_per_mtok=0.3,
            cache_write_per_mtok=3.75,
        ),
    )
    tokens = {"input": 1000, "output": 500, "cache_write": 200_000, "cache_read": 800_000}
    expected = 1000 / 1e6 * 3.0 + 500 / 1e6 * 15.0 + 200_000 / 1e6 * 3.75 + 800_000 / 1e6 * 0.3
    assert sink._cost_usd(tokens) == pytest.approx(expected)  # type: ignore[attr-defined]

    # The negative half: pricing the plain pair alone is what shipped, and
    # it is ~90x lower on this input. Pin that it is NOT what comes back.
    plain_only = 1000 / 1e6 * 3.0 + 500 / 1e6 * 15.0
    assert sink._cost_usd(tokens) != pytest.approx(plain_only)  # type: ignore[attr-defined]


def test_a_cached_run_with_unknown_cache_rates_reports_no_cost(tmp_path: Path) -> None:
    """No figure beats a wrong figure.

    Most models sit in ``reference_prices``, which carries no cache rates.
    Billing their cache buckets at the plain input rate would overstate
    reads roughly tenfold; dropping the buckets understates the run by
    orders of magnitude. Both are worse than declining to answer.
    """
    from rote.cli import _Rates

    sink = _sink(tmp_path, _Rates(input_per_mtok=3.0, output_per_mtok=15.0))
    assert sink._cost_usd({"input": 10, "output": 5, "cache_read": 900_000}) is None  # type: ignore[attr-defined]
    assert sink._cost_usd({"input": 10, "output": 5, "cache_write": 900_000}) is None  # type: ignore[attr-defined]
    # A run that genuinely touched no cache is still priceable, so the
    # rule cannot be satisfied by returning None unconditionally.
    assert sink._cost_usd({"input": 1000, "output": 500}) == pytest.approx(0.0105)  # type: ignore[attr-defined]


def test_the_runtimes_own_billed_cost_wins_over_the_priced_estimate(tmp_path: Path) -> None:
    """``total_cost_usd`` from the agent runtime outranks a local estimate.

    It knows the per-model split and the exact cache rates that applied.
    """
    from rote.cli import _Rates

    path = tmp_path / "p.ndjson"
    from rote.cli import _JsonlProgressSink

    sink = _JsonlProgressSink.__new__(_JsonlProgressSink)
    sink._path = path  # type: ignore[attr-defined]
    sink._model_id = "m"  # type: ignore[attr-defined]
    sink._prices = _Rates(input_per_mtok=3.0, output_per_mtok=15.0)  # type: ignore[attr-defined]
    sink._file = path.open("w", encoding="utf-8")  # type: ignore[attr-defined]

    digest = {"type": "summary", "total_tokens": {"input": 1000, "output": 500}}
    sink.write_summary(dict(digest), reported_cost=2.71)  # type: ignore[attr-defined]
    sink.close()  # type: ignore[attr-defined]
    written = json.loads(path.read_text(encoding="utf-8").strip())
    assert written["cost_usd"] == 2.71

    # Without one, the local estimate is still used, so the preference is
    # a real branch rather than the estimate having been removed.
    sink2 = _sink(tmp_path, _Rates(input_per_mtok=3.0, output_per_mtok=15.0))
    sink2._file = (tmp_path / "q.ndjson").open("w", encoding="utf-8")  # type: ignore[attr-defined]
    sink2.write_summary(dict(digest))  # type: ignore[attr-defined]
    sink2.close()  # type: ignore[attr-defined]
    fallback = json.loads((tmp_path / "q.ndjson").read_text(encoding="utf-8").strip())
    assert fallback["cost_usd"] == pytest.approx(0.0105)


def test_token_note_reports_the_whole_input_volume() -> None:
    """The ``in`` figure includes cache writes and reads.

    Reporting the plain field alone announced a fully prompt-cached
    compilation as ``in 113`` when it had moved six figures of input.
    """
    from rote.cli import _format_token_note

    note = _format_token_note(
        {"input": 113, "output": 1919, "cache_write": 122_956, "cache_read": 41_000}
    )
    assert "in 164.1k" in note
    assert "164.0k cached" in note
    assert "113" not in note

    # An uncached run keeps the plain rendering, with no cache clause.
    assert _format_token_note({"input": 1000, "output": 500}) == " (in 1.0k / out 500 tok)"


#: The real ``_resolve_prices``, captured at import time.
#:
#: ``conftest`` replaces this attribute for every test so no unit test
#: reaches the network, and every pricing test re-stubs the same seam.
#: The consequence is that the function's own body never runs, so it
#: could drop the cache rates entirely with the suite fully green (a
#: mutation confirmed exactly that). Grabbing the underlying function at
#: module import, before any fixture applies, is what lets the mapping
#: itself be tested.
_REAL_RESOLVE_PRICES = _SinkForCapture.__dict__["_resolve_prices"].__func__


def _lineup_with_cache_rates() -> dict[str, object]:
    def _m(inp: float, out: float, released: str, cr: float, cw: float) -> dict[str, object]:
        return {
            "name": "",
            "cost": {"input": inp, "output": out, "cache_read": cr, "cache_write": cw},
            "release_date": released,
        }

    return {
        "anthropic": {
            "models": {
                "claude-fable-5": _m(10, 50, "2026-06-09", 1, 12.5),
                "claude-sonnet-5": _m(2, 10, "2026-06-30", 0.2, 2.5),
                "claude-haiku-4-5": _m(1, 5, "2025-10-15", 0.1, 1.25),
                # Not a tier representative: the case that was unpriceable.
                "claude-sonnet-4-6": _m(3, 15, "2026-02-17", 0.3, 3.75),
            }
        }
    }


def test_resolve_prices_carries_every_rate_off_the_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All four rates must survive the catalog-to-sink mapping.

    Exercises the real ``_resolve_prices`` by patching ``fetch_catalog``
    rather than the method, because patching the method is precisely what
    hid this code path from the suite.
    """
    from rote.eval.pricing import build_catalog

    catalog = build_catalog(_lineup_with_cache_rates(), None, provider="anthropic", fetched_at="t")
    monkeypatch.setattr("rote.eval.pricing.fetch_catalog", lambda provider: catalog)

    rates = _REAL_RESOLVE_PRICES("claude-sonnet-4-6")
    assert rates is not None
    assert rates.input_per_mtok == 3.0
    assert rates.output_per_mtok == 15.0
    assert rates.cache_read_per_mtok == 0.3
    assert rates.cache_write_per_mtok == 3.75

    # An unknown model resolves to nothing rather than raising.
    assert _REAL_RESOLVE_PRICES("not-a-model") is None


def test_resolve_prices_survives_an_offline_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed price fetch skips cost enrichment; it never sinks a run."""
    from rote.eval.pricing import PricingError

    def _down(provider: str) -> object:
        raise PricingError("offline")

    monkeypatch.setattr("rote.eval.pricing.fetch_catalog", _down)
    assert _REAL_RESOLVE_PRICES("claude-sonnet-4-6") is None
