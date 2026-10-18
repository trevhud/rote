"""What a newcomer sees on the first-run path: install hints, stub
reporting, run preflight, failure excerpts, the ``--json`` stdout contract,
and help text that matches behavior.

Each test here guards a defect found by a fresh-install run of the
published package (isolated HOME, no extras). They assert the value the
user reads, not just that some line was printed.
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path
from typing import Any

import pytest

from rote.adapters import get_adapter
from rote.adapters._common import judge_override_readme, unimplemented_stubs
from rote.cli import main as cli_main
from rote.eval.empirical import failure_excerpt
from rote.extras import EXTRAS, extra_for_missing_module, install_hint
from rote.ir import Pipeline, load_pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src" / "rote"
OPS_PIPELINE = REPO_ROOT / "examples" / "ops-report" / "expected" / "pipeline.yaml"
BDR_PIPELINE = REPO_ROOT / "examples" / "bdr-outreach" / "expected" / "pipeline.yaml"


# ───────── 3. install hints name the right distribution ─────────


def test_no_install_hint_in_src_names_the_bare_rote_package() -> None:
    """``rote`` on PyPI is an unrelated library; every hint must say ``rote-cli``."""
    bare = re.compile(r"(?<![\w-])rote\[")
    offenders = [
        f"{path.relative_to(REPO_ROOT)}:{lineno}: {line.strip()}"
        for path in sorted(SRC.rglob("*.py"))
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if bare.search(line)
    ]
    assert offenders == []


def test_install_commands_are_only_spelled_in_rote_extras() -> None:
    """A hand-typed ``rote-cli[...]`` is correct today and drifts tomorrow;
    every hint goes through :func:`rote.extras.install_hint`."""
    typed = re.compile(r"rote-cli\[")
    offenders = [
        f"{path.relative_to(REPO_ROOT)}:{lineno}: {line.strip()}"
        for path in sorted(SRC.rglob("*.py"))
        if path != SRC / "extras.py"
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if typed.search(line)
    ]
    assert offenders == []


def test_install_hint_spells_the_pypi_distribution() -> None:
    assert install_hint("dbos") == "pip install 'rote-cli[dbos]'"


def test_every_known_extra_exists_in_pyproject() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    declared = set(pyproject["project"]["optional-dependencies"])
    assert set(EXTRAS) <= declared


def test_doctor_hints_use_rote_cli(capsys: pytest.CaptureFixture[str]) -> None:
    from rote.cli import _render_doctor_text

    report = {
        "version": "0",
        "python": "3",
        "drivers": [{"name": "api", "available": False, "reason": "missing"}],
        "runtimes": [{"name": "dbos", "installed": False}],
        "mcp_servers": [],
        "apps": [],
        "ok": False,
    }
    assert "  ✗ dbos — pip install 'rote-cli[dbos]'" in _render_doctor_text(report).splitlines()


# ───────── 1/2a. failure excerpts: the end of stderr, plus the hint ─────────


def test_failure_excerpt_keeps_the_end_of_long_output() -> None:
    noise = "\n".join(f"12:00:0{i % 10} [INFO] (dbos) startup line {i}" for i in range(400))
    stderr = noise + "\ndbos._error.DBOSMaxStepRetriesExceeded: Step fetch_dock_pending failed"
    text = failure_excerpt(stderr)
    assert text.endswith("DBOSMaxStepRetriesExceeded: Step fetch_dock_pending failed")
    assert "startup line 0\n" not in text
    assert "earlier characters omitted" in text
    assert len(text) < 1700


def test_failure_excerpt_short_output_is_untouched() -> None:
    assert failure_excerpt("boom\n") == "boom"


def test_failure_excerpt_appends_the_extra_for_a_missing_module() -> None:
    stderr = (
        'Traceback (most recent call last):\n  File "main.py", line 31\n'
        "ModuleNotFoundError: No module named 'dbos'\n"
    )
    text = failure_excerpt(stderr)
    assert text.splitlines()[-1] == (
        "hint: the app imports `dbos`, which the dbos extra provides: pip install 'rote-cli[dbos]'"
    )


def test_missing_module_outside_the_extras_gets_no_hint() -> None:
    assert extra_for_missing_module("ModuleNotFoundError: No module named 'requests'") is None
    assert extra_for_missing_module("ModuleNotFoundError: No module named 'openai.types'") == (
        "openai",
        "openai-api",
    )


# ───────── 2b/1. `rote run` preflight ─────────


def _emit(pipeline_yaml: Path, runtime: str, out: Path) -> dict[str, Path]:
    pipeline = load_pipeline(pipeline_yaml)
    return dict(get_adapter(runtime).emit(pipeline, out))


def test_run_refuses_stubs_before_starting_the_app(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    app = tmp_path / "app"
    _emit(OPS_PIPELINE, "dbos", app)
    started: list[Any] = []
    monkeypatch.setattr("rote.runners.run_pipeline_trial", lambda *a, **k: started.append(a))
    rc = cli_main(["run", str(app), "--input", "{}"])
    err = capsys.readouterr().err
    assert rc == 2
    assert started == []
    assert "error: 7 emitted module(s) are still stubs" in err
    assert f"  {app.resolve() / 'extracted' / 'google_drive.py'}" in err.splitlines()
    assert f"then re-run: rote run {app.resolve()}" in err


def test_run_preflight_comes_before_input_and_gate_prompts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A gated pipeline with no --input would otherwise derive an input
    (an LLM call) and demand gate payloads, only to refuse afterwards."""
    import shutil

    out = tmp_path / "out"
    (out / "compiled").mkdir(parents=True)
    shutil.copy(OPS_PIPELINE, out / "compiled" / "pipeline.yaml")
    _emit(OPS_PIPELINE, "dbos", out / "runtime" / "dbos")
    derived: list[Any] = []
    monkeypatch.setattr("rote.runners.derive_input_payload", lambda *a: derived.append(a))
    rc = cli_main(["run", str(out)])
    err = capsys.readouterr().err
    assert rc == 2
    assert derived == []
    assert err.startswith("error: 7 emitted module(s) are still stubs")
    assert "HITL gate" not in err


def test_run_preflight_passes_once_stubs_are_filled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from rote.eval.empirical import MeasuredRun

    app = tmp_path / "app"
    _emit(OPS_PIPELINE, "dbos", app)
    for stub in unimplemented_stubs(app):
        stub.write_text("def implemented():\n    return {}\n", encoding="utf-8")
    monkeypatch.setattr(
        "rote.runners.run_pipeline_trial",
        lambda *a, **k: MeasuredRun(wall_seconds=0.1, output={"ok": True}),
    )
    rc = cli_main(["run", str(app), "--input", "{}"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out) == {"ok": True}


@pytest.mark.parametrize(("runtime", "extra"), [("dbos", "dbos"), ("temporal", "temporal")])
def test_run_names_the_missing_extra(
    runtime: str,
    extra: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import importlib.util

    app = tmp_path / "app"
    app.mkdir()
    if runtime == "dbos":
        (app / "main.py").write_text("", encoding="utf-8")
        (app / "dbos-config.yaml").write_text("", encoding="utf-8")
    else:
        (app / "workflow.py").write_text("", encoding="utf-8")
        (app / "activities.py").write_text("", encoding="utf-8")
    real_find_spec = importlib.util.find_spec
    module = EXTRAS[extra]
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a: None if name == module else real_find_spec(name, *a),
    )
    rc = cli_main(["run", str(app), "--input", "{}"])
    assert rc == 2
    assert capsys.readouterr().err.strip() == (
        f"error: running a {runtime} app requires the {extra} extra: "
        f"pip install 'rote-cli[{extra}]'"
    )


# ───────── 4. `rote emit` reports stubs in plain text ─────────


def test_emit_text_lists_the_same_stubs_as_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "app"
    assert cli_main(["emit", str(OPS_PIPELINE), "--runtime", "dbos", "--out", str(out)]) == 0
    text = capsys.readouterr().out.splitlines()
    assert (
        cli_main(["emit", str(OPS_PIPELINE), "--runtime", "dbos", "--out", str(out), "--json"]) == 0
    )
    stubs = json.loads(capsys.readouterr().out)["unimplemented_stubs"]
    assert len(stubs) == 7
    assert "  stubs: 7 module(s) still raise until you implement them:" in text
    for stub in stubs:
        assert f"    {stub}" in text
    assert f"  next: fill those in, then rote run {out}" in text


def test_emit_text_with_no_stubs_points_at_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "app"
    assert cli_main(["emit", str(OPS_PIPELINE), "--runtime", "dbos", "--out", str(out)]) == 0
    for stub in unimplemented_stubs(out):
        stub.write_text("def implemented():\n    return {}\n", encoding="utf-8")
    capsys.readouterr()
    assert cli_main(["emit", str(OPS_PIPELINE), "--runtime", "dbos", "--out", str(out)]) == 0
    text = capsys.readouterr().out.splitlines()
    assert "  stubs: none, every extracted module is implemented" in text
    assert f"  next: rote run {out}" in text


def test_stub_detection_covers_typescript_layouts(tmp_path: Path) -> None:
    out = tmp_path / "cf"
    _emit(BDR_PIPELINE, "cloudflare", out)
    stubs = {p.relative_to(out.resolve()).as_posix() for p in unimplemented_stubs(out)}
    assert "src/extracted/hubspot_upsert.ts" in stubs
    assert not any(s.endswith("_roteMcp.ts") for s in stubs)


REAL_IMPL_WITH_UNSUPPORTED_BRANCH = """\
def fetch_file_content(**payload):
    region = payload.get("region", "us")
    if region != "us":
        raise NotImplementedError("region not supported")
    return {"content": "a,b\\n1,2\\n"}
"""


def test_real_implementation_raising_not_implemented_is_not_a_stub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A filled-in module that raises NotImplementedError on one branch is
    working code; the preflight must not refuse to start it."""
    from rote.eval.empirical import MeasuredRun

    app = tmp_path / "app"
    _emit(OPS_PIPELINE, "dbos", app)
    for stub in unimplemented_stubs(app):
        stub.write_text(REAL_IMPL_WITH_UNSUPPORTED_BRANCH, encoding="utf-8")
    assert unimplemented_stubs(app) == []
    monkeypatch.setattr(
        "rote.runners.run_pipeline_trial",
        lambda *a, **k: MeasuredRun(wall_seconds=0.1, output={"ok": True}),
    )
    assert cli_main(["run", str(app), "--input", "{}"]) == 0
    assert json.loads(capsys.readouterr().out) == {"ok": True}


def test_edited_module_with_a_stub_left_is_still_a_stub(tmp_path: Path) -> None:
    app = tmp_path / "app"
    _emit(OPS_PIPELINE, "dbos", app)
    gmail = app.resolve() / "extracted" / "gmail.py"
    # The user implements one function and adds a second, still a stub in
    # the shape a compile agent writes (no sentinel, bare raise).
    gmail.write_text(
        "def search_threads(**payload):\n"
        "    return {'threads': []}\n\n\n"
        "def fetch_thread(**payload):\n"
        '    """Fetch one thread."""\n'
        "    raise NotImplementedError('fill me in')\n",
        encoding="utf-8",
    )
    assert gmail in unimplemented_stubs(app)
    # Editing a rote stub's docstring keeps its sentinel raise: still a stub.
    drive = app.resolve() / "extracted" / "google_drive.py"
    drive.write_text(drive.read_text(encoding="utf-8").replace("STUB", "TODO"), encoding="utf-8")
    assert drive in unimplemented_stubs(app)


def test_typescript_stub_left_after_editing_another_is_still_a_stub(tmp_path: Path) -> None:
    out = tmp_path / "cf"
    _emit(BDR_PIPELINE, "cloudflare", out)
    stub = out.resolve() / "src" / "extracted" / "hubspot_upsert.ts"
    text = stub.read_text(encoding="utf-8")
    stub.write_text("export const helper = 1;\n" + text, encoding="utf-8")
    assert stub in unimplemented_stubs(out)
    stub.write_text("export async function hubspotUpsert() { return {}; }\n", encoding="utf-8")
    assert stub not in unimplemented_stubs(out)


def test_allow_stubs_starts_the_run_anyway(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from rote.eval.empirical import MeasuredRun

    app = tmp_path / "app"
    _emit(OPS_PIPELINE, "dbos", app)
    assert unimplemented_stubs(app)
    monkeypatch.setattr(
        "rote.runners.run_pipeline_trial",
        lambda *a, **k: MeasuredRun(wall_seconds=0.1, output={"ok": True}),
    )
    assert cli_main(["run", str(app), "--input", "{}", "--allow-stubs"]) == 0
    assert json.loads(capsys.readouterr().out) == {"ok": True}


def test_stub_refusal_points_at_allow_stubs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    app = tmp_path / "app"
    _emit(OPS_PIPELINE, "dbos", app)
    assert cli_main(["run", str(app), "--input", "{}"]) == 2
    assert (
        capsys.readouterr()
        .err.rstrip()
        .endswith("If the run will not reach them, pass --allow-stubs.")
    )


def test_untracked_output_says_so_instead_of_none(tmp_path: Path) -> None:
    """Output whose code imports implementations from outside the dir (the
    temporal adapter before it emitted extracted/) must not read as
    "stubs: none, next: rote run"."""
    import io

    from rote.adapters._common import MANIFEST_NAME, stubs_tracked
    from rote.cli import _print_stub_todo

    pipeline = load_pipeline(OPS_PIPELINE)
    out = tmp_path / "app"
    out.mkdir()
    (out / MANIFEST_NAME).write_text(
        json.dumps({"version": 1, "files": {"activities.py": "0", "workflow.py": "0"}}),
        encoding="utf-8",
    )
    assert stubs_tracked(out, pipeline) is False
    buf = io.StringIO()
    _print_stub_todo([], out, tracked=False, stream=buf)
    text = buf.getvalue()
    assert text.startswith("  stubs: not tracked for this output")
    assert "next: rote run" not in text
    assert "none" not in text


def test_emitted_runtimes_track_stubs(tmp_path: Path) -> None:
    """Every adapter that emits extracted/ modules is vouched for, so its
    empty list really means nothing is left to implement."""
    from rote.adapters._common import stubs_tracked

    for runtime in ("dbos", "temporal", "cloudflare", "dbos-ts", "inngest"):
        out = tmp_path / runtime
        _emit(BDR_PIPELINE, runtime, out)
        assert stubs_tracked(out, load_pipeline(BDR_PIPELINE)), runtime
        assert unimplemented_stubs(out), runtime


def test_emit_json_reports_whether_stubs_are_tracked(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "app"
    assert cli_main(["emit", str(OPS_PIPELINE), "--out", str(out), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["stubs_tracked"] is True


def test_stub_detection_without_a_manifest_finds_nothing(tmp_path: Path) -> None:
    (tmp_path / "extracted").mkdir()
    (tmp_path / "extracted" / "x.py").write_text("raise NotImplementedError\n", encoding="utf-8")
    assert unimplemented_stubs(tmp_path) == []


# ───────── 5. `--json` failures still write JSON to stdout ─────────


def test_emit_json_failure_writes_an_error_object(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = cli_main(
        ["emit", str(BDR_PIPELINE), "--runtime", "python", "--out", str(tmp_path / "o"), "--json"]
    )
    out = capsys.readouterr().out
    assert rc == 1
    payload = json.loads(out)
    assert payload["ok"] is False
    assert payload["exit_code"] == 1
    assert payload["error"].startswith("python adapter: pipeline 'bdr-campaign' contains hitl_gate")


def test_json_error_object_carries_multiline_detail(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    app = tmp_path / "app"
    _emit(OPS_PIPELINE, "dbos", app)
    rc = cli_main(["run", str(app), "--input", "{}", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 2
    assert payload["exit_code"] == 2
    assert payload["error"].splitlines()[0] == (
        "7 emitted module(s) are still stubs and would raise on their first call:"
    )
    assert (
        payload["error"].splitlines()[-1] == "If the run will not reach them, pass --allow-stubs."
    )


def test_json_success_is_not_wrapped(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli_main(["emit", str(OPS_PIPELINE), "--out", str(tmp_path / "o"), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert "ok" not in payload
    assert payload["pipeline"]["name"] == "ops-report"


def test_missing_pipeline_file_json_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = cli_main(["emit", str(tmp_path / "nope.yaml"), "--out", str(tmp_path / "o"), "--json"])
    assert rc == 2
    assert json.loads(capsys.readouterr().out)["error"] == (
        f"pipeline file not found: {tmp_path / 'nope.yaml'}"
    )


# ───────── 6/7. help text derives from the real behavior ─────────


def _help(*argv: str, capsys: pytest.CaptureFixture[str]) -> str:
    """Help text with argparse's line wrapping undone (it breaks at hyphens)."""
    with pytest.raises(SystemExit):
        cli_main([*argv, "--help"])
    return re.sub(r"-\s+", "-", " ".join(capsys.readouterr().out.split()))


def test_run_help_lists_every_runnable_runtime(capsys: pytest.CaptureFixture[str]) -> None:
    from rote.runners import RUNNABLE_RUNTIMES

    text = _help("run", capsys=capsys)
    assert f"({', '.join(sorted(RUNNABLE_RUNTIMES))})" in text
    assert "in-process today" not in text
    assert "native dev command" not in text


def test_model_help_names_the_actual_default(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import rote.compiler.drivers.anthropic_api as api
    import rote.compiler.drivers.claude as claude

    monkeypatch.setattr(claude, "DEFAULT_MODEL", "claude-test-default-9")
    monkeypatch.setattr(api, "DEFAULT_MODEL", "claude-test-default-9")
    text = _help("compile", capsys=capsys)
    assert "claude-test-default-9 for the claude and api drivers" in text
    assert "Sonnet 4.6" not in text
    assert "(default: claude-test-default-9)" in _help("run", capsys=capsys)


# ───────── 8. README env knobs come from the pipeline's judges ─────────


def test_readme_without_judges_documents_no_judge_knobs(tmp_path: Path) -> None:
    from rote.adapters.python import PythonAdapterConfig
    from rote.adapters.python import emit_readme as python_readme

    readmes = {"python": python_readme(load_pipeline(OPS_PIPELINE), PythonAdapterConfig())}
    for runtime in ("dbos", "dbos-ts"):
        _emit(OPS_PIPELINE, runtime, tmp_path / runtime)
        readmes[runtime] = (tmp_path / runtime / "README.md").read_text(encoding="utf-8")
    for runtime, readme in readmes.items():
        assert "ROTE_MODEL_" not in readme, runtime
        assert "ROTE_BASE_URL_" not in readme, runtime
        assert "This pipeline has no LLM-judge nodes, so it needs no model API keys." in readme


def test_readme_lists_each_real_judge_knob(tmp_path: Path) -> None:
    pipeline = load_pipeline(BDR_PIPELINE)
    out = tmp_path / "dbos"
    _emit(BDR_PIPELINE, "dbos", out)
    readme = (out / "README.md").read_text(encoding="utf-8")
    judges = [n.id for n in pipeline.nodes if n.kind.value == "llm_judge"]
    assert judges
    for judge in judges:
        assert f"export ROTE_MODEL_{judge.upper()}=..." in readme
        assert f"export ROTE_BASE_URL_{judge.upper()}=..." in readme
    assert "ROTE_MODEL_<NODE_ID>" not in readme


def test_readme_temperature_knob_only_when_the_ir_sets_one() -> None:
    pipeline = load_pipeline(BDR_PIPELINE)
    judge = next(n for n in pipeline.nodes if n.signature_spec is not None)
    assert judge.signature_spec is not None
    assert "ROTE_TEMPERATURE_" not in judge_override_readme(pipeline)
    data = pipeline.model_dump()
    for node in data["nodes"]:
        if node["id"] == judge.id:
            node["signature_spec"]["temperature"] = 0.2
    with_temp = Pipeline.model_validate(data)
    assert f"export ROTE_TEMPERATURE_{judge.id.upper()}=..." in judge_override_readme(with_temp)


# ───────── 9. `rote config` columns line up ─────────


def test_config_rows_align(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli_main(["config"]) == 0
    rows = [
        line
        for line in capsys.readouterr().out.splitlines()
        if re.match(r"  (runtime|deploy|agent|model|inference) ", line)
    ]
    assert len(rows) == 5
    # The key is one word, so the value starts at the first non-space after
    # it. Suite isolation leaves every key on its built-in default, so each
    # row's source starts at "built-in".
    value_cols = {len(row) - len(row[2:].split(" ", 1)[1].lstrip()) for row in rows}
    source_cols = {row.index("built-in") for row in rows}
    assert len(value_cols) == 1, rows
    assert len(source_cols) == 1, rows
