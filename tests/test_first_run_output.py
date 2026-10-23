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

from rote import __version__
from rote.adapters import get_adapter
from rote.adapters._common import judge_override_readme, unimplemented_stubs
from rote.cli import main as cli_main
from rote.eval.empirical import failure_excerpt
from rote.extras import (
    ALIAS_EXTRAS,
    BASE_MODULES,
    EXTRAS,
    install_hint,
    missing_module_in,
    reinstall_hint,
)
from rote.ir import Pipeline, load_pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src" / "rote"
OPS_PIPELINE = REPO_ROOT / "examples" / "ops-report" / "expected" / "pipeline.yaml"
BDR_PIPELINE = REPO_ROOT / "examples" / "bdr-outreach" / "expected" / "pipeline.yaml"
#: A real compile of ops-report: every pure_function implemented, and the
#: four fetches mcp-bound, with the agent's `--backend api` stubs beside them.
OPS_COMPILED = (
    REPO_ROOT / "examples" / "ops-report" / "runs" / "2026-10-18-implements-pure-functions"
)


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
    # Pinned: installing an extra must never move rote itself.
    assert install_hint("temporal") == f"pip install 'rote-cli[temporal]=={__version__}'"
    assert reinstall_hint() == f"pip install 'rote-cli=={__version__}'"


def test_install_hint_refuses_an_alias_extra() -> None:
    """``rote-cli[mcp]`` installs nothing beyond the base set, so a hint
    naming it would send someone to a command that changes nothing."""
    for alias in ALIAS_EXTRAS:
        with pytest.raises(ValueError, match="empty alias"):
            install_hint(alias)


def test_every_known_extra_exists_in_pyproject() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    optional = pyproject["project"]["optional-dependencies"]
    assert set(EXTRAS) <= set(optional)
    # Every extra rote knows installs something, and the aliases resolve
    # (declared) and install nothing (empty), so an old `rote-cli[mcp]`
    # command keeps working and means the base install.
    assert all(optional[extra] for extra in EXTRAS)
    assert {alias: optional[alias] for alias in ALIAS_EXTRAS} == {a: [] for a in ALIAS_EXTRAS}
    base = {re.split(r"[<>=!~\[ ]", d, maxsplit=1)[0] for d in pyproject["project"]["dependencies"]}
    assert {dep.distribution for dep in BASE_MODULES.values()} <= base
    # An extra proven by another distribution's extra must install that
    # extra: serve's full fastmcp requires fastmcp-slim[...server...].
    from importlib.metadata import requires

    for name, extra in EXTRAS.items():
        if extra.requires_extra_of is None:
            continue
        dist, wanted = extra.requires_extra_of
        brought = [
            r
            for d in optional[name]
            for r in requires(re.split(r"[<>=!~\[ ]", d, maxsplit=1)[0]) or []
            if "extra ==" not in r
        ]
        pattern = rf"^{re.escape(dist)}\[[^\]]*\b{re.escape(wanted)}\b"
        assert any(re.match(pattern, r) for r in brought), (name, brought)


def test_doctor_hints_use_rote_cli(capsys: pytest.CaptureFixture[str]) -> None:
    from rote.cli import _render_doctor_text

    report = {
        "version": "0",
        "python": "3",
        "drivers": [{"name": "api", "available": False, "reason": "missing"}],
        "dependencies": [{"module": "fastmcp", "used_for": "x", "installed": False}],
        "runtimes": [{"name": "temporal", "used_for": "y", "installed": False}],
        "mcp_servers": [],
        "apps": [],
        "ok": False,
    }
    lines = _render_doctor_text(report).splitlines()
    assert f"  ✗ [temporal] extra (y) — pip install 'rote-cli[temporal]=={__version__}'" in lines
    reinstall = f"pip install 'rote-cli=={__version__}'"
    assert "  ✗ fastmcp (x) — missing" in lines
    assert lines[-1] == f"✗ this install is missing fastmcp: {reinstall}"


def test_doctor_is_not_ready_while_a_base_dependency_is_missing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`rote doctor` said "ready to compile" with the MCP client missing,
    and every MCP login and tool grounding then failed. A missing extra
    (dbos, temporal) is normal and does not fail it."""
    import importlib.util

    monkeypatch.setattr("rote.cli.available_drivers", lambda: [("claude", True, "")])
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a: None if name == "fastmcp" else real_find_spec(name, *a),
    )
    assert cli_main(["doctor", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is False
    assert {d["module"]: d["installed"] for d in report["dependencies"]} == {
        "fastmcp": False,
        "httpx": True,
    }
    # Negative half: the same machine with fastmcp present is ready, even
    # with both runtime extras missing.
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a: None if name in ("dbos", "temporalio") else real_find_spec(name, *a),
    )
    assert cli_main(["doctor", "--json"]) == 0
    runtimes = {r["name"]: r["installed"] for r in json.loads(capsys.readouterr().out)["runtimes"]}
    assert (runtimes["dbos"], runtimes["temporal"]) == (False, False)
    monkeypatch.setattr(importlib.util, "find_spec", real_find_spec)
    assert cli_main(["doctor"]) == 0
    assert capsys.readouterr().out.splitlines()[-1] == "✓ ready to compile"


# ───────── 1/2a. failure excerpts: the end of stderr, plus the hint ─────────


def test_failure_excerpt_keeps_the_end_of_long_output() -> None:
    noise = "\n".join(f"12:00:0{i % 10} [INFO] (dbos) startup line {i}" for i in range(400))
    stderr = noise + "\ndbos._error.DBOSMaxStepRetriesExceeded: Step fetch_dock_pending failed"
    text = failure_excerpt(stderr)
    assert text.endswith("DBOSMaxStepRetriesExceeded: Step fetch_dock_pending failed")
    assert "startup line 0\n" not in text
    assert "earlier characters omitted" in text
    assert len(text) < 1700


def test_failure_excerpt_keeps_a_node_errors_message_above_its_stack() -> None:
    """Node prints the message first and the stack after, so the tail of a
    long DBOS-TS failure held only `at ...` frames and never said why."""
    frames = "\n".join(f"    at frame{i} (/app/node_modules/x.js:{i}:1)" for i in range(80))
    stderr = "startup\nError: pipeline input is missing required field(s): region\n" + frames
    text = failure_excerpt(stderr)
    assert text.startswith("Error: pipeline input is missing required field(s): region\n")
    assert "earlier characters omitted" in text
    assert text.endswith("x.js:79:1)")


def test_failure_excerpt_short_output_is_untouched() -> None:
    assert failure_excerpt("boom\n") == "boom"


def test_failure_excerpt_appends_the_fix_for_a_missing_module() -> None:
    def tail(module: str) -> str:
        stderr = (
            'Traceback (most recent call last):\n  File "main.py", line 31\n'
            f"ModuleNotFoundError: No module named '{module}'\n"
        )
        return failure_excerpt(stderr).splitlines()[-1]

    assert tail("fastmcp") == (
        "hint: the app imports `fastmcp`, a rote-cli dependency missing from this install "
        f"(restore it with: pip install 'rote-cli=={__version__}')"
    )
    assert tail("dbos") == (
        "hint: the app imports `dbos`, from the dbos extra "
        f"(pip install 'rote-cli[dbos]=={__version__}')"
    )
    assert tail("temporalio") == (
        "hint: the app imports `temporalio`, from the temporal extra "
        f"(pip install 'rote-cli[temporal]=={__version__}')"
    )


def test_missing_module_outside_the_extras_gets_no_hint() -> None:
    assert missing_module_in("ModuleNotFoundError: No module named 'requests'") is None
    assert missing_module_in("ModuleNotFoundError: No module named 'openai.types'") == "openai"
    assert missing_module_in("ModuleNotFoundError: No module named 'fastmcp.client'") == "fastmcp"


# ───────── 2b/1. `rote run` preflight ─────────


def _emit(pipeline_yaml: Path, runtime: str, out: Path) -> dict[str, Path]:
    pipeline = load_pipeline(pipeline_yaml)
    # The real modules beside the IR, as `rote emit` sources them.
    return dict(get_adapter(runtime, extracted_source_dir=pipeline_yaml.parent).emit(pipeline, out))


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


# A missing runtime extra is offered as an install plan: tests/test_install_extra.py.


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


def test_emit_next_hint_is_a_command_that_runs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A bare pipeline.yaml has no skill to derive an input from, so a
    pipeline with required inputs needs --input. The hint points at the
    example's input file, and following it verbatim works."""
    import shlex

    draft = REPO_ROOT / "examples" / "expense-audit" / "expected" / "pipeline-draft.yaml"
    out = tmp_path / "app"
    assert cli_main(["emit", str(draft), "--runtime", "python", "--out", str(out)]) == 0
    (hint,) = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("  next: ")]
    expected_input = REPO_ROOT / "examples" / "expense-audit" / "input.json"
    assert hint == f"  next: rote run {out} --input {expected_input}"
    assert cli_main(shlex.split(hint.removeprefix("  next: "))[1:]) == 0
    assert json.loads(capsys.readouterr().out)["file_report"]["batch_total_usd"] == 3304.15

    # No input file to point at: the placeholder names the required fields.
    lone = tmp_path / "lone" / "pipeline.yaml"
    lone.parent.mkdir()
    lone.write_text(draft.read_text(encoding="utf-8"), encoding="utf-8")
    assert cli_main(["emit", str(lone), "--runtime", "python", "--out", str(tmp_path / "a2")]) == 0
    text = capsys.readouterr().out
    assert "--input '<JSON object with period, submissions>'" in text


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


@pytest.mark.parametrize(
    ("source", "is_stub"),
    [
        # BDR's zoominfo.enrich_batch: validates, then raises unconditionally.
        (
            "def enrich_batch(contacts, batch_size=10):\n"
            '    """Enrich a batch."""\n'
            "    if len(contacts) > batch_size:\n"
            "        raise ValueError('too many')\n"
            "    raise NotImplementedError('implement against ZoomInfo')\n",
            True,
        ),
        ("def f(**payload):\n    raise NotImplementedError\n", True),
        # The unsupported case is a branch: working code.
        (
            "def f(region):\n"
            "    if region == 'us':\n"
            "        return {'ok': True}\n"
            "    else:\n"
            "        raise NotImplementedError(region)\n",
            False,
        ),
        # Dispatch that falls through to the raise: it returns on its real paths.
        (
            "def f(kind):\n"
            "    if kind == 'a':\n"
            "        return 1\n"
            "    raise NotImplementedError(kind)\n",
            False,
        ),
        # A nested helper's return does not make the outer stub real.
        (
            "def f():\n    def helper():\n        return 1\n    raise NotImplementedError\n",
            True,
        ),
        (
            "import abc\n\nclass Base(abc.ABC):\n"
            "    @abc.abstractmethod\n"
            "    def run(self):\n"
            "        raise NotImplementedError\n",
            False,
        ),
    ],
    ids=["preamble-then-raise", "bare", "if-else-branch", "fallthrough", "nested", "abstract"],
)
def test_python_stub_shapes(source: str, is_stub: bool) -> None:
    from rote.adapters._common import _python_module_is_stub

    assert _python_module_is_stub(source) is is_stub


def test_bdr_reports_every_stub_module_including_zoominfo(tmp_path: Path) -> None:
    """BDR has four stub modules. zoominfo's validates its batch size before
    raising and used to be missed, leaving three."""
    app = tmp_path / "app"
    _emit(BDR_PIPELINE, "dbos", app)
    names = sorted(p.name for p in unimplemented_stubs(app))
    assert "zoominfo.py" in names
    assert names == ["exclusion_checks.py", "hubspot.py", "taxonomy.py", "zoominfo.py"]


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


@pytest.mark.parametrize(
    ("runtime", "backend", "stubs"),
    [
        ("dbos", "mcp", []),
        ("dbos", "api", ["drive.py", "gmail.py"]),
        # No MCP client: the step calls the module whatever the backend.
        ("python", "mcp", ["drive.py", "gmail.py"]),
        ("temporal", "mcp", ["drive.py", "gmail.py"]),
    ],
)
def test_run_and_compile_agree_on_which_mcp_bound_stubs_stop_a_run(
    runtime: str, backend: str, stubs: list[str], tmp_path: Path
) -> None:
    """ops-report's fetch modules are `--backend api` stubs. On the default
    backend their steps call the MCP tool and never the function, so
    neither the compile nor `rote run` may call them blocking; on the api
    backend, and on python and temporal (no MCP client, the step calls the
    module), both must. One answer, read from the IR and from the disk."""
    from rote.contracts import implementation_status

    pipeline = load_pipeline(OPS_COMPILED / "pipeline.yaml")
    impls = implementation_status(pipeline, OPS_COMPILED, runtime, backend)
    blocking = sorted({i.impl.split(":")[0].rsplit("/", 1)[1] for i in impls if i.blocks_run})
    assert blocking == stubs
    # The python adapter refuses ops-report's gate, so only the compile's
    # side exists there; every other runtime is read back off the disk too.
    if runtime != "python":
        app = tmp_path / "app"
        options: dict[str, Any] = {"extracted_source_dir": OPS_COMPILED}
        if runtime == "dbos":
            options["external_backend"] = backend
        get_adapter(runtime, **options).emit(pipeline, app)
        assert sorted(p.name for p in unimplemented_stubs(app)) == stubs
    # Non-degenerate: the stubs are there either way, only their verdict moves.
    assert {i.node_id for i in impls if i.status == "stub"} == {
        n.id for n in pipeline.nodes if n.mcp is not None
    }


def test_run_gets_past_the_stub_check_on_an_mcp_backed_compile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from rote.eval.empirical import MeasuredRun

    app = tmp_path / "app"
    _emit(OPS_COMPILED / "pipeline.yaml", "dbos", app)
    ran: list[Path] = []

    def trial(target: Any, *a: Any, **k: Any) -> MeasuredRun:
        ran.append(target)
        return MeasuredRun(wall_seconds=0.1, output={"ok": True})

    monkeypatch.setattr("rote.runners.run_pipeline_trial", trial)
    signal = 'duty_manager_data_provided={"ok": true}'
    assert cli_main(["run", str(app), "--input", "{}", "--signal", signal]) == 0
    assert ran, capsys.readouterr().err


def test_a_stub_beside_an_mcp_backed_function_still_counts(tmp_path: Path) -> None:
    """Only the functions the manifest names as MCP-backed are exempt: a
    second stub in the same module is still called by nothing rote knows
    of, so it keeps the module listed."""
    app = tmp_path / "app"
    _emit(OPS_COMPILED / "pipeline.yaml", "dbos", app)
    gmail = app.resolve() / "extracted" / "gmail.py"
    assert gmail not in unimplemented_stubs(app)
    gmail.write_text(
        gmail.read_text(encoding="utf-8")
        + "\n\ndef fetch_thread(**payload):\n    raise NotImplementedError('fill me in')\n",
        encoding="utf-8",
    )
    assert gmail in unimplemented_stubs(app)


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


@pytest.mark.parametrize(
    ("runtime", "fix", "not_said"),
    [
        ("dbos", "re-emit with the real modules in an extracted/ directory", "TypeScript"),
        ("cloudflare", "Implement them in TypeScript in the files above", "re-emit"),
        ("inngest", "Implement them in TypeScript in the files above", "re-emit"),
        ("dbos-ts", "Implement them in TypeScript in the files above", "re-emit"),
    ],
)
def test_stub_refusal_fix_matches_the_runtimes_language(
    runtime: str, fix: str, not_said: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A TS app's implementations are TypeScript: telling its user to put
    Python modules beside the pipeline.yaml cannot work."""
    app = tmp_path / "app"
    _emit(BDR_PIPELINE, runtime, app)
    assert cli_main(["run", str(app), "--input", "{}"]) == 2
    err = capsys.readouterr().err
    assert fix in err
    assert not_said not in err.split("on their first call:")[1].split("then re-run")[0]


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


def _emitted_module(tmp_path: Path, runtime: str, rel: str) -> tuple[Path, Path]:
    out = tmp_path / runtime
    _emit(BDR_PIPELINE if runtime == "cloudflare" else OPS_PIPELINE, runtime, out)
    return out, out.resolve() / rel


def test_python_marker_in_a_comment_or_docstring_is_not_a_stub(tmp_path: Path) -> None:
    out, mod = _emitted_module(tmp_path, "dbos", "extracted/google_drive.py")
    assert mod in unimplemented_stubs(out)
    mod.write_text(
        '"""Was a [rote stub]; now calls Drive directly."""\n\n\n'
        "# TODO: this used to be a [rote stub]\n"
        "def fetch_file_content(**payload):\n"
        '    """Replaces the [rote stub] that rote emitted."""\n'
        '    note = "see [rote stub] history in git"\n'
        "    return {'content': note}\n",
        encoding="utf-8",
    )
    assert mod not in unimplemented_stubs(out)


def test_python_marker_inside_a_raise_is_a_stub(tmp_path: Path) -> None:
    """Rote's template moved into a branch: still a stub, because the
    sentinel is a raise argument (and the body is no longer bare)."""
    out, mod = _emitted_module(tmp_path, "dbos", "extracted/google_drive.py")
    mod.write_text(
        "def fetch_file_content(**payload):\n"
        "    if payload:\n"
        "        raise NotImplementedError('[rote stub] google_drive: implement me')\n"
        "    return {}\n",
        encoding="utf-8",
    )
    assert mod in unimplemented_stubs(out)


def test_python_stub_method_on_a_class_is_a_stub(tmp_path: Path) -> None:
    """Agent-written modules wrap vendor clients in classes."""
    out, mod = _emitted_module(tmp_path, "dbos", "extracted/google_drive.py")
    mod.write_text(
        "class DriveClient:\n"
        "    def __init__(self):\n"
        "        self.base = 'https://www.googleapis.com/drive/v3'\n\n"
        "    def fetch(self, file_id):\n"
        '        """Download one file."""\n'
        "        raise NotImplementedError('call the Drive API')\n\n\n"
        "def fetch_file_content(**payload):\n"
        "    return DriveClient().fetch(payload['file_id'])\n",
        encoding="utf-8",
    )
    assert mod in unimplemented_stubs(out)


def test_python_abstract_method_is_not_a_stub(tmp_path: Path) -> None:
    out, mod = _emitted_module(tmp_path, "dbos", "extracted/google_drive.py")
    mod.write_text(
        "import abc\n\n\n"
        "class Source(abc.ABC):\n"
        "    @abc.abstractmethod\n"
        "    def fetch(self, file_id):\n"
        "        raise NotImplementedError\n\n\n"
        "class Drive(Source):\n"
        "    def fetch(self, file_id):\n"
        "        return {'content': file_id}\n\n\n"
        "def fetch_file_content(**payload):\n"
        "    return Drive().fetch(payload['file_id'])\n",
        encoding="utf-8",
    )
    assert mod not in unimplemented_stubs(out)


def test_typescript_marker_in_a_comment_is_not_a_stub(tmp_path: Path) -> None:
    out, mod = _emitted_module(tmp_path, "cloudflare", "src/extracted/hubspot_upsert.ts")
    assert mod in unimplemented_stubs(out)
    mod.write_text(
        "// TODO was a [rote stub]\n"
        "/* the [rote stub] that rote emitted\n   is gone */\n"
        "/** Upserts contacts. Replaces the [rote stub]. */\n"
        "export async function hubspotUpsert(_input: unknown) {\n"
        '    const url = "https://api.hubapi.com/crm/v3"; // not a [rote stub]\n'
        "    return { url };\n"
        "}\n",
        encoding="utf-8",
    )
    assert mod not in unimplemented_stubs(out)


def test_strip_ts_comments_keeps_strings() -> None:
    from rote.adapters._common import strip_ts_comments

    src = (
        'const u = "https://x"; // gone\n'
        'const t = `a//b`; /* gone */ throw new Error("[rote stub] y");'
    )
    cleaned = strip_ts_comments(src)
    assert "gone" not in cleaned
    assert '"https://x"' in cleaned
    assert "`a//b`" in cleaned
    assert '"[rote stub] y"' in cleaned


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


# How the hint follows the install (uv tool, uvx, pip, a checkout):
# tests/test_install_extra.py.


def test_an_apps_one_line_error_survives_runtime_noise_after_it() -> None:
    """A DBOS child workflow still finishing at shutdown logs after the app's
    `error:` line, so the line is found wherever it sits."""
    from rote.eval.empirical import usage_error

    stderr = (
        "12:00:01 [    INFO] (dbos) DBOS launched!\n"
        "error: this run is read-only, and briefs.publish_brief may write\n"
        "Traceback (most recent call last):\n"
        '  File "dbos/_core.py", line 1126, in _execute_workflow_wthread\n'
        "dbos._error.DBOSException: System database accessed before DBOS was launched\n"
    )
    assert usage_error(stderr) == "this run is read-only, and briefs.publish_brief may write"
    assert usage_error("Traceback (most recent call last):\nValueError: boom\n") is None
