"""``rote setup``: every consent point, the agent protocol, skip and resume. No inference.

The world is a fixture Claude Code config (in the conftest-isolated client
home) naming the two fixture MCP servers, and a copy of docs-digest in that
home's ``~/.claude/skills``: one of "the user's" skills. The compile is
``tests/test_try.py``'s fake (docs-digest's committed compile); the paired
eval is the real :func:`rote.eval.skill_ab.run_subject` with the fake
``claude``. A person is a :class:`ScriptedPrompter`; an agent is the real
non-interactive path (``CLAUDE_CODE_CHILD_SESSION=1``).
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from rote import demo
from rote.cli import main
from rote.interact.prompter import DEFAULT, Answer, ScriptedPrompter
from rote.mcp.registry import load_registry
from rote.setup import first_run, try_skill
from tests._skill_driver import FixtureServer, fixture_server
from tests.test_try import FAKE_CLAUDE, World

pytest.importorskip("fastmcp")

REPO = Path(__file__).resolve().parent.parent
YES = Answer("confirm", "--yes", True)
NO = Answer("confirm", "--yes", False)


@pytest.fixture(scope="module")
def servers(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, FixtureServer]]:
    hits = tmp_path_factory.mktemp("setup-mcp")
    with (
        fixture_server(hits / "mslearn.jsonl", ["microsoft_docs_search"]) as mslearn,
        fixture_server(hits / "deepwiki.jsonl", ["read_wiki_structure"]) as deepwiki,
    ):
        yield {"mslearn": mslearn, "deepwiki": deepwiki}


class SetupWorld(World):
    @property
    def skill(self) -> Path:
        return Path(os.environ["ROTE_CLIENT_HOME_DIR"]) / ".claude" / "skills" / "docs-digest"


@pytest.fixture
def world(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, servers: dict[str, FixtureServer]
) -> SetupWorld:
    from rote.interact.agents import SIGNALS
    from rote.skill_target.trust import CA_VARS, PROXY_VARS

    for var in (*PROXY_VARS, *CA_VARS, "AI_AGENT", "ROTE_AGENT_MODE", *(s.var for s in SIGNALS)):
        monkeypatch.delenv(var, raising=False)
    home = Path(os.environ["ROTE_CLIENT_HOME_DIR"])
    home.mkdir(parents=True, exist_ok=True)
    config = {"mcpServers": {n: {"type": "http", "url": s.url} for n, s in servers.items()}}
    (home / ".claude.json").write_text(json.dumps(config))
    shutil.copytree(REPO / "examples/docs-digest/skill", home / ".claude/skills/docs-digest")
    original = tmp_path / "original-output.json"
    original.write_text(json.dumps(demo.load_reference().skill_output))
    monkeypatch.setenv("FAKE_ORIGINAL_OUTPUT", str(original))
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    w = SetupWorld(work)
    monkeypatch.setattr(
        try_skill,
        "RUNNERS",
        lambda: try_skill.Runners(
            compile=w.compile, run_subject=w.run_subject, executable=lambda: str(FAKE_CLAUDE)
        ),
    )
    return w


def _agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAUDE_CODE_CHILD_SESSION", "1")


def _setup(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, Any, str]:
    status = main(["setup", *argv])
    captured = capsys.readouterr()
    payload = json.loads(captured.out) if captured.out.strip() else None
    return status, payload, captured.err


def _person(monkeypatch: pytest.MonkeyPatch, *answers: Answer) -> ScriptedPrompter:
    prompter = ScriptedPrompter(list(answers))
    monkeypatch.setattr("rote.interact.prompter.prompter_for", lambda mode: prompter)
    return prompter


#: The flags an agent has carried by the time it reaches the skill step.
THROUGH_IMPORT = ("--json", "--only", "mslearn,deepwiki", "--approve", "mcp-import")


# ───────── the agent protocol: each consent point stops, for its own step ─────────


def test_an_agent_is_asked_which_servers_to_import_and_nothing_is_written(
    world: SetupWorld, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _agent(monkeypatch)
    status, payload, err = _setup(capsys, "--json")
    assert status == 2, err
    assert (payload["reason"], payload["missing"]) == ("missing_answer", ["--only"])
    assert payload["step"]["id"] == "mcp-import"
    assert sorted(payload["choices"]) == ["deepwiki", "mslearn"]
    assert payload["ask"].startswith("Which servers should rote import?")
    assert "--skip mcp-import" in payload["answer_with"]
    # What was found, for the agent to relay: hosts, never URLs.
    assert {s["name"] for s in payload["servers"]} == {"mslearn", "deepwiki"}
    assert all("url" not in s for s in payload["servers"])
    assert payload["next"] == [{"command": "rote setup --json --only <mslearn|deepwiki>"}]
    assert load_registry().servers == {}
    assert "Running under Claude Code (CLAUDE_CODE_CHILD_SESSION)" in err


def test_an_agents_yes_is_for_one_step_and_never_covers_the_next(
    world: SetupWorld, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _agent(monkeypatch)
    status, payload, _ = _setup(capsys, "--json", "--only", "mslearn,deepwiki")
    assert (status, payload["reason"]) == (2, "confirmation_required")
    assert payload["missing"] == ["--approve mcp-import"]
    yes, no = payload["next"]
    assert yes["command"].endswith("--approve mcp-import") and yes["when"] == "the user says yes"
    assert no["command"].endswith("--skip mcp-import")
    assert "Register 2 MCP servers" in json.dumps(payload["plan"])
    assert payload["ask"].endswith("Go ahead?")
    assert load_registry().servers == {}

    status, payload, _ = _setup(capsys, *THROUGH_IMPORT)
    # The import applied; the login step's plan is a new consent point.
    assert set(load_registry().servers) == {"mslearn", "deepwiki"}
    assert (status, payload["reason"], payload["step"]["id"]) == (2, "missing_answer", "login")
    status, payload, _ = _setup(capsys, *THROUGH_IMPORT, "--login", "mslearn")
    assert (status, payload["reason"], payload["step"]["id"]) == (
        2,
        "confirmation_required",
        "login",
    )
    assert payload["missing"] == ["--approve login"]


def test_yes_alone_never_spends_and_approve_alone_never_spends(
    world: SetupWorld, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _agent(monkeypatch)
    picked = (
        *THROUGH_IMPORT,
        "--skip",
        "login",
        "--skill",
        str(world.skill),
        "--read-only",
        "mslearn",
        "--read-only",
        "deepwiki",
        "--approve",
        "skill",
    )
    status, payload, _ = _setup(capsys, *picked)
    assert (status, payload["step"]["id"]) == (2, "compile")
    assert payload["missing"] == ["--approve compile", "--live"]
    assert "spend $" in payload["ask"] and "In all, rote try spends" in payload["ask"]
    status, payload, _ = _setup(capsys, *picked, "--approve", "compile")
    assert (status, payload["missing"]) == (2, ["--live"])
    status, payload, _ = _setup(capsys, *picked, "--live")
    assert (status, payload["missing"]) == (2, ["--approve compile"])
    assert world.compiles == []
    status, payload, _ = _setup(capsys, *picked, "--approve", "compile", "--live")
    # Compiled; the install relies on read-only servers, so only the user may approve it.
    assert len(world.compiles) == 1
    assert (status, payload["reason"], payload["step"]["id"]) == (2, "person_required", "install")
    assert payload["next"][0]["command"].startswith("rote skill install ")
    assert payload["next"][-1]["command"].startswith("rote setup ")


def test_the_skill_step_asks_for_each_live_server_and_saves_the_pick(
    world: SetupWorld, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _agent(monkeypatch)
    base = (*THROUGH_IMPORT, "--skip", "login")
    status, payload, _ = _setup(capsys, *base)
    assert (status, payload["missing"]) == (2, ["--skill"])
    # The user's own skill first, rote's demo last.
    assert [s["demo"] for s in payload["skills"]] == [False, True]
    assert payload["choices"][0] == str(world.skill)
    assert "docs-digest (claude-code user)" in payload["ask"]

    status, payload, _ = _setup(capsys, *base, "--skill", str(world.skill))
    assert (status, payload["missing"]) == (2, ["--read-only deepwiki"])
    # A yes/no: the next command passes the switch with the server's name.
    assert payload["next"][0]["command"].endswith("--read-only deepwiki")
    status, payload, _ = _setup(
        capsys, *base, "--skill", str(world.skill), "--read-only", "deepwiki"
    )
    assert payload["missing"] == ["--read-only mslearn"]
    assert not first_run.state_path().exists()
    full = (*base, "--skill", str(world.skill), "--read-only", "deepwiki,mslearn")
    status, payload, _ = _setup(capsys, *full)
    assert (status, payload["reason"], payload["missing"]) == (
        2,
        "confirmation_required",
        ["--approve skill"],
    )
    assert not first_run.state_path().exists()
    assert not any(s.read_only for s in load_registry().servers.values())
    status, payload, _ = _setup(capsys, *full, "--approve", "skill")
    assert payload["step"]["id"] == "compile"
    assert first_run.read_state().skill == world.skill.resolve()
    assert all(s.read_only for s in load_registry().servers.values())


def test_a_login_for_an_agent_hands_over_the_authorization_url(
    world: SetupWorld, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from rote.mcp import pending_login

    started: list[str] = []

    def fake_start(server: str, config: Any, **_: Any) -> pending_login.Started:
        started.append(server)
        record = pending_login.PendingLogin.new(server, config.url)
        return pending_login.Started("authorize", record, authorization_url="https://as/auth")

    monkeypatch.setattr(pending_login, "start_pending_login", fake_start)
    monkeypatch.setattr("webbrowser.open", lambda url: False)
    _agent(monkeypatch)
    status, payload, _ = _setup(capsys, *THROUGH_IMPORT, "--login", "mslearn")
    assert started == [] and payload["reason"] == "confirmation_required"
    status, payload, _ = _setup(capsys, *THROUGH_IMPORT, "--login", "mslearn", "--approve", "login")
    assert started == ["mslearn"]
    assert (status, payload["reason"]) == (2, "authorization_pending")
    (login,) = payload["logins"]
    assert (login["server"], login["authorization_url"]) == ("mslearn", "https://as/auth")
    assert "https://as/auth" in payload["ask"]
    assert payload["next"][0]["command"].startswith("rote mcp login --complete ")
    assert payload["next"][-1]["when"] == "every login above completed"


# ───────── skip ─────────


def test_skipping_import_goes_on_and_skipping_the_skill_ends_setup(
    world: SetupWorld, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _agent(monkeypatch)
    status, payload, _ = _setup(capsys, "--json", "--skip", "mcp-import", "--skip", "login")
    assert (status, payload["step"]["id"]) == (2, "skill")
    assert [s["state"] for s in payload["steps"]] == ["skipped", "skipped"]
    assert load_registry().servers == {}
    status, payload, err = _setup(capsys, "--json", "--skip", "mcp-import,login,skill")
    assert (status, payload["stopped_at"]) == (1, "skill")
    assert "stopped at skill (skipped)" in err


def test_a_step_that_is_not_one_is_refused(
    world: SetupWorld, capsys: pytest.CaptureFixture[str]
) -> None:
    status, payload, _ = _setup(capsys, "--json", "--approve", "everything")
    assert status == 2
    assert payload["error"].startswith("--approve everything: not a step")


# ───────── a person ─────────


def test_a_person_walks_through_every_step_to_the_verdict_and_a_rerun_costs_nothing(
    world: SetupWorld, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    prompter = _person(
        monkeypatch,
        Answer("text", "--work-domain", ""),
        Answer("checkbox", "--only", ["mslearn", "deepwiki"]),
        YES,
        Answer("checkbox", "--login", DEFAULT),
        YES,
        Answer("select", "--skill", DEFAULT),
        Answer("confirm", "--read-only deepwiki", True),
        Answer("confirm", "--read-only mslearn", True),
        YES,
        YES,
        YES,
        YES,
    )
    status, _, err = _setup(capsys)
    assert status == 0, err
    assert prompter.answers == []
    assert len(world.compiles) == 1 and len(world.evals) == 1
    lines = err.splitlines()
    # The pick is explained in one line, and the verdict leads the result.
    assert any(line.startswith("  Recommended: docs-digest (claude-code user): ") for line in lines)
    verdict = next(i for i, line in enumerate(lines) if line.startswith("rote try docs-digest: "))
    assert "Same answer" in lines[verdict]
    assert lines.index("Next:") > verdict
    # Logged in for real against the fixture servers: they need no auth.
    from rote.setup.first_run import needs_login

    assert needs_login(()) == []
    # Resumed: every step reads done, nothing is asked, nothing runs again.
    again = _person(monkeypatch)
    status, _, err = _setup(capsys)
    assert status == 0, err
    assert again.asked == []
    assert err.count("done already") == 6
    assert len(world.compiles) == 1 and len(world.evals) == 1
    assert "Same answer" in err


def test_a_person_who_declines_the_import_goes_on_to_the_next_step(
    world: SetupWorld, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    prompter = _person(
        monkeypatch,
        Answer("text", "--work-domain", ""),
        Answer("checkbox", "--only", ["mslearn", "deepwiki"]),
        NO,
        # Nothing was registered, so the login step is done and asks nothing.
        Answer("select", "--skill", DEFAULT),
        NO,
    )
    status, _, err = _setup(capsys)
    assert prompter.answers == []
    assert "skipped; later: rote mcp import" in err
    # A no to the skill's plan ends setup there: the rest builds on it.
    assert status == 1 and "stopped at skill (not applied)" in err
    assert load_registry().servers == {} and not first_run.state_path().exists()


def test_a_server_the_person_does_not_mark_read_only_stops_setup_before_any_spend(
    world: SetupWorld, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    prompter = _person(
        monkeypatch,
        Answer("text", "--work-domain", ""),
        Answer("checkbox", "--only", ["mslearn", "deepwiki"]),
        YES,
        Answer("checkbox", "--login", []),
        Answer("select", "--skill", DEFAULT),
        Answer("confirm", "--read-only deepwiki", True),
        Answer("confirm", "--read-only mslearn", False),
        YES,
    )
    status, _, err = _setup(capsys)
    assert prompter.answers == []
    assert "mslearn: not marked read-only, so the comparison cannot call it" in err
    assert status == 1
    assert "Checked before the compile: nothing was spent" in err
    assert "rote mcp add mslearn <url> --read-only" in err
    assert world.compiles == []
    # Re-run: the skill step is pending again, so the question comes back.
    assert first_run.SkillStep(first_run.SetupFlags()).status(None).state == "pending"  # type: ignore[arg-type]


def test_rote_try_refuses_before_the_compile_when_a_server_is_not_read_only(
    world: SetupWorld, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from rote.mcp.registry import McpRegistry, McpServerConfig, save_registry

    save_registry(McpRegistry(servers={"mslearn": McpServerConfig(url="http://127.0.0.1:9/mcp")}))
    status = main(["try", str(world.skill), "--json", "--yes", "--live"])
    out = json.loads(capsys.readouterr().out)
    assert status == 1
    assert "rote mcp add mslearn <url> --read-only" in out["error"]
    assert world.compiles == []


def test_the_parser_names_every_step_setup_takes() -> None:
    from rote.cli import _build_parser

    parser = _build_parser()
    action = next(a for a in parser._subparsers._actions if a.dest == "command")  # type: ignore[union-attr]
    description = action.choices["setup"].description
    assert f"Steps: {', '.join(first_run.STEP_IDS)}." in description
