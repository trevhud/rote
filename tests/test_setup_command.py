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
import shlex
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


def _yes(capsys: pytest.CaptureFixture[str], payload: Any) -> tuple[int, Any, str]:
    """Run a stop's yes command, exactly as an agent would."""
    return _setup(capsys, *shlex.split(payload["next"][0]["command"])[2:])


def _through_import(capsys: pytest.CaptureFixture[str]) -> tuple[int, Any, str]:
    """Pick both servers and approve the import: what every later test starts from."""
    _, payload, _ = _setup(capsys, "--json", "--only", "mslearn,deepwiki")
    result = _yes(capsys, payload)
    assert set(load_registry().servers) == {"mslearn", "deepwiki"}
    return result


def _through_skill(capsys: pytest.CaptureFixture[str]) -> tuple[int, Any, str]:
    """Import both servers, then pick and approve the user's docs-digest:
    the login step comes next, scoped to the servers it names."""
    _through_import(capsys)
    _, payload, _ = _setup(capsys, "--json", "--skill", "docs-digest@claude-code-user")
    assert payload["step"]["id"] == "skill"
    return _yes(capsys, payload)


def _install_as_the_user(monkeypatch: pytest.MonkeyPatch, capsys: Any, command: str) -> None:
    """The user runs the install in their own terminal."""
    from rote.interact import prompter as prompting

    real = prompting.prompter_for
    monkeypatch.delenv("CLAUDE_CODE_CHILD_SESSION")
    _person(monkeypatch, YES, Answer("confirm", "--save-allow-rule", False))
    assert main(shlex.split(command)[1:]) == 0
    capsys.readouterr()
    monkeypatch.setattr(prompting, "prompter_for", real)
    _agent(monkeypatch)


# ───────── the agent protocol: each consent point stops, for its own step ─────────


@pytest.fixture
def unmarked(monkeypatch: pytest.MonkeyPatch) -> None:
    """Servers that mark none of their tools read-only: only the user's word
    can let the comparison call them."""

    async def _nothing_marked(names: list[str]) -> tuple[list[str], list[str], dict[str, str]]:
        return [], [], {n: "no tools declare readOnlyHint" for n in names}

    monkeypatch.setattr("rote.eval.baseline.read_only_allowlist", _nothing_marked)


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


def test_an_agents_yes_is_for_one_plan_and_never_covers_the_next(
    world: SetupWorld, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _agent(monkeypatch)
    status, payload, _ = _setup(capsys, "--json", "--only", "mslearn,deepwiki")
    assert (status, payload["reason"]) == (2, "confirmation_required")
    code = payload["approval"]["code"]
    assert len(code) == 12 and payload["approval"]["step"] == "mcp-import"
    assert payload["missing"] == [f"--approve mcp-import={code}"]
    yes, no = payload["next"]
    assert yes["command"] == f"rote setup --json --approve mcp-import={code}"
    assert no["command"] == "rote setup --json --skip mcp-import"
    assert "Register 2 MCP servers" in json.dumps(payload["plan"])
    assert payload["ask"].endswith("Go ahead?")
    assert load_registry().servers == {}
    # A bare step name approves nothing: the code is what ties a yes to a plan.
    status, payload, _ = _setup(capsys, "--json", "--approve", "mcp-import")
    assert status == 2 and "an approval names the plan it approves" in payload["error"]

    status, payload, _ = _through_import(capsys)
    # The import applied; the skill comes next, before any login.
    assert (status, payload["reason"], payload["step"]["id"]) == (2, "missing_answer", "skill")
    status, payload, _ = _setup(capsys, "--json", "--skill", "docs-digest@claude-code-user")
    assert (status, payload["reason"], payload["step"]["id"]) == (
        2,
        "confirmation_required",
        "skill",
    )
    status, payload, _ = _yes(capsys, payload)
    # The skill's plan applied; the login step's plan is a new consent point,
    # offering only what docs-digest names.
    assert (status, payload["reason"], payload["step"]["id"]) == (2, "missing_answer", "login")
    assert sorted(payload["choices"]) == ["deepwiki", "mslearn"]
    status, payload, _ = _setup(capsys, "--json", "--login", "mslearn")
    assert (status, payload["reason"], payload["step"]["id"]) == (
        2,
        "confirmation_required",
        "login",
    )
    assert payload["missing"] == [f"--approve login={payload['approval']['code']}"]
    # The login plan in the user's words.
    assert payload["ask"] == (
        "Log in to mslearn? rote opens a browser page for each one for you to approve, and "
        "waits up to 10 min."
    )
    # The import's approval, replayed, does not approve the login.
    status, payload, _ = _setup(capsys, "--json", "--approve", f"mcp-import={code}")
    assert (status, payload["step"]["id"]) == (2, "login")


def test_yes_alone_never_spends_and_approve_alone_never_spends(
    world: SetupWorld, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _agent(monkeypatch)
    _through_import(capsys)
    _, payload, _ = _setup(
        capsys,
        "--json",
        "--skip",
        "login",
        "--skill",
        str(world.skill),
        "--read-only",
        "mslearn,deepwiki",
    )
    status, payload, _ = _yes(capsys, payload)
    assert (status, payload["step"]["id"]) == (2, "compile")
    code = payload["approval"]["code"]
    assert payload["missing"] == [f"--approve compile={code}", "--live"]
    assert payload["next"][0]["command"] == f"rote setup --json --approve compile={code} --live"
    assert "spend $" in payload["ask"] and "In all, rote try spends" in payload["ask"]
    status, payload, _ = _setup(capsys, "--json", "--approve", f"compile={code}")
    assert (status, payload["missing"]) == (2, [f"--approve compile={code}", "--live"])
    status, payload, _ = _setup(capsys, "--json", "--live")
    assert (status, payload["missing"]) == (2, [f"--approve compile={code}"])
    assert world.compiles == []
    status, payload, _ = _yes(
        capsys,
        payload | {"next": [{"command": f"rote setup --json --approve compile={code} --live"}]},
    )
    # Compiled; the install relies on read-only servers, so only the user may approve it.
    assert len(world.compiles) == 1
    assert (status, payload["reason"], payload["step"]["id"]) == (2, "person_required", "install")
    assert payload["next"][0]["command"].startswith("rote skill install ")
    assert payload["next"][-1]["command"] == "rote setup --json"


def test_a_replayed_approval_never_spends_twice(
    world: SetupWorld, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _agent(monkeypatch)
    _through_import(capsys)
    _, payload, _ = _setup(
        capsys,
        "--json",
        "--skip",
        "login",
        "--skill",
        str(world.skill),
        "--read-only",
        "mslearn,deepwiki",
        "--trials",
        "1",
    )
    _, payload, _ = _yes(capsys, payload)
    status, payload, _ = _yes(capsys, payload)
    assert (status, payload["reason"]) == (2, "person_required")
    _install_as_the_user(monkeypatch, capsys, payload["next"][0]["command"])
    status, eval_stop, _ = _setup(capsys, "--json")
    assert eval_stop["step"]["id"] == "eval"
    status, payload, err = _yes(capsys, eval_stop)
    assert status == 0, err
    assert len(world.evals) == 1
    # The result is deleted and the very same command is run again: it stops,
    # says what was already spent, and asks with a new code.
    (world.out / try_skill.RESULT_FILE).unlink()
    status, payload, _ = _yes(capsys, eval_stop)
    assert (status, payload["reason"], payload["step"]["id"]) == (
        2,
        "confirmation_required",
        "eval",
    )
    assert payload["approval"]["code"] != eval_stop["approval"]["code"]
    assert "Already spent here: $" in payload["ask"]
    assert len(world.evals) == 1


def test_a_used_approval_does_not_apply_the_same_plan_again(
    world: SetupWorld, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from rote.mcp.registry import registry_path

    _agent(monkeypatch)
    _, import_stop, _ = _setup(capsys, "--json", "--only", "mslearn,deepwiki")
    _yes(capsys, import_stop)
    assert set(load_registry().servers) == {"mslearn", "deepwiki"}
    # The registry is gone, so the import's plan is exactly the one approved
    # before. Replaying that approval does not write it again: it asks.
    registry_path().unlink()
    status, payload, _ = _yes(capsys, import_stop)
    assert (status, payload["step"]["id"], payload["reason"]) == (
        2,
        "mcp-import",
        "confirmation_required",
    )
    assert payload["plan"] == import_stop["plan"]
    assert payload["approval"]["code"] != import_stop["approval"]["code"]
    assert load_registry().servers == {}


def test_an_approval_for_one_skill_does_not_approve_another(
    world: SetupWorld, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _agent(monkeypatch)
    _through_import(capsys)
    _, payload, _ = _setup(
        capsys,
        "--json",
        "--skip",
        "login",
        "--skill",
        str(world.skill),
        "--read-only",
        "mslearn,deepwiki",
    )
    _, compile_stop, _ = _yes(capsys, payload)
    assert compile_stop["step"]["id"] == "compile"
    # The user picks rote's demo instead, and approves that pick.
    _, payload, _ = _setup(capsys, "--json", "--skill", "docs-digest@demo")
    assert payload["step"]["id"] == "skill"
    _, payload, _ = _yes(capsys, payload)
    assert first_run.read_state().skill == demo.skill_dir().resolve()
    # The first skill's compile approval, replayed: nothing compiles.
    status, payload, _ = _yes(capsys, compile_stop)
    assert (status, payload["step"]["id"], payload["reason"]) == (
        2,
        "compile",
        "confirmation_required",
    )
    assert payload["approval"]["code"] != compile_stop["approval"]["code"]
    assert world.compiles == []


def test_the_skill_step_asks_for_each_live_server_and_saves_the_pick(
    unmarked: None,
    world: SetupWorld,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _agent(monkeypatch)
    _through_import(capsys)
    status, payload, _ = _setup(capsys, "--json", "--skip", "login")
    assert (status, payload["missing"]) == (2, ["--skill"])
    # The user's own skill first, rote's demo last.
    assert [s["demo"] for s in payload["skills"]] == [False, True]
    # The demo's hint says what is true: rote try compiles it live like any skill.
    assert payload["skills"][1]["reason"] == (
        "the demo rote try runs: compiled and compared live, on public servers that need no login"
    )
    # Short keys, never paths: two skills share a name, so each says where it is.
    assert payload["choices"] == ["docs-digest@claude-code-user", "docs-digest@demo"]
    assert payload["skills"][0]["path"] == str(world.skill)
    assert "docs-digest@claude-code-user (claude-code user)" in payload["ask"]

    status, payload, _ = _setup(capsys, "--json", "--skill", "docs-digest@claude-code-user")
    assert (status, payload["missing"]) == (2, ["--read-only deepwiki"])
    assert payload["ask"] == (
        "Let rote call any tool on deepwiki (127.0.0.1) without asking? Say yes only if none "
        "of its tools can change anything."
    )
    # A yes/no: the next command passes the switch with the server's name.
    assert payload["next"][0]["command"] == "rote setup --json --read-only deepwiki"
    status, payload, _ = _yes(capsys, payload)
    assert payload["missing"] == ["--read-only mslearn"]
    status, payload, _ = _yes(capsys, payload)
    assert (status, payload["reason"]) == (2, "confirmation_required")
    assert first_run.read_state().skill is None
    assert not any(s.read_only for s in load_registry().servers.values())
    status, payload, _ = _yes(capsys, payload)
    assert payload["step"]["id"] == "compile"
    assert first_run.read_state().skill == world.skill.resolve()
    assert all(s.read_only for s in load_registry().servers.values())


def test_yes_never_vouches_for_a_server(
    unmarked: None,
    world: SetupWorld,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _agent(monkeypatch)
    status, payload, _ = _setup(
        capsys,
        "--json",
        "--yes",
        "--only",
        "mslearn,deepwiki",
        "--skip",
        "login",
        "--skill",
        "docs-digest@claude-code-user",
    )
    assert (status, payload["reason"], payload["missing"]) == (
        2,
        "missing_answer",
        ["--read-only deepwiki"],
    )
    assert "--yes does not answer this" in payload["message"]
    assert world.compiles == [] and first_run.read_state().skill is None


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
    _through_skill(capsys)
    status, payload, _ = _setup(capsys, "--json", "--login", "mslearn")
    assert started == [] and payload["reason"] == "confirmation_required"
    status, payload, _ = _yes(capsys, payload)
    assert started == ["mslearn"]
    assert (status, payload["reason"]) == (2, "authorization_pending")
    (login,) = payload["logins"]
    assert (login["server"], login["authorization_url"]) == ("mslearn", "https://as/auth")
    assert "https://as/auth" in payload["ask"]
    assert payload["next"][0]["command"].startswith("rote mcp login --complete ")
    assert payload["next"][-1] == {
        "command": "rote setup --json",
        "when": "every login above completed",
    }


# ───────── skip ─────────


def test_skipping_import_goes_on_and_skipping_the_skill_ends_setup(
    world: SetupWorld, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _agent(monkeypatch)
    status, payload, _ = _setup(capsys, "--json", "--skip", "mcp-import", "--skip", "login")
    assert (status, payload["step"]["id"]) == (2, "skill")
    assert [s["state"] for s in payload["steps"]] == ["skipped"]
    assert load_registry().servers == {}
    status, payload, err = _setup(capsys, "--json", "--skip", "mcp-import,login,skill")
    assert (status, payload["stopped_at"]) == (1, "skill")
    assert "stopped at skill (skipped)" in err


def test_a_step_that_is_not_one_is_refused(
    world: SetupWorld, capsys: pytest.CaptureFixture[str]
) -> None:
    status, payload, _ = _setup(capsys, "--json", "--approve", "everything=abc")
    assert status == 2
    assert payload["error"].startswith("--approve everything=abc: not a step")


# ───────── a person ─────────


def test_a_person_walks_through_every_step_to_the_verdict_and_a_rerun_costs_nothing(
    unmarked: None,
    world: SetupWorld,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompter = _person(
        monkeypatch,
        Answer("text", "--work-domain", ""),
        Answer("checkbox", "--only", ["mslearn", "deepwiki"]),
        YES,
        Answer("select", "--skill", DEFAULT),
        Answer("confirm", "--read-only deepwiki", True),
        Answer("confirm", "--read-only mslearn", True),
        YES,
        Answer("checkbox", "--login", DEFAULT),
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
    # No pick is recommended by rank: the pick says what compiles well, and
    # the verdict leads the result.
    assert not any("Recommended" in line for line in lines)
    assert any(line.startswith("  Pick a skill whose work is mostly tool calls") for line in lines)
    # The login was offered for the two servers docs-digest names, both checked.
    login_q = next(q for q in prompter.asked if q[1] == "--login")
    assert login_q[2].startswith("docs-digest uses these. Log in to which")
    verdict = next(i for i, line in enumerate(lines) if line.startswith("rote try docs-digest: "))
    assert "Same answer" in lines[verdict]
    assert lines.index("Next:") > verdict
    # Inside setup, measuring again is setup's own command, and it asks first.
    assert any("and run rote setup again (it asks before it spends again)" in ln for ln in lines)
    assert not any("run rote try again" in ln for ln in lines)
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
    assert load_registry().servers == {} and first_run.read_state().skill is None


def test_a_server_the_person_does_not_mark_read_only_stops_setup_before_any_spend(
    unmarked: None,
    world: SetupWorld,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompter = _person(
        monkeypatch,
        Answer("text", "--work-domain", ""),
        Answer("checkbox", "--only", ["mslearn", "deepwiki"]),
        YES,
        Answer("select", "--skill", DEFAULT),
        Answer("confirm", "--read-only deepwiki", True),
        Answer("confirm", "--read-only mslearn", False),
        YES,
        Answer("checkbox", "--login", []),
    )
    status, _, err = _setup(capsys)
    assert prompter.answers == []
    assert "mslearn: not marked read-only, so the comparison cannot call it" in err
    assert status == 1
    assert "Checked before the compile: nothing was spent" in err
    assert "rote setup --read-only mslearn" in err
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
    assert "which is not registered read-only, and its tools could not be checked" in out["error"]
    assert "rote setup --read-only mslearn" in out["error"]
    assert world.compiles == []


def test_the_parser_names_every_step_setup_takes() -> None:
    from rote.cli import _build_parser

    parser = _build_parser()
    action = next(a for a in parser._subparsers._actions if a.dest == "command")  # type: ignore[union-attr]
    description = action.choices["setup"].description
    assert f"Steps: {', '.join(first_run.STEP_IDS)}." in description


# ───────── answers are remembered, so next stays short ─────────


def test_next_is_short_and_carries_only_the_new_answer(
    unmarked: None,
    world: SetupWorld,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _agent(monkeypatch)
    mode = ("--agent-mode", "yes", "--json")
    status, payload, _ = _setup(capsys, *mode, "--only", "mslearn")
    code = payload["approval"]["code"]
    assert payload["next"][0]["command"] == (
        f"rote setup --agent-mode yes --json --approve mcp-import={code}"
    )
    # Run exactly that: the pick is remembered, not asked again.
    status, payload, _ = _yes(capsys, payload)
    assert set(load_registry().servers) == {"mslearn"}
    assert payload["step"]["id"] == "skill"
    status, payload, _ = _setup(capsys, *mode, "--skill", "docs-digest@claude-code-user")
    assert payload["next"][0]["command"] == "rote setup --agent-mode yes --json --read-only mslearn"
    status, payload, _ = _yes(capsys, payload)
    code = payload["approval"]["code"]
    assert payload["next"][0]["command"] == (
        f"rote setup --agent-mode yes --json --approve skill={code}"
    )
    assert all("/" not in n["command"] for n in payload["next"])
    status, payload, _ = _yes(capsys, payload)
    assert payload["next"] == [{"command": "rote setup --agent-mode yes --json --login <mslearn>"}]


def test_reset_forgets_the_saved_answers(
    world: SetupWorld, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _agent(monkeypatch)
    _setup(capsys, "--json", "--only", "mslearn")
    assert first_run.read_state().answers["only"] == ["mslearn"]
    status, payload, err = _setup(capsys, "--json", "--reset")
    assert "forgot the saved answers" in err
    assert (payload["reason"], payload["missing"]) == ("missing_answer", ["--only"])


# ───────── a partial pick is not asked again ─────────


def test_a_partial_import_and_a_declined_login_are_not_offered_again(
    world: SetupWorld, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from rote.interact.prompter import PromptCancelled

    stop = Answer("confirm", "--yes", PromptCancelled("stop here"))
    _person(
        monkeypatch,
        Answer("text", "--work-domain", ""),
        Answer("checkbox", "--only", ["mslearn"]),
        YES,
        Answer("select", "--skill", DEFAULT),
        YES,
        Answer("checkbox", "--login", []),
        stop,
    )
    assert _setup(capsys)[0] == 130
    assert set(load_registry().servers) == {"mslearn"}
    again = _person(monkeypatch, stop)
    _, _, err = _setup(capsys)
    # Only the compile is asked: deepwiki (not picked) and mslearn's login
    # (declined) are not offered again.
    assert [q[1] for q in again.asked] == ["--yes"]
    assert "1 you did not pick (rote mcp import adds them)" in err
    assert "you chose not to log in to mslearn" in err


def test_setup_says_why_rote_inits_defaults_are_not_a_step(
    world: SetupWorld, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _agent(monkeypatch)
    _, _, err = _setup(capsys, "--json")
    assert "rote init's defaults are not a step here: rote try pins its own runtime" in err


def test_a_server_that_marks_its_read_only_tools_is_not_asked_about(
    world: SetupWorld, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fixture servers mark every tool read-only, as Atlassian's do: the
    comparison calls only those, so nobody is asked to vouch for the server
    and the registry is not changed to say it only reads."""
    _agent(monkeypatch)
    _through_import(capsys)
    _setup(capsys, "--json", "--skip", "login")
    status, payload, err = _setup(capsys, "--json", "--skill", "docs-digest@claude-code-user")
    assert (status, payload["reason"]) == (2, "confirmation_required")
    assert "--read-only" not in json.dumps(payload["next"])
    assert "deepwiki: the comparison calls only the 1 tool(s) it marks read-only" in err
    status, payload, _ = _yes(capsys, payload)
    assert payload["step"]["id"] == "compile"
    assert not any(s.read_only for s in load_registry().servers.values())


# ───────── logins come after the pick, for that skill's servers ─────────


def _needs(name: str) -> first_run.NeedsLogin:
    from rote.mcp.registry import McpServerConfig

    return first_run.NeedsLogin(
        name, McpServerConfig(url=f"https://{name}.example/mcp"), f"{name}.example", "none", False
    )


def test_a_skill_that_names_servers_is_offered_only_their_logins(tmp_path: Path) -> None:
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        "---\nname: s\ndescription: d\n---\nSearch with the `crm` MCP server.\n"
    )
    offered, named = first_run.skill_logins(skill, [_needs("crm"), _needs("figma"), _needs("pd")])
    assert ([n.name for n in offered], named) == (["crm"], True)


def test_a_skill_that_names_no_server_is_offered_every_login_to_pick_from(tmp_path: Path) -> None:
    """Atlassian's skills call getJiraIssue(...) and never name a server: the
    tool's server is unknown until a login, so whoever answers reads the
    skill and picks."""
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: s\ndescription: d\n---\nCall getJiraIssue(...).\n")
    pending = [_needs("atlassian"), _needs("figma")]
    offered, named = first_run.skill_logins(skill, pending)
    assert ([n.name for n in offered], named) == (["atlassian", "figma"], False)
    assert first_run.skill_logins(None, pending) == ([], False)


def test_identical_copies_of_a_skill_are_offered_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    def copy(where: str, body: str) -> Any:
        d = tmp_path / where / "audit"
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(body)
        root = SimpleNamespace(scope="user", client=where, plugin=None)
        return SimpleNamespace(
            name="audit",
            path=d,
            found_in=(root,),
            readiness=SimpleNamespace(state="ready", reason="r"),
        )

    same = "---\nname: audit\ndescription: d\n---\nbody\n"
    skills = [copy("claude", same), copy("agents", same), copy("codex", same + "more\n")]
    monkeypatch.setattr("rote.skills.discover", lambda: SimpleNamespace(skills=skills))
    # The two identical copies are one choice; the one that differs is its own.
    assert [o.where for o in first_run.offered_skills(10)] == ["claude user", "codex user"]


def test_own_skills_are_offered_before_plugin_skills(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    def skill(name: str, plugin: str | None) -> Any:
        root = SimpleNamespace(
            scope="plugin" if plugin else "user", client="claude-code", plugin=plugin
        )
        return SimpleNamespace(
            name=name,
            path=Path(f"/s/{name}"),
            found_in=(root,),
            readiness=SimpleNamespace(state="ready", reason="r"),
        )

    # discover() ranks the plugin's skill first; setup still lists the user's own first.
    ranked = [skill("aa-plugin", "acme-tools"), skill("zz-mine", None)]
    monkeypatch.setattr("rote.skills.discover", lambda: SimpleNamespace(skills=ranked))
    assert [o.name for o in first_run.offered_skills(10)] == ["zz-mine", "aa-plugin"]
    assert [o.name for o in first_run.offered_skills(1)] == ["zz-mine"]


def test_a_machine_with_no_client_config_is_told_so_and_the_demo_needs_nothing(
    world: SetupWorld, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A brand-new user: no AI-client config, so nothing to import, and rote's
    demo calls public servers that need no registration, so nothing waits."""
    (Path(os.environ["ROTE_CLIENT_HOME_DIR"]) / ".claude.json").unlink()
    _agent(monkeypatch)
    status, payload, err = _setup(capsys, "--json", "--skill", "docs-digest@demo")
    assert (status, payload["step"]["id"]) == (2, "skill")
    assert "no MCP servers found in your AI clients' configs to import" in err
    assert "0 server(s) registered" not in err
    assert "deepwiki: a public server rote's demo calls, no login needed" in err
    assert "the compile waits for" not in err
