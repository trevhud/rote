"""Mode is decided once, from injected streams and env (rote.interact.mode/agents)."""

from __future__ import annotations

import io
from typing import Any

import pytest

from rote.interact.agents import detect
from rote.interact.mode import Mode
from rote.interact.prompter import NonInteractivePrompter, QuestionaryPrompter, prompter_for


class _Tty(io.StringIO):
    def isatty(self) -> bool:
        return True


def _mode(env: dict[str, str] | None = None, **kw: Any) -> Mode:
    streams = {"stdin": _Tty(), "stdout": _Tty(), "stderr": _Tty()}
    streams.update({k: kw.pop(k) for k in list(kw) if k in streams})
    return Mode.resolve(env=env or {}, **streams, **kw)


@pytest.mark.parametrize(
    ("env", "client", "role", "var"),
    [
        ({"AI_AGENT": "claude-code_2-1-302_agent"}, "claude-code", "caller", "AI_AGENT"),
        ({"AI_AGENT": "github_copilot_vscode_agent"}, "vscode", "caller", "AI_AGENT"),
        ({"AI_AGENT": "someone-new"}, "someone-new", "caller", "AI_AGENT"),
        ({"CLAUDE_CODE_CHILD_SESSION": "1"}, "claude-code", "caller", "CLAUDE_CODE_CHILD_SESSION"),
        ({"CLAUDECODE": "1"}, "claude-code", "host", "CLAUDECODE"),
        ({"CODEX_THREAD_ID": "t"}, "codex", "caller", "CODEX_THREAD_ID"),
        ({"CURSOR_TRACE_ID": "x"}, "cursor", "host", "CURSOR_TRACE_ID"),
        ({"CURSOR_AGENT": "1"}, "cursor", "caller", "CURSOR_AGENT"),
        ({"GEMINI_CLI": "1"}, "gemini", "caller", "GEMINI_CLI"),
        ({"AGENT": "amp"}, "amp", "caller", "AGENT"),
        ({"ZED_TERM": "true"}, "zed", "host", "ZED_TERM"),
        # A caller outranks a host, whatever the order of the table.
        ({"CLAUDECODE": "1", "CODEX_SANDBOX": "seatbelt"}, "codex", "caller", "CODEX_SANDBOX"),
    ],
)
def test_detect_names_the_client_role_and_variable(
    env: dict[str, str], client: str, role: str, var: str
) -> None:
    found = detect(env)
    assert found is not None
    assert (found.client, found.role, found.var) == (client, role, var)


@pytest.mark.parametrize(
    "env",
    [
        {},
        {"CLAUDECODE": "0"},  # value must be 1
        {"AGENT": "someone"},  # AGENT counts only as amp
        {"AI_AGENT": "  "},
        {"CODEX_HOME": "/x"},  # config location, not a launcher
    ],
)
def test_detect_ignores_what_does_not_name_a_client(env: dict[str, str]) -> None:
    assert detect(env) is None


def test_ai_agent_prefix_needs_a_boundary() -> None:
    found = detect({"AI_AGENT": "codexfoo"})
    assert found is not None and found.client == "codexfoo"


def test_a_person_at_terminals_is_interactive() -> None:
    mode = _mode()
    assert mode.interactive
    assert mode.why_not_interactive is None
    assert isinstance(prompter_for(mode), QuestionaryPrompter)


@pytest.mark.parametrize(
    ("kw", "env", "reason"),
    [
        ({"json": True}, {}, "--json was passed"),
        ({"yes": True}, {}, "--yes was passed"),
        ({}, {"ROTE_PROMPT_DISABLED": "1"}, "ROTE_PROMPT_DISABLED is set"),
        ({}, {"CLAUDE_CODE_CHILD_SESSION": "1"}, "running under claude-code"),
        ({"agent_mode": "yes"}, {}, "--agent-mode yes was passed"),
        ({"stdin": io.StringIO()}, {}, "stdin is not a terminal"),
        ({"stderr": io.StringIO()}, {}, "stderr is not a terminal"),
    ],
)
def test_each_condition_alone_turns_prompts_off(
    kw: dict[str, Any], env: dict[str, str], reason: str
) -> None:
    mode = _mode(env, **kw)
    assert not mode.interactive
    assert mode.why_not_interactive is not None and reason in mode.why_not_interactive
    prompter = prompter_for(mode)
    assert isinstance(prompter, NonInteractivePrompter)
    assert prompter.accept_defaults is bool(kw.get("yes"))


def test_stdout_is_not_part_of_the_decision() -> None:
    """Prompts draw on stderr, so `rote init | tee log` can still ask."""
    assert _mode(stdout=io.StringIO()).interactive


def test_a_host_terminal_is_still_a_person() -> None:
    """CLAUDECODE=1 is set in IDE terminals, where a person types."""
    mode = _mode({"CLAUDECODE": "1"})
    assert mode.detected is not None and mode.detected.client == "claude-code"
    assert not mode.agent
    assert mode.interactive


def test_agent_mode_no_overrides_a_detected_caller() -> None:
    env = {"CLAUDE_CODE_CHILD_SESSION": "1"}
    assert not _mode(env).interactive
    assert _mode(env, agent_mode="no").interactive
    assert _mode({"ROTE_AGENT_MODE": "no", **env}).interactive


def test_agent_mode_env_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match="ROTE_AGENT_MODE='maybe'"):
        _mode({"ROTE_AGENT_MODE": "maybe"})


def test_color_follows_stderr_and_no_color() -> None:
    assert _mode().color
    assert not _mode({"NO_COLOR": "1"}).color
    assert not _mode(stderr=io.StringIO()).color


def test_to_dict_reports_the_decision() -> None:
    assert _mode({"CLAUDE_CODE_CHILD_SESSION": "1"}).to_dict() == {
        "interactive": False,
        "why_not_interactive": (
            "running under claude-code (CLAUDE_CODE_CHILD_SESSION); "
            "--agent-mode no if a person is here"
        ),
        "agent": True,
        "detected_client": "claude-code",
        "detected_role": "caller",
        "detected_by": "CLAUDE_CODE_CHILD_SESSION",
    }
