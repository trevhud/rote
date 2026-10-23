"""The three prompters (rote.interact.prompter). The questionary one runs on
prompt_toolkit's pipe input, never a terminal."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from typing import Any

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from rote.interact.prompter import (
    DEFAULT,
    Answer,
    Choice,
    NonInteractivePrompter,
    PromptCancelled,
    PromptRequired,
    QuestionaryPrompter,
    ScriptedPrompter,
)

DOWN, ENTER, SPACE, CTRL_C = "\x1b[B", "\r", " ", "\x03"

CHOICES = [Choice("a", "A"), Choice("b", "B", disabled="not importable"), Choice("c", "C")]


@pytest.fixture
def keys() -> Iterator[Any]:
    with create_pipe_input() as pipe:
        yield pipe


def _q(pipe: Any, text: str) -> QuestionaryPrompter:
    pipe.send_text(text)
    return QuestionaryPrompter(input=pipe, output=DummyOutput())


# ── Non-interactive ──


def test_without_yes_every_question_names_its_flag() -> None:
    p = NonInteractivePrompter(why="stdin is not a terminal")
    with pytest.raises(PromptRequired) as e:
        p.select("Pick one?", CHOICES, flag="--pick", default="a")
    assert e.value.flag == "--pick"
    assert e.value.choices == ("a", "c")  # disabled options are not offered
    assert str(e.value) == (
        "Pick one: pass --pick <a|c> (stdin is not a terminal; or --yes to take the default)"
    )
    with pytest.raises(PromptRequired, match="pass --ok"):
        p.confirm("OK?", flag="--ok", default=True)


def test_with_yes_defaults_are_taken() -> None:
    p = NonInteractivePrompter(why="--yes was passed", accept_defaults=True)
    assert p.select("Pick?", CHOICES, flag="--pick", default="c") == "c"
    assert p.confirm("OK?", flag="--ok", default=False) is False
    assert p.text("Name?", flag="--name", default="") == ""
    # An empty checkbox default is a real answer: import nothing.
    assert p.checkbox("Which?", CHOICES, flag="--only", default=()) == []
    assert p.checkbox("Which?", CHOICES, flag="--only", default=["a", "c"]) == ["a", "c"]


def test_with_yes_a_question_without_a_default_still_stops() -> None:
    p = NonInteractivePrompter(why="--yes was passed", accept_defaults=True)
    with pytest.raises(PromptRequired, match="this question has no default"):
        p.select("Pick?", CHOICES, flag="--pick")
    with pytest.raises(PromptRequired, match="pass --name"):
        p.text("Name?", flag="--name")


def test_non_interactive_never_reads_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
    class Exploding:
        def read(self, *a: Any) -> str:
            raise AssertionError("stdin was read")

        readline = read

        def isatty(self) -> bool:
            return False

    monkeypatch.setattr(sys, "stdin", Exploding())
    p = NonInteractivePrompter(why="x", accept_defaults=True)
    assert p.select("Pick?", CHOICES, flag="--pick", default="a") == "a"


def test_a_default_must_be_an_enabled_choice() -> None:
    p = NonInteractivePrompter(why="x", accept_defaults=True)
    with pytest.raises(ValueError, match="is disabled"):
        p.select("Pick?", CHOICES, flag="--pick", default="b")
    with pytest.raises(ValueError, match="not one of the choices"):
        p.select("Pick?", CHOICES, flag="--pick", default="z")
    with pytest.raises(ValueError, match="not an enabled choice"):
        p.checkbox("Which?", CHOICES, flag="--only", default=["b"])


# ── Scripted ──


def test_scripted_answers_are_checked_against_the_question() -> None:
    p = ScriptedPrompter([Answer("select", "--pick", "c"), Answer("confirm", "--ok", DEFAULT)])
    assert p.select("Pick?", CHOICES, flag="--pick", default="a") == "c"
    assert p.confirm("OK?", flag="--ok", default=True) is True
    assert p.asked == [("select", "--pick", "Pick?"), ("confirm", "--ok", "OK?")]


def test_scripted_fails_where_the_flow_diverges() -> None:
    p = ScriptedPrompter([Answer("select", "--runtime", "dbos")])
    with pytest.raises(AssertionError, match="reached select question 'Pick\\?' \\(--pick\\)"):
        p.select("Pick?", CHOICES, flag="--pick")
    with pytest.raises(AssertionError, match="unscripted"):
        ScriptedPrompter([]).confirm("OK?", flag="--ok", default=True)
    with pytest.raises(AssertionError, match="not an enabled choice"):
        ScriptedPrompter([Answer("select", "--pick", "b")]).select("Pick?", CHOICES, flag="--pick")


# ── questionary, on pipe input ──


def test_questionary_select_moves_and_starts_at_the_default(keys: Any) -> None:
    assert _q(keys, ENTER).select("Pick?", CHOICES, flag="--pick", default="c") == "c"


def test_questionary_select_skips_disabled_choices(keys: Any) -> None:
    assert _q(keys, DOWN + ENTER).select("Pick?", CHOICES, flag="--pick", default="a") == "c"


def test_questionary_checkbox_preselects_the_default(keys: Any) -> None:
    # Toggle a on, move past disabled b to c, toggle c (preselected) off.
    picked = _q(keys, SPACE + DOWN + SPACE + ENTER).checkbox(
        "Which?", CHOICES, flag="--only", default=["c"]
    )
    assert picked == ["a"]


def test_questionary_checkbox_enter_keeps_the_default(keys: Any) -> None:
    assert _q(keys, ENTER).checkbox("Which?", CHOICES, flag="--only", default=["c"]) == ["c"]


def test_questionary_confirm_and_text(keys: Any) -> None:
    assert _q(keys, "n").confirm("OK?", flag="--ok", default=True) is False


def test_questionary_text_validates(keys: Any) -> None:
    def no_spaces(v: str) -> str | None:
        return "no spaces" if " " in v else None

    # "a b" fails validation, so the prompt stays; erase two chars, then "c".
    answer = _q(keys, "a b" + ENTER + "\x7f\x7f" + "c" + ENTER).text(
        "Name?", flag="--name", validate=no_spaces
    )
    assert answer == "ac"


def test_questionary_ctrl_c_is_a_cancel_not_a_none(keys: Any) -> None:
    with pytest.raises(PromptCancelled):
        _q(keys, CTRL_C).select("Pick?", CHOICES, flag="--pick", default="a")


def test_questionary_draws_on_stderr_by_default() -> None:
    p = QuestionaryPrompter()
    assert p._output.stdout is sys.stderr  # stdout stays the command's result


def test_the_piped_stdin_trap_this_layer_exists_for(keys: Any) -> None:
    """What happens without the Mode check: questionary reads piped bytes as
    keys and returns an answer, with no error. Here a lone Enter "chooses"
    the default. This is why prompter_for decides before any prompt, and
    why NonInteractivePrompter never touches stdin."""
    import questionary

    keys.send_text(ENTER)
    answer = questionary.select("Pick?", choices=["a", "c"], input=keys, output=DummyOutput()).ask()
    assert answer == "a"
