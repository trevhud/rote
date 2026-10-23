"""Plans: every side effect previewed, confirmed by the right flags, applied in order."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import pytest

from rote.interact.mode import Mode
from rote.interact.output import Output
from rote.interact.plan import (
    Action,
    ConfirmationRequired,
    Contact,
    FileWrite,
    Launch,
    Plan,
    Spend,
    apply_plan,
    atomic_write_text,
    confirm_plan,
    render_plan,
    write_file_action,
)
from rote.interact.prompter import Answer, NonInteractivePrompter, ScriptedPrompter
from rote.interact.report import action_required


def _noop() -> None:
    return None


def _action(id: str, *effects: object, run: object = _noop) -> Action:
    return Action(id=id, title=id, effects=effects, run=run)  # type: ignore[arg-type]


def _write(tmp_path: Path, replaces: bool = False) -> Action:
    path = tmp_path / "f.txt"
    if replaces:
        path.write_text("old", encoding="utf-8")
    return write_file_action(
        id="w", title="write", path=path, content="new", summary="s", replaces=replaces
    )


SPEND = Spend(low_usd=0.9, high_usd=1.7, payer="your Claude subscription", flag="--live")
NOBODY = NonInteractivePrompter(why="stdin is not a terminal")


def test_non_interactive_needs_yes(tmp_path: Path) -> None:
    plan = Plan("p", (_write(tmp_path),))
    with pytest.raises(ConfirmationRequired) as e:
        confirm_plan(plan, NOBODY, yes=False)
    assert e.value.missing == ("--yes",)
    assert confirm_plan(plan, NOBODY, yes=True) is True


def test_replacing_a_file_needs_force_beyond_yes(tmp_path: Path) -> None:
    plan = Plan("p", (_write(tmp_path, replaces=True),))
    assert plan.needed_flags() == ("--force",)
    with pytest.raises(ConfirmationRequired) as e:
        confirm_plan(plan, NOBODY, yes=True)
    assert e.value.missing == ("--force",)
    assert confirm_plan(plan, NOBODY, yes=True, granted={"--force"}) is True


def test_yes_never_spends_money() -> None:
    plan = Plan("p", (_action("run", Contact("api.anthropic.com", "compile"), SPEND),))
    with pytest.raises(ConfirmationRequired) as e:
        confirm_plan(plan, NOBODY, yes=True, granted={"--force"})
    assert e.value.missing == ("--live",)
    with pytest.raises(ConfirmationRequired) as both:
        confirm_plan(plan, NOBODY, yes=False)
    assert both.value.missing == ("--yes", "--live")
    assert confirm_plan(plan, NOBODY, yes=True, granted={"--live"}) is True


@pytest.mark.parametrize(("effect_needs", "default"), [(False, True), (True, False)])
def test_a_person_is_asked_and_the_risky_default_is_no(
    tmp_path: Path, effect_needs: bool, default: bool
) -> None:
    plan = Plan("p", (_write(tmp_path, replaces=effect_needs),))
    seen: list[bool] = []

    class Recording(ScriptedPrompter):
        def confirm(self, message: str, *, flag: str, default: bool) -> bool:
            seen.append(default)
            return super().confirm(message, flag=flag, default=default)

    prompter = Recording([Answer("confirm", "--yes", False)])
    assert confirm_plan(plan, prompter, yes=False) is False
    assert seen == [default]


def test_an_empty_plan_needs_nothing() -> None:
    assert confirm_plan(Plan("p"), NOBODY, yes=False) is True
    assert Plan("p").to_dict()["needs"] == []


def test_apply_runs_in_order_and_stops_at_the_first_failure() -> None:
    ran: list[str] = []

    def boom() -> None:
        ran.append("b")
        raise OSError("disk full")

    plan = Plan(
        "p",
        (
            _action("a", Launch("x", "y"), run=lambda: ran.append("a")),
            _action("b", Launch("x", "y"), run=boom),
            _action("c", Launch("x", "y"), run=lambda: ran.append("c")),
        ),
    )
    outcomes = apply_plan(plan)
    assert ran == ["a", "b"]
    assert [(o.action_id, o.status) for o in outcomes] == [
        ("a", "done"),
        ("b", "failed"),
        ("c", "not-run"),
    ]
    assert outcomes[1].error == "OSError: disk full"


def test_write_action_writes_atomically_and_only_when_applied(tmp_path: Path) -> None:
    action = _write(tmp_path)
    assert not (tmp_path / "f.txt").exists()
    apply_plan(Plan("p", (action,)))
    assert (tmp_path / "f.txt").read_text(encoding="utf-8") == "new"
    assert action.effects[0].to_dict() == {
        "kind": "write",
        "path": str(tmp_path / "f.txt"),
        "summary": "s",
        "existed": False,
        "replaces": False,
    }


def test_a_failed_atomic_write_keeps_the_old_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "keep.json"
    path.write_text("old", encoding="utf-8")

    def fail(*a: object) -> None:
        raise OSError("crash")

    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(OSError):
        atomic_write_text(path, "new")
    assert path.read_text(encoding="utf-8") == "old"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["keep.json"]  # no temp left


def test_atomic_write_sets_the_mode(tmp_path: Path) -> None:
    path = tmp_path / "secret.json"
    atomic_write_text(path, "{}", file_mode=0o600)
    assert path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("host", ["https://x.com", "x.com/path", "x.com?k=v", "u@x.com", ""])
def test_contact_takes_a_host_never_a_url(host: str) -> None:
    with pytest.raises(ValueError, match="bare host"):
        Contact(host, "p")


def test_invalid_records_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no effects"):
        Action(id="x", title="x", effects=(), run=_noop)
    with pytest.raises(ValueError, match="repeats an action id"):
        Plan("p", (_action("a", Launch("x", "y")), _action("a", Launch("x", "y"))))
    with pytest.raises(ValueError, match="cannot replace a file that does not exist"):
        FileWrite(path=tmp_path / "nope", summary="s", replaces=True)
    with pytest.raises(ValueError, match="long flag"):
        Spend(1, 2, "you", flag="live")
    with pytest.raises(ValueError, match="low <= high"):
        Spend(2, 1, "you", flag="--live")


def test_render_spells_out_every_effect(tmp_path: Path) -> None:
    plan = Plan(
        "Import servers",
        (
            _write(tmp_path, replaces=True),
            _action("login", Contact("mcp.figma.com", "OAuth"), Launch("your browser", "approve")),
            _action("run", SPEND),
        ),
        notes=("figma: already registered",),
    )
    assert render_plan(plan) == [
        "Plan: Import servers",
        "  1. write",
        f"       - replace {tmp_path / 'f.txt'} (s)",
        "  2. login",
        "       - contact mcp.figma.com (OAuth)",
        "       - start your browser (approve)",
        "  3. run",
        "       - spend $0.90 to $1.70, billed to your Claude subscription",
        "  note: figma: already registered",
    ]
    assert plan.to_dict()["needs"] == ["--yes", "--force", "--live"]


def test_action_required_payload_for_a_plan(tmp_path: Path) -> None:
    plan = Plan("Save", (_write(tmp_path, replaces=True),))
    with pytest.raises(ConfirmationRequired) as e:
        confirm_plan(plan, NOBODY, yes=False)
    payload = action_required(e.value, ["init", "--deploy", "none"])
    assert payload["ok"] is False
    assert payload["reason"] == "confirmation_required"
    assert payload["missing"] == ["--yes", "--force"]
    assert payload["next"] == [{"command": "rote init --deploy none --yes --force"}]
    assert payload["plan"] == plan.to_dict()
    json.dumps(payload)  # one JSON object, serializable as is


def test_output_never_interprets_markup_from_config_names() -> None:
    err = io.StringIO()
    mode = Mode.resolve(env={}, stdin=io.StringIO(), stdout=io.StringIO(), stderr=err)
    Output(mode, stderr=err).say("server [bold]x[/bold] [link=https://evil]y[/link]")
    assert err.getvalue() == "server [bold]x[/bold] [link=https://evil]y[/link]\n"


def test_output_result_goes_to_stdout_only_under_json() -> None:
    for json_flag, expected in ((False, ""), (True, '{\n  "ok": true\n}\n')):
        out, err = io.StringIO(), io.StringIO()
        mode = Mode.resolve(json=json_flag, env={}, stdin=io.StringIO(), stdout=out, stderr=err)
        Output(mode, stdout=out, stderr=err).result({"ok": True})
        assert out.getvalue() == expected
