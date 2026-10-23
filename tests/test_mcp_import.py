"""``rote mcp import``: read the AI clients, pick, plan, confirm, apply.

Every test reads fixture configs from a tmp home through ``ClientContext``
(the real home is never read: ``Path.home`` is made to fail), writes only
the conftest-isolated registry and config, and contacts nothing. Logins are
an injected function that records its calls.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from rote import isolation
from rote.clients import ClientContext, ImportCandidate, SourceReading
from rote.config import load_config_file, load_layers, user_config_path
from rote.interact.mode import Mode
from rote.interact.output import Output
from rote.interact.plan import (
    ConfirmationRequired,
    FileWrite,
    apply_plan,
    confirm_plan,
)
from rote.interact.prompter import (
    DEFAULT,
    Answer,
    NonInteractivePrompter,
    PromptRequired,
    ScriptedPrompter,
)
from rote.mcp import McpRegistry, McpServerConfig, load_registry, save_registry
from rote.mcp.registry import registry_path
from rote.setup.mcp_import import ImportFlags, ImportPlan, ImportPlanner, McpImportError

ZAPIER = "https://actions.zapier.com/mcp/sk-ak-9fQ2xLm4Rt7Vb8Np3Kd6Wz1Hc5Jy0Ts/sse"


def _http(url: str, **extra: Any) -> dict[str, Any]:
    return {"type": "http", "url": url, **extra}


DEFAULT_SERVERS: dict[str, Any] = {
    "deepwiki": _http("https://mcp.deepwiki.com/mcp"),
    "mslearn": _http("https://learn.microsoft.com/api/mcp"),
    "acme-crm": _http(
        "https://mcp.acme-work.example/mcp", headers={"Authorization": "Bearer ${ACME_TOKEN}"}
    ),
    "acme-wiki": _http("https://wiki.acme-work.example/mcp"),
    "zap": {"type": "sse", "url": ZAPIER},
    "local-tool": {"type": "stdio", "command": "uvx", "args": ["x"]},
}


@pytest.fixture(autouse=True)
def _no_real_home(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse() -> Path:
        raise AssertionError("mcp import read Path.home(); use ClientContext")

    monkeypatch.setattr(Path, "home", staticmethod(refuse))


def _machine(tmp_path: Path, servers: dict[str, Any] | None = None) -> ClientContext:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    data = {"mcpServers": DEFAULT_SERVERS if servers is None else servers}
    (home / ".claude.json").write_text(json.dumps(data), encoding="utf-8")
    cwd = tmp_path / "cwd"
    (cwd / ".git").mkdir(parents=True, exist_ok=True)
    return ClientContext(home=home, cwd=cwd, env={})


def _out() -> tuple[Output, io.StringIO]:
    err = io.StringIO()
    mode = Mode.resolve(env={}, stdin=io.StringIO(), stdout=io.StringIO(), stderr=err)
    return Output(mode, stdout=io.StringIO(), stderr=err), err


class _Logins:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.registry_at_call: list[set[str]] = []

    def __call__(self, name: str) -> None:
        self.calls.append(name)
        self.registry_at_call.append(set(load_registry().servers))


def _plan(
    tmp_path: Path,
    prompter: Any,
    flags: ImportFlags | None = None,
    *,
    servers: dict[str, Any] | None = None,
    logins: _Logins | None = None,
    wanted: Callable[[Path, Sequence[str]], dict[str, str]] | None = None,
) -> ImportPlan:
    out, _ = _out()
    planner = ImportPlanner(
        flags=flags or ImportFlags(),
        prompter=prompter,
        out=out,
        ctx=_machine(tmp_path, servers),
        layers=load_layers(),
        registry_path=registry_path(),
        user_config_path=user_config_path(),
        login=logins,
        wanted_by_skill=wanted,
    )
    return planner.build()


def _rows(planned: ImportPlan) -> dict[str, tuple[str, str]]:
    return {r.candidate.original_name: (r.status, r.name) for r in planned.rows}


def _person(*answers: Answer) -> ScriptedPrompter:
    return ScriptedPrompter(list(answers))


NONE_AT_WORK = Answer("text", "--work-domain", "")
ACME_IS_WORK = Answer("text", "--work-domain", "acme-work.example")
NO_LOGINS = Answer("checkbox", "--login", [])
YES = NonInteractivePrompter(why="test", accept_defaults=True)


def _apply(planned: ImportPlan, *, force: bool = False) -> None:
    granted = {"--force"} if force else set()
    assert confirm_plan(planned.plan, YES, yes=True, granted=granted)
    outcomes = apply_plan(planned.plan)
    assert [o.status for o in outcomes] == ["done"] * len(outcomes), outcomes


# ───────── picking ─────────


def test_nothing_is_preselected_without_a_skill(tmp_path: Path) -> None:
    """The person's default answer imports nothing at all."""
    prompter = _person(NONE_AT_WORK, Answer("checkbox", "--only", DEFAULT))
    planned = _plan(tmp_path, prompter)
    assert planned.plan.actions[0].id == "write-work-domains"  # only the answer is saved
    assert [a.id for a in planned.plan.actions] == ["write-work-domains"]
    assert {s for s, _ in _rows(planned).values()} == {"not-selected", "not-importable"}
    assert [kind for kind, _flag, _msg in prompter.asked] == ["text", "checkbox"]


def test_a_skill_preselects_its_servers_but_never_work_or_risky_ones(tmp_path: Path) -> None:
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("# s\n", encoding="utf-8")
    asked: list[Sequence[str]] = []

    def wanted(path: Path, names: Sequence[str]) -> dict[str, str]:
        assert path == skill
        asked.append(names)
        return {n: "named" for n in ("deepwiki", "acme-crm", "acme-wiki", "zap")}

    prompter = _person(
        ACME_IS_WORK, Answer("checkbox", "--only", DEFAULT), Answer("checkbox", "--login", DEFAULT)
    )
    planned = _plan(tmp_path, prompter, ImportFlags(servers_for=skill), wanted=wanted)
    rows = _rows(planned)
    assert rows["deepwiki"] == ("add", "deepwiki")
    # Named by the skill, but on a work domain or carrying a key: not picked.
    assert rows["acme-crm"][0] == rows["acme-wiki"][0] == rows["zap"][0] == "not-selected"
    assert rows["mslearn"][0] == "not-selected"
    assert {"deepwiki", "acme-crm", "acme_crm"} <= set(asked[0])
    # The login default leaves out work servers; deepwiki is the only pick.
    assert [a.id for a in planned.plan.actions] == [
        "write-work-domains",
        "write-registry",
        "login-deepwiki",
    ]


def test_servers_for_reads_the_real_skill(tmp_path: Path) -> None:
    """No injection: tool_grounding decides what the skill names."""
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        "# Research\n\nAsk the `deepwiki` MCP server about the repo.\n", encoding="utf-8"
    )
    prompter = _person(NONE_AT_WORK, Answer("checkbox", "--only", DEFAULT), NO_LOGINS)
    planned = _plan(tmp_path, prompter, ImportFlags(servers_for=skill))
    assert [r.name for r in planned.rows if r.writes] == ["deepwiki"]


def test_the_picker_shows_host_scope_work_and_reasons(tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    class Recording(ScriptedPrompter):
        def checkbox(self, message: str, choices: Any, *, flag: str, default: Any = ()) -> Any:
            if flag == "--only":
                seen["choices"] = {c.label: c for c in choices}
            return super().checkbox(message, choices, flag=flag, default=default)

    servers = {**DEFAULT_SERVERS, "deepwiki": _http("https://mcp.deepwiki.com/mcp")}
    save_registry(
        McpRegistry(servers={"deepwiki": McpServerConfig(url="https://mcp.deepwiki.com/mcp")})
    )
    prompter = Recording([ACME_IS_WORK, Answer("checkbox", "--only", [])])
    _plan(tmp_path, prompter, servers=servers)
    choices = seen["choices"]
    assert "mcp.acme-work.example" in choices["acme_crm"].hint
    assert "Claude Code user" in choices["acme_crm"].hint
    assert "WORK" in choices["acme_crm"].hint
    assert "needs your yes" in choices["zap"].hint and "API key" in choices["zap"].hint
    assert "9fQ2" not in choices["zap"].hint  # the key itself is never shown
    assert choices["deepwiki"].disabled == "already in rote's registry"
    assert "stdio" not in {c.label for c in choices.values()}
    assert "local_tool" not in choices  # not importable: a row, not a choice
    assert "WORK" not in choices["mslearn"].hint


def test_only_picks_by_name_and_refuses_what_it_cannot_import(tmp_path: Path) -> None:
    planned = _plan(tmp_path, YES, ImportFlags(only=("mslearn",)))
    assert [r.name for r in planned.rows if r.writes] == ["mslearn"]
    with pytest.raises(McpImportError, match="local-tool: not importable: a stdio server"):
        _plan(tmp_path, YES, ImportFlags(only=("local-tool",)))
    with pytest.raises(McpImportError, match="no such server was found; found: "):
        _plan(tmp_path, YES, ImportFlags(only=("figma",)))


# ───────── work domains ─────────


def test_work_domains_are_asked_once_and_saved(tmp_path: Path) -> None:
    planned = _plan(tmp_path, _person(ACME_IS_WORK, Answer("checkbox", "--only", [])))
    _apply(planned)
    assert load_config_file(user_config_path()) == {"work_domains": ("acme-work.example",)}
    # The second run asks nothing about domains: the scripted prompter
    # fails on any question it was not given.
    again = _plan(tmp_path, _person(Answer("checkbox", "--only", [])))
    assert again.work_domains == ("acme-work.example",)
    assert again.plan.actions == ()
    assert {r.work for r in again.rows if r.candidate.original_name.startswith("acme")} == {True}


def test_a_blank_answer_is_saved_as_none(tmp_path: Path) -> None:
    _apply(_plan(tmp_path, _person(NONE_AT_WORK, Answer("checkbox", "--only", []))))
    assert load_config_file(user_config_path()) == {"work_domains": ()}


def test_the_work_domain_write_keeps_the_other_settings(tmp_path: Path) -> None:
    user_config_path().parent.mkdir(parents=True, exist_ok=True)
    user_config_path().write_text("runtime: temporal\nmodel: m\n", encoding="utf-8")
    planned = _plan(tmp_path, YES, ImportFlags(only=(), work_domains=("acme-work.example",)))
    [action] = planned.plan.actions
    [effect] = action.effects
    assert isinstance(effect, FileWrite) and (effect.existed, effect.replaces) == (True, False)
    _apply(planned)
    assert load_config_file(user_config_path()) == {
        "runtime": "temporal",
        "model": "m",
        "work_domains": ("acme-work.example",),
    }


def test_nobody_at_the_terminal_must_answer_work_domains_for_a_skill(tmp_path: Path) -> None:
    """Defaults decide the pick under --servers-for, and a default cannot know
    which domains are work, so even --yes stops on the question."""
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("# s\n", encoding="utf-8")
    with pytest.raises(PromptRequired) as e:
        _plan(tmp_path, YES, ImportFlags(servers_for=skill), wanted=lambda _p, _n: {})
    assert e.value.flag == "--work-domain"
    planned = _plan(
        tmp_path,
        YES,
        ImportFlags(servers_for=skill, work_domains=("none",)),
        wanted=lambda _p, _n: {},
    )
    assert planned.plan.actions[0].id == "write-work-domains"


def test_only_the_user_answers_the_work_domain_question(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """review: a project rote.yaml (or ROTE_WORK_DOMAINS) adds domains but is
    not the user's answer. Before, a committed `work_domains: []` skipped the
    question, let work servers be preselected, and was saved into the user's
    config as their answer."""
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("# s\n", encoding="utf-8")
    project = Path(os.environ["ROTE_PROJECT_CONFIG_PATH"])
    project.parent.mkdir(parents=True, exist_ok=True)
    project.write_text("work_domains: []\n", encoding="utf-8")
    flags = ImportFlags(servers_for=skill)
    with pytest.raises(PromptRequired) as e:
        _plan(tmp_path, YES, flags, wanted=lambda _p, _n: {})
    assert e.value.flag == "--work-domain"
    monkeypatch.setenv("ROTE_WORK_DOMAINS", "acme-work.example")
    with pytest.raises(PromptRequired):
        _plan(tmp_path, YES, flags, wanted=lambda _p, _n: {})
    # A person's answer is saved alone; the env's domain still applies to the run.
    planned = _plan(
        tmp_path,
        _person(
            Answer("text", "--work-domain", "corp.example"),
            Answer("checkbox", "--only", DEFAULT),
            NO_LOGINS,
        ),
        flags,
        wanted=lambda _p, _n: {"acme-wiki": "named", "deepwiki": "named"},
    )
    assert planned.work_domains == ("acme-work.example", "corp.example")
    assert [r.name for r in planned.rows if r.writes] == ["deepwiki"]
    _apply(planned)
    assert load_config_file(user_config_path())["work_domains"] == ("corp.example",)
    # A project/env value never writes the user's config.
    user_config_path().unlink()
    planned = _plan(tmp_path, YES, ImportFlags(only=("mslearn",)))
    assert [a.id for a in planned.plan.actions] == ["write-registry"]


def test_saving_work_domains_keeps_the_users_comments(tmp_path: Path) -> None:
    user_config_path().parent.mkdir(parents=True, exist_ok=True)
    user_config_path().write_text("# keep me\nruntime: temporal  # and me\n", encoding="utf-8")
    planned = _plan(tmp_path, YES, ImportFlags(only=(), work_domains=("acme-work.example",)))
    [action] = planned.plan.actions
    [effect] = action.effects
    assert isinstance(effect, FileWrite) and effect.replaces is False
    _apply(planned)
    text = user_config_path().read_text(encoding="utf-8")
    assert text.startswith("# keep me\nruntime: temporal  # and me\nwork_domains: ")
    # A layout the line edit cannot handle is a marked replacement (--force).
    user_config_path().write_text("{runtime: temporal}\n", encoding="utf-8")
    planned = _plan(tmp_path, YES, ImportFlags(only=(), work_domains=("corp.example",)))
    [action] = planned.plan.actions
    assert planned.plan.needed_flags() == ("--force",)
    assert "comments are lost" in action.effects[0].describe()


def test_an_explicit_pick_needs_no_work_domain_answer(tmp_path: Path) -> None:
    planned = _plan(tmp_path, YES, ImportFlags(only=("deepwiki",)))
    assert [a.id for a in planned.plan.actions] == ["write-registry"]
    assert not user_config_path().exists()


def test_bad_work_domains_are_refused(tmp_path: Path) -> None:
    with pytest.raises(McpImportError, match="no scheme, path or port"):
        _plan(tmp_path, YES, ImportFlags(only=(), work_domains=("https://acme.example",)))
    with pytest.raises(McpImportError, match="cannot be combined"):
        _plan(tmp_path, YES, ImportFlags(only=(), work_domains=("none", "acme.example")))


# ───────── the registry write ─────────


def test_the_registry_write_is_additive_and_rerunning_changes_nothing(tmp_path: Path) -> None:
    save_registry(McpRegistry(servers={"mine": McpServerConfig(url="https://mine.example/mcp")}))
    planned = _plan(tmp_path, YES, ImportFlags(only=("deepwiki", "acme-crm")))
    [action] = planned.plan.actions
    [effect] = action.effects
    assert isinstance(effect, FileWrite)
    assert (effect.existed, effect.replaces, effect.summary) == (
        True,
        False,
        "add deepwiki, acme_crm",
    )
    assert planned.plan.needed_flags() == ()  # --yes is enough: nothing is lost
    _apply(planned)
    registry = load_registry()
    assert sorted(registry.servers) == ["acme_crm", "deepwiki", "mine"]
    assert registry.servers["mine"].url == "https://mine.example/mcp"
    assert registry_path().stat().st_mode & 0o777 == 0o600
    before = registry_path().read_bytes()

    again = _plan(tmp_path, YES, ImportFlags(only=("deepwiki", "acme-crm")))
    assert again.plan.actions == ()
    assert _rows(again)["deepwiki"] == ("present", "deepwiki")
    assert registry_path().read_bytes() == before


def test_a_header_template_is_written_as_the_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ACME_TOKEN", "the-real-secret")
    planned = _plan(tmp_path, YES, ImportFlags(only=("acme-crm",)))
    _apply(planned)
    text = registry_path().read_text(encoding="utf-8")
    assert "Bearer ${ACME_TOKEN}" in text
    assert "the-real-secret" not in text
    assert load_registry().servers["acme_crm"].headers == {"Authorization": "Bearer ${ACME_TOKEN}"}
    assert any("reads ACME_TOKEN from the environment" in n for n in planned.plan.notes)


def test_an_edit_made_after_the_plan_is_never_overwritten(tmp_path: Path) -> None:
    planned = _plan(tmp_path, YES, ImportFlags(only=("deepwiki",)))
    save_registry(McpRegistry(servers={"late": McpServerConfig(url="https://late.example/mcp")}))
    [outcome] = apply_plan(planned.plan)
    assert outcome.status == "failed"
    assert outcome.error is not None and "was created since the plan was made" in outcome.error
    assert sorted(load_registry().servers) == ["late"]


# ───────── collisions ─────────


def _registered(**servers: str) -> None:
    save_registry(
        McpRegistry(servers={name: McpServerConfig(url=url) for name, url in servers.items()})
    )


def test_a_name_registered_elsewhere_gets_a_suffix_and_keeps_the_old_entry(tmp_path: Path) -> None:
    _registered(deepwiki="https://old.example/mcp", deepwiki_2="https://older.example/mcp")
    planned = _plan(tmp_path, YES, ImportFlags(only=("deepwiki",)))
    row = next(r for r in planned.rows if r.candidate.original_name == "deepwiki")
    assert (row.status, row.name) == ("add", "deepwiki_3")
    assert "deepwiki is registered for old.example; added as deepwiki_3" in row.detail
    assert [c.kind for c in planned.collisions] == ["registered-elsewhere"]
    _apply(planned)
    servers = load_registry().servers
    assert servers["deepwiki"].url == "https://old.example/mcp"  # never written over
    assert servers["deepwiki_3"].url == "https://mcp.deepwiki.com/mcp"


def test_rename_picks_the_name_and_refuses_a_taken_one(tmp_path: Path) -> None:
    _registered(deepwiki="https://old.example/mcp", other="https://o.example/mcp")
    planned = _plan(tmp_path, YES, ImportFlags(only=("deepwiki",), rename=(("deepwiki", "dw"),)))
    assert _rows(planned)["deepwiki"] == ("add", "dw")
    with pytest.raises(McpImportError, match="other is already registered for o.example"):
        _plan(tmp_path, YES, ImportFlags(only=("deepwiki",), rename=(("deepwiki", "other"),)))
    with pytest.raises(McpImportError, match="not an identifier"):
        _plan(tmp_path, YES, ImportFlags(only=("deepwiki",), rename=(("deepwiki", "a-b"),)))
    with pytest.raises(McpImportError, match="not a server being imported"):
        _plan(tmp_path, YES, ImportFlags(only=("deepwiki",), rename=(("mslearn", "ms"),)))


def test_replace_overwrites_only_with_force(tmp_path: Path) -> None:
    _registered(deepwiki="https://old.example/mcp")
    planned = _plan(tmp_path, YES, ImportFlags(only=("deepwiki",), replace=("deepwiki",)))
    assert _rows(planned)["deepwiki"] == ("replace", "deepwiki")
    assert planned.plan.needed_flags() == ("--force",)
    with pytest.raises(ConfirmationRequired) as e:
        confirm_plan(planned.plan, YES, yes=True)
    assert e.value.missing == ("--force",)
    assert load_registry().servers["deepwiki"].url == "https://old.example/mcp"
    _apply(planned, force=True)
    assert load_registry().servers["deepwiki"].url == "https://mcp.deepwiki.com/mcp"
    with pytest.raises(McpImportError, match="--replace mslearn: not a registered server"):
        _plan(tmp_path, YES, ImportFlags(only=("mslearn",), replace=("mslearn",)))


def test_two_picks_that_want_one_name_are_both_kept(tmp_path: Path) -> None:
    servers = {
        "acme-crm": _http("https://a.example/mcp"),
        "acme_crm": _http("https://b.example/mcp"),
    }
    planned = _plan(tmp_path, YES, ImportFlags(only=("acme-crm", "acme_crm")), servers=servers)
    rows = _rows(planned)
    assert rows == {"acme-crm": ("add", "acme_crm"), "acme_crm": ("add", "acme_crm_2")}
    assert [c.kind for c in planned.collisions] == ["same-identifier"]
    _apply(planned)
    servers_now = load_registry().servers
    assert servers_now["acme_crm"].url == "https://a.example/mcp"
    assert servers_now["acme_crm_2"].url == "https://b.example/mcp"


def test_an_endpoint_registered_under_another_name_is_not_added_again(tmp_path: Path) -> None:
    _registered(wiki="https://MCP.deepwiki.com/mcp/")
    planned = _plan(tmp_path, YES, ImportFlags(only=("deepwiki",)))
    assert _rows(planned)["deepwiki"] == ("present", "wiki")
    assert planned.plan.actions == ()


def test_changed_settings_are_kept_unless_replaced(tmp_path: Path) -> None:
    save_registry(
        McpRegistry(
            servers={
                "acme_crm": McpServerConfig(
                    url="https://mcp.acme-work.example/mcp", headers={"X-Old": "v"}
                )
            }
        )
    )
    planned = _plan(tmp_path, YES, ImportFlags(only=("acme-crm",)))
    row = next(r for r in planned.rows if r.candidate.original_name == "acme-crm")
    assert row.status == "changed" and "--replace acme_crm" in row.detail
    assert planned.plan.actions == ()
    replaced = _plan(tmp_path, YES, ImportFlags(only=("acme-crm",), replace=("acme_crm",)))
    _apply(replaced, force=True)
    assert load_registry().servers["acme_crm"].headers == {"Authorization": "Bearer ${ACME_TOKEN}"}


# ───────── risky URLs ─────────


def test_a_risky_url_needs_its_own_yes(tmp_path: Path) -> None:
    declined = _plan(tmp_path, YES, ImportFlags(only=("zap",)))
    assert _rows(declined)["zap"][0] == "declined"
    assert declined.plan.actions == ()
    with pytest.raises(PromptRequired) as e:
        _plan(tmp_path, NonInteractivePrompter(why="test"), ImportFlags(only=("zap",)))
    assert e.value.flag == "--allow-url-secret zap" and e.value.takes_value is False
    allowed = _plan(tmp_path, YES, ImportFlags(only=("zap",), allow_url_secret=("zap",)))
    assert _rows(allowed)["zap"] == ("add", "zap")
    person = _plan(
        tmp_path,
        _person(
            NONE_AT_WORK,
            Answer("checkbox", "--only", ["zap"]),
            Answer("confirm", "--allow-url-secret zap", True),
            NO_LOGINS,
        ),
    )
    assert _rows(person)["zap"] == ("add", "zap")


@dataclass(frozen=True)
class _WithTrust(ImportCandidate):
    """The trust state PR #245 adds, ahead of it: unknown trust is not preselectable."""

    trust: str = "trusted"

    @property
    def preselectable(self) -> bool:
        return super().preselectable and self.trust == "trusted"


@dataclass(frozen=True)
class _NewReason(ImportCandidate):
    """A non-preselectable reason this version does not know."""

    @property
    def preselectable(self) -> bool:
        return False


class _FakeSource:
    id = "fake-client"
    label = "Fake Client"

    def __init__(self, candidates: Sequence[ImportCandidate]) -> None:
        self.candidates = tuple(candidates)

    def config_paths(self, ctx: ClientContext) -> tuple[Path, ...]:
        return ()

    def detect(self, ctx: ClientContext) -> bool:
        return True

    def read(self, ctx: ClientContext) -> SourceReading:
        return SourceReading(client=self.id, candidates=self.candidates)


def _fake(cls: type[ImportCandidate], name: str, **kw: Any) -> ImportCandidate:
    return cls(
        client="fake-client",
        name=name,
        original_name=name,
        scope="project",
        origin=Path("/fake/.mcp.json"),
        transport="streamable-http",
        url=f"https://{name}.example/mcp",
        **kw,
    )


def test_any_reason_not_to_preselect_needs_its_own_yes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Built against preselectable, so a trust state a client source adds is
    covered with no change here: never a default, a bare --yes declines it,
    and its own flag (or a person's yes) imports it."""
    import rote.setup.mcp_import as mcp_import

    source = _FakeSource(
        [
            _fake(_WithTrust, "untrusted", trust="unknown"),
            _fake(_NewReason, "novel"),
            _fake(ImportCandidate, "off", inactive="disabled in Fake Client"),
            _fake(ImportCandidate, "plain"),
        ]
    )
    monkeypatch.setattr(mcp_import, "ordered_sources", lambda _detected: (source,))
    names = ("untrusted", "novel", "off", "plain")

    declined = _plan(tmp_path, YES, ImportFlags(only=names))
    rows = {r.candidate.original_name: r for r in declined.rows}
    assert rows["untrusted"].status == "declined"
    assert rows["untrusted"].detail == (
        "trust in Fake Client could not be checked; pass --allow-untrusted untrusted to import it"
    )
    assert rows["novel"].status == "declined"
    assert "--allow-untrusted novel" in rows["novel"].detail
    # Inactive is the person's own pick to make: no extra yes.
    assert (rows["off"].status, rows["plain"].status) == ("add", "add")

    with pytest.raises(PromptRequired) as e:
        _plan(tmp_path, NonInteractivePrompter(why="test"), ImportFlags(only=("untrusted",)))
    assert e.value.flag == "--allow-untrusted untrusted"

    allowed = _plan(tmp_path, YES, ImportFlags(only=names, allow_untrusted=("untrusted", "novel")))
    assert {r.candidate.original_name for r in allowed.rows if r.writes} == set(names)
    command = allowed.plan.actions[0].command or ""
    assert "--allow-untrusted untrusted" in command and "--allow-untrusted novel" in command

    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("# s\n", encoding="utf-8")
    preselected = _plan(
        tmp_path,
        _person(NONE_AT_WORK, Answer("checkbox", "--only", DEFAULT), NO_LOGINS),
        ImportFlags(servers_for=skill),
        wanted=lambda _p, _n: dict.fromkeys(names, "named"),
    )
    assert [r.candidate.original_name for r in preselected.rows if r.writes] == ["plain"]


# ───────── logins ─────────


def test_logins_run_after_the_write_and_skip_header_servers(tmp_path: Path) -> None:
    logins = _Logins()
    prompter = _person(
        ACME_IS_WORK,
        Answer("checkbox", "--only", ["deepwiki", "acme-crm", "acme-wiki"]),
        Answer("checkbox", "--login", DEFAULT),
    )
    planned = _plan(tmp_path, prompter, logins=logins)
    ids = [a.id for a in planned.plan.actions]
    # acme-crm carries a header (no OAuth login); acme-wiki is work, so not a default.
    assert ids == ["write-work-domains", "write-registry", "login-deepwiki"]
    login = planned.plan.actions[-1]
    assert login.command == "rote mcp login deepwiki"
    assert [e.to_dict() for e in login.effects] == [
        {"kind": "contact", "host": "mcp.deepwiki.com", "purpose": "OAuth discovery and sign-in"},
        {
            "kind": "launch",
            "program": "your browser",
            "purpose": "approve rote's access to deepwiki",
            "paid": False,
        },
    ]
    assert logins.calls == []  # planning contacts nothing
    _apply(planned)
    assert logins.calls == ["deepwiki"]
    assert "deepwiki" in logins.registry_at_call[0]  # registered before the login ran


def test_a_login_needs_a_person(tmp_path: Path) -> None:
    with pytest.raises(McpImportError, match="--login needs a person"):
        _plan(tmp_path, YES, ImportFlags(only=("deepwiki",), login=("deepwiki",)))
    with pytest.raises(McpImportError, match="--login acme_crm: not a server this import"):
        _plan(tmp_path, YES, ImportFlags(only=("acme-crm",), login=("acme_crm",)))
    planned = _plan(tmp_path, YES, ImportFlags(only=("deepwiki",)))
    assert "deepwiki: next: rote mcp login deepwiki" in planned.plan.notes


def test_the_diff_lists_what_changes_and_counts_the_rest(tmp_path: Path) -> None:
    from rote.setup.mcp_import import render_rows

    _registered(mslearn="https://learn.microsoft.com/api/mcp")
    planned = _plan(
        tmp_path, YES, ImportFlags(only=("deepwiki",), work_domains=("acme-work.example",))
    )
    lines = render_rows(planned.rows)
    body = "\n".join(lines)
    assert "+ deepwiki" in body and "mcp.deepwiki.com  add" in body
    assert "= mslearn" in body and "present: already registered" in body
    assert "x local_tool" in body and "a stdio server" in body
    # acme-crm, acme-wiki and zap are listed only as a count.
    assert lines[-1] == "  3 more not picked (2 on your work domains)"
    assert "acme" not in body and "zap" not in body


# ───────── the command, end to end ─────────


def _cli(tmp_path: Path, *args: str, stdin: str = "") -> tuple[int, str, str]:
    env = isolation.isolated_env(tmp_path / "state")
    home = tmp_path / "home"
    _machine(tmp_path)  # writes tmp_path/home/.claude.json
    env["ROTE_CLIENT_HOME_DIR"] = str(home)
    for var in ("CLAUDECODE", "CLAUDE_CODE_CHILD_SESSION", "AI_AGENT", "ROTE_AGENT_MODE"):
        env.pop(var, None)
    run = subprocess.run(
        [sys.executable, "-m", "rote", "mcp", "import", *args],
        input=stdin,
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path / "cwd",
        timeout=120,
    )
    return run.returncode, run.stdout, run.stderr


def _state_registry(tmp_path: Path) -> Path:
    return tmp_path / "state" / isolation.STATE_PATHS["ROTE_MCP_CONFIG"]


def test_cli_json_without_a_pick_is_action_required(tmp_path: Path) -> None:
    rc, out, _ = _cli(tmp_path, "--json")
    assert rc == 2
    payload = json.loads(out)  # one object on stdout
    assert (payload["status"], payload["reason"], payload["missing"]) == (
        "action_required",
        "missing_answer",
        ["--only"],
    )
    assert set(payload["choices"]) == {"deepwiki", "mslearn", "acme-crm", "acme-wiki", "zap"}
    assert payload["next"][0]["command"].startswith("rote mcp import --json --only <")
    hosts = {s["name"]: s["host"] for s in payload["servers"]}
    assert hosts["zap"] == "actions.zapier.com"
    assert "9fQ2" not in out and "url" not in {k for s in payload["servers"] for k in s}
    assert not _state_registry(tmp_path).exists()


def test_cli_json_shows_the_plan_and_applies_only_with_yes(tmp_path: Path) -> None:
    rc, out, _ = _cli(tmp_path, "--only", "deepwiki", "--json")
    assert rc == 2
    payload = json.loads(out)
    assert (payload["reason"], payload["missing"]) == ("confirmation_required", ["--yes"])
    assert [a["id"] for a in payload["plan"]["actions"]] == ["write-registry"]
    assert payload["next"][0]["command"] == "rote mcp import --only deepwiki --json --yes"
    assert not _state_registry(tmp_path).exists()

    rc, out, _ = _cli(tmp_path, "--only", "deepwiki", "--json", "--yes")
    assert rc == 0
    payload = json.loads(out)
    assert payload["applied"] is True
    assert payload["outcomes"] == [{"id": "write-registry", "status": "done", "error": None}]
    registry = json.loads(_state_registry(tmp_path).read_text(encoding="utf-8"))
    assert registry["servers"] == {
        "deepwiki": {"url": "https://mcp.deepwiki.com/mcp", "transport": "streamable-http"}
    }


def test_cli_a_risky_pick_names_its_flag_bare(tmp_path: Path) -> None:
    rc, out, _ = _cli(tmp_path, "--only", "zap", "--json")
    assert rc == 2
    payload = json.loads(out)
    assert payload["missing"] == ["--allow-url-secret zap"]
    assert payload["next"][0]["command"] == (
        "rote mcp import --only zap --json --allow-url-secret zap"
    )


def test_cli_piped_answers_never_reach_a_prompt(tmp_path: Path) -> None:
    rc, out, err = _cli(tmp_path, stdin="\r\r\ry\r\r")
    assert rc == 2
    assert out == ""
    assert "pass --only" in err
    assert not _state_registry(tmp_path).exists()


def test_cli_bad_flags_exit_2_naming_them(tmp_path: Path) -> None:
    rc, _, err = _cli(tmp_path, "--rename", "nope", "--yes")
    assert rc == 2 and "--rename expects old=new" in err
    rc, _, err = _cli(tmp_path, "--from", "nowhere", "--yes")
    assert rc == 2 and "no MCP config reader for 'nowhere'" in err
    assert not _state_registry(tmp_path).exists()
