"""rote.skills.discover against fixture homes, never the real one.

Every test builds a home and a project under tmp_path and passes them as a
ClientContext, the only door the module has to the machine.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from rote.cli import main as cli_main
from rote.clients.base import ClientContext
from rote.skills import RegistryView, discover
from rote.skills.discovery import FRONTMATTER_BYTES, read_frontmatter

REPO = Path(__file__).resolve().parent.parent


def _skill(root: Path, dirname: str, body: str = "Do the thing.\n", **front: str) -> Path:
    directory = root / dirname
    directory.mkdir(parents=True, exist_ok=True)
    lines = ["---", *(f"{k}: {json.dumps(v)}" for k, v in front.items()), "---", "", body]
    (directory / "SKILL.md").write_text("\n".join(lines), encoding="utf-8")
    return directory


@pytest.fixture()
def machine(tmp_path: Path) -> tuple[Path, Path]:
    """An empty home, and a project two levels inside a repository."""
    home = tmp_path / "home"
    repo = tmp_path / "work" / "repo"
    (repo / ".git").mkdir(parents=True)
    project = repo / "packages" / "app"
    project.mkdir(parents=True)
    home.mkdir()
    return home, project


def _ctx(home: Path, cwd: Path, **env: str) -> ClientContext:
    return ClientContext(home=home, cwd=cwd, env=env)


def _found(ctx: ClientContext, registry: RegistryView | None = None) -> dict[str, dict]:
    result = discover(ctx, registry=registry or RegistryView(), include_bundled=False)
    return {s.name: s.to_dict() for s in result.skills}


def _enable_plugin(home: Path, key: str, install: Path) -> None:
    plugins = home / ".claude" / "plugins"
    plugins.mkdir(parents=True, exist_ok=True)
    path = plugins / "installed_plugins.json"
    data = json.loads(path.read_text()) if path.exists() else {"version": 2, "plugins": {}}
    data["plugins"][key] = [{"scope": "user", "installPath": str(install)}]
    path.write_text(json.dumps(data))
    settings = home / ".claude" / "settings.json"
    current = json.loads(settings.read_text()) if settings.exists() else {}
    current.setdefault("enabledPlugins", {})[key] = True
    settings.write_text(json.dumps(current))


def test_every_clients_skill_dirs_are_found(machine: tuple[Path, Path], tmp_path: Path) -> None:
    home, project = machine
    repo = project.parent.parent
    _skill(home / ".claude" / "skills", "claude-user")
    _skill(repo / ".claude" / "skills", "claude-repo-root")
    _skill(project / ".claude" / "skills", "claude-project")
    _skill(repo.parent / ".claude" / "skills", "above-the-repo")  # outside the repo: not read
    _skill(home / ".agents" / "skills", "agents-user")
    _skill(project / ".agents" / "skills", "agents-project")
    codex_home = tmp_path / "codex-home"
    _skill(codex_home / "skills", "codex-user")
    _skill(home / ".codex" / "skills", "codex-default-home")  # moved by CODEX_HOME
    _skill(home / ".gemini" / "skills", "gemini-user")
    _skill(repo / ".gemini" / "skills", "gemini-project")
    xdg = tmp_path / "xdg"
    _skill(xdg / "opencode" / "skills", "opencode-user")
    _skill(project / ".opencode" / "skills", "opencode-project")
    on = tmp_path / "plugins" / "helper"
    off = tmp_path / "plugins" / "dormant"
    _skill(on / "skills", "plugin-skill")
    _skill(off / "skills", "disabled-plugin-skill")
    _enable_plugin(home, "helper@market", on)
    installed = home / ".claude" / "plugins" / "installed_plugins.json"
    data = json.loads(installed.read_text())
    data["plugins"]["dormant@market"] = [{"scope": "user", "installPath": str(off)}]
    installed.write_text(json.dumps(data))
    (project / ".claude" / "skills" / "not-a-skill").mkdir()  # no SKILL.md

    found = _found(_ctx(home, project, CODEX_HOME=str(codex_home), XDG_CONFIG_HOME=str(xdg)))

    where = {name: (s["client"], s["scope"], s["plugin"]) for name, s in found.items()}
    assert where == {
        "claude-user": ("claude-code", "user", None),
        "claude-repo-root": ("claude-code", "project", None),
        "claude-project": ("claude-code", "project", None),
        "agents-user": ("agents", "user", None),
        "agents-project": ("agents", "project", None),
        "codex-user": ("codex", "user", None),
        "gemini-user": ("gemini", "user", None),
        "gemini-project": ("gemini", "project", None),
        "opencode-user": ("opencode", "user", None),
        "opencode-project": ("opencode", "project", None),
        "plugin-skill": ("claude-code", "plugin", "helper"),
    }


def test_claude_config_dir_moves_the_user_skills(
    machine: tuple[Path, Path], tmp_path: Path
) -> None:
    home, project = machine
    _skill(home / ".claude" / "skills", "default-dir")
    moved = tmp_path / "claude-config"
    _skill(moved / "skills", "moved-dir")
    assert set(_found(_ctx(home, project, CLAUDE_CONFIG_DIR=str(moved)))) == {"moved-dir"}


def test_a_skill_linked_into_two_roots_is_one_skill(machine: tuple[Path, Path]) -> None:
    home, project = machine
    real = _skill(home / ".agents" / "skills", "shared", name="shared")
    (home / ".claude" / "skills").mkdir(parents=True)
    (home / ".claude" / "skills" / "shared").symlink_to(real)

    found = _found(_ctx(home, project))

    assert list(found) == ["shared"]
    assert found["shared"]["path"] == str(real.resolve())
    assert [(r["client"], r["scope"]) for r in found["shared"]["found_in"]] == [
        ("claude-code", "user"),
        ("agents", "user"),
    ]


def test_frontmatter_is_read_safely(machine: tuple[Path, Path]) -> None:
    home, project = machine
    root = home / ".claude" / "skills"
    _skill(root, "dir-name-a", name="Named", description="line one\nline two‮​ end")
    plain = root / "no-frontmatter"
    plain.mkdir(parents=True)
    (plain / "SKILL.md").write_text("# Just a heading\n", encoding="utf-8")
    for name, text in {
        "bad-yaml": "---\nname: [unclosed\n---\n",
        "unclosed": "---\nname: x\n" + "y: z\n" * 10,
        "a-list": "---\n- one\n- two\n---\n",
        "python-tag": "---\nname: !!python/object/apply:os.system ['echo hi']\n---\n",
    }.items():
        (root / name).mkdir()
        (root / name / "SKILL.md").write_text(text, encoding="utf-8")
    long = _skill(root, "long", description="word " * 500)

    found = _found(_ctx(home, project))

    assert found["Named"]["description"] == "line one line two end"
    assert found["no-frontmatter"]["description"] is None
    assert found["no-frontmatter"]["problems"] == []
    assert found["bad-yaml"]["problems"] == ["its frontmatter is not valid YAML (line 2)"]
    assert found["unclosed"]["problems"] == ["its frontmatter does not close within 64 KB"]
    assert found["a-list"]["problems"] == ["its frontmatter is not a mapping"]
    assert found["python-tag"]["problems"][0].startswith("its frontmatter is not valid YAML")
    assert len(found["long"]["description"]) == 300
    assert read_frontmatter(long / "SKILL.md")[1] is None


def test_frontmatter_reads_are_bounded(tmp_path: Path) -> None:
    skill_md = tmp_path / "SKILL.md"
    skill_md.write_text("---\n" + "x: y\n" * (FRONTMATTER_BYTES // 5 + 10) + "---\n")
    assert read_frontmatter(skill_md) == ({}, "its frontmatter does not close within 64 KB")


def test_ranking_and_readiness_reasons(machine: tuple[Path, Path]) -> None:
    home, project = machine
    root = home / ".claude" / "skills"
    _skill(root, "b-no-servers")
    _skill(root, "a-ready", "Call `mcp__crm__get_contact`.")
    _skill(root, "d-one-login", "Call `mcp__crm__get_contact` and `mcp__mail__send`.")
    _skill(root, "c-two-logins", "Use `mcp__mail__send` and `mcp__docs__read`.")
    _skill(root, "e-unregistered", "Use `mcp__ghost__haunt` and `mcp__mail__send`.")
    bad = root / "f-binary"
    bad.mkdir()
    (bad / "SKILL.md").write_bytes(b"---\nname: f-binary\n---\n\xff\xfe not utf-8")
    registry = RegistryView(
        {"crm": "authenticated", "mail": "not authenticated", "docs": "auth unknown"}
    )

    result = discover(_ctx(home, project), registry=registry, include_bundled=False)

    order = [(s.name, s.readiness.state) for s in result.skills]
    assert order == [
        ("a-ready", "ready"),
        ("b-no-servers", "ready"),
        ("d-one-login", "needs_login"),
        ("c-two-logins", "needs_login"),
        ("e-unregistered", "needs_server"),
        ("f-binary", "unreadable"),
    ]
    by = {s.name: s.readiness for s in result.skills}
    assert by["a-ready"].reason == "every server it names is registered and logged in: crm"
    assert by["b-no-servers"].reason == "it names no MCP server"
    assert by["d-one-login"].reason == "1 login(s) away: mail (not authenticated)"
    assert by["d-one-login"].next == ("rote mcp login mail",)
    assert by["c-two-logins"].logins_needed == 2
    assert by["e-unregistered"].next == (
        "rote mcp add ghost <url>",
        "rote mcp login ghost",
        "rote mcp login mail",
    )
    assert [s.to_dict() for s in by["d-one-login"].servers] == [
        {
            "name": "crm",
            "evidence": "mcp__crm__get_contact",
            "registered": True,
            "auth": "authenticated",
            "usable": True,
        },
        {
            "name": "mail",
            "evidence": "mcp__mail__send",
            "registered": True,
            "auth": "not authenticated",
            "usable": False,
        },
    ]
    assert by["f-binary"].reason == "rote could not read the skill (UnicodeDecodeError)"


def test_readiness_reads_the_real_registry_and_token_store(machine: tuple[Path, Path]) -> None:
    from rote.mcp import McpRegistry, McpServerConfig, save_registry, write_token_file

    home, project = machine
    _skill(home / ".claude" / "skills", "uses-crm", "Call `mcp__crm__get_contact`.")
    save_registry(McpRegistry(servers={"crm": McpServerConfig(url="https://crm.example/mcp")}))
    ctx = _ctx(home, project)
    before = discover(ctx, include_bundled=False).skills[0].readiness
    assert (before.state, before.servers[0].auth) == ("needs_login", "auth unknown")

    write_token_file(
        "crm",
        {
            "server_url": "https://crm.example/mcp",
            "tokens": {"access_token": "t", "token_type": "Bearer"},
            "expires_at": time.time() + 3600,
        },
    )
    after = discover(ctx, include_bundled=False).skills[0].readiness
    assert (after.state, after.servers[0].auth) == ("ready", "authenticated")


def test_the_bundled_demo_comes_first(machine: tuple[Path, Path]) -> None:
    from rote.demo import DEMO_NAME
    from rote.examples import BUNDLED

    home, project = machine
    _skill(home / ".claude" / "skills", "aaa-first-by-name")
    result = discover(_ctx(home, project), registry=RegistryView())
    others = [n for n, e in BUNDLED.items() if "skill" in e.files and n != DEMO_NAME]
    shipped = [DEMO_NAME, *others]
    assert DEMO_NAME in BUNDLED and others  # the ordering below is not vacuous
    head = result.skills[: len(shipped)]
    assert [s.readiness.state for s in head] == ["bundled"] * len(shipped)
    assert [s.path.parent.name for s in head] == shipped
    assert head[0].readiness.reason.startswith("the demo rote try runs:")
    assert head[1].readiness.reason.startswith("bundled:")
    assert result.skills[len(shipped)].name == "aaa-first-by-name"


def test_the_default_context_never_reads_the_real_home() -> None:
    """conftest moves ROTE_CLIENT_HOME_DIR, so only the bundled demo shows."""
    result = discover(registry=RegistryView())
    assert {s.readiness.state for s in result.skills} == {"bundled"}
    assert all(str(Path.home()) not in str(r.path) for r in result.roots if r.scope == "user")


def test_hundreds_of_skills_stay_fast(machine: tuple[Path, Path]) -> None:
    home, project = machine
    body = "## Steps\n\n| tool | use |\n|---|---|\n| `get_contact` | read |\n\n" * 20
    for index in range(400):
        _skill(home / ".claude" / "skills", f"skill-{index:03}", body, description="x" * 200)
    registry = RegistryView({"crm": "authenticated", "mail": "auth unknown"})
    began = time.perf_counter()
    result = discover(_ctx(home, project), registry=registry, include_bundled=False)
    elapsed = time.perf_counter() - began
    assert len(result.skills) == 400
    assert elapsed < 5.0, f"400 skills took {elapsed:.2f}s"


def test_tool_registry_matches_the_full_parse() -> None:
    """The fast table read rote.skills uses is the full parse's own registry."""
    from rote.compiler.system1.parse import parse_skill, tool_registry

    skills = sorted(p.parent for p in REPO.glob("examples/*/skill/SKILL.md"))
    assert len(skills) >= 4
    for skill_dir in skills:
        assert tool_registry(skill_dir) == parse_skill(skill_dir).tool_registry, skill_dir


def test_skills_list_command(
    machine: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    home, project = machine
    _skill(home / ".claude" / "skills", "uses-crm", "Call `mcp__crm__get_contact`.")
    monkeypatch.setenv("ROTE_CLIENT_HOME_DIR", str(home))
    monkeypatch.chdir(project)

    assert cli_main(["skills", "list", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    states = [(s["name"], s["readiness"]["state"]) for s in payload["skills"]]
    assert states[-1] == ("uses-crm", "needs_server")
    assert states[0][1] == "bundled"

    # The bundled set grows as demos ship, so the counts come from the
    # listing; two limits, so no constant can match both.
    total = payload["total"]
    assert total == len(payload["skills"]) >= 3
    for limit in (1, 2):
        assert cli_main(["skills", "list", "--limit", str(limit)]) == 0
        out = capsys.readouterr().out.splitlines()
        assert out[0].split()[0] == "demo"
        assert len(out) == limit + 1
        assert out[-1] == f"  ... {total - limit} more (drop --limit to see them)"


# ── hostile skills (a .claude/skills/<x> in a cloned repository) ─────────


def _sparse(path: Path, size: int, head: str = "") -> None:
    """A file of ``size`` bytes that costs no disk and no time to make."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(head.encode())
        f.truncate(size)


def test_deeply_nested_frontmatter_is_a_problem_not_a_crash(machine: tuple[Path, Path]) -> None:
    home, project = machine
    root = home / ".claude" / "skills"
    nested = root / "nested"
    nested.mkdir(parents=True)
    (nested / "SKILL.md").write_text("---\nname: " + "[" * 5000 + "]" * 5000 + "\n---\n")
    _skill(root, "fine")
    found = _found(_ctx(home, project))
    assert set(found) == {"nested", "fine"}
    assert found["nested"]["problems"] == ["its frontmatter nests too deeply to read"]
    assert found["fine"]["readiness"]["state"] == "ready"


def test_a_skill_that_raises_anything_is_listed_unreadable(
    machine: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    home, project = machine
    _skill(home / ".claude" / "skills", "boom")

    def explode(*_: object, **__: object) -> None:
        raise MemoryError("the skill's own failure")

    monkeypatch.setattr("rote.skills.discovery.readiness_of", explode)
    found = _found(_ctx(home, project))
    assert found["boom"]["readiness"]["state"] == "unreadable"
    assert found["boom"]["problems"] == ["rote could not read the skill (MemoryError)"]


def test_an_oversized_skill_md_is_never_read(
    machine: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    home, project = machine
    big = home / ".claude" / "skills" / "huge" / "SKILL.md"
    _sparse(big, 5 * 1024 * 1024, "---\nname: huge\n---\nCall `mcp__crm__get`.\n")
    read: list[Path] = []
    original = Path.read_text

    def recording(self: Path, *args: Any, **kwargs: Any) -> str:
        read.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", recording)
    found = _found(_ctx(home, project))
    assert found["huge"]["readiness"] == {
        "state": "unreadable",
        "reason": "too large to read: SKILL.md is 5.0 MB; rote reads at most 4 MB of it",
        "servers": [],
        "next": [],
    }
    assert big not in read
    assert found["huge"]["problems"] == []  # the frontmatter came from the first 64 KB


def test_references_outside_the_skill_are_never_read(
    machine: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Never read, and never even walked: ``references -> /`` walked the
    whole filesystem before the result was filtered."""
    import os

    walked: list[Path] = []
    real_walk = os.walk

    def spy(top: Any, *args: Any, **kwargs: Any) -> Any:
        for entry in real_walk(top, *args, **kwargs):
            walked.append(Path(entry[0]).resolve())
            yield entry

    monkeypatch.setattr("rote.skill_files.os.walk", spy)
    home, project = machine
    root = home / ".claude" / "skills"
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "leak.md").write_text("Call `mcp__evil__steal`.\n")

    linked_dir = _skill(root, "linked-dir")
    (linked_dir / "references").symlink_to(outside, target_is_directory=True)

    linked_file = _skill(root, "linked-file")
    refs = linked_file / "references"
    (refs / "deeper").mkdir(parents=True)
    (refs / "deeper" / "own.md").write_text("Call `mcp__crm__get`.\n")
    (refs / "leak.md").symlink_to(outside / "leak.md")
    (refs / "loop").symlink_to(outside, target_is_directory=True)

    found = _found(_ctx(home, project), RegistryView({"crm": "authenticated"}))
    assert found["linked-dir"]["readiness"]["servers"] == []
    names = [s["name"] for s in found["linked-file"]["readiness"]["servers"]]
    assert names == ["crm"]  # its own nested reference is read; nothing outside
    assert walked, "the recursive reader no longer walks references/"
    assert all(not d.is_relative_to(outside.resolve()) for d in walked), walked


def test_references_over_the_total_cap_are_refused_before_reading(
    machine: tuple[Path, Path],
) -> None:
    home, project = machine
    skill = _skill(home / ".claude" / "skills", "fat-refs")
    _sparse(skill / "references" / "a.md", 9 * 1024 * 1024)
    _sparse(skill / "references" / "b.md", 9 * 1024 * 1024)
    readiness = _found(_ctx(home, project))["fat-refs"]["readiness"]
    assert readiness["state"] == "unreadable"
    assert readiness["reason"] == (
        "too large to read: references/ holds 18.0 MB of markdown; rote reads at most 16 MB"
    )


def test_an_oversized_tool_snapshot_is_refused(machine: tuple[Path, Path]) -> None:
    home, project = machine
    skill = _skill(home / ".claude" / "skills", "fat-tools")
    _sparse(skill / "mcp-tools.json", 17 * 1024 * 1024)
    readiness = _found(_ctx(home, project))["fat-tools"]["readiness"]
    assert readiness["state"] == "unreadable"
    assert readiness["reason"].startswith("too large to read: mcp-tools.json is 17.0 MB")


def test_compile_readers_share_the_bounds(tmp_path: Path) -> None:
    """Tool grounding and the System-1 parser read a chosen skill through the
    same bounded reader, so a compile gets the same protection."""
    from rote.compiler.system1.parse import parse_skill, tool_registry
    from rote.compiler.tool_grounding import servers_named
    from rote.skill_files import SkillTooLarge

    skill = tmp_path / "huge"
    _sparse(skill / "SKILL.md", 5 * 1024 * 1024, "# Huge\n")
    for reader in (servers_named, parse_skill, tool_registry):
        with pytest.raises(SkillTooLarge, match="SKILL.md is 5.0 MB"):
            reader(skill)


def test_refreshable_auth_counts_as_ready_and_plain_expiry_does_not(
    machine: tuple[Path, Path],
) -> None:
    home, project = machine
    _skill(home / ".claude" / "skills", "uses-crm", "Call `mcp__crm__get_contact`.")
    for auth, state in (
        ("expired (refreshable)", "ready"),
        ("static headers", "ready"),
        ("no auth required", "ready"),
        ("expired", "needs_login"),
        ("auth unknown", "needs_login"),
    ):
        found = _found(_ctx(home, project), RegistryView({"crm": auth}))
        assert found["uses-crm"]["readiness"]["state"] == state, auth


def test_tool_registry_refuses_an_unsafe_reference_name(tmp_path: Path) -> None:
    from rote.compiler.system1.parse import UnsafeSkillText, parse_skill, tool_registry

    skill = _skill(tmp_path, "s", "| tool | use |\n|---|---|\n| `get_x` | read |\n")
    (skill / "references").mkdir()
    (skill / "references" / "bad name.md").write_text("text\n")
    for reader in (tool_registry, parse_skill):
        with pytest.raises(UnsafeSkillText):
            reader(skill)


def test_dot_directories_are_not_skills(machine: tuple[Path, Path]) -> None:
    home, project = machine
    root = home / ".claude" / "skills"
    _skill(root, ".hidden")
    _skill(root, "visible")
    assert set(_found(_ctx(home, project))) == {"visible"}
