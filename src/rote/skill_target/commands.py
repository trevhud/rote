"""``rote skill install / trust / list / runs``: the person's side of a compiled skill.

``rote skill run`` (:mod:`rote.skill_target.launch`) is the agent's side,
and it runs only what these commands recorded. Each one that writes builds
a :class:`~rote.interact.plan.Plan` and applies it only once confirmed
(``rote.interact``, ``docs/design/first-run.md``):

- **install** records the skill and the hash of every file in it, after
  checking the dir is exactly what rote emitted. Saving the Claude Code
  allow rule is a second, separate Action (:mod:`rote.skill_target.allow_rule`):
  a person is asked with "no" as the default, and without one only
  ``--save-allow-rule`` saves it, never ``--yes``.
- **trust** shows what changed since the last install or trust, as a diff
  summary (``--diff`` prints the diffs), and records the new hashes.
  ``--allow-writes`` lists every tool the plan calls that may write and
  records the consent, bound to this plan. That consent is what lets the
  driver call a write tool, and the driver takes it from nowhere else. It
  needs a person at a terminal, whatever flags are passed: ``--yes`` is what
  an agent passes, and an agent is who the consent guards against, since the
  driver's tool calls skip the client's own per-tool prompts.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from rote.app_registry import RegisteredApp, apps_path, parse_apps, render_apps
from rote.interact.output import Output
from rote.interact.plan import (
    FORCE_FLAG,
    Action,
    ConfirmationRequired,
    FileWrite,
    Plan,
    Snapshot,
    apply_plan,
    atomic_write_text,
    confirm_plan,
    snapshot,
)
from rote.interact.prompter import PromptRequired
from rote.interact.report import PersonRequired
from rote.skill_target import trust
from rote.skill_target.trust import SkillTrustError

SAVE_RULE_FLAG = "--save-allow-rule"


def _mode(args: argparse.Namespace) -> Any:
    from rote.interact.mode import Mode

    return Mode.resolve(
        json=bool(getattr(args, "json", False)),
        yes=bool(getattr(args, "yes", False)),
        agent_mode=getattr(args, "agent_mode", None),
    )


def _fail(out: Output, error: SkillTrustError, **extra: Any) -> int:
    out.error(f"{error.cause} Fix: {error.fix}")
    out.result({"ok": False, "error": error.cause, "fix": error.fix, **extra})
    return 2 if error.usage else 1


def _registry() -> tuple[Snapshot, list[RegisteredApp]]:
    read = snapshot(apps_path())
    try:
        return read, parse_apps(read.text())
    except ValueError as e:
        raise SkillTrustError(
            f"the app registry {read.path} cannot be read: {e}.",
            "fix or remove that file, then run the command again",
        ) from None


def _record_action(
    *,
    id: str,
    title: str,
    read: Snapshot,
    apps: list[RegisteredApp],
    root: Path,
    files: dict[str, str],
    summary: str,
    replaces: bool,
    command: str,
) -> Action:
    """Write the registry (stale-checked against ``read``) and keep copies of ``files``."""
    content = render_apps(apps)
    state = read.state
    store = trust.store_dir()

    def run() -> Path:
        trust.keep_copies(root, files)
        atomic_write_text(read.path, content, expect=state)
        return read.path

    return Action(
        id=id,
        title=title,
        effects=(
            FileWrite(read.path, summary, existed=read.existed, replaces=replaces),
            FileWrite(
                store,
                f"keep a copy of each file, for rote skill trust's diffs ({len(files)} files)",
                existed=store.exists(),
            ),
        ),
        run=run,
        command=command,
    )


def _stop_for_action(
    out: Output, error: PromptRequired | ConfirmationRequired | PersonRequired, argv: list[str]
) -> int:
    """Exit 2 with the ``action_required`` object: nothing was written."""
    from rote.interact.report import EXIT_ACTION_REQUIRED, action_required, person_required

    out.error(str(error))
    if isinstance(error, PersonRequired):
        out.result(person_required(error))
    else:
        out.result(action_required(error, argv))
    return EXIT_ACTION_REQUIRED


def _report(out: Output, plan: Plan) -> tuple[bool, list[dict[str, Any]]]:
    outcomes = apply_plan(plan)
    for outcome in outcomes:
        line = f"  {outcome.status}: {outcome.action_id}"
        if outcome.error:
            line += f" ({outcome.error})"
        out.say(line)
    return all(o.status == "done" for o in outcomes), [o.to_dict() for o in outcomes]


# ───────── install ─────────


def install_plan(
    skill: trust.SkillDir, read: Snapshot, apps: list[RegisteredApp], now: trust.Scan
) -> Plan:
    """The plan that records ``skill`` as installed, with every file in ``now``."""
    existing = next((a for a in apps if a.path == skill.root), None)
    previous = existing.trust if existing is not None else None
    displaced = [
        a
        for a in apps
        if a.runtime == trust.SKILL_RUNTIME
        and a.name == skill.name
        and a.trust is not None
        and a.path != skill.root
    ]
    notes: list[str] = []
    writes = None
    if previous is not None:
        same_files = previous.get("files") == now.files
        same_paths = previous.get("bindings") == trust.bindings()
        if same_files and same_paths and not displaced:
            return Plan(
                f"install {skill.name}",
                notes=(f"{skill.name} is already installed from {skill.root}",),
            )
        recorded = previous.get("files") or {}
        notes += trust.diff_summary(skill.root, trust.drift(now, recorded), recorded)
        old_writes = previous.get("writes")
        if isinstance(old_writes, dict):
            if old_writes.get("plan_sha256") == trust.plan_sha256(skill):
                writes = old_writes
            else:
                notes.append(
                    "the plan changed, so the earlier write consent is dropped "
                    f"(rote skill trust {skill.name} --allow-writes to give it again)"
                )
    for other in displaced:
        notes.append(f"replaces the install of {skill.name} from {other.path}")
    record = trust.install_record(now.files, writes=writes)
    updated = [
        replace(a, trust=None) if a in displaced else a for a in apps if a.path != skill.root
    ]
    updated.append(
        RegisteredApp(
            path=skill.root,
            runtime=trust.SKILL_RUNTIME,
            pipeline=skill.pipeline,
            name=skill.name,
            trust=record,
        )
    )
    action = _record_action(
        id=f"install-{skill.name}",
        title=f"Install {skill.name} from {skill.root}",
        read=read,
        apps=updated,
        root=skill.root,
        files=now.files,
        summary=f"record {skill.name} and the hashes of its {len(now.files)} files",
        replaces=bool(displaced),
        command=f"rote skill install {skill.root}",
    )
    return Plan(f"install {skill.name}", (action,), tuple(notes))


def cmd_install(args: argparse.Namespace) -> int:
    from rote.interact import prompter as prompting
    from rote.interact.prompter import PromptCancelled, PromptRequired

    try:
        mode = _mode(args)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    out = Output(mode)
    prompter = prompting.prompter_for(mode)
    argv: list[str] = list(getattr(args, "raw_argv", sys.argv[1:]))
    try:
        skill = trust.read_skill_dir(Path(args.dir))
        now = trust.scan(skill.root)
        found = trust.against_manifest(skill, now)
        if not found.clean:
            raise SkillTrustError(
                f"{skill.root} is not what rote emitted ({found.describe()}).",
                "re-emit it (rote emit <pipeline.yaml> --runtime skill --out "
                f"{skill.root}), or remove the files listed",
            )
        read, apps = _registry()
        plan = install_plan(skill, read, apps, now)
    except SkillTrustError as e:
        return _fail(out, e)
    out.plan(plan)
    try:
        granted = {FORCE_FLAG} if args.force else set()
        if not confirm_plan(plan, prompter, yes=mode.yes, granted=granted):
            out.say(f"rote skill install: {skill.name} was not installed")
            out.result({"ok": False, "applied": False, "plan": plan.to_dict()})
            return 1
    except (PromptRequired, ConfirmationRequired) as e:
        return _stop_for_action(out, e, argv)
    except PromptCancelled:
        out.say("rote skill install: cancelled; nothing was written")
        return 130
    ok, outcomes = _report(out, plan)
    payload: dict[str, Any] = {
        "ok": ok,
        "applied": True,
        "skill": skill.name,
        "path": str(skill.root),
        "plan": plan.to_dict(),
        "outcomes": outcomes,
    }
    if not ok:
        out.result(payload)
        return 1
    try:
        payload["allow_rule"] = _offer_allow_rule(args, skill, prompter, out)
    except PromptCancelled:
        out.say("rote skill install: the allow rule was not saved")
        payload["allow_rule"] = {"saved": False, "why": "cancelled"}
    except ValueError as e:
        out.error(str(e))
        payload["allow_rule"] = {"saved": False, "why": str(e)}
        payload["ok"] = False
        out.result(payload)
        return 1
    out.say(f"Installed {skill.name}. An agent runs it with: rote skill run {skill.name} start ...")
    out.result(payload)
    return 0 if payload["allow_rule"].get("error") is None else 1


def _offer_allow_rule(
    args: argparse.Namespace, skill: trust.SkillDir, prompter: Any, out: Output
) -> dict[str, Any]:
    """Save the Claude Code allow rule: asked (default no), or ``--save-allow-rule``.

    Never ``--yes``: it changes another program's permissions."""
    from rote.skill_target.allow_rule import allow_rule_action, default_project, rule_for

    project = (
        Path(args.claude_project).resolve()
        if args.claude_project
        else default_project(skill.root, Path.cwd())
    )
    command = f"rote skill install {skill.root} {SAVE_RULE_FLAG} --claude-project {project}"
    action = allow_rule_action(skill.name, project, command)
    rule = rule_for(skill.name)
    if action is None:
        out.say(f"Claude Code already allows {rule} in {project}")
        return {"saved": True, "rule": rule, "already": True}
    plan = Plan(f"save the Claude Code allow rule for {skill.name}", (action,))
    if args.save_allow_rule:
        approved = True
        out.plan(plan)
    elif prompter.interactive:
        out.plan(plan)
        approved = prompter.confirm(
            f"Save {rule} for Claude Code in {project}?", flag=SAVE_RULE_FLAG, default=False
        )
    else:
        out.say(
            f"Not saving the Claude Code allow rule (it needs {SAVE_RULE_FLAG}; --yes does not "
            f"cover it): {command}"
        )
        return {"saved": False, "rule": rule, "next": [{"command": command}]}
    if not approved:
        return {"saved": False, "rule": rule, "next": [{"command": command}]}
    (outcome,) = apply_plan(plan)
    out.say(
        f"  {outcome.status}: {outcome.action_id}"
        + (f" ({outcome.error})" if outcome.error else "")
    )
    result: dict[str, Any] = {"saved": outcome.status == "done", "rule": rule}
    if outcome.error:
        result["error"] = outcome.error
    return result


# ───────── trust ─────────


def cmd_trust(args: argparse.Namespace) -> int:
    from rote.interact import prompter as prompting
    from rote.interact.prompter import PromptCancelled, PromptRequired
    from rote.interact.report import PersonRequired

    try:
        mode = _mode(args)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    out = Output(mode)
    prompter = prompting.prompter_for(mode)
    argv: list[str] = list(getattr(args, "raw_argv", sys.argv[1:]))
    report: dict[str, Any] = {}
    try:
        read, apps = _registry()
        app = trust.installed(args.name, apps)
        plan, report = trust_plan(app, read, apps, allow_writes=args.allow_writes, out=out)
        if args.diff:
            recorded = (app.trust or {}).get("files") or {}
            found = trust.drift(trust.scan(app.path), recorded)
            for line in trust.unified_diffs(app.path, found, recorded):
                out.say(line)
        out.plan(plan)
        if report.get("grants_writes") and not prompter.interactive:
            raise PersonRequired(
                f"letting {app.name}'s tools write needs the user at a terminal: --yes cannot "
                "stand in for them (the driver's tool calls skip the client's own prompts). "
                "Nothing was written.",
                f"rote skill trust {app.name} --allow-writes",
                plan,
            )
        if not confirm_plan(plan, prompter, yes=mode.yes):
            out.say(f"rote skill trust: nothing was recorded for {app.name}")
            out.result({"ok": False, "applied": False, "plan": plan.to_dict(), **report})
            return 1
    except SkillTrustError as e:
        return _fail(out, e)
    except (PromptRequired, ConfirmationRequired, PersonRequired) as e:
        return _stop_for_action(out, e, argv)
    except PromptCancelled:
        out.say("rote skill trust: cancelled; nothing was written")
        return 130
    ok, outcomes = _report(out, plan)
    out.result(
        {
            "ok": ok,
            "applied": True,
            "skill": app.name,
            "plan": plan.to_dict(),
            "outcomes": outcomes,
            **report,
        }
    )
    return 0 if ok else 1


def trust_plan(
    app: RegisteredApp,
    read: Snapshot,
    apps: list[RegisteredApp],
    *,
    allow_writes: bool,
    out: Output | None = None,
) -> tuple[Plan, dict[str, Any]]:
    """The plan that records ``app``'s files as they are now (and, with
    ``allow_writes``, consent for its write tools); and what it found."""
    assert app.trust is not None and app.name is not None
    name = app.name
    root = app.path
    if not root.is_dir() or root.is_symlink():
        raise SkillTrustError(
            f"{name} is installed from {root}, which is not a directory any more.",
            "rote skill install <the skill's dir>",
        )
    recorded: dict[str, str] = app.trust.get("files") or {}
    now = trust.scan(root)
    found = trust.drift(now, recorded)
    lines = trust.diff_summary(root, found, recorded)
    if found.untrustable:
        raise SkillTrustError(
            f"{name}'s dir holds what no record can vouch for: {found.describe()}.",
            "remove those entries (copy a symlink's target in its place if it is needed), "
            f"then run rote skill trust {name} again",
        )
    skill = trust.read_skill_dir(root)
    moved = trust.moved_bindings(app.trust.get("bindings") or {}, trust.bindings())
    writes = app.trust.get("writes")
    notes = [*lines, *(f"moved: {m}" for m in moved)]
    if isinstance(writes, dict) and writes.get("plan_sha256") != trust.plan_sha256(skill):
        notes.append(
            "the plan changed, so the earlier write consent is dropped"
            + ("" if allow_writes else f" (rote skill trust {name} --allow-writes gives it again)")
        )
        writes = None
    report: dict[str, Any] = {"drift": found.to_dict(), "moved": moved, "grants_writes": False}
    if allow_writes:
        tools = trust.write_tools(skill)
        report["write_tools"] = list(tools.may_write)
        report["read_only_tools"] = list(tools.read_only)
        for tool in tools.read_only:
            notes.append(f"read-only (its server is registered read-only): {tool}")
        if not tools.may_write:
            notes.append(f"no tool {name} calls needs write consent")
        elif not (isinstance(writes, dict) and writes.get("tools") == list(tools.may_write)):
            writes = {
                "plan_sha256": trust.plan_sha256(skill),
                "tools": list(tools.may_write),
                "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            report["grants_writes"] = True
            notes += [f"may write: {tool}" for tool in tools.may_write]
            notes.append(
                "listed because their servers are not registered read-only (rote mcp add "
                "--read-only); a tool its server marks readOnlyHint runs without this consent"
            )
    changes_files = not found.clean or bool(moved)
    if not changes_files and not report["grants_writes"] and writes == app.trust.get("writes"):
        return Plan(f"trust {name}", notes=(*notes, f"{name} is as the user installed it")), report
    record = trust.install_record(now.files, writes=writes)
    updated = [replace(a, trust=record) if a.path == root else a for a in apps]
    parts = []
    if not found.clean:
        parts.append(f"{len(lines)} file change(s)")
    if moved:
        parts.append(f"{len(moved)} moved path(s)")
    if report["grants_writes"]:
        parts.append(f"write consent for {len(report['write_tools'])} tool(s)")
    if not parts:
        parts.append("drop the write consent")
    action = _record_action(
        id=f"trust-{name}",
        title=f"Trust {name} as it is now ({root})",
        read=read,
        apps=updated,
        root=root,
        files=now.files,
        summary="record " + ", ".join(parts),
        replaces=False,
        command=f"rote skill trust {name}" + (" --allow-writes" if allow_writes else ""),
    )
    return Plan(f"trust {name}", (action,), tuple(notes)), report


# ───────── list and runs ─────────


def _status(app: RegisteredApp) -> tuple[str, str | None]:
    if app.trust is None:
        return "not installed", f"rote skill install {app.path}"
    try:
        trust.check(app)
    except SkillTrustError as e:
        return f"refused: {e.cause}", e.fix
    return "installed", None


def cmd_list(args: argparse.Namespace) -> int:
    from rote.app_registry import registered_apps

    try:
        apps = [a for a in registered_apps() if a.runtime == trust.SKILL_RUNTIME and a.name]
    except (OSError, ValueError) as e:
        print(f"error: the app registry {apps_path()} cannot be read: {e}", file=sys.stderr)
        return 1
    rows = []
    for app in apps:
        status, fix = _status(app)
        writes = (app.trust or {}).get("writes")
        rows.append(
            {
                "name": app.name,
                "path": str(app.path),
                "status": "installed" if status == "installed" else status.split(":")[0],
                "detail": None if status in ("installed", "not installed") else status,
                "fix": fix,
                "writes": list(writes.get("tools") or []) if isinstance(writes, dict) else [],
            }
        )
    if getattr(args, "json", False):
        import json

        print(json.dumps({"ok": True, "skills": rows}, indent=2))
        return 0
    if not rows:
        print("No compiled skills. Emit one with --runtime skill, then: rote skill install <dir>")
        return 0
    for row in rows:
        writes_note = f"  writes: {', '.join(row['writes'])}" if row["writes"] else ""
        print(f"{row['name']}  {row['status']}  {row['path']}{writes_note}")
        if row["detail"]:
            print(f"    {row['detail']}")
        if row["fix"]:
            print(f"    fix: {row['fix']}")
    return 0


def cmd_runs(args: argparse.Namespace) -> int:
    from rote.app_registry import registered_apps
    from rote.skill_target._driver import Driver, Stop, _iso

    try:
        apps = [
            a
            for a in registered_apps()
            if a.runtime == trust.SKILL_RUNTIME and a.name and a.trust is not None
        ]
    except (OSError, ValueError) as e:
        print(f"error: the app registry {apps_path()} cannot be read: {e}", file=sys.stderr)
        return 1
    rows: list[dict[str, Any]] = []
    pruned: dict[str, list[str]] = {}
    for app in apps:
        assert app.name is not None
        scripts = app.path / "scripts"
        if not (scripts / "plan.json").is_file():
            continue
        try:
            # Reads plan.json and the run journals only: no skill code is imported.
            driver = Driver(scripts, skill=app.name)
        except (Stop, OSError, ValueError, KeyError):
            continue
        if args.prune:
            pruned[app.name] = driver.sweep()
        for path, state in sorted(driver._runs(), key=lambda r: r[1].get("updated", 0)):
            rows.append(
                {
                    "skill": app.name,
                    "run": path.name,
                    "state": state.get("state"),
                    "updated": _iso(float(state.get("updated", 0))),
                    "pending": (state.get("pending") or {}).get("step"),
                }
            )
    if getattr(args, "json", False):
        import json

        payload: dict[str, Any] = {"ok": True, "runs": rows}
        if args.prune:
            payload["pruned"] = pruned
        print(json.dumps(payload, indent=2))
        return 0
    for row in rows:
        pending = f"  waiting on {row['pending']}" if row["pending"] else ""
        print(f"{row['skill']}  {row['run']}  {row['state']}  {row['updated']}{pending}")
    if args.prune:
        total = sum(len(v) for v in pruned.values())
        print(f"pruned {total} expired run(s)")
    elif not rows:
        print("No runs.")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from rote.skill_target.launch import run

    return run(args.name, list(args.driver_args))
