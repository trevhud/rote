"""Compiled-skill adapter (``--runtime skill``): a pipeline as a skill any agent client loads.

Layer 3 of rote, for the one runtime whose "orchestrator" is the agent that
loaded the skill. The emitted directory is a skill: the original skill's
``name`` and ``description`` (so every client triggers on it exactly as it
did on the original), a short SKILL.md body that says how to start a run
and answer what it asks, and a ``scripts/`` dir that does the work in code.
See ``docs/design/compiled-skill-target.md``.

Output layout::

    <skill>/
        SKILL.md                    # original name + description; a ~12-line body
        README.md                   # for people: what it runs, how to run and test it
        .rote-manifest.json         # EmitWriter's hashes for every file, plus the
                                    # skill's name and judge mode
        scripts/
            run.py                  # entry point (PEP 723 header): argv -> _rote_skill.main
            plan.json               # rote.plan.build_plan, live: data, no code
            _rote_skill.py          # verbatim rote.skill_target._driver
            _rote_schema.py         # verbatim rote.plan._schema
            _rote_mcp.py            # verbatim rote.mcp._runtime_helper, when a step calls MCP
            _rote_inference.py      # verbatim rote.inference._runtime_helper, --judges helper
            extracted/<module>.py   # the compile's real implementations

There is no generated workflow code. The driver interprets ``plan.json``,
which the one plan builder writes (:mod:`rote.plan`), so the program a
compiled skill runs is the one compile-time verification checked.

The manifest covers SKILL.md and README.md too, not only the scripts: a
skill's instructions are as much its program as its code.

Refused at emit, before any file is written:

- a skill identity rote cannot vouch for (:func:`identity_for`);
- a plan the driver cannot run: a :class:`rote.plan.PlanError` (a legacy
  ``signature:`` judge, a node with nothing to run on the backend) or a
  step kind the driver does not run yet (an agent loop), naming the node;
- an ``extracted/`` module that is still a stub, unless ``allow_stubs``
  (the preflight ``rote run`` applies, moved to emit because a compiled
  skill is run by an agent that cannot fix one);
- impls that cannot take their node's constants;
- under ``--judges helper``, a judge output schema structured outputs
  cannot express (as every Python runtime refuses it).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from rote.adapters._common import (
    MANIFEST_NAME,
    EmitWriter,
    _is_stub_module,
    escape_invisible,
    mcp_backed,
    py_str_literal,
    safe_docstring_line,
)
from rote.adapters._py_common import (
    _extracted_layout,
    _impl_path_parts,
    emit_extracted_module,
    helper_module_names,
    require_constant_parameters,
    resolve_extracted_source,
    spec_judges,
    vendor_sdk_requirements,
    write_extracted_package,
)
from rote.ir import Pipeline, is_field_name
from rote.plan import JUDGE_MODES, JudgeMode, PlanError, build_plan, plan_hash, plan_text

#: The manifest's record of what this emit decided (:func:`emitted_skill`).
MANIFEST_KEY = "skill"
SKILL_FILE = "SKILL.md"
GENERATED_BY = "rote.adapters.skill"
_CALLER_NOTE = (
    "the compiled skill's driver (scripts/_rote_skill.py)\ncalls these with the step "
    "payload as keyword arguments."
)
#: agentskills.io caps ``compatibility`` at 500 characters.
COMPATIBILITY_MAX = 500


class SkillIdentityError(ValueError):
    """rote cannot say which skill this pipeline was compiled from."""


@dataclass(frozen=True)
class SkillIdentity:
    """What the compiled skill keeps of the original: its name and description."""

    name: str
    description: str
    skill_md: Path


@dataclass(frozen=True)
class SkillAdapterConfig:
    #: The source skill's directory, when the caller knows it (``compile``
    #: does; ``emit --skill``). Otherwise the pipeline's ``source_skill``,
    #: vouched for by the provenance beside pipeline.yaml (:func:`identity_for`).
    skill_dir: Path | None = None
    #: Directory of the pipeline.yaml: its ``extracted/`` modules ship
    #: verbatim, and its ``provenance.json`` vouches for ``source_skill``.
    extracted_source_dir: Path | None = None
    external_backend: Literal["mcp", "api"] = "mcp"
    judges: JudgeMode = "session"
    allow_stubs: bool = False


# ───────── the source skill's identity ─────────


def read_identity(skill_dir: Path) -> SkillIdentity:
    """The ``name`` and ``description`` of the SKILL.md in ``skill_dir``.

    Refused, never guessed: a missing SKILL.md, a frontmatter that does not
    parse, a missing name (a client would fall back to the folder's name,
    which is not something a compiled skill should copy), a name the
    agentskills.io rule rejects (it is a folder under the run root and a
    word in the shell command the SKILL.md gives), or no description.
    """
    from rote.skill_target._driver import SKILL_NAME, SKILL_NAME_MAX
    from rote.skills.discovery import read_frontmatter

    skill_md = Path(skill_dir) / SKILL_FILE
    if not skill_md.is_file():
        raise SkillIdentityError(f"{skill_md} does not exist")
    meta, problem = read_frontmatter(skill_md)
    if problem:
        raise SkillIdentityError(f"{skill_md}: {problem}")
    name, description = meta.get("name"), meta.get("description")
    if not isinstance(name, str) or not name:
        raise SkillIdentityError(
            f"{skill_md} has no `name:` in its frontmatter, and a compiled skill keeps the "
            "original's name"
        )
    if len(name) > SKILL_NAME_MAX or not SKILL_NAME.fullmatch(name):
        raise SkillIdentityError(
            f"{skill_md}: name {name!r} is not a skill name: lowercase letters, digits and "
            f"single hyphens, at most {SKILL_NAME_MAX} characters (the agentskills.io rule)"
        )
    if not isinstance(description, str) or not description.strip():
        raise SkillIdentityError(
            f"{skill_md} has no `description:` in its frontmatter, and a compiled skill "
            "keeps the original's description"
        )
    return SkillIdentity(name=name, description=description, skill_md=skill_md)


def _skill_md_sha256(skill_md: Path) -> str:
    """The digest ``provenance.json`` records (rote.skill_source.build_provenance)."""
    from rote.skill_files import read_skill_md

    return hashlib.sha256(read_skill_md(skill_md.parent).encode("utf-8")).hexdigest()


def _recorded_sha256(provenance: Path) -> str:
    try:
        data = json.loads(provenance.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise SkillIdentityError(f"{provenance} cannot be read: {e}") from e
    digest = data.get("skill_md_sha256") if isinstance(data, dict) else None
    if not isinstance(digest, str):
        raise SkillIdentityError(f"{provenance} records no skill_md_sha256")
    return digest


def identity_for(
    pipeline: Pipeline, *, skill_dir: Path | None, source_dir: Path | None
) -> SkillIdentity:
    """The source skill's identity, from a SKILL.md rote can vouch for.

    With ``skill_dir`` (``rote compile``, or ``rote emit --skill <dir>``),
    that SKILL.md, held to the ``provenance.json`` beside pipeline.yaml when
    there is one: a skill edited since the compile is refused, because the
    compiled skill would claim a name and description its pipeline was not
    compiled from. Without it, the pipeline's ``source_skill`` resolved from
    pipeline.yaml's directory, and then the provenance is required: the
    path is written by the compile agent, so only the digest rote recorded
    says it is the skill this pipeline came from.
    """
    from rote.skill_source import PROVENANCE_FILENAME

    provenance = source_dir / PROVENANCE_FILENAME if source_dir is not None else None
    if skill_dir is not None:
        identity = read_identity(skill_dir)
        if (
            provenance is not None
            and provenance.is_file()
            and _skill_md_sha256(identity.skill_md) != _recorded_sha256(provenance)
        ):
            raise SkillIdentityError(
                f"{identity.skill_md} has changed since this pipeline was compiled "
                f"(its digest is not the one {provenance} records), so the compiled skill "
                "would carry a name and description it was not compiled from. Recompile "
                "it (rote compile --update)"
            )
        return identity
    fix = "pass the source skill's directory with --skill <dir>"
    if source_dir is None or not pipeline.source_skill:
        raise SkillIdentityError(
            f"the compiled skill keeps the source skill's name and description, and this "
            f"pipeline does not say where its source skill is: {fix}"
        )
    candidate = (source_dir / pipeline.source_skill).resolve()
    assert provenance is not None
    if not provenance.is_file():
        raise SkillIdentityError(
            f"pipeline.yaml names its source skill as {pipeline.source_skill!r}, but no "
            f"{PROVENANCE_FILENAME} beside it records which SKILL.md it was compiled from, so "
            f"rote cannot vouch for that path: {fix}"
        )
    if not (candidate / SKILL_FILE).is_file():
        raise SkillIdentityError(
            f"pipeline.yaml names its source skill as {pipeline.source_skill!r}, and "
            f"{candidate / SKILL_FILE} does not exist: {fix}"
        )
    identity = read_identity(candidate)
    if _skill_md_sha256(identity.skill_md) != _recorded_sha256(provenance):
        raise SkillIdentityError(
            f"{identity.skill_md} is not the SKILL.md this pipeline was compiled from (its "
            f"digest is not the one {provenance} records): {fix}, or recompile"
        )
    return identity


# ───────── what a plan can print ─────────


def refusal(pipeline: Pipeline, backend: Literal["mcp", "api"] = "mcp") -> str | None:
    """Why the skill runtime refuses ``pipeline`` whatever skill it came from; or None.

    ``rote analyze`` lists the runtime as untargetable with this reason.
    """
    try:
        plan = build_plan(pipeline, mcp="live", backend=backend)
    except PlanError as e:
        return str(e)
    from rote.skill_target._driver import unrunnable_step

    return unrunnable_step(plan)


@dataclass(frozen=True)
class _Kinds:
    decide: bool
    loop: bool
    gate: bool
    login: bool


def _kinds(plan: dict[str, Any]) -> _Kinds:
    from rote.skill_target import calls_mcp

    steps = plan["steps"]
    return _Kinds(
        decide=plan.get("judges") != "helper" and any(s["kind"] == "judge" for s in steps),
        loop=any(s["kind"] == "loop" for s in steps),
        gate=any(s["kind"] == "gate" for s in steps),
        login=calls_mcp(plan),
    )


# ───────── SKILL.md ─────────

_PLACEHOLDER_BY_TYPE = {
    "string": "...",
    "integer": 0,
    "number": 0,
    "boolean": True,
    "array": [],
    "object": {},
}


def _field_type(schema: dict[str, Any], name: str) -> str:
    prop = (schema.get("properties") or {}).get(name)
    kind = prop.get("type") if isinstance(prop, dict) else None
    if isinstance(kind, str) and kind in _PLACEHOLDER_BY_TYPE:
        return kind
    if isinstance(kind, list) and kind and all(k in (*_PLACEHOLDER_BY_TYPE, "null") for k in kind):
        return " or ".join(kind)
    return "any"


def _input_lines(pipeline: Pipeline) -> tuple[str, str]:
    """``(the input line, the start example's JSON)`` for the SKILL.md body.

    Only identifier-shaped field names appear (every name a node can bind
    is one), and the example is JSON with every ``'`` spelled ``\\u0027``,
    so it is safe inside the single-quoted shell argument it is shown in.
    """
    schema = pipeline.input.input_schema or {}
    required = list(pipeline.required_input_fields)
    optional = sorted(n for n in pipeline.optional_input_fields if is_field_name(n))
    shape = ", ".join(f"{json.dumps(n)}: {_field_type(schema, n)}" for n in [*required, *optional])
    parts = [f"Input, one JSON object: {{{shape}}}."]
    if required:
        parts.append("Required: " + ", ".join(required) + ".")
    if optional:
        parts.append("Optional: " + ", ".join(optional) + ".")
    if not required and not optional:
        parts = ["Input: one JSON object (`{}` when there is nothing to pass)."]
    example = {n: _PLACEHOLDER_BY_TYPE.get(_field_type(schema, n), "...") for n in required}
    shown = json.dumps(example, ensure_ascii=True).replace("'", "\\u0027")
    return " ".join(parts), shown


def _servers(plan: dict[str, Any]) -> list[str]:
    """Where a run's tool calls go: each endpoint's host, else the server's name."""
    found: dict[str, None] = {}
    for step in plan["steps"]:
        if step["kind"] != "mcp":
            continue
        host = urlparse(step.get("url") or "").hostname
        safe = host if host and all(c.isalnum() or c in ".-" for c in host) else None
        found.setdefault(safe or f"the {step['server']} MCP server", None)
    return list(found)


def _compatibility(plan: dict[str, Any]) -> str:
    from rote.extras import DISTRIBUTION

    text = f"Needs rote (pip install {DISTRIBUTION})"
    servers = _servers(plan)
    if servers:
        text += " and network access to " + ", ".join(servers)
    text += "."
    return text if len(text) <= COMPATIBILITY_MAX else text[: COMPATIBILITY_MAX - 1] + "…"


def skill_md(identity: SkillIdentity, pipeline: Pipeline, plan: dict[str, Any]) -> str:
    """The compiled skill's SKILL.md: the original's name and description, and a short body.

    The body is all the agent reads before its first call, and it stays in
    context for the rest of the session, so it says only how to start a
    run and what to do with each block this plan can print. Each block
    names its own next command in full. The description is a YAML
    double-quoted scalar that parses to the original's exact value, with
    invisible characters spelled out (invariant #7: a bidi override in it
    cannot hide a line of this file from a reviewer).
    """
    name = identity.name
    command = f"rote skill run {name}"
    digest = plan_hash(plan_text(plan))
    input_line, example = _input_lines(pipeline)
    kinds = _kinds(plan)
    frontmatter = [
        "---",
        f"name: {name}",
        f"description: {py_str_literal(identity.description)}",
        f"allowed-tools: Bash({command} *)",
        f"compatibility: {py_str_literal(_compatibility(plan))}",
        "metadata:",
        "  rote-compiled: " + py_str_literal(f"{pipeline.name} {pipeline.version} {digest[:12]}"),
        "---",
    ]
    blocks = [
        "- DONE: give the user the summary. Read the result.json it names only if they need more.",
        "- ERROR: tell the user what it says. Do not work around it.",
    ]
    if kinds.decide:
        blocks.append(
            "- DECIDE: yours to answer. Decide it yourself, as it asks; do not ask the user. "
            "Then run its `answer` command."
        )
    if kinds.loop:
        blocks.append(
            "- LOOP: yours to do. Do the task yourself with only the tools it lists; do not "
            "ask the user. Then run its `answer` command."
        )
    if kinds.gate:
        blocks.append(
            "- HUMAN: for the user, never for you. Ask them, then run its command with "
            "`--said '<their exact words>'`."
        )
    elif kinds.login:
        blocks.append(
            "- HUMAN: for the user (a login), never for you. Ask them to do it, then run its "
            "command."
        )
    if kinds.decide or kinds.loop or kinds.gate or kinds.login:
        blocks.append(f"- Lost track? `{command} status` prints the pending block again.")
    body = [
        f"# {name} (compiled by rote)",
        "",
        f"Run this skill only through `{command}`. Do not do its steps yourself.",
        "",
        input_line,
        "",
        "Start with one shell command, the input as one single-quoted argument:",
        "",
        f"    {command} start '{example}'",
        "",
        "If the input is already in a file, pass `--file <path>` instead.",
        "",
        "It prints one block, and each block gives its next command in full:",
        *blocks,
    ]
    return "\n".join([*frontmatter, "", *body]) + "\n"


# ───────── README.md and run.py ─────────


def _runs_as(step: dict[str, Any], plan: dict[str, Any]) -> str:
    kind = step["kind"]
    if kind == "call":
        return f"code: `extracted/{step['module'].split('.')[-1]}.py:{step['symbol']}`"
    if kind == "mcp":
        return f"MCP tool `{step['server']}.{step['tool']}`"
    if kind == "judge":
        if plan.get("judges") == "helper":
            return f"inference helper, `{step['lane']['model']}`"
        return "the agent decides (DECIDE)"
    if kind == "gate":
        return "the person answers (HUMAN)"
    return "the agent, with the loop's tools (LOOP)"


def readme(identity: SkillIdentity, pipeline: Pipeline, plan: dict[str, Any]) -> str:
    """For people: what the compiled skill runs, how to run it, and what it guarantees."""
    name = identity.name
    command = f"rote skill run {name}"
    _, example = _input_lines(pipeline)
    by_id = {n.id: n for n in pipeline.nodes}
    rows = [
        f"| `{s['id']}` | {s['node_kind']} | {_runs_as(s, plan)} | "
        f"{safe_docstring_line(by_id[s['id']].description).replace('|', '/')} |"
        for s in plan["steps"]
    ]
    helper = plan.get("judges") == "helper" and any(s["kind"] == "judge" for s in plan["steps"])
    lines = [
        f"# {name}, compiled by rote",
        "",
        escape_invisible(identity.description.strip()),
        "",
        f"This directory is a skill, compiled by rote from `{pipeline.name}` "
        f"v{pipeline.version}. It keeps the original skill's name and description, so a "
        "client triggers on it as it did on the original. Its steps run in code "
        "(`scripts/`); the agent that loaded it answers only what needs a model, and a "
        "person only what needs a person.",
        "",
        "## What it runs",
        "",
        "| Step | Kind | Runs as | What it does |",
        "| --- | --- | --- | --- |",
        *rows,
        "",
        "## Running it",
        "",
        "An agent runs it the way SKILL.md says, one command per block:",
        "",
        f"    {command} start '{example}'",
        "",
        "Without rote installed, for a skill whose servers need no login, `uv` runs the "
        "same driver (the PEP 723 header in `scripts/run.py` names its dependencies):",
        "",
        f"    uv run scripts/run.py start '{example}'",
        "",
        "Runs live under `$ROTE_SKILL_RUN_DIR`, else "
        f"`$XDG_STATE_HOME/rote/skill-runs/{name}/`, else "
        f"`~/.local/state/rote/skill-runs/{name}/`, one folder per run, mode 0700.",
        "",
        "## What holds",
        "",
        "Steps run as compiled. Whether the agent runs the skill through its driver, and "
        "what it answers at DECIDE, is up to the agent. HUMAN answers are recorded with "
        "the person's words.",
        "",
        "- Every tool call is read-only unless the user trusts the skill with writes.",
        "- A tool call that finished is never repeated; one interrupted mid-call runs again "
        "on `resume` (at least once).",
    ]
    if helper:
        lines.append(
            "- Judges run in code through `scripts/_rote_inference.py` on the IR's model, "
            "one call per element (`--judges helper`). Who pays is resolved at run time: "
            "the Claude subscription, an API key, or rote cloud (`ROTE_INFERENCE`)."
        )
    lines += [
        "",
        "## Files",
        "",
        f"`{MANIFEST_NAME}` holds the hash of every file here. Do not edit them: re-emit "
        "instead (`rote emit --runtime skill`), and a file that differs from its hash is "
        "refused before it runs.",
        "",
    ]
    return "\n".join(lines)


def run_py(identity: SkillIdentity, pipeline: Pipeline, plan: dict[str, Any]) -> str:
    """``scripts/run.py``: the entry point, with the dependencies a bare ``uv run`` needs."""
    from rote.mcp._runtime_helper import FASTMCP_REQUIREMENT
    from rote.skill_target import calls_mcp, helper_judges

    deps: list[str] = []
    if calls_mcp(plan):
        deps.append(FASTMCP_REQUIREMENT)
    if helper_judges(plan):
        deps += vendor_sdk_requirements(pipeline)
    return (
        "# /// script\n"
        '# requires-python = ">=3.11"\n'
        f"# dependencies = {json.dumps(deps)}\n"
        "# ///\n"
        f'"""The compiled skill {identity.name}: run one driver command.\n'
        "\n"
        f"    rote skill run {identity.name} <command> ...\n"
        "\n"
        f"Generated by {GENERATED_BY}; do not edit (the manifest check refuses it).\n"
        '"""\n'
        "\n"
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "sys.path.insert(0, str(Path(__file__).resolve().parent))\n"
        "\n"
        "from _rote_skill import main  # noqa: E402 (after the path it imports from)\n"
        "\n"
        'if __name__ == "__main__":\n'
        f"    sys.exit(main(sys.argv[1:], skill={py_str_literal(identity.name)}))\n"
    )


# ───────── the adapter ─────────


def emitted_skill(output_dir: str | Path) -> dict[str, Any] | None:
    """What a skill emit recorded in ``output_dir``'s manifest (name, judges, plan), or None."""
    manifest = Path(output_dir) / MANIFEST_NAME
    if not manifest.is_file():
        return None
    record = json.loads(manifest.read_text(encoding="utf-8")).get(MANIFEST_KEY)
    return record if isinstance(record, dict) else None


class SkillAdapter:
    """Emit a compiled skill from a :class:`rote.ir.Pipeline`."""

    def __init__(self, config: SkillAdapterConfig | None = None) -> None:
        self.config = config or SkillAdapterConfig()
        if self.config.judges not in JUDGE_MODES:
            raise ValueError(
                f"judges must be one of {', '.join(JUDGE_MODES)}, got {self.config.judges!r}"
            )

    def _plan(self, pipeline: Pipeline) -> dict[str, Any]:
        from rote.skill_target._driver import unrunnable_step

        cfg = self.config
        try:
            plan = build_plan(pipeline, mcp="live", backend=cfg.external_backend, judges=cfg.judges)
        except PlanError as e:
            raise PlanError(f"the skill runtime cannot run this pipeline: {e}") from e
        problem = unrunnable_step(plan)
        if problem:
            raise PlanError(
                f"the skill runtime cannot run this pipeline: {problem} Compile it for a "
                "workflow runtime (--runtime dbos)"
            )
        if cfg.judges == "helper":
            from rote.inference import structured_output_schema

            for node in spec_judges(pipeline):
                assert node.signature_spec is not None
                if node.signature_spec.client == "anthropic":
                    # The helper's request schema, refused here the way every
                    # Python runtime refuses it at emit. The helper raises a
                    # plain RuntimeError (it must pickle); an emit refusal is
                    # a ValueError, which `rote emit` reports as one.
                    try:
                        structured_output_schema(node.signature_spec.output_schema, node_id=node.id)
                    except RuntimeError as e:
                        raise ValueError(f"{e} (--judges helper)") from e
        return plan

    def _stubs(self, pipeline: Pipeline, scripts: EmitWriter) -> list[str]:
        """The ``extracted/`` modules this emit would leave as stubs, as ``write_extracted_package``
        would write them (the agent's module, a user edit the writer keeps, or a stub)."""
        cfg = self.config
        layout = _extracted_layout(pipeline)
        found = []
        for module, nodes in sorted(layout.items()):
            fresh = resolve_extracted_source(cfg.extracted_source_dir, module) or (
                emit_extracted_module(
                    module, nodes, generated_by=GENERATED_BY, caller_note=_CALLER_NOTE
                )
            )
            text = scripts.landed_content("extracted", f"{module}.py", content=fresh)
            backed = frozenset(
                _impl_path_parts(n.impl)[1]
                for n in nodes
                if n.impl is not None and mcp_backed(n, "skill", cfg.external_backend)
            )
            if _is_stub_module(Path(f"{module}.py"), text, backed):
                found.append(f"scripts/extracted/{module}.py")
        for module in helper_module_names(cfg.extracted_source_dir, set(layout)):
            fresh = resolve_extracted_source(cfg.extracted_source_dir, module) or ""
            text = scripts.landed_content("extracted", f"{module}.py", content=fresh)
            if _is_stub_module(Path(f"{module}.py"), text):
                found.append(f"scripts/extracted/{module}.py")
        return found

    def emit(self, pipeline: Pipeline, output_dir: str | Path) -> dict[str, Path]:
        cfg = self.config
        # Everything that can refuse runs before the first write.
        identity = identity_for(
            pipeline, skill_dir=cfg.skill_dir, source_dir=cfg.extracted_source_dir
        )
        plan = self._plan(pipeline)

        writer = EmitWriter(output_dir)
        scripts = writer.scoped("scripts")
        require_constant_parameters(pipeline, scripts, cfg.extracted_source_dir)
        stubs = [] if cfg.allow_stubs else self._stubs(pipeline, scripts)
        if stubs:
            raise ValueError(
                f"{len(stubs)} extracted/ module(s) would still be stubs, and a compiled skill "
                f"stops on its first call to one: {', '.join(stubs)}. Implement them beside "
                "pipeline.yaml (or rote compile again), then re-emit. If no run reaches them, "
                "pass --allow-stubs"
            )

        from rote.skill_target import HELPER_FILES, script_files

        written: dict[str, Path] = {
            "SKILL": writer.write(SKILL_FILE, content=skill_md(identity, pipeline, plan)),
            "README": writer.write("README.md", content=readme(identity, pipeline, plan)),
            "run": scripts.write("run.py", content=run_py(identity, pipeline, plan)),
        }
        shipped = script_files(plan)
        for name, text in shipped.items():
            written[name.removesuffix(".py").removesuffix(".json")] = scripts.write(
                name, content=text
            )
        for name in HELPER_FILES.keys() - shipped.keys():
            # A helper an earlier emit shipped and this plan does not need
            # (re-emitted with --judges session, or with no MCP step left):
            # a leftover would be code the skill can import but nothing vouches for.
            scripts.retire(name)
        written.update(
            {
                f"scripts/{label}": path
                for label, path in write_extracted_package(
                    scripts,
                    pipeline,
                    source_dir=cfg.extracted_source_dir,
                    generated_by=GENERATED_BY,
                    caller_note=_CALLER_NOTE,
                ).items()
            }
        )
        for module, nodes in _extracted_layout(pipeline).items():
            backed = [
                _impl_path_parts(n.impl)[1]
                for n in nodes
                if n.impl is not None and mcp_backed(n, "skill", cfg.external_backend)
            ]
            if backed:
                scripts.record_mcp_backed("extracted", f"{module}.py", functions=backed)
        writer.record(
            MANIFEST_KEY,
            {
                "name": identity.name,
                "pipeline": pipeline.name,
                "judges": cfg.judges,
                "backend": cfg.external_backend,
                "plan_sha256": plan_hash(plan_text(plan)),
            },
        )
        writer.finalize()
        return written
