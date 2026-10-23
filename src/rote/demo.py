"""The demo skill that ``rote try`` and ``rote setup`` run: docs-digest.

docs-digest searches Microsoft Learn and reads a repository's DeepWiki
outline, two public MCP servers that need no login, so a first run can
show a real skill becoming a real pipeline with nothing to configure but
two ``rote mcp add`` lines. It ships in the wheel the same way as the
``rote example`` bundles (one home, ``examples/docs-digest/``, listed in
:data:`rote.examples.BUNDLED` and force-included by ``pyproject.toml``).

Next to the skill and its compiled pipeline sits ``reference.json``: one
recorded run of each side on the skill's own ``input.json``. A fresh
pipeline run is compared against it field by field. The format
(:data:`REFERENCE_FORMAT`) is::

    {
      "format": "rote-demo-reference/1",
      "skill": "docs-digest",
      "recorded_on": "YYYY-MM-DD",
      "rote_version": "...", "claude_cli_version": "...",
      "input": {...},                      # the payload both sides ran on
      "servers": {name: {"url", "tool"}},  # what to `rote mcp add --read-only`
      "skill_run": {"model", "wall_seconds", "turns", "cost_usd",
                    "tokens": {"input", "output", "cache_read", "cache_write"},
                    "output": {...}},      # what the agent returned
      "compile": {"runtime", "models", "wall_seconds", "turns", "cost_usd",
                  "tokens": {...}},
      "pipeline_run": {"runtime", "wall_seconds", "model_calls", "cost_usd",
                       "output": {...},    # what the pipeline returned
                       "tool_calls": [     # each MCP step's call, as made
                         {"node", "server", "tool", "arguments", "result"}]},
      "fields": [{"name", "pipeline", "skill", "section", "measure",
                  "expect", "why"}],
      "drift": {...}                       # what changed between recorded runs
    }

Each entry of ``fields`` names one comparable value. ``pipeline`` and
``skill`` are JSON pointers (RFC 6901) into the two ``output`` objects,
since the pipeline returns its exit node's result under the node id.
``section``, when set, narrows a markdown string to the body under the
``## <section>`` heading. ``measure``, when set, reduces the value to a
property of it: ``"headings"`` (the list of ``##`` headings) or
``"bullets"`` (the number of top-level ``- `` bullets). ``expect`` is
``"equal"`` (a fresh run must match the reference) or ``"drifts"`` (live
data: a difference is reported, never a failure). Only values that are
the same on every run of the same input are ``"equal"``; everything a
server can change between runs is ``"drifts"``, with ``why`` saying which
server and how. A reference needs at least one ``"equal"`` field: with
none, every comparison would pass whatever a run returned.

``tool_calls`` are the MCP calls the recorded pipeline run made, each with
the arguments it sent and the value the step received. They are the
recorded-call stubs :func:`rote.compiler.prefix.run_prefix` takes, so the
committed compiled code can be replayed offline against the recording.

Nothing in the file is secret or machine-specific: no paths, hosts other
than the two servers, or account names.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

#: The ``format`` value this module reads. Bump the suffix on any change a
#: reader of the old format would misread.
REFERENCE_FORMAT = "rote-demo-reference/1"

#: The demo's directory name under the bundled examples.
DEMO_NAME = "docs-digest"

Expect = Literal["equal", "drifts"]
Measure = Literal["headings", "bullets"]
_EXPECTS: tuple[Expect, ...] = ("equal", "drifts")
_MEASURES: tuple[Measure, ...] = ("headings", "bullets")


class DemoError(RuntimeError):
    """The bundled demo or its reference is missing or malformed."""


@dataclass(frozen=True)
class Field:
    """One value a fresh run is compared on (see the module docstring)."""

    name: str
    pipeline: str
    skill: str
    section: str | None
    measure: Measure | None
    expect: Expect
    why: str


@dataclass(frozen=True)
class Server:
    """One MCP server the demo needs, as ``rote mcp add <name> <url>`` takes it."""

    name: str
    url: str
    tool: str


@dataclass(frozen=True)
class ToolCall:
    """One MCP call the recorded pipeline run made, and what its step received."""

    node: str
    server: str
    tool: str
    arguments: dict[str, Any]
    result: Any


@dataclass(frozen=True)
class Reference:
    """``reference.json``, checked on load. ``raw`` keeps the whole record."""

    raw: dict[str, Any]
    input: dict[str, Any]
    servers: tuple[Server, ...]
    skill_output: dict[str, Any]
    pipeline_output: dict[str, Any]
    tool_calls: tuple[ToolCall, ...]
    fields: tuple[Field, ...]


def docs_digest_dir() -> Path:
    """``examples/docs-digest/`` in this install: ``skill/``, ``compiled/``,
    ``reference.json`` and a README."""
    from rote.examples import examples_root

    return examples_root() / DEMO_NAME


def skill_dir() -> Path:
    """The demo skill (``SKILL.md``, ``input.json``, ``mcp-tools.json``)."""
    return docs_digest_dir() / "skill"


def pipeline_path() -> Path:
    """The recorded compile's ``pipeline.yaml``; ``rote emit`` reads its
    ``extracted/`` beside it."""
    return docs_digest_dir() / "compiled" / "pipeline.yaml"


def reference_path() -> Path:
    return docs_digest_dir() / "reference.json"


def load_reference(path: Path | None = None) -> Reference:
    """Read and check ``reference.json``.

    Every section is type-checked, every server must be one ``rote mcp
    add`` can take as written (the README and ``rote example`` tell the
    user to run exactly that), with no key in its path, field names are unique, at least one field
    is ``"equal"``, and every field resolves in both recorded outputs, so
    a reference that cannot be compared never loads.
    """
    path = path or reference_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise DemoError(f"{path}: cannot read the demo reference: {e}") from e

    def fail(why: str) -> DemoError:
        return DemoError(f"{path}: {why}")

    if not isinstance(raw, dict) or raw.get("format") != REFERENCE_FORMAT:
        found = raw.get("format") if isinstance(raw, dict) else None
        raise fail(f"expected format {REFERENCE_FORMAT!r}, found {found!r}")
    skill_run = _object(raw, "skill_run", fail)
    pipeline_run = _object(raw, "pipeline_run", fail)
    skill_output = _object(skill_run, "output", fail, "skill_run.output")
    pipeline_output = _object(pipeline_run, "output", fail, "pipeline_run.output")
    servers = tuple(
        _server(name, entry, fail) for name, entry in _object(raw, "servers", fail).items()
    )
    if not servers:
        raise fail("servers is empty: the demo names the MCP servers it needs")
    calls = pipeline_run.get("tool_calls")
    if not isinstance(calls, list) or not all(isinstance(c, dict) for c in calls):
        raise fail("pipeline_run.tool_calls must be a list of objects")
    tool_calls = tuple(_tool_call(c, servers, fail) for c in calls)
    nodes = [c.node for c in tool_calls]
    if len(set(nodes)) != len(nodes):
        raise fail("pipeline_run.tool_calls names a node twice")
    entries = raw.get("fields")
    if not isinstance(entries, list) or not all(isinstance(e, dict) for e in entries):
        raise fail("fields must be a list of objects")
    fields = tuple(_field(entry, i, fail) for i, entry in enumerate(entries))
    names = [f.name for f in fields]
    duplicated = sorted({n for n in names if names.count(n) > 1})
    if duplicated:
        raise fail(f"field names must be unique; repeated: {', '.join(duplicated)}")
    if not any(f.expect == "equal" for f in fields):
        raise fail('no field is "equal": with every field allowed to drift, any run would match')
    for field in fields:
        value_at(pipeline_output, field.pipeline, field.section, field.measure)
        value_at(skill_output, field.skill, field.section, field.measure)
    return Reference(
        raw=raw,
        input=_object(raw, "input", fail),
        servers=servers,
        skill_output=skill_output,
        pipeline_output=pipeline_output,
        tool_calls=tool_calls,
        fields=fields,
    )


def _object(
    parent: Mapping[str, Any], key: str, fail: Any, where: str | None = None
) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise fail(f"{where or key} must be an object, found {type(value).__name__}")
    return value


def _server(name: str, entry: object, fail: Any) -> Server:
    from rote.clients.normalize import CLAUDE_CODE_SYNTAX, url_problem, url_secret_risk
    from rote.ir import _IDENTIFIER_RE, _MCP_TOOL_RE

    if not _IDENTIFIER_RE.fullmatch(name):
        raise fail(f"server name {name!r} is not an identifier")
    if not isinstance(entry, dict):
        raise fail(f"server {name!r} must be an object with url and tool")
    url, tool = entry.get("url"), entry.get("tool")
    if not isinstance(url, str):
        raise fail(f"server {name!r} has no url string")
    problem = url_problem(url, CLAUDE_CODE_SYNTAX) or url_secret_risk(url)
    if problem is None and not url.startswith("https://"):
        problem = "the URL is not https"
    if problem is not None:
        raise fail(f"server {name!r}: {problem}")
    if not isinstance(tool, str) or not _MCP_TOOL_RE.fullmatch(tool):
        raise fail(f"server {name!r}: tool must be a bare MCP tool name, found {tool!r}")
    return Server(name=name, url=url, tool=tool)


def _tool_call(entry: dict[str, Any], servers: tuple[Server, ...], fail: Any) -> ToolCall:
    node, server, tool = entry.get("node"), entry.get("server"), entry.get("tool")
    if not isinstance(node, str) or not node:
        raise fail("each tool call needs its node id")
    known = {s.name: s.tool for s in servers}
    if not isinstance(server, str) or server not in known or tool != known[server]:
        raise fail(
            f"tool call {node!r}: {server}.{tool} is not one of the servers' tools "
            f"({', '.join(f'{k}.{v}' for k, v in known.items())})"
        )
    arguments = entry.get("arguments")
    if not isinstance(arguments, dict):
        raise fail(f"tool call {node!r}: arguments must be an object")
    if "result" not in entry:
        raise fail(f"tool call {node!r} has no result")
    return ToolCall(node, server, known[server], arguments, entry["result"])


def _field(entry: dict[str, Any], index: int, fail: Any) -> Field:
    name = entry.get("name")
    where = f"field {name!r}" if isinstance(name, str) else f"field #{index}"
    if not isinstance(name, str) or not name:
        raise fail(f"{where} needs a name")
    for key in ("pipeline", "skill", "why"):
        if not isinstance(entry.get(key), str):
            raise fail(f"{where}: {key} must be a string")
    section, measure, expect = entry.get("section"), entry.get("measure"), entry.get("expect")
    if section is not None and not isinstance(section, str):
        raise fail(f"{where}: section must be a string")
    if measure is not None and measure not in _MEASURES:
        raise fail(f"{where}: measure must be one of {', '.join(_MEASURES)}")
    if expect not in _EXPECTS:
        raise fail(f"{where}: expect must be equal or drifts")
    return Field(
        name=name,
        pipeline=entry["pipeline"],
        skill=entry["skill"],
        section=section,
        measure=cast("Measure | None", measure),
        expect=cast(Expect, expect),
        why=entry["why"],
    )


def value_at(
    document: Any, pointer: str, section: str | None = None, measure: Measure | None = None
) -> Any:
    """The value at JSON ``pointer`` in ``document``, narrowed to the body
    of markdown heading ``## <section>`` when ``section`` is given, then
    reduced by ``measure``.

    Raises :class:`DemoError` naming what is missing, because a run whose
    output lacks a compared field is a finding, not a blank.
    """
    value = document
    if pointer:
        if not pointer.startswith("/"):
            raise DemoError(f"{pointer!r} is not a JSON pointer")
        for token in pointer[1:].split("/"):
            key = token.replace("~1", "/").replace("~0", "~")
            if isinstance(value, dict) and key in value:
                value = value[key]
            elif isinstance(value, list) and key.isdigit() and int(key) < len(value):
                value = value[int(key)]
            else:
                raise DemoError(f"no value at {pointer!r} (missing {key!r})")
    if section is None and measure is None:
        return value
    if not isinstance(value, str):
        raise DemoError(f"{pointer!r} is not markdown text")
    if section is not None:
        value = markdown_section(value, section)
    if measure == "headings":
        return [line[3:].strip() for line in value.splitlines() if line.startswith("## ")]
    if measure == "bullets":
        return sum(1 for line in value.splitlines() if re.match(r"- \S", line))
    return value


def markdown_section(text: str, heading: str) -> str:
    """The body under ``## <heading>``, up to the next ``#``/``##`` heading,
    with surrounding blank lines removed."""
    lines = text.splitlines()
    marker = f"## {heading}"
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == marker)
    except StopIteration:
        raise DemoError(f"no '{marker}' heading") from None
    body: list[str] = []
    for line in lines[start + 1 :]:
        if line.startswith("# ") or line.startswith("## "):
            break
        body.append(line)
    return "\n".join(body).strip("\n")
