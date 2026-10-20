"""Ground a compile's MCP tool arguments in the tools' own input schemas.

A skill says "search Gmail for label:dock-activity"; it never says the
tool's parameter is called ``query``. A compile that guesses
``search_query`` passes every check rote had, and a real server refuses
the call. Measured on main: ops-report's compile did exactly that, and its
fidelity fell to 0.44 once the fixtures refused it the way a server would.

The schema is data the server publishes, so rote, not the agent, fetches
it:

* :func:`discover` lists the tools of every registered MCP server it can
  reach (``tools/list``, through the same non-interactive client emitted
  pipelines use; read-only, never a tool call). Those schemas are
  *declared*.
* :func:`recorded` infers schemas from a ``rote baseline`` run's recorded
  calls (:func:`rote.probe.infer_tool_schemas`). Those are evidence, not
  declarations: they show the keys someone sent, not the ones the tool
  takes, so they only ever raise open questions.

:func:`brief` hands the schemas to the agent before it starts.
:func:`tool_findings` checks every ``mcp:`` binding against them after it
stops, with the same reading of a schema the bench's fixture servers use
(:mod:`rote.tool_schema`). With no schema for a bound tool, the arguments
the compile chose are listed for review (:func:`tool_questions`).
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rote.contracts import ContractFinding
from rote.ir import Node, Pipeline
from rote.tool_schema import is_json_type, names_problem

#: Per-server cap on one ``tools/list``. A server that does not answer in
#: this long is skipped, not waited on: the compile has not started yet.
LIST_TIMEOUT_SECONDS = 15.0

TOOL_SCHEMAS_FILENAME = "tool-schemas.json"


@dataclass(frozen=True)
class ToolSchema:
    """One tool's input schema, and how rote knows it."""

    server: str
    tool: str
    input_schema: dict[str, Any]
    #: Where it came from, for messages.
    source: str
    #: True from the server's own ``tools/list``; False when inferred from
    #: recorded calls, which cannot say which keys the tool does not take.
    declared: bool

    def as_dict(self) -> dict[str, Any]:
        return {"inputSchema": self.input_schema, "source": self.source, "declared": self.declared}


ToolSchemas = dict[tuple[str, str], ToolSchema]


async def discover(on_skip: Callable[[str, str], None] | None = None) -> ToolSchemas:
    """Every registered server's tool schemas, from a live ``tools/list``.

    A server that is unreachable, needs a login, or lacks the ``mcp``
    extra's client is skipped with a reason through ``on_skip``: grounding
    is best-effort, and the check falls back to an open question.
    """
    from rote.mcp import load_registry

    servers = sorted(load_registry().servers)
    if not servers:
        return {}
    try:
        from rote.mcp._runtime_helper import mcp_client
    except ImportError as e:  # the mcp extra is not installed
        for name in servers:
            if on_skip is not None:
                on_skip(name, f"no MCP client ({e})")
        return {}

    found: ToolSchemas = {}
    for name in servers:
        try:
            async with asyncio.timeout(LIST_TIMEOUT_SECONDS):
                async with mcp_client(name, None) as client:
                    tools = await client.list_tools()
        except Exception as e:  # noqa: BLE001 (any failure skips this server only)
            if on_skip is not None:
                on_skip(name, f"{type(e).__name__}: {e}"[:300])
            continue
        for tool in tools:
            schema = tool.input_schema  # MCP's inputSchema (SDK v2 name)
            if isinstance(schema, dict):
                found[(name, tool.name)] = ToolSchema(
                    name, tool.name, schema, f"tools/list on the registered {name!r}", True
                )
    return found


def recorded(observations: tuple[Any, ...]) -> ToolSchemas:
    """Schemas inferred from a baseline's recorded calls (not declarations)."""
    if not observations:
        return {}
    from rote.probe import infer_tool_schemas

    return {
        (s.server, s.tool): ToolSchema(
            s.server,
            s.tool,
            s.input_schema,
            f"{s.samples} recorded call(s) in a baseline run",
            False,
        )
        for s in infer_tool_schemas(list(observations))
    }


def merge(declared: ToolSchemas, inferred: ToolSchemas) -> ToolSchemas:
    """Declared schemas win; inferred ones fill the tools nothing declared."""
    return {**inferred, **declared}


def materialize(ctx_dir: Path, tools: ToolSchemas) -> Path:
    """Write ``tool-schemas.json`` (``{server: {tool: {...}}}``) for the agent."""
    ctx_dir.mkdir(parents=True, exist_ok=True)
    doc: dict[str, dict[str, Any]] = {}
    for (server, tool), schema in sorted(tools.items()):
        doc.setdefault(server, {})[tool] = schema.as_dict()
    path = ctx_dir / TOOL_SCHEMAS_FILENAME
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return path


def _signature(schema: dict[str, Any]) -> str:
    required = set(schema.get("required") or [])
    params = [
        f"`{name}` ({(prop or {}).get('type', 'any')}{', required' if name in required else ''})"
        for name, prop in (schema.get("properties") or {}).items()
    ]
    return ", ".join(params) or "no arguments"


def brief(tools: ToolSchemas, skill_text: str, path: Path) -> str:
    """The prompt section naming each tool's real parameters.

    Tools the skill names (its text contains the tool's name) are spelled
    out; every other discovered tool is in ``path`` only, so a server with
    fifty tools does not fill the prompt.
    """
    if not tools:
        return ""
    named = [t for t in tools.values() if re.search(rf"\b{re.escape(t.tool)}\b", skill_text)]
    lines = [
        f"- `{t.server}` / `{t.tool}`: {_signature(t.input_schema)}"
        + ("" if t.declared else " (inferred from recorded calls, not declared)")
        for t in sorted(named, key=lambda t: (t.server, t.tool))
    ]
    servers = ", ".join(f"`{s}`" for s in sorted({t.server for t in tools.values()}))
    return (
        "TOOL CONTRACTS (each MCP tool's own input schema, fetched by rote; not open "
        f"to interpretation). Every tool schema is in {path}; the servers are {servers}, "
        "and an `mcp:` binding's `server` is one of these names. An mcp-bound node "
        "sends its `inputs:` keys as the tool's arguments, or, with `mcp.args`, exactly "
        "the keys `mcp.args` names: those keys must be the tool's own parameter names, "
        "every required one present, none the tool does not declare. When a node's "
        "inputs are named differently, map them with `mcp.args` (`{query: "
        "search_text}`).\n" + "\n".join(lines)
    )


# ───────── checks ─────────

_UNKNOWN = object()


def sent_arguments(node: Node) -> dict[str, Any]:
    """``{argument name: value}`` an mcp-bound node sends; the value is known
    for a constant or literal and ``_UNKNOWN`` for one taken from the payload.

    The same resolution every emitter uses
    (:func:`rote.adapters._common.mcp_tool_arguments`).
    """
    from rote.adapters._common import mcp_tool_arguments

    sources = mcp_tool_arguments(node)
    if sources is None:
        return {param: _UNKNOWN for param in node.inputs or {}}
    return {
        arg: value if origin in ("constant", "literal") else _UNKNOWN
        for arg, origin, value in sources
    }


def tool_findings(pipeline: Pipeline, tools: ToolSchemas) -> list[ContractFinding]:
    """Every mcp binding's arguments against its tool's declared schema."""
    findings: list[ContractFinding] = []
    for node in pipeline.nodes:
        if node.mcp is None:
            continue
        schema = tools.get((node.mcp.server, node.mcp.tool))
        if schema is None or not schema.declared:
            continue
        sent = sent_arguments(node)
        problem = names_problem(schema.input_schema, sent)
        if problem is None:
            properties = schema.input_schema.get("properties") or {}
            for name, value in sent.items():
                want = (properties.get(name) or {}).get("type")
                if value is not _UNKNOWN and want is not None and not is_json_type(value, want):
                    problem = f"argument {name!r} must be {want}, got {type(value).__name__}"
                    break
        if problem is None:
            continue
        findings.append(
            ContractFinding(
                node_id=node.id,
                kind="tool_arguments_invalid",
                severity="error",
                message=(
                    f"{node.id} calls {node.mcp.server}.{node.mcp.tool} with "
                    f"{', '.join(f'`{k}`' for k in sent) or 'no arguments'}: {problem}. The "
                    f"tool's schema ({schema.source}) takes "
                    f"{_signature(schema.input_schema)}. Rename the node's `inputs:` keys to "
                    f"the tool's parameter names, or map them with `mcp.args`"
                ),
                path="pipeline.yaml",
            )
        )
    return findings


def tool_questions(pipeline: Pipeline, tools: ToolSchemas) -> list[tuple[str, str]]:
    """``(node id, text)`` for every binding whose arguments nothing declared."""
    questions: list[tuple[str, str]] = []
    for node in pipeline.nodes:
        if node.mcp is None:
            continue
        schema = tools.get((node.mcp.server, node.mcp.tool))
        names = ", ".join(f"`{k}`" for k in sent_arguments(node)) or "no arguments"
        call = f"{node.mcp.server}.{node.mcp.tool}"
        if schema is None:
            questions.append(
                (
                    node.id,
                    f"arguments to {call} ({names}) are guessed: no schema for this tool "
                    f"was available (register the server with `rote mcp add` so rote can "
                    f"read its tools/list)",
                )
            )
        elif not schema.declared:
            seen = set(schema.input_schema.get("properties") or {})
            unseen = sorted(set(sent_arguments(node)) - seen)
            if unseen:
                questions.append(
                    (
                        node.id,
                        f"{call} is sent {', '.join(f'`{k}`' for k in unseen)}, which no "
                        f"recorded call used ({schema.source}); the tool's declared schema "
                        f"was not available to check it",
                    )
                )
    return questions
