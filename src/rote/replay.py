"""A replay case from a recorded run of a stored skill (#335).

``rote baseline`` records one real run of a skill: its input and every MCP
call it made, with the real results. ``rote eval --run --runtime skill``
already runs a skill and its compiled skill side by side against fixture MCP
servers (:mod:`rote.bench.fixture_mcp`) that answer only from a case's
recorded responses, log every call with its arguments, and never forward
one. What was missing is the case: fidelity cases were hand-written into an
example's ``fidelity/cases.yaml``. :func:`prepare` writes one from the
recording, for a person to review before anything replays it:

- the recorded input, and the gate answers the person supplies;
- each recorded read as a fixture that answers that exact call, with the
  result text the real server sent;
- the tools of every server the run called, with their schemas, and
  whether each writes (the server's own ``readOnlyHint``). A write has no
  recorded response (a recording runs read-only), and none is invented:
  the replay logs the write either side proposes and answers it with an
  error, so a run that needed its response ends incomplete, never passes;
- the fields to compare, from the compiled pipeline's exit outputs.

The tool list comes from the skill's bundled ``mcp-tools.json`` when it has
one, else from a read-only ``tools/list`` of each observed server.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

#: The case id a replay dir holds.
CASE_ID = "recorded"


class ReplayError(ValueError):
    """The recording cannot become a replay case, and why."""


@dataclass(frozen=True)
class Prepared:
    """What :func:`prepare` wrote, and what a person should review."""

    out_dir: Path
    reads: int
    #: ``server.tool`` for every tool that writes (no response is recorded).
    writes: tuple[str, ...]
    #: Things the replay cannot answer as recorded, for the person.
    notes: tuple[str, ...] = field(default_factory=tuple)


def _observations(baseline: Path) -> list[dict[str, Any]]:
    from rote.eval.baseline import OBSERVED_TOOLS_FILENAME

    path = baseline / OBSERVED_TOOLS_FILENAME
    if not path.is_file():
        raise ReplayError(f"{path} does not exist: run `rote baseline` on the skill first")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [o for o in raw if isinstance(o, dict)]


def _recorded_input(baseline: Path) -> dict[str, Any]:
    from rote.eval.baseline import METRICS_FILENAME

    metrics = json.loads((baseline / METRICS_FILENAME).read_text(encoding="utf-8"))
    payload = metrics.get("input")
    if not isinstance(payload, dict):
        raise ReplayError(f"{baseline / METRICS_FILENAME} records no input object")
    succeeded = [r for r in metrics.get("runs") or [] if r.get("succeeded")]
    if not succeeded:
        raise ReplayError(
            f"{baseline}: no recorded trial succeeded, so its calls are not a run to replay"
        )
    return payload


def _result_text(result: Any) -> Any:
    """What the fixture answers: the text the real server sent, verbatim when
    it sent text, so both sides read exactly what the recorded run read."""
    if isinstance(result, list):
        blocks = [b for b in result if isinstance(b, dict) and b.get("type") == "text"]
        texts = [b["text"] for b in blocks if isinstance(b.get("text"), str)]
        if blocks and len(texts) == len(blocks):
            return "".join(texts)
    from rote.probe import parse_tool_result

    return parse_tool_result(result)


async def _listed_tools(servers: Sequence[str]) -> dict[str, list[dict[str, Any]]]:
    """``tools/list`` of each server, read-only, as tools/list returns them."""
    from rote.mcp._runtime_helper import mcp_client

    listed: dict[str, list[dict[str, Any]]] = {}
    for server in servers:
        try:
            async with mcp_client(server, None) as client:
                tools = await client.list_tools()
        except Exception as e:  # noqa: BLE001 (every failure means the same thing here)
            raise ReplayError(
                f"cannot list {server}'s tools ({type(e).__name__}: {e}): the replay needs "
                f"each tool's schema. Log in (`rote mcp login {server}`) or bundle the "
                "skill's tools as mcp-tools.json"
            ) from e
        # By alias: the wire names (inputSchema, readOnlyHint), as tools/list
        # sends them and as a bundled mcp-tools.json holds them. Dumped by
        # field name, every tool read as a write and the snapshot was refused.
        listed[server] = [
            t.model_dump(mode="json", by_alias=True, exclude_none=True) for t in tools
        ]
    return listed


def _tool_snapshot(skill_dir: Path, servers: Sequence[str]) -> dict[str, list[dict[str, Any]]]:
    """``{server: [tool, ...]}``: the bundled snapshot when it covers every
    observed server, else a live read-only listing."""
    import asyncio

    bundled_path = skill_dir / "mcp-tools.json"
    if bundled_path.is_file():
        raw = json.loads(bundled_path.read_text(encoding="utf-8")).get("servers") or {}
        if set(servers) <= set(raw):
            return {s: list(raw[s].get("tools") or []) for s in servers}
    return asyncio.run(_listed_tools(servers))


def _writes(server: str, tools: Sequence[Mapping[str, Any]]) -> set[str]:
    """The tools that write: anything the server does not annotate read-only,
    unless the user registered the server read-only."""
    from rote.mcp.registry import declared_read_only

    if declared_read_only(server):
        return set()
    return {
        str(t["name"])
        for t in tools
        if not (isinstance(t.get("annotations"), dict) and t["annotations"].get("readOnlyHint"))
    }


def prepare(
    skill_dir: Path,
    baseline_dir: Path,
    compiled_pipeline: Path,
    out_dir: Path,
    *,
    gates: Sequence[Any] = (),
) -> Prepared:
    """Write ``out_dir/fidelity/cases.yaml`` and ``out_dir/mcp-tools.json``."""
    from rote.eval.baseline import BASELINE_DIRNAME
    from rote.ir import load_pipeline
    from rote.proof import fields_from_ir

    baseline = baseline_dir / BASELINE_DIRNAME
    if not baseline.is_dir():
        baseline = baseline_dir
    payload = _recorded_input(baseline)
    observed = _observations(baseline)
    pipeline = load_pipeline(compiled_pipeline)
    gate_count = sum(1 for n in pipeline.nodes if n.kind.value == "hitl_gate")
    notes: list[str] = []
    if gate_count and len(gates) < gate_count:
        notes.append(
            f"the compiled pipeline has {gate_count} gate(s) and {len(gates)} answer(s) were "
            "given: a gate with no answer stops the replay (pass --gates)"
        )
    servers = sorted({str(o["server"]) for o in observed})
    snapshot = _tool_snapshot(skill_dir, servers)
    fixtures: dict[str, dict[str, list[dict[str, Any]]]] = {}
    seen: dict[tuple[str, str, str], Any] = {}
    for o in observed:
        if o.get("is_error"):
            continue
        server, tool = str(o["server"]), str(o["tool"])
        arguments = o.get("input") if isinstance(o.get("input"), dict) else {}
        key = (server, tool, json.dumps(arguments, sort_keys=True))
        result = _result_text(o.get("result"))
        if key in seen:
            if seen[key] != result:
                notes.append(
                    f"{server}.{tool} was called twice with the same arguments and answered "
                    "differently: the replay answers the first"
                )
            continue
        seen[key] = result
        fixtures.setdefault(server, {}).setdefault(tool, []).append(
            {"when": arguments, "result": result}
        )
    writes: list[str] = []
    servers_spec: dict[str, Any] = {}
    for server in servers:
        tools = snapshot.get(server) or []
        writing = _writes(server, tools)
        writes += [f"{server}.{t}" for t in sorted(writing)]
        servers_spec[server] = {
            "aliases": [],
            "tools": {str(t["name"]): {"side_effect": str(t["name"]) in writing} for t in tools},
        }
    output_fields = {
        f.original.split("/", 1)[1] if "/" in f.original else f.original: (
            f.why or "as the original run returned it"
        )
        for f in fields_from_ir(pipeline)
    }
    if not output_fields:
        raise ReplayError(f"{compiled_pipeline} declares no exit output to compare")
    out = out_dir / "fidelity"
    out.mkdir(parents=True, exist_ok=True)
    (out_dir / "mcp-tools.json").write_text(
        json.dumps({"servers": {s: {"tools": snapshot.get(s) or []} for s in servers}}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    case = {
        "id": CASE_ID,
        "description": f"replay of a recorded run of {skill_dir.name} ({len(seen)} recorded reads)",
        "input": payload,
        "gates": list(gates),
        "fixtures": fixtures,
    }
    document = {
        "tools_dir": "..",
        "output_fields": output_fields,
        "servers": servers_spec,
        "cases": [case],
    }
    header = (
        f"# Replay case for {skill_dir.name}, written by `rote replay prepare` from\n"
        f"# {baseline}. Review it before replaying: every fixture answers one\n"
        "# exact call with what the real server returned. Writes have no response:\n"
        "# the replay logs what each side proposes and answers it with an error.\n"
    )
    (out / "cases.yaml").write_text(
        header + yaml.safe_dump(document, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return Prepared(out_dir=out_dir, reads=len(seen), writes=tuple(writes), notes=tuple(notes))
