"""The run plan: a pipeline as data a stdlib interpreter can execute.

One builder serves two interpreters. :mod:`rote.compiler._prefix_runner`
runs a plan on a sample input at compile time, with recorded MCP results
standing in for the calls (``mcp="recorded"``). A compiled skill's driver
(:mod:`rote.skill_target._driver`) runs the same plan for real, calling MCP
live (``mcp="live"``) and stopping at the steps that need a model or a
person. Both interpreters are dumb on purpose: every semantic decision
lives here, and each one is the emitted step bodies' own:

- a payload is built from ``inputs:`` (``bindings``), an omitted optional
  pipeline-input field is left out, not None;
- constants are merged last into an impl's keyword arguments;
- a ``fan_out`` node calls once per element of the list
  :func:`rote.ir.fan_out_element_param` names;
- an MCP tool's arguments come from :func:`rote.adapters._common
  .mcp_tool_arguments`;
- a node on the ``mcp`` backend calls its binding, on ``api`` its impl
  (:func:`rote.adapters._common.mcp_backed`).

So verification checks the same program a compiled skill runs. A second
copy of these rules (generated Python, say) is how payload semantics drift.

Step kinds, each a JSON object with ``id``, ``node_kind``, ``wave`` and
``bindings`` (``[param, source, node_id, field]`` per input):

- ``call``: ``module``, ``symbol``, ``constants``, ``fan_param``, ``retry``.
- ``mcp``: ``server``, ``tool``, ``url``, ``args``, ``fan_param``,
  ``retry``; recorded plans add ``observed`` and ``probe_shapes``.
- ``judge``: ``prompt``, ``references`` (the dotted paths the prompt
  reads), ``input_schema``, ``output_schema``, ``fan_param``. A live plan
  built with ``judges="helper"`` adds ``lane`` (``client``, ``model``,
  ``base_url``, ``temperature``: what the emitted Python judge passes the
  inference helper) and ``retry``; the driver then calls the helper itself
  instead of asking the agent.
- ``gate``: ``description``, ``message``, ``answer_schema``, ``timeout_s``,
  ``signal``; a recorded plan whose sample answered the gate adds
  ``payload``.
- ``loop``: ``description``, ``termination``, ``max_iterations``,
  ``tools``, ``tool_servers``, ``body`` (each sub-node as a full step), ``output_schema``,
  ``unwrap_key``.
- ``skip`` (recorded plans only): ``reason``. The prefix does not run it.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Literal

from rote.adapters._common import PROMPT_PLACEHOLDER
from rote.ir import (
    CONSTANT_ARGUMENT_KINDS,
    Node,
    NodeKind,
    Pipeline,
    fan_out_element_param,
    parse_input_ref,
)

#: Bumped when a step's fields change meaning; a driver refuses a plan
#: with another format rather than misreading it.
PLAN_FORMAT = "rote-plan/1"

#: The ``{{ dotted.path }}`` placeholder of a judge prompt, built from the one
#: source the emitted ``_interpolate`` and TS ``interpolate`` are built from.
PLACEHOLDER = re.compile(PROMPT_PLACEHOLDER)

McpMode = Literal["recorded", "live"]

#: Who answers a live plan's judges: the agent running the skill
#: (``session``: a DECIDE block) or the inference helper on the IR's model
#: (``helper``). Chosen at emit, recorded in the plan, never chosen at run time.
JudgeMode = Literal["session", "helper"]
JUDGE_MODES: tuple[JudgeMode, ...] = ("session", "helper")


class PlanError(ValueError):
    """The pipeline has a step this plan cannot express for a live run."""


def _module_of(impl: str) -> tuple[str, str]:
    rel, _, symbol = impl.partition(":")
    return rel.removesuffix(".py").replace("/", "."), symbol


def _bindings(node: Node) -> list[list[Any]]:
    bindings: list[list[Any]] = []
    for param, ref in (node.inputs or {}).items():
        parsed = parse_input_ref(ref)
        if parsed.node_id is None:
            source = "input" if parsed.field is None else "input_field"
            bindings.append([param, source, None, parsed.field])
        else:
            bindings.append([param, "node", parsed.node_id, parsed.field])
    return bindings


def _retry(node: Node) -> dict[str, Any] | None:
    if node.retry is None or node.retry.max == 0:
        return None
    return {"attempts": node.retry.max + 1, "backoff": node.retry.backoff}


def judge_references(prompt: str) -> list[str]:
    """The dotted paths ``prompt`` reads, in first-use order, each once."""
    seen: dict[str, None] = {}
    for match in PLACEHOLDER.finditer(prompt):
        seen.setdefault(match.group(1), None)
    return list(seen)


def gate_answer_schema(node: Node) -> dict[str, Any]:
    """The schema a person's answer at ``node`` must fit: its declared keys.

    The answer becomes the gate's result, so its keys are the ones the steps
    after it read: every declared key required, none other allowed. A
    single type name is the answer itself. A gate that declares nothing
    takes any object.
    """
    from rote.adapters._common import ir_type, keyed_output_schema

    evidence = node.output_schema or {}
    if isinstance(node.output, dict) and node.output:
        schema, _, _ = keyed_output_schema(node.output, evidence)
        return schema
    if isinstance(node.output, str) and node.output.strip():
        return evidence or ir_type(node.output)[0]
    return evidence or {"type": "object"}


def _gate_timeout_s(node: Node, pipeline: Pipeline) -> float:
    from rote.adapters._common import _duration_to_seconds

    return _duration_to_seconds(node.timeout or pipeline.config.hitl.default_timeout)


def build_plan(
    pipeline: Pipeline,
    *,
    mcp: McpMode,
    implementations: dict[str, str] | None = None,
    observations: tuple[Any, ...] = (),
    tools: dict[tuple[str, str], Any] | None = None,
    gate_answers: dict[str, dict[str, Any]] | None = None,
    backend: Literal["mcp", "api"] = "mcp",
    judges: JudgeMode = "session",
) -> dict[str, Any]:
    """The plan for ``pipeline``, as JSON-ready data.

    ``mcp="recorded"`` is the compile-time prefix: an MCP step carries the
    baseline's ``observations`` (matched on exact arguments by the runner),
    a node whose ``implementations`` status is not ``implemented`` and an
    impl-only external_call (vendor code, never run in the sandbox) are
    ``skip``, and a gate in ``gate_answers`` carries the sample's answer.
    ``mcp="live"`` is a compiled skill: every node runs, so a step the
    driver cannot run (a legacy ``signature:`` judge, a fan_out it cannot
    resolve, a node with no body on ``backend``) raises :class:`PlanError`.
    ``judges`` says who answers a live plan's judges (:data:`JudgeMode`).
    """
    from rote.adapters._common import (
        DEFAULT_AGENT_MAX_ITERATIONS,
        DEFAULT_ANTHROPIC_MODEL,
        DEFAULT_OPENAI_MODEL,
        agent_loop_output,
        mcp_tool_arguments,
    )

    if judges not in JUDGE_MODES:
        raise ValueError(f"judges must be one of {', '.join(JUDGE_MODES)}, got {judges!r}")

    live = mcp == "live"
    recorded = _recordings(observations, tools) if not live else {}
    answers = gate_answers or {}

    def fields(node: Node, kind: str, fan: str | None) -> dict[str, Any]:
        """What a ``kind`` step needs to run ``node``, beyond its id and bindings."""
        if kind == "gate":
            gate: dict[str, Any] = {
                "signal": node.signal,
                "description": node.description.strip(),
                "message": (node.notify.message_template or "").strip() if node.notify else "",
                "answer_schema": gate_answer_schema(node),
                "timeout_s": _gate_timeout_s(node, pipeline),
            }
            if node.id in answers:
                # The sample says what the person answered: the gate's
                # result, so the nodes after it run on the sample too.
                gate["payload"] = answers[node.id]
            return gate
        if kind == "mcp":
            assert node.mcp is not None
            call: dict[str, Any] = {
                "server": node.mcp.server,
                "tool": node.mcp.tool,
                "url": node.mcp.url,
                "args": mcp_tool_arguments(node),
                "fan_param": fan,
                "retry": _retry(node),
            }
            if not live:
                declared = (tools or {}).get((node.mcp.server, node.mcp.tool))
                call["observed"] = recorded.get((node.mcp.server, node.mcp.tool), [])
                call["probe_shapes"] = probe_shapes(getattr(declared, "output_schema", None))
            return call
        if kind == "call":
            assert node.impl is not None
            module, symbol = _module_of(node.impl)
            return {
                "module": module,
                "symbol": symbol,
                "constants": node.constants,
                "fan_param": fan,
                "retry": _retry(node),
            }
        if kind == "judge":
            spec = node.signature_spec
            assert spec is not None
            judge: dict[str, Any] = {
                "prompt": spec.prompt,
                "references": judge_references(spec.prompt),
                "input_schema": spec.input_schema,
                "output_schema": spec.output_schema,
                "fan_param": fan,
            }
            if live and judges == "helper":
                # What the emitted Python judge passes call_judge: the IR's
                # model or the adapters' default, its endpoint, its
                # temperature (rote.adapters._py_common.emit_signature_module).
                default = (
                    DEFAULT_OPENAI_MODEL if spec.client == "openai" else DEFAULT_ANTHROPIC_MODEL
                )
                judge["lane"] = {
                    "client": spec.client,
                    "model": spec.model or default,
                    "base_url": spec.base_url,
                    "temperature": spec.temperature,
                }
                judge["retry"] = _retry(node)
            return judge
        contract = agent_loop_output(node)
        return {
            "description": node.description.strip(),
            "termination": node.termination.condition if node.termination else None,
            "max_iterations": (
                node.termination.max_iterations
                if node.termination
                else DEFAULT_AGENT_MAX_ITERATIONS
            ),
            "tools": list(node.tools or ()),
            "tool_servers": dict(node.tool_servers or {}),
            "body": [body_step(pipeline.node_by_id(i)) for i in node.loop_body or ()],
            "output_schema": contract.schema,
            "unwrap_key": contract.unwrap_key,
        }

    def body_step(node: Node) -> dict[str, Any]:
        """A loop-body sub-node in full, so the driver never reads the IR. The
        loop calls it with arguments the model builds, so its ``inputs:``
        describe rather than bind, and it does not fan out."""
        kind = _step_kind(node, live, backend, implementations or {}, answers)
        if kind not in ("call", "mcp", "judge"):
            if live:
                raise PlanError(_no_body(node, backend))
            return {"id": node.id, "node_kind": str(node.kind), "kind": "skip"}
        return {
            "id": node.id,
            "node_kind": str(node.kind),
            "kind": kind,
            "description": node.description.strip(),
            "input_schema": node.input_schema,
            **fields(node, kind, None),
        }

    steps: list[dict[str, Any]] = []
    for index, wave in enumerate(pipeline.execution_waves()):
        for node in wave:
            step: dict[str, Any] = {
                "id": node.id,
                "node_kind": str(node.kind),
                "wave": index,
                "bindings": _bindings(node),
            }
            kind = _step_kind(node, live, backend, implementations or {}, answers)
            if kind is None:
                raise PlanError(_no_body(node, backend))
            if kind == "skip":
                steps.append(step | {"kind": "skip", "reason": _skip_reason(node, implementations)})
                continue
            fan: str | None = None
            if node.fan_out and kind in ("mcp", "call", "judge"):
                try:
                    fan = fan_out_element_param(node, pipeline)
                except ValueError as e:
                    if live:
                        raise PlanError(f"{node.id}: fan_out: {e}") from e
                    steps.append(step | {"kind": "skip", "reason": f"fan_out: {e}"})
                    continue
            steps.append(step | {"kind": kind} | fields(node, kind, fan))
    plan: dict[str, Any] = {
        "format": PLAN_FORMAT,
        "name": pipeline.name,
        "version": pipeline.version,
        "required_inputs": list(pipeline.required_input_fields),
        "exit_nodes": list(pipeline.exit_nodes),
        "steps": steps,
    }
    if live:
        plan["judges"] = judges
    return plan


#: What an MCP result may be when the tool declares no output schema: a bare
#: JSON list (Gmail's search_threads) or text (Drive's read_file_content).
UNDECLARED_SHAPES = ("list", "text")
_SHAPE_OF_TYPE = {"array": "list", "string": "text", "object": "object"}


def probe_shapes(output_schema: dict[str, Any] | None) -> list[str]:
    """The result shapes the prefix probes a consumer with: the declared one,
    else both shapes an undeclared result is known to take.

    FastMCP wraps a non-object result as ``{"result": ...}`` and marks it
    ``x-fastmcp-wrap-result``; rote's MCP helper unwraps it, so the declared
    shape is the inner one."""
    schema = output_schema
    if isinstance(schema, dict) and schema.get("x-fastmcp-wrap-result"):
        schema = (schema.get("properties") or {}).get("result")
    kind = schema.get("type") if isinstance(schema, dict) else None
    if isinstance(kind, str) and kind in _SHAPE_OF_TYPE:
        return [_SHAPE_OF_TYPE[kind]]
    return list(UNDECLARED_SHAPES)


def _no_body(node: Node, backend: str) -> str:
    if node.kind is NodeKind.LLM_JUDGE:
        return (
            f"{node.id}: a judge needs a signature_spec (prompt and schemas); the legacy "
            "signature: form names a Python class, which a compiled skill cannot show an agent"
        )
    return f"{node.id}: {node.kind} has nothing to run on the {backend} backend"


def _recordings(
    observations: tuple[Any, ...], tools: dict[tuple[str, str], Any] | None
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """``{(server, tool): [{input, output}]}`` from a baseline's tool calls."""
    from rote.probe import parse_tool_result

    recorded: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for o in observations:
        declared = (tools or {}).get((o.server, o.tool))
        output = parse_tool_result(o.result, getattr(declared, "output_schema", None))
        if output is not None and not o.is_error:
            recorded.setdefault((o.server, o.tool), []).append({"input": o.input, "output": output})
    return recorded


def plan_text(plan: dict[str, Any]) -> str:
    """The plan as the bytes a compiled skill ships (``scripts/plan.json``)."""
    return json.dumps(plan, indent=1, sort_keys=True, ensure_ascii=True, default=str) + "\n"


def plan_hash(text: str) -> str:
    """The identity a run journals: the sha256 of the shipped plan bytes."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _step_kind(
    node: Node,
    live: bool,
    backend: str,
    implementations: dict[str, str],
    answers: dict[str, dict[str, Any]],
) -> str | None:
    """Which step ``node`` becomes; None when a live run has nothing to run."""
    from rote.adapters._common import mcp_backed

    if node.kind is NodeKind.HITL_GATE:
        return "gate" if live or node.id in answers else "skip"
    if node.mcp is not None and (not live or mcp_backed(node, "skill", backend)):
        return "mcp"
    if node.kind in CONSTANT_ARGUMENT_KINDS and node.impl is not None:
        if live:
            return "call"
        # The prefix runs written pure functions only: an external_call's
        # impl is vendor code, and the sandbox has no network.
        ready = implementations.get(node.id) == "implemented"
        return "call" if node.kind is NodeKind.PURE_FUNCTION and ready else "skip"
    if not live:
        return "skip"
    if node.kind is NodeKind.LLM_JUDGE and node.signature_spec is not None:
        return "judge"
    if node.kind is NodeKind.AGENT_LOOP:
        return "loop"
    return None


def _skip_reason(node: Node, implementations: dict[str, str] | None) -> str:
    if node.kind is NodeKind.PURE_FUNCTION:
        return f"pure_function {(implementations or {}).get(node.id, 'without impl')}"
    return str(node.kind)
