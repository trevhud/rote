"""Run a deliverable's deterministic prefix on the skill's sample input.

Generated tests check each function against cases the agent chose. They
cannot catch a contract the agent invented: a test written by the model
that wrote ``line["expense_id"]`` feeds it rows with ``expense_id``. The
sample input is data the agent did not write, so running the pipeline on
it is the check that can fail.

The prefix is every node that runs without a model or a person, in
execution order: ``pure_function`` nodes with a written implementation,
and MCP-bound nodes whose call a ``rote baseline`` run recorded (the
recorded result stands in for the call, matched on the exact arguments),
and gates the sample answers (its ``signals`` payload is the gate's result).
A judge, an agent loop, an unanswered gate, an unrecorded tool call or a stub ends
the prefix on its branch: nodes that need its output are reported as not
reached, never guessed at.

:func:`build_plan` turns the IR into a plan that mirrors the emitted step
bodies (payload from ``inputs:``, an omitted optional field left out,
constants merged last, one call per element for ``fan_out``, results
through ``_serialize``). The plan runs in :mod:`rote.compiler._prefix_runner`,
a stdlib-only script, under the same sandbox and environment as the
generated tests (:func:`rote.compiler.verify.run_confined`). A raised
exception or a binding the upstream result cannot satisfy becomes an
error finding with the exact message and the file and line it came from.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from rote.compiler.grounding import Sample
from rote.contracts import ContractFinding, NodeImplementation
from rote.ir import (
    CONSTANT_ARGUMENT_KINDS,
    NodeKind,
    Pipeline,
    fan_out_element_param,
    parse_input_ref,
)

#: Wall-clock cap on one prefix run. The prefix is pure functions over one
#: sample, milliseconds of work; hitting this is an unbounded loop.
PREFIX_TIMEOUT_SECONDS = 120.0

RUNNER_PATH = Path(__file__).with_name("_prefix_runner.py")

#: The scratch file the runner writes its report to (never stdout, which
#: the agent's code shares and could forge).
REPORT_FILENAME = "prefix-report.json"

PrefixStatus = Literal["ran", "unverified", "failed"]


@dataclass(frozen=True)
class PrefixRun:
    """The verdict of one prefix run."""

    status: PrefixStatus
    summary: str
    #: ``{node id: {"status": ..., ...}}`` from the runner.
    steps: dict[str, dict[str, Any]] = field(default_factory=dict)
    findings: list[ContractFinding] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {"status": self.status, "summary": self.summary, "steps": self.steps}


def _module_of(impl: str) -> tuple[str, str]:
    rel, _, symbol = impl.partition(":")
    return rel.removesuffix(".py").replace("/", "."), symbol


def build_plan(
    pipeline: Pipeline,
    root: Path,
    sample: Sample,
    implementations: list[NodeImplementation],
    observations: tuple[Any, ...] = (),
    tools: dict[tuple[str, str], Any] | None = None,
) -> dict[str, Any]:
    """The JSON plan :mod:`rote.compiler._prefix_runner` executes."""
    from rote.adapters._common import mcp_tool_arguments
    from rote.probe import parse_tool_result

    status = {i.node_id: i.status for i in implementations}
    answers = _gate_answers(pipeline, sample)
    recorded: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for o in observations:
        output = parse_tool_result(o.result)
        if output is not None and not o.is_error:
            recorded.setdefault((o.server, o.tool), []).append({"input": o.input, "output": output})
    steps: list[dict[str, Any]] = []
    for wave in pipeline.execution_waves():
        for node in wave:
            bindings: list[list[Any]] = []
            for param, ref in (node.inputs or {}).items():
                parsed = parse_input_ref(ref)
                if parsed.node_id is None:
                    source = "input" if parsed.field is None else "input_field"
                    bindings.append([param, source, None, parsed.field])
                else:
                    bindings.append([param, "node", parsed.node_id, parsed.field])
            step: dict[str, Any] = {"id": node.id, "bindings": bindings}
            if node.kind is NodeKind.HITL_GATE and node.id in answers:
                # The sample says what the person answered: the gate's result,
                # so the nodes after it run on the sample too.
                step |= {"kind": "gate", "payload": answers[node.id]}
            elif node.mcp is not None:
                declared = (tools or {}).get((node.mcp.server, node.mcp.tool))
                try:
                    # One call per element, as every emitter fans an MCP node.
                    fan = fan_out_element_param(node, pipeline) if node.fan_out else None
                except ValueError as e:
                    step |= {"kind": "skip", "reason": f"fan_out: {e}"}
                    steps.append(step)
                    continue
                step |= {
                    "kind": "mcp",
                    "args": mcp_tool_arguments(node),
                    "fan_param": fan,
                    "observed": recorded.get((node.mcp.server, node.mcp.tool), []),
                    "probe_shapes": probe_shapes(getattr(declared, "output_schema", None)),
                }
            elif (
                node.kind is NodeKind.PURE_FUNCTION
                and node.impl is not None
                and status.get(node.id) == "implemented"
            ):
                module, symbol = _module_of(node.impl)
                try:
                    fan = fan_out_element_param(node, pipeline) if node.fan_out else None
                except ValueError as e:
                    step |= {"kind": "skip", "reason": f"fan_out: {e}"}
                    steps.append(step)
                    continue
                step |= {
                    "kind": "call",
                    "module": module,
                    "symbol": symbol,
                    "constants": node.constants if node.kind in CONSTANT_ARGUMENT_KINDS else None,
                    "fan_param": fan,
                }
            else:
                reason = str(node.kind)
                if node.kind is NodeKind.PURE_FUNCTION:
                    reason = f"pure_function {status.get(node.id, 'without impl')}"
                step |= {"kind": "skip", "reason": reason}
            steps.append(step)
    return {"root": str(root.resolve()), "sample": sample.payload, "steps": steps}


#: What an MCP result may be when the tool declares no output schema: a bare
#: JSON list (Gmail's search_threads) or text (Drive's read_file_content).
UNDECLARED_SHAPES = ("list", "text")
_SHAPE_OF_TYPE = {"array": "list", "string": "text", "object": "object"}


def probe_shapes(output_schema: dict[str, Any] | None) -> list[str]:
    """The result shapes to probe a consumer with: the declared one, else both
    shapes an undeclared result is known to take.

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


def _gate_answers(pipeline: Pipeline, sample: Sample) -> dict[str, dict[str, Any]]:
    """``{gate id: the sample's answer}`` for the gates the sample surely answers.

    Pairing comes from grounding's one matcher, but running past a gate
    takes more than a pairing: the answer must carry the gate's signal, or
    only keys the gate declares. The matcher's last resort pairs a leftover
    answer with the next unmatched gate even when nothing matches (that is
    right for the gate check, which then reports the mismatch). Run on such
    an answer, the nodes after the gate would fail on keys it never had and
    blame correct code.
    """
    if not sample.gates:
        return {}
    from rote.compiler.grounding import GateView, match_sample_gates

    views = {
        n.id: GateView(n.id, n.signal, n.output_keys)
        for wave in pipeline.execution_waves()
        for n in wave
        if n.kind is NodeKind.HITL_GATE
    }
    answers: dict[str, dict[str, Any]] = {}
    for signal, gate_id in match_sample_gates(list(views.values()), sample).items():
        gate, answer = views[gate_id], sample.gates[signal]
        if gate.signal == signal or (answer and set(answer) <= gate.keys):
            answers[gate_id] = answer
    return answers


def _where(at: dict[str, Any] | None) -> str:
    if not at:
        return ""
    code = f" `{at['code']}`" if at.get("code") else ""
    return f" at {at['path']}:{at['line']}{code}"


def findings_from_report(
    report: dict[str, dict[str, Any]], sample: Sample
) -> list[ContractFinding]:
    """Error findings for every node that failed on the sample."""
    findings: list[ContractFinding] = []
    tail = (
        f"rote ran the pipeline's deterministic steps on the skill's sample input "
        f"({sample.source}). The pipeline must run on that input as it is: fix the "
        f"code or the IR, and if a test encodes the same wrong name, fix the test"
    )
    for node_id, step in report.items():
        if step.get("status") == "error":
            at = step.get("at") or {}
            findings.append(
                ContractFinding(
                    node_id=node_id,
                    kind="prefix_run_failed",
                    severity="error",
                    message=(
                        f"{node_id} raised {step.get('type')}: {step.get('message')}"
                        f"{_where(at)}. {tail}"
                    ),
                    path=at.get("path"),
                )
            )
        elif step.get("status") == "shape_assumption":
            at = step.get("at") or {}
            upstream = ", ".join(step.get("upstream") or [])
            findings.append(
                ContractFinding(
                    node_id=node_id,
                    kind="prefix_shape_assumption",
                    severity="error",
                    message=(
                        f"{node_id} crashes when {upstream} returns "
                        f"{step.get('shape', 'a JSON list')}: "
                        f"{step.get('type')}: {step.get('message')}{_where(at)}. The tool "
                        f"declares no result shape, and an MCP tool may return a bare list "
                        f"(Gmail's search_threads does) or text (Drive's read_file_content "
                        f"returns CSV), so accept a dict, a list or a string, and test each"
                    ),
                    path=at.get("path"),
                )
            )
        elif step.get("status") == "not_json":
            findings.append(
                ContractFinding(
                    node_id=node_id,
                    kind="prefix_not_json",
                    severity="error",
                    message=(
                        f"{node_id} returned a value that is not plain JSON: "
                        f"{step.get('message')}. A step result crosses a durable boundary "
                        f"as JSON on every runtime, so return lists, dicts, strings, numbers, "
                        f"booleans and None only (a set becomes a sorted list). {tail}"
                    ),
                )
            )
        elif step.get("status") == "binding_error":
            findings.append(
                ContractFinding(
                    node_id=node_id,
                    kind="prefix_binding_failed",
                    severity="error",
                    message=f"{node_id}: {step.get('message')}. {tail}",
                    path="pipeline.yaml",
                )
            )
    return findings


def _argument_findings(
    pipeline: Pipeline, report: dict[str, dict[str, Any]], tools: dict[tuple[str, str], Any]
) -> list[ContractFinding]:
    """The values an MCP node would send on the sample, against the declared schema."""
    from rote.tool_schema import argument_problem, names_problem

    findings: list[ContractFinding] = []
    for node in pipeline.nodes:
        step = report.get(node.id) or {}
        if node.mcp is None or "arguments" not in step:
            continue
        schema = tools.get((node.mcp.server, node.mcp.tool))
        if schema is None or not schema.declared:
            continue
        # A fan_out node calls the tool once per element: each call is checked.
        calls = step["arguments"] if step.get("fan") else [step["arguments"]]
        for i, arguments in enumerate(calls):
            if names_problem(schema.input_schema, arguments) is not None:
                break  # the static check already names it (tool_findings)
            problem = argument_problem(schema.input_schema, arguments)
            if problem is None:
                continue
            where = f"{node.id}[{i}]" if step.get("fan") else node.id
            findings.append(
                ContractFinding(
                    node_id=node.id,
                    kind="tool_arguments_invalid",
                    severity="error",
                    message=(
                        f"on the sample input, {where} would call "
                        f"{node.mcp.server}.{node.mcp.tool} with {arguments!r}: "
                        f"{problem} (schema: {schema.source})"
                    ),
                    path="pipeline.yaml",
                )
            )
            break
    return findings


def output_findings(
    pipeline: Pipeline, report: dict[str, dict[str, Any]], sample: Sample
) -> list[ContractFinding]:
    """Each result on the sample, against its node's declared ``output_schema``.

    The schema is the contract the node's consumers were written against
    (System-1 fixes it before any module is written), so a module that
    returns another shape fails here, on data no model wrote, instead of in
    the next node. A fan_out node's result is one value per element, each
    checked. A node without an ``output_schema`` declares nothing to check.
    """
    from rote.compiler.grounding import schema_mismatches

    findings: list[ContractFinding] = []
    for node in pipeline.nodes:
        step = report.get(node.id) or {}
        if not node.output_schema or step.get("status") != "ok" or "output" not in step:
            continue
        values = step["output"] if node.fan_out else [step["output"]]
        if not isinstance(values, list):
            continue
        for i, value in enumerate(values):
            where = f"{node.id}[{i}]" if node.fan_out else node.id
            problems = schema_mismatches(node.output_schema, value, where)
            if problems:
                findings.append(
                    ContractFinding(
                        node_id=node.id,
                        kind="output_schema_mismatch",
                        severity="error",
                        message=(
                            f"on the sample input ({sample.source}), {node.id} returned a "
                            f"value its output_schema does not allow: {'; '.join(problems[:5])}. "
                            "Its consumers were written against that schema: return what it "
                            "says"
                        ),
                        path=node.impl.partition(":")[0] if node.impl else "pipeline.yaml",
                    )
                )
                break  # one element names the problem; the rest repeat it
    return findings


def summarize(report: dict[str, dict[str, Any]]) -> str:
    counts: dict[str, int] = {}
    for step in report.values():
        counts[step["status"]] = counts.get(step["status"], 0) + 1
    ran = counts.get("ok", 0)
    failed = counts.get("error", 0) + counts.get("binding_error", 0)
    rest = ", ".join(
        f"{n} {s.replace('_', ' ')}"
        for s, n in sorted(counts.items())
        if s not in ("ok", "error", "binding_error")
    )
    return (
        f"{ran} of {len(report)} node(s) ran on the sample input"
        + (f", {failed} failed" if failed else "")
        + (f" ({rest})" if rest else "")
    )


def run_prefix(
    pipeline: Pipeline,
    root: Path,
    sample: Sample,
    implementations: list[NodeImplementation],
    *,
    observations: tuple[Any, ...] = (),
    trust_skill: bool = False,
    tools: dict[tuple[str, str], Any] | None = None,
) -> PrefixRun:
    """Build the plan, run it confined, and turn the report into findings.

    ``tools`` are the MCP tools' schemas (:mod:`rote.compiler.tool_grounding`).
    An MCP node the prefix reaches reports the arguments it would send, and
    their values are checked against the tool's declared schema here: the
    static check can only see the names of arguments taken from the payload.
    """
    from rote.compiler.verify import run_confined

    plan = build_plan(pipeline, root, sample, implementations, observations, tools)

    def _args(scratch: Path) -> list[str]:
        runner = scratch / RUNNER_PATH.name
        runner.write_text(RUNNER_PATH.read_text(encoding="utf-8"), encoding="utf-8")
        plan_path = scratch / "prefix-plan.json"
        plan_path.write_text(json.dumps(plan, default=str), encoding="utf-8")
        return [str(runner), str(plan_path), str(scratch / REPORT_FILENAME)]

    run = run_confined(
        root,
        _args,
        trust_skill=trust_skill,
        timeout=PREFIX_TIMEOUT_SECONDS,
        read_back=(REPORT_FILENAME,),
    )
    if run.unconfined_reason is not None:
        return PrefixRun(
            status="unverified",
            summary=(
                f"deterministic prefix NOT run on the sample: no OS sandbox "
                f"({run.unconfined_reason}); pass --trust-skill to run it without one"
            ),
        )
    if run.timed_out:
        finding = ContractFinding(
            node_id="(prefix)",
            kind="prefix_timeout",
            severity="error",
            message=(
                f"the deterministic steps did not finish on the sample input within "
                f"{PREFIX_TIMEOUT_SECONDS:.0f}s; look for an unbounded loop"
            ),
        )
        return PrefixRun(status="failed", summary="timed out", findings=[finding])
    written = run.files.get(REPORT_FILENAME)
    if not written:
        detail = (run.stderr or run.stdout).strip()[-1500:]
        finding = ContractFinding(
            node_id="(prefix)",
            kind="prefix_crashed",
            severity="error",
            message=(
                f"running the deterministic steps on the sample input crashed before "
                f"reporting (exit {run.returncode}): {detail}"
            ),
        )
        return PrefixRun(status="failed", summary="crashed", findings=[finding])
    report: dict[str, dict[str, Any]] = json.loads(written)
    findings = (
        findings_from_report(report, sample)
        + _argument_findings(pipeline, report, tools or {})
        + output_findings(pipeline, report, sample)
    )
    for step in report.values():
        step.pop("output", None)  # checked; the verdict carries no payloads
    return PrefixRun(
        status="failed" if findings else "ran",
        summary=summarize(report),
        steps=report,
        findings=findings,
    )
