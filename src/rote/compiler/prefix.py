"""Run a deliverable's deterministic prefix on the skill's sample input.

Generated tests check each function against cases the agent chose. They
cannot catch a contract the agent invented: a test written by the model
that wrote ``line["expense_id"]`` feeds it rows with ``expense_id``. The
sample input is data the agent did not write, so running the pipeline on
it is the check that can fail.

The prefix is every node that runs without a model or a person, in
execution order: ``pure_function`` nodes with a written implementation,
and MCP-bound nodes whose call a ``rote baseline`` run recorded (the
recorded result stands in for the call, matched on the exact arguments).
A judge, an agent loop, a gate, an unrecorded tool call or a stub ends
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
) -> dict[str, Any]:
    """The JSON plan :mod:`rote.compiler._prefix_runner` executes."""
    from rote.adapters._common import mcp_tool_arguments
    from rote.probe import parse_tool_result

    status = {i.node_id: i.status for i in implementations}
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
            if node.mcp is not None:
                step |= {
                    "kind": "mcp",
                    "args": mcp_tool_arguments(node),
                    "observed": recorded.get((node.mcp.server, node.mcp.tool), []),
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
        if names_problem(schema.input_schema, step["arguments"]) is not None:
            continue  # the static check already names it (tool_findings)
        problem = argument_problem(schema.input_schema, step["arguments"])
        if problem is not None:
            findings.append(
                ContractFinding(
                    node_id=node.id,
                    kind="tool_arguments_invalid",
                    severity="error",
                    message=(
                        f"on the sample input, {node.id} would call "
                        f"{node.mcp.server}.{node.mcp.tool} with {step['arguments']!r}: "
                        f"{problem} (schema: {schema.source})"
                    ),
                    path="pipeline.yaml",
                )
            )
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

    plan = build_plan(pipeline, root, sample, implementations, observations)

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
    findings = findings_from_report(report, sample) + _argument_findings(
        pipeline, report, tools or {}
    )
    return PrefixRun(
        status="failed" if findings else "ran",
        summary=summarize(report),
        steps=report,
        findings=findings,
    )
