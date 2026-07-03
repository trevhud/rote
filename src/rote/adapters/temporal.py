"""Temporal adapter — emits Workflow + Activities Python from a Pipeline IR.

The adapter consumes a :class:`rote.ir.Pipeline` and produces two source
files:

* ``activities.py`` — one ``@activity.defn`` per non-HITL node, each a
  thin wrapper around the corresponding ``extracted/`` function or
  ``signatures/`` class. **No MCP runtime ever appears in this file.**
  External calls are direct API calls (deterministic Python), and
  LLM judge calls are typed signatures. The graduation of an MCP tool
  call into a deterministic activity happens *here*.
* ``workflow.py`` — one ``@workflow.defn`` class with the orchestration:
  topologically-sorted activity calls grouped into parallel waves,
  signal handlers and ``wait_condition`` calls for HITL gates, and
  the workflow class name versioned by a hash of the pipeline so
  regenerated workflows become new types (avoiding determinism errors
  on in-flight workflows).

The adapter is intentionally template-driven and not clever. The interesting
work is the IR design and the topological sort; the code emission itself
is plain string templates.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from pathlib import Path

from rote.adapters._common import _execution_waves, _pipeline_hash, _to_pascal_case
from rote.ir import Node, NodeKind, Pipeline

# ───────── Adapter configuration ─────────


@dataclass(frozen=True)
class TemporalAdapterConfig:
    """Per-emission configuration for the Temporal adapter.

    These knobs are project-specific and should be set by the caller (or
    eventually by an adapter config file in the source skill bundle).
    Defaults match the BDR example so the adapter is usable out of the
    box.
    """

    # Python import paths used by the emitted code.
    types_module: str = "expected.types"
    extracted_module: str = "expected.extracted"
    signatures_module: str = "expected.signatures"

    # Default activity timeouts when the IR doesn't specify one.
    default_activity_timeout: str = "5m"
    default_hitl_timeout: str = "7d"


# ───────── Helpers ─────────


def _impl_path_parts(impl: str) -> tuple[str, str]:
    """Parse 'extracted/foo.py:bar' into ('foo', 'bar')."""
    if ":" not in impl:
        raise ValueError(f"impl path missing ':function_name': {impl!r}")
    file_part, func = impl.split(":", 1)
    # Strip leading directories and .py suffix
    module_name = file_part.rsplit("/", 1)[-1].removesuffix(".py")
    return module_name, func


def _signature_path_parts(signature: str) -> tuple[str, str]:
    """Parse 'signatures/foo.py:Foo' into ('foo', 'Foo')."""
    return _impl_path_parts(signature)


def _activity_timeout(node: Node, default: str) -> str:
    return node.timeout or default


def _retry_policy_args(node: Node) -> str:
    """Render a Temporal RetryPolicy from the node's retry config.

    Returns a string like ``RetryPolicy(maximum_attempts=5, ...)`` or an
    empty string if no retry policy is specified.
    """
    if not node.retry:
        return ""
    parts = [f"maximum_attempts={node.retry.max + 1}"]  # Temporal counts initial + retries
    backoff = node.retry.backoff
    if backoff == "exponential":
        parts.append('backoff_coefficient=2.0')
    elif backoff == "linear":
        parts.append('backoff_coefficient=1.0')
    return f"RetryPolicy({', '.join(parts)})"


# ───────── Activities emission ─────────


def _emit_activity_for_pure_or_external(node: Node, cfg: TemporalAdapterConfig) -> str:
    assert node.impl is not None
    module_name, func_name = _impl_path_parts(node.impl)

    constants_block = ""
    if node.constants:
        constants_block = (
            "    # Constants from the IR (extracted from the source skill prose):\n"
            + "".join(f"    #   {k} = {v!r}\n" for k, v in node.constants.items())
        )

    mandatory_marker = ""
    if node.mandatory:
        mandatory_marker = (
            "    # MANDATORY: this node was marked mandatory in the source\n"
            "    # skill. The workflow always calls it; do not make it conditional.\n"
        )

    return textwrap.dedent(
        f'''
        @activity.defn(name="{node.id}")
        async def {node.id}(payload: dict) -> dict:
            """{node.description.strip().splitlines()[0]}

            Graduated from MCP tool call → deterministic API call. See
            ``{cfg.extracted_module}.{module_name}`` for implementation.
            """
        '''
    ).lstrip("\n") + (
        mandatory_marker
        + constants_block
        + f"    from {cfg.extracted_module}.{module_name} import {func_name}\n"
        + f"    result = await {func_name}(**payload)\n"
        + "    return _serialize(result)\n"
    )


def _emit_activity_for_llm_judge(node: Node, cfg: TemporalAdapterConfig) -> str:
    assert node.signature is not None
    module_name, class_name = _signature_path_parts(node.signature)
    input_class = f"{class_name}Input"

    return textwrap.dedent(
        f'''
        @activity.defn(name="{node.id}")
        async def {node.id}(payload: dict) -> dict:
            """{node.description.strip().splitlines()[0]}

            LLM judge — typed input/output, bounded decision space. The
            non-determinism lives inside this activity, not the workflow.
            """
        '''
    ).lstrip("\n") + (
        f"    from {cfg.signatures_module}.{module_name} import (\n"
        f"        {class_name},\n"
        f"        {input_class},\n"
        f"    )\n"
        f"    judge = {class_name}()\n"
        f"    result = await judge.forward({input_class}(**payload))\n"
        f"    return result.model_dump()\n"
    )


def _emit_activity_for_agent_loop(node: Node, cfg: TemporalAdapterConfig) -> str:
    """Emit a stub activity for an agent_loop node.

    Agent loops require an LLM agent runtime (e.g., the Anthropic SDK
    with bounded iterations). The v0 adapter emits a stub that raises
    NotImplementedError so the workflow at least imports cleanly. Real
    implementations are a per-project decision and should call out to
    whatever agent harness the project already uses.
    """
    tools_doc = "\n".join(f"    #   - {t}" for t in (node.tools or []))
    sub_nodes_doc = ""
    if node.loop_body:
        sub_nodes_doc = (
            "    # Loop body sub-nodes (call these for each iteration):\n"
            + "\n".join(f"    #   - {sn}" for sn in node.loop_body)
            + "\n"
        )

    return textwrap.dedent(
        f'''
        @activity.defn(name="{node.id}")
        async def {node.id}(payload: dict) -> dict:
            """{node.description.strip().splitlines()[0]}

            STUB — agent loops require an LLM agent runtime. Implement
            this against the project's preferred agent harness (Anthropic
            Agent SDK, OpenAI Agents SDK, LangGraph, etc.).
            """
            # Tools the agent should be allowed to call inside the loop:
        '''
    ).lstrip("\n") + (tools_doc + "\n" if tools_doc else "") + sub_nodes_doc + (
        f'    raise NotImplementedError("agent_loop activity {node.id!r}: requires an agent runtime")\n'
    )


def emit_activities(pipeline: Pipeline, cfg: TemporalAdapterConfig | None = None) -> str:
    """Render the activities.py source for a pipeline."""
    cfg = cfg or TemporalAdapterConfig()

    parts: list[str] = []

    parts.append(
        textwrap.dedent(
            f'''
            """Auto-generated by rote.adapters.temporal.

            Pipeline: {pipeline.name} v{pipeline.version}
            Source skill: {pipeline.source_skill or "unknown"}

            DO NOT EDIT BY HAND. Re-run ``rote emit`` to regenerate.

            Architecture note: every external_call activity in this file
            wraps a deterministic Python function from the ``extracted/``
            package. None of them call MCP tools at runtime — the MCP
            tool calls from the source skill have been graduated into
            direct API calls during the rote emission step.
            """

            from __future__ import annotations

            from typing import Any

            from temporalio import activity
            '''
        ).lstrip("\n")
    )

    parts.append(
        textwrap.dedent(
            '''

            def _serialize(obj: Any) -> Any:
                """Convert pydantic models / dataclasses / lists to plain dicts.

                Activities return JSON-serializable payloads so the workflow
                doesn't carry typed objects through Temporal's history.
                """
                if hasattr(obj, "model_dump"):
                    return obj.model_dump()
                if isinstance(obj, list):
                    return [_serialize(x) for x in obj]
                if isinstance(obj, tuple):
                    return [_serialize(x) for x in obj]
                if isinstance(obj, dict):
                    return {k: _serialize(v) for k, v in obj.items()}
                return obj
            '''
        )
    )

    # Emit one activity per non-HITL node (including loop_body sub-nodes
    # so they can be tested in isolation).
    for node in pipeline.nodes:
        if node.kind is NodeKind.HITL_GATE:
            continue
        parts.append("\n")
        if node.kind in (NodeKind.PURE_FUNCTION, NodeKind.EXTERNAL_CALL):
            parts.append(_emit_activity_for_pure_or_external(node, cfg))
        elif node.kind is NodeKind.LLM_JUDGE:
            parts.append(_emit_activity_for_llm_judge(node, cfg))
        elif node.kind is NodeKind.AGENT_LOOP:
            parts.append(_emit_activity_for_agent_loop(node, cfg))

    return "".join(parts)


# ───────── Workflow emission ─────────


def _emit_signal_handlers(pipeline: Pipeline) -> str:
    """Emit __init__ + signal handlers for every HITL gate.

    The output is indented for class-body inclusion (4-space prefix on
    every line). The caller concatenates this directly under the class
    header.
    """
    gates = [n for n in pipeline.nodes if n.kind is NodeKind.HITL_GATE]
    if not gates:
        return (
            "    def __init__(self) -> None:\n"
            "        pass\n\n"
        )

    init_lines = ["    def __init__(self) -> None:"]
    for gate in gates:
        init_lines.append(f"        self._{gate.signal}_payload: dict | None = None")
    init_block = "\n".join(init_lines) + "\n\n"

    handler_blocks: list[str] = []
    for gate in gates:
        # First describe at module-level indent, then re-indent everything
        # by 4 spaces so the handlers land inside the class body.
        raw = textwrap.dedent(
            f'''
            @workflow.signal(name="{gate.signal}")
            def {gate.signal}(self, payload: dict) -> None:
                """{gate.description.strip().splitlines()[0]}"""
                self._{gate.signal}_payload = payload
            '''
        ).lstrip("\n")
        handler_blocks.append(textwrap.indent(raw, prefix="    "))
    return init_block + "\n".join(handler_blocks)


def _emit_wave_call(node: Node, cfg: TemporalAdapterConfig) -> str:
    """Emit a multi-line call to ``workflow.execute_activity`` for one node.

    Produces something like::

        foo_result = await workflow.execute_activity(
            "foo",
            {},
            start_to_close_timeout=timedelta(minutes=...),
            retry_policy=RetryPolicy(...),
        )
    """
    if node.kind is NodeKind.HITL_GATE:
        # HITL gates are not activities — they're handled separately.
        return ""

    timeout = _activity_timeout(node, cfg.default_activity_timeout)
    retry = _retry_policy_args(node)

    lines = [
        f'        {node.id}_result = await workflow.execute_activity(',
        f'            "{node.id}",',
        '            {},  # TODO: pass real payload from upstream nodes',
        f'            start_to_close_timeout=timedelta(minutes=_parse_minutes("{timeout}")),',
    ]
    if retry:
        lines.append(f'            retry_policy={retry},')
    lines.append('        )')
    return "\n".join(lines) + "\n"


def _emit_hitl_block(node: Node) -> str:
    assert node.kind is NodeKind.HITL_GATE
    assert node.signal is not None
    return (
        f'        # ─── HITL gate: {node.id} ───\n'
        f'        # Workflow suspends here until the {node.signal!r} signal\n'
        f'        # arrives. Survives worker restarts; resumes immediately\n'
        f'        # when the signal fires.\n'
        f'        await workflow.wait_condition(\n'
        f'            lambda: self._{node.signal}_payload is not None\n'
        f'        )\n'
        f'        {node.id}_result = self._{node.signal}_payload\n'
    )


def _emit_workflow_run(pipeline: Pipeline, cfg: TemporalAdapterConfig) -> str:
    waves = _execution_waves(pipeline)

    body_lines: list[str] = ['    @workflow.run', '    async def run(self, brief: dict) -> dict:']
    body_lines.append(
        f'        """{pipeline.description.strip().splitlines()[0] if pipeline.description else pipeline.name}"""'
    )

    for wave_idx, wave in enumerate(waves, start=1):
        non_hitl = [n for n in wave if n.kind is not NodeKind.HITL_GATE]
        hitl = [n for n in wave if n.kind is NodeKind.HITL_GATE]

        body_lines.append("")
        body_lines.append(f"        # ─── Wave {wave_idx} ───")

        if len(non_hitl) == 1:
            body_lines.append(_emit_wave_call(non_hitl[0], cfg).rstrip("\n"))
        elif len(non_hitl) > 1:
            # Parallel via asyncio.gather
            body_lines.append("        (")
            for n in non_hitl:
                body_lines.append(f"            {n.id}_result,")
            body_lines.append("        ) = await asyncio.gather(")
            for n in non_hitl:
                timeout = _activity_timeout(n, cfg.default_activity_timeout)
                body_lines.append(
                    f'            workflow.execute_activity(\n'
                    f'                "{n.id}",\n'
                    f'                {{}},\n'
                    f'                start_to_close_timeout=timedelta(minutes=_parse_minutes("{timeout}")),\n'
                    f'            ),'
                )
            body_lines.append("        )")

        for h in hitl:
            body_lines.append(_emit_hitl_block(h).rstrip("\n"))

    body_lines.append("")
    body_lines.append("        return {")
    for exit_id in pipeline.exit_nodes:
        body_lines.append(f'            "{exit_id}": {exit_id}_result,')
    body_lines.append("        }")

    return "\n".join(body_lines) + "\n"


def emit_workflow(pipeline: Pipeline, cfg: TemporalAdapterConfig | None = None) -> str:
    """Render the workflow.py source for a pipeline."""
    cfg = cfg or TemporalAdapterConfig()

    pascal_name = _to_pascal_case(pipeline.name)
    workflow_hash = _pipeline_hash(pipeline)
    versioned_workflow_name = f"{pascal_name}_{workflow_hash}"
    class_name = f"{pascal_name}Workflow"

    parts: list[str] = []

    parts.append(
        textwrap.dedent(
            f'''
            """Auto-generated by rote.adapters.temporal.

            Pipeline: {pipeline.name} v{pipeline.version}
            Source skill: {pipeline.source_skill or "unknown"}
            Pipeline hash: {workflow_hash}

            DO NOT EDIT BY HAND. Re-run ``rote emit`` to regenerate.

            Workflow versioning: the registered workflow type name
            includes a hash of the pipeline so regenerated workflows
            become a new type. In-flight workflows on the old type
            continue running the old code; new workflows use the new
            code. This is the simplest way to avoid Temporal's
            determinism errors when regenerating from a changed skill.
            """

            from __future__ import annotations

            import asyncio
            from datetime import timedelta

            from temporalio import workflow
            from temporalio.common import RetryPolicy


            def _parse_minutes(s: str) -> float:
                """Parse a duration string like '5m', '30s', '7d' into minutes."""
                s = s.strip()
                if s.endswith("ms"):
                    return float(s[:-2]) / 60_000
                if s.endswith("s"):
                    return float(s[:-1]) / 60
                if s.endswith("m"):
                    return float(s[:-1])
                if s.endswith("h"):
                    return float(s[:-1]) * 60
                if s.endswith("d"):
                    return float(s[:-1]) * 60 * 24
                return float(s)


            @workflow.defn(name="{versioned_workflow_name}")
            class {class_name}:
                """Graduated workflow for {pipeline.name}."""

            '''
        ).lstrip("\n")
    )

    parts.append(_emit_signal_handlers(pipeline))
    parts.append("\n")
    parts.append(_emit_workflow_run(pipeline, cfg))

    # Re-indent the workflow run + signal handlers under the class
    # (they were emitted at module level above for readability).
    return "".join(parts)


# ───────── TemporalAdapter facade ─────────


class TemporalAdapter:
    """Facade that emits a Temporal workflow + activities pair from a Pipeline."""

    def __init__(self, config: TemporalAdapterConfig | None = None) -> None:
        self.config = config or TemporalAdapterConfig()

    def emit_activities(self, pipeline: Pipeline) -> str:
        return emit_activities(pipeline, self.config)

    def emit_workflow(self, pipeline: Pipeline) -> str:
        return emit_workflow(pipeline, self.config)

    def emit(self, pipeline: Pipeline, output_dir: str | Path) -> dict[str, Path]:
        """Write activities.py + workflow.py + __init__.py into output_dir."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        activities_path = out / "activities.py"
        workflow_path = out / "workflow.py"
        init_path = out / "__init__.py"

        activities_path.write_text(self.emit_activities(pipeline), encoding="utf-8")
        workflow_path.write_text(self.emit_workflow(pipeline), encoding="utf-8")
        if not init_path.exists():
            init_path.write_text(
                f'"""Emitted Temporal artifacts for {pipeline.name}."""\n',
                encoding="utf-8",
            )

        return {
            "activities": activities_path,
            "workflow": workflow_path,
            "__init__": init_path,
        }
