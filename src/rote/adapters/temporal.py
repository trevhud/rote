"""Temporal adapter — emits Workflow + Activities Python from a Pipeline IR.

The adapter consumes a :class:`rote.ir.Pipeline` and produces two source
files:

* ``activities.py`` — one ``@activity.defn`` per non-HITL node, each a
  thin wrapper around the corresponding ``extracted/`` function or
  ``signatures/`` class. **No MCP runtime ever appears in this file.**
  External calls are direct API calls (deterministic Python), and
  LLM judge calls are typed signatures. The compilation of an MCP tool
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

from rote.adapters._common import (
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_OPENAI_MODEL,
    EmitWriter,
    _execution_waves,
    _pipeline_hash,
    _to_pascal_case,
    check_input_refs_available,
    fan_out_nodes,
    refuse_mcp_only_nodes,
    safe_docstring_line,
)
from rote.adapters._py_common import (
    _impl_path_parts,
    _payload_literal,
    _signature_path_parts,
    agent_loop_call,
    fan_out_binding,
    fan_out_list_helper,
    write_signature_package,
)
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

    # Python import paths used by the emitted code. ``signatures_module``
    # only applies to the *legacy* ``signature: path:Class`` form —
    # judges with a ``signature_spec`` import from the generated
    # ``signatures/`` package emitted alongside activities.py.
    types_module: str = "expected.types"
    extracted_module: str = "expected.extracted"
    signatures_module: str = "expected.signatures"

    # Default activity timeouts when the IR doesn't specify one.
    default_activity_timeout: str = "5m"
    default_hitl_timeout: str = "7d"

    # Vendor-default models for generated signature modules when the
    # spec doesn't pin one (same defaults as the DBOS adapter).
    anthropic_default_model: str = DEFAULT_ANTHROPIC_MODEL
    openai_default_model: str = DEFAULT_OPENAI_MODEL


# ───────── Helpers ─────────


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
        parts.append("backoff_coefficient=2.0")
    elif backoff == "linear":
        parts.append("backoff_coefficient=1.0")
    return f"RetryPolicy({', '.join(parts)})"


# ───────── Activities emission ─────────


def _emit_activity_for_pure_or_external(node: Node, cfg: TemporalAdapterConfig) -> str:
    # Type narrowing only. An mcp-only external_call is valid IR but has
    # no module to import here; emit_activities() already refused the
    # whole pipeline with an actionable message before reaching this.
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
            """{safe_docstring_line(node.description)}

            Compiled from MCP tool call → deterministic API call. See
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
    header = textwrap.dedent(
        f'''
        @activity.defn(name="{node.id}")
        async def {node.id}(payload: dict) -> dict:
            """{safe_docstring_line(node.description)}

            LLM judge — typed input/output, bounded decision space. The
            non-determinism lives inside this activity, not the workflow.
            """
        '''
    ).lstrip("\n")

    if node.signature_spec is not None:
        # Preferred: generated signature module (signatures/<node_id>.py,
        # emitted alongside this file). Its forward() is synchronous, so
        # it runs in a worker thread to keep the activity event loop free
        # during the LLM call.
        class_name = _to_pascal_case(node.id)
        input_class = f"{class_name}Input"
        return header + (
            f"    from signatures.{node.id} import {class_name}, {input_class}\n"
            f"\n"
            f"    judge = {class_name}()\n"
            f"    result = await asyncio.to_thread(judge.forward, {input_class}(**payload))\n"
            f"    return result.model_dump(by_alias=True)\n"
        )

    # Legacy path: user-maintained module with an async forward().
    assert node.signature is not None
    module_name, class_name = _signature_path_parts(node.signature)
    input_class = f"{class_name}Input"
    return header + (
        f"    from {cfg.signatures_module}.{module_name} import (\n"
        f"        {class_name},\n"
        f"        {input_class},\n"
        f"    )\n"
        f"    judge = {class_name}()\n"
        f"    result = await judge.forward({input_class}(**payload))\n"
        f"    return result.model_dump(by_alias=True)\n"
    )


def _emit_activity_for_agent_loop(node: Node, cfg: TemporalAdapterConfig) -> str:
    """Emit a real, bounded agent-loop activity.

    Not a stub: the provider (Claude subscription / API key / rote cloud)
    is resolved at runtime by ``signatures/_rote_inference.py``, the same
    helper the judges use.

    Unlike the DBOS and python adapters this one passes no
    ``local_tools``. A Temporal ``loop_body`` sub-node is its own
    ``@activity.defn`` coroutine — reaching into one from inside another
    activity would bypass the scheduler that makes it retryable and
    history-checkpointed, which is the whole reason to be on Temporal.
    Those sub-nodes stay workflow-orchestrated; the agent drives its MCP
    tools. Declared MCP tools work identically on every adapter.
    """
    # run_agent_loop is synchronous (it spawns a subprocess, or blocks on
    # the SDK's tool runner), so it goes to a worker thread — same reason
    # and same shape as the spec-judge activities above.
    inner = agent_loop_call(
        node,
        default_model=cfg.anthropic_default_model,
        indent="        ",
        include_local_tools=False,
    )
    return (
        f'\n@activity.defn(name="{node.id}")\n'
        f"async def {node.id}(payload: dict) -> dict:\n"
        f'    """{safe_docstring_line(node.description)}\n'
        f"\n"
        f"    Bounded agent loop. loop_body sub-nodes stay workflow-orchestrated\n"
        f"    activities on this runtime; the agent drives its declared MCP tools.\n"
        f'    """\n'
        f"\n"
        f"    def _run() -> dict:\n"
        f"{inner}"
        f"\n"
        f"    return await asyncio.to_thread(_run)\n"
    )


def emit_activities(pipeline: Pipeline, cfg: TemporalAdapterConfig | None = None) -> str:
    """Render the activities.py source for a pipeline.

    Raises ``ValueError`` if any ``external_call`` is bound to MCP with
    no ``impl`` — this adapter has no MCP backend (see
    :func:`refuse_mcp_only_nodes`).
    """
    cfg = cfg or TemporalAdapterConfig()
    refuse_mcp_only_nodes(pipeline, "temporal")

    parts: list[str] = []

    # Spec-carrying judges call their generated module's synchronous
    # forward() via asyncio.to_thread — only then is the import needed.
    has_spec_judges = any(
        n.kind is NodeKind.LLM_JUDGE and n.signature_spec is not None for n in pipeline.nodes
    )
    has_agent_loops = any(n.kind is NodeKind.AGENT_LOOP for n in pipeline.nodes)

    preamble = textwrap.dedent(
        f'''
        """Auto-generated by rote.adapters.temporal.

        Pipeline: {pipeline.name} v{pipeline.version}
        Source skill: {pipeline.source_skill or "unknown"}

        DO NOT EDIT BY HAND. Re-run ``rote emit`` to regenerate.

        Architecture note: every external_call activity in this file
        wraps a deterministic Python function from the ``extracted/``
        package. None of them call MCP tools at runtime — the MCP
        tool calls from the source skill have been compiled into
        direct API calls during the rote emission step.
        """

        from __future__ import annotations

        from typing import Any

        from temporalio import activity
        '''
    ).lstrip("\n")
    if has_spec_judges or has_agent_loops:
        preamble = preamble.replace(
            "from typing import Any",
            "import asyncio\nfrom typing import Any",
        )
    if has_agent_loops:
        # The agent-loop activity renders its task payload as JSON and
        # reads a ROTE_MODEL_<NODE_ID> override.
        preamble = preamble.replace(
            "from typing import Any",
            "import json\nimport os\nfrom typing import Any",
        )
    parts.append(preamble)

    parts.append(
        textwrap.dedent(
            '''

            def _serialize(obj: Any) -> Any:
                """Convert pydantic models / dataclasses / lists to plain dicts.

                Activities return JSON-serializable payloads so the workflow
                doesn't carry typed objects through Temporal's history.
                """
                if hasattr(obj, "model_dump"):
                    return obj.model_dump(by_alias=True)
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
        return "    def __init__(self) -> None:\n        pass\n\n"

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
                """{safe_docstring_line(gate.description)}"""
                self._{gate.signal}_payload = payload
            '''
        ).lstrip("\n")
        handler_blocks.append(textwrap.indent(raw, prefix="    "))
    return init_block + "\n".join(handler_blocks)


def _execute_activity_expr(
    node: Node, cfg: TemporalAdapterConfig, payload: str, indent: str
) -> str:
    """The ``workflow.execute_activity(...)`` expression for one node.

    The first line carries no indentation (the caller prefixes it with
    an assignment or a comma-separated position); continuation lines and
    the closing paren are indented against ``indent``.

    This is the single renderer behind all three dispatch shapes —
    lone node, parallel wave, and fan_out. It exists because the wave
    and single-node branches were once separate copies, and the wave
    copy quietly omitted ``retry_policy``: a node lost its declared
    retry budget merely by gaining a sibling, in exactly the parallel
    fetch waves where flaky network calls live. One renderer means a
    new field cannot be added to one shape and forgotten in the others.
    """
    inner = indent + "    "
    timeout = _activity_timeout(node, cfg.default_activity_timeout)
    lines = [
        "workflow.execute_activity(",
        f'{inner}"{node.id}",',
        f"{inner}{payload},",
        f'{inner}start_to_close_timeout=timedelta(minutes=_parse_minutes("{timeout}")),',
    ]
    retry = _retry_policy_args(node)
    if retry:
        lines.append(f"{inner}retry_policy={retry},")
    lines.append(f"{indent})")
    return "\n".join(lines)


def _emit_wave_call(node: Node, cfg: TemporalAdapterConfig) -> str:
    """Emit an awaited activity execution for a node alone in its wave."""
    if node.kind is NodeKind.HITL_GATE:
        # HITL gates are not activities — they're handled separately.
        return ""
    payload = _payload_literal(node, indent=" " * 12)
    expr = _execute_activity_expr(node, cfg, payload, indent=" " * 8)
    return f"        {node.id}_result = await {expr}\n"


def _emit_fan_out_call(node: Node, pipeline: Pipeline, cfg: TemporalAdapterConfig) -> str:
    """Emit a ``fan_out`` node as one activity execution per element.

    ``asyncio.gather`` preserves argument order, so ``<id>_result[i]``
    corresponds to element ``i`` of the bound list — the same ordering
    guarantee the other adapters give.
    """
    element_param, list_expr, scalars = fan_out_binding(node, pipeline, indent=" " * 20)
    shared = "".join(f', "{param}": {expr}' for param, expr in sorted(scalars.items()))
    payload = f'{{"{element_param}": _item{shared}}}'
    expr = _execute_activity_expr(node, cfg, payload, indent=" " * 20)
    return "\n".join(
        [
            f"        # fan_out: {node.id} runs once per element of the bound",
            "        # list, each as its own activity execution.",
            f"        {node.id}_result = list(",
            "            await asyncio.gather(",
            "                *(",
            f"                    {expr}",
            f"                    for _item in {list_expr}",
            "                )",
            "            )",
            "        )",
        ]
    )


def _emit_hitl_block(node: Node) -> str:
    assert node.kind is NodeKind.HITL_GATE
    assert node.signal is not None
    return (
        f"        # ─── HITL gate: {node.id} ───\n"
        f"        # Workflow suspends here until the {node.signal!r} signal\n"
        f"        # arrives. Survives worker restarts; resumes immediately\n"
        f"        # when the signal fires.\n"
        f"        await workflow.wait_condition(\n"
        f"            lambda: self._{node.signal}_payload is not None\n"
        f"        )\n"
        f"        {node.id}_result = self._{node.signal}_payload\n"
    )


def _emit_workflow_run(pipeline: Pipeline, cfg: TemporalAdapterConfig) -> str:
    waves = _execution_waves(pipeline)

    body_lines: list[str] = [
        "    @workflow.run",
        "    async def run(self, pipeline_input: dict) -> dict:",
    ]
    desc = safe_docstring_line(pipeline.description, fallback=pipeline.name)
    body_lines.append(f'        """{desc}"""')

    # Node ids whose results are bound by the time each wave starts —
    # used to reject inputs that reference a later wave at emit time.
    available: set[str] = set()

    for wave_idx, wave in enumerate(waves, start=1):
        non_hitl = [n for n in wave if n.kind is not NodeKind.HITL_GATE]
        hitl = [n for n in wave if n.kind is NodeKind.HITL_GATE]

        for n in non_hitl:
            check_input_refs_available(n, available)

        body_lines.append("")
        body_lines.append(f"        # ─── Wave {wave_idx} ───")

        # fan_out nodes dispatch once per element of their bound list —
        # they never share the single/parallel payload shapes below.
        fanned, plain = fan_out_nodes(non_hitl)

        if len(plain) == 1:
            body_lines.append(_emit_wave_call(plain[0], cfg).rstrip("\n"))
        elif len(plain) > 1:
            # Parallel via asyncio.gather
            body_lines.append("        (")
            for n in plain:
                body_lines.append(f"            {n.id}_result,")
            body_lines.append("        ) = await asyncio.gather(")
            for n in plain:
                payload = _payload_literal(n, indent=" " * 16)
                expr = _execute_activity_expr(n, cfg, payload, indent=" " * 12)
                body_lines.append(f"            {expr},")
            body_lines.append("        )")

        for n in fanned:
            body_lines.append(_emit_fan_out_call(n, pipeline, cfg))

        for h in hitl:
            body_lines.append(_emit_hitl_block(h).rstrip("\n"))

        available.update(n.id for n in wave)

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

            '''
        ).lstrip("\n")
    )

    if any(n.fan_out for n in pipeline.nodes):
        parts.append("\n" + fan_out_list_helper() + "\n")

    parts.append(
        textwrap.dedent(
            f'''
            @workflow.defn(name="{versioned_workflow_name}")
            class {class_name}:
                """Compiled workflow for {pipeline.name}."""

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
        """Write activities.py + workflow.py + __init__.py into output_dir.

        Judges carrying a ``signature_spec`` additionally get a generated
        ``signatures/<id>.py`` (shared Python signature emitter) that
        their activity imports; legacy ``signature: path:Class`` judges
        keep importing the user-maintained ``signatures_module``.
        """
        # Refuse before touching the filesystem: no partial output.
        refuse_mcp_only_nodes(pipeline, "temporal")

        writer = EmitWriter(output_dir)

        written = {
            "activities": writer.write("activities.py", content=self.emit_activities(pipeline)),
            "workflow": writer.write("workflow.py", content=self.emit_workflow(pipeline)),
            "__init__": writer.write(
                "__init__.py",
                content=f'"""Emitted Temporal artifacts for {pipeline.name}."""\n',
            ),
        }

        written.update(
            write_signature_package(
                writer,
                pipeline,
                anthropic_default_model=self.config.anthropic_default_model,
                openai_default_model=self.config.openai_default_model,
                generated_by="rote.adapters.temporal",
                regen_command="rote emit --runtime temporal",
                context_note=(
                    "The non-determinism lives inside this module; the activity\n"
                    "that calls it stays a retryable, history-checkpointed unit."
                ),
            )
        )

        writer.finalize()
        return written
