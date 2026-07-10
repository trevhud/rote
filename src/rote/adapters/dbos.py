"""DBOS adapter — emits a durable Python app from a Pipeline IR.

Layer 3 of rote: takes a validated :class:`rote.ir.Pipeline` and emits a
runnable DBOS Transact application. DBOS is the "no workflow runner"
target — durable execution as a Python library checkpointing to Postgres
(or SQLite for local dev). There is no orchestrator process to deploy;
``python main.py`` *is* the runtime.

Output layout::

    out/
        main.py                     # @DBOS.workflow + one @DBOS.step per node
        dbos-config.yaml            # CLI/Cloud tooling config (`dbos start`)
        README.md                   # how to run, signal HITL gates, deploy
        extracted/<module>.py       # stubs for pure_function / external_call
        extracted/<agent_loop>.py   # stubs for agent_loop nodes
        signatures/<llm_judge>.py   # typed Pydantic + vendor-SDK signatures

Key design choices vs. the other adapters:

* **Single process, single file orchestration.** DBOS has no
  workflow/activity split (Temporal) and no platform entrypoint class
  (Cloudflare). Steps and the workflow live together in ``main.py``;
  durability comes from the library checkpointing every step result to
  the system database.

* **Parallel waves via ``Queue``.** DBOS's documented concurrency
  primitive is enqueueing work onto a :class:`dbos.Queue`, which returns
  a ``WorkflowHandle`` per enqueued function. Multi-node waves enqueue
  every node then join with ``get_result()``; single-node waves call the
  step directly. This keeps the emitted code synchronous and the
  step-checkpoint ordering deterministic (asyncio.gather over steps has
  subtler replay-ordering semantics).

* **HITL gates via durable messages.** ``DBOS.recv(topic=...)`` parks
  the workflow in the system database until ``DBOS.send(workflow_id,
  payload, topic=...)`` delivers the message — the officially documented
  human-in-the-loop pattern. The IR ``signal`` name maps to the topic.

* **Data-flow threading matches the Temporal adapter.** The workflow
  takes the pipeline input dict as ``pipeline_input``, binds every
  node's result (including HITL gate resume payloads) as
  ``<id>_result``, and builds each step's payload from the node's
  ``inputs:`` bindings via the shared reference grammar
  (:func:`rote.ir.parse_input_ref`). Forward references are rejected at
  emit time by :func:`rote.adapters._common.check_input_refs_available`.

* **Typed LLM signatures as generated Pydantic modules.** When the IR
  carries ``signature_spec`` (preferred), the adapter converts the JSON
  Schemas to Pydantic model source and emits a direct vendor-SDK call
  with structured (tool-use / JSON-schema) output — the Python analog of
  the Cloudflare adapter's Zod emission. The legacy ``signature`` path
  form falls back to importing the user's module, matching Temporal.

The emitted code never imports MCP runtime — same architectural invariant
as the other adapters, enforced by AST tests.
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from rote.adapters._common import (
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_OPENAI_MODEL,
    EmitWriter,
    _duration_to_seconds,
    _execution_waves,
    _pipeline_hash,
    _seconds_literal,
    _to_pascal_case,
    check_input_refs_available,
    safe_docstring_line,
)
from rote.adapters._py_common import (
    _extracted_layout,
    _impl_path_parts,
    _payload_literal,
    _signature_path_parts,
    serialize_helper,
)
from rote.adapters._py_common import emit_extracted_module as _shared_emit_extracted_module
from rote.adapters._py_common import emit_signature_module as _shared_emit_signature_module
from rote.ir import Node, NodeKind, Pipeline

# ───────── Adapter configuration ─────────


@dataclass(frozen=True)
class DbosAdapterConfig:
    """Per-emission knobs for the DBOS adapter.

    Defaults work out-of-the-box for the BDR example: SQLite system
    database for local dev (overridable via ``DBOS_SYSTEM_DATABASE_URL``
    for Postgres in production), admin server off.
    """

    anthropic_default_model: str = DEFAULT_ANTHROPIC_MODEL
    openai_default_model: str = DEFAULT_OPENAI_MODEL
    # Module path used for *legacy* `signature: path.py:Class` judges,
    # which import a user-maintained Python module instead of a
    # generated one. Mirrors TemporalAdapterConfig.signatures_module.
    legacy_signatures_module: str = "signatures"
    # Steps enqueued for parallel waves land on this queue. None means
    # derive "<pipeline-name>-queue".
    queue_name: str | None = None
    # Backend for external_call nodes that carry an ``mcp`` binding:
    #   "mcp" → emit a working Streamable-HTTP call to the MCP tool (runs
    #           out of the box against the server the source skill used)
    #   "api" → emit the direct vendor-SDK path via ``impl`` (leaner; one
    #           key in .env). Nodes without an ``mcp`` binding always use
    #           ``impl`` regardless of this setting.
    external_backend: Literal["mcp", "api"] = "mcp"


# ───────── Retry mapping ─────────


def _step_decorator(node: Node) -> str:
    """Render the ``@DBOS.step(...)`` decorator line for a node.

    Maps the IR's ``RetryPolicy`` onto DBOS step retry parameters:

    | IR field      | DBOS parameter                                |
    |---------------|-----------------------------------------------|
    | ``max``       | ``max_attempts`` (= max + 1; DBOS counts the  |
    |               | initial attempt, like Temporal)               |
    | ``backoff``   | ``backoff_rate`` (exponential → 2.0;          |
    |               | constant → 1.0; linear → 1.0, the closest     |
    |               | available approximation — DBOS delay is       |
    |               | ``interval_seconds * backoff_rate**attempt``) |
    | ``retry_on``  | not mapped (DBOS retries any exception; its   |
    |               | ``should_retry`` predicate takes a callable,  |
    |               | not categories — surfaced as a comment)       |
    """
    args = [f'name="{node.id}"']
    if node.retry and node.retry.max > 0:
        args.append("retries_allowed=True")
        args.append(f"max_attempts={node.retry.max + 1}")
        backoff_rate = 2.0 if node.retry.backoff == "exponential" else 1.0
        args.append(f"backoff_rate={backoff_rate}")
    return f"@DBOS.step({', '.join(args)})"


def _retry_on_comment(node: Node) -> str:
    """Emit a comment documenting retry_on categories DBOS can't express."""
    if node.retry and node.retry.retry_on:
        cats = ", ".join(node.retry.retry_on)
        return (
            f"    # retry_on categories from the IR: {cats}. DBOS retries any\n"
            f"    # exception; narrow with @DBOS.step(should_retry=...) if needed.\n"
        )
    return ""


def _timeout_comment(node: Node) -> str:
    """DBOS steps have no per-step timeout primitive; document the IR value."""
    if node.timeout:
        return (
            f"    # IR timeout {node.timeout!r}: DBOS has no per-step timeout\n"
            f"    # primitive; enforce inside the implementation if required.\n"
        )
    return ""


# ───────── signatures/<id>.py emission ─────────


def emit_signature_module(node: Node, cfg: DbosAdapterConfig) -> str:
    """Emit signatures/<node_id>.py for an llm_judge node with a spec.

    Delegates to the shared Python signature emitter
    (:func:`rote.adapters._py_common.emit_signature_module`) with this
    adapter's identity strings: generated Pydantic models from the
    ``signature_spec`` JSON Schemas plus a typed judge class that calls
    the vendor SDK with structured output. See the shared module for the
    full emission contract.
    """
    return _shared_emit_signature_module(
        node,
        anthropic_default_model=cfg.anthropic_default_model,
        openai_default_model=cfg.openai_default_model,
        generated_by="rote.adapters.dbos",
        regen_command="rote emit --runtime dbos",
        context_note=(
            "The non-determinism lives inside this module; the workflow step that\n"
            "calls it stays a checkpointed, retryable unit."
        ),
    )


# ───────── extracted/<module>.py emission ─────────


def emit_extracted_module(module_name: str, nodes: list[Node]) -> str:
    """Emit extracted/<module_name>.py with one stub function per node.

    Delegates to the shared Python stub emitter
    (:func:`rote.adapters._py_common.emit_extracted_module`); the stubs
    raise ``NotImplementedError`` until the user fills them in with
    direct vendor API calls.
    """
    return _shared_emit_extracted_module(
        module_name,
        nodes,
        generated_by="rote.adapters.dbos",
        caller_note=(
            "the DBOS steps in main.py call these\nwith the step payload as keyword arguments."
        ),
    )


# ───────── main.py emission ─────────


def _emit_step_pure_or_external(node: Node) -> str:
    assert node.impl is not None
    module_name, func_name = _impl_path_parts(node.impl)
    mandatory_marker = ""
    if node.mandatory:
        mandatory_marker = (
            "    # MANDATORY: this node was marked mandatory in the source\n"
            "    # skill. The workflow always calls it; do not make it conditional.\n"
        )
    return (
        f"{_step_decorator(node)}\n"
        f"def {node.id}(payload: dict) -> dict:\n"
        f'    """{safe_docstring_line(node.description)}\n'
        f"\n"
        f"    Graduated from MCP tool call → deterministic API call. See\n"
        f"    ``extracted.{module_name}`` for the implementation.\n"
        f'    """\n'
        f"{mandatory_marker}"
        f"{_retry_on_comment(node)}"
        f"{_timeout_comment(node)}"
        f"    from extracted.{module_name} import {func_name}\n"
        f"\n"
        f"    return _serialize({func_name}(**payload))\n"
    )


def _emit_step_mcp(node: Node) -> str:
    """Emit a step that calls the node's MCP tool over Streamable HTTP.

    Unlike :func:`_emit_step_pure_or_external` (which imports a
    ``NotImplementedError`` stub), this produces a *working* body: it opens
    an authenticated FastMCP client to the resolved server endpoint (via
    the emitted ``extracted/_rote_mcp.py`` helper — env var > rote registry
    > the endpoint recorded here, with OAuth credentials from
    ``rote mcp login`` refreshing durably in place), calls the tool, and
    returns the structured result. The call runs inside the ``@DBOS.step``
    so its result is checkpointed and retried like any other step.
    """
    binding = node.mcp
    assert binding is not None
    url_literal = json.dumps(binding.url) if binding.url is not None else "None"
    if binding.args:
        arg_items = ", ".join(
            f"{json.dumps(tool_arg)}: payload[{json.dumps(payload_key)}]"
            for tool_arg, payload_key in binding.args.items()
        )
        args_line = f"    arguments = {{{arg_items}}}\n"
    else:
        args_line = "    arguments = payload  # tool arg names match the threaded payload keys\n"
    mandatory_marker = ""
    if node.mandatory:
        mandatory_marker = (
            "    # MANDATORY: this node was marked mandatory in the source\n"
            "    # skill. The workflow always calls it; do not make it conditional.\n"
        )
    return (
        f"{_step_decorator(node)}\n"
        f"def {node.id}(payload: dict) -> dict:\n"
        f'    """{safe_docstring_line(node.description)}\n'
        f"\n"
        f"    MCP-backed external_call → invokes tool {binding.tool!r} on MCP\n"
        f"    server {binding.server!r} over Streamable HTTP, authenticated\n"
        f"    from the rote credential store (`rote mcp login {binding.server}`).\n"
        f"    The result is checkpointed like any other durable step. Swap to\n"
        f"    a direct vendor-SDK call with `rote emit --backend api`.\n"
        f'    """\n'
        f"{mandatory_marker}"
        f"{_retry_on_comment(node)}"
        f"{_timeout_comment(node)}"
        f"    import asyncio\n"
        f"\n"
        f"    from extracted._rote_mcp import mcp_client\n"
        f"\n"
        f"{args_line}"
        f"\n"
        f"    async def _call() -> object:\n"
        f"        async with mcp_client({binding.server!r}, {url_literal}) as _client:\n"
        f"            _result = await _client.call_tool({json.dumps(binding.tool)}, arguments)\n"
        f"            return _result.data\n"
        f"\n"
        f"    return _serialize(asyncio.run(_call()))\n"
    )


def _emit_step_llm_judge(node: Node, cfg: DbosAdapterConfig) -> str:
    pascal_parts: tuple[str, str]
    if node.signature_spec is not None:
        # Preferred: generated signature module (signatures/<node_id>.py).
        module_name = node.id
        class_name = _to_pascal_case(node.id)
        import_line = f"    from signatures.{module_name} import {class_name}, {class_name}Input\n"
        call_lines = (
            f"    judge = {class_name}()\n"
            f"    return judge.forward({class_name}Input(**payload)).model_dump()\n"
        )
    else:
        # Legacy Temporal-style path: user-maintained module with an
        # async ``forward`` (the Temporal adapter's convention).
        assert node.signature is not None
        pascal_parts = _signature_path_parts(node.signature)
        module_name, class_name = pascal_parts
        import_line = (
            f"    import asyncio\n"
            f"\n"
            f"    from {cfg.legacy_signatures_module}.{module_name} import (\n"
            f"        {class_name},\n"
            f"        {class_name}Input,\n"
            f"    )\n"
        )
        call_lines = (
            f"    judge = {class_name}()\n"
            f"    result = asyncio.run(judge.forward({class_name}Input(**payload)))\n"
            f"    return result.model_dump()\n"
        )
    return (
        f"{_step_decorator(node)}\n"
        f"def {node.id}(payload: dict) -> dict:\n"
        f'    """{safe_docstring_line(node.description)}\n'
        f"\n"
        f"    LLM judge — typed input/output, bounded decision space. The\n"
        f"    non-determinism lives inside this step, not the workflow.\n"
        f'    """\n'
        f"{_retry_on_comment(node)}"
        f"{_timeout_comment(node)}"
        f"{import_line}"
        f"\n"
        f"{call_lines}"
    )


def _emit_step_agent_loop(node: Node) -> str:
    return (
        f"{_step_decorator(node)}\n"
        f"def {node.id}(payload: dict) -> dict:\n"
        f'    """{safe_docstring_line(node.description)}\n'
        f"\n"
        f"    Agent loop — bounded, tool-restricted. The stub in\n"
        f"    ``extracted.{node.id}`` raises until implemented against an\n"
        f"    agent harness.\n"
        f'    """\n'
        f"{_retry_on_comment(node)}"
        f"{_timeout_comment(node)}"
        f"    from extracted.{node.id} import {node.id} as _impl\n"
        f"\n"
        f"    return _serialize(_impl(**payload))\n"
    )


def _emit_hitl_wait(node: Node, pipeline: Pipeline) -> str:
    assert node.signal is not None
    timeout_str = node.timeout or pipeline.config.hitl.default_timeout
    timeout_seconds = _duration_to_seconds(timeout_str)
    return (
        f"    # ─── HITL gate: {node.id} ───\n"
        f"    # Durable receive: the workflow parks in the system database until\n"
        f"    # DBOS.send(workflow_id, payload, topic={node.signal!r})\n"
        f"    # delivers the resume message. Survives process restarts.\n"
        f"    {node.id}_result = DBOS.recv(\n"
        f'        topic="{node.signal}",\n'
        f"        timeout_seconds={_seconds_literal(timeout_seconds)},  # {timeout_str}\n"
        f"    )\n"
        f"    if {node.id}_result is None:\n"
        f"        raise TimeoutError(\n"
        f'            "HITL gate {node.id!r} timed out after {timeout_str} waiting "\n'
        f'            "for signal {node.signal!r}"\n'
        f"        )\n"
    )


def _emit_workflow_body(pipeline: Pipeline) -> str:
    waves = _execution_waves(pipeline)
    lines: list[str] = []

    # Node ids whose results are bound by the time each wave starts —
    # used to reject inputs that reference a later wave at emit time.
    # HITL gates count: their DBOS.recv payload binds `<id>_result`, so
    # gate resume payloads participate as that gate's result downstream.
    available: set[str] = set()

    for wave_idx, wave in enumerate(waves, start=1):
        non_hitl = [n for n in wave if n.kind is not NodeKind.HITL_GATE]
        hitl = [n for n in wave if n.kind is NodeKind.HITL_GATE]

        for n in non_hitl:
            check_input_refs_available(n, available)

        lines.append("")
        lines.append(f"    # ─── Wave {wave_idx} ───")

        if len(non_hitl) == 1:
            node = non_hitl[0]
            payload = _payload_literal(node, indent=" " * 8)
            if payload == "{}":
                lines.append(f"    {node.id}_result = {node.id}({{}})")
            else:
                lines.append(f"    {node.id}_result = {node.id}(")
                lines.append(f"        {payload}")
                lines.append("    )")
        elif len(non_hitl) > 1:
            lines.append("    # Parallel fan-out: enqueue every node in the wave, then")
            lines.append("    # join. Each enqueued step runs as its own one-step")
            lines.append("    # workflow; get_result() blocks durably.")
            for node in non_hitl:
                payload = _payload_literal(node, indent=" " * 8)
                if payload == "{}":
                    lines.append(f"    {node.id}_handle = queue.enqueue({node.id}, {{}})")
                else:
                    lines.append(f"    {node.id}_handle = queue.enqueue(")
                    lines.append(f"        {node.id},")
                    lines.append(f"        {payload},")
                    lines.append("    )")
            for node in non_hitl:
                lines.append(f"    {node.id}_result = {node.id}_handle.get_result()")

        for gate in hitl:
            lines.append(_emit_hitl_wait(gate, pipeline).rstrip("\n"))

        available.update(n.id for n in wave)

    lines.append("")
    lines.append("    return {")
    for exit_id in pipeline.exit_nodes:
        lines.append(f'        "{exit_id}": {exit_id}_result,')
    lines.append("    }")
    return "\n".join(lines)


def emit_main(pipeline: Pipeline, cfg: DbosAdapterConfig | None = None) -> str:
    """Render the main.py source for a pipeline."""
    cfg = cfg or DbosAdapterConfig()

    pascal = _to_pascal_case(pipeline.name)
    pipeline_h = _pipeline_hash(pipeline)
    workflow_name = f"{pascal}_{pipeline_h}"
    queue_name = cfg.queue_name or f"{pipeline.name}-queue"
    sqlite_file = f"{pipeline.name}.dbos.sqlite"
    desc_first = safe_docstring_line(pipeline.description, fallback=pipeline.name)

    header = textwrap.dedent(
        f'''\
        """Auto-generated by rote.adapters.dbos.

        Pipeline: {pipeline.name} v{pipeline.version}
        Source skill: {pipeline.source_skill or "unknown"}
        Pipeline hash: {pipeline_h}

        DO NOT EDIT BY HAND. Re-run ``rote emit --runtime dbos`` to regenerate.

        Workflow versioning: the registered workflow name includes a hash of
        the pipeline so regenerated pipelines become a new workflow type.
        In-flight workflows recover onto the code they started with (DBOS
        recovers by workflow name + application_version); new starts use the
        new code.

        Architecture note: every step in this file wraps a deterministic
        function from ``extracted/`` or a typed LLM signature from
        ``signatures/``. None of them call MCP tools at runtime — the MCP
        tool calls from the source skill were graduated into direct API
        calls during the rote emission step.
        """

        from __future__ import annotations

        import json
        import os
        import sys
        import threading
        from pathlib import Path
        from typing import Any

        from dbos import DBOS, DBOSConfig, Queue

        # ───────── DBOS configuration ─────────
        #
        # Local dev runs on SQLite (zero infrastructure). Production should
        # point DBOS_SYSTEM_DATABASE_URL at Postgres — SQLite is
        # single-process only.
        _APP_DIR = Path(__file__).resolve().parent

        config: DBOSConfig = {{
            "name": "{pipeline.name}",
            "system_database_url": os.environ.get(
                "DBOS_SYSTEM_DATABASE_URL",
                f"sqlite:///{{_APP_DIR / '{sqlite_file}'}}",
            ),
            # The admin server (port 3001) is opt-in; enable it when you
            # want the DBOS console / management API.
            "run_admin_server": False,
        }}
        DBOS(config=config)

        # Parallel waves fan out onto this queue; each enqueued step runs as
        # a one-step workflow with its own durable handle.
        queue = Queue("{queue_name}")
        '''
    )

    header += "\n\n" + serialize_helper(
        "Steps return JSON-serializable payloads so workflow state stays\n"
        "    portable across the system database."
    )

    step_parts: list[str] = []
    for node in pipeline.nodes:
        if node.kind is NodeKind.HITL_GATE:
            continue
        step_parts.append("\n\n")
        if node.kind is NodeKind.PURE_FUNCTION:
            step_parts.append(_emit_step_pure_or_external(node))
        elif node.kind is NodeKind.EXTERNAL_CALL:
            if node.mcp is not None and cfg.external_backend == "mcp":
                step_parts.append(_emit_step_mcp(node))
            else:
                step_parts.append(_emit_step_pure_or_external(node))
        elif node.kind is NodeKind.LLM_JUDGE:
            step_parts.append(_emit_step_llm_judge(node, cfg))
        elif node.kind is NodeKind.AGENT_LOOP:
            step_parts.append(_emit_step_agent_loop(node))

    workflow_block = (
        f"\n\n@DBOS.workflow(name={json.dumps(workflow_name)})\n"
        f"def run_pipeline(pipeline_input: dict) -> dict:\n"
        f'    """{desc_first}"""\n'
        f"{_emit_workflow_body(pipeline)}\n"
    )

    main_block = textwrap.dedent(
        """\


        if __name__ == "__main__":
            # Launch DBOS (connects to the system database, starts queue
            # workers and recovery). Two modes:
            #
            #   python main.py --serve
            #       Run as a long-lived worker: execute runs enqueued
            #       externally (`rote serve` MCP tools / DBOSClient.enqueue)
            #       against the same system database. This is what
            #       `dbos start` and production deployments run.
            #
            #   python main.py '{"your": "input"}'
            #       Start one pipeline run with the input from argv[1]
            #       (JSON) and block until it completes.
            #
            # HITL gates are resumed from another process either way — see
            # README.md.
            DBOS.launch()
            try:
                if "--serve" in sys.argv[1:]:
                    print("serving: waiting for enqueued runs (Ctrl-C to stop)", file=sys.stderr)
                    threading.Event().wait()
                else:
                    pipeline_input = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
                    handle = DBOS.start_workflow(run_pipeline, pipeline_input)
                    print(f"workflow started: {handle.workflow_id}", file=sys.stderr)
                    print(json.dumps(handle.get_result(), indent=2, default=str))
            finally:
                DBOS.destroy()
        """
    )

    return header + "".join(step_parts) + workflow_block + main_block


# ───────── dbos-config.yaml + README emission ─────────


def emit_dbos_config(pipeline: Pipeline) -> str:
    """Emit dbos-config.yaml so ``dbos start`` (and DBOS Cloud) work.

    Runtime configuration is programmatic (the ``DBOSConfig`` dict in
    main.py); this file only feeds the CLI/Cloud tooling.
    """
    return (
        "# Auto-generated by rote.adapters.dbos. Used by the `dbos` CLI and\n"
        "# DBOS Cloud; runtime config lives in main.py's DBOSConfig.\n"
        f"name: {pipeline.name}\n"
        "language: python\n"
        "runtimeConfig:\n"
        "  start:\n"
        "    # Long-lived worker mode: executes runs enqueued externally\n"
        "    # (rote serve / DBOSClient). One-shot runs: python3 main.py '{...}'.\n"
        "    - python3 main.py --serve\n"
    )


def emit_readme(pipeline: Pipeline, cfg: DbosAdapterConfig) -> str:
    mcp_servers = sorted(
        {
            n.mcp.server
            for n in pipeline.nodes
            if n.kind is NodeKind.EXTERNAL_CALL
            and n.mcp is not None
            and cfg.external_backend == "mcp"
        }
    )
    if mcp_servers:
        env_lines = "\n".join(
            f"export ROTE_MCP_{s.upper()}_URL=...   # {s!r} MCP server (Streamable HTTP)"
            for s in mcp_servers
        )
        mcp_note = (
            "\n## MCP-backed steps\n\n"
            "Some `external_call` steps call MCP tools over Streamable HTTP\n"
            "(the `mcp` backend). To run them, install `fastmcp` and set one\n"
            "endpoint env var per server:\n\n"
            "```sh\n"
            "pip install fastmcp\n"
            f"{env_lines}\n"
            "```\n\n"
            "Prefer direct vendor-SDK calls? Re-emit with "
            "`rote emit --runtime dbos --backend api` and fill in the\n"
            "`extracted/` stubs (one key in `.env` per vendor).\n"
        )
    else:
        mcp_note = ""

    gates = [n for n in pipeline.nodes if n.kind is NodeKind.HITL_GATE]
    gate_lines = "\n".join(
        f"| `{g.id}` | `{g.signal}` | {g.timeout or pipeline.config.hitl.default_timeout} |"
        for g in gates
    )
    first_signal = gates[0].signal if gates else "example_signal"
    # Dedent BEFORE interpolating: a multi-line gate_lines value would put
    # column-0 lines inside the template, making dedent a no-op and shipping
    # the whole README indented (Markdown renders that as one code block).
    template = textwrap.dedent(
        """\
        # {pipeline_name} — DBOS runtime

        Auto-generated by `rote emit --runtime dbos`. Do not edit generated
        files by hand; re-run the emitter to regenerate.

        DBOS is durable execution as a library: there is no orchestrator to
        deploy. `main.py` checkpoints every step to a system database and
        resumes from the last completed step after a crash.

        ## Layout

        - `main.py` — the `@DBOS.workflow` DAG plus one `@DBOS.step` per node
        - `extracted/` — stubs for deterministic nodes; fill these in with
          direct vendor API calls (`NotImplementedError` until you do)
        - `signatures/` — typed LLM judges generated from the pipeline IR
          (Pydantic models + direct vendor SDK calls)
        - `dbos-config.yaml` — CLI/Cloud tooling config for `dbos start`

        ## Run locally

        ```sh
        pip install dbos
        python main.py '{{"your": "input"}}'   # one run, blocks until done
        python main.py --serve                 # long-lived worker (see below)
        ```

        `--serve` keeps the process alive executing runs enqueued externally
        — `rote register --runtime dbos` + `rote serve` expose this app as an
        MCP tool whose trigger enqueues onto this app's queue. `dbos start`
        runs the same mode.

        By default the system database is a SQLite file next to `main.py` —
        zero infrastructure, ideal for development. For production, point
        DBOS at Postgres (SQLite is single-process only):

        ```sh
        export DBOS_SYSTEM_DATABASE_URL="postgresql://user:pass@host:5432/db"
        python main.py '{{...}}'   # or: dbos start
        ```

        LLM judge steps call the vendor SDK directly and read the standard
        `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` environment variables. Each
        judge also honors two per-node overrides, so operators can change
        the model or point at an OpenAI-compatible endpoint (Ollama, vLLM,
        a gateway) without re-emitting:

        ```sh
        export ROTE_MODEL_<NODE_ID>=...      # e.g. ROTE_MODEL_VET_CONTACT
        export ROTE_BASE_URL_<NODE_ID>=...   # custom endpoint for that judge
        ```
        {mcp_note}
        ## HITL gates

        The workflow parks durably at each gate until a message arrives on
        the gate's topic:

        | Gate | Signal (topic) | Timeout |
        | --- | --- | --- |
        {gate_lines}

        Resume a parked workflow from any process that can reach the system
        database:

        ```python
        from dbos import DBOSClient

        client = DBOSClient(system_database_url="...")
        client.send(workflow_id, {{"approved": True}}, topic="{first_signal}")
        ```

        (In-process, `DBOS.send(workflow_id, payload, topic=...)` does the
        same.) A gate that times out raises `TimeoutError` and fails the
        run — silence is not approval.
        """
    )
    return template.format(
        pipeline_name=pipeline.name,
        gate_lines=gate_lines,
        first_signal=first_signal,
        mcp_note=mcp_note,
    )


# ───────── Adapter facade ─────────


class DbosAdapter:
    """Facade that emits a DBOS application from a Pipeline IR.

    Output layout::

        out/
            main.py
            dbos-config.yaml
            README.md
            extracted/__init__.py
            extracted/<module>.py
            signatures/__init__.py
            signatures/<llm_judge_id>.py

    The directory runs with ``python main.py`` once the user fills in the
    extracted/ stubs (SQLite system DB by default; Postgres via
    ``DBOS_SYSTEM_DATABASE_URL``).
    """

    def __init__(self, config: DbosAdapterConfig | None = None) -> None:
        self.config = config or DbosAdapterConfig()

    def emit_main(self, pipeline: Pipeline) -> str:
        return emit_main(pipeline, self.config)

    def emit(self, pipeline: Pipeline, output_dir: str | Path) -> dict[str, Path]:
        writer = EmitWriter(output_dir)

        written: dict[str, Path] = {}

        written["main"] = writer.write("main.py", content=self.emit_main(pipeline))

        extracted_modules = _extracted_layout(pipeline)
        mcp_backed = self.config.external_backend == "mcp" and any(
            n.mcp is not None for n in pipeline.nodes
        )
        if extracted_modules or mcp_backed:
            written["extracted/__init__"] = writer.write(
                "extracted",
                "__init__.py",
                content=f'"""Extracted deterministic modules for {pipeline.name}."""\n',
            )
            for module_name, nodes in sorted(extracted_modules.items()):
                written[f"extracted/{module_name}"] = writer.write(
                    "extracted",
                    f"{module_name}.py",
                    content=emit_extracted_module(module_name, nodes),
                )
        if mcp_backed:
            # The connection helper is the *source text* of
            # rote.mcp._runtime_helper — one tested implementation,
            # emitted verbatim so the app stays standalone (no rote
            # import at runtime; auth comes from `rote mcp login`).
            from rote.mcp import _runtime_helper

            written["extracted/_rote_mcp"] = writer.write(
                "extracted",
                "_rote_mcp.py",
                content=Path(_runtime_helper.__file__).read_text(encoding="utf-8"),
            )

        spec_judges = [
            n
            for n in pipeline.nodes
            if n.kind is NodeKind.LLM_JUDGE and n.signature_spec is not None
        ]
        if spec_judges:
            written["signatures/__init__"] = writer.write(
                "signatures",
                "__init__.py",
                content=f'"""Generated LLM signatures for {pipeline.name}."""\n',
            )
            for node in spec_judges:
                written[f"signatures/{node.id}"] = writer.write(
                    "signatures",
                    f"{node.id}.py",
                    content=emit_signature_module(node, self.config),
                )

        written["dbos-config"] = writer.write(
            "dbos-config.yaml", content=emit_dbos_config(pipeline)
        )

        written["README"] = writer.write("README.md", content=emit_readme(pipeline, self.config))

        writer.finalize()
        return written
