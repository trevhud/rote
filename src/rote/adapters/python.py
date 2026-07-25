"""Raw Python adapter — emits a plain, orchestrator-free script from a Pipeline IR.

Layer 3 of rote: takes a validated :class:`rote.ir.Pipeline` and emits a
dependency-light Python script. This is the **maximum-legibility
target** — no durability library, no workflow engine, no decorator
magic. A developer can read ``main.py`` top to bottom, review every
retry loop and every payload wire, and paste the pieces into an
existing codebase.

Output layout::

    out/
        main.py                     # one plain function per node + run_pipeline()
        requirements.txt            # only the vendor SDKs the judges need
        README.md                   # how to run, what you give up vs. durable
        extracted/<module>.py       # stubs for pure_function / external_call
        extracted/<agent_loop>.py   # stubs for agent_loop nodes
        signatures/<llm_judge>.py   # typed Pydantic + vendor-SDK signatures

Key design choices vs. the durable adapters:

* **No durable execution — and loud about it.** Pipelines containing a
  ``hitl_gate`` are refused at emit time
  (:attr:`rote.ir.Pipeline.requires_durable_execution`): a plain script
  cannot durably park for human approval, and pretending otherwise
  (polling loops, pickled state files) would betray the legibility
  goal. The error points at ``--runtime dbos``.

* **Retries are visible for-loops.** The IR ``RetryPolicy`` becomes an
  inline ``for attempt in range(...)`` with ``time.sleep`` backoff in
  the function body — the whole point of this target is that nothing
  hides behind a decorator.

* **Parallel waves via the stdlib.** Multi-node waves run on a
  ``concurrent.futures.ThreadPoolExecutor`` and join before the next
  wave; single-node waves are a direct call.

* **Data-flow threading matches the other adapters.** ``run_pipeline``
  takes the pipeline input dict, binds every node's result as
  ``<id>_result``, and builds each payload from the node's ``inputs:``
  bindings via the shared reference grammar
  (:func:`rote.ir.parse_input_ref`). Forward references are rejected at
  emit time by :func:`rote.adapters._common.check_input_refs_available`.

* **Signatures and stubs are shared machinery.** ``signatures/*.py``
  and ``extracted/*.py`` come from :mod:`rote.adapters._py_common`, the
  same emitters the DBOS adapter uses — only the identity strings
  differ.

The emitted code never imports MCP runtime — same architectural
invariant as the other adapters, enforced by AST tests.
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
    safe_docstring_line,
)
from rote.adapters._py_common import (
    _extracted_layout,
    _impl_path_parts,
    _payload_literal,
    _signature_path_parts,
    resolve_extracted_source,
    serialize_helper,
)
from rote.adapters._py_common import (
    emit_extracted_module as _shared_emit_extracted_module,
)
from rote.adapters._py_common import (
    emit_signature_module as _shared_emit_signature_module,
)
from rote.ir import Node, NodeKind, Pipeline

# ───────── Adapter configuration ─────────


@dataclass(frozen=True)
class PythonAdapterConfig:
    """Per-emission knobs for the raw Python adapter."""

    anthropic_default_model: str = DEFAULT_ANTHROPIC_MODEL
    openai_default_model: str = DEFAULT_OPENAI_MODEL
    # Module path used for *legacy* `signature: path.py:Class` judges,
    # which import a user-maintained Python module instead of a
    # generated one. Mirrors DbosAdapterConfig.legacy_signatures_module.
    legacy_signatures_module: str = "signatures"
    # Base delay (seconds) between retry attempts in the emitted retry
    # loops. Exponential backoff doubles it per attempt; linear grows it
    # linearly; constant repeats it.
    retry_base_delay_seconds: float = 1.0
    # Directory holding the graduation's pipeline.yaml. When its
    # extracted/<module>.py exists (the agent's real, test-verified
    # implementation), emission uses that file verbatim instead of an
    # IR-derived NotImplementedError stub.
    extracted_source_dir: Path | None = None


# ───────── HITL refusal ─────────


def _refuse_if_durable(pipeline: Pipeline) -> None:
    """Refuse pipelines that need durable execution, loudly and early.

    A plain script cannot durably park for human approval — the wait
    would die with the process, and faking it (polling loops, pickled
    state files) would betray the adapter's legibility contract.
    """
    if pipeline.requires_durable_execution:
        gates = ", ".join(n.id for n in pipeline.nodes_by_kind(NodeKind.HITL_GATE))
        raise ValueError(
            f"python adapter: pipeline {pipeline.name!r} contains hitl_gate "
            f"node(s) [{gates}] — a plain script cannot durably park for human "
            f"approval; the wait would die with the process. Emit onto a "
            f"durable runtime instead: "
            f"`rote emit <pipeline.yaml> --runtime dbos` "
            f"(or --runtime temporal / --runtime cloudflare)."
        )


# ───────── Per-node comments ─────────


def _retry_on_comment(node: Node) -> str:
    """Document retry_on categories the plain loop can't express."""
    if node.retry and node.retry.retry_on:
        cats = ", ".join(node.retry.retry_on)
        return (
            f"    # retry_on categories from the IR: {cats}. The loop below\n"
            f"    # retries any exception; narrow its `except` clause if needed.\n"
        )
    return ""


def _timeout_comment(node: Node) -> str:
    """No clean stdlib per-step timeout for sync functions; document it."""
    if node.timeout:
        return (
            f"    # IR timeout {node.timeout!r}: a plain synchronous script has no\n"
            f"    # per-step timeout primitive; enforce inside the implementation\n"
            f"    # if required.\n"
        )
    return ""


def _mandatory_comment(node: Node) -> str:
    if node.mandatory:
        return (
            "    # MANDATORY: this node was marked mandatory in the source\n"
            "    # skill. The pipeline always calls it; do not make it conditional.\n"
        )
    return ""


# ───────── Inline retry loop ─────────

_BACKOFF_SLEEP_EXPR = {
    "exponential": "RETRY_BASE_DELAY_SECONDS * 2 ** (attempt - 1)",
    "linear": "RETRY_BASE_DELAY_SECONDS * attempt",
    "constant": "RETRY_BASE_DELAY_SECONDS",
}


def _emit_return(node: Node, call_expr: str) -> str:
    """Emit the function tail: a bare return, or a visible retry loop.

    The retry loop is a plain for-loop with ``time.sleep`` — the IR's
    ``RetryPolicy`` made flesh, with nothing hidden behind a decorator.
    """
    if not node.retry or node.retry.max == 0:
        return f"    return {call_expr}\n"

    sleep_expr = _BACKOFF_SLEEP_EXPR.get(node.retry.backoff)
    if sleep_expr is None:
        raise ValueError(
            f"python adapter: node {node.id!r} has unknown retry backoff "
            f"{node.retry.backoff!r} (expected one of "
            f"{sorted(_BACKOFF_SLEEP_EXPR)})"
        )
    attempts = node.retry.max + 1
    return (
        f"    attempts = {attempts}  # 1 initial call + {node.retry.max} "
        f"retries (IR retry policy)\n"
        f"    for attempt in range(1, attempts + 1):\n"
        f"        try:\n"
        f"            return {call_expr}\n"
        f"        except Exception:\n"
        f"            if attempt == attempts:\n"
        f"                raise\n"
        f"            # {node.retry.backoff} backoff, visible and in-process\n"
        f"            time.sleep({sleep_expr})\n"
        f'    raise AssertionError("unreachable: the retry loop returns or raises")\n'
    )


# ───────── Node function emission ─────────


def _emit_node_pure_or_external(node: Node) -> str:
    assert node.impl is not None
    module_name, func_name = _impl_path_parts(node.impl)
    return (
        f"def {node.id}(payload: dict) -> dict:\n"
        f'    """{safe_docstring_line(node.description)}\n'
        f"\n"
        f"    Graduated from MCP tool call → deterministic API call. See\n"
        f"    ``extracted.{module_name}`` for the implementation.\n"
        f'    """\n'
        f"{_mandatory_comment(node)}"
        f"{_retry_on_comment(node)}"
        f"{_timeout_comment(node)}"
        f"    from extracted.{module_name} import {func_name}\n"
        f"\n"
        f"{_emit_return(node, f'_serialize({func_name}(**payload))')}"
    )


def _emit_node_llm_judge(node: Node, cfg: PythonAdapterConfig) -> str:
    if node.signature_spec is not None:
        # Preferred: generated signature module (signatures/<node_id>.py).
        class_name = _to_pascal_case(node.id)
        import_block = f"    from signatures.{node.id} import {class_name}, {class_name}Input\n"
        call_expr = (
            f"{class_name}().forward({class_name}Input(**payload)).model_dump(by_alias=True)"
        )
    else:
        # Legacy Temporal-style path: user-maintained module with an
        # async ``forward`` (the Temporal adapter's convention).
        assert node.signature is not None
        module_name, class_name = _signature_path_parts(node.signature)
        import_block = (
            f"    import asyncio\n"
            f"\n"
            f"    from {cfg.legacy_signatures_module}.{module_name} import (\n"
            f"        {class_name},\n"
            f"        {class_name}Input,\n"
            f"    )\n"
        )
        call_expr = (
            f"asyncio.run({class_name}().forward({class_name}Input(**payload)))"
            f".model_dump(by_alias=True)"
        )
    return (
        f"def {node.id}(payload: dict) -> dict:\n"
        f'    """{safe_docstring_line(node.description)}\n'
        f"\n"
        f"    LLM judge — typed input/output, bounded decision space. The\n"
        f"    non-determinism lives inside the signature module, not here.\n"
        f'    """\n'
        f"{_retry_on_comment(node)}"
        f"{_timeout_comment(node)}"
        f"{import_block}"
        f"\n"
        f"{_emit_return(node, call_expr)}"
    )


def _emit_node_agent_loop(node: Node) -> str:
    return (
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
        f"{_emit_return(node, '_serialize(_impl(**payload))')}"
    )


# ───────── run_pipeline emission ─────────


def _has_parallel_wave(pipeline: Pipeline) -> bool:
    return any(len(wave) > 1 for wave in _execution_waves(pipeline))


def _has_retry(pipeline: Pipeline) -> bool:
    return any(n.retry and n.retry.max > 0 for n in pipeline.nodes)


def _emit_pipeline_body(pipeline: Pipeline) -> str:
    waves = _execution_waves(pipeline)
    lines: list[str] = []

    # Node ids whose results are bound by the time each wave starts —
    # used to reject inputs that reference a later wave at emit time.
    available: set[str] = set()

    for wave_idx, wave in enumerate(waves, start=1):
        for n in wave:
            check_input_refs_available(n, available)

        lines.append("")
        lines.append(f"    # ─── Wave {wave_idx} ───")

        if len(wave) == 1:
            node = wave[0]
            payload = _payload_literal(node, indent=" " * 8)
            if payload == "{}":
                lines.append(f"    {node.id}_result = {node.id}({{}})")
            else:
                lines.append(f"    {node.id}_result = {node.id}(")
                lines.append(f"        {payload}")
                lines.append("    )")
        else:
            lines.append("    # Independent nodes: run concurrently on a thread pool and")
            lines.append("    # join before the next wave starts.")
            lines.append("    with ThreadPoolExecutor() as pool:")
            for node in wave:
                payload = _payload_literal(node, indent=" " * 12)
                if payload == "{}":
                    lines.append(f"        {node.id}_future = pool.submit({node.id}, {{}})")
                else:
                    lines.append(f"        {node.id}_future = pool.submit(")
                    lines.append(f"            {node.id},")
                    lines.append(f"            {payload},")
                    lines.append("        )")
            for node in wave:
                lines.append(f"        {node.id}_result = {node.id}_future.result()")

        available.update(n.id for n in wave)

    lines.append("")
    lines.append("    return {")
    for exit_id in pipeline.exit_nodes:
        lines.append(f'        "{exit_id}": {exit_id}_result,')
    lines.append("    }")
    return "\n".join(lines)


def emit_main(pipeline: Pipeline, cfg: PythonAdapterConfig | None = None) -> str:
    """Render the main.py source for a pipeline.

    Raises ``ValueError`` if the pipeline requires durable execution
    (see :func:`_refuse_if_durable`).
    """
    cfg = cfg or PythonAdapterConfig()
    _refuse_if_durable(pipeline)

    pipeline_h = _pipeline_hash(pipeline)
    desc_first = safe_docstring_line(pipeline.description, fallback=pipeline.name)
    has_retry = _has_retry(pipeline)
    has_parallel = _has_parallel_wave(pipeline)

    import_lines = ["import json", "import sys"]
    if has_retry:
        import_lines.append("import time")
    if has_parallel:
        import_lines.append("from concurrent.futures import ThreadPoolExecutor")
    import_lines.append("from typing import Any")
    imports_block = "\n".join(import_lines)

    # Dedent BEFORE interpolating: imports_block is multi-line, and its
    # column-0 continuation lines would make an f-string + dedent a no-op
    # (the same bug emit_readme's comment documents).
    header = textwrap.dedent(
        '''\
        """Auto-generated by rote.adapters.python.

        Pipeline: {name} v{version}
        Source skill: {source_skill}
        Pipeline hash: {pipeline_h}

        DO NOT EDIT BY HAND. Re-run ``rote emit --runtime python`` to regenerate.

        This is the maximum-legibility target: a plain Python script with no
        workflow engine, no durability library, and no orchestrator process.
        Every retry loop and every payload wire is visible in this file. If
        the process dies mid-run, the run is gone — re-run from the start.
        Pipelines that must park for human approval or survive crashes should
        be emitted onto a durable runtime instead (``rote emit --runtime dbos``).

        Architecture note: every function in this file wraps a deterministic
        function from ``extracted/`` or a typed LLM signature from
        ``signatures/``. None of them call MCP tools at runtime — the MCP
        tool calls from the source skill were graduated into direct API
        calls during the rote emission step.
        """

        from __future__ import annotations

        {imports_block}
        '''
    ).format(
        name=pipeline.name,
        version=pipeline.version,
        source_skill=pipeline.source_skill or "unknown",
        pipeline_h=pipeline_h,
        imports_block=imports_block,
    )

    retry_const_block = ""
    if has_retry:
        retry_const_block = (
            "\n# Base delay between retry attempts, in seconds. Exponential\n"
            "# backoff doubles it per attempt; linear grows it linearly;\n"
            "# constant repeats it.\n"
            f"RETRY_BASE_DELAY_SECONDS = {cfg.retry_base_delay_seconds!r}\n"
        )

    serialize_block = "\n\n" + serialize_helper(
        "Node functions return JSON-serializable payloads so the final\n"
        "    result prints cleanly and pastes into other systems."
    )

    node_parts: list[str] = []
    for node in pipeline.nodes:
        node_parts.append("\n\n")
        if node.kind in (NodeKind.PURE_FUNCTION, NodeKind.EXTERNAL_CALL):
            node_parts.append(_emit_node_pure_or_external(node))
        elif node.kind is NodeKind.LLM_JUDGE:
            node_parts.append(_emit_node_llm_judge(node, cfg))
        elif node.kind is NodeKind.AGENT_LOOP:
            node_parts.append(_emit_node_agent_loop(node))
        else:  # pragma: no cover — hitl_gate is refused before emission
            raise ValueError(f"python adapter: unexpected node kind {node.kind!r}")

    pipeline_block = (
        f"\n\ndef run_pipeline(pipeline_input: dict) -> dict:\n"
        f'    """{desc_first}"""\n'
        f"{_emit_pipeline_body(pipeline)}\n"
    )

    main_block = textwrap.dedent(
        """\


        if __name__ == "__main__":
            # No engine, no daemon: parse the JSON input from argv[1], run the
            # pipeline in-process, print the result.
            pipeline_input = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
            result = run_pipeline(pipeline_input)
            print(json.dumps(result, indent=2, default=str))
        """
    )

    return (
        header
        + retry_const_block
        + serialize_block
        + "".join(node_parts)
        + pipeline_block
        + main_block
    )


# ───────── requirements.txt + README emission ─────────


def emit_requirements(pipeline: Pipeline) -> str:
    """Emit requirements.txt with only what the emitted code imports.

    The script itself is stdlib-only; third-party needs come exclusively
    from LLM-judge signatures (Pydantic for the typed models, plus the
    vendor SDK per ``signature_spec.client``). A pipeline with no judges
    gets an empty file with an explanatory comment.
    """
    judges = pipeline.nodes_by_kind(NodeKind.LLM_JUDGE)
    if not judges:
        return (
            "# Auto-generated by rote.adapters.python.\n"
            "# This pipeline has no LLM-judge nodes: the emitted script runs on\n"
            "# the Python standard library alone. Add whatever your extracted/\n"
            "# implementations need here.\n"
        )
    clients = {j.signature_spec.client for j in judges if j.signature_spec is not None}
    lines = [
        "# Auto-generated by rote.adapters.python. Only what the emitted",
        "# code imports: Pydantic for the typed judge signatures, plus the",
        "# vendor SDK per judge. Add whatever your extracted/ implementations",
        "# need here.",
        "pydantic>=2.7",
    ]
    if "anthropic" in clients:
        lines.append("anthropic>=0.89")
    if "openai" in clients:
        lines.append("openai>=1.40")
    return "\n".join(lines) + "\n"


def emit_readme(pipeline: Pipeline, cfg: PythonAdapterConfig) -> str:
    judges = pipeline.nodes_by_kind(NodeKind.LLM_JUDGE)
    judge_note = (
        "LLM judge steps call the vendor SDK directly and read the standard\n"
        "`ANTHROPIC_API_KEY` / `OPENAI_API_KEY` environment variables. Each\n"
        "judge also honors per-node `ROTE_MODEL_<NODE_ID>` and\n"
        "`ROTE_BASE_URL_<NODE_ID>` overrides, so you can swap the model or\n"
        "point at an OpenAI-compatible endpoint without re-emitting.\n"
        if judges
        else "This pipeline has no LLM-judge nodes — no API keys required.\n"
    )
    # Dedent BEFORE interpolating: a multi-line interpolated value would put
    # column-0 lines inside the template, making dedent a no-op and shipping
    # the whole README indented (Markdown renders that as one code block).
    template = textwrap.dedent(
        """\
        # {pipeline_name} — plain Python runtime

        Auto-generated by `rote emit --runtime python`. Do not edit generated
        files by hand; re-run the emitter to regenerate.

        This is the maximum-legibility target: `main.py` is a plain script a
        reviewer can read top to bottom. There is no workflow engine and no
        durability library — every retry loop is a visible `for` loop, every
        payload wire is an explicit dict literal, and parallel waves use the
        standard library's `ThreadPoolExecutor`.

        ## Layout

        - `main.py` — one plain function per node plus `run_pipeline()`
        - `extracted/` — stubs for deterministic nodes; fill these in with
          direct vendor API calls (`NotImplementedError` until you do)
        - `signatures/` — typed LLM judges generated from the pipeline IR
          (Pydantic models + direct vendor SDK calls)
        - `requirements.txt` — only what the emitted code imports

        ## Run

        ```sh
        pip install -r requirements.txt
        python main.py '{{"your": "input"}}'
        ```

        {judge_note}
        ## What you give up vs. a durable runtime

        - **No crash recovery.** If the process dies mid-run, the run is
          gone — re-run from the start. Steps are not checkpointed.
        - **No HITL gates.** Pipelines with `hitl_gate` nodes are refused at
          emit time; a plain script cannot durably park for human approval.
        - **In-process retries only.** Backoff is `time.sleep` inside the
          function; a machine reboot forgets the attempt count.

        Need any of those? Emit the same pipeline onto a durable runtime:

        ```sh
        rote emit <pipeline.yaml> --runtime dbos
        ```
        """
    )
    return template.format(pipeline_name=pipeline.name, judge_note=judge_note)


# ───────── Adapter facade ─────────


class PythonAdapter:
    """Facade that emits a plain Python script from a Pipeline IR.

    Output layout::

        out/
            main.py
            requirements.txt
            README.md
            extracted/__init__.py
            extracted/<module>.py
            signatures/__init__.py
            signatures/<llm_judge_id>.py

    The directory runs with ``python main.py '<json input>'`` once the
    user fills in the extracted/ stubs. Pipelines that require durable
    execution (any ``hitl_gate``) are refused at emit time — see
    :func:`_refuse_if_durable`.
    """

    def __init__(self, config: PythonAdapterConfig | None = None) -> None:
        self.config = config or PythonAdapterConfig()

    def emit_main(self, pipeline: Pipeline) -> str:
        return emit_main(pipeline, self.config)

    def emit(self, pipeline: Pipeline, output_dir: str | Path) -> dict[str, Path]:
        # Refuse before touching the filesystem: no partial output.
        _refuse_if_durable(pipeline)

        writer = EmitWriter(output_dir)

        written: dict[str, Path] = {}

        written["main"] = writer.write("main.py", content=self.emit_main(pipeline))

        extracted_modules = _extracted_layout(pipeline)
        if extracted_modules:
            written["extracted/__init__"] = writer.write(
                "extracted",
                "__init__.py",
                content=f'"""Extracted deterministic modules for {pipeline.name}."""\n',
            )
            for module_name, nodes in sorted(extracted_modules.items()):
                written[f"extracted/{module_name}"] = writer.write(
                    "extracted",
                    f"{module_name}.py",
                    content=resolve_extracted_source(self.config.extracted_source_dir, module_name)
                    or _shared_emit_extracted_module(
                        module_name,
                        nodes,
                        generated_by="rote.adapters.python",
                        caller_note=(
                            "the plain functions in main.py call these\n"
                            "with the step payload as keyword arguments."
                        ),
                    ),
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
                    content=_shared_emit_signature_module(
                        node,
                        anthropic_default_model=self.config.anthropic_default_model,
                        openai_default_model=self.config.openai_default_model,
                        generated_by="rote.adapters.python",
                        regen_command="rote emit --runtime python",
                        context_note=(
                            "The non-determinism lives inside this module; the plain\n"
                            "function in main.py that calls it stays a reviewable,\n"
                            "retryable unit."
                        ),
                    ),
                )

        written["requirements"] = writer.write(
            "requirements.txt", content=emit_requirements(pipeline)
        )
        written["README"] = writer.write("README.md", content=emit_readme(pipeline, self.config))

        writer.finalize()
        return written
