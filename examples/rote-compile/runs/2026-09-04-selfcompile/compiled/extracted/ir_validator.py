"""
Validate a pipeline.yaml using rote.ir.load_pipeline().

Checks all Phase 6 checklist items programmatically:
  - Every pure_function / external_call node has an impl: field.
  - Every llm_judge node has at least one of signature: or signature_spec:.
  - Every agent_loop node has a tools: list.
  - Every hitl_gate node has a signal: field.
  - Every node referenced in entry_nodes, exit_nodes, edges, and loop_body
    exists in nodes:.
  - Node IDs are unique.
  - All inputs: references resolve.

Contract:
  Input:  pipeline_yaml_path — path to the pipeline.yaml file to validate
  Output: dict with keys:
            is_valid   — True if validation passed
            node_count — number of nodes in the pipeline
            warnings   — list of non-fatal warning strings

Raises:
  ValidationError  — on any constraint violation (caller should not retry)
  FileNotFoundError — if pipeline_yaml_path does not exist
"""

from __future__ import annotations

from pathlib import Path


def validate_ir(pipeline_yaml_path: str) -> dict:
    """Load and validate pipeline.yaml; raise ValidationError on any violation."""
    path = Path(pipeline_yaml_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"pipeline.yaml not found at {pipeline_yaml_path}. "
            "assemble_ir must write it before validate_ir runs."
        )

    # rote.ir.load_pipeline is the authoritative validator. It enforces all
    # Phase 6 checklist constraints via Pydantic field validators and a
    # post-init DAG check. If this import fails, rote is not installed.
    try:
        from rote.ir import load_pipeline
    except ImportError as exc:
        raise RuntimeError(
            "rote package not found. Install it with: pip install rote-cli"
        ) from exc

    pipeline = load_pipeline(path)

    warnings: list[str] = []

    # Warn on agent_loop nodes with incomplete tool_servers maps.
    for node in pipeline.nodes:
        if node.kind == "agent_loop" and node.tools:
            unmapped = [t for t in node.tools if t not in (node.tool_servers or {})]
            if unmapped:
                warnings.append(
                    f"Node '{node.id}': tools without tool_servers entries: "
                    f"{unmapped}. MCP requirements will be under-reported."
                )

    return {
        "is_valid": True,
        "node_count": len(pipeline.nodes),
        "warnings": warnings,
    }
