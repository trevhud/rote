"""Intermediate representation for graduated skills.

The IR is a runtime-agnostic DAG of typed nodes. Adapters
(``rote/adapters/temporal.py``, etc.) consume an :class:`Pipeline` and emit
runnable code for a specific durable execution engine.

The schema is intentionally driven by what real complex skills (BDR
outreach is the canonical example) need to express. Adding fields here
should always be motivated by a concrete skill that needs them, not
speculation.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class NodeKind(StrEnum):
    """The five kinds a graduated step can be classified as.

    Every node in a graduated pipeline is exactly one kind. Adapters
    decide how to emit each kind for their target runtime.
    """

    PURE_FUNCTION = "pure_function"
    LLM_JUDGE = "llm_judge"
    AGENT_LOOP = "agent_loop"
    HITL_GATE = "hitl_gate"
    EXTERNAL_CALL = "external_call"


class RetryPolicy(BaseModel):
    """Retry behavior for a node. Adapters map this onto runtime primitives."""

    model_config = ConfigDict(extra="forbid")

    max: int = Field(ge=0, description="Maximum retry attempts (0 = no retry)")
    backoff: str = Field(default="exponential", description="linear | exponential | constant")
    retry_on: list[str] | None = Field(
        default=None,
        description=(
            "Error categories to retry on. None = retry all transient errors. "
            "Named retry_on (not 'on') because YAML 1.1 parses bare 'on' as a boolean."
        ),
    )


class CacheConfig(BaseModel):
    """Caching for nodes whose output is stable across runs (e.g., taxonomy IDs)."""

    model_config = ConfigDict(extra="forbid")

    strategy: str = Field(description="persistent | memory")
    ttl: str = Field(description="Duration string, e.g., '30d', '1h'")


class TerminationConfig(BaseModel):
    """Termination criteria for an agent_loop node."""

    model_config = ConfigDict(extra="forbid")

    condition: str = Field(description="Human-readable termination condition")
    max_iterations: int = Field(ge=1, description="Hard upper bound on iterations")


class NotifyConfig(BaseModel):
    """How a hitl_gate notifies the human reviewer."""

    model_config = ConfigDict(extra="forbid")

    channel: str = Field(description="slack | email | webhook")
    target: str = Field(description="Channel name, email address, or webhook URL")
    message_template: str | None = Field(
        default=None,
        description="Template string with {placeholders} from upstream node outputs",
    )


class LLMSignature(BaseModel):
    """Runtime-agnostic, structured form of an llm_judge signature.

    Carries everything an adapter needs to emit a working LLM call in any
    target language: JSON Schema for the I/O contract (which Pydantic and
    Zod both derive cleanly from), a prompt template, and the vendor
    client config. The legacy ``signature: 'path/to/file.py:Class'`` form
    on :class:`Node` continues to work for Temporal back-compat; new
    runtimes (Cloudflare Workflows, future TS targets) require this
    structured form because there's no shared Python module to import.
    """

    model_config = ConfigDict(extra="forbid")

    input_schema: dict[str, Any] = Field(
        description="JSON Schema for the typed input. Adapters convert to Pydantic / Zod.",
    )
    output_schema: dict[str, Any] = Field(
        description="JSON Schema for the typed output. Adapters convert to Pydantic / Zod.",
    )
    prompt: str = Field(
        description=(
            "Prompt template (Jinja-style {{ var }} interpolation). "
            "Variables resolve against the input schema's properties."
        ),
    )
    client: str = Field(
        default="anthropic",
        description="LLM vendor identifier. Currently 'anthropic' | 'openai'.",
    )
    model: str | None = Field(
        default=None,
        description="Vendor-specific model id. None = adapter chooses default.",
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature; None = vendor default.",
    )

    @field_validator("client")
    @classmethod
    def _validate_client(cls, v: str) -> str:
        allowed = {"anthropic", "openai"}
        if v not in allowed:
            raise ValueError(f"client must be one of {sorted(allowed)}, got {v!r}")
        return v


class Node(BaseModel):
    """A single step in the graduated pipeline.

    The fields used depend on the node ``kind``. Cross-field validation
    enforces that, e.g., ``llm_judge`` nodes have a ``signature`` and
    ``hitl_gate`` nodes have a ``signal``.
    """

    model_config = ConfigDict(extra="forbid")

    # Common fields
    id: str = Field(description="Unique identifier within the pipeline")
    kind: NodeKind
    phase: str | None = Field(
        default=None,
        description="Metadata pointing back to the source skill's phase number",
    )

    @field_validator("phase", mode="before")
    @classmethod
    def _coerce_phase_to_string(cls, v: Any) -> str | None:
        """BDR has 'Phase 1.5' which YAML parses as a float; coerce to string."""
        if v is None:
            return None
        return str(v)

    description: str = Field(description="What this node does, in prose")
    input: dict[str, str] = Field(
        default_factory=dict,
        description="Input field name → type name (free-form for v0)",
    )
    output: dict[str, str] | str = Field(
        default_factory=dict,
        description="Output field name → type name, or a single type name",
    )
    timeout: str | None = Field(
        default=None,
        description="Duration string, e.g., '5m', '60s'",
    )
    retry: RetryPolicy | None = None
    mandatory: bool = Field(
        default=False,
        description="If true, this node cannot be skipped or made conditional",
    )
    constants: dict[str, Any] | None = Field(
        default=None,
        description="Hard-coded constants extracted from the source skill",
    )
    cache: CacheConfig | None = None
    fan_out: bool = Field(
        default=False,
        description="If true, this node is invoked once per input element",
    )

    # pure_function / external_call fields
    impl: str | None = Field(
        default=None,
        description="Path to extracted function, e.g., 'extracted/foo.py:bar'",
    )

    # llm_judge fields
    signature: str | None = Field(
        default=None,
        description=(
            "Legacy: path to a Python signature class, e.g., 'signatures/foo.py:Foo'. "
            "The Temporal adapter accepts this for back-compat. New runtimes "
            "(Cloudflare, etc.) require ``signature_spec`` instead — see below."
        ),
    )
    signature_spec: LLMSignature | None = Field(
        default=None,
        description=(
            "Structured, runtime-agnostic signature: JSON Schema in/out + prompt "
            "+ client config. Required for non-Python adapters; optional for "
            "Temporal (which can fall back to ``signature`` path)."
        ),
    )
    eval_set: str | None = Field(
        default=None,
        description="Path to seed eval set",
    )

    # agent_loop fields
    tools: list[str] | None = Field(
        default=None,
        description="MCP tool names the agent can call inside the loop",
    )
    loop_body: list[str] | None = Field(
        default=None,
        description="IDs of nodes invoked from inside each loop iteration",
    )
    termination: TerminationConfig | None = None

    # hitl_gate fields
    signal: str | None = Field(
        default=None,
        description="Signal name the workflow waits for (resume on signal)",
    )
    notify: NotifyConfig | None = None

    @model_validator(mode="after")
    def _validate_kind_specific_fields(self) -> Self:
        """Each node kind has required fields. Enforce them here."""
        kind = self.kind
        missing: list[str] = []

        if kind in (NodeKind.PURE_FUNCTION, NodeKind.EXTERNAL_CALL):
            if not self.impl:
                missing.append("impl")
        elif kind is NodeKind.LLM_JUDGE:
            # Either the legacy path or the structured spec is acceptable.
            # Both being absent is the failure mode.
            if not self.signature and self.signature_spec is None:
                missing.append("signature")
        elif kind is NodeKind.AGENT_LOOP:
            if not self.tools:
                missing.append("tools")
        elif kind is NodeKind.HITL_GATE and not self.signal:
            missing.append("signal")

        if missing:
            raise ValueError(
                f"Node {self.id!r} of kind {kind.value} is missing required field(s): "
                f"{', '.join(missing)}"
            )

        # mandatory only meaningful for non-agentic kinds (you can't make an
        # agent loop "mandatory" — that's a no-op)
        if self.mandatory and kind is NodeKind.AGENT_LOOP:
            raise ValueError(f"Node {self.id!r}: mandatory=true is not allowed on agent_loop nodes")

        return self


class Edge(BaseModel):
    """A directed edge in the pipeline DAG."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    from_: str = Field(alias="from", description="Source node id")
    to: str = Field(description="Destination node id")
    on_signal: str | None = Field(
        default=None,
        description="If set, edge only activates when the named signal fires",
    )
    fan_out: bool = Field(
        default=False,
        description="If true, the destination is invoked once per element of the source's output",
    )


class ObservabilityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    traces: bool = True
    eval_set_dir: str | None = None


class HITLConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default_timeout: str = "7d"


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schedule: str | None = None
    on_failure: str = "notify_owner"
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    hitl: HITLConfig = Field(default_factory=HITLConfig)


class PipelineInput(BaseModel):
    """The pipeline's input contract."""

    model_config = ConfigDict(extra="forbid")

    type: str = Field(description="Type name for the pipeline input")
    required: list[str] = Field(default_factory=list)
    optional: list[str] = Field(default_factory=list)


class Pipeline(BaseModel):
    """A graduated skill, ready to be emitted into a runtime adapter."""

    model_config = ConfigDict(extra="forbid")

    name: str
    version: str = "0.1.0"
    source_skill: str | None = Field(
        default=None,
        description="Path to the source skill bundle this was graduated from",
    )
    description: str = ""
    config: PipelineConfig = Field(default_factory=PipelineConfig)
    input: PipelineInput
    nodes: list[Node]
    edges: list[Edge]
    entry_nodes: list[str] = Field(default_factory=list)
    exit_nodes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_dag_integrity(self) -> Self:
        """Cross-references must point at real nodes; entry/exit must exist."""
        node_ids = {n.id for n in self.nodes}
        if len(node_ids) != len(self.nodes):
            seen: set[str] = set()
            dups = [n.id for n in self.nodes if n.id in seen or seen.add(n.id)]  # type: ignore[func-returns-value]
            raise ValueError(f"Duplicate node ids: {dups}")

        for edge in self.edges:
            if edge.from_ not in node_ids:
                raise ValueError(f"Edge from unknown node: {edge.from_!r}")
            if edge.to not in node_ids:
                raise ValueError(f"Edge to unknown node: {edge.to!r}")

        for entry in self.entry_nodes:
            if entry not in node_ids:
                raise ValueError(f"entry_nodes references unknown node: {entry!r}")
        for exit_id in self.exit_nodes:
            if exit_id not in node_ids:
                raise ValueError(f"exit_nodes references unknown node: {exit_id!r}")

        # loop_body references must also be valid node IDs
        for node in self.nodes:
            if node.loop_body:
                for body_id in node.loop_body:
                    if body_id not in node_ids:
                        raise ValueError(
                            f"Node {node.id!r} loop_body references unknown node: {body_id!r}"
                        )

        return self

    def node_by_id(self, node_id: str) -> Node:
        for n in self.nodes:
            if n.id == node_id:
                return n
        raise KeyError(node_id)

    def nodes_by_kind(self, kind: NodeKind) -> list[Node]:
        return [n for n in self.nodes if n.kind is kind]


def load_pipeline(path: str | Path) -> Pipeline:
    """Load and validate a pipeline.yaml file."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Pipeline.model_validate(raw)
