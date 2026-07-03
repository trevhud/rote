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

import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ───────── Data-flow input references ─────────
#
# A node declares where its runtime inputs come from via ``inputs:``,
# a mapping of parameter name → source reference. The grammar is
# deliberately tiny — four forms, no expression language:
#
#   pipeline.input               the whole pipeline input payload
#   pipeline.input.<field>       one top-level field of the pipeline input
#   <node_id>.output             the whole output of an upstream node
#   <node_id>.output.<field>     one top-level field of an upstream node's output
#
# Anything fancier (aggregation, arithmetic, deep paths) belongs in an
# extracted pure_function node, not in the reference syntax.

_INPUT_REF_RE = re.compile(
    r"^(?:pipeline\.input|(?P<node>[A-Za-z_][A-Za-z0-9_]*)\.output)"
    r"(?:\.(?P<field>[A-Za-z_][A-Za-z0-9_]*))?$"
)

# ───────── Identifier / code-safety constraints ─────────
#
# Several string fields are interpolated *verbatim* into emitted source
# code (Python def names, Temporal signal-handler methods, `from … import`
# targets) and into emitted filenames (``signatures/<id>.py``,
# ``extracted/<id>.ts``). Without a charset constraint a crafted
# pipeline.yaml can inject code or traverse the filesystem at emit time —
# so these fields are pinned to a safe identifier shape at the IR
# boundary, which every adapter inherits. See ``Node.id`` / ``Node.signal``
# / ``Node.impl`` validators below.
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# A path segment in an ``impl``/``signature`` reference (the part before
# ``:``). Directory and filename segments allow ``.`` and ``-`` but never
# ``..`` (traversal) or path separators inside a segment.
_PATH_SEGMENT_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def _validate_impl_ref(value: str, field_name: str) -> str:
    """Validate a ``'relative/path.py:symbol'`` reference.

    Both halves reach emitted source code: the trailing symbol is spliced
    in as a ``from … import <symbol>`` target and call site, and the
    module component (last path segment, ``.py`` stripped) becomes an
    import module name. Constrain them to safe identifiers, and forbid
    absolute paths and ``..`` traversal in the path half.
    """
    if ":" not in value:
        raise ValueError(f"{field_name} {value!r} must be of the form 'path/to/file.py:symbol'")
    file_part, symbol = value.split(":", 1)
    if not _IDENTIFIER_RE.fullmatch(symbol):
        raise ValueError(
            f"{field_name} symbol {symbol!r} must be a valid identifier "
            f"(letters, digits, underscore; not starting with a digit)"
        )
    if not file_part or file_part.startswith("/"):
        raise ValueError(f"{field_name} path {file_part!r} must be a non-absolute relative path")
    segments = file_part.split("/")
    for seg in segments:
        if seg in ("", "..") or not _PATH_SEGMENT_RE.fullmatch(seg):
            raise ValueError(
                f"{field_name} path {file_part!r} has an unsafe segment {seg!r} "
                f"(no '..', no empty segments, no path-breaking characters)"
            )
    module_name = segments[-1].removesuffix(".py")
    if not _IDENTIFIER_RE.fullmatch(module_name):
        raise ValueError(
            f"{field_name} module name {module_name!r} (from {file_part!r}) "
            f"must be a valid identifier"
        )
    return value


@dataclass(frozen=True)
class InputRef:
    """A parsed ``inputs:`` source reference.

    ``node_id is None`` means the reference targets the pipeline input;
    otherwise it targets the named node's output. ``field`` is the
    optional top-level field selector.
    """

    node_id: str | None
    field: str | None


def parse_input_ref(ref: str) -> InputRef:
    """Parse an ``inputs:`` source reference string.

    This is the single source of truth for the reference grammar —
    the IR validator and every adapter resolve references through it.
    Raises ``ValueError`` for anything outside the four allowed forms.
    """
    m = _INPUT_REF_RE.fullmatch(ref.strip())
    if not m:
        raise ValueError(
            f"Invalid input reference {ref!r}. Allowed forms: "
            f"'pipeline.input', 'pipeline.input.<field>', "
            f"'<node_id>.output', '<node_id>.output.<field>'."
        )
    return InputRef(node_id=m.group("node"), field=m.group("field"))


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

    @field_validator("id")
    @classmethod
    def _validate_id(cls, v: str) -> str:
        """``id`` is emitted verbatim as a code identifier and a filename.

        Adapters splice it into ``async def {id}``, ``@activity.defn(name=…)``,
        and ``signatures/{id}.py`` / ``extracted/{id}.ts`` paths. Pinning it
        to a valid identifier closes both the code-injection and the
        filesystem-traversal vectors at the source.
        """
        if not _IDENTIFIER_RE.fullmatch(v):
            raise ValueError(
                f"Node id {v!r} must be a valid identifier "
                f"(letters, digits, underscore; not starting with a digit)"
            )
        return v

    @field_validator("signal")
    @classmethod
    def _validate_signal(cls, v: str | None) -> str | None:
        """``signal`` becomes a Temporal signal-handler method / DBOS topic."""
        if v is not None and not _IDENTIFIER_RE.fullmatch(v):
            raise ValueError(f"Node signal {v!r} must be a valid identifier")
        return v

    @field_validator("impl", "signature")
    @classmethod
    def _validate_impl_signature(cls, v: str | None) -> str | None:
        """``impl``/``signature`` halves reach emitted imports and call sites."""
        if v is None:
            return None
        return _validate_impl_ref(v, "impl/signature reference")

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
    inputs: dict[str, str] | None = Field(
        default=None,
        description=(
            "Data-flow bindings: parameter name → source reference. "
            "References use the grammar in parse_input_ref: 'pipeline.input', "
            "'pipeline.input.<field>', '<node_id>.output', or "
            "'<node_id>.output.<field>'. Complements ``input`` (which documents "
            "types); ``inputs`` binds where the values come from at runtime. "
            "Nodes without ``inputs`` receive an empty payload (back-compat). "
            "For loop_body sub-nodes the bindings describe what the parent "
            "loop passes per iteration — adapters do not resolve them at the "
            "top level."
        ),
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
    input_schema: dict[str, Any] | None = Field(
        default=None,
        description=(
            "JSON Schema for the pipeline input payload (the same schema "
            "Pydantic emits via model_json_schema()). Runtime-agnostic: "
            "adapters may derive input validation from it before the "
            "workflow starts. Optional for back-compat with pipelines "
            "that only declare required/optional field names."
        ),
    )


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

        # inputs references must parse and point at real sources.
        # Pipeline-input field names are checked against the declared
        # contract (required + optional + input_schema properties) when
        # one exists; a pipeline with an empty contract skips the check.
        declared_fields = set(self.input.required) | set(self.input.optional)
        if self.input.input_schema is not None:
            props = self.input.input_schema.get("properties")
            if isinstance(props, dict):
                declared_fields |= set(props.keys())

        for node in self.nodes:
            if not node.inputs:
                continue
            for param, ref in node.inputs.items():
                try:
                    parsed = parse_input_ref(ref)
                except ValueError as e:
                    raise ValueError(f"Node {node.id!r} input {param!r}: {e}") from e
                if parsed.node_id is None:
                    if parsed.field and declared_fields and parsed.field not in declared_fields:
                        raise ValueError(
                            f"Node {node.id!r} input {param!r} references pipeline input "
                            f"field {parsed.field!r}, which is not declared in the "
                            f"pipeline's input contract"
                        )
                else:
                    if parsed.node_id not in node_ids:
                        raise ValueError(
                            f"Node {node.id!r} input {param!r} references unknown "
                            f"node: {parsed.node_id!r}"
                        )
                    if parsed.node_id == node.id:
                        raise ValueError(
                            f"Node {node.id!r} input {param!r} references its own output"
                        )

        return self

    def node_by_id(self, node_id: str) -> Node:
        for n in self.nodes:
            if n.id == node_id:
                return n
        raise KeyError(node_id)

    def nodes_by_kind(self, kind: NodeKind) -> list[Node]:
        return [n for n in self.nodes if n.kind is kind]

    @property
    def requires_durable_execution(self) -> bool:
        """Whether this pipeline needs a durable-execution runtime.

        Derived, not stored — the IR stays runtime-agnostic; this just
        names a property the DAG already has. A pipeline that parks on a
        human approval (``hitl_gate``) cannot run as a plain in-process
        script: the park must survive process restarts, which requires
        checkpointed state. Adapters without a durable parking primitive
        refuse such pipelines at emit time; durable runtimes may use it
        for warnings or ignore it.
        """
        return any(n.kind is NodeKind.HITL_GATE for n in self.nodes)


def load_pipeline(path: str | Path) -> Pipeline:
    """Load and validate a pipeline.yaml file."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Pipeline.model_validate(raw)
