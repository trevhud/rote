"""Intermediate representation for compiled skills.

The IR is a runtime-agnostic DAG of typed nodes. Adapters
(``rote/adapters/temporal.py``, etc.) consume an :class:`Pipeline` and emit
runnable code for a specific durable execution engine.

The schema is intentionally driven by what real complex skills (BDR
outreach is the canonical example) need to express. Adding fields here
should always be motivated by a concrete skill that needs them, not
speculation.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

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

# An LLM endpoint override (``LLMSignature.base_url``). Reaches emitted
# source as a string literal (always via json.dumps), so the shape check
# here is belt-and-braces: require an http(s) scheme and standard URL
# characters only — no whitespace, quotes, or backslashes.
_BASE_URL_RE = re.compile(r"^https?://[A-Za-z0-9._~:/?#\[\]@!$&()*+,;=%-]+$")

# An MCP tool name (``MCPBinding.tool``). Reaches emitted source as a string
# literal (always via json.dumps), so this is belt-and-braces: MCP tool
# names are letters/digits with ``_ . - /`` (namespacing) and nothing that
# could break out of a string literal.
_MCP_TOOL_RE = re.compile(r"^[A-Za-z0-9_./-]+$")

# A pipeline name (``Pipeline.name``). Reaches emitted code two ways: verbatim
# in docstrings / block comments / string literals, AND through
# ``_to_pascal_case`` to form workflow / class / queue names. The pascal-case
# split only understands ``-`` and ``_`` as word separators, so the name must
# be an identifier-safe kebab/snake string starting with a letter — anything
# else either breaks out of the emitted syntax or produces an invalid class
# name. See ``Pipeline._validate_name``.
_PIPELINE_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")

# A pipeline version (``Pipeline.version``). Reaches emitted docstrings / JSDoc
# verbatim, so restrict it to the characters a semver-ish version needs and
# nothing that could close a docstring (``\"\"\"``), a block comment (``*/``),
# or inject a newline. See ``Pipeline._validate_version``.
#: A ``notify.message_template`` placeholder: ``{name}``, one key of the
#: node's ``inputs:``. The compiled skill's driver fills it with this exact
#: pattern (``rote.skill_target._driver._GATE_PLACEHOLDER``, pinned by a
#: parity test). No other runtime sends the message.
GATE_MESSAGE_PLACEHOLDER = r"\{([A-Za-z_][A-Za-z0-9_]*)\}"
_GATE_MESSAGE_PLACEHOLDER_RE = re.compile(GATE_MESSAGE_PLACEHOLDER)

_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")


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


def is_field_name(name: str) -> bool:
    """Whether ``name`` can be a pipeline input field: an identifier, the
    only shape a ``pipeline.input.<field>`` reference can bind."""
    return _IDENTIFIER_RE.fullmatch(name) is not None


@dataclass(frozen=True)
class InputRef:
    """A parsed ``inputs:`` source reference.

    ``node_id is None`` means the reference targets the pipeline input;
    otherwise it targets the named node's output. ``field`` is the
    optional top-level field selector.
    """

    node_id: str | None
    field: str | None


def _closed_object_properties(schema: Any) -> dict[str, Any] | None:
    """A JSON Schema's properties when it is an object that admits no other key
    (``additionalProperties: false``), else None: an open schema lists only
    some of its keys."""
    if not isinstance(schema, dict) or schema.get("additionalProperties") is not False:
        return None
    props = schema.get("properties")
    return props if isinstance(props, dict) else {}


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


def _schema_patterns(schema: Any) -> list[str]:
    """Every ``pattern`` a JSON Schema states, read only where a keyword holds a schema
    (so a property *named* ``pattern``, or a ``default`` holding one, is not a pattern)."""
    if not isinstance(schema, dict):
        return []
    found = [schema["pattern"]] if isinstance(schema.get("pattern"), str) else []
    children: list[Any] = [schema.get("items"), schema.get("additionalProperties")]
    for key in ("properties", "$defs"):
        if isinstance(schema.get(key), dict):
            children += schema[key].values()
    for key in ("anyOf", "oneOf", "allOf"):
        if isinstance(schema.get(key), list):
            children += schema[key]
    for child in children:
        found += _schema_patterns(child)
    return found


def check_schema_patterns(schema: dict[str, Any] | None) -> dict[str, Any] | None:
    """Refuse a ``pattern`` Python's ``re`` cannot compile, at IR load.

    The validator every runtime checks answers with (``schema_errors`` in
    ``rote.inference._runtime_helper``) would otherwise refuse it only when
    the first answer arrives. Whether JavaScript's engine compiles it too
    cannot be checked here without node; the TS validator refuses one it
    cannot compile with the same kind of named error.
    """
    for pattern in _schema_patterns(schema):
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ValueError(
                f"schema pattern {pattern!r} is not a regular expression Python can compile ({exc})"
            ) from exc
    return schema


class NodeKind(StrEnum):
    """The five kinds a compiled step can be classified as.

    Every node in a compiled pipeline is exactly one kind. Adapters
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

    # Both optional: a skill rarely names where its approval goes, the compile
    # must not invent one, and no runtime sends to them (the compiled skill
    # reaches the person through its agent, with the message alone).
    channel: str | None = Field(default=None, description="slack | email | webhook")
    target: str | None = Field(
        default=None, description="Channel name, email address, or webhook URL"
    )
    message_template: str | None = Field(
        default=None,
        description=(
            "Message for the person. Each {name} placeholder is a key of the "
            "gate's inputs: (validated) and is filled with that input's value."
        ),
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
        description="LLM vendor identifier. 'anthropic' | 'openai' | 'workers-ai'.",
    )
    model: str | None = Field(
        default=None,
        description="Vendor-specific model id. None = adapter chooses default.",
    )
    base_url: str | None = Field(
        default=None,
        description=(
            "Custom API endpoint. With client 'openai' this reaches every "
            "OpenAI-compatible server (Ollama, vLLM, Azure, gateways/proxies). "
            "None = the vendor SDK's default endpoint. Emitted code also "
            "honors a ROTE_BASE_URL_<NODE_ID> environment override at runtime."
        ),
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature; None = vendor default.",
    )

    @field_validator("input_schema", "output_schema")
    @classmethod
    def _validate_patterns(cls, v: dict[str, Any]) -> dict[str, Any]:
        check_schema_patterns(v)
        return v

    @field_validator("client")
    @classmethod
    def _validate_client(cls, v: str) -> str:
        allowed = {"anthropic", "openai", "workers-ai"}
        if v not in allowed:
            raise ValueError(f"client must be one of {sorted(allowed)}, got {v!r}")
        return v

    @field_validator("base_url")
    @classmethod
    def _validate_base_url(cls, v: str | None) -> str | None:
        """``base_url`` is spliced into emitted source (via json.dumps, but
        keep the IR boundary strict per the charset-constraint invariant)."""
        if v is None:
            return None
        if not _BASE_URL_RE.fullmatch(v):
            raise ValueError(
                f"base_url {v!r} must be an http(s) URL with no whitespace, quotes, or backslashes"
            )
        return v


class SourceRef(BaseModel):
    """Provenance: which SKILL.md section a node was derived from.

    ``section`` is the heading text (any level, without the ``#``
    markers) of the source skill section — written by the compiler
    agent, which knows the mapping. ``content_hash`` is the sha256 of
    that section's text; it's stamped by rote tooling (agents can't
    compute hashes) and normally lives in the ``provenance.json``
    sidecar rather than here — see :mod:`rote.skill_source`.

    Pure metadata: adapters ignore it, and it is excluded from the
    pipeline identity hash so annotating provenance never re-versions
    an emitted workflow type.
    """

    model_config = ConfigDict(extra="forbid")

    section: str = Field(description="Heading text of the source SKILL.md section")
    content_hash: str | None = Field(
        default=None,
        description="sha256 hex of the section text at compilation time (tool-stamped)",
    )

    @field_validator("content_hash")
    @classmethod
    def _validate_content_hash(cls, v: str | None) -> str | None:
        if v is not None and not re.fullmatch(r"[0-9a-f]{64}", v):
            raise ValueError(f"content_hash {v!r} must be a lowercase sha256 hex digest")
        return v


class MCPLiteral(BaseModel):
    """A fixed value for one MCP tool argument: ``{literal: "BDR do not contact"}``.

    The mapping form keeps it distinct from a reference: a bare string in
    ``mcp.args`` always names a payload key or a constant, so a literal
    can never be misread as one, or the reverse. Only scalars: a string, a
    finite number, or a boolean. Strict types, so ``"3"`` stays a string
    and ``true`` stays a boolean instead of being coerced. A list or an
    object belongs in the node's ``constants``, which ``mcp.args`` can
    already name, and ``null`` is better said by leaving the argument out.

    The value reaches emitted code, so emitters render it only as an
    escaped quoted literal (``json.dumps`` / a Python string literal),
    never spliced (invariant #7).
    """

    model_config = ConfigDict(extra="forbid")

    literal: StrictBool | StrictInt | StrictFloat | StrictStr

    @field_validator("literal")
    @classmethod
    def _validate_literal(cls, v: bool | int | float | str) -> bool | int | float | str:
        if isinstance(v, float) and not math.isfinite(v):
            raise ValueError(f"mcp.args literal {v!r} must be a finite number")
        return v


class MCPBinding(BaseModel):
    """Binding from an ``external_call`` node to an MCP tool.

    When present, the ``mcp`` backend emits a call to ``tool`` on the MCP
    server ``server`` over Streamable HTTP — a working step body, not a
    ``NotImplementedError`` stub — while the ``api`` backend falls back to
    the node's ``impl`` (a direct vendor-SDK call). A node may carry both;
    the backend chooses which is emitted.

    The shape mirrors the official MCP registry's ``remotes[]`` entries
    (``url`` + ``transport``) so a resolved registry entry maps straight
    onto it. ``server`` is a *logical* name. At run time the endpoint is
    the ``ROTE_MCP_<SERVER>_URL`` env var if set, else the rote registry
    entry for ``server``, else ``url`` (the endpoint recorded at
    compilation time).
    """

    model_config = ConfigDict(extra="forbid")

    server: str = Field(
        description=(
            "Logical MCP server name; at run time ROTE_MCP_<SERVER>_URL, then the "
            "rote registry, then url"
        ),
    )
    tool: str = Field(
        description="MCP tool name to invoke, e.g. 'hubspot_get_contact_emails'",
    )
    args: dict[str, str | MCPLiteral] | None = Field(
        default=None,
        description=(
            "Optional map: MCP tool-argument name → where its value comes from. "
            "A string names a key in the node's runtime payload, or one of its "
            "constants; {literal: <string, number or boolean>} is a fixed value. "
            "Omit (null) to pass the payload through unchanged; {} calls the tool "
            "with no arguments."
        ),
    )
    url: str | None = Field(
        default=None,
        description=(
            "Streamable-HTTP endpoint recorded at compilation time; used when neither "
            "ROTE_MCP_<SERVER>_URL nor the rote registry names one"
        ),
    )
    transport: Literal["streamable-http", "sse"] = Field(default="streamable-http")

    @field_validator("server")
    @classmethod
    def _validate_server(cls, v: str) -> str:
        """``server`` becomes part of an env-var name and an auth-park signal."""
        if not _IDENTIFIER_RE.fullmatch(v):
            raise ValueError(f"mcp.server {v!r} must be a valid identifier")
        return v

    @field_validator("tool")
    @classmethod
    def _validate_tool(cls, v: str) -> str:
        """``tool`` is emitted as a string literal; charset-limit it anyway."""
        if not _MCP_TOOL_RE.fullmatch(v):
            raise ValueError(f"mcp.tool {v!r} must contain only letters, digits, and _./-")
        return v

    @field_validator("url")
    @classmethod
    def _validate_url(cls, v: str | None) -> str | None:
        if v is not None and not _BASE_URL_RE.fullmatch(v):
            raise ValueError(f"mcp.url {v!r} must be an http(s) URL")
        return v

    @field_validator("args")
    @classmethod
    def _validate_args(
        cls, v: dict[str, str | MCPLiteral] | None
    ) -> dict[str, str | MCPLiteral] | None:
        """Keys and references reach emitted dict literals and payload lookups,
        so they are identifiers. A literal is validated by :class:`MCPLiteral`."""
        if v is None:
            return None
        for arg_name, source in v.items():
            if not _IDENTIFIER_RE.fullmatch(arg_name):
                raise ValueError(f"mcp.args key {arg_name!r} must be a valid identifier")
            if isinstance(source, str) and not _IDENTIFIER_RE.fullmatch(source):
                raise ValueError(
                    f"mcp.args value {source!r} must be a valid identifier (a payload key "
                    f"or constant name); write {{literal: {source!r}}} for a fixed value"
                )
        return v


#: Node kinds whose ``constants`` are passed to their ``impl`` as keyword
#: arguments. A judge's constants are not inputs to its typed signature,
#: and an MCP-bound step forwards one only when ``mcp.args`` names it.
CONSTANT_ARGUMENT_KINDS = frozenset({NodeKind.PURE_FUNCTION, NodeKind.EXTERNAL_CALL})


class Node(BaseModel):
    """A single step in the compiled pipeline.

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
    source: SourceRef | None = Field(
        default=None,
        description=(
            "Provenance: the SKILL.md section this node was derived from. "
            "Enables incremental re-compilation (only nodes whose source "
            "section changed get re-derived). Metadata only — adapters "
            "ignore it and it never affects the pipeline identity hash."
        ),
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

    @field_validator("tools")
    @classmethod
    def _validate_tools(cls, v: list[str] | None) -> list[str] | None:
        """Each tool name is spliced verbatim into emitted docstrings / block
        comments (Python ``#``/``\"\"\"`` and TS ``/* */``). Pin them to the
        MCP tool-name charset so a crafted name can't close a comment and
        inject code."""
        if v is None:
            return None
        for tool in v:
            if not _MCP_TOOL_RE.fullmatch(tool):
                raise ValueError(f"tool name {tool!r} must contain only letters, digits, and _./-")
        return v

    @field_validator("tool_servers")
    @classmethod
    def _validate_tool_servers(cls, v: dict[str, str] | None) -> dict[str, str] | None:
        """Keys are tool names, values are server names — and BOTH reach
        emitted source: the server becomes a `ROTE_MCP_<SERVER>_URL` lookup
        and a literal in the emitted tool map. Same charsets their
        single-binding equivalents already enforce (invariant #7)."""
        if v is None:
            return None
        for tool, server in v.items():
            if not _MCP_TOOL_RE.fullmatch(tool):
                raise ValueError(
                    f"tool_servers key {tool!r} must contain only letters, digits, and _./-"
                )
            if not _IDENTIFIER_RE.fullmatch(server):
                raise ValueError(f"tool_servers[{tool!r}] server {server!r} must be an identifier")
        return v

    @field_validator("input_schema", "output_schema")
    @classmethod
    def _validate_schema_patterns(cls, v: dict[str, Any] | None) -> dict[str, Any] | None:
        return check_schema_patterns(v)

    @field_validator("constants")
    @classmethod
    def _validate_constants(cls, v: dict[str, Any] | None) -> dict[str, Any] | None:
        """Constant *keys* are emitted verbatim into docstrings / comments, so
        pin them to identifiers. Values stay arbitrary — they're escaped at
        emission (repr into Python docstrings, ``safe_block_comment_line`` into
        TS JSDoc)."""
        if v is None:
            return None
        for key in v:
            if not _IDENTIFIER_RE.fullmatch(key):
                raise ValueError(f"constant key {key!r} must be a valid identifier")
        return v

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
    input_schema: dict[str, Any] | None = Field(
        default=None,
        description=(
            "JSON Schema for this node's input payload. Runtime-agnostic "
            "data contract (same convention as PipelineInput.input_schema); "
            "typically populated from observed baseline traffic "
            "(rote.probe) or by the compiler. Adapters may render it into "
            "stub documentation and typed signatures."
        ),
    )
    output_schema: dict[str, Any] | None = Field(
        default=None,
        description=(
            "JSON Schema for this node's result. Same provenance and use "
            "as input_schema — evidence from real payloads beats guessed "
            "types."
        ),
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
    mcp: MCPBinding | None = Field(
        default=None,
        description=(
            "external_call only: binding to an MCP tool. When set, the mcp "
            "backend emits a working Streamable-HTTP call; the api backend "
            "uses ``impl`` instead. A node may carry both."
        ),
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
    tool_servers: dict[str, str] | None = Field(
        default=None,
        description=(
            "agent_loop only: tool name → the MCP server that provides it. "
            "``tools`` names tools WITHOUT a server (one loop may reach several, "
            "and an MCPBinding is a single server/tool pair), which leaves the "
            "pipeline's MCP requirements incomplete: a loop-only pipeline would "
            "report no required servers at all. This closes that gap. Keys must "
            "appear in ``tools``; a tool omitted here is still resolved at run "
            "time (the local registry, or ROTE_MCP_SERVERS), so the field is "
            "additive and partial maps are valid."
        ),
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
    def _validate_constants_do_not_shadow_inputs(self) -> Self:
        """An impl receives its constants and its inputs as one set of
        keyword arguments, so a shared name would make one silently win.
        Other kinds' constants never reach a call, so they may reuse a name."""
        if self.kind not in CONSTANT_ARGUMENT_KINDS:
            return self
        clash = sorted(set(self.constants or {}) & set(self.inputs or {}))
        if clash:
            raise ValueError(
                f"node {self.id!r}: {', '.join(clash)} is both a constant and an input; rename one"
            )
        return self

    @model_validator(mode="after")
    def _validate_gate_shows_what_it_binds(self) -> Self:
        """A gate shows the person its bound ``inputs:`` and nothing else.
        So what it says it shows, the keys its ``input:`` documents and the
        ``{name}`` placeholders of its message, must each be bound, or the
        person reads ``{review_count}`` and has nothing to decide on. Every
        committed compile that broke this documented the data and bound none."""
        if self.kind is not NodeKind.HITL_GATE:
            return self
        bound = set(self.inputs or {})
        template = (self.notify.message_template if self.notify else None) or ""
        documented = sorted(set(self.input) - bound)
        placeholders = sorted(set(_GATE_MESSAGE_PLACEHOLDER_RE.findall(template)) - bound)
        problems = ([f"input: documents {', '.join(documented)}"] if documented else []) + (
            [f"notify.message_template uses {', '.join(f'{{{p}}}' for p in placeholders)}"]
            if placeholders
            else []
        )
        if problems:
            raise ValueError(
                f"hitl_gate {self.id!r} shows the person only its inputs: "
                f"({', '.join(sorted(bound)) or 'none bound'}), "
                f"but its {' and its '.join(problems)}. "
                "Bind each in inputs: to the data the person decides on, or remove it"
            )
        return self

    @model_validator(mode="after")
    def _validate_kind_specific_fields(self) -> Self:
        """Each node kind has required fields. Enforce them here."""
        kind = self.kind
        missing: list[str] = []

        if kind is NodeKind.PURE_FUNCTION:
            if not self.impl:
                missing.append("impl")
        elif kind is NodeKind.EXTERNAL_CALL:
            # An external_call needs a body: either a direct-API impl or an
            # MCP binding (or both — the backend picks which is emitted).
            if not self.impl and self.mcp is None:
                missing.append("impl or mcp")
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

        # An MCP binding is only meaningful on an external_call (it's how that
        # node reaches a vendor). Reject it elsewhere so the field can't drift.
        if self.mcp is not None and kind is not NodeKind.EXTERNAL_CALL:
            raise ValueError(f"Node {self.id!r}: mcp is only allowed on external_call nodes")

        # tool_servers resolves `tools`, so it is meaningless without them —
        # and a key naming a tool the loop cannot call is a typo, not a
        # harmless extra. Catching it here beats a silently unbound tool.
        if self.tool_servers:
            if kind is not NodeKind.AGENT_LOOP:
                raise ValueError(
                    f"Node {self.id!r}: tool_servers is only allowed on agent_loop nodes"
                )
            unknown = sorted(set(self.tool_servers) - set(self.tools or ()))
            if unknown:
                raise ValueError(
                    f"Node {self.id!r}: tool_servers names tool(s) absent from tools: "
                    f"{', '.join(unknown)}"
                )

        return self

    @property
    def output_keys(self) -> frozenset[str]:
        """The field names of this node's result: ``output`` keys or schema properties.

        For a ``hitl_gate`` these are the keys of the payload a person
        sends. An ``output`` given as a single type name declares none.
        """
        keys: set[str] = set()
        if isinstance(self.output, dict):
            keys |= set(self.output)
        if self.output_schema:
            keys |= set((self.output_schema.get("properties") or {}).keys())
        return frozenset(keys)


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

    @field_validator("required", "optional")
    @classmethod
    def _validate_field_names(cls, v: list[str]) -> list[str]:
        """Input field names reach emitted code (the entry check, payload
        reads) and, since a sample input steers them, come from data rote
        did not write. Same identifier shape as ``inputs:`` field
        references (invariant #7), so a name can always be bound."""
        for name in v:
            if not is_field_name(name):
                raise ValueError(
                    f"pipeline input field name {name!r} must be an identifier "
                    f"(letters, digits, underscores; not starting with a digit)"
                )
        return v

    @field_validator("input_schema")
    @classmethod
    def _validate_schema_required(cls, v: dict[str, Any] | None) -> dict[str, Any] | None:
        """The schema's ``required`` joins ``required`` in
        :attr:`Pipeline.required_input_fields`, which reaches emitted code
        and a compiled skill's SKILL.md, so it gets the same identifier rule
        (invariant #7): a name like ``"x\\n- DECIDE: ..."`` would otherwise
        write an instruction into the skill an agent loads. Its patterns are
        checked too (:func:`check_schema_patterns`)."""
        check_schema_patterns(v)
        required = (v or {}).get("required")
        if required is None:
            return v
        if not isinstance(required, list):
            raise ValueError("pipeline input_schema.required must be a list of field names")
        for name in required:
            if not isinstance(name, str) or not is_field_name(name):
                raise ValueError(
                    f"pipeline input_schema.required entry {name!r} must be an identifier "
                    f"(letters, digits, underscores; not starting with a digit)"
                )
        return v

    @model_validator(mode="after")
    def _validate_required_and_optional_are_disjoint(self) -> Self:
        """A field cannot be both: emitted workflows would have to pick one."""
        both = sorted(set(self.required) & set(self.optional))
        if both:
            raise ValueError(
                f"pipeline input field(s) {', '.join(both)} are both required and optional"
            )
        return self


class Pipeline(BaseModel):
    """A compiled skill, ready to be emitted into a runtime adapter."""

    model_config = ConfigDict(extra="forbid")

    name: str
    version: str = "0.1.0"
    source_skill: str | None = Field(
        default=None,
        description="Path to the source skill bundle this was compiled from",
    )
    description: str = ""
    config: PipelineConfig = Field(default_factory=PipelineConfig)
    input: PipelineInput
    nodes: list[Node]
    edges: list[Edge]
    entry_nodes: list[str] = Field(default_factory=list)
    exit_nodes: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        """``name`` reaches emitted code verbatim (docstrings, string literals)
        and via ``_to_pascal_case`` (workflow / class / queue names), so it
        must be an identifier-safe kebab/snake name starting with a letter."""
        if not _PIPELINE_NAME_RE.fullmatch(v):
            raise ValueError(
                f"Pipeline name {v!r} must start with a letter and contain only "
                f"letters, digits, hyphens, and underscores"
            )
        return v

    @field_validator("version")
    @classmethod
    def _validate_version(cls, v: str) -> str:
        """``version`` reaches emitted docstrings / JSDoc verbatim; constrain it
        so it cannot close a docstring, close a block comment, or inject a
        newline."""
        if not _VERSION_RE.fullmatch(v):
            raise ValueError(
                f"Pipeline version {v!r} must start with a letter or digit and "
                f"contain only letters, digits, and .+-_"
            )
        return v

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

        # loop_body references must also be valid node IDs, of a kind a loop
        # can call per iteration. A hitl_gate cannot be one: a durable human
        # pause is not a tool call. And a sub-node runs only when the loop's
        # agent chooses to call it, so it cannot be mandatory (invariant #4).
        by_id = {n.id: n for n in self.nodes}
        callable_kinds = (NodeKind.PURE_FUNCTION, NodeKind.EXTERNAL_CALL, NodeKind.LLM_JUDGE)
        for node in self.nodes:
            if node.loop_body:
                for body_id in node.loop_body:
                    if body_id not in node_ids:
                        raise ValueError(
                            f"Node {node.id!r} loop_body references unknown node: {body_id!r}"
                        )
                    sub = by_id[body_id]
                    if sub.kind not in callable_kinds:
                        raise ValueError(
                            f"Node {node.id!r} loop_body has {body_id!r}, a {sub.kind.value}: a "
                            "loop calls only pure_function, external_call and llm_judge nodes "
                            "per iteration"
                        )
                    if sub.mandatory:
                        raise ValueError(
                            f"Node {node.id!r} loop_body has {body_id!r}, marked mandatory: the "
                            "loop's agent decides whether to call it, so it cannot be mandatory"
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

    @model_validator(mode="after")
    def _validate_judge_input_bindings(self) -> Self:
        """A judge's ``inputs:`` must satisfy its ``signature_spec.input_schema``.

        Runs after DAG integrity (pydantic applies after-validators in
        definition order), so every reference here already resolves.

        Scope, and why:

        * Only ``llm_judge`` nodes with a ``signature_spec``. The legacy
          ``signature:`` path points at a hand-maintained Python class the
          IR cannot see, and no other kind validates its payload against a
          schema (``Node.input_schema`` is observed documentation).
        * Not ``loop_body`` sub-nodes. Their parent loop calls them with
          model-chosen arguments; their ``inputs:`` describe, they do not
          bind.
        """
        loop_body_ids = {sub for n in self.nodes for sub in (n.loop_body or ())}
        errors = [
            error
            for node in self.nodes
            if node.kind is NodeKind.LLM_JUDGE
            and node.signature_spec is not None
            and node.id not in loop_body_ids
            and (error := _judge_binding_error(node, self)) is not None
        ]
        if errors:
            raise ValueError("\n".join(errors))
        return self

    @model_validator(mode="after")
    def _validate_input_reachability(self) -> Self:
        """Every node-output input must be bound before its node runs.

        Adapters emit the DAG wave by wave and bind each node's result as
        they go, so an input naming a node in the same or a later wave, or
        a ``loop_body`` sub-node (which the loop calls per iteration and
        which never binds a top-level result), would emit code reading an
        undefined variable. This used to be checked only at emit time, in
        every adapter, after the compile had already declared success: a
        real bdr-outreach compile paid $4.46 for an IR whose loop took its
        own body sub-node's output as an input, then failed to emit. Here
        it is a validation error the compile's repair loop sends back.

        Loop-body sub-nodes' own ``inputs:`` are not checked: their parent
        loop calls them with model-chosen arguments, so those bindings
        describe and do not bind (same scope rule as the judge check).
        """
        waves = self.execution_waves()
        wave_of = {n.id: i for i, wave in enumerate(waves) for n in wave}
        owners: dict[str, list[str]] = {}
        for n in self.nodes:
            for sub in n.loop_body or ():
                owners.setdefault(sub, []).append(n.id)
        errors: list[str] = []
        for node in self.nodes:
            if node.id in owners or not node.inputs:
                continue
            for param, ref in node.inputs.items():
                source = parse_input_ref(ref).node_id
                if source is None:
                    continue
                head = (
                    f"Node {node.id!r} input {param!r} references {ref!r}, but node "
                    f"{source!r} has no result available at this point in the workflow"
                )
                if source in owners:
                    if node.id in owners[source]:
                        errors.append(
                            f"{head}: it is in {node.id!r}'s own loop_body. The loop "
                            f"calls its body sub-nodes on each iteration, so no "
                            f"sub-node result exists before the loop starts. Remove "
                            f"this input; what the body produces comes back as the "
                            f"loop's own output."
                        )
                    else:
                        errors.append(
                            f"{head}: it is a loop_body sub-node of "
                            f"{', '.join(repr(o) for o in owners[source])}, which never "
                            f"binds a top-level result. Reference the loop's output "
                            f"instead."
                        )
                elif wave_of[source] >= wave_of[node.id]:
                    errors.append(
                        f"{head}: it does not run before {node.id!r}. Add an edge from "
                        f"{source!r} to {node.id!r} (directly or through other nodes) "
                        f"so it finishes first."
                    )
        if errors:
            raise ValueError("\n".join(errors))
        return self

    @model_validator(mode="after")
    def _validate_input_fields_declared(self) -> Self:
        """An input reading ``<node>.output.<field>`` names a field that node declares.

        Where a node lists its output keys (a non-empty ``output`` mapping),
        a reference to any other key reads a field the result never has: the
        run fails with a KeyError (Python) or carries ``undefined`` on (TS).
        A System-1 review found a gate bound to ``render_digest.output.items``
        of a node whose output had no ``items``, and the IR accepted it.
        An empty mapping or a type name declares no keys, so it is not
        checked, and neither is an MCP node's ``output`` (a guess at the
        tool's result) unless its ``output_schema`` closes the object.
        Loop-body sub-nodes' inputs describe rather than bind, the same
        exemption as the reachability rule above."""
        by_id = {n.id: n for n in self.nodes}
        owned = {sub for n in self.nodes for sub in n.loop_body or ()}
        errors: list[str] = []
        for node in self.nodes:
            if node.id in owned:
                continue
            for param, ref in (node.inputs or {}).items():
                parsed = parse_input_ref(ref)
                source = by_id.get(parsed.node_id) if parsed.node_id is not None else None
                if source is None or parsed.field is None:
                    continue
                declared: dict[str, Any] | str = source.output
                if source.mcp is not None:
                    # An MCP result's keys are a guess (inferred from a recording,
                    # an open outputSchema, the interface's reading of the skill),
                    # so `output` is not a complete list: a read of another key
                    # may be right. Only a schema that closes the object lists them.
                    closed = _closed_object_properties(source.output_schema)
                    if closed is None:
                        continue
                    declared = closed
                if isinstance(declared, dict) and declared and parsed.field not in declared:
                    errors.append(
                        f"Node {node.id!r} input {param!r} references {ref!r}, but "
                        f"{source.id!r} declares only output keys {sorted(declared)}. "
                        f"Read one of those, or add {parsed.field!r} to its output."
                    )
        if errors:
            raise ValueError("\n".join(errors))
        return self

    def execution_waves(self) -> list[list[Node]]:
        """Topologically sort the top-level nodes into parallel waves.

        A wave is a set of nodes whose dependencies have all finished by
        the time it starts; nodes within a wave may run concurrently, with
        one exception every adapter holds to: a ``hitl_gate`` waits only
        after every other node in its wave has finished. A gate with no
        dependency on a wave's reads (one that takes only the pipeline
        input) therefore still parks after them, so a person is never asked
        before the data the same wave pulls in has arrived. Nothing in the
        IR orders a gate after its wave siblings; the adapters do, and
        ``tests/test_gate_after_wave.py`` holds every gate-capable runtime
        to it.
        Nodes in another node's ``loop_body`` are excluded: they run
        inside their parent loop, not as workflow steps. Ties sort by id,
        so emission order is deterministic. Raises ``ValueError`` on a
        cycle.

        Lives on the IR, not in an adapter, because the reachability rule
        above is defined by it and every adapter emits from it: one
        definition, so "valid IR" and "emittable IR" cannot disagree.
        """
        nested = {sub for n in self.nodes for sub in (n.loop_body or ())}
        eligible = [n for n in self.nodes if n.id not in nested]
        eligible_ids = {n.id for n in eligible}
        in_degree = {n.id: 0 for n in eligible}
        for edge in self.edges:
            if edge.from_ in eligible_ids and edge.to in eligible_ids:
                in_degree[edge.to] += 1
        waves: list[list[Node]] = []
        remaining = set(in_degree)
        while remaining:
            wave_ids = sorted(nid for nid in remaining if in_degree[nid] == 0)
            if not wave_ids:
                raise ValueError(
                    f"Cycle detected in pipeline {self.name!r}; "
                    f"remaining nodes: {sorted(remaining)}"
                )
            waves.append([self.node_by_id(nid) for nid in wave_ids])
            for nid in wave_ids:
                remaining.remove(nid)
                for edge in self.edges:
                    if edge.from_ == nid and edge.to in eligible_ids:
                        in_degree[edge.to] -= 1
        return waves

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

    @property
    def optional_input_fields(self) -> frozenset[str]:
        """Pipeline input fields a run may omit.

        A field is optional when ``input.optional`` lists it, or when
        ``input.input_schema`` declares it as a property without listing it
        in the schema's ``required``. Declaring it required anywhere wins:
        the validator on :class:`PipelineInput` rejects a field that is
        listed both ways, so "wins" only settles the schema-only case.
        """
        schema = self.input.input_schema or {}
        schema_required = set(schema.get("required") or [])
        schema_optional = set(schema.get("properties") or {}) - schema_required
        declared_required = set(self.input.required) | schema_required
        return frozenset((set(self.input.optional) | schema_optional) - declared_required)

    @property
    def required_input_fields(self) -> tuple[str, ...]:
        """Pipeline input fields every run must supply, sorted.

        The declared ones (``input.required`` and the schema's
        ``required``) plus any field a node reads that nothing declares
        optional. Derived, never stored: emitted workflows check these on
        entry so a missing field fails naming itself, and they read every
        pipeline input field without raising, and an omitted optional one
        is simply left out of a step's payload.
        """
        schema = self.input.input_schema or {}
        fields = set(self.input.required) | set(schema.get("required") or [])
        optional = self.optional_input_fields
        for node in self.nodes:
            for ref in (node.inputs or {}).values():
                parsed = parse_input_ref(ref)
                if (
                    parsed.node_id is None
                    and parsed.field is not None
                    and parsed.field not in optional
                ):
                    fields.add(parsed.field)
        return tuple(sorted(fields))

    @property
    def needs_input_check(self) -> bool:
        """Whether emitted workflows check the pipeline input on entry.

        True when a field is required, or when any node reads a pipeline
        input field: a run started with no payload at all (or a non-object
        one) must fail naming the problem, not with a TypeError inside
        generated code.
        """
        if self.required_input_fields:
            return True
        return any(
            parse_input_ref(ref).node_id is None and parse_input_ref(ref).field is not None
            for node in self.nodes
            for ref in (node.inputs or {}).values()
        )

    @property
    def required_mcp_servers(self) -> dict[str, list[str]]:
        """Logical MCP server name → sorted ids of the nodes bound to it.

        Derived, not stored — the requirements manifest is a property the
        DAG already has, so persisting it would only let it drift. Includes
        loop-body sub-nodes: a binding anywhere in the pipeline means the
        server must be reachable (and authenticated) at run time.

        Two kinds of requirement count, because the IR declares MCP in two
        places: a node's ``mcp:`` binding, and an ``agent_loop``'s
        ``tool_servers``. Counting only the former made a pipeline whose
        only MCP usage is a loop report *zero* required servers — so
        ``analyze``/``emit``/``compile`` and the ``rote mcp login``
        recommendations all said there was nothing to authenticate, which
        is a wrong answer rather than a missing feature.

        A loop's tools with no resolved server contribute nothing here:
        there is no server name to report, and inventing one would be
        worse than admitting the gap (the emitted module says so at the
        call site instead).
        """
        by_server: dict[str, list[str]] = {}
        for n in self.nodes:
            if n.mcp is not None:
                by_server.setdefault(n.mcp.server, []).append(n.id)
            for server in sorted(set((n.tool_servers or {}).values())):
                by_server.setdefault(server, []).append(n.id)
        return {server: sorted(set(ids)) for server, ids in sorted(by_server.items())}


# ───────── Fan-out element + judge input contract ─────────


def fan_out_element_param(node: Node, pipeline: Pipeline) -> str:
    """Which ``inputs`` param carries the list a ``fan_out`` node fans over.

    Every other input is shared verbatim by all invocations. Which param
    is the list, in precedence order:

    1. the param bound to the source of an incoming ``fan_out: true``
       edge (the IR's explicit marker);
    2. else the param bound to a node with any incoming edge — inputs
       may also reference nodes *without* an edge (shared context, e.g.
       BDR's ``intel``), which is what makes "the only node-bound
       param" too naive;
    3. else the only node-bound param;
    4. else, when NO param is node-bound, the only param bound to a
       pipeline-input field that can hold a list
       (``invoice: pipeline.input.invoices``). A field the input schema
       declares with a non-array type is excluded, an undeclared or
       untyped one is not. Fanning over a list the caller passes in is a
       legitimate shape, and before this tier it crashed ``rote compile``
       after the agent had already been paid. It only applies when no
       node-bound param exists, so a pipeline-input list bound as shared
       context next to a fanned node output keeps resolving as before.

    Ambiguity after all four is an emit-time error, never a guess:
    dispatching over the wrong list would silently judge the wrong
    things.

    This lives in the IR module because the answer is a property of the
    IR, not of the target language: every adapter must fan over the
    *same* input or the same pipeline would mean different things on
    different runtimes, and the judge-binding validator
    (:meth:`Pipeline._validate_judge_input_bindings`) must agree with
    the emitters about which param is the element.
    """
    if not node.inputs:
        raise ValueError(f"fan_out node {node.id!r} has no inputs: nothing to fan over")
    node_bound = {
        param: parsed.node_id
        for param, ref in node.inputs.items()
        if (parsed := parse_input_ref(ref)).node_id is not None
    }
    edge_sources = {e.from_ for e in pipeline.edges if e.to == node.id}
    fan_edge_sources = {e.from_ for e in pipeline.edges if e.to == node.id and e.fan_out}

    # The tiers are tried narrowest-first, but the ORDER is provably
    # unobservable: fan_edge_sources ⊆ edge_sources by construction, so
    # matching(fan) ⊆ matching(edge), and a tier is accepted only when it
    # singles out exactly one param. If the broader tier singles one out,
    # it is necessarily the same one. (Mutation testing flags swapping
    # these as a surviving mutant — it is equivalent, not a test gap.)
    # The tiers still earn their place: they express which signal the
    # author gave, and tier 2 is what handles a marker pointing at a node
    # no input references.
    for sources in (fan_edge_sources, edge_sources):
        matching = sorted(p for p, src in node_bound.items() if src in sources)
        if len(matching) == 1:
            return matching[0]
    if not node_bound:
        input_lists = sorted(
            param
            for param, ref in node.inputs.items()
            if _pipeline_input_field_may_be_list(parse_input_ref(ref), pipeline)
        )
        if len(input_lists) == 1:
            return input_lists[0]
        raise ValueError(
            f"fan_out node {node.id!r}: cannot identify the element param: no input "
            f"is bound to an upstream node, and the pipeline-input fields that could "
            f"hold the list ({input_lists or 'none'}) don't single one out. Bind "
            f"exactly one param to the list (`<param>: pipeline.input.<list_field>`), "
            f"declare that field `type: array` in input.input_schema, and give any "
            f"other pipeline-input field a non-array type"
        )
    if len(node_bound) != 1:
        raise ValueError(
            f"fan_out node {node.id!r}: cannot identify the element param — "
            f"node-bound inputs {sorted(node_bound)} and incoming edges "
            f"{sorted(edge_sources)} don't single one out; mark the list edge "
            f"with `fan_out: true`"
        )
    return next(iter(node_bound))


def _pipeline_input_field_may_be_list(ref: InputRef, pipeline: Pipeline) -> bool:
    """Whether ``ref`` names a pipeline-input field a fan could iterate.

    Only ``pipeline.input.<field>`` qualifies (the whole input is an
    object). The input schema can rule a field out by giving it a
    non-array ``type``; it cannot rule one in that it does not describe,
    because ``input_schema`` is optional and most IRs leave it partial.
    """
    if ref.node_id is not None or ref.field is None:
        return False
    schema = pipeline.input.input_schema or {}
    props = schema.get("properties")
    field_schema = props.get(ref.field) if isinstance(props, dict) else None
    if not isinstance(field_schema, dict) or "type" not in field_schema:
        return True
    declared = field_schema["type"]
    types = declared if isinstance(declared, list) else [declared]
    return "array" in types


def _schema_root(schema: dict[str, Any]) -> dict[str, Any]:
    """The object a judge's input schema describes, after any root ``$ref``.

    Follows ``#/$defs/<name>`` within the same schema, which is the only
    ``$ref`` form the signature emitters accept (a definition belongs to
    its root document). Anything else raises: emission would refuse it
    too, and a clear message at load time beats one at emit time.
    """
    root = schema
    seen: set[str] = set()
    while "$ref" in root:
        ref = root["$ref"]
        if not isinstance(ref, str) or not ref.startswith("#/$defs/"):
            raise ValueError(f"unsupported root $ref {ref!r} (only '#/$defs/<name>' resolves)")
        name = ref[len("#/$defs/") :]
        if name in seen:
            raise ValueError(f"cyclic root $ref through {name!r}")
        seen.add(name)
        defs = schema.get("$defs")
        if not isinstance(defs, dict) or not isinstance(defs.get(name), dict):
            raise ValueError(f"root $ref {ref!r} names no definition in this schema's $defs")
        root = defs[name]
    return root


def _judge_binding_error(node: Node, pipeline: Pipeline) -> str | None:
    """Where a judge's ``inputs:`` and its input schema disagree, if anywhere.

    Every adapter builds a judge's input payload from ``inputs:`` and
    nothing else: one key per bound param (on a ``fan_out`` node the
    element param carries one list item, the rest are shared), then
    validates it against ``signature_spec.input_schema`` before the
    vendor call. So a required property ``inputs:`` does not bind fails
    every call. A bound param the schema does not declare is rejected
    whatever ``additionalProperties`` says: under ``false`` every call
    fails, and otherwise the value is silently dropped, so either way it
    is a binding the judge's contract cannot name. Both are decidable
    from the IR alone.
    """
    assert node.signature_spec is not None
    schema = node.signature_spec.input_schema
    title = schema.get("title")
    schema_name = "signature_spec.input_schema" + (f" ({title})" if isinstance(title, str) else "")
    prefix = f"Node {node.id!r} (llm_judge): "
    try:
        root = _schema_root(schema)
    except ValueError as e:
        return f"{prefix}{schema_name}: {e}"

    props = root.get("properties")
    declared = sorted(props) if isinstance(props, dict) else []
    required = [r for r in root.get("required") or [] if isinstance(r, str)]
    bound = node.inputs or {}
    missing = [p for p in required if p not in bound]
    undeclared = sorted(p for p in bound if p not in declared)
    if not missing and not undeclared:
        return None

    problems = []
    if missing:
        problems.append(f"it requires {missing}, which `inputs:` does not bind")
    if undeclared:
        problems.append(
            f"`inputs:` binds {undeclared}, which it does not declare (properties: {declared})"
        )
    message = (
        f"{prefix}`inputs:` ({sorted(bound) if bound else 'nothing bound'}) does not "
        f"satisfy {schema_name}: {'; and '.join(problems)}. The emitted judge builds its "
        f"input from `inputs:` alone, so every call would fail validation."
    )
    if node.fan_out and bound:
        try:
            element: str | None = fan_out_element_param(node, pipeline)
        except ValueError:
            # Ambiguous fan: emission raises its own, more specific error.
            element = None
        if element is not None:
            message += (
                f" On this fan_out node {element!r} is the element (one item of "
                f"{bound[element]} per call): give the schema a property named {element!r} "
                f"describing one whole element, and read per-element fields through "
                f"it (e.g. {{{{ {element}.<field> }}}}) instead of declaring them "
                f"beside it."
            )
    return message


def load_pipeline(path: str | Path) -> Pipeline:
    """Load and validate a pipeline.yaml file."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Pipeline.model_validate(raw)


def save_pipeline(pipeline: Pipeline, path: str | Path) -> Path:
    """Write a pipeline back to YAML, round-trippable through load_pipeline.

    Dumped from the model (``by_alias`` so ``Edge.from_`` serializes as
    ``from:``, ``exclude_none``/``exclude_defaults`` to keep the file
    close to what the compiler writes). Formatting and comments from the
    original file are not preserved — callers rewrite pipeline.yaml only
    when they materially changed the IR (e.g. observed-schema enrichment).
    """
    p = Path(path)
    data = pipeline.model_dump(mode="json", by_alias=True, exclude_none=True, exclude_defaults=True)
    p.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=100),
        encoding="utf-8",
    )
    return p
