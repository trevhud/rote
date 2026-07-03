"""Manifest registry for graduated pipelines served over MCP.

The registry is a single JSON file (default ``~/.rote/registry.json``)
listing every graduated pipeline the user wants exposed as an MCP tool:
tool name, description, the pipeline.yaml it came from, a JSON Schema
for the tool's input, and the runtime trigger config (Temporal address /
task queue, Cloudflare workflow URL, …).

``rote register`` writes entries; ``rote serve`` reads them. The file is
the contract between the two commands — same philosophy as the driver
layer's ``work_dir/pipeline.yaml`` contract.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from rote.ir import Pipeline, PipelineInput

REGISTRY_SCHEMA_VERSION = 1

#: MCP tool names must be safe for every client; Cloudflare's signal-name
#: constraint ([A-Za-z0-9_-]+) is the strictest charset we target, so the
#: registry enforces the same one for tool names.
_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def default_registry_path() -> Path:
    """The registry location used when ``--registry`` is not given."""
    return Path.home() / ".rote" / "registry.json"


class TemporalTrigger(BaseModel):
    """How to start a graduated pipeline on a Temporal cluster."""

    model_config = ConfigDict(extra="forbid")

    runtime: Literal["temporal"] = "temporal"
    address: str = Field(
        default="localhost:7233",
        description="Temporal frontend address, host:port",
    )
    namespace: str = Field(default="default")
    task_queue: str = Field(description="Task queue the graduated worker polls")
    workflow_name: str = Field(
        description=(
            "Registered workflow *type* name. The Temporal adapter emits a "
            "versioned name (PascalCase pipeline name + pipeline hash), so "
            "this must match the emitted @workflow.defn name exactly."
        ),
    )


class CloudflareTrigger(BaseModel):
    """How to start a graduated pipeline deployed as a Cloudflare Worker.

    The emitted ``src/index.ts`` fetch handler accepts a JSON POST body
    (the workflow params) and responds with ``{id, status}``.
    """

    model_config = ConfigDict(extra="forbid")

    runtime: Literal["cloudflare"] = "cloudflare"
    url: str = Field(description="Deployed worker URL (the fetch trigger endpoint)")
    status_url: str | None = Field(
        default=None,
        description=(
            "Optional status endpoint template containing '{workflow_id}'. "
            "The emitted worker does not expose a status route by default, "
            "so without this the companion status tool directs users to "
            "`wrangler workflows instances describe` instead."
        ),
    )


Trigger = Annotated[TemporalTrigger | CloudflareTrigger, Field(discriminator="runtime")]


class RegistryEntry(BaseModel):
    """One graduated pipeline exposed as one MCP tool."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="MCP tool name (unique within the registry)")
    description: str = ""
    pipeline_yaml: str = Field(description="Absolute path to the graduated pipeline.yaml")
    input_schema: dict[str, Any] = Field(
        description="JSON Schema for the tool input (the pipeline's input contract)",
    )
    trigger: Trigger
    registered_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="ISO-8601 timestamp of the last register/update",
    )

    @field_validator("name")
    @classmethod
    def _validate_tool_name(cls, v: str) -> str:
        if not _TOOL_NAME_RE.match(v):
            raise ValueError(
                f"tool name {v!r} must match [A-Za-z0-9][A-Za-z0-9_-]* (MCP tool-name safe charset)"
            )
        return v


class Registry(BaseModel):
    """The full registry file."""

    model_config = ConfigDict(extra="forbid")

    version: int = REGISTRY_SCHEMA_VERSION
    entries: list[RegistryEntry] = Field(default_factory=list)

    def get(self, name: str) -> RegistryEntry | None:
        for entry in self.entries:
            if entry.name == name:
                return entry
        return None

    def upsert(self, entry: RegistryEntry) -> bool:
        """Insert or replace by tool name. Returns True if an entry was replaced."""
        for i, existing in enumerate(self.entries):
            if existing.name == entry.name:
                self.entries[i] = entry
                return True
        self.entries.append(entry)
        return False

    @classmethod
    def load(cls, path: str | Path) -> Registry:
        """Load the registry; a missing file is an empty registry."""
        p = Path(path)
        if not p.exists():
            return cls()
        raw = json.loads(p.read_text(encoding="utf-8"))
        return cls.model_validate(raw)

    def save(self, path: str | Path) -> None:
        """Write the registry atomically (tmp file + rename)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")
        tmp.replace(p)


def input_schema_for(pipeline_input: PipelineInput) -> dict[str, Any]:
    """Derive the MCP tool's inputSchema from the pipeline's input contract.

    Prefers a structured ``input_schema`` field if the loaded model carries
    one (being added to :class:`rote.ir.PipelineInput` concurrently — coded
    defensively via ``getattr`` so this works before and after that lands).
    Otherwise synthesizes a permissive object schema from the required /
    optional name lists, which are untyped in the current IR.
    """
    explicit = getattr(pipeline_input, "input_schema", None)
    if isinstance(explicit, dict) and explicit:
        return dict(explicit)

    properties: dict[str, Any] = {
        name: {} for name in [*pipeline_input.required, *pipeline_input.optional]
    }
    return {
        "type": "object",
        "title": pipeline_input.type,
        "description": (
            f"Input contract for the graduated pipeline "
            f"(type {pipeline_input.type}; field types are untyped in the IR v0)"
        ),
        "properties": properties,
        "required": list(pipeline_input.required),
        "additionalProperties": True,
    }


def entry_from_pipeline(
    pipeline: Pipeline,
    pipeline_yaml: Path,
    trigger: TemporalTrigger | CloudflareTrigger,
    name: str | None = None,
) -> RegistryEntry:
    """Build a registry entry from a loaded pipeline IR.

    Tool name defaults to ``pipeline.name`` with any charset-unsafe
    characters replaced by ``-``; description comes from
    ``pipeline.description`` (first paragraph, whitespace-normalized).
    """
    tool_name = name if name is not None else re.sub(r"[^A-Za-z0-9_-]+", "-", pipeline.name)
    return RegistryEntry(
        name=tool_name,
        description=pipeline.description.strip(),
        pipeline_yaml=str(pipeline_yaml.resolve()),
        input_schema=input_schema_for(pipeline.input),
        trigger=trigger,
    )
