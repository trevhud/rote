"""Tests for the serve registry (rote.serve.registry) and `rote register`.

Covers the manifest model round-trip, tool-schema synthesis from the BDR
pipeline's input contract, and the CLI subcommand end-to-end against a
temp registry file. No network, no MCP server — those live in
test_serve_server.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from rote.cli import main as cli_main
from rote.ir import PipelineInput, load_pipeline
from rote.serve.registry import (
    CloudflareTrigger,
    DbosTrigger,
    Registry,
    RegistryEntry,
    TemporalTrigger,
    entry_from_pipeline,
    input_schema_for,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
BDR_PIPELINE_YAML = REPO_ROOT / "examples" / "bdr-outreach" / "expected" / "pipeline.yaml"


def _temporal_entry(name: str = "bdr-campaign") -> RegistryEntry:
    return RegistryEntry(
        name=name,
        description="Test pipeline",
        pipeline_yaml=str(BDR_PIPELINE_YAML),
        input_schema={"type": "object", "properties": {"x": {"type": "string"}}},
        trigger=TemporalTrigger(task_queue="q", workflow_name="Wf_abc123"),
    )


# ───────── Model round-trip ─────────


def test_registry_round_trip(tmp_path: Path) -> None:
    registry = Registry()
    registry.upsert(_temporal_entry())
    registry.upsert(
        RegistryEntry(
            name="cf-pipeline",
            description="Cloudflare-deployed pipeline",
            pipeline_yaml=str(BDR_PIPELINE_YAML),
            input_schema={"type": "object"},
            trigger=CloudflareTrigger(url="https://wf.example.workers.dev"),
        )
    )
    registry.upsert(
        RegistryEntry(
            name="dbos-pipeline",
            description="DBOS-deployed pipeline",
            pipeline_yaml=str(BDR_PIPELINE_YAML),
            input_schema={"type": "object"},
            trigger=DbosTrigger(
                system_database_url="sqlite:////tmp/app/bdr-campaign.dbos.sqlite",
                workflow_name="BdrCampaign_abc123",
                queue_name="bdr-campaign-queue",
                gate_signals=["contact_review_approved"],
            ),
        )
    )

    path = tmp_path / "registry.json"
    registry.save(path)
    loaded = Registry.load(path)

    assert loaded == registry
    # Discriminated union survives serialization with the right types.
    assert isinstance(loaded.entries[0].trigger, TemporalTrigger)
    assert isinstance(loaded.entries[1].trigger, CloudflareTrigger)
    assert isinstance(loaded.entries[2].trigger, DbosTrigger)
    assert loaded.entries[2].trigger.gate_signals == ["contact_review_approved"]


def test_registry_load_missing_file_is_empty(tmp_path: Path) -> None:
    registry = Registry.load(tmp_path / "nope.json")
    assert registry.entries == []
    assert registry.version == 1


def test_registry_upsert_replaces_by_name() -> None:
    registry = Registry()
    assert registry.upsert(_temporal_entry()) is False

    updated = _temporal_entry()
    updated.description = "Updated"
    assert registry.upsert(updated) is True
    assert len(registry.entries) == 1
    assert registry.entries[0].description == "Updated"


def test_registry_get() -> None:
    registry = Registry()
    registry.upsert(_temporal_entry())
    assert registry.get("bdr-campaign") is not None
    assert registry.get("missing") is None


def test_tool_name_charset_enforced() -> None:
    with pytest.raises(ValidationError, match="tool name"):
        _temporal_entry(name="bad name!")


def test_registry_save_creates_parent_dirs(tmp_path: Path) -> None:
    path = tmp_path / "deep" / "nested" / "registry.json"
    Registry().save(path)
    assert json.loads(path.read_text())["entries"] == []


# ───────── inputSchema synthesis ─────────


def test_input_schema_uses_bdr_typed_schema() -> None:
    """The BDR baseline carries a typed input_schema; it must flow through
    verbatim rather than falling back to name-list synthesis."""
    pipeline = load_pipeline(BDR_PIPELINE_YAML)
    schema = input_schema_for(pipeline.input)

    assert schema == pipeline.input.input_schema
    assert schema["type"] == "object"
    # Typed fields, not the synthesized empty-dict placeholders.
    assert schema["properties"]["drug_brand"] == {
        "title": "Drug Brand",
        "type": "string",
    }
    assert set(pipeline.input.required) <= set(schema["properties"])


def test_input_schema_synthesized_from_name_lists_without_schema() -> None:
    pipeline_input = PipelineInput(
        type="CampaignBrief",
        required=["drug_brand", "drug_generic"],
        optional=["job_focus"],
    )
    schema = input_schema_for(pipeline_input)

    assert schema["type"] == "object"
    assert schema["title"] == "CampaignBrief"
    assert set(schema["required"]) == {"drug_brand", "drug_generic"}
    # Optional fields are properties but not required.
    assert "job_focus" in schema["properties"]
    assert "job_focus" not in schema["required"]
    assert schema["additionalProperties"] is True


def test_input_schema_prefers_structured_field_when_present() -> None:
    """A concurrent change is adding input_schema to PipelineInput; when a
    loaded model carries one, it must win over the synthesized fallback."""

    class PipelineInputWithSchema(PipelineInput):
        input_schema: dict[str, object] | None = None

    explicit = {
        "type": "object",
        "properties": {"drug_brand": {"type": "string"}},
        "required": ["drug_brand"],
    }
    pipeline_input = PipelineInputWithSchema(
        type="CampaignBrief",
        required=["drug_brand"],
        input_schema=explicit,
    )
    assert input_schema_for(pipeline_input) == explicit


def test_input_schema_falls_back_when_structured_field_empty() -> None:
    class PipelineInputWithSchema(PipelineInput):
        input_schema: dict[str, object] | None = None

    pipeline_input = PipelineInputWithSchema(
        type="Brief",
        required=["a"],
        optional=["b"],
        input_schema=None,
    )
    schema = input_schema_for(pipeline_input)
    assert schema["required"] == ["a"]
    assert set(schema["properties"]) == {"a", "b"}


# ───────── entry_from_pipeline ─────────


def test_entry_from_pipeline_bdr() -> None:
    pipeline = load_pipeline(BDR_PIPELINE_YAML)
    entry = entry_from_pipeline(
        pipeline,
        BDR_PIPELINE_YAML,
        TemporalTrigger(task_queue="q", workflow_name="Wf_abc123"),
    )

    assert entry.name == "bdr-campaign"
    assert entry.description.startswith("End-to-end BDR outreach campaign")
    assert entry.pipeline_yaml == str(BDR_PIPELINE_YAML.resolve())
    assert entry.input_schema["title"] == "CampaignBrief"


def test_entry_from_pipeline_name_override() -> None:
    pipeline = load_pipeline(BDR_PIPELINE_YAML)
    entry = entry_from_pipeline(
        pipeline,
        BDR_PIPELINE_YAML,
        CloudflareTrigger(url="https://wf.example.workers.dev"),
        name="my_tool",
    )
    assert entry.name == "my_tool"


# ───────── `rote register` CLI ─────────


def test_register_bdr_temporal_defaults(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    registry_path = tmp_path / "registry.json"
    rc = cli_main(
        [
            "register",
            str(BDR_PIPELINE_YAML),
            "--registry",
            str(registry_path),
            "--runtime",
            "temporal",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "registered tool 'bdr-campaign'" in out

    registry = Registry.load(registry_path)
    assert len(registry.entries) == 1
    entry = registry.entries[0]
    assert isinstance(entry.trigger, TemporalTrigger)
    assert entry.trigger.task_queue == "bdr-campaign"

    # The default workflow name must match the versioned @workflow.defn
    # name the Temporal adapter emits — otherwise triggering fails.
    from rote.adapters.temporal import _pipeline_hash, _to_pascal_case

    pipeline = load_pipeline(BDR_PIPELINE_YAML)
    expected = f"{_to_pascal_case(pipeline.name)}_{_pipeline_hash(pipeline)}"
    assert entry.trigger.workflow_name == expected


def test_register_twice_updates_single_entry(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.json"
    assert (
        cli_main(
            [
                "register",
                str(BDR_PIPELINE_YAML),
                "--registry",
                str(registry_path),
                "--runtime",
                "temporal",
            ]
        )
        == 0
    )
    assert (
        cli_main(
            [
                "register",
                str(BDR_PIPELINE_YAML),
                "--registry",
                str(registry_path),
                "--runtime",
                "temporal",
                "--task-queue",
                "custom-queue",
            ]
        )
        == 0
    )

    registry = Registry.load(registry_path)
    assert len(registry.entries) == 1
    trigger = registry.entries[0].trigger
    assert isinstance(trigger, TemporalTrigger)
    assert trigger.task_queue == "custom-queue"


def test_register_cloudflare_requires_url(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = cli_main(
        [
            "register",
            str(BDR_PIPELINE_YAML),
            "--registry",
            str(tmp_path / "r.json"),
            "--runtime",
            "cloudflare",
        ]
    )
    assert rc == 2
    assert "--url is required" in capsys.readouterr().err


def test_register_cloudflare_with_url(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.json"
    rc = cli_main(
        [
            "register",
            str(BDR_PIPELINE_YAML),
            "--registry",
            str(registry_path),
            "--runtime",
            "cloudflare",
            "--url",
            "https://wf.example.workers.dev",
            "--status-url",
            "https://wf.example.workers.dev/status/{workflow_id}",
        ]
    )
    assert rc == 0
    trigger = Registry.load(registry_path).entries[0].trigger
    assert isinstance(trigger, CloudflareTrigger)
    assert trigger.url == "https://wf.example.workers.dev"
    assert trigger.status_url == "https://wf.example.workers.dev/status/{workflow_id}"


def test_register_dbos_defaults(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """dbos is the default runtime; workflow/queue names derive from the
    pipeline exactly like the DBOS adapter emits them, and the pipeline's
    HITL gate signals are captured for the _signal tool."""
    registry_path = tmp_path / "registry.json"
    rc = cli_main(
        [
            "register",
            str(BDR_PIPELINE_YAML),
            "--registry",
            str(registry_path),
            "--system-database-url",
            "sqlite:////tmp/app/bdr-campaign.dbos.sqlite",
        ]
    )
    assert rc == 0
    assert "registered tool 'bdr-campaign' (dbos)" in capsys.readouterr().out

    entry = Registry.load(registry_path).entries[0]
    trigger = entry.trigger
    assert isinstance(trigger, DbosTrigger)
    assert trigger.system_database_url == "sqlite:////tmp/app/bdr-campaign.dbos.sqlite"
    assert trigger.queue_name == "bdr-campaign-queue"

    # The default workflow name must match the versioned @DBOS.workflow
    # name the DBOS adapter emits — otherwise enqueued runs never execute.
    from rote.adapters._common import _pipeline_hash, _to_pascal_case

    pipeline = load_pipeline(BDR_PIPELINE_YAML)
    assert trigger.workflow_name == f"{_to_pascal_case(pipeline.name)}_{_pipeline_hash(pipeline)}"

    # Gate signals captured from the IR, in pipeline order.
    assert trigger.gate_signals == ["contact_review_approved", "bdr_enrollment_complete"]


def test_register_dbos_env_var_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DBOS_SYSTEM_DATABASE_URL", "postgresql://u:p@db:5432/rote")
    registry_path = tmp_path / "registry.json"
    rc = cli_main(["register", str(BDR_PIPELINE_YAML), "--registry", str(registry_path)])
    assert rc == 0
    trigger = Registry.load(registry_path).entries[0].trigger
    assert isinstance(trigger, DbosTrigger)
    assert trigger.system_database_url == "postgresql://u:p@db:5432/rote"


def test_register_dbos_requires_system_database_url(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bare pipeline.yaml, no env var, no emitted app dir nearby: rather
    than fabricate a database nobody uses, registration must fail with
    guidance (mirrors cloudflare's required --url)."""
    monkeypatch.delenv("DBOS_SYSTEM_DATABASE_URL", raising=False)
    # Copy the pipeline into an empty dir so no main.py is discoverable
    # (the repo's expected/ tree has emitted runtimes/ nearby).
    (tmp_path / "pipeline.yaml").write_text(
        BDR_PIPELINE_YAML.read_text(encoding="utf-8"), encoding="utf-8"
    )
    rc = cli_main(
        ["register", str(tmp_path / "pipeline.yaml"), "--registry", str(tmp_path / "r.json")]
    )
    assert rc == 2
    assert "--system-database-url is required" in capsys.readouterr().err


def test_register_resolves_graduate_out_layout(tmp_path: Path) -> None:
    """A `rote graduate --out` dir nests pipeline.yaml under graduated/;
    with the default dbos runtime, the system DB URL derives from the
    emitted app dir at runtime/dbos/."""
    out_dir = tmp_path / "bdr-out"
    (out_dir / "graduated").mkdir(parents=True)
    (out_dir / "graduated" / "pipeline.yaml").write_text(
        BDR_PIPELINE_YAML.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (out_dir / "runtime" / "dbos").mkdir(parents=True)
    (out_dir / "runtime" / "dbos" / "main.py").write_text("# emitted app\n", encoding="utf-8")

    registry_path = tmp_path / "registry.json"
    rc = cli_main(["register", str(out_dir), "--registry", str(registry_path)])
    assert rc == 0
    entry = Registry.load(registry_path).entries[0]
    assert entry.pipeline_yaml == str((out_dir / "graduated" / "pipeline.yaml").resolve())
    assert isinstance(entry.trigger, DbosTrigger)
    expected_sqlite = (out_dir / "runtime" / "dbos" / "bdr-campaign.dbos.sqlite").resolve()
    assert entry.trigger.system_database_url == f"sqlite:///{expected_sqlite}"


def test_register_missing_pipeline_yaml(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli_main(["register", str(tmp_path), "--registry", str(tmp_path / "r.json")])
    assert rc == 2
    assert "no pipeline.yaml found" in capsys.readouterr().err
