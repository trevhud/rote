"""Tests for incremental re-graduation (`rote graduate --update`).

The update flow is: diff the current SKILL.md against the previous
run's provenance sidecar → build an UpdatePlan → materialize the
previous pipeline + an UPDATE.md brief into the driver's work dir →
run the agent with minimal-patch instructions → enforce that
provenance-preserved node ids survived → file-level merge into the
output (so untouched, possibly user-filled modules survive).

Drivers are faked, as in test_graduator.py — these tests exercise the
orchestrator's update machinery, not an agent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rote.graduator import Graduator, GraduatorError
from rote.graduator.drivers import DriverResult, GraduatorDriver
from rote.graduator.update import (
    UPDATE_CONTEXT_DIRNAME,
    build_update_plan,
    render_update_brief,
)
from rote.ir import load_pipeline
from rote.skill_source import load_provenance

SKILL_MD_V1 = """\
# Demo skill

## Phase 1: Gather

Gather instructions.

## Phase 2: Judge

Judge instructions v1.
"""

# v2 changes only the Judge section.
SKILL_MD_V2 = SKILL_MD_V1.replace("Judge instructions v1.", "Judge instructions v2, stricter.")

PIPELINE_YAML = """\
name: demo
version: "0.1.0"
description: demo pipeline
input:
  type: In
  required: [foo]
nodes:
  - id: gather
    kind: pure_function
    description: gather things
    impl: extracted/gather.py:gather
    source:
      section: "Phase 1: Gather"
  - id: judge
    kind: pure_function
    description: judge things
    impl: extracted/judge.py:judge
    source:
      section: "Phase 2: Judge"
edges:
  - {from: gather, to: judge}
entry_nodes: [gather]
exit_nodes: [judge]
"""

PIPELINE_YAML_WITHOUT_GATHER = """\
name: demo
version: "0.1.0"
description: demo pipeline
input:
  type: In
  required: [foo]
nodes:
  - id: judge
    kind: pure_function
    description: judge things
    impl: extracted/judge.py:judge
    source:
      section: "Phase 2: Judge"
edges: []
entry_nodes: [judge]
exit_nodes: [judge]
"""


class _RecordingDriver(GraduatorDriver):
    """Fake driver that records its invocation and writes canned files."""

    name = "fake"

    def __init__(self, pipeline_yaml: str, extras: dict[str, str] | None = None) -> None:
        self._pipeline_yaml = pipeline_yaml
        self._extras = extras or {}
        self.extra_instructions: str | None = None
        self.saw_update_context: dict[str, bool] = {}
        self.invocations = 0

    def is_available(self) -> tuple[bool, str]:
        return (True, "")

    async def run(
        self,
        skill_dir: Path,
        graduator_skill_dir: Path,
        work_dir: Path,
        extra_instructions: str | None = None,
        on_event: object = None,
    ) -> DriverResult:
        self.invocations += 1
        self.extra_instructions = extra_instructions
        ctx = work_dir / UPDATE_CONTEXT_DIRNAME
        self.saw_update_context = {
            "brief": (ctx / "UPDATE.md").is_file(),
            "pipeline": (ctx / "pipeline.yaml").is_file(),
            "extracted": (ctx / "extracted" / "gather.py").is_file(),
        }
        pipeline_yaml = work_dir / "pipeline.yaml"
        pipeline_yaml.write_text(self._pipeline_yaml, encoding="utf-8")
        for rel, content in self._extras.items():
            target = work_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        return DriverResult(
            pipeline_yaml_path=pipeline_yaml,
            work_dir=work_dir,
            driver_name=self.name,
            metadata={},
        )


@pytest.fixture
def graduator_skill_dir(tmp_path: Path) -> Path:
    d = tmp_path / "rote-graduate"
    d.mkdir()
    (d / "SKILL.md").write_text("# fake graduator", encoding="utf-8")
    return d


@pytest.fixture
def skill_dir(tmp_path: Path) -> Path:
    d = tmp_path / "demo-skill"
    d.mkdir()
    (d / "SKILL.md").write_text(SKILL_MD_V1, encoding="utf-8")
    return d


async def _full_graduation(
    skill_dir: Path, graduator_skill_dir: Path, output_dir: Path
) -> _RecordingDriver:
    driver = _RecordingDriver(
        PIPELINE_YAML,
        extras={
            "extracted/gather.py": "def gather():\n    raise NotImplementedError\n",
            "extracted/judge.py": "def judge():\n    raise NotImplementedError\n",
        },
    )
    graduator = Graduator(graduator_skill_dir=graduator_skill_dir)
    graduator.select_driver = lambda: driver  # type: ignore[method-assign]
    await graduator.graduate(skill_dir, output_dir)
    return driver


# ───────── UpdatePlan unit behavior ─────────


def test_build_update_plan_classifies_sections_and_nodes(tmp_path: Path) -> None:
    from rote.skill_source import build_provenance

    pipeline = _load_yaml_pipeline(tmp_path)
    prov = build_provenance(pipeline, SKILL_MD_V1)
    plan = build_update_plan(pipeline, prov, SKILL_MD_V2)

    assert plan.changed_sections == ("Phase 2: Judge",)
    assert plan.added_sections == ()
    assert plan.removed_sections == ()
    assert plan.preserved_node_ids == ("gather",)
    assert plan.stale_node_ids == ("judge",)
    assert not plan.is_noop


def test_build_update_plan_noop_when_skill_unchanged(tmp_path: Path) -> None:
    from rote.skill_source import build_provenance

    pipeline = _load_yaml_pipeline(tmp_path)
    prov = build_provenance(pipeline, SKILL_MD_V1)
    plan = build_update_plan(pipeline, prov, SKILL_MD_V1)

    assert plan.is_noop
    assert plan.preserved_node_ids == ("gather", "judge")


def test_node_without_provenance_is_stale(tmp_path: Path) -> None:
    """No verifiable provenance → re-derive; never silently keep."""
    from rote.skill_source import build_provenance

    pipeline = _load_yaml_pipeline(tmp_path)
    prov = build_provenance(pipeline, SKILL_MD_V1)
    del prov["nodes"]["gather"]
    plan = build_update_plan(pipeline, prov, SKILL_MD_V2)
    assert "gather" in plan.stale_node_ids


def test_render_update_brief_names_the_contract(tmp_path: Path) -> None:
    from rote.skill_source import build_provenance

    pipeline = _load_yaml_pipeline(tmp_path)
    prov = build_provenance(pipeline, SKILL_MD_V1)
    plan = build_update_plan(pipeline, prov, SKILL_MD_V2)
    brief = render_update_brief(plan)

    assert "Phase 2: Judge" in brief
    assert "- gather" in brief  # preserved list
    assert "VERBATIM" in brief
    assert "COMPLETE updated pipeline.yaml" in brief


def _load_yaml_pipeline(tmp_path: Path):  # noqa: ANN202
    p = tmp_path / "p.yaml"
    p.write_text(PIPELINE_YAML, encoding="utf-8")
    return load_pipeline(p)


# ───────── Orchestrator update flow ─────────


@pytest.mark.asyncio
async def test_update_requires_previous_run(
    skill_dir: Path, graduator_skill_dir: Path, tmp_path: Path
) -> None:
    graduator = Graduator(graduator_skill_dir=graduator_skill_dir)
    with pytest.raises(GraduatorError, match="previous graduation"):
        await graduator.graduate(skill_dir, tmp_path / "empty-out", update=True)


@pytest.mark.asyncio
async def test_update_requires_provenance_sidecar(
    skill_dir: Path, graduator_skill_dir: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "pipeline.yaml").write_text(PIPELINE_YAML, encoding="utf-8")

    graduator = Graduator(graduator_skill_dir=graduator_skill_dir)
    with pytest.raises(GraduatorError, match="provenance"):
        await graduator.graduate(skill_dir, output_dir, update=True)


@pytest.mark.asyncio
async def test_update_with_unchanged_skill_skips_the_agent(
    skill_dir: Path, graduator_skill_dir: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "out"
    await _full_graduation(skill_dir, graduator_skill_dir, output_dir)

    update_driver = _RecordingDriver(PIPELINE_YAML)
    graduator = Graduator(graduator_skill_dir=graduator_skill_dir)
    graduator.select_driver = lambda: update_driver  # type: ignore[method-assign]

    result = await graduator.graduate(skill_dir, output_dir, update=True)

    assert update_driver.invocations == 0
    assert result.driver_name == "(no-op)"
    assert "no source sections changed" in result.driver_metadata["update"]
    assert result.pipeline.name == "demo"


@pytest.mark.asyncio
async def test_update_run_gets_context_and_merges_output(
    skill_dir: Path, graduator_skill_dir: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "out"
    await _full_graduation(skill_dir, graduator_skill_dir, output_dir)

    # The user filled in a stub since graduation; the skill's Judge
    # section then changed.
    user_impl = "def gather():\n    return fetch_from_api()\n"
    (output_dir / "extracted" / "gather.py").write_text(user_impl, encoding="utf-8")
    (skill_dir / "SKILL.md").write_text(SKILL_MD_V2, encoding="utf-8")

    update_driver = _RecordingDriver(
        PIPELINE_YAML,
        extras={"extracted/judge.py": "def judge():\n    raise NotImplementedError  # v2\n"},
    )
    graduator = Graduator(graduator_skill_dir=graduator_skill_dir)
    graduator.select_driver = lambda: update_driver  # type: ignore[method-assign]

    result = await graduator.graduate(skill_dir, output_dir, update=True)

    # The agent saw the brief, the previous pipeline, and the previous stubs.
    assert update_driver.saw_update_context == {
        "brief": True,
        "pipeline": True,
        "extracted": True,
    }
    assert update_driver.extra_instructions is not None
    assert "UPDATE.md" in update_driver.extra_instructions

    # File-level merge: the rewritten module updated, the untouched
    # (user-filled) module survived, the context dir did not leak out.
    assert "# v2" in (output_dir / "extracted" / "judge.py").read_text()
    assert (output_dir / "extracted" / "gather.py").read_text() == user_impl
    assert not (output_dir / UPDATE_CONTEXT_DIRNAME).exists()

    # Provenance was re-stamped against the new skill text.
    prov = load_provenance(output_dir / "provenance.json")
    assert prov["nodes"]["judge"]["content_hash"] == prov["sections"]["Phase 2: Judge"]
    assert result.pipeline.node_by_id("judge") is not None


@pytest.mark.asyncio
async def test_update_rejects_dropped_preserved_node(
    skill_dir: Path, graduator_skill_dir: Path, tmp_path: Path
) -> None:
    """An update that loses a provenance-preserved node id would orphan
    in-flight workflows — refuse it and leave the previous output alone."""
    output_dir = tmp_path / "out"
    await _full_graduation(skill_dir, graduator_skill_dir, output_dir)
    (skill_dir / "SKILL.md").write_text(SKILL_MD_V2, encoding="utf-8")
    before = (output_dir / "pipeline.yaml").read_text()

    update_driver = _RecordingDriver(PIPELINE_YAML_WITHOUT_GATHER)
    graduator = Graduator(graduator_skill_dir=graduator_skill_dir)
    graduator.select_driver = lambda: update_driver  # type: ignore[method-assign]

    with pytest.raises(GraduatorError, match="gather"):
        await graduator.graduate(skill_dir, output_dir, update=True)

    assert (output_dir / "pipeline.yaml").read_text() == before


# ───────── CLI wiring ─────────


def test_cli_graduate_exposes_update_flag() -> None:
    from rote.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args(["graduate", "some-skill", "--out", "o", "--update"])
    assert args.update is True
    args = parser.parse_args(["graduate", "some-skill", "--out", "o"])
    assert args.update is False


def test_provenance_survives_roundtrip_through_update_brief(tmp_path: Path) -> None:
    """Sanity: the sidecar written by a full run parses back into the
    exact structures build_update_plan consumes."""
    from rote.skill_source import write_provenance

    pipeline = _load_yaml_pipeline(tmp_path)
    sidecar = tmp_path / "provenance.json"
    write_provenance(sidecar, pipeline, SKILL_MD_V1)
    prov = load_provenance(sidecar)
    plan = build_update_plan(pipeline, prov, SKILL_MD_V2)
    assert plan.stale_node_ids == ("judge",)
    # The file is valid JSON with sorted, stable keys (diff-friendly).
    raw = json.loads(sidecar.read_text())
    assert list(raw["sections"]) == sorted(raw["sections"])
