"""Tests for the high-level Graduator orchestrator.

The orchestrator is mocked at the driver layer — we don't actually
run an agent. The test driver is a class that implements the
GraduatorDriver Protocol and writes a canned pipeline.yaml + a few
extra files into the work directory, mimicking what a real driver
would produce.

Coverage:

* Happy path: skill → driver.run → load_pipeline → move to output
* Output dir contains all artifacts the driver wrote (not just yaml)
* Auto-detect: when no driver available, helpful error
* Explicit agent: unknown name → GraduatorError
* Explicit agent: unavailable → GraduatorError with reason
* Skill validation: missing dir, missing SKILL.md → GraduatorError
* Pipeline validation: invalid yaml → GraduatorError
* Driver error → GraduatorError with details preserved
* Default graduator skill dir resolves to the bundled rote-graduate skill
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from rote.graduator import Graduator, GraduatorError
from rote.graduator.drivers import DriverError, DriverResult, GraduatorDriver

REPO_ROOT = Path(__file__).resolve().parent.parent
BUNDLED_GRADUATOR_SKILL = REPO_ROOT / "skills" / "rote-graduate"


# ───────── Fakes ─────────


class _FakeDriver(GraduatorDriver):
    """A fake driver that writes a canned pipeline.yaml + extra stubs.

    Used by tests that want to exercise the orchestrator without
    bringing up the real Anthropic SDK loop.
    """

    name = "fake"

    def __init__(
        self,
        pipeline_yaml: str,
        extras: dict[str, str] | None = None,
        metadata: dict[str, object] | None = None,
        is_available_result: tuple[bool, str] = (True, ""),
    ) -> None:
        self._pipeline_yaml = pipeline_yaml
        self._extras = extras or {}
        self._metadata = metadata or {"tokens": 100}
        self._is_available = is_available_result
        self.run_called_with: dict[str, Path] | None = None

    def is_available(self) -> tuple[bool, str]:
        return self._is_available

    async def run(
        self,
        skill_dir: Path,
        graduator_skill_dir: Path,
        work_dir: Path,
    ) -> DriverResult:
        self.run_called_with = {
            "skill_dir": skill_dir,
            "graduator_skill_dir": graduator_skill_dir,
            "work_dir": work_dir,
        }
        pipeline_yaml = work_dir / "pipeline.yaml"
        pipeline_yaml.write_text(self._pipeline_yaml, encoding="utf-8")
        for rel_path, content in self._extras.items():
            target = work_dir / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        return DriverResult(
            pipeline_yaml_path=pipeline_yaml,
            work_dir=work_dir,
            driver_name=self.name,
            metadata=self._metadata,
        )


class _FailingDriver(GraduatorDriver):
    name = "failing"

    def is_available(self) -> tuple[bool, str]:
        return (True, "")

    async def run(
        self, skill_dir: Path, graduator_skill_dir: Path, work_dir: Path
    ) -> DriverResult:
        raise DriverError("simulated failure", details="extra context here")


VALID_YAML = """\
name: fake-pipeline
version: "0.1.0"
description: |
  A minimal pipeline used for graduator orchestrator testing.

input:
  type: FakeInput
  required: [foo]

nodes:
  - id: only_node
    kind: pure_function
    description: The one and only node.
    impl: extracted/foo.py:only_node

edges: []
entry_nodes: [only_node]
exit_nodes: [only_node]
"""

INVALID_YAML = """\
name: bad
nodes:
  - id: a
    kind: bogus_kind   # not a real NodeKind
    description: x
"""


# ───────── Skill bundle fixture ─────────


@pytest.fixture
def fake_skill_dir(tmp_path: Path) -> Path:
    skill_dir = tmp_path / "fake-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: fake\n---\n\n# Fake skill\n", encoding="utf-8"
    )
    return skill_dir


@pytest.fixture
def fake_graduator_skill_dir(tmp_path: Path) -> Path:
    grad_dir = tmp_path / "fake-graduator-skill"
    grad_dir.mkdir()
    (grad_dir / "SKILL.md").write_text("# fake graduator", encoding="utf-8")
    return grad_dir


# ───────── Happy path ─────────


@pytest.mark.asyncio
async def test_graduate_happy_path(
    fake_skill_dir: Path,
    fake_graduator_skill_dir: Path,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "output"

    fake_driver = _FakeDriver(
        pipeline_yaml=VALID_YAML,
        extras={
            "extracted/foo.py": "# stub\ndef only_node():\n    pass\n",
            "signatures/__init__.py": "",
        },
        metadata={"tokens": 1234, "iterations": 5},
    )

    graduator = Graduator(graduator_skill_dir=fake_graduator_skill_dir)
    # Bypass select_driver so we don't need any real backend
    graduator.select_driver = lambda: fake_driver  # type: ignore[method-assign]

    result = await graduator.graduate(fake_skill_dir, output_dir)

    # Validated pipeline returned
    assert result.pipeline.name == "fake-pipeline"
    assert result.pipeline.version == "0.1.0"
    assert result.driver_name == "fake"
    assert result.driver_metadata == {"tokens": 1234, "iterations": 5}
    assert result.output_dir == output_dir.resolve()

    # All artifacts moved to output dir
    assert (output_dir / "pipeline.yaml").is_file()
    assert (output_dir / "extracted" / "foo.py").is_file()
    assert (output_dir / "signatures" / "__init__.py").is_file()
    assert "only_node" in (output_dir / "extracted" / "foo.py").read_text()

    # Driver was called with the right paths
    assert fake_driver.run_called_with is not None
    assert fake_driver.run_called_with["skill_dir"] == fake_skill_dir.resolve()
    assert (
        fake_driver.run_called_with["graduator_skill_dir"]
        == fake_graduator_skill_dir
    )


@pytest.mark.asyncio
async def test_graduate_overwrites_existing_output_files(
    fake_skill_dir: Path,
    fake_graduator_skill_dir: Path,
    tmp_path: Path,
) -> None:
    """If the output dir already has a pipeline.yaml, it gets replaced."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "pipeline.yaml").write_text("OLD CONTENT")

    fake_driver = _FakeDriver(pipeline_yaml=VALID_YAML)
    graduator = Graduator(graduator_skill_dir=fake_graduator_skill_dir)
    graduator.select_driver = lambda: fake_driver  # type: ignore[method-assign]

    await graduator.graduate(fake_skill_dir, output_dir)

    assert (output_dir / "pipeline.yaml").read_text() == VALID_YAML


# ───────── Driver selection ─────────


@pytest.mark.asyncio
async def test_explicit_unknown_agent_raises(
    fake_skill_dir: Path,
    fake_graduator_skill_dir: Path,
    tmp_path: Path,
) -> None:
    graduator = Graduator(
        agent="bogus", graduator_skill_dir=fake_graduator_skill_dir
    )
    with pytest.raises(GraduatorError, match="bogus"):
        await graduator.graduate(fake_skill_dir, tmp_path / "out")


@pytest.mark.asyncio
async def test_explicit_unavailable_agent_raises_with_reason(
    fake_skill_dir: Path,
    fake_graduator_skill_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """User asks for `--agent claude` but claude isn't installed."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with patch(
        "rote.graduator.drivers.claude.which", return_value=None
    ):
        graduator = Graduator(
            agent="claude", graduator_skill_dir=fake_graduator_skill_dir
        )
        with pytest.raises(GraduatorError, match="not available"):
            await graduator.graduate(fake_skill_dir, tmp_path / "out")


@pytest.mark.asyncio
async def test_auto_detect_with_no_drivers_available_raises_helpful_error(
    fake_skill_dir: Path,
    fake_graduator_skill_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with patch(
        "rote.graduator.drivers.claude.which", return_value=None
    ), patch(
        "rote.graduator.drivers.codex.which", return_value=None
    ):
        graduator = Graduator(graduator_skill_dir=fake_graduator_skill_dir)
        with pytest.raises(GraduatorError) as excinfo:
            await graduator.graduate(fake_skill_dir, tmp_path / "out")

    msg = str(excinfo.value)
    assert "claude" in msg
    assert "codex" in msg
    assert "api" in msg


# ───────── Skill validation ─────────


@pytest.mark.asyncio
async def test_missing_skill_dir_raises(
    fake_graduator_skill_dir: Path,
    tmp_path: Path,
) -> None:
    graduator = Graduator(graduator_skill_dir=fake_graduator_skill_dir)
    with pytest.raises(GraduatorError, match="does not exist"):
        await graduator.graduate(
            tmp_path / "nonexistent", tmp_path / "out"
        )


@pytest.mark.asyncio
async def test_skill_dir_without_skill_md_raises(
    fake_graduator_skill_dir: Path,
    tmp_path: Path,
) -> None:
    skill_dir = tmp_path / "not-a-skill"
    skill_dir.mkdir()  # but no SKILL.md inside
    graduator = Graduator(graduator_skill_dir=fake_graduator_skill_dir)
    with pytest.raises(GraduatorError, match="no SKILL.md"):
        await graduator.graduate(skill_dir, tmp_path / "out")


# ───────── Driver error → orchestrator error ─────────


@pytest.mark.asyncio
async def test_driver_error_wrapped_with_details(
    fake_skill_dir: Path,
    fake_graduator_skill_dir: Path,
    tmp_path: Path,
) -> None:
    graduator = Graduator(graduator_skill_dir=fake_graduator_skill_dir)
    graduator.select_driver = lambda: _FailingDriver()  # type: ignore[method-assign]

    with pytest.raises(GraduatorError) as excinfo:
        await graduator.graduate(fake_skill_dir, tmp_path / "out")
    msg = str(excinfo.value)
    assert "failing" in msg
    assert "simulated failure" in msg
    assert "extra context here" in msg


@pytest.mark.asyncio
async def test_invalid_pipeline_yaml_raises_validation_error(
    fake_skill_dir: Path,
    fake_graduator_skill_dir: Path,
    tmp_path: Path,
) -> None:
    fake_driver = _FakeDriver(pipeline_yaml=INVALID_YAML)
    graduator = Graduator(graduator_skill_dir=fake_graduator_skill_dir)
    graduator.select_driver = lambda: fake_driver  # type: ignore[method-assign]

    with pytest.raises(GraduatorError, match="invalid pipeline.yaml"):
        await graduator.graduate(fake_skill_dir, tmp_path / "out")


# ───────── Default graduator skill dir resolution ─────────


@pytest.mark.asyncio
async def test_model_override_flows_through_to_driver(
    fake_skill_dir: Path,
    fake_graduator_skill_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the user passes ``model=...`` to Graduator, it should be
    forwarded to the driver's constructor via ``get_driver`` kwargs.
    We check this by spying on the registry factory that constructs
    the api driver."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    from rote.graduator.drivers.anthropic_api import AnthropicApiDriver

    constructed_models: list[str] = []

    class _SpyDriver(AnthropicApiDriver):
        def __init__(self, **kwargs: object) -> None:
            super().__init__(**kwargs)  # type: ignore[arg-type]
            constructed_models.append(self.model)

        async def run(self, skill_dir, graduator_skill_dir, work_dir):  # noqa: ANN001
            (work_dir / "pipeline.yaml").write_text(VALID_YAML, encoding="utf-8")
            from rote.graduator.drivers import DriverResult

            return DriverResult(
                pipeline_yaml_path=work_dir / "pipeline.yaml",
                work_dir=work_dir,
                driver_name="api",
                metadata={},
            )

    # Patch the registry factory so get_driver("api") returns our spy
    monkeypatch.setitem(
        # DRIVERS is the dict the registry consults
        __import__("rote.graduator.drivers", fromlist=["DRIVERS"]).DRIVERS,
        "api",
        lambda **kwargs: _SpyDriver(**kwargs),
    )

    graduator = Graduator(
        agent="api",
        graduator_skill_dir=fake_graduator_skill_dir,
        model="claude-opus-4-6",
    )
    await graduator.graduate(fake_skill_dir, tmp_path / "out")

    assert constructed_models == ["claude-opus-4-6"]


def test_default_graduator_skill_dir_finds_bundled_skill() -> None:
    """In an editable install, the default skill dir is the one in
    the rote source tree."""
    if not BUNDLED_GRADUATOR_SKILL.is_dir():
        pytest.skip("rote-graduate skill not found in source tree")

    graduator = Graduator()
    assert graduator.graduator_skill_dir == BUNDLED_GRADUATOR_SKILL
    assert (graduator.graduator_skill_dir / "SKILL.md").is_file()
