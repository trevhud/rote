"""Tests for the high-level Compiler orchestrator.

The orchestrator is mocked at the driver layer — we don't actually
run an agent. The test driver is a class that implements the
CompilerDriver Protocol and writes a canned pipeline.yaml + a few
extra files into the work directory, mimicking what a real driver
would produce.

Coverage:

* Happy path: skill → driver.run → load_pipeline → move to output
* Output dir contains all artifacts the driver wrote (not just yaml)
* Auto-detect: when no driver available, helpful error
* Explicit agent: unknown name → CompilerError
* Explicit agent: unavailable → CompilerError with reason
* Skill validation: missing dir, missing SKILL.md → CompilerError
* Pipeline validation: invalid yaml → CompilerError
* Driver error → CompilerError with details preserved
* Default compiler skill dir resolves to the bundled rote-compile skill
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from rote.compiler import Compiler, CompilerError
from rote.compiler.drivers import CompilerDriver, DriverError, DriverResult
from rote.ir import load_pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
BUNDLED_COMPILER_SKILL = REPO_ROOT / "skills" / "rote-compile"


# ───────── Fakes ─────────


class _FakeDriver(CompilerDriver):
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
        compiler_skill_dir: Path,
        work_dir: Path,
        extra_instructions: str | None = None,
        on_event: object = None,
    ) -> DriverResult:
        self.run_called_with = {
            "skill_dir": skill_dir,
            "compiler_skill_dir": compiler_skill_dir,
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


class _FailingDriver(CompilerDriver):
    name = "failing"

    def is_available(self) -> tuple[bool, str]:
        return (True, "")

    async def run(
        self,
        skill_dir: Path,
        compiler_skill_dir: Path,
        work_dir: Path,
        extra_instructions: str | None = None,
        on_event: object = None,
    ) -> DriverResult:
        raise DriverError("simulated failure", details="extra context here")


VALID_YAML = """\
name: fake-pipeline
version: "0.1.0"
description: |
  A minimal pipeline used for compiler orchestrator testing.

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
    (skill_dir / "SKILL.md").write_text("---\nname: fake\n---\n\n# Fake skill\n", encoding="utf-8")
    return skill_dir


@pytest.fixture
def fake_compiler_skill_dir(tmp_path: Path) -> Path:
    grad_dir = tmp_path / "fake-compiler-skill"
    grad_dir.mkdir()
    (grad_dir / "SKILL.md").write_text("# fake compiler", encoding="utf-8")
    return grad_dir


# ───────── Happy path ─────────


@pytest.mark.asyncio
async def test_compile_happy_path(
    fake_skill_dir: Path,
    fake_compiler_skill_dir: Path,
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

    compiler = Compiler(compiler_skill_dir=fake_compiler_skill_dir)
    # Bypass select_driver so we don't need any real backend
    compiler.select_driver = lambda: fake_driver  # type: ignore[method-assign]

    result = await compiler.compile(fake_skill_dir, output_dir)

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
    assert fake_driver.run_called_with["compiler_skill_dir"] == fake_compiler_skill_dir


@pytest.mark.asyncio
async def test_compile_repoints_dead_source_skill_pointer(
    fake_skill_dir: Path,
    fake_compiler_skill_dir: Path,
    tmp_path: Path,
) -> None:
    """The agent records source_skill relative to its temp work dir, which
    is deleted after the run — a dead pointer that costs `rote eval` its
    before-side baseline. The orchestrator must re-point it to resolve
    from the pipeline.yaml's final location."""
    output_dir = tmp_path / "output"
    dead_pointer_yaml = VALID_YAML.replace(
        'version: "0.1.0"\n',
        'version: "0.1.0"\nsource_skill: ../../rote-compile-gone/skill\n',
    )
    compiler = Compiler(compiler_skill_dir=fake_compiler_skill_dir)
    compiler.select_driver = lambda: _FakeDriver(pipeline_yaml=dead_pointer_yaml)  # type: ignore[method-assign]

    result = await compiler.compile(fake_skill_dir, output_dir)

    reloaded = load_pipeline(output_dir / "pipeline.yaml")
    assert reloaded.source_skill is not None
    assert result.pipeline.source_skill == reloaded.source_skill
    resolved = (output_dir / reloaded.source_skill).resolve()
    assert resolved == fake_skill_dir.resolve()
    assert (resolved / "SKILL.md").is_file()
    # And the eval-side resolver actually finds it (the symptom that
    # motivated this: "source_skill did not resolve — after-side only").
    from rote.cli import _resolve_eval_skill_dir

    assert (
        _resolve_eval_skill_dir(None, output_dir / "pipeline.yaml", reloaded.source_skill)
        == resolved
    )


@pytest.mark.asyncio
async def test_compile_repoints_eval_sidecar_source_skill(
    fake_skill_dir: Path,
    fake_compiler_skill_dir: Path,
    tmp_path: Path,
) -> None:
    """eval.yaml suffers the same dead-pointer failure as pipeline.yaml
    (the agent records source_skill relative to its deleted temp work
    dir) — found on the push-to-coupa compilation, whose sidecar kept
    `../../skill` while pipeline.yaml was corrected."""
    output_dir = tmp_path / "output"
    dead_sidecar = (
        "version: 1\n"
        "source_skill: ../../skill\n"
        "steps:\n"
        "  - description: push each row\n"
        "    estimated_turns: {low: 3, high: 6}\n"
        "    iterations: {low: 20, high: 90}\n"
    )
    compiler = Compiler(compiler_skill_dir=fake_compiler_skill_dir)
    compiler.select_driver = lambda: _FakeDriver(  # type: ignore[method-assign]
        pipeline_yaml=VALID_YAML, extras={"eval.yaml": dead_sidecar}
    )

    await compiler.compile(fake_skill_dir, output_dir)

    from rote.eval.sidecar import load_eval_estimates

    sidecar = load_eval_estimates(output_dir / "eval.yaml")
    assert sidecar.source_skill is not None
    resolved = (output_dir / sidecar.source_skill).resolve()
    assert resolved == fake_skill_dir.resolve()
    # The rest of the agent's sidecar survives the surgical rewrite.
    assert sidecar.steps[0].iterations is not None
    assert sidecar.turn_range().high == 540


@pytest.mark.asyncio
async def test_compile_adds_source_skill_when_agent_omits_it(
    fake_skill_dir: Path,
    fake_compiler_skill_dir: Path,
    tmp_path: Path,
) -> None:
    """VALID_YAML has no source_skill at all — the orchestrator inserts a
    resolvable one rather than leaving the baseline undiscoverable."""
    output_dir = tmp_path / "output"
    compiler = Compiler(compiler_skill_dir=fake_compiler_skill_dir)
    compiler.select_driver = lambda: _FakeDriver(pipeline_yaml=VALID_YAML)  # type: ignore[method-assign]

    result = await compiler.compile(fake_skill_dir, output_dir)

    assert result.pipeline.source_skill is not None
    assert (output_dir / result.pipeline.source_skill).resolve() == fake_skill_dir.resolve()


@pytest.mark.asyncio
async def test_compile_overwrites_existing_output_files(
    fake_skill_dir: Path,
    fake_compiler_skill_dir: Path,
    tmp_path: Path,
) -> None:
    """If the output dir already has a pipeline.yaml, it gets replaced."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "pipeline.yaml").write_text("OLD CONTENT")

    fake_driver = _FakeDriver(pipeline_yaml=VALID_YAML)
    compiler = Compiler(compiler_skill_dir=fake_compiler_skill_dir)
    compiler.select_driver = lambda: fake_driver  # type: ignore[method-assign]

    await compiler.compile(fake_skill_dir, output_dir)

    written = (output_dir / "pipeline.yaml").read_text()
    assert "OLD CONTENT" not in written
    # Byte-identical to the driver's output except for the repointed
    # source_skill line the orchestrator inserts (see
    # test_compile_adds_source_skill_when_agent_omits_it).
    assert written.replace("source_skill: ../fake-skill\n", "") == VALID_YAML


@pytest.mark.asyncio
async def test_compile_stamps_provenance_sidecar(
    fake_skill_dir: Path,
    fake_compiler_skill_dir: Path,
    tmp_path: Path,
) -> None:
    """A successful compilation writes provenance.json next to the
    pipeline: section hashes for the whole SKILL.md plus the per-node
    section mapping the agent recorded."""
    import json

    yaml_with_source = VALID_YAML.replace(
        "    impl: extracted/foo.py:only_node\n",
        "    impl: extracted/foo.py:only_node\n    source:\n      section: Fake skill\n",
    )
    fake_driver = _FakeDriver(pipeline_yaml=yaml_with_source)
    compiler = Compiler(compiler_skill_dir=fake_compiler_skill_dir)
    compiler.select_driver = lambda: fake_driver  # type: ignore[method-assign]

    output_dir = tmp_path / "output"
    await compiler.compile(fake_skill_dir, output_dir)

    prov = json.loads((output_dir / "provenance.json").read_text())
    assert prov["version"] == 1
    assert "Fake skill" in prov["sections"]
    assert prov["nodes"]["only_node"]["section"] == "Fake skill"
    assert prov["nodes"]["only_node"]["content_hash"] == prov["sections"]["Fake skill"]


# ───────── Driver selection ─────────


@pytest.mark.asyncio
async def test_explicit_unknown_agent_raises(
    fake_skill_dir: Path,
    fake_compiler_skill_dir: Path,
    tmp_path: Path,
) -> None:
    compiler = Compiler(agent="bogus", compiler_skill_dir=fake_compiler_skill_dir)
    with pytest.raises(CompilerError, match="bogus"):
        await compiler.compile(fake_skill_dir, tmp_path / "out")


@pytest.mark.asyncio
async def test_explicit_unavailable_agent_raises_with_reason(
    fake_skill_dir: Path,
    fake_compiler_skill_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """User asks for `--agent claude` but claude isn't installed."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with patch("rote.compiler.drivers.claude.which", return_value=None):
        compiler = Compiler(agent="claude", compiler_skill_dir=fake_compiler_skill_dir)
        with pytest.raises(CompilerError, match="not available"):
            await compiler.compile(fake_skill_dir, tmp_path / "out")


@pytest.mark.asyncio
async def test_auto_detect_with_no_drivers_available_raises_helpful_error(
    fake_skill_dir: Path,
    fake_compiler_skill_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with (
        patch("rote.compiler.drivers.claude.which", return_value=None),
        patch("rote.compiler.drivers.codex.which", return_value=None),
    ):
        compiler = Compiler(compiler_skill_dir=fake_compiler_skill_dir)
        with pytest.raises(CompilerError) as excinfo:
            await compiler.compile(fake_skill_dir, tmp_path / "out")

    msg = str(excinfo.value)
    assert "claude" in msg
    assert "codex" in msg
    assert "api" in msg


# ───────── Skill validation ─────────


@pytest.mark.asyncio
async def test_missing_skill_dir_raises(
    fake_compiler_skill_dir: Path,
    tmp_path: Path,
) -> None:
    compiler = Compiler(compiler_skill_dir=fake_compiler_skill_dir)
    with pytest.raises(CompilerError, match="does not exist"):
        await compiler.compile(tmp_path / "nonexistent", tmp_path / "out")


@pytest.mark.asyncio
async def test_skill_dir_without_skill_md_raises(
    fake_compiler_skill_dir: Path,
    tmp_path: Path,
) -> None:
    skill_dir = tmp_path / "not-a-skill"
    skill_dir.mkdir()  # but no SKILL.md inside
    compiler = Compiler(compiler_skill_dir=fake_compiler_skill_dir)
    with pytest.raises(CompilerError, match="no SKILL.md"):
        await compiler.compile(skill_dir, tmp_path / "out")


# ───────── Driver error → orchestrator error ─────────


@pytest.mark.asyncio
async def test_driver_error_wrapped_with_details(
    fake_skill_dir: Path,
    fake_compiler_skill_dir: Path,
    tmp_path: Path,
) -> None:
    compiler = Compiler(compiler_skill_dir=fake_compiler_skill_dir)
    compiler.select_driver = lambda: _FailingDriver()  # type: ignore[method-assign]

    with pytest.raises(CompilerError) as excinfo:
        await compiler.compile(fake_skill_dir, tmp_path / "out")
    msg = str(excinfo.value)
    assert "failing" in msg
    assert "simulated failure" in msg
    assert "extra context here" in msg


@pytest.mark.asyncio
async def test_invalid_pipeline_yaml_raises_validation_error(
    fake_skill_dir: Path,
    fake_compiler_skill_dir: Path,
    tmp_path: Path,
) -> None:
    fake_driver = _FakeDriver(pipeline_yaml=INVALID_YAML)
    compiler = Compiler(compiler_skill_dir=fake_compiler_skill_dir)
    compiler.select_driver = lambda: fake_driver  # type: ignore[method-assign]

    with pytest.raises(CompilerError, match="invalid pipeline.yaml"):
        await compiler.compile(fake_skill_dir, tmp_path / "out")


# ───────── Default compiler skill dir resolution ─────────


@pytest.mark.asyncio
async def test_model_override_flows_through_to_driver(
    fake_skill_dir: Path,
    fake_compiler_skill_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the user passes ``model=...`` to Compiler, it should be
    forwarded to the driver's constructor via ``get_driver`` kwargs.
    We check this by spying on the registry factory that constructs
    the api driver."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    from rote.compiler.drivers.anthropic_api import AnthropicApiDriver

    constructed_models: list[str] = []

    class _SpyDriver(AnthropicApiDriver):
        def __init__(self, **kwargs: object) -> None:
            super().__init__(**kwargs)  # type: ignore[arg-type]
            constructed_models.append(self.model)

        async def run(  # noqa: ANN001
            self, skill_dir, compiler_skill_dir, work_dir, extra_instructions=None, on_event=None
        ):
            (work_dir / "pipeline.yaml").write_text(VALID_YAML, encoding="utf-8")
            from rote.compiler.drivers import DriverResult

            return DriverResult(
                pipeline_yaml_path=work_dir / "pipeline.yaml",
                work_dir=work_dir,
                driver_name="api",
                metadata={},
            )

    # Patch the registry factory so get_driver("api") returns our spy
    monkeypatch.setitem(
        # DRIVERS is the dict the registry consults
        __import__("rote.compiler.drivers", fromlist=["DRIVERS"]).DRIVERS,
        "api",
        lambda **kwargs: _SpyDriver(**kwargs),
    )

    compiler = Compiler(
        agent="api",
        compiler_skill_dir=fake_compiler_skill_dir,
        model="claude-opus-4-6",
    )
    await compiler.compile(fake_skill_dir, tmp_path / "out")

    assert constructed_models == ["claude-opus-4-6"]


def test_default_compiler_skill_dir_finds_bundled_skill() -> None:
    """In an editable install, the default skill dir is the one in
    the rote source tree."""
    if not BUNDLED_COMPILER_SKILL.is_dir():
        pytest.skip("rote-compile skill not found in source tree")

    compiler = Compiler()
    assert compiler.compiler_skill_dir == BUNDLED_COMPILER_SKILL
    assert (compiler.compiler_skill_dir / "SKILL.md").is_file()


def test_completion_message_reports_the_whole_input_volume() -> None:
    """The one-line summary must not report a cached run as ~0 input.

    This line is what a human sees at the end of a compilation. With a
    prompt-cached driver the plain ``input_tokens`` field holds almost
    nothing, so printing it alone announced a 22-minute, 62-turn run as
    ``tokens in=113``.
    """
    result = DriverResult(
        driver_name="claude",
        pipeline_yaml_path=Path("pipeline.yaml"),
        work_dir=Path("."),
        metadata={
            "input_tokens": 113,
            "output_tokens": 1919,
            "cache_write_tokens": 122_956,
            "cache_read_tokens": 41_000,
            "num_turns": 62,
            "cost_usd": 2.71,
        },
    )
    message = Compiler._completion_message(result)
    assert "in=164069" in message
    assert "(163956 cached)" in message
    assert "in=113" not in message
    assert "turns=62" in message
    assert "cost=$2.71" in message

    # An uncached driver keeps the plain rendering and grows no cache clause.
    plain = Compiler._completion_message(
        DriverResult(
            driver_name="api",
            pipeline_yaml_path=Path("pipeline.yaml"),
            work_dir=Path("."),
            metadata={"input_tokens": 1000, "output_tokens": 500},
        )
    )
    assert "tokens in=1000 out=500" in plain
    assert "cached" not in plain
