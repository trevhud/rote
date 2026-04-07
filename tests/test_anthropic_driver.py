"""Tests for the AnthropicApiDriver run() implementation.

We mock ``anthropic.AsyncAnthropic`` to return canned message
responses so the tool-use loop runs end-to-end without an API call.
The driver doesn't care that the responses come from a fake — it
only inspects the duck-typed attributes (``content``, ``stop_reason``,
``usage``, ``type``, ``id``, ``name``, ``input``).

Coverage:

* Happy path: read SKILL.md → write pipeline.yaml → end_turn
* Tool dispatch: read_file, list_directory, write_file all wired up
* Path security: read outside read_roots and write outside work_dir
  both rejected as tool errors (returned to the LLM, not crashing
  the driver)
* Failure: agent ends without producing pipeline.yaml → DriverError
* Failure: max_iterations exceeded → DriverError
* Token accounting: input/output tokens summed across turns
* Path traversal: ``..`` in a path is resolved before security check
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from rote.graduator.drivers import DriverError
from rote.graduator.drivers.anthropic_api import (
    AnthropicApiDriver,
    _handle_list_directory,
    _handle_read_file,
    _handle_write_file,
)


# ───────── Fake Anthropic SDK ─────────


class _FakeMessages:
    """Stand-in for ``client.messages``.

    The driver calls ``await client.messages.create(**kwargs)``. We
    return canned responses in order, recording each call's kwargs so
    tests can introspect.

    Note: ``messages`` is captured by snapshot (a shallow copy of the
    list) because the driver mutates its local ``messages`` list
    in-place by appending assistant/tool-result turns. Without the
    snapshot, every recorded call would point at the same final list
    by reference and tests couldn't see what was actually sent.
    """

    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        snapshot = dict(kwargs)
        if "messages" in snapshot:
            snapshot["messages"] = list(snapshot["messages"])
        self.calls.append(snapshot)
        if not self._responses:
            raise RuntimeError(
                "FakeMessages out of canned responses; "
                "test set up too few turns"
            )
        return self._responses.pop(0)


class _FakeAsyncAnthropic:
    """Replaces ``anthropic.AsyncAnthropic``."""

    def __init__(self, responses: list[Any]) -> None:
        self.messages = _FakeMessages(responses)


@pytest.fixture
def fake_anthropic(monkeypatch: pytest.MonkeyPatch):  # noqa: ANN201
    """Patch ``anthropic.AsyncAnthropic`` to return a fake client.

    Usage::

        client = fake_anthropic([response1, response2, ...])
        # ... run the driver ...
        assert len(client.messages.calls) == N
    """
    holder: dict[str, Any] = {}

    def _patch(responses: list[Any]) -> _FakeAsyncAnthropic:
        client = _FakeAsyncAnthropic(responses)
        holder["client"] = client
        # Patch the symbol the driver actually calls
        monkeypatch.setattr(
            "rote.graduator.drivers.anthropic_api.anthropic.AsyncAnthropic",
            lambda *a, **k: client,
        )
        # Bypass the env-var check
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
        return client

    return _patch


# ───────── Helpers for building canned responses ─────────


def _tool_use(tool_id: str, name: str, input_: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        type="tool_use",
        id=tool_id,
        name=name,
        input=input_,
    )


def _text(content: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=content)


def _msg(
    content: list[SimpleNamespace],
    *,
    stop_reason: str,
    input_tokens: int = 100,
    output_tokens: int = 50,
) -> SimpleNamespace:
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
    )


# A small but valid pipeline.yaml the test agent will write.
VALID_PIPELINE_YAML = """\
name: fake-pipeline
version: "0.1.0"
description: |
  A minimal pipeline used for driver testing.

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


# ───────── Skill bundle fixture ─────────


@pytest.fixture
def fake_skills(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create a fake source skill, fake graduator skill, and work dir."""
    skill_dir = tmp_path / "fake-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: fake\n---\n\n# Fake skill\n\nThis is the source skill.\n",
        encoding="utf-8",
    )

    graduator_dir = tmp_path / "fake-graduator-skill"
    graduator_dir.mkdir()
    (graduator_dir / "SKILL.md").write_text(
        "# Fake rote-graduate skill\n\nDo the graduation thing.\n",
        encoding="utf-8",
    )
    (graduator_dir / "references").mkdir()

    work_dir = tmp_path / "work"
    return skill_dir, graduator_dir, work_dir


# ───────── Happy path ─────────


@pytest.mark.asyncio
async def test_happy_path_read_then_write_then_end(
    fake_skills: tuple[Path, Path, Path],
    fake_anthropic,  # noqa: ANN001
) -> None:
    """The driver reads a file, writes pipeline.yaml, ends. We check
    the result and the file."""
    skill_dir, graduator_dir, work_dir = fake_skills
    work_dir.mkdir()

    # Use the resolved path because the driver resolves before passing
    # to the system prompt and to security checks.
    skill_md_abs = (skill_dir / "SKILL.md").resolve()
    pipeline_yaml_abs = (work_dir / "pipeline.yaml").resolve()

    responses = [
        # Turn 1: model reads SKILL.md
        _msg(
            [_tool_use("tu1", "read_file", {"path": str(skill_md_abs)})],
            stop_reason="tool_use",
            input_tokens=200,
            output_tokens=40,
        ),
        # Turn 2: model writes pipeline.yaml
        _msg(
            [
                _tool_use(
                    "tu2",
                    "write_file",
                    {
                        "path": str(pipeline_yaml_abs),
                        "content": VALID_PIPELINE_YAML,
                    },
                )
            ],
            stop_reason="tool_use",
            input_tokens=300,
            output_tokens=500,
        ),
        # Turn 3: model says it's done
        _msg(
            [_text("Done. Pipeline written.")],
            stop_reason="end_turn",
            input_tokens=100,
            output_tokens=20,
        ),
    ]
    fake_client = fake_anthropic(responses)

    driver = AnthropicApiDriver()
    result = await driver.run(
        skill_dir=skill_dir,
        graduator_skill_dir=graduator_dir,
        work_dir=work_dir,
    )

    # Result shape
    assert result.driver_name == "api"
    assert result.pipeline_yaml_path == pipeline_yaml_abs
    assert result.pipeline_yaml_path.is_file()
    assert result.pipeline_yaml_path.read_text() == VALID_PIPELINE_YAML

    # Metadata
    from rote.graduator.drivers.anthropic_api import DEFAULT_MODEL
    assert result.metadata["model"] == DEFAULT_MODEL
    assert result.metadata["iterations"] == 3
    assert result.metadata["input_tokens"] == 600  # 200+300+100
    assert result.metadata["output_tokens"] == 560  # 40+500+20

    # Three calls were made to messages.create
    assert len(fake_client.messages.calls) == 3


@pytest.mark.asyncio
async def test_list_directory_tool_works(
    fake_skills: tuple[Path, Path, Path],
    fake_anthropic,  # noqa: ANN001
) -> None:
    """Verify the list_directory tool path is wired correctly."""
    skill_dir, graduator_dir, work_dir = fake_skills
    work_dir.mkdir()

    # Add an extra file so the listing has something to show
    (skill_dir / "extra.md").write_text("hi")

    responses = [
        _msg(
            [_tool_use("tu1", "list_directory", {"path": str(skill_dir.resolve())})],
            stop_reason="tool_use",
        ),
        _msg(
            [
                _tool_use(
                    "tu2",
                    "write_file",
                    {
                        "path": str((work_dir / "pipeline.yaml").resolve()),
                        "content": VALID_PIPELINE_YAML,
                    },
                )
            ],
            stop_reason="tool_use",
        ),
        _msg([_text("done")], stop_reason="end_turn"),
    ]
    fake_anthropic(responses)

    driver = AnthropicApiDriver()
    result = await driver.run(skill_dir, graduator_dir, work_dir)
    assert result.pipeline_yaml_path.is_file()


# ───────── Path security ─────────


def test_read_file_rejects_path_outside_roots(tmp_path: Path) -> None:
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("hi")

    other = tmp_path / "other"
    other.mkdir()
    (other / "secret.md").write_text("nope")

    with pytest.raises(PermissionError, match="not within allowed read roots"):
        _handle_read_file(str(other / "secret.md"), [skill_dir])


def test_read_file_resolves_traversal_attempts(tmp_path: Path) -> None:
    """`..` segments are resolved before the security check, so a path
    that *appears* to be inside the root but actually escapes is
    blocked."""
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("hi")

    sneaky = skill_dir / ".." / "outside.md"
    (tmp_path / "outside.md").write_text("oops")

    with pytest.raises(PermissionError):
        _handle_read_file(str(sneaky), [skill_dir])


def test_read_file_rejects_directory(tmp_path: Path) -> None:
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        _handle_read_file(str(skill_dir), [skill_dir])


def test_write_file_rejects_path_outside_work_dir(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    other = tmp_path / "other"
    other.mkdir()

    with pytest.raises(PermissionError, match="not within allowed write root"):
        _handle_write_file(str(other / "evil.py"), "content", work_dir)


def test_write_file_creates_parent_directories(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    target = work_dir / "extracted" / "deep" / "foo.py"

    msg = _handle_write_file(str(target), "x = 1\n", work_dir)
    assert "Wrote" in msg
    assert target.is_file()
    assert target.read_text() == "x = 1\n"


def test_list_directory_rejects_outside_roots(tmp_path: Path) -> None:
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    other = tmp_path / "other"
    other.mkdir()

    with pytest.raises(PermissionError):
        _handle_list_directory(str(other), [skill_dir])


@pytest.mark.asyncio
async def test_path_security_violations_returned_to_model_as_tool_errors(
    fake_skills: tuple[Path, Path, Path],
    fake_anthropic,  # noqa: ANN001
    tmp_path: Path,
) -> None:
    """When the model tries to read outside the allowed roots, the
    driver returns a tool_result with is_error=True instead of crashing.
    The model can then recover."""
    skill_dir, graduator_dir, work_dir = fake_skills
    work_dir.mkdir()

    # Some file outside both allowed read roots
    forbidden = tmp_path / "forbidden.md"
    forbidden.write_text("nope")

    responses = [
        # Turn 1: model attempts forbidden read
        _msg(
            [_tool_use("tu1", "read_file", {"path": str(forbidden)})],
            stop_reason="tool_use",
        ),
        # Turn 2: model writes pipeline.yaml (after seeing the error)
        _msg(
            [
                _tool_use(
                    "tu2",
                    "write_file",
                    {
                        "path": str((work_dir / "pipeline.yaml").resolve()),
                        "content": VALID_PIPELINE_YAML,
                    },
                )
            ],
            stop_reason="tool_use",
        ),
        _msg([_text("done")], stop_reason="end_turn"),
    ]
    fake_client = fake_anthropic(responses)

    driver = AnthropicApiDriver()
    result = await driver.run(skill_dir, graduator_dir, work_dir)

    # The driver completed despite the forbidden read
    assert result.pipeline_yaml_path.is_file()

    # The second messages.create call should contain a tool_result
    # marked is_error=True for the forbidden read.
    second_call_messages = fake_client.messages.calls[1]["messages"]
    user_turn = second_call_messages[-1]
    assert user_turn["role"] == "user"
    tool_results = user_turn["content"]
    assert len(tool_results) == 1
    assert tool_results[0]["is_error"] is True
    assert "not within allowed read roots" in tool_results[0]["content"]


# ───────── Failure modes ─────────


@pytest.mark.asyncio
async def test_max_iterations_exceeded_raises_driver_error(
    fake_skills: tuple[Path, Path, Path],
    fake_anthropic,  # noqa: ANN001
) -> None:
    """If the model never stops calling tools, the driver gives up
    after max_iterations."""
    skill_dir, graduator_dir, work_dir = fake_skills
    work_dir.mkdir()

    # Build a long chain of tool_use responses; never end_turn
    responses = [
        _msg(
            [_tool_use(f"tu{i}", "read_file", {"path": str(skill_dir / "SKILL.md")})],
            stop_reason="tool_use",
        )
        for i in range(10)
    ]
    fake_anthropic(responses)

    driver = AnthropicApiDriver(max_iterations=3)
    with pytest.raises(DriverError, match="did not complete within 3 iterations"):
        await driver.run(skill_dir, graduator_dir, work_dir)


@pytest.mark.asyncio
async def test_missing_pipeline_yaml_raises_driver_error(
    fake_skills: tuple[Path, Path, Path],
    fake_anthropic,  # noqa: ANN001
) -> None:
    """If the agent ends without producing pipeline.yaml, error out."""
    skill_dir, graduator_dir, work_dir = fake_skills
    work_dir.mkdir()

    responses = [
        # Just end the turn without writing anything
        _msg([_text("I am done but did nothing")], stop_reason="end_turn"),
    ]
    fake_anthropic(responses)

    driver = AnthropicApiDriver()
    with pytest.raises(DriverError, match="did not produce"):
        await driver.run(skill_dir, graduator_dir, work_dir)


@pytest.mark.asyncio
async def test_missing_graduator_skill_md_raises_driver_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If graduator_skill_dir is missing SKILL.md, error before the
    LLM is called."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")

    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("hi")

    bad_graduator = tmp_path / "no-skill-md"
    bad_graduator.mkdir()  # but no SKILL.md inside

    work_dir = tmp_path / "work"

    driver = AnthropicApiDriver()
    with pytest.raises(DriverError, match="rote-graduate SKILL.md not found"):
        await driver.run(skill_dir, bad_graduator, work_dir)
