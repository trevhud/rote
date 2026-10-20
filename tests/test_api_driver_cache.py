"""The api driver prompt-caches its conversation and prices its own run.

Uncached, every turn re-billed the whole conversation at the full input
rate (a measured expense-audit compile: $3.67 over 28 turns, 948k uncached
input tokens, vs $0.69 on the cached subscription lane). And its cost only
reached progress events, so compile-metrics.json had none and `rote bench
--budget` counted an api compile as $0.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from rote.compiler.drivers.anthropic_api import AnthropicApiDriver, cached_messages
from tests.test_anthropic_driver import (
    VALID_PIPELINE_YAML,
    _text,
    _tool_use,
    fake_anthropic,  # noqa: F401 (fixture)
    fake_skills,  # noqa: F401 (fixture)
)


def _msg(content: list[Any], stop: str, **usage: int) -> SimpleNamespace:
    return SimpleNamespace(
        content=content,
        stop_reason=stop,
        usage=SimpleNamespace(
            input_tokens=usage.get("input", 10),
            output_tokens=usage.get("output", 5),
            cache_creation_input_tokens=usage.get("write", 0),
            cache_read_input_tokens=usage.get("read", 0),
        ),
    )


def _breakpoints(request: dict[str, Any]) -> list[str]:
    where = []
    where += ["tools"] * sum("cache_control" in t for t in request["tools"])
    system = request["system"]
    where += ["system"] * sum("cache_control" in b for b in system if isinstance(b, dict))
    for i, m in enumerate(request["messages"]):
        content = m["content"]
        if isinstance(content, list):
            where += [f"messages[{i}]"] * sum(
                isinstance(b, dict) and "cache_control" in b for b in content
            )
    return where


def _catalog(rates: Any) -> Any:
    return SimpleNamespace(rates_for=lambda model: rates)


async def _run(fake_anthropic: Any, fake_skills: Any) -> tuple[Any, Any]:  # noqa: F811
    skill_dir, compiler_dir, work_dir = fake_skills
    work_dir.mkdir()
    client = fake_anthropic(
        [
            _msg(
                [_tool_use("r", "read_file", {"path": str(skill_dir / "SKILL.md")})],
                "tool_use",
                input=900,
                write=12000,
                read=0,
            ),
            _msg(
                [
                    _tool_use(
                        "w",
                        "write_file",
                        {"path": str(work_dir / "pipeline.yaml"), "content": VALID_PIPELINE_YAML},
                    )
                ],
                "tool_use",
                input=40,
                write=600,
                read=12000,
            ),
            _msg([_text("done")], "end_turn", input=8, write=300, read=12600, output=50),
        ]
    )
    result = await AnthropicApiDriver().run(skill_dir, compiler_dir, work_dir)
    return client, result


@pytest.mark.asyncio
async def test_every_request_caches_tools_system_and_the_latest_user_turn(
    fake_anthropic: Any,  # noqa: F811
    fake_skills: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _ = await _run(fake_anthropic, fake_skills)
    for i, request in enumerate(client.messages.calls):
        # One rolling breakpoint, always on the last message (a user turn):
        # three of the API's four, every turn.
        last = len(request["messages"]) - 1
        assert _breakpoints(request) == ["tools", "system", f"messages[{last}]"], i
        assert request["system"][0]["text"]


@pytest.mark.asyncio
async def test_the_history_is_not_marked_only_each_request_is(
    fake_anthropic: Any,  # noqa: F811
    fake_skills: Any,  # noqa: F811
) -> None:
    client, _ = await _run(fake_anthropic, fake_skills)
    # The third request's history holds the first user turn unmarked: had the
    # breakpoint been written into the history, it would accumulate past four.
    third = client.messages.calls[2]["messages"]
    assert isinstance(third[0]["content"], str)


def test_a_trailing_assistant_turn_keeps_the_breakpoint_on_the_user_turn_before() -> None:
    messages = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": [SimpleNamespace(type="server_tool_use")]},
    ]
    sent = cached_messages(messages)
    assert sent[0]["content"] == [
        {"type": "text", "text": "go", "cache_control": {"type": "ephemeral"}}
    ]
    assert sent[1] is messages[1]
    assert messages[0]["content"] == "go"


@pytest.mark.asyncio
async def test_all_four_buckets_are_summed_and_priced_into_the_metadata(
    fake_anthropic: Any,  # noqa: F811
    fake_skills: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Sonnet 4.6's published rates per Mtok: input, output, cache read, cache write.
    monkeypatch.setattr(
        "rote.eval.pricing.fetch_catalog", lambda provider: _catalog((3.0, 15.0, 0.3, 3.75))
    )
    _, result = await _run(fake_anthropic, fake_skills)
    meta = result.metadata
    assert (
        meta["input_tokens"],
        meta["output_tokens"],
        meta["cache_write_tokens"],
        meta["cache_read_tokens"],
    ) == (948, 60, 12900, 24600)
    expected = (948 * 3.0 + 60 * 15.0 + 24600 * 0.3 + 12900 * 3.75) / 1e6
    assert meta["cost_usd"] == pytest.approx(expected)
    # The negative half: pricing the plain pair alone undercounts it.
    assert meta["cost_usd"] != pytest.approx((948 * 3.0 + 60 * 15.0) / 1e6)


@pytest.mark.asyncio
@pytest.mark.parametrize("rates", [None, (3.0, 15.0, None, None)])
async def test_a_run_that_cannot_be_priced_reports_no_cost_not_a_wrong_one(
    fake_anthropic: Any,  # noqa: F811
    fake_skills: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    rates: Any,
) -> None:
    monkeypatch.setattr("rote.eval.pricing.fetch_catalog", lambda provider: _catalog(rates))
    _, result = await _run(fake_anthropic, fake_skills)
    assert result.metadata["cost_usd"] is None


@pytest.mark.asyncio
async def test_bench_budget_sees_an_api_compiles_cost(
    tmp_path: Path,
    fake_anthropic: Any,  # noqa: F811
    fake_skills: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The metadata reaches compile-metrics.json, which bench's collect reads."""
    from rote.bench.collect import read_compile_metrics
    from rote.bench.runner import _Budget
    from rote.compiler import _write_compile_metrics

    monkeypatch.setattr(
        "rote.eval.pricing.fetch_catalog", lambda provider: _catalog((3.0, 15.0, 0.3, 3.75))
    )
    _, result = await _run(fake_anthropic, fake_skills)
    out = tmp_path / "compiled"
    out.mkdir()
    _write_compile_metrics(result, out)
    assert json.loads((out / "compile-metrics.json").read_text())["metadata"]["cost_usd"] > 0
    budget = _Budget(limit=result.metadata["cost_usd"] / 2)
    budget.add(read_compile_metrics(out).get("cost_usd"))
    assert budget.exhausted


@pytest.mark.asyncio
async def test_the_api_lane_gets_the_claude_lanes_task_and_the_real_reference_list(
    fake_anthropic: Any,  # noqa: F811
    fake_skills: Any,  # noqa: F811
) -> None:
    """The api lane's own prompt never mentioned tests/ and told the agent to
    stop once pipeline.yaml was written; its system prompt listed four
    hardcoded references, none of which a System-1 rubric has."""
    from rote.compiler.drivers.claude import ClaudeDriver

    skill_dir, compiler_dir, work_dir = fake_skills
    (compiler_dir / "references" / "implementation.md").write_text("x", encoding="utf-8")
    client, _ = await _run(fake_anthropic, fake_skills)
    first = client.messages.calls[0]
    task = first["messages"][0]["content"][0]["text"]
    resolved = (skill_dir.resolve(), compiler_dir.resolve(), work_dir.resolve())
    assert task == ClaudeDriver()._build_user_prompt(*resolved)
    assert f"their tests to {work_dir.resolve()}/tests/" in task
    system = first["system"][0]["text"]
    assert str(compiler_dir.resolve() / "references" / "implementation.md") in system
    assert "node-kinds.md" not in system
    assert "Do not call any further tools" not in system


# ───────── The work directory is readable, and relative paths mean it ─────────


@pytest.mark.asyncio
async def test_the_agent_reads_the_system1_skeleton_from_the_work_dir_by_relative_path(
    tmp_path: Path,
    fake_anthropic: Any,  # noqa: F811
    fake_skills: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """System-1 writes pipeline.yaml and system1/brief.md before the agent
    starts. The api lane could not read either (the work dir was not a read
    root), so a live compile wrote its own plan beside System-1's and every
    function disagreed with its node's inputs."""
    skill_dir, compiler_dir, work_dir = fake_skills
    (work_dir / "system1").mkdir(parents=True)
    (work_dir / "pipeline.yaml").write_text(VALID_PIPELINE_YAML, encoding="utf-8")
    (work_dir / "system1" / "brief.md").write_text("the brief", encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)  # the rote process's cwd is not the work dir
    client = fake_anthropic(
        [
            _msg(
                [
                    _tool_use("a", "read_file", {"path": "pipeline.yaml"}),
                    _tool_use("b", "read_file", {"path": "system1/brief.md"}),
                    _tool_use("c", "list_directory", {"path": "system1"}),
                    _tool_use("d", "write_file", {"path": "extracted/x.py", "content": "y = 1\n"}),
                    _tool_use("e", "write_file", {"path": "../escape.py", "content": "z"}),
                    _tool_use("f", "read_file", {"path": str(tmp_path / "secret.txt")}),
                ],
                "tool_use",
            ),
            _msg([_text("done")], "end_turn"),
        ]
    )
    (tmp_path / "secret.txt").write_text("s", encoding="utf-8")
    await AnthropicApiDriver().run(skill_dir, compiler_dir, work_dir)
    results = {b["tool_use_id"]: b for b in client.messages.calls[1]["messages"][-1]["content"]}

    def text(i: str) -> str:
        content = results[i]["content"]
        return content if isinstance(content, str) else "".join(c["text"] for c in content)

    assert text("a") == VALID_PIPELINE_YAML and not results["a"].get("is_error")
    assert text("b") == "the brief"
    assert text("c") == "brief.md"
    assert (work_dir / "extracted" / "x.py").read_text() == "y = 1\n"
    assert not (elsewhere / "extracted").exists()
    assert results["e"].get("is_error") and not (work_dir.parent / "escape.py").exists()
    assert results["f"].get("is_error") and text("f") != "s"
