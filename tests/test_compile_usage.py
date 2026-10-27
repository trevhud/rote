"""A compile's token usage reads as measured, partial or unavailable (#295).

An empty driver metadata object used to become four zeros and
``usage_complete: true``: a codex compile that certainly spent tokens read
as free. These pin each status and the stages that make up the total.
"""

from __future__ import annotations

import json
from pathlib import Path

from rote.compiler.system1.skeleton import BRIEF_DIRNAME, DECISIONS_FILENAME
from rote.compiler.usage import compile_usage


def _decisions(out: Path, doc: dict[str, object]) -> None:
    path = out / BRIEF_DIRNAME / DECISIONS_FILENAME
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(doc), encoding="utf-8")


def _row(**tokens: object) -> dict[str, object]:
    return {"model": "m", **tokens}


def test_empty_driver_metadata_is_unavailable_not_zero(tmp_path: Path) -> None:
    usage = compile_usage("codex", {}, tmp_path)
    assert usage.status == "unavailable"
    assert usage.complete is False
    assert usage.tokens is None
    assert usage.agent_tokens is None
    assert usage.missing == ["agent: the codex driver reported no token usage"]


def test_an_explicit_zero_is_a_measured_zero(tmp_path: Path) -> None:
    """Zero reported is complete; zero inferred from nothing is not (above)."""
    usage = compile_usage("api", {"input_tokens": 0, "output_tokens": 0}, tmp_path)
    assert usage.status == "complete"
    assert usage.complete is True
    assert usage.tokens == {"input": 0, "output": 0, "cache_write": 0, "cache_read": 0}
    assert usage.missing == []


def test_an_unreported_structure_pass_keeps_the_total_partial(tmp_path: Path) -> None:
    """The agent's tokens are kept, and the stage nobody measured is named."""
    _decisions(
        tmp_path,
        {"structure": {"usage": [_row(input_tokens=None, output_tokens=None)]}},
    )
    agent = {"input_tokens": 10, "output_tokens": 4, "cache_read_tokens": 6}
    usage = compile_usage("codex", agent, tmp_path)
    assert usage.status == "partial"
    assert usage.complete is False
    assert usage.tokens == {"input": 10, "output": 4, "cache_write": 0, "cache_read": 6}
    assert usage.missing == ["structure: none of its 1 call(s) reported tokens"]
    assert [s.stage for s in usage.stages] == ["structure", "agent"]


def test_measured_stages_add_up_and_only_the_agent_is_priced_as_one_model(
    tmp_path: Path,
) -> None:
    _decisions(
        tmp_path,
        {
            "structure": {
                "usage": [
                    _row(input_tokens=100, output_tokens=20, cache_read_tokens=50),
                    _row(input_tokens=7, output_tokens=3),
                ]
            },
            "tiebreak": {"usage": [_row(input_tokens=1, output_tokens=1)]},
        },
    )
    agent = {"input_tokens": 1000, "output_tokens": 200, "cache_write_tokens": 5}
    usage = compile_usage("claude", agent, tmp_path)
    assert usage.status == "complete"
    assert usage.tokens == {"input": 1108, "output": 224, "cache_write": 5, "cache_read": 50}
    # System-1's calls are priced on their own (system1_cost_usd).
    assert usage.agent_tokens == {"input": 1000, "output": 200, "cache_write": 5, "cache_read": 0}


def test_a_structure_call_missing_its_counts_makes_that_stage_partial(tmp_path: Path) -> None:
    _decisions(
        tmp_path,
        {"structure": {"usage": [_row(input_tokens=9, output_tokens=2), _row(input_tokens=None)]}},
    )
    usage = compile_usage("claude", {"input_tokens": 1, "output_tokens": 1}, tmp_path)
    assert usage.status == "partial"
    assert usage.missing == ["structure: 1 of its 2 call(s) reported no tokens"]
    assert usage.tokens == {"input": 10, "output": 3, "cache_write": 0, "cache_read": 0}


def test_the_decision_model_records_cost_not_tokens(tmp_path: Path) -> None:
    _decisions(tmp_path, {"decider": {"name": "clef", "cost_usd": 0.001}})
    usage = compile_usage("claude", {"input_tokens": 1, "output_tokens": 1}, tmp_path)
    assert usage.status == "partial"
    assert usage.missing == ["decider: the clef model records its cost, not its tokens"]


def test_every_stage_unreported_is_unavailable(tmp_path: Path) -> None:
    _decisions(tmp_path, {"structure": {"usage": [_row(output_tokens=None)]}})
    usage = compile_usage("codex", {}, tmp_path)
    assert usage.status == "unavailable"
    assert usage.tokens is None
