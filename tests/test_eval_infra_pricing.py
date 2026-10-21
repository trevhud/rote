"""Cloudflare Workflows infrastructure pricing: parse, cache, per-run cost.

The fixture is a verbatim copy of
``cloudflare/cloudflare-docs@production:src/content/partials/workflows/workflows-pricing.mdx``
as fetched on 2026-10-18 (last changed upstream in 5cbc81ed, 2026-09-09).
No test touches the network.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from rote.cli import main
from rote.eval.estimate import FanOutWidth, Range
from rote.eval.infra_pricing import (
    WORKFLOWS_PRICING_PATH,
    cloudflare_infra_cost,
    cloudflare_steps_per_run,
    fetch_workflows_pricing,
    parse_workflows_pricing,
)
from rote.eval.pricing import PricingError
from rote.eval.priors import Priors
from rote.ir import Pipeline

FIXTURE = Path(__file__).parent / "data" / "cloudflare" / "workflows-pricing.mdx"
MDX = FIXTURE.read_text(encoding="utf-8")
SOURCE = f"cloudflare/cloudflare-docs@production:{WORKFLOWS_PRICING_PATH}"


def _parse(mdx: str = MDX):  # type: ignore[no-untyped-def]
    return parse_workflows_pricing(mdx, source=SOURCE, fetched_at="2026-10-18T00:00:00Z")


# ───────── parsing ─────────


def test_parses_every_paid_rate_from_the_real_table() -> None:
    pricing = _parse()

    assert pricing.steps.included_per_month == 500_000
    assert pricing.steps.usd == 0.80
    assert pricing.steps.per == 100_000
    assert pricing.requests.included_per_month == 10_000_000
    assert pricing.requests.usd == 0.30
    assert pricing.requests.per == 1_000_000
    assert pricing.cpu_ms.included_per_month == 30_000_000
    assert pricing.cpu_ms.usd == 0.02
    assert pricing.cpu_ms.per == 1_000_000
    assert pricing.storage_gb_month.included_per_month == 1
    assert pricing.storage_gb_month.usd == 0.20
    assert pricing.storage_gb_month.per == 1


@pytest.mark.parametrize("row", ["Requests", "CPU time", "Storage", "Steps"])
def test_a_table_missing_any_row_is_rejected(row: str) -> None:
    """A missing row must fail loudly. A parser that skipped it would price
    that unit at nothing and the scorecard would understate the run."""
    without = "\n".join(line for line in MDX.splitlines() if not line.startswith(f"| {row}"))
    assert without != MDX
    with pytest.raises(PricingError, match=f"expected one {row!r} row, found 0"):
        _parse(without)


def test_a_reworded_cell_is_rejected_not_guessed() -> None:
    reworded = MDX.replace(
        "500,000 included per month + $0.80/ additional 100,000 per month",
        "500,000 per month, then 80 cents for each 100,000 more",
    )
    with pytest.raises(PricingError, match="'Steps' paid-plan cell no longer parses"):
        _parse(reworded)


@pytest.mark.parametrize(
    ("published", "reworded"),
    [
        # "100K" would read as a per of 100: a 1000x overstatement.
        (
            "500,000 included per month + $0.80/ additional 100,000 per month",
            "500,000 included per month + $0.80 per additional 100K steps",
        ),
        # "1M" would read as a per of 1: a millionfold overstatement.
        (
            "10 million included per month + $0.30 per additional million",
            "10 million included per month + $0.30 per additional 1M",
        ),
    ],
)
def test_an_unknown_unit_phrase_is_rejected_not_misread(published: str, reworded: str) -> None:
    """The cell grammar is closed and anchored at both ends, so wording it
    has not seen raises instead of parsing a prefix of it."""
    assert published in MDX
    with pytest.raises(PricingError, match="paid-plan cell no longer parses"):
        _parse(MDX.replace(published, reworded))


def test_a_table_without_the_paid_column_is_rejected() -> None:
    with pytest.raises(PricingError, match="no 'Workers Paid' column"):
        _parse(MDX.replace("Workers Paid", "Workers Standard"))


# ───────── fetch + cache ─────────


def test_offline_with_no_cache_is_a_pricing_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def _down(url: str, timeout: float) -> str:
        raise PricingError(f"fetch of {url} failed: offline")

    monkeypatch.setattr("rote.eval.infra_pricing.get_text", _down)
    with pytest.raises(PricingError, match="offline"):
        fetch_workflows_pricing(cache_path=tmp_path / "cf.json")


def test_a_fresh_fetch_is_cached_and_reused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []

    def _fetch(url: str, timeout: float) -> str:
        calls.append(url)
        return MDX

    monkeypatch.setattr("rote.eval.infra_pricing.get_text", _fetch)
    cache = tmp_path / "cf.json"
    first = fetch_workflows_pricing(cache_path=cache)
    second = fetch_workflows_pricing(cache_path=cache)

    assert calls == [
        "https://raw.githubusercontent.com/cloudflare/cloudflare-docs/production/"
        "src/content/partials/workflows/workflows-pricing.mdx"
    ]
    assert second == first
    assert first.source == SOURCE


def test_a_stale_cache_refetches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cache = tmp_path / "cf.json"
    cache.write_text(
        json.dumps(
            {
                "cached_at_epoch": time.time() - 25 * 3600,
                "fetched_at": "2026-10-16T00:00:00Z",
                "mdx": MDX,
            }
        ),
        encoding="utf-8",
    )

    def _down(url: str, timeout: float) -> str:
        raise PricingError("offline")

    monkeypatch.setattr("rote.eval.infra_pricing.get_text", _down)
    with pytest.raises(PricingError, match="offline"):
        fetch_workflows_pricing(cache_path=cache)


# ───────── per-run cost ─────────


_PIPELINE = Pipeline.model_validate(
    {
        "name": "cf-bill",
        "input": {"type": "In", "required": [], "optional": []},
        "nodes": [
            {
                "id": "list_posts",
                "kind": "pure_function",
                "description": "list",
                "impl": "posts.py:list_posts",
            },
            {
                "id": "score_post",
                "kind": "pure_function",
                "description": "score one",
                "impl": "posts.py:score_post",
                "fan_out": True,
                "inputs": {"post": "list_posts.output.posts"},
            },
            {"id": "approve", "kind": "hitl_gate", "description": "approve", "signal": "go"},
            {
                "id": "research",
                "kind": "agent_loop",
                "description": "research",
                "tools": ["web_search"],
                "loop_body": ["summarize"],
            },
            {
                "id": "summarize",
                "kind": "pure_function",
                "description": "loop body only",
                "impl": "posts.py:summarize",
            },
        ],
        "edges": [
            {"from": "list_posts", "to": "score_post", "fan_out": True},
            {"from": "score_post", "to": "approve"},
            {"from": "approve", "to": "research"},
        ],
        "entry_nodes": ["list_posts"],
        "exit_nodes": ["research"],
    }
)


def test_steps_count_fan_out_elements_and_gates_but_not_loop_bodies() -> None:
    priors = Priors(fan_out_elements_per_run=7)
    # list_posts 1 + score_post 7 + approve 1 + research 1; summarize runs
    # inside research's step.
    assert cloudflare_steps_per_run(_PIPELINE, {}, priors) == Range.exact(10)


def test_steps_use_the_same_fan_width_as_the_token_estimate() -> None:
    """deal-monitor's eval.yaml says 3 to 20 threads. The LLM dollars used
    that range, so the step count must too, not the prior's 10."""
    width = FanOutWidth(node_id="score_post", elements=Range(3, 20), source="eval.yaml iterations")
    steps = cloudflare_steps_per_run(_PIPELINE, {"score_post": width}, Priors())
    assert steps == Range(1 + 3 + 1 + 1, 1 + 20 + 1 + 1)


def test_per_run_dollars_follow_the_published_rates() -> None:
    # Default retention is the Workers Paid account maximum, 30 days.
    priors = Priors(fan_out_elements_per_run=7)
    assert priors.cloudflare_retention_days == 30
    infra = cloudflare_infra_cost(_PIPELINE, _parse(), priors, runtime_label="Cloudflare")

    by_label = {line.label: line.usd.low for line in infra.lines}
    assert by_label["Steps"] == pytest.approx(10 * 0.80 / 100_000)
    assert by_label["Requests"] == pytest.approx(0.30 / 1_000_000)
    assert by_label["CPU time"] == pytest.approx(10 * 5.0 * 0.02 / 1_000_000)
    assert by_label["Storage"] == pytest.approx(10 * 16_384 / 1e9 * 30 / 30 * 0.20)
    assert infra.usd.low == infra.usd.high == pytest.approx(sum(by_label.values()))
    assert infra.runs_within_included_steps == Range.exact(50_000)


# ───────── rote eval ─────────


@pytest.fixture()
def offline_prices(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.test_eval_cli import FAKE_CATALOG

    monkeypatch.setattr("rote.eval.pricing.fetch_catalog", lambda **kwargs: FAKE_CATALOG)
    monkeypatch.setattr("rote.eval.infra_pricing.get_text", lambda url, timeout: MDX)


def _compile_out(tmp_path: Path, runtimes: list[str]) -> Path:
    from rote.ir import save_pipeline

    out = tmp_path / "out"
    (out / "compiled").mkdir(parents=True)
    save_pipeline(_PIPELINE, out / "compiled" / "pipeline.yaml")
    for runtime in runtimes:
        (out / "runtime" / runtime).mkdir(parents=True)
    return out


def test_eval_prices_cloudflare_infrastructure_with_its_source(
    offline_prices: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = _compile_out(tmp_path, ["cloudflare"])
    assert main(["eval", str(out)]) == 0
    text = capsys.readouterr().out

    assert "## Infrastructure per run (Cloudflare Workflows, Workers Paid)" in text
    assert f"from `{SOURCE}`" in text
    assert "| Steps | 13 steps |" in text  # 1 + 10 fan_out + 1 gate + 1 loop
    assert "(widths: `score_post` 10 (prior fan_out_elements_per_run))" in text
    assert "A conservative assumption" in text


def test_eval_json_carries_the_infrastructure(
    offline_prices: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = _compile_out(tmp_path, ["temporal"])
    assert main(["eval", str(out), "--runtime", "rote-cloud", "--json"]) == 0
    infra = json.loads(capsys.readouterr().out)["infrastructure"]

    assert infra["runtime"] == "rote cloud (Cloudflare Dynamic Workflows)"
    assert infra["steps_per_run"] == {"low": 13, "high": 13}
    assert infra["price_source"] == SOURCE


def test_eval_adds_no_infrastructure_for_runtimes_that_do_not_bill_it(
    offline_prices: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = _compile_out(tmp_path, ["temporal"])
    assert main(["eval", str(out), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["infrastructure"] is None


def test_eval_fails_loudly_when_the_cloudflare_table_will_not_parse(
    offline_prices: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "rote.eval.infra_pricing.get_text", lambda url, timeout: MDX.replace("| Steps", "| Stairs")
    )
    out = _compile_out(tmp_path, ["cloudflare"])
    assert main(["eval", str(out)]) == 1
    assert "expected one 'Steps' row, found 0" in capsys.readouterr().err
