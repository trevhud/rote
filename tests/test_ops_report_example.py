"""ops-report's Gmail window comes from report_date, not the wall clock.

The source skill's ``label:dock-activity newer_than:1d`` is resolved by
Gmail at search time, so a retry or backfill of one report date saw
different threads. ``build_dock_email_query`` now derives a calendar
window from ``report_date``; these tests run the emitted module.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from rote.adapters import get_adapter
from rote.ir import load_pipeline

PIPELINE = (
    Path(__file__).resolve().parent.parent
    / "examples"
    / "ops-report"
    / "expected"
    / "pipeline.yaml"
)


@pytest.fixture(scope="module")
def query_module(tmp_path_factory: pytest.TempPathFactory) -> ModuleType:
    out = tmp_path_factory.mktemp("ops-report")
    get_adapter("dbos", extracted_source_dir=PIPELINE.parent).emit(load_pipeline(PIPELINE), out)
    spec = importlib.util.spec_from_file_location(
        "gmail_query", out / "extracted" / "gmail_query.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _constants() -> dict[str, object]:
    node = next(n for n in load_pipeline(PIPELINE).nodes if n.id == "build_dock_email_query")
    return dict(node.constants or {})


def test_no_node_reads_the_clock_through_its_query() -> None:
    pipeline = load_pipeline(PIPELINE)
    for node in pipeline.nodes:
        for value in (node.constants or {}).values():
            assert "newer_than" not in str(value), node.id
    search = next(n for n in pipeline.nodes if n.id == "search_dock_emails")
    assert search.inputs == {"query": "build_dock_email_query.output"}
    builder = next(n for n in pipeline.nodes if n.id == "build_dock_email_query")
    assert builder.inputs == {"report_date": "pipeline.input.report_date"}


def test_the_window_is_the_day_before_the_report_date(query_module: ModuleType) -> None:
    query = query_module.build_dock_email_query(**_constants(), report_date="2026-10-19")
    assert query == "label:dock-activity after:2026/10/18 before:2026/10/19"
    # Same date, same query: nothing here depends on when it runs.
    assert query_module.build_dock_email_query(**_constants(), report_date="2026-10-19") == query


def test_a_wider_window_reaches_further_back(query_module: ModuleType) -> None:
    constants = {**_constants(), "window_days": 3}
    query = query_module.build_dock_email_query(**constants, report_date="2026-03-02")
    assert query == "label:dock-activity after:2026/02/27 before:2026/03/02"


def test_an_omitted_report_date_means_today(query_module: ModuleType) -> None:
    before = dt.date.today()
    query = query_module.build_dock_email_query(**_constants())
    after = dt.date.today()
    expected = {
        f"label:dock-activity after:{d - dt.timedelta(days=1):%Y/%m/%d} before:{d:%Y/%m/%d}"
        for d in (before, after)
    }
    assert query in expected


def test_a_bad_report_date_fails_instead_of_searching(query_module: ModuleType) -> None:
    with pytest.raises(ValueError):
        query_module.build_dock_email_query(**_constants(), report_date="10/19/2026")
