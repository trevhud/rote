"""A fan over one field of a result reads another field of it as shared (#329).

The ticket-creation fan-out needed the prepared items and, beside them, the
signed-in account id the same judge returned. Two params bound to one producer
used to be an ambiguous element param, so the binding #329 needed could not be
written at all. The param whose value the producer declares as a list is the
element; every other param, including the shared field, reaches each element
as is. This runs the emitted code: each item keeps its own fields, and every
one gets the same account.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from rote.adapters import get_adapter
from rote.eval.empirical import run_pipeline_trial
from rote.ir import Pipeline, fan_out_element_param

PIPELINE = Pipeline.model_validate(
    {
        "name": "fan-shared",
        "version": "0.1.0",
        "input": {"type": "In", "required": ["titles"]},
        "nodes": [
            {
                "id": "prepare",
                "kind": "pure_function",
                "description": "Prepare the tickets.",
                "impl": "extracted/steps.py:prepare",
                "inputs": {"titles": "pipeline.input.titles"},
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "items": {"type": "array", "items": {"type": "object"}},
                        "assignee_account_id": {"type": "string"},
                    },
                },
            },
            {
                "id": "create",
                "kind": "pure_function",
                "description": "Create one ticket per item.",
                "impl": "extracted/steps.py:create",
                "fan_out": True,
                "inputs": {
                    "ticket": "prepare.output.items",
                    "who": "prepare.output.assignee_account_id",
                },
            },
        ],
        "edges": [{"from": "prepare", "to": "create", "fan_out": True}],
        "entry_nodes": ["prepare"],
        "exit_nodes": ["create"],
    }
)

STEPS = (
    "def prepare(titles):\n"
    "    return {\n"
    "        'items': [{'title': t, 'n': i} for i, t in enumerate(titles)],\n"
    "        'assignee_account_id': 'ACCOUNT-1',\n"
    "    }\n"
    "\n"
    "\n"
    "def create(ticket, who):\n"
    "    return {'title': ticket['title'], 'n': ticket['n'], 'assignee': who}\n"
)

INPUT = {"titles": ["zeta", "alpha", "mid"]}
EXPECTED = {
    "create": [
        {"title": "zeta", "n": 0, "assignee": "ACCOUNT-1"},
        {"title": "alpha", "n": 1, "assignee": "ACCOUNT-1"},
        {"title": "mid", "n": 2, "assignee": "ACCOUNT-1"},
    ]
}


def test_the_declared_list_is_the_element_param() -> None:
    assert fan_out_element_param(PIPELINE.node_by_id("create"), PIPELINE) == "ticket"


def test_without_a_declared_list_it_is_still_an_error() -> None:
    raw = PIPELINE.model_dump(mode="json", by_alias=True, exclude_none=True)
    raw["nodes"][0].pop("output_schema")
    undeclared = Pipeline.model_validate(raw)
    with pytest.raises(ValueError, match="cannot identify the element param"):
        fan_out_element_param(undeclared.node_by_id("create"), undeclared)


def _emit(runtime: str, tmp_path: Path) -> Path:
    source = tmp_path / "compiled"
    (source / "extracted").mkdir(parents=True)
    (source / "extracted" / "steps.py").write_text(STEPS, encoding="utf-8")
    out = tmp_path / runtime
    get_adapter(runtime, extracted_source_dir=source).emit(PIPELINE, out)
    return out


def _check(output: Any) -> None:
    assert output == EXPECTED


def test_python_gives_every_element_the_shared_field(tmp_path: Path) -> None:
    run = run_pipeline_trial(_emit("python", tmp_path), INPUT, timeout_seconds=60)
    assert run.error is None, run.error
    _check(run.output)


def test_temporal_gives_every_element_the_shared_field(tmp_path: Path) -> None:
    from rote.runners.temporal import run_temporal

    app = _emit("temporal", tmp_path)
    try:
        run = run_temporal(app, INPUT, signals={}, timeout_seconds=60)
    finally:
        sys.path.remove(str(app))
        for name in [
            m
            for m in sys.modules
            if m.split(".")[0] in ("workflow", "activities", "extracted", "signatures")
        ]:
            del sys.modules[name]
    assert run.error is None, run.error
    _check(run.output)


@pytest.mark.slow
def test_dbos_gives_every_element_the_shared_field(tmp_path: Path) -> None:
    run = run_pipeline_trial(_emit("dbos", tmp_path), INPUT, timeout_seconds=120)
    assert run.error is None, run.error
    _check(run.output)


@pytest.mark.parametrize("runtime", ["cloudflare", "inngest", "dbos-ts"])
def test_ts_maps_over_the_list_and_shares_the_field(runtime: str, tmp_path: Path) -> None:
    """The TS emitters take the same element param: they map over the list
    and pass the shared field to every call."""
    out = tmp_path / runtime
    get_adapter(runtime).emit(PIPELINE, out)
    text = "\n".join(
        p.read_text() for p in (out / "src").rglob("*.ts") if "extracted" not in p.parts
    )
    assert 'fanOutList(prepare_result["items"]' in text or 'prepare_result["items"]' in text
    assert 'who: prepare_result["assignee_account_id"]' in text
