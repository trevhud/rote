"""The TS cases of the conditional gate parity test (see test_conditional_gate_parity.py).

Cloudflare and Inngest here; DBOS-TS, which needs Docker Postgres, in
test_conditional_gate_parity_dbos_ts_e2e.py. The step implementations are
typed, so ``tsc`` checks that a gate's ``Record | null`` result reaches an
implementation that declares it nullable.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from rote.adapters import get_adapter
from rote.mcp.parked import watch_factory
from tests.test_conditional_gate_parity import CASES, PIPELINE, SIGNALS, check

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("npm") is None, reason="requires node + npm"),
]

_IMPLS = {
    "detect_owner": (
        "export async function detectOwner(input: {\n"
        "    notes: string;\n"
        "    flag: unknown;\n"
        "}): Promise<{ ambiguous: unknown; candidates: string[] }> {\n"
        '    return { ambiguous: input.flag, candidates: ["ana", "bo"] };\n'
        "}\n"
    ),
    "recap": (
        "export async function recap(input: {\n"
        "    notes: string;\n"
        "    answer: { owner: string } | null;\n"
        "}): Promise<{ recap: string; owner: string; asked: boolean }> {\n"
        "    if (input.answer === null) {\n"
        '        return { recap: input.notes, owner: "me", asked: false };\n'
        "    }\n"
        "    return { recap: input.notes, owner: input.answer.owner, asked: true };\n"
        "}\n"
    ),
}


def run_case(runtime: str, case: str, tmp_path: Path) -> Any:
    app = tmp_path / runtime
    get_adapter(runtime).emit(PIPELINE, app)
    for node_id, source in _IMPLS.items():
        (app / "src" / "extracted" / f"{node_id}.ts").write_text(source, encoding="utf-8")
    watch = watch_factory(False)
    pipeline_input = CASES[case][0]
    if runtime == "cloudflare":
        from rote.runners.cloudflare import run_cloudflare

        return run_cloudflare(
            app,
            pipeline_input,
            signals=SIGNALS,
            gate_order=["owner"],
            timeout_seconds=180,
            park_watch=watch,
        )
    if runtime == "inngest":
        from rote.runners.inngest import run_inngest

        return run_inngest(
            app,
            pipeline_input,
            pipeline=PIPELINE,
            signals=SIGNALS,
            gate_order=["owner"],
            timeout_seconds=180,
            park_watch=watch,
        )
    from rote.runners.dbos_ts import run_dbos_ts

    return run_dbos_ts(app, pipeline_input, signals=SIGNALS, timeout_seconds=180, park_watch=watch)


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("runtime", ["cloudflare", "inngest"])
def test_ts_asks_only_when_the_condition_holds(
    runtime: str, case: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = run_case(runtime, case, tmp_path)
    check(case, run, capsys.readouterr().err)
