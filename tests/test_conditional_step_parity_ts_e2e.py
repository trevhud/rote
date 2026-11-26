"""The TS cases of the conditional step parity test (see test_conditional_step_parity.py).

Cloudflare and Inngest here; DBOS-TS, which needs Docker Postgres, in
test_conditional_step_parity_dbos_ts_e2e.py. The step implementations are
typed with interfaces, the way the implement skill has users write them, so
``tsc`` checks that a skipped step's ``T | null`` result reaches an
implementation that declares it nullable, and that the condition's own node
may return an interface.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from rote.adapters import get_adapter
from rote.mcp.parked import watch_factory
from tests.test_conditional_step_parity import CASES, PEOPLE, PIPELINE, check

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("npm") is None, reason="requires node + npm"),
]

_IMPLS = {
    "check": (
        "export interface Checked {\n"
        "    connected: boolean | string;\n"
        "    channel: string;\n"
        "    people: string[];\n"
        "}\n"
        "\n"
        "export async function check(input: { flag: boolean | string }): Promise<Checked> {\n"
        f'    return {{ connected: input.flag, channel: "#ops", people: {PEOPLE!r} }};\n'.replace(
            "'", '"'
        )
        + "}\n"
    ),
    "draft": (
        "export async function draft(input: { notes: string }): Promise<{ text: string }> {\n"
        "    return { text: input.notes.toUpperCase() };\n"
        "}\n"
    ),
    "post": (
        "export interface Posted {\n"
        "    posted_to: string;\n"
        "}\n"
        "\n"
        "export async function post(input: {\n"
        "    channel: string;\n"
        "    notes: string;\n"
        "}): Promise<Posted> {\n"
        "    return { posted_to: input.channel };\n"
        "}\n"
    ),
    "ping": (
        "export async function ping(input: { person: string }): Promise<string> {\n"
        '    return "pinged " + input.person;\n'
        "}\n"
    ),
    "recap": (
        'import type { Posted } from "./post";\n'
        "\n"
        "export async function recap(input: {\n"
        "    text: string;\n"
        "    post: Posted | null;\n"
        "    pings: string[] | null;\n"
        "}): Promise<{ text: string; posted_to: string | null; pings: string[] | null }> {\n"
        "    return {\n"
        "        text: input.text,\n"
        "        posted_to: input.post === null ? null : input.post.posted_to,\n"
        "        pings: input.pings,\n"
        "    };\n"
        "}\n"
    ),
}


def emit(runtime: str, tmp_path: Path) -> Path:
    app = tmp_path / runtime
    get_adapter(runtime).emit(PIPELINE, app)
    for node_id, source in _IMPLS.items():
        target = app / "src" / "extracted" / f"{node_id}.ts"
        assert target.is_file(), target
        target.write_text(source, encoding="utf-8")
    return app


def run_case(runtime: str, case: str, tmp_path: Path) -> Any:
    app = emit(runtime, tmp_path)
    watch = watch_factory(False)
    pipeline_input = CASES[case][0]
    if runtime == "cloudflare":
        from rote.runners.cloudflare import run_cloudflare

        return run_cloudflare(
            app, pipeline_input, signals={}, gate_order=[], timeout_seconds=180, park_watch=watch
        )
    if runtime == "inngest":
        from rote.runners.inngest import run_inngest

        return run_inngest(
            app,
            pipeline_input,
            pipeline=PIPELINE,
            signals={},
            gate_order=[],
            timeout_seconds=180,
            park_watch=watch,
        )
    from rote.runners.dbos_ts import run_dbos_ts

    return run_dbos_ts(app, pipeline_input, signals={}, timeout_seconds=180, park_watch=watch)


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("runtime", ["cloudflare", "inngest"])
def test_ts_runs_a_step_only_when_its_condition_holds(
    runtime: str, case: str, tmp_path: Path
) -> None:
    check(case, run_case(runtime, case, tmp_path))


def _tsc(app: Path) -> subprocess.CompletedProcess[str]:
    from tests import _node_trees as node_trees

    node_trees.install(app)
    return subprocess.run(
        ["npx", "--no-install", "tsc", "--noEmit"],
        cwd=app,
        capture_output=True,
        text=True,
        timeout=180,
    )


@pytest.mark.parametrize("runtime", ["cloudflare", "dbos-ts", "inngest"])
def test_typed_implementations_compile_around_a_conditional_step(
    runtime: str, tmp_path: Path
) -> None:
    tsc = _tsc(emit(runtime, tmp_path))
    assert tsc.returncode == 0, tsc.stdout


@pytest.mark.parametrize("runtime", ["cloudflare", "dbos-ts", "inngest"])
def test_a_reader_that_does_not_admit_null_fails_tsc(runtime: str, tmp_path: Path) -> None:
    """The negative half: the skipped step's result is typed ``T | null``, so
    a reader that drops the null and reads a field through it fails tsc. A
    result typed away to ``any`` would let this compile."""
    app = emit(runtime, tmp_path)
    workflow = next(
        p
        for p in (app / "src").rglob("*.ts")
        if "post_result as ImplInput<typeof recap>" in p.read_text(encoding="utf-8")
    )
    text = workflow.read_text(encoding="utf-8")
    broken = text.replace(
        'post_result as ImplInput<typeof recap>["post"]', "{ posted_to: post_result.posted_to }"
    )
    assert broken != text
    workflow.write_text(broken, encoding="utf-8")
    tsc = _tsc(app)
    assert tsc.returncode != 0
    assert "TS18047" in tsc.stdout, tsc.stdout
