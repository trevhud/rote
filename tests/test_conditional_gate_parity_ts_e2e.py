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


#: The condition's node typed as an interface, the way the implement skill
#: has users write it. An interface has no implicit index signature, so a
#: `as Record<string, unknown>` cast of its result fails tsc (TS2352).
_INTERFACE_IMPLS = {
    **_IMPLS,
    "detect_owner": (
        "export interface Detected {\n"
        "    ambiguous: boolean;\n"
        "    candidates: string[];\n"
        "}\n"
        "\n"
        "export async function detectOwner(input: {\n"
        "    notes: string;\n"
        "    flag: unknown;\n"
        "}): Promise<Detected> {\n"
        '    return { ambiguous: input.flag === true, candidates: ["ana", "bo"] };\n'
        "}\n"
    ),
}


@pytest.mark.parametrize("runtime", ["cloudflare", "dbos-ts", "inngest"])
def test_a_condition_on_an_interface_typed_result_compiles(runtime: str, tmp_path: Path) -> None:
    """Found by review: the emitted condition cast its node's result to a
    Record, which tsc refuses for an interface on Cloudflare and DBOS-TS."""
    import subprocess

    from tests import _node_trees as node_trees

    app = tmp_path / runtime
    get_adapter(runtime).emit(PIPELINE, app)
    for node_id, source in _INTERFACE_IMPLS.items():
        (app / "src" / "extracted" / f"{node_id}.ts").write_text(source, encoding="utf-8")
    node_trees.install(app)
    tsc = subprocess.run(
        ["npx", "--no-install", "tsc", "--noEmit"],
        cwd=app,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert tsc.returncode == 0, tsc.stdout


def test_cloudflare_runs_past_a_skipped_gate_into_a_slow_step(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Found by review: `rote run` inferred a park from a step slower than
    its stability window after a skipped gate, sent the gate's payload,
    nothing consumed it, and the stuck-gate check failed a good run. The
    runner now answers only a gate that has announced itself."""
    from rote.runners.cloudflare import run_cloudflare

    app = tmp_path / "cloudflare"
    get_adapter("cloudflare").emit(PIPELINE, app)
    slow = _IMPLS["recap"].replace(
        "}): Promise<{ recap: string; owner: string; asked: boolean }> {\n",
        "}): Promise<{ recap: string; owner: string; asked: boolean }> {\n"
        "    await new Promise((resolve) => setTimeout(resolve, 6000));\n",
    )
    assert slow != _IMPLS["recap"]
    for node_id, source in {**_IMPLS, "recap": slow}.items():
        (app / "src" / "extracted" / f"{node_id}.ts").write_text(source, encoding="utf-8")
    run = run_cloudflare(
        app,
        CASES["skips"][0],
        signals=SIGNALS,
        gate_order=["owner"],
        timeout_seconds=180,
        park_watch=watch_factory(False),
    )
    check("skips", run, capsys.readouterr().err)
