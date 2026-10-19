"""The TS runtimes produce an agent loop's declared output keys too.

Same fixture as ``test_agent_loop_output_parity.py`` (the Python
runtimes): a downstream node reads two declared keys of an agent loop,
the loop runs against a stub Messages endpoint and a real fastmcp server,
and the pipeline's output is what that node made of them. Also typechecks
invoice-push on the three TS runtimes: before the fix, its downstream
read of ``process_invoices_loop_result["invoice_results"]`` failed TS7053
on DBOS-TS and Inngest, and Cloudflare compiled only because it cast the
result to ``Record<string, unknown>``.

Slow: npm install, tsc and the real dev servers (wrangler, inngest-cli,
DBOS-TS on Docker Postgres, the last in its own file).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from rote.adapters import get_adapter
from rote.ir import Pipeline, load_pipeline
from tests.test_agent_loop_output_parity import (
    EXPECTED,
    PIPELINE,
    SUBMITTED,
    MessagesStub,
    assert_the_model_was_offered_the_declared_output,
    loop_env,
    mcp_server,  # noqa: F401 — fixture
    messages_stub,  # noqa: F401 — fixture
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("npm") is None, reason="requires node + npm"),
]

REPORT_TS = """\
export async function report(input: Record<string, unknown>): Promise<{ line: string }> {
    return { line: `${String(input.summary)} / ${Number(input.score)}` };
}
"""

INVOICE_PUSH = (
    Path(__file__).resolve().parent.parent
    / "examples"
    / "invoice-push"
    / "expected"
    / "pipeline.yaml"
)


def run_ts(runtime: str, env: dict[str, str], tmp_path: Path) -> Any:
    app = tmp_path / runtime
    get_adapter(runtime).emit(PIPELINE, app)
    (app / "src" / "extracted" / "report.ts").write_text(REPORT_TS, encoding="utf-8")
    payload = {"account": "acme"}
    if runtime == "cloudflare":
        from rote.runners.cloudflare import run_cloudflare

        # workerd reads its bindings from .dev.vars, not the process env.
        lines = [f'{key}="{value}"' for key, value in env.items()]
        (app / ".dev.vars").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return run_cloudflare(app, payload, signals={}, gate_order=[], timeout_seconds=180)
    if runtime == "inngest":
        from rote.runners.inngest import run_inngest

        return run_inngest(
            app, payload, pipeline=PIPELINE, signals={}, gate_order=[], timeout_seconds=180
        )
    from rote.runners.dbos_ts import run_dbos_ts

    return run_dbos_ts(app, payload, signals={}, timeout_seconds=180)


def check_ts_runtime(
    runtime: str,
    stub: str,
    mcp: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = loop_env(stub, mcp, tmp_path)
    # The Node runtimes inherit the process env; Cloudflare gets .dev.vars.
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    run = run_ts(runtime, env, tmp_path)
    assert run.error is None, run.error
    assert run.output == EXPECTED
    assert_the_model_was_offered_the_declared_output()


#: dbos-ts needs Docker Postgres, so its case lives in
#: test_agent_loop_output_parity_dbos_ts_e2e.py (the DBOS-TS CI job).
@pytest.mark.parametrize("runtime", ["cloudflare", "inngest"])
def test_downstream_reads_the_loops_declared_keys(
    runtime: str,
    messages_stub: str,  # noqa: F811
    mcp_server: str,  # noqa: F811
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    check_ts_runtime(runtime, messages_stub, mcp_server, tmp_path, monkeypatch)


@pytest.mark.parametrize("runtime", ["cloudflare", "dbos-ts", "inngest"])
def test_invoice_push_typechecks(runtime: str, tmp_path: Path) -> None:
    app = tmp_path / runtime
    get_adapter(runtime).emit(load_pipeline(INVOICE_PUSH), app)
    install = subprocess.run(
        ["npm", "install", "--no-audit", "--no-fund"],
        cwd=app,
        capture_output=True,
        text=True,
        timeout=600,
        env={**os.environ, "npm_config_progress": "false"},
    )
    assert install.returncode == 0, install.stderr
    tsc = subprocess.run(
        ["npx", "--no-install", "tsc", "--noEmit"],
        cwd=app,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert tsc.returncode == 0, tsc.stdout + tsc.stderr


# ───────── the TS lane's own guarantees, run through the emitted module ─────────

_DRIVER_TS = """\
import { research } from "./extracted/research";

async function main(): Promise<void> {
    try {
        console.log(JSON.stringify({ output: await research({ account: "acme" }) }));
    } catch (err) {
        console.log(JSON.stringify({ error: (err as Error).message }));
    }
}

void main();
"""


@pytest.fixture(scope="module")
def loop_app(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The emitted DBOS-TS app, compiled, with a driver that runs the loop
    module alone (no Postgres: only the loop runs)."""
    app = tmp_path_factory.mktemp("loop-module")
    get_adapter("dbos-ts").emit(PIPELINE, app)
    (app / "src" / "extracted" / "report.ts").write_text(REPORT_TS, encoding="utf-8")
    (app / "src" / "driver.ts").write_text(_DRIVER_TS, encoding="utf-8")
    _npm_install(app)
    build = subprocess.run(["npx", "tsc"], cwd=app, capture_output=True, text=True, timeout=300)
    assert build.returncode == 0, build.stdout + build.stderr
    return app


def _drive(app: Path, env: dict[str, str]) -> dict[str, Any]:
    run = subprocess.run(
        ["node", "dist/driver.js"],
        cwd=app,
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, **env},
    )
    assert run.returncode == 0, run.stderr
    result: dict[str, Any] = json.loads(run.stdout.strip().splitlines()[-1])
    return result


def test_ts_loop_that_never_submits_gets_one_forced_turn(
    loop_app: Path,
    messages_stub: str,  # noqa: F811
    mcp_server: str,  # noqa: F811
    tmp_path: Path,
) -> None:
    MessagesStub.mode = "forced"
    result = _drive(loop_app, loop_env(messages_stub, mcp_server, tmp_path))
    assert result == {"output": SUBMITTED}
    forced = MessagesStub.requests[-1]
    assert forced["tool_choice"] == {"type": "tool", "name": "submit_result"}
    assert "tool_choice" not in MessagesStub.requests[0]
    assert forced["messages"][-1]["content"][-1]["text"].startswith("Call submit_result now")


def test_ts_submission_is_validated_with_the_shared_message(
    loop_app: Path,
    messages_stub: str,  # noqa: F811
    mcp_server: str,  # noqa: F811
    tmp_path: Path,
) -> None:
    MessagesStub.mode = "bad"
    result = _drive(loop_app, loop_env(messages_stub, mcp_server, tmp_path))
    assert result == {
        "error": "agent loop research (api) did not hand back its declared output: "
        "output.score: expected integer, got string"
    }


def test_ts_submit_tool_is_strict_when_the_schema_allows(
    loop_app: Path,
    messages_stub: str,  # noqa: F811
    mcp_server: str,  # noqa: F811
    tmp_path: Path,
) -> None:
    _drive(loop_app, loop_env(messages_stub, mcp_server, tmp_path))
    tools = {tool["name"]: tool for tool in MessagesStub.requests[0]["tools"]}
    assert tools["submit_result"]["strict"] is True
    # A server's tool is not marked: rote owns only its own schema.
    assert "strict" not in tools["web_search"]


# ───────── a named output type the user narrows once ─────────

_NAMED = Pipeline.model_validate(
    {
        **PIPELINE.model_dump(mode="json", by_alias=True, exclude_none=True),
        "nodes": [
            {
                **PIPELINE.model_dump(mode="json", by_alias=True, exclude_none=True)["nodes"][0],
                "output": {"summary": "str", "findings": "list[Finding]"},
            },
            {
                "id": "report",
                "kind": "pure_function",
                "description": "Write the one-line report.",
                "impl": "extracted/report.py:report",
                "inputs": {
                    "summary": "research.output.summary",
                    "findings": "research.output.findings",
                },
                "output": {"line": "str"},
            },
        ],
    }
)

_TYPED_REPORT_TS = """\
import type { Finding } from "../types";

export async function report(input: {
    summary: string;
    findings: Finding[];
}): Promise<{ line: string }> {
    return { line: `${input.summary}: ${input.findings.map((f) => f.title).join(", ")}` };
}
"""

_NARROWED = "export interface Finding {\n    title: string;\n}\n"


@pytest.mark.parametrize(
    "runtime",
    [
        "cloudflare",
        "inngest",
        pytest.param(
            "dbos-ts",
            marks=pytest.mark.xfail(
                strict=True,
                reason="DBOS-TS steps take Record<string, unknown>, so a closed input "
                "fails (fix/dbos-ts-typed-step-inputs)",
            ),
        ),
    ],
)
def test_a_typed_reader_compiles_once_the_named_type_is_narrowed(
    runtime: str, tmp_path: Path
) -> None:
    app = tmp_path / runtime
    get_adapter(runtime).emit(_NAMED, app)
    types_ts = app / "src" / "types.ts"
    assert "export type Finding = JsonValue;" in types_ts.read_text(encoding="utf-8")
    (app / "src" / "extracted" / "report.ts").write_text(_TYPED_REPORT_TS, encoding="utf-8")
    _npm_install(app)

    # As emitted, Finding is any JSON value, so reading .title fails.
    before = _tsc(app)
    assert before.returncode != 0
    assert "title" in before.stdout, before.stdout

    # The user narrows the alias once; the loop and its reader follow.
    types_ts.write_text(_NARROWED, encoding="utf-8")
    after = _tsc(app)
    assert after.returncode == 0, after.stdout

    # And a re-emit keeps the narrowed alias.
    get_adapter(runtime).emit(_NAMED, app)
    assert types_ts.read_text(encoding="utf-8") == _NARROWED


def _npm_install(app: Path) -> None:
    install = subprocess.run(
        ["npm", "install", "--no-audit", "--no-fund"],
        cwd=app,
        capture_output=True,
        text=True,
        timeout=600,
        env={**os.environ, "npm_config_progress": "false"},
    )
    assert install.returncode == 0, install.stderr


def _tsc(app: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["npx", "--no-install", "tsc", "--noEmit"],
        cwd=app,
        capture_output=True,
        text=True,
        timeout=300,
    )
