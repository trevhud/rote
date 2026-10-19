"""Constants reach an impl from every caller, and reach a tool only by name.

Two callers the payload-literal approach missed:

* An agent loop drives its ``loop_body`` sub-nodes as tools, with a payload
  the model builds. BDR's ``enrich_contact_batch`` is one, and it needs
  ``batch_size``. The step body now merges its node's constants, so the
  model's call carries them without the tool schema exposing them.
* An MCP-bound step passes its payload to the tool as the arguments, and a
  tool rejects arguments it does not declare. Constants stay out of that
  payload; a constant reaches a tool only when ``mcp.args`` names it, and
  every runtime builds the identical arguments.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

from rote.adapters import get_adapter
from rote.eval.empirical import run_pipeline_trial
from rote.ir import Pipeline

# ───────── an agent loop calling a constant-bearing sub-node ─────────

_LOOP_PIPELINE = Pipeline.model_validate(
    {
        "name": "loop-constants",
        "input": {"type": "In", "required": ["account"], "optional": []},
        "nodes": [
            {
                "id": "research",
                "kind": "agent_loop",
                "description": "Research the account with the lookup tool.",
                "tools": ["web_search"],
                "loop_body": ["lookup_account"],
                "termination": {"condition": "account looked up", "max_iterations": 4},
                "inputs": {"account": "pipeline.input.account"},
            },
            {
                "id": "lookup_account",
                "kind": "external_call",
                "description": "Look up one account.",
                "impl": "extracted/crm.py:lookup",
                "constants": {"batch_size": 25, "fields": ["tier", "owner"]},
            },
        ],
        "edges": [],
        "entry_nodes": ["research"],
        "exit_nodes": ["research"],
    }
)

_CRM = (
    "def lookup(account, *, batch_size, fields):\n"
    "    return {'account': account, 'batch_size': batch_size, 'fields': fields}\n"
)


class _LoopStub(BaseHTTPRequestHandler):
    """Turn 1 calls ``lookup_account`` with a model-built payload; turn 2
    answers with whatever the tool returned."""

    requests: list[dict[str, Any]] = []

    def do_POST(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler's API
        payload = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
        type(self).requests.append(payload)
        if len(type(self).requests) == 1:
            message: dict[str, Any] = {
                "stop_reason": "tool_use",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "lookup_account",
                        "input": {"payload": {"account": "acme"}},
                    }
                ],
            }
        else:
            result = payload["messages"][-1]["content"][0]["content"]
            text = result if isinstance(result, str) else result[0]["text"]
            message = {"stop_reason": "end_turn", "content": [{"type": "text", "text": text}]}
        message.update(
            {
                "id": f"msg_{len(type(self).requests)}",
                "type": "message",
                "role": "assistant",
                "model": payload["model"],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
        )
        raw = json.dumps(message).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args: object) -> None:
        return


@pytest.fixture()
def loop_stub() -> Iterator[str]:
    _LoopStub.requests = []
    server = HTTPServer(("127.0.0.1", 0), _LoopStub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


def test_a_loop_body_tool_call_receives_its_nodes_constants(
    loop_stub: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real emitted app runs a real tool runner against a local Messages
    endpoint. The model's payload names only ``account``, yet the impl is
    keyword-only on its constants and gets them, and the tool schema the
    model saw never mentions them."""
    source = tmp_path / "compiled"
    (source / "extracted").mkdir(parents=True)
    (source / "extracted" / "crm.py").write_text(_CRM, encoding="utf-8")
    app = tmp_path / "python"
    get_adapter("python", extracted_source_dir=source).emit(_LOOP_PIPELINE, app)

    monkeypatch.setenv("ROTE_INFERENCE", "api")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", loop_stub)
    # No MCP server resolves, so the loop has only its local tool.
    monkeypatch.setenv("ROTE_MCP_CONFIG", str(tmp_path / "no-mcp.json"))
    run = run_pipeline_trial(app, {"account": "acme"}, timeout_seconds=60)

    assert run.error is None, run.error
    tool_result = json.loads(run.output["research"]["result"])  # type: ignore[index]
    assert tool_result == {"account": "acme", "batch_size": 25, "fields": ["tier", "owner"]}
    (tool,) = _LoopStub.requests[0]["tools"]
    assert tool["name"] == "lookup_account"
    schema = json.dumps(tool["input_schema"])
    assert "batch_size" not in schema
    assert "fields" not in schema


# ───────── MCP tool arguments, identical on every runtime ─────────


def _mcp_pipeline(args: dict[str, str] | None) -> Pipeline:
    mcp: dict[str, Any] = {"server": "gmail", "tool": "search_threads"}
    if args is not None:
        mcp["args"] = args
    return Pipeline.model_validate(
        {
            "name": "mcp-args",
            "input": {"type": "In", "required": ["query"], "optional": []},
            "nodes": [
                {
                    "id": "search_mail",
                    "kind": "external_call",
                    "description": "Search mail.",
                    "mcp": mcp,
                    "constants": {"email_window_days": 7, "fixed_queries": ["a", "b"]},
                    "inputs": {"query": "pipeline.input.query"},
                }
            ],
            "edges": [],
            "entry_nodes": ["search_mail"],
            "exit_nodes": ["search_mail"],
        }
    )


_PAYLOAD = {"query": "invoices"}


def _python_arguments(pipeline: Pipeline, tmp_path: Path) -> Any:
    out = tmp_path / "dbos"
    get_adapter("dbos").emit(pipeline, out)
    main = (out / "main.py").read_text(encoding="utf-8")
    (line,) = [ln for ln in main.splitlines() if ln.strip().startswith("arguments = ")]
    scope: dict[str, Any] = {"payload": dict(_PAYLOAD)}
    exec(line.strip(), scope)  # noqa: S102 — our own emitted literal
    return scope["arguments"]


def _ts_arguments(pipeline: Pipeline, tmp_path: Path, runtime: str) -> Any:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    out = tmp_path / runtime
    get_adapter(runtime).emit(pipeline, out)
    module = next(out.rglob("search_mail.ts")).read_text(encoding="utf-8")
    match = re.search(r"const args = (.*?);", module)
    assert match is not None, module
    script = (
        f"const payload = {json.dumps(_PAYLOAD)}; console.log(JSON.stringify({match.group(1)}));"
    )
    return json.loads(
        subprocess.run([node, "-e", script], capture_output=True, text=True, check=True).stdout
    )


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        # No mapping: the payload is the arguments, and constants are not in it.
        (None, {"query": "invoices"}),
        # A mapping that names a constant forwards that constant.
        ({"q": "query", "days": "email_window_days"}, {"q": "invoices", "days": 7}),
    ],
)
def test_every_runtime_sends_a_tool_the_same_arguments(
    args: dict[str, str] | None, expected: dict[str, Any], tmp_path: Path
) -> None:
    pipeline = _mcp_pipeline(args)
    assert _python_arguments(pipeline, tmp_path) == expected
    for runtime in ("dbos-ts", "cloudflare", "inngest"):
        assert _ts_arguments(pipeline, tmp_path, runtime) == expected, runtime


_TS_ENTRY_FILES = {
    "cloudflare": "src/workflow.ts",
    "dbos-ts": "src/main.ts",
    "inngest": "src/inngest/pipeline.ts",
}


@pytest.mark.parametrize("runtime", sorted(_TS_ENTRY_FILES))
def test_ts_workflows_keep_constants_out_of_an_mcp_steps_payload(
    runtime: str, tmp_path: Path
) -> None:
    """The TS impl calls carry constants now, but an MCP-backed step's
    payload is its tool arguments, so its constants must stay out."""
    out = tmp_path / runtime
    get_adapter(runtime).emit(_mcp_pipeline(None), out)
    entry = (out / _TS_ENTRY_FILES[runtime]).read_text(encoding="utf-8")
    assert 'query: pipelineInput["query"]' in entry
    assert "email_window_days" not in entry
    assert "fixed_queries" not in entry


@pytest.mark.parametrize("runtime", sorted(_TS_ENTRY_FILES))
def test_ts_loop_body_tool_calls_get_constants_the_model_never_sees(
    runtime: str, tmp_path: Path
) -> None:
    out = tmp_path / runtime
    get_adapter(runtime).emit(_LOOP_PIPELINE, out)
    loop = next(out.rglob("research.ts")).read_text(encoding="utf-8")
    call = (
        "lookupAccount({ ...(args as Record<string, unknown>), batch_size: 25, "
        'fields: ["tier", "owner"] })'
    )
    assert call in loop, loop
    tool_block = loop[loop.index('name: "lookup_account"') : loop.index("run: async (args)")]
    assert "batch_size" not in tool_block


def test_a_judge_may_reuse_an_input_name_as_a_constant() -> None:
    """A judge's constants never reach a call, so the impl-only clash rule
    must not reject them."""
    Pipeline.model_validate(
        {
            "name": "judge-constants",
            "input": {"type": "In", "required": ["score"], "optional": []},
            "nodes": [
                {
                    "id": "grade",
                    "kind": "llm_judge",
                    "description": "Grade.",
                    "constants": {"score": 85},
                    "inputs": {"score": "pipeline.input.score"},
                    "signature_spec": {
                        "input_schema": {
                            "type": "object",
                            "properties": {"score": {"type": "number"}},
                        },
                        "output_schema": {
                            "type": "object",
                            "properties": {"ok": {"type": "boolean"}},
                        },
                        "prompt": "Is {{ score }} enough?",
                    },
                }
            ],
            "edges": [],
            "entry_nodes": ["grade"],
            "exit_nodes": ["grade"],
        }
    )


# ───────── a compile whose agent hard-codes a constant ─────────


_FETCH_PIPELINE = Pipeline.model_validate(
    {
        "name": "fetch-sheet",
        "input": {"type": "In", "required": ["region"], "optional": []},
        "nodes": [
            {
                "id": "fetch_sheet",
                "kind": "external_call",
                "description": "Fetch the sheet.",
                "impl": "extracted/drive.py:fetch",
                "constants": {"file_id": "F-1"},
                "inputs": {"region": "pipeline.input.region"},
            }
        ],
        "edges": [],
        "entry_nodes": ["fetch_sheet"],
        "exit_nodes": ["fetch_sheet"],
    }
)


@pytest.mark.parametrize("runtime", ["dbos", "python", "temporal"])
def test_compile_refuses_cleanly_when_an_impl_rejects_its_constant(
    runtime: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The way 37 of 41 committed agent-written impls are shaped: the
    constant is a module-level value, not a parameter. The paid-for compile
    must not die with a traceback, must keep pipeline.yaml, and must leave
    no half-new app in the runtime dir."""
    from rote.cli import main as cli_main
    from rote.compiler import CompilationResult
    from rote.ir import save_pipeline

    class _AgentThatHardCodes:
        def __init__(self, agent: str | None = None, **kwargs: object) -> None:
            pass

        async def compile(self, skill_path, output_dir, update=False, **_kwargs):  # type: ignore[no-untyped-def]  # noqa: ANN001
            out = Path(output_dir)
            (out / "extracted").mkdir(parents=True, exist_ok=True)
            save_pipeline(_FETCH_PIPELINE, out / "pipeline.yaml")
            (out / "extracted" / "drive.py").write_text(
                "FILE_ID = 'F-1'\n\n\ndef fetch(region):\n    return {'id': FILE_ID}\n",
                encoding="utf-8",
            )
            return CompilationResult(
                pipeline=_FETCH_PIPELINE, output_dir=out, driver_name="fake", driver_metadata={}
            )

    monkeypatch.setattr("rote.cli.Compiler", _AgentThatHardCodes)
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: t\n---\n", encoding="utf-8")
    out = tmp_path / "out"

    rc = cli_main(["compile", str(skill), "--runtime", runtime, "--out", str(out), "--no-eval"])

    err = capsys.readouterr().err
    assert rc == 1
    assert "Traceback" not in err
    agent_module = out / "compiled" / "extracted" / "drive.py"
    assert f"node 'fetch_sheet': {agent_module} fetch() takes no file_id" in err
    assert f"rote emit {out / 'compiled' / 'pipeline.yaml'} --runtime {runtime}" in err
    assert (out / "compiled" / "pipeline.yaml").is_file()
    runtime_dir = out / "runtime" / runtime
    assert not runtime_dir.exists() or not any(runtime_dir.rglob("*.py")), sorted(
        runtime_dir.rglob("*")
    )


def test_a_user_edit_the_writer_would_preserve_is_checked_too(tmp_path: Path) -> None:
    """No module beside pipeline.yaml, so the first emit writes a stub. The
    user then fills it in without the constant. The re-emit would preserve
    that file, so it is the one checked, and nothing is rewritten."""
    out = tmp_path / "python"
    get_adapter("python").emit(_FETCH_PIPELINE, out)
    (out / "extracted" / "drive.py").write_text(
        "def fetch(region):\n    return {'region': region}\n", encoding="utf-8"
    )
    main_before = (out / "main.py").read_text(encoding="utf-8")

    with pytest.raises(ValueError) as refused:
        get_adapter("python").emit(_FETCH_PIPELINE, out)
    # It names the user's edit in the runtime dir, the file that needs fixing.
    assert f"{out / 'extracted' / 'drive.py'} fetch() takes no file_id" in str(refused.value)
    assert (out / "main.py").read_text(encoding="utf-8") == main_before
    assert not (out / "extracted" / "drive.py.new").exists()
