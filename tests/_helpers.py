"""Shared helpers for adapter tests.

Everything here exists in 3+ adapter test files otherwise — and the
copies had already drifted (the Cloudflare MCP scan was missing the
backtick template-literal alternative, so a legitimate "mcp" mention
inside a template string failed only that adapter's test).
"""

from __future__ import annotations

import ast
import re
import socket
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from rote.adapters._common import strip_ts_comments
from rote.ir import Node, Pipeline

# ───────── MCP invariant scan (TS) ─────────
#
# Comments and JSDoc may *mention* MCP to explain the compilation
# history; only executable code is forbidden from referencing it. So
# strip /* ... */ blocks, // line comments, and string literals —
# including backtick template literals — before scanning.

_JS_STRING = re.compile(r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|`(?:[^`\\]|\\.)*`')


def assert_no_mcp_in_ts(
    emit_result: dict[str, Path], *, min_files: int, expected: set[str] | None = None
) -> None:
    """Assert no emitted .ts file references MCP in executable code.

    ``expected`` names the labels that legitimately DO — the MCP helper
    itself, and any agent_loop module, whose ``tools:`` are MCP tool
    names by the IR's own definition. Naming them (rather than skipping
    every file that happens to mention MCP) keeps the scan a real
    constraint: an unexpected MCP reference still fails, and an expected
    site that stops referencing MCP fails too.

    ``min_files`` guards against the scan silently checking nothing
    (e.g. a refactor that changes emitted file suffixes).
    """
    expected = expected or set()
    checked = 0
    for label, path in emit_result.items():
        if not str(path).endswith(".ts"):
            continue
        if label in expected:
            assert "mcp" in path.read_text(encoding="utf-8").lower(), (
                f"{label} was declared an expected MCP site but references no MCP"
            )
            continue
        src = path.read_text(encoding="utf-8")
        cleaned = _JS_STRING.sub('""', strip_ts_comments(src))
        assert "mcp" not in cleaned.lower(), (
            f"{label} ({path.name}) contains forbidden substring 'mcp' in executable code"
        )
        checked += 1
    assert checked >= min_files, f"MCP scan only saw {checked} .ts files (expected >= {min_files})"


# ───────── Minimal pipeline factory ─────────


def mini_pipeline(node: Node) -> Pipeline:
    """A single-node pipeline for focused emission tests."""
    return Pipeline(
        name="mini",
        input={"type": "In", "required": [], "optional": []},
        nodes=[node],
        edges=[],
        entry_nodes=[node.id],
        exit_nodes=[node.id],
    )


# ───────── fan_out mock support (live e2e) ─────────
#
# Every live e2e replaces node implementations with recording mocks. A
# fan_out node dispatches once per element of an upstream list, so the
# mock standing in for that upstream node has to return a real list —
# otherwise the fan either dispatches zero times (empty list, node
# silently never runs) or trips the fanOutList guard (missing key).
#
# Derived from the IR rather than hardcoded per suite: the three TS e2e
# files had already encoded `exclusion_check_sequence` returning
# `passed: []`, which made the BDR fan a no-op that no assertion caught.

#: Elements per fanned list in mocked runs. Must exceed 1, or
#: per-element dispatch is indistinguishable from batch dispatch.
FAN_OUT_ELEMENTS = 3


def fan_out_element(index: int) -> dict[str, int]:
    """One canned element of a mocked fanned list."""
    return {"fanElement": index}


def fan_out_source_keys(pipeline: Pipeline) -> dict[str, dict[str, list[dict[str, int]]]]:
    """``{upstream_node: {field: [element, ...]}}`` for every fan_out node.

    Uses the adapters' own element-param resolution, so a mock can never
    disagree with the emitted code about which input is the fanned list.
    """
    from rote.ir import fan_out_element_param, parse_input_ref

    extra: dict[str, dict[str, list[dict[str, int]]]] = {}
    for node in pipeline.nodes:
        if not node.fan_out or not node.inputs:
            continue
        parsed = parse_input_ref(node.inputs[fan_out_element_param(node, pipeline)])
        if parsed.node_id is None or parsed.field is None:
            continue
        extra.setdefault(parsed.node_id, {})[parsed.field] = [
            fan_out_element(i) for i in range(FAN_OUT_ELEMENTS)
        ]
    return extra


# ───────── Unconfigured MCP endpoint (config-error e2e cases) ─────────
#
# A node bound to (or a loop pinned to) a server with no URL anywhere: no
# IR url, no registry entry, no ROTE_MCP_VENDOR_URL. Every runtime must
# run the step exactly ONCE and fail naming the variable. The count is
# the assertion that matters: a message can be right while the step
# still burned its retry budget.

ATTEMPT_MARKER = "ROTE_E2E_ATTEMPT"


def unconfigured_mcp_pipelines(name: str) -> dict[str, Pipeline]:
    """One bound-step pipeline and one agent-loop pipeline, both with retries."""
    from rote.ir import MCPBinding, NodeKind, PipelineInput, RetryPolicy, TerminationConfig

    bound = Node(
        id="lookup",
        kind=NodeKind.EXTERNAL_CALL,
        description="Look up a contact on the vendor MCP server.",
        input={"contact_id": "str"},
        inputs={"contact_id": "pipeline.input.contact_id"},
        output="dict",
        mcp=MCPBinding(server="vendor", tool="enrich_contact", args={"contact_id": "contact_id"}),
        retry=RetryPolicy(max=3, backoff="exponential"),
    )
    loop = Node(
        id="research",
        kind=NodeKind.AGENT_LOOP,
        description="Research the contact with the vendor's tools.",
        input={"contact_id": "str"},
        inputs={"contact_id": "pipeline.input.contact_id"},
        output="dict",
        tools=["enrich_contact"],
        tool_servers={"enrich_contact": "vendor"},
        termination=TerminationConfig(condition="the contact is enriched", max_iterations=3),
        retry=RetryPolicy(max=3, backoff="exponential"),
    )
    return {
        node.id: Pipeline(
            name=f"{name}_{node.id}",
            description="Unconfigured MCP endpoint e2e pipeline.",
            input=PipelineInput(type="Req", required=["contact_id"]),
            nodes=[node],
            edges=[],
            entry_nodes=[node.id],
            exit_nodes=[node.id],
        )
        for node in (bound, loop)
    }


def count_ts_step_attempts(module: Path, node_id: str) -> None:
    """Make an emitted TS node module log one marker line per invocation.

    The marker goes first in the exported function, so it counts step
    attempts whatever the step does next (bound call or tool discovery).
    """
    src = module.read_text(encoding="utf-8")
    match = re.search(r"export async function \w+\([\s\S]*?\)[^\n]*\{\n", src)
    assert match, f"no exported async function in {module}"
    marker = f'    console.log("{ATTEMPT_MARKER} {node_id}");\n'
    module.write_text(src[: match.end()] + marker + src[match.end() :], encoding="utf-8")


def attempts_in(text: str, node_id: str) -> int:
    """How many times :func:`count_ts_step_attempts`'s marker was printed."""
    return text.count(f"{ATTEMPT_MARKER} {node_id}")


def payload_dict_keys(payload: ast.Dict) -> set[str]:
    """The keys an emitted payload dict literal can carry.

    A pipeline input field is spread in as
    ``**_input_field(pipeline_input, "<key>", "<field>")``, so its key is
    the call's second argument rather than a literal dict key.
    """
    keys: set[str] = set()
    for key, value in zip(payload.keys, payload.values, strict=True):
        if isinstance(key, ast.Constant):
            keys.add(key.value)
        elif (
            key is None
            and isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "_input_field"
            and isinstance(value.args[1], ast.Constant)
        ):
            keys.add(value.args[1].value)
    return keys


@contextmanager
def occupied_ports(*ports: int) -> Iterator[None]:
    """Make ``ports`` unavailable to the login's callback-port selection.

    The goal is the state "these ports are busy", not "this test owns
    them". A port a previous test left in TIME_WAIT must not fail the setup
    (SO_REUSEADDR, as uvicorn and the code under test both set), and a port
    some other listener already holds is busy too, so it is kept as is
    rather than raising EADDRINUSE. The precondition is then checked the
    way the code under test sees it: it cannot bind any of them.
    """
    from rote.mcp.auth import bind_callback_sockets

    held: list[socket.socket] = []
    try:
        for port in ports:
            sock = socket.socket()
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
                sock.listen()
            except OSError:
                sock.close()  # already held by a live listener: busy either way
                continue
            held.append(sock)
        for port in ports:
            assert bind_callback_sockets(port) is None, f"port {port} is still free"
        yield
    finally:
        for sock in held:
            sock.close()


def fill_system1_judges(work: Path) -> None:
    """Write what the System-1 text agent writes for each judge (a prompt and
    output fields) into ``work``'s text.yaml, and settle it.

    A bare System-1 skeleton is not a deliverable: its judges have no prompt
    and an output schema with no properties, which every adapter refuses at
    emit (structured outputs cannot express a free-form object)."""
    import yaml

    from rote.compiler.system1.enforce import enforce

    doc = yaml.safe_load((work / "pipeline.yaml").read_text())
    text = {
        n["id"]: {
            "signature_spec": {
                "prompt": "Decide from {{ request }}.",
                "output_schema": {
                    "type": "object",
                    "properties": {"verdict": {"type": "string"}},
                    "required": ["verdict"],
                },
            }
        }
        for n in doc["nodes"]
        if n["kind"] == "llm_judge"
    }
    (work / "system1" / "text.yaml").write_text(yaml.safe_dump(text))
    enforce(work, final=True)


# ───────── System-1 calls on an OpenAI-compatible compile ─────────

#: An openai-api compile behind a Cloudflare AI Gateway, as rote-cloud runs it.
GATEWAY_URL = "https://gateway.example/v1/openai"
GATEWAY_HEADERS = {"cf-aig-authorization": "Bearer gw-token", "cf-aig-metadata": '{"tenant":"t1"}'}
#: An Anthropic key in the environment that no request may carry.
STRAY_ANTHROPIC_KEY = "sk-ant-must-not-be-sent"


def refuse_anthropic(monkeypatch: Any) -> None:
    """Make every way to reach Anthropic fail the test: its SDK clients, a
    ``claude`` lookup or subprocess. An Anthropic key sits in the environment
    so a request carrying it is visible, and pricing is offline."""
    import shutil
    import subprocess

    import anthropic

    from rote.eval import pricing

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(f"a non-Anthropic compile reached Anthropic: {args!r}")

    def offline(*args: Any, **kwargs: Any) -> Any:
        raise pricing.PricingError("offline in tests")

    real_which, real_run = shutil.which, subprocess.run

    def run(argv: Any, *args: Any, **kwargs: Any) -> Any:
        if "claude" in str(argv[0] if isinstance(argv, (list, tuple)) else argv):
            refuse(argv)
        return real_run(argv, *args, **kwargs)

    monkeypatch.setattr(anthropic, "Anthropic", refuse)
    monkeypatch.setattr(anthropic, "AsyncAnthropic", refuse)
    monkeypatch.setattr(
        shutil,
        "which",
        lambda name, *a, **k: refuse(name) if "claude" in str(name) else real_which(name, *a, **k),
    )
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(pricing, "fetch_catalog", offline)
    monkeypatch.setenv("ANTHROPIC_API_KEY", STRAY_ANTHROPIC_KEY)
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", STRAY_ANTHROPIC_KEY)


def openai_gateway(monkeypatch: Any, answers: list[Any]) -> list[Any]:
    """Serve every ``openai.OpenAI`` client from an in-process gateway that
    answers ``answers`` in turn (a dict as the model's JSON, an int as that
    HTTP error status), and return the requests it saw. Anthropic
    is refused (:func:`refuse_anthropic`)."""
    import json

    import httpx
    import openai

    seen: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        body = json.loads(request.content)
        if isinstance(answers[0], int):  # an HTTP error status, answered once
            return httpx.Response(answers.pop(0), json={"error": {"message": "gateway says no"}})
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": json.dumps(answers.pop(0))},
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            },
        )

    real = openai.OpenAI

    def client(**kwargs: Any) -> Any:
        return real(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(answer)))

    monkeypatch.setattr(openai, "OpenAI", client)
    refuse_anthropic(monkeypatch)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return seen


def fake_codex(tmp_path: Path, answers: list[dict[str, Any]], exit_code: int = 0) -> Path:
    """An executable standing in for ``codex``: it records its argv, stdin and
    the schema file it was given, prints ``--json`` events (a schema-shaped
    progress message, then the answer, then usage) and exits ``exit_code``.
    Each run answers the next of ``answers``."""
    import json
    import sys

    state = tmp_path / "fake-codex"
    state.mkdir(exist_ok=True)
    (state / "answers.json").write_text(json.dumps(answers))
    script = state / "codex"
    script.write_text(
        f"""#!{sys.executable}
import json, sys
from pathlib import Path
state = Path({str(state)!r})
argv = sys.argv[1:]
runs = sorted(state.glob("run-*.json"))
n = len(runs)
schema = argv[argv.index("--output-schema") + 1]
record = {{"argv": argv, "stdin": sys.stdin.read(), "schema": json.loads(Path(schema).read_text())}}
(state / f"run-{{n}}.json").write_text(json.dumps(record))
answers = json.loads((state / "answers.json").read_text())
def emit(event):
    print(json.dumps(event), flush=True)
def message(i, text):
    item = {{"id": f"item_{{i}}", "type": "agent_message", "text": text}}
    emit({{"type": "item.completed", "item": item}})
emit({{"type": "thread.started", "thread_id": "t"}})
message(0, "{{}}")
if {exit_code}:
    emit({{"type": "turn.failed", "error": {{"message": "usage limit reached"}}}})
    sys.exit({exit_code})
message(1, json.dumps(answers[n]))
usage = {{"input_tokens": 1000, "cached_input_tokens": 400, "output_tokens": 50}}
emit({{"type": "turn.completed", "usage": usage}})
"""
    )
    script.chmod(0o755)
    return script


def fake_codex_runs(script: Path) -> list[dict[str, Any]]:
    import json

    return [json.loads(p.read_text()) for p in sorted(script.parent.glob("run-*.json"))]


def openai_compile_driver(model: str = "openai/gpt-5.5") -> Any:
    from rote.compiler.drivers.openai_api import OpenAIApiDriver

    return OpenAIApiDriver(model=model, base_url=GATEWAY_URL, default_headers=GATEWAY_HEADERS)


def assert_sent_through_the_gateway(
    requests: list[Any], model: str = "openai/gpt-5.5", max_completion_tokens: int | None = None
) -> None:
    """Each request went to the compile's gateway, with its headers and model,
    schema-locked, and none carried the stray Anthropic key."""
    import json

    assert requests
    for r in requests:
        assert str(r.url) == f"{GATEWAY_URL}/chat/completions"
        assert r.headers["cf-aig-authorization"] == GATEWAY_HEADERS["cf-aig-authorization"]
        assert r.headers["cf-aig-metadata"] == GATEWAY_HEADERS["cf-aig-metadata"]
        assert "x-api-key" not in r.headers
        assert all(STRAY_ANTHROPIC_KEY not in v for v in r.headers.values())
        body = json.loads(r.content)
        assert body["model"] == model
        assert body["response_format"]["type"] == "json_schema"
        assert body["response_format"]["json_schema"]["strict"] is True
        assert body.get("max_completion_tokens") == max_completion_tokens
