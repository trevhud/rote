"""The judge inference helper: provider selection and both transports.

This is where the *vendor call* is tested now. Emission tests assert
that judges delegate here; these assert what happens when they do.
"""

from __future__ import annotations

import json
import pickle
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from rote.adapters import ADAPTERS
from rote.inference import _runtime_helper as helper
from rote.ir import Node, NodeKind, Pipeline
from tests._helpers import mini_pipeline

# ───────── fixtures / doubles ─────────


class _FakeAPIError(Exception):
    """Shaped like the vendor SDKs' error classes: reconstructible only
    with keyword-only ``response`` / ``body``, so pickle cannot rebuild it
    from ``BaseException.__reduce__``'s ``(cls, args)``. Module-level so
    it is picklable at all — the failure under test is on *load*."""

    def __init__(self, message: str, *, response: object, body: object) -> None:
        super().__init__(message)
        self.response = response
        self.body = body
        self.status_code = 400


def _fake_anthropic(create: Any) -> types.ModuleType:
    module = types.ModuleType("anthropic")
    module.APIError = _FakeAPIError  # type: ignore[attr-defined]
    module.Anthropic = lambda **kwargs: types.SimpleNamespace(  # type: ignore[attr-defined]
        _kwargs=kwargs, messages=types.SimpleNamespace(create=create)
    )
    return module


def _tool_use_response(payload: dict[str, Any]) -> Any:
    return types.SimpleNamespace(
        content=[types.SimpleNamespace(type="tool_use", input=payload)],
        usage=types.SimpleNamespace(input_tokens=11, output_tokens=7),
    )


@pytest.fixture(autouse=True)
def _no_ambient_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralize the developer's own machine: a real ``claude`` binary or
    a real API key in the environment must not decide these tests."""
    for var in (
        "ROTE_INFERENCE",
        "ROTE_INFERENCE_GRADE_ESSAY",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "OPENAI_API_KEY",
        "ROTE_CLOUD_URL",
        "ROTE_CLOUD_TOKEN",
        "ROTE_USAGE_LOG",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(helper.shutil, "which", lambda _name: None)


# ───────── provider selection ─────────


def test_workload_shape_decides_the_lane_not_just_price(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The orders are inverses, and that is the design.

    A judge is one structured call, so the CLI's process spawn plus agent
    harness is pure overhead (measured 3.6-5.0s vs ~1.3s) — a key serves
    it. An agent loop amortizes that same overhead over every turn and is
    the expensive part of a pipeline, so the subscription serves it.
    """
    monkeypatch.setattr(helper.shutil, "which", lambda _name: "/usr/bin/claude")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-whatever")
    assert helper.select_provider("grade_essay", "anthropic") == "api"
    assert helper.select_provider("research", "anthropic", workload="agent") == "claude-cli"


def test_subscription_serves_judges_when_there_is_no_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cliff this whole feature exists to remove: a subscriber with no
    API key must still be able to run what they graduated."""
    monkeypatch.setattr(helper.shutil, "which", lambda _name: "/usr/bin/claude")
    assert helper.select_provider("grade_essay", "anthropic") == "claude-cli"


def test_an_agent_loop_with_local_tools_cannot_use_the_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`claude -p` reaches tools only over MCP, so loop_body sub-nodes
    bound as in-process callables push the loop to the SDK runner."""
    monkeypatch.setattr(helper.shutil, "which", lambda _name: "/usr/bin/claude")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-whatever")
    provider = helper.select_provider("lead_gen", "anthropic", workload="agent", local_tools=True)
    assert provider == "api"


def test_api_wins_when_no_cli_is_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A deployed image has no `claude` binary, so the first candidate
    drops out on its own — no separate 'deployed' branch needed."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-whatever")
    assert helper.select_provider("grade_essay", "anthropic") == "api"


def test_rote_cloud_is_the_last_resort(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cred = tmp_path / "cloud.json"
    cred.write_text(json.dumps({"base_url": "https://cloud.example", "token": "rote_x"}))
    monkeypatch.setenv("ROTE_CLOUD_CRED_PATH", str(cred))
    assert helper.select_provider("grade_essay", "anthropic") == "rote-cloud"


def test_an_explicit_endpoint_rules_out_the_subscription(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The proven gateway config (ROTE_BASE_URL_<ID> + a bearer token)
    must not be silently ignored just because `claude` is on PATH — the
    operator pointed this judge somewhere on purpose."""
    monkeypatch.setattr(helper.shutil, "which", lambda _name: "/usr/bin/claude")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "cf-token")
    provider = helper.select_provider(
        "grade_essay", "anthropic", base_url="https://gateway.example/ai"
    )
    assert provider == "api"


def test_a_claude_subscription_cannot_serve_an_openai_judge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(helper.shutil, "which", lambda _name: "/usr/bin/claude")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    assert helper.select_provider("grade_essay", "openai") == "api"


def test_explicit_choice_beats_auto_detect(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(helper.shutil, "which", lambda _name: "/usr/bin/claude")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-whatever")
    monkeypatch.setenv("ROTE_INFERENCE", "api")
    assert helper.select_provider("grade_essay", "anthropic") == "api"


def test_per_node_choice_beats_the_global_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(helper.shutil, "which", lambda _name: "/usr/bin/claude")
    monkeypatch.setenv("ROTE_INFERENCE", "api")
    monkeypatch.setenv("ROTE_INFERENCE_GRADE_ESSAY", "claude-cli")
    assert helper.select_provider("grade_essay", "anthropic") == "claude-cli"


def test_an_unavailable_explicit_choice_is_an_error_not_a_downgrade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Never silently bill a lane the operator did not pick. Falling back
    from `claude-cli` to `api` here would move the cost from a flat
    subscription onto a metered key without anyone saying so."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-whatever")
    monkeypatch.setenv("ROTE_INFERENCE", "claude-cli")  # not on PATH
    with pytest.raises(RuntimeError, match="cannot serve grade_essay"):
        helper.select_provider("grade_essay", "anthropic")


def test_an_unknown_provider_name_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROTE_INFERENCE", "free-lunch")
    with pytest.raises(RuntimeError, match="unknown inference provider"):
        helper.select_provider("grade_essay", "anthropic")


def test_nothing_available_lists_every_setup_step() -> None:
    with pytest.raises(RuntimeError) as excinfo:
        helper.select_provider("grade_essay", "anthropic")
    message = str(excinfo.value)
    assert "claude login" in message
    assert "ANTHROPIC_API_KEY" in message
    assert "rote login" in message


# ───────── SDK transport ─────────


def test_judge_translates_sdk_errors_so_they_survive_a_durable_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A durable runtime persists a failed step by pickling the exception,
    and the vendor SDKs' error classes take keyword-only ``response`` /
    ``body`` arguments — pickle cannot rebuild them. Raising one straight
    out of a judge replaced a real 400 (rejecting ``temperature``) with a
    bare ``TypeError`` at the fan-out join, hiding the cause entirely.
    """
    raised = _FakeAPIError("temperature is not supported on this model", response=object(), body={})
    with pytest.raises(TypeError):  # the bug this translation exists to avoid
        pickle.loads(pickle.dumps(raised))

    def _boom(**_kwargs: object) -> object:
        raise raised

    monkeypatch.setitem(sys.modules, "anthropic", _fake_anthropic(_boom))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-whatever")

    with pytest.raises(RuntimeError) as excinfo:
        helper.call_judge(
            node_id="grade_essay",
            client="anthropic",
            model="claude-sonnet-5",
            prompt="Grade it.",
            output_schema={"type": "object", "properties": {}},
        )

    message = str(excinfo.value)
    assert "APIError" in message
    assert "HTTP 400" in message
    assert "temperature is not supported" in message
    assert "via api" in message  # which lane was billed for the failure
    # The whole point: this one crosses the boundary intact.
    assert str(pickle.loads(pickle.dumps(excinfo.value))) == message


def test_api_provider_leaves_credentials_to_the_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """The proven Cloudflare AI Gateway config works because the SDK does
    its own env resolution — passing a credential explicitly here would
    suppress that and break it."""
    seen: dict[str, Any] = {}

    def _create(**kwargs: Any) -> Any:
        return _tool_use_response({"grade": 9})

    module = _fake_anthropic(_create)

    def _ctor(**kwargs: Any) -> Any:
        seen.update(kwargs)
        return types.SimpleNamespace(messages=types.SimpleNamespace(create=_create))

    module.Anthropic = _ctor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", module)
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "cf-token")

    helper.call_judge(
        node_id="grade_essay",
        client="anthropic",
        model="anthropic/claude-sonnet-5",
        base_url="https://gateway.example/ai",
        prompt="Grade it.",
        output_schema={"type": "object", "properties": {}},
    )
    assert seen == {"base_url": "https://gateway.example/ai"}


def test_rote_cloud_passes_its_token_explicitly_so_no_key_rides_along(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An explicit ``auth_token`` also suppresses the SDK's credential env
    reads ("explicit ctor args are total"), which is what keeps a user's
    ambient ANTHROPIC_API_KEY out of a request to our proxy."""
    cred = tmp_path / "cloud.json"
    cred.write_text(json.dumps({"base_url": "https://cloud.example/", "token": "rote_secret"}))
    monkeypatch.setenv("ROTE_CLOUD_CRED_PATH", str(cred))
    monkeypatch.setenv("ROTE_INFERENCE", "rote-cloud")

    seen: dict[str, Any] = {}

    def _create(**kwargs: Any) -> Any:
        return _tool_use_response({"grade": 9})

    module = _fake_anthropic(_create)

    def _ctor(**kwargs: Any) -> Any:
        seen.update(kwargs)
        return types.SimpleNamespace(messages=types.SimpleNamespace(create=_create))

    module.Anthropic = _ctor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", module)

    result = helper.call_judge(
        node_id="grade_essay",
        client="anthropic",
        model="claude-sonnet-5",
        prompt="Grade it.",
        output_schema={"type": "object", "properties": {}},
    )
    assert result == {"grade": 9}
    assert seen == {
        "base_url": "https://cloud.example/v1/inference/anthropic",
        "auth_token": "rote_secret",
    }


def test_usage_log_records_the_provider_that_served_the_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The same token count costs differently per billing lane, so a
    subscription trial and a key trial are not interchangeable rows."""
    log = tmp_path / "usage.jsonl"
    monkeypatch.setenv("ROTE_USAGE_LOG", str(log))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-whatever")
    monkeypatch.setitem(
        sys.modules,
        "anthropic",
        _fake_anthropic(lambda **_kwargs: _tool_use_response({"grade": 9})),
    )

    helper.call_judge(
        node_id="grade_essay",
        client="anthropic",
        model="claude-sonnet-5",
        prompt="Grade it.",
        output_schema={"type": "object", "properties": {}},
    )
    record = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
    assert record == {
        "node": "grade_essay",
        "provider": "api",
        "model": "claude-sonnet-5",
        "input_tokens": 11,
        "output_tokens": 7,
    }


# ───────── CLI transport ─────────


def _cli_envelope(**overrides: Any) -> str:
    envelope = {
        "is_error": False,
        "structured_output": {"grade": 9},
        "result": '{"grade": 9}',
        "usage": {"input_tokens": 740, "output_tokens": 638},
    }
    envelope.update(overrides)
    return json.dumps(envelope)


def test_cli_transport_asks_for_schema_locked_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``--json-schema`` makes ``claude -p`` do a forced tool call, so the
    subscription path has the same structural guarantee as the SDK path.
    The prompt goes on stdin: judge prompts carry whole documents, and
    argv is both size-limited and visible in ``ps``."""
    monkeypatch.setattr(helper.shutil, "which", lambda _name: "/usr/bin/claude")
    seen: dict[str, Any] = {}

    def _run(command: list[str], **kwargs: Any) -> Any:
        seen["command"] = command
        seen["input"] = kwargs["input"]
        seen["env"] = kwargs["env"]
        return types.SimpleNamespace(stdout=_cli_envelope(), stderr="", returncode=0)

    monkeypatch.setattr(helper.subprocess, "run", _run)
    schema = {"type": "object", "properties": {"grade": {"type": "integer"}}}

    result = helper.call_judge(
        node_id="grade_essay",
        client="anthropic",
        model="anthropic/claude-sonnet-5",
        prompt="Grade it.",
        output_schema=schema,
    )

    assert result == {"grade": 9}
    assert seen["input"] == "Grade it."
    command = seen["command"]
    assert "--json-schema" in command
    assert json.loads(command[command.index("--json-schema") + 1]) == schema
    # A gateway-qualified id means nothing to the CLI.
    assert command[command.index("--model") + 1] == "claude-sonnet-5"
    # A judge reasons and answers; it must not read files or run commands.
    # The empty value is load-bearing: the flag is variadic, so a bare
    # `--tools` makes the CLI exit 1 before it ever runs the judge.
    assert command[command.index("--tools") + 1] == ""
    assert command[command.index("--setting-sources") + 1] == ""


def test_cli_transport_forces_subscription_billing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole reason this provider exists: in ``claude -p`` these env
    vars beat an active OAuth session, so a stray key would silently move
    the bill from the subscription to per-token billing."""
    monkeypatch.setattr(helper.shutil, "which", lambda _name: "/usr/bin/claude")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-whatever")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "bearer-whatever")
    monkeypatch.setenv("ROTE_INFERENCE", "claude-cli")
    seen: dict[str, Any] = {}

    def _run(command: list[str], **kwargs: Any) -> Any:
        seen["env"] = kwargs["env"]
        return types.SimpleNamespace(stdout=_cli_envelope(), stderr="", returncode=0)

    monkeypatch.setattr(helper.subprocess, "run", _run)
    helper.call_judge(
        node_id="grade_essay",
        client="anthropic",
        model="claude-sonnet-5",
        prompt="Grade it.",
        output_schema={"type": "object", "properties": {}},
    )
    assert "ANTHROPIC_API_KEY" not in seen["env"]
    assert "ANTHROPIC_AUTH_TOKEN" not in seen["env"]


def test_cli_transport_surfaces_the_envelopes_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """``claude -p`` reports API failures inside a successful exit — the
    returncode alone would call this a win."""
    monkeypatch.setattr(helper.shutil, "which", lambda _name: "/usr/bin/claude")
    monkeypatch.setattr(
        helper.subprocess,
        "run",
        lambda command, **kwargs: types.SimpleNamespace(
            stdout=_cli_envelope(is_error=True, result="model not found"),
            stderr="",
            returncode=0,
        ),
    )
    with pytest.raises(RuntimeError, match="model not found"):
        helper.call_judge(
            node_id="grade_essay",
            client="anthropic",
            model="claude-sonnet-5",
            prompt="Grade it.",
            output_schema={"type": "object", "properties": {}},
        )


def test_cli_transport_falls_back_to_the_result_string(monkeypatch: pytest.MonkeyPatch) -> None:
    """``structured_output`` is the documented field, but the same JSON is
    also in ``result``; a missing key should not lose a paid answer."""
    monkeypatch.setattr(helper.shutil, "which", lambda _name: "/usr/bin/claude")
    envelope = json.loads(_cli_envelope())
    del envelope["structured_output"]
    monkeypatch.setattr(
        helper.subprocess,
        "run",
        lambda command, **kwargs: types.SimpleNamespace(
            stdout=json.dumps(envelope), stderr="", returncode=0
        ),
    )
    assert helper.call_judge(
        node_id="grade_essay",
        client="anthropic",
        model="claude-sonnet-5",
        prompt="Grade it.",
        output_schema={"type": "object", "properties": {}},
    ) == {"grade": 9}


# ───────── the verbatim-copy contract ─────────


def _judge_pipeline() -> Pipeline:
    return mini_pipeline(
        Node(
            id="grade_essay",
            kind=NodeKind.LLM_JUDGE,
            description="Grade an essay.",
            signature_spec={
                "input_schema": {
                    "type": "object",
                    "properties": {"essay": {"type": "string"}},
                    "required": ["essay"],
                },
                "output_schema": {
                    "type": "object",
                    "properties": {"grade": {"type": "integer"}},
                    "required": ["grade"],
                },
                "prompt": "Grade: {{ essay }}",
            },
        )
    )


@pytest.mark.parametrize("runtime", ["dbos", "python", "temporal"])
def test_emitted_helper_is_the_source_module_byte_for_byte(runtime: str, tmp_path: Path) -> None:
    """Same contract ``extracted/_rote_mcp.py`` has: the emitted file IS
    the module's source. A hand-edited copy would be a second
    implementation of the billing decision, drifting silently."""
    out = tmp_path / runtime
    ADAPTERS[runtime]().emit(_judge_pipeline(), out)
    emitted = (out / "signatures" / "_rote_inference.py").read_text(encoding="utf-8")
    assert emitted == Path(helper.__file__).read_text(encoding="utf-8")


def test_the_helper_imports_nothing_but_stdlib_at_module_level() -> None:
    """It executes inside emitted apps where rote is not installed, and
    where the vendor SDKs may not be installed either — that constraint
    is what makes the verbatim copy safe rather than a fork."""
    import ast

    tree = ast.parse(Path(helper.__file__).read_text(encoding="utf-8"))
    roots: list[str] = []
    for node in tree.body:  # module level only; lazy imports live in functions
        if isinstance(node, ast.Import):
            roots += [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            roots.append(node.module.split(".")[0])
    assert set(roots) <= set(sys.stdlib_module_names) | {"__future__"}


# ───────── CLI transport: agent loops ─────────


def _agent_envelope(**overrides: Any) -> str:
    envelope = {
        "is_error": False,
        "result": "Researched 3 accounts.",
        "num_turns": 4,
        "usage": {"input_tokens": 900, "output_tokens": 120},
    }
    envelope.update(overrides)
    return json.dumps(envelope)


def _capture_cli(monkeypatch: pytest.MonkeyPatch, envelope: str) -> dict[str, Any]:
    monkeypatch.setattr(helper.shutil, "which", lambda _name: "/usr/bin/claude")
    seen: dict[str, Any] = {}

    def _run(command: list[str], **kwargs: Any) -> Any:
        seen["command"] = command
        seen["input"] = kwargs["input"]
        return types.SimpleNamespace(stdout=envelope, stderr="", returncode=0)

    monkeypatch.setattr(helper.subprocess, "run", _run)
    return seen


def test_agent_loop_cli_never_passes_a_bare_tools_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--tools`` is variadic, so a bare flag exits 1 before the loop runs.

    This shipped broken and no test caught it: every emitted agent loop on
    the subscription lane died with "option '--tools <tools...>' argument
    missing" the moment no MCP server resolved — the common case on a
    machine with no registry. Asserting the flag's *value* is the point;
    asserting its presence is what let the bug through.
    """
    monkeypatch.setattr(helper, "_mcp_config_for", lambda _tools: ({}, []))
    seen = _capture_cli(monkeypatch, _agent_envelope())

    result = helper.run_agent_loop(
        node_id="target_research",
        description="Research the account.",
        task='{"account": "acme"}',
        model="claude-sonnet-4-6",
        tools=["bright_data_search"],
        max_iterations=6,
    )

    assert result["result"] == "Researched 3 accounts."
    command = seen["command"]
    assert command[command.index("--tools") + 1] == ""
    # No servers resolved means no allowlist and no mcp-config to pass.
    assert "--allowedTools" not in command
    assert "--mcp-config" not in command
    # Bounded by the caller's cap, not the CLI's default.
    assert command[command.index("--max-turns") + 1] == "6"


def test_agent_loop_cli_wires_resolved_servers_and_allowlists_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With servers resolved, the allowlist — not the wiring — is the boundary.

    Over-wiring is safe precisely because ``--allowedTools`` constrains
    what the agent may call; the IR named the tools, so a server offering
    anything else is unreachable.
    """
    servers = {"vendor": {"type": "http", "url": "https://mcp.example.com/mcp"}}
    allowed = ["mcp__vendor__bright_data_search"]
    monkeypatch.setattr(helper, "_mcp_config_for", lambda _tools: (servers, allowed))
    seen = _capture_cli(monkeypatch, _agent_envelope())

    helper.run_agent_loop(
        node_id="target_research",
        description="Research the account.",
        task='{"account": "acme"}',
        model="claude-sonnet-4-6",
        tools=["bright_data_search"],
        max_iterations=6,
    )

    command = seen["command"]
    assert command[command.index("--allowedTools") + 1] == "mcp__vendor__bright_data_search"
    assert json.loads(command[command.index("--mcp-config") + 1]) == {"mcpServers": servers}
    # The tool-free form must not also appear — it would contradict the
    # allowlist and reload the default toolset.
    assert "--tools" not in command
