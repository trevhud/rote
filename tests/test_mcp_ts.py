"""Unit tests for MCP-backed emission in the Node TS adapters.

The DBOS-TS and Inngest adapters emit working, authenticated MCP calls
for nodes with an ``mcp:`` binding (backend "mcp", the default) — the
TS twin of the DBOS Python adapter's behavior. The Cloudflare adapter
must NOT gain this: Workers have no filesystem for the token store
(that's PR3's provisioning design).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rote.adapters import get_adapter
from rote.ir import Pipeline


def _bound_pipeline() -> Pipeline:
    return Pipeline.model_validate(
        {
            "name": "ts_mcp_demo",
            "input": {"type": "In", "required": ["q"]},
            "nodes": [
                {
                    "id": "pull_data",
                    "kind": "external_call",
                    "description": "pull data over MCP",
                    "inputs": {"q": "pipeline.input.q"},
                    "mcp": {"server": "vendor", "tool": "get_data", "args": {"q": "q"}},
                },
                {
                    "id": "plain_call",
                    "kind": "external_call",
                    "description": "no binding — stays a stub",
                    "impl": "extracted/plain_call.py:plain_call",
                },
            ],
            "edges": [{"from": "pull_data", "to": "plain_call"}],
        }
    )


@pytest.mark.parametrize("runtime", ["dbos-ts", "inngest"])
def test_node_ts_adapters_emit_working_mcp_calls(runtime: str, tmp_path: Path) -> None:
    get_adapter(runtime).emit(_bound_pipeline(), tmp_path)

    bound = (tmp_path / "src" / "extracted" / "pull_data.ts").read_text(encoding="utf-8")
    assert 'import { callMcpTool } from "./_roteMcp";' in bound
    assert 'callMcpTool("vendor", null, "get_data"' in bound
    assert "throw new Error" not in bound  # a working body, not a stub

    helper = (tmp_path / "src" / "extracted" / "_roteMcp.ts").read_text(encoding="utf-8")
    assert "refreshAccessToken" in helper
    assert "grant_type" in helper
    assert "rote mcp login" in helper

    unbound = (tmp_path / "src" / "extracted" / "plain_call.ts").read_text(encoding="utf-8")
    assert "throw new Error" in unbound  # binding-less nodes still stub

    package = json.loads((tmp_path / "package.json").read_text(encoding="utf-8"))
    assert "@modelcontextprotocol/sdk" in package["dependencies"]


@pytest.mark.parametrize("runtime", ["dbos-ts", "inngest"])
def test_api_backend_keeps_stubs_and_skips_the_helper(runtime: str, tmp_path: Path) -> None:
    get_adapter(runtime, external_backend="api").emit(_bound_pipeline(), tmp_path)
    bound = (tmp_path / "src" / "extracted" / "pull_data.ts").read_text(encoding="utf-8")
    assert "throw new Error" in bound
    assert not (tmp_path / "src" / "extracted" / "_roteMcp.ts").exists()
    package = json.loads((tmp_path / "package.json").read_text(encoding="utf-8"))
    assert "@modelcontextprotocol/sdk" not in package["dependencies"]


@pytest.mark.parametrize("runtime", ["dbos-ts", "inngest"])
def test_helper_not_emitted_without_bindings(runtime: str, tmp_path: Path) -> None:
    pipeline = Pipeline.model_validate(
        {
            "name": "no_mcp",
            "input": {"type": "In"},
            "nodes": [
                {
                    "id": "plain_call",
                    "kind": "external_call",
                    "description": "plain",
                    "impl": "extracted/plain_call.py:plain_call",
                }
            ],
            "edges": [],
        }
    )
    get_adapter(runtime).emit(pipeline, tmp_path)
    assert not (tmp_path / "src" / "extracted" / "_roteMcp.ts").exists()
    package = json.loads((tmp_path / "package.json").read_text(encoding="utf-8"))
    assert "@modelcontextprotocol/sdk" not in package["dependencies"]


def test_cloudflare_emits_the_workers_helper_not_the_node_one(tmp_path: Path) -> None:
    """Workers have no filesystem — Cloudflare gets the provisioned-secrets
    + KV helper, never the Node token-store reader."""
    get_adapter("cloudflare").emit(_bound_pipeline(), tmp_path)
    helper = (tmp_path / "src" / "extracted" / "_roteMcp.ts").read_text(encoding="utf-8")
    assert "node:fs" not in helper  # the load-bearing difference
    assert "KVNamespace" in helper
    assert "ROTE_MCP_TOKENS" in helper
    assert "grant_type" in helper
    bound = (tmp_path / "src" / "extracted" / "pull_data.ts").read_text(encoding="utf-8")
    assert "callMcpTool(env" in bound
    assert "RoteMcpEnv" in bound


def test_cloudflare_provisioning_surfaces(tmp_path: Path) -> None:
    """The wrangler config, Env interface, and .dev.vars carry the
    provisioning surface for each bound server."""
    import json as _json

    get_adapter("cloudflare").emit(_bound_pipeline(), tmp_path)
    wrangler = (tmp_path / "wrangler.jsonc").read_text(encoding="utf-8")
    assert '"binding": "ROTE_MCP_TOKENS"' in wrangler
    workflow = (tmp_path / "src" / "workflow.ts").read_text(encoding="utf-8")
    assert "ROTE_MCP_VENDOR_REFRESH_TOKEN: string;" in workflow
    assert "ROTE_MCP_TOKENS?: KVNamespace;" in workflow
    dev_vars = (tmp_path / ".dev.vars.example").read_text(encoding="utf-8")
    assert "ROTE_MCP_VENDOR_TOKEN_ENDPOINT=" in dev_vars
    package = _json.loads((tmp_path / "package.json").read_text(encoding="utf-8"))
    assert "@modelcontextprotocol/sdk" in package["dependencies"]


def test_cloudflare_api_backend_keeps_stubs(tmp_path: Path) -> None:
    get_adapter("cloudflare", external_backend="api").emit(_bound_pipeline(), tmp_path)
    assert not (tmp_path / "src" / "extracted" / "_roteMcp.ts").exists()
    bound = (tmp_path / "src" / "extracted" / "pull_data.ts").read_text(encoding="utf-8")
    assert "throw new Error" in bound
    wrangler = (tmp_path / "wrangler.jsonc").read_text(encoding="utf-8")
    assert "ROTE_MCP_TOKENS" not in wrangler


# ───────── Park-on-auth (DBOS-TS) ─────────


def _bound_pipeline_with_retry_and_parallel() -> Pipeline:
    """One retried MCP node in wave 1, a parallel wave of MCP + stub."""
    return Pipeline.model_validate(
        {
            "name": "ts_park_demo",
            "input": {"type": "In", "required": ["q"]},
            "nodes": [
                {
                    "id": "first_pull",
                    "kind": "external_call",
                    "description": "pull data over MCP",
                    "inputs": {"q": "pipeline.input.q"},
                    "mcp": {"server": "vendor", "tool": "get_data", "args": {"q": "q"}},
                    "retry": {"max": 3, "backoff": "exponential"},
                },
                {
                    "id": "par_mcp",
                    "kind": "external_call",
                    "description": "parallel MCP pull",
                    "inputs": {"q": "first_pull.output.q"},
                    "mcp": {"server": "crm", "tool": "push"},
                },
                {
                    "id": "par_stub",
                    "kind": "external_call",
                    "description": "parallel stub",
                    "inputs": {"q": "first_pull.output.q"},
                    "impl": "extracted/par_stub.py:par_stub",
                },
            ],
            "edges": [
                {"from": "first_pull", "to": "par_mcp"},
                {"from": "first_pull", "to": "par_stub"},
            ],
        }
    )


def test_dbos_ts_parks_on_auth(tmp_path: Path) -> None:
    """MCP-backed dispatch routes through the auth-park wrapper: dead
    credentials suspend the workflow (DBOS.recv on rote:auth:<server>)
    instead of failing it, with the wait advertised PORTABLY so the
    Python CLI can read it."""
    get_adapter("dbos-ts").emit(_bound_pipeline_with_retry_and_parallel(), tmp_path)
    main = (tmp_path / "src" / "main.ts").read_text(encoding="utf-8")
    assert 'import { isRoteMcpAuthNeeded } from "./extracted/_roteMcp";' in main
    assert "runWithAuthPark(" in main
    assert "`rote:auth:${server}`" in main
    assert '{ serializationType: "portable" }' in main
    # Retried MCP steps exempt auth failures from the retry budget.
    assert "shouldRetry: mcpShouldRetry" in main
    # Parallel wave: payload bound once, settled rejection re-run through
    # the park wrapper.
    assert "const par_mcp_payload =" in main
    assert "par_mcp_result = unwrap(par_mcp_settled);" in main
    assert "if (!isRoteMcpAuthNeeded(err)) throw err;" in main
    # The helper carries the typed error both sides detect by NAME.
    helper = (tmp_path / "src" / "extracted" / "_roteMcp.ts").read_text(encoding="utf-8")
    assert "class RoteMcpAuthNeeded extends Error" in helper
    assert 'e.name === "RoteMcpAuthNeeded"' in helper


def test_dbos_ts_api_backend_emits_no_park_machinery(tmp_path: Path) -> None:
    get_adapter("dbos-ts", external_backend="api").emit(
        _bound_pipeline_with_retry_and_parallel(), tmp_path
    )
    main = (tmp_path / "src" / "main.ts").read_text(encoding="utf-8")
    assert "runWithAuthPark" not in main
    assert "rote_auth_status" not in main
    assert "shouldRetry" not in main


# ───────── Park-on-auth (Inngest) ─────────


def test_inngest_parks_on_auth(tmp_path: Path) -> None:
    """MCP-backed steps route through runParkable: auth failures wrap in
    NonRetriableError (skipping the retry budget), the run parks on the
    pipeline's rote.auth.<server> event, and one broadcast wakes every
    parked run — Inngest events fan out, so no discovery is needed."""
    get_adapter("inngest").emit(_bound_pipeline(), tmp_path)
    src = (tmp_path / "src" / "inngest" / "pipeline.ts").read_text(encoding="utf-8")
    assert 'import { isRoteMcpAuthNeeded } from "../extracted/_roteMcp";' in src
    assert 'runParkable(step, "pull_data", "ts_mcp_demo/rote.auth.vendor", "vendor"' in src
    assert "new NonRetriableError(" in src
    assert "-auth-wait-" in src  # fresh waitForEvent id per park
    assert 'timeout: "30d"' in src
    # Binding-less nodes keep the plain step.run.
    assert 'step.run("plain_call"' in src


def test_inngest_api_backend_emits_no_park_machinery(tmp_path: Path) -> None:
    get_adapter("inngest", external_backend="api").emit(_bound_pipeline(), tmp_path)
    src = (tmp_path / "src" / "inngest" / "pipeline.ts").read_text(encoding="utf-8")
    assert "runParkable" not in src
    assert "rote.auth." not in src


# ───────── Park-on-auth (Cloudflare) ─────────


def test_cloudflare_parks_on_auth(tmp_path: Path) -> None:
    """MCP-backed step.do wraps auth failures in NonRetryableError (no
    should-retry predicate exists) and parks the instance on a
    rote_auth_<server> waitForEvent with an explicit long timeout —
    the default is 24h and expiry THROWS, failing the instance."""
    get_adapter("cloudflare").emit(_bound_pipeline(), tmp_path)
    src = (tmp_path / "src" / "workflow.ts").read_text(encoding="utf-8")
    assert 'import { NonRetryableError } from "cloudflare:workflows";' in src
    assert 'import { isRoteMcpAuthNeeded } from "./extracted/_roteMcp";' in src
    assert 'type: "rote_auth_vendor", timeout: "30 days"' in src
    assert "auth retry" in src  # fresh step names per attempt
    assert "function stepNeedsAuth" in src
    # Binding-less nodes keep the plain step.do.
    assert 'step.do(\n            "plain_call"' in src


def test_cloudflare_api_backend_emits_no_park_machinery(tmp_path: Path) -> None:
    get_adapter("cloudflare", external_backend="api").emit(_bound_pipeline(), tmp_path)
    src = (tmp_path / "src" / "workflow.ts").read_text(encoding="utf-8")
    assert "NonRetryableError" not in src
    assert "rote_auth_" not in src
    assert "stepNeedsAuth" not in src


# ───────── Binding-backed Workers MCP helper (mcp_client="binding") ─────────
#
# `emit(..., mcp_client="binding")` swaps the direct helper (per-server
# secrets + OAuth refresh + KV token cache) for a variant that delegates
# every MCP operation to a platform-provisioned ROTE_MCP service binding.
# The exported surface — and the workflow's park-on-auth machinery — stay
# identical; only the provisioning surface changes.


def test_cloudflare_binding_mode_delegates_to_the_service_binding(tmp_path: Path) -> None:
    get_adapter("cloudflare").emit(_bound_pipeline(), tmp_path, mcp_client="binding")
    helper = (tmp_path / "src" / "extracted" / "_roteMcp.ts").read_text(encoding="utf-8")
    assert "env.ROTE_MCP.call(auth, server, tool, args)" in helper
    assert "env.ROTE_MCP.listTools(auth, server)" in helper
    # The platform proxy's auth contract, translated back into the same
    # error the park loop already detects by name.
    assert '"ROTE_MCP_AUTH_NEEDED:"' in helper
    assert "class RoteMcpAuthNeeded extends Error" in helper
    assert 'e.name === "RoteMcpAuthNeeded"' in helper
    # No token machinery in this variant: no OAuth grant, no KV cache,
    # no MCP SDK client.
    assert "grant_type" not in helper
    assert "refreshAccessToken" not in helper
    assert "KVNamespace" not in helper
    assert "@modelcontextprotocol" not in helper
    # The emitted node module is unchanged — same import, same call shape.
    bound = (tmp_path / "src" / "extracted" / "pull_data.ts").read_text(encoding="utf-8")
    assert 'import { callMcpTool, type RoteMcpEnv } from "./_roteMcp";' in bound
    assert "callMcpTool(env" in bound


def test_cloudflare_binding_mode_signs_every_proxy_call(tmp_path: Path) -> None:
    """The proxy's caller-auth contract: every call carries {tenant_id,
    pipeline, run_id?, sig} built from dispatcher-injected env vars, a
    missing ROTE_MCP_SIG is a config error naming it, and the proxy's
    reply shapes (is_error, snake_case input_schema) are normalized to
    the direct helper's behavior."""
    get_adapter("cloudflare").emit(_bound_pipeline(), tmp_path, mcp_client="binding")
    helper = (tmp_path / "src" / "extracted" / "_roteMcp.ts").read_text(encoding="utf-8")
    # The auth object and its env sources.
    assert 'tenant_id: requireVar(env, "ROTE_TENANT_ID")' in helper
    assert 'pipeline: requireVar(env, "ROTE_PIPELINE")' in helper
    assert 'requireVar(env, "ROTE_MCP_SIG")' in helper
    assert "env.ROTE_RUN_ID" in helper  # optional — only sent when present
    # A missing injected var throws a config error naming the variable.
    assert "`the ${name} variable is not set" in helper
    # The proxy's detail leads with the server name (its parse token) —
    # translate() strips it so the parked-error reason doesn't repeat it.
    assert "reason.startsWith(`${server} `)" in helper
    assert "reason.slice(server.length + 1)" in helper
    # is_error maps to the same thrown-error path as the direct helper's
    # result.isError; string content is JSON-parsed like text blocks.
    assert "if (result.is_error)" in helper
    assert "returned an error" in helper
    assert "JSON.parse(content)" in helper
    # listTools' snake_case wire shape is normalized into McpToolSpec.
    assert "tool.input_schema" in helper
    assert 'tool.description ?? ""' in helper


def test_cloudflare_binding_mode_env_and_config_surfaces(tmp_path: Path) -> None:
    """Binding mode swaps the provisioning surface: Env declares the one
    ROTE_MCP binding instead of per-server secrets + the KV cache, wrangler
    provisions no KV namespace, .dev.vars declares no MCP secrets, and the
    MCP SDK leaves package.json."""
    get_adapter("cloudflare").emit(_bound_pipeline(), tmp_path, mcp_client="binding")
    workflow = (tmp_path / "src" / "workflow.ts").read_text(encoding="utf-8")
    assert "ROTE_MCP: RoteMcpBinding;" in workflow
    assert "type RoteMcpBinding" in workflow  # imported from the helper
    # The dispatcher-injected caller identity is part of the declared Env.
    assert "ROTE_TENANT_ID: string;" in workflow
    assert "ROTE_PIPELINE: string;" in workflow
    assert "ROTE_RUN_ID?: string;" in workflow
    assert "ROTE_RUN_ID: string;" not in workflow
    assert "ROTE_MCP_SIG: string;" in workflow
    assert "ROTE_MCP_VENDOR_REFRESH_TOKEN" not in workflow
    assert "ROTE_MCP_TOKENS" not in workflow
    wrangler = (tmp_path / "wrangler.jsonc").read_text(encoding="utf-8")
    assert "kv_namespaces" not in wrangler
    dev_vars = (tmp_path / ".dev.vars.example").read_text(encoding="utf-8")
    assert "ROTE_MCP_VENDOR" not in dev_vars
    package = json.loads((tmp_path / "package.json").read_text(encoding="utf-8"))
    assert "@modelcontextprotocol/sdk" not in package["dependencies"]
    # The manifest's requirements key is mode-independent.
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["mcp_servers"] == {"vendor": ["pull_data"]}


def test_cloudflare_binding_mode_keeps_the_park_loop(tmp_path: Path) -> None:
    """Parkable steps, auth_event_type names, and the NonRetryableError
    opt-out are identical in both client modes — only the helper changed."""
    get_adapter("cloudflare").emit(_bound_pipeline(), tmp_path, mcp_client="binding")
    src = (tmp_path / "src" / "workflow.ts").read_text(encoding="utf-8")
    assert 'import { NonRetryableError } from "cloudflare:workflows";' in src
    assert "isRoteMcpAuthNeeded" in src
    assert 'type: "rote_auth_vendor", timeout: "30 days"' in src
    assert "function stepNeedsAuth" in src


def test_cloudflare_binding_mode_binds_agent_tools_through_the_proxy(
    bdr_pipeline: Pipeline, tmp_path: Path
) -> None:
    """BDR's agent loops declare MCP tools without any `mcp:`-bound node —
    binding mode still emits the helper, and the loops import the same
    bindAgentTools surface (now backed by ROTE_MCP.listTools)."""
    get_adapter("cloudflare").emit(bdr_pipeline, tmp_path, mcp_client="binding")
    helper = (tmp_path / "src" / "extracted" / "_roteMcp.ts").read_text(encoding="utf-8")
    assert "bindAgentTools" in helper
    assert "env.ROTE_MCP.listTools(auth, server)" in helper
    loop = (tmp_path / "src" / "extracted" / "lead_generation_loop.ts").read_text(encoding="utf-8")
    assert 'import { bindAgentTools, type RoteMcpEnv } from "./_roteMcp";' in loop
    workflow = (tmp_path / "src" / "workflow.ts").read_text(encoding="utf-8")
    assert "ROTE_MCP: RoteMcpBinding;" in workflow


def test_cloudflare_binding_mode_via_factory_option(tmp_path: Path) -> None:
    get_adapter("cloudflare", mcp_client="binding").emit(_bound_pipeline(), tmp_path)
    helper = (tmp_path / "src" / "extracted" / "_roteMcp.ts").read_text(encoding="utf-8")
    assert "env.ROTE_MCP.call" in helper


def test_cloudflare_default_mode_is_direct(tmp_path: Path) -> None:
    """Without the kwarg the existing behavior holds — the negative half,
    so binding mode can't silently become the default."""
    get_adapter("cloudflare").emit(_bound_pipeline(), tmp_path)
    helper = (tmp_path / "src" / "extracted" / "_roteMcp.ts").read_text(encoding="utf-8")
    assert "refreshAccessToken" in helper
    assert "env.ROTE_MCP.call" not in helper
    wrangler = (tmp_path / "wrangler.jsonc").read_text(encoding="utf-8")
    assert '"binding": "ROTE_MCP_TOKENS"' in wrangler


def test_cloudflare_rejects_unknown_mcp_client(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mcp_client"):
        get_adapter("cloudflare").emit(_bound_pipeline(), tmp_path, mcp_client="proxy")
    with pytest.raises(ValueError, match="mcp_client"):
        get_adapter("cloudflare", mcp_client="proxy")


def test_binding_mode_emit_wires_the_mode_into_the_node_module(tmp_path: Path) -> None:
    """The adapter must pass its mode down to the node-module emitter.

    Deliberately goes through `emit(...)` rather than calling the emitter
    directly: the unit-level test below passes even when the adapter
    forgets to forward `mcp_client`, which is exactly the wiring bug that
    leaves an operator reading the wrong provisioning instructions.
    """
    get_adapter("cloudflare").emit(_bound_pipeline(), tmp_path, mcp_client="binding")
    modules = list((tmp_path / "src" / "extracted").glob("*.ts"))
    headers = [m.read_text(encoding="utf-8") for m in modules]
    mcp_headers = [h for h in headers if "MCP-backed" in h]
    assert mcp_headers, "expected at least one MCP-backed node module"
    for src in mcp_headers:
        assert "ROTE_MCP service" in src
        assert "rote mcp export" not in src


def test_binding_mode_node_module_does_not_advertise_worker_secrets() -> None:
    """The emitted node module must name the mechanism it was built for.

    Binding mode reads no per-server secrets, so telling the operator to
    run `rote mcp export <server>` sends them to provision secrets this
    build will never read. This header is the file someone opens when a
    call fails, which is exactly when a wrong instruction costs the most.
    """
    from rote.adapters._ts_common import emit_workers_mcp_call_module
    from rote.ir import MCPBinding, Node, NodeKind

    node = Node(
        id="fetch_thing",
        kind=NodeKind.EXTERNAL_CALL,
        description="Fetch a thing.",
        impl="extracted/fetch.py:fetch",
        mcp=MCPBinding(server="vendorapi", tool="get_thing"),
    )

    binding_src = emit_workers_mcp_call_module(node, generated_by="test", mcp_client="binding")
    assert "ROTE_MCP service" in binding_src
    assert "rote mcp export" not in binding_src
    assert "Worker secrets" not in binding_src

    # The negative half: direct mode still says what it has always said,
    # so this cannot be satisfied by deleting the sentence outright.
    direct_src = emit_workers_mcp_call_module(node, generated_by="test", mcp_client="direct")
    assert "rote mcp export vendorapi" in direct_src
    assert "ROTE_MCP service" not in direct_src
