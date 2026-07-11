"""End-to-end: park-on-auth on Cloudflare Workflows (wrangler dev, workerd).

The instance-addressed release loop, live on the real Workers runtime:

1. Emit a Cloudflare app for an MCP-backed pipeline and run it under
   ``wrangler dev --local`` with NO provisioned ``ROTE_MCP_*`` secrets —
   the Workers helper throws RoteMcpAuthNeeded (not provisioned), the
   step wraps it in NonRetryableError (no retry budget burned), and the
   instance parks on ``step.waitForEvent`` for ``rote_auth_vendor``.
2. Prove release precision: a *different* server's event leaves it
   parked.
3. Fix the credential by writing a token into the ``ROTE_MCP_TOKENS``
   KV cache (the production analog is re-provisioned secrets — KV
   state deliberately supersedes them, which is what makes an
   in-session fix possible at all), send the real release event, and
   the instance resumes: the retried step drives ``callMcpTool``
   through the real ``@modelcontextprotocol/sdk`` streamable-HTTP
   client *inside workerd* against a live FastMCP server and the run
   completes with real data. (That call closes the long-standing
   "live workerd run of the TS MCP client" gap — previously the
   Workers MCP output was only ever typechecked.)

The release event is delivered via the overlay's ``sendEvent`` route —
the same per-instance send `wrangler workflows instances send-event
--local` and the REST channel perform; the REST channel itself is
covered by unit tests (local dev has no REST surface).

Slow: npm install + wrangler dev (workerd) + a live FastMCP server.
Park detection uses the local-dev step-output-stability technique from
tests/test_cloudflare_e2e.py (status stays "running" while parked).
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import textwrap
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from rote.adapters.cloudflare import CloudflareAdapter, workflow_class_name
from rote.ir import MCPBinding, Node, NodeKind, Pipeline, PipelineInput, RetryPolicy

pytest.importorskip("fastmcp")

from tests.test_cloudflare_e2e import (  # noqa: E402 — reuse the proven live harness
    _http_get,
    _http_post,
    _node_available,
    _wait_until_parked_or_complete,
)

pytestmark = pytest.mark.slow

_PIPELINE_NAME = "cf_park_e2e"

_MOCK_SERVER = textwrap.dedent(
    """
    import sys
    from fastmcp import FastMCP

    mcp = FastMCP("mock-vendor")

    @mcp.tool
    def enrich_contact(contact_id: str) -> dict:
        return {"contact_id": contact_id, "source": "mock-vendor-mcp"}

    if __name__ == "__main__":
        mcp.run(transport="http", host="127.0.0.1", port=int(sys.argv[1]))
    """
)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="module")
def mcp_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    src = tmp_path_factory.mktemp("mcpsrv") / "server.py"
    src.write_text(_MOCK_SERVER)
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, str(src), str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + 20
    while time.time() < deadline:
        with socket.socket() as s:
            s.settimeout(0.5)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                break
        time.sleep(0.2)
    else:
        proc.kill()
        raise RuntimeError(f"mock MCP server did not come up on :{port}")
    try:
        yield f"http://127.0.0.1:{port}/mcp/"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def _park_pipeline(mcp_url: str) -> Pipeline:
    node = Node(
        id="enrich",
        kind=NodeKind.EXTERNAL_CALL,
        description="Enrich a contact via the vendor MCP server.",
        input={"contact_id": "str"},
        inputs={"contact_id": "pipeline.input.contact_id"},
        output="dict",
        # The endpoint rides in the binding so the Worker needs no
        # ROTE_MCP_VENDOR_URL var — only credentials are "missing".
        mcp=MCPBinding(
            server="vendor",
            tool="enrich_contact",
            url=mcp_url,
            args={"contact_id": "contact_id"},
        ),
        # Retries on: the NonRetryableError wrap must keep auth failures
        # from burning retries.limit worth of delay before the park.
        retry=RetryPolicy(max=3, backoff="exponential"),
    )
    return Pipeline(
        name=_PIPELINE_NAME,
        description="Park-on-auth e2e pipeline (Cloudflare).",
        input=PipelineInput(type="Req", required=["contact_id"]),
        nodes=[node],
        edges=[],
        entry_nodes=["enrich"],
        exit_nodes=["enrich"],
    )


def _test_index_ts(class_name: str) -> str:
    """Overlay index.ts: instance control + a KV credential-fix route."""
    return f"""\
import {{ {class_name}, type Env, type Params }} from "./workflow";

export {{ {class_name} }};

export default {{
    async fetch(req: Request, env: Env): Promise<Response> {{
        const url = new URL(req.url);
        if (url.pathname === "/start" || url.pathname === "/") {{
            const raw = await req.json().catch(() => ({{}}));
            const params = (raw ?? {{}}) as Params;
            const inst = await env.PIPELINE.create({{ params }});
            return Response.json({{ id: inst.id, status: await inst.status() }});
        }}
        let m = url.pathname.match(/^\\/status\\/([^/]+)$/);
        if (m) {{
            const inst = await env.PIPELINE.get(m[1]);
            return Response.json(await inst.status());
        }}
        m = url.pathname.match(/^\\/event\\/([^/]+)\\/([^/]+)$/);
        if (m) {{
            const inst = await env.PIPELINE.get(m[1]);
            const payload = await req.json().catch(() => ({{}}));
            await inst.sendEvent({{ type: m[2], payload }});
            return Response.json({{ ok: true }});
        }}
        m = url.pathname.match(/^\\/kvfix\\/([^/]+)$/);
        if (m) {{
            // The "credential fix": production re-provisions secrets via
            // `rote mcp export` + wrangler; the KV token cache supersedes
            // them, so writing a fresh token here is the same code path
            // the helper reads on its next attempt.
            await env.ROTE_MCP_TOKENS!.put(
                `mcp-token:${{m[1]}}`,
                JSON.stringify({{ access_token: "kv-fixed-token", expires_at: null }}),
            );
            return Response.json({{ ok: true }});
        }}
        return new Response("not found", {{ status: 404 }});
    }},
}} satisfies ExportedHandler<Env>;
"""


def test_cf_instance_parks_and_event_release_completes_it(mcp_server: str, tmp_path: Path) -> None:
    if not _node_available():
        pytest.skip("Node / npm not available")
    import os

    pipeline = _park_pipeline(mcp_server)
    out = tmp_path / "app"
    CloudflareAdapter().emit(pipeline, out)
    class_name = workflow_class_name(pipeline)
    (out / "src" / "index.ts").write_text(_test_index_ts(class_name), encoding="utf-8")
    # The emitted KV namespace id is a REPLACE-me placeholder; local dev
    # only needs a syntactically plausible id.
    wrangler_path = out / "wrangler.jsonc"
    wrangler = wrangler_path.read_text(encoding="utf-8")
    wrangler = wrangler.replace(
        "REPLACE-run: npx wrangler kv namespace create rote-mcp-tokens", "0" * 32
    )
    wrangler_path.write_text(wrangler, encoding="utf-8")

    npm = subprocess.run(
        ["npm", "install", "--no-audit", "--no-fund"],
        cwd=out,
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "npm_config_progress": "false"},
    )
    assert npm.returncode == 0, f"npm install failed:\n{npm.stdout}\n{npm.stderr}"

    port = _free_port()
    log_path = out / "wrangler-dev.log"
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            ["npx", "wrangler", "dev", "--port", str(port), "--local"],
            cwd=out,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "WRANGLER_LOG": "info"},
        )
    try:
        ready_deadline = time.time() + 60
        while time.time() < ready_deadline:
            assert proc.poll() is None, (
                f"wrangler dev exited (code {proc.returncode}); log:\n{log_path.read_text()}"
            )
            try:
                _http_get(f"http://127.0.0.1:{port}/status/nonexistent")
            except Exception:  # noqa: BLE001 — boot poll; any response means listening
                import urllib.error

                try:
                    _http_post(f"http://127.0.0.1:{port}/start", {"contact_id": "boot-check"})
                    break
                except urllib.error.URLError:
                    time.sleep(0.5)
                    continue
            break
        else:
            pytest.fail(f"wrangler dev did not start in 60s; log:\n{log_path.read_text()}")

        started = _http_post(f"http://127.0.0.1:{port}/start", {"contact_id": "lead-5"})
        instance_id = started["id"]

        # ── The instance parks: step outputs stay empty and stable while
        # local-dev status remains "running" (no "waiting" in local dev).
        state = _wait_until_parked_or_complete(
            port, instance_id, min_step_count=0, timeout=90, stability_polls=6
        )
        assert state.get("status") == "running", f"expected a parked instance: {state}"

        # ── Release precision: the wrong server's event wakes nothing ──
        _http_post(f"http://127.0.0.1:{port}/event/{instance_id}/rote_auth_other", {})
        time.sleep(4)
        wrong = _http_get(f"http://127.0.0.1:{port}/status/{instance_id}")
        assert wrong.get("status") == "running"
        assert not wrong.get("__LOCAL_DEV_STEP_OUTPUTS")

        # ── Fix the credential (KV supersedes secrets), then release ──
        _http_post(f"http://127.0.0.1:{port}/kvfix/vendor", {})
        _http_post(
            f"http://127.0.0.1:{port}/event/{instance_id}/rote_auth_vendor",
            {"server": "vendor", "released_by": "rote mcp"},
        )

        # ── The instance wakes; the retried step calls the live MCP
        # server through the real TS SDK inside workerd and completes.
        deadline = time.time() + 90
        final: dict = {}
        while time.time() < deadline:
            final = _http_get(f"http://127.0.0.1:{port}/status/{instance_id}")
            if final.get("status") == "complete":
                break
            assert final.get("status") != "errored", f"instance errored: {final}"
            time.sleep(1)
        assert final.get("status") == "complete", f"never completed: {final}"

        outputs = json.dumps(final.get("__LOCAL_DEV_STEP_OUTPUTS", []))
        assert "mock-vendor-mcp" in outputs  # live data, through the SDK, in workerd
        assert "lead-5" in outputs
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
