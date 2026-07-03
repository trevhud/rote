"""Slow integration tests for the Cloudflare adapter.

Three tiers of validation, each requiring more of the host toolchain:

1. ``test_emitted_typescript_compiles`` — runs ``tsc --noEmit`` over the
   emitted output. Catches type errors, missing imports, and SDK
   surface drift.
2. ``test_node_modules_contains_expected_packages`` — confirms the
   emitted ``package.json`` resolves against the real npm registry.
3. ``test_workflow_executes_through_hitl_gates`` — boots ``wrangler dev``
   against a copy of the emitted output with stubs replaced by mocks,
   then drives a real Cloudflare Workflow instance through both HITL
   gates using ``inst.sendEvent``, polling status until ``complete``.
   This is the only tier that exercises the actual durable-execution
   machinery — ``step.do`` retries, ``step.waitForEvent`` event delivery,
   step output threading, etc.

All tests are gated by ``@pytest.mark.slow`` so they're easy to skip
during fast iteration: ``pytest -m 'not slow'``. The Node toolchain is
detected at runtime; if missing, tests skip with a clear message.
CI should have Node available.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import pytest

from rote.adapters.cloudflare import CloudflareAdapter
from rote.ir import Pipeline, load_pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
BDR_PIPELINE_YAML = REPO_ROOT / "examples" / "bdr-outreach" / "expected" / "pipeline.yaml"


pytestmark = pytest.mark.slow


def _node_available() -> bool:
    return shutil.which("node") is not None and shutil.which("npm") is not None


@pytest.fixture(scope="module")
def bdr_pipeline() -> Pipeline:
    return load_pipeline(BDR_PIPELINE_YAML)


@pytest.fixture(scope="module")
def emitted_dir(bdr_pipeline: Pipeline, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Emit the BDR pipeline + run npm install once per module."""
    if not _node_available():
        pytest.skip("Node / npm not available — skipping cloudflare e2e tests")

    out = tmp_path_factory.mktemp("cf-e2e")
    CloudflareAdapter().emit(bdr_pipeline, out)

    # Install dependencies. Use a fresh cache to avoid stale lockfile drift,
    # but keep silent unless something fails (the install is noisy).
    proc = subprocess.run(
        ["npm", "install", "--no-audit", "--no-fund"],
        cwd=out,
        capture_output=True,
        text=True,
        timeout=180,
        env={**os.environ, "npm_config_progress": "false"},
    )
    if proc.returncode != 0:
        pytest.fail(
            f"npm install failed in {out}:\n--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}"
        )
    return out


def test_emitted_typescript_compiles(emitted_dir: Path) -> None:
    """Run `tsc --noEmit` over the emitted output. Zero diagnostics expected.

    This is the strongest static signal short of running the workflow.
    If this passes, the emitted code typechecks against the
    @cloudflare/workers-types definitions, the Anthropic SDK, and Zod.
    """
    proc = subprocess.run(
        ["npx", "--no-install", "tsc", "--noEmit"],
        cwd=emitted_dir,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"tsc --noEmit reported errors in emitted Cloudflare code:\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )


def test_node_modules_contains_expected_packages(emitted_dir: Path) -> None:
    """Sanity check that the emitted package.json's deps are real and resolvable."""
    node_modules = emitted_dir / "node_modules"
    assert (node_modules / "@anthropic-ai" / "sdk").exists()
    assert (node_modules / "zod").exists()
    assert (node_modules / "@cloudflare" / "workers-types").exists()
    assert (node_modules / "typescript").exists()
    assert (node_modules / "wrangler").exists()


# ───────── Live workflow execution ─────────


def _to_camel_case(s: str) -> str:
    parts = s.replace("-", "_").split("_")
    if not parts or not parts[0]:
        return ""
    return parts[0] + "".join(p.capitalize() for p in parts[1:])


# Test wrapper for the emitted index.ts. Adds /event/<id>/<type> and
# /status/<id> routes so the test can drive a real instance through
# its HITL gates over plain HTTP.
_TEST_INDEX_TS = """\
import { BdrCampaignWorkflow, type Env, type Params } from "./workflow";

export { BdrCampaignWorkflow };

export default {
    async fetch(req: Request, env: Env): Promise<Response> {
        const url = new URL(req.url);
        if (url.pathname === "/start" || url.pathname === "/") {
            const raw = await req.json().catch(() => ({}));
            const params = (raw ?? {}) as Params;
            const inst = await env.PIPELINE.create({ params });
            return Response.json({ id: inst.id, status: await inst.status() });
        }
        let m = url.pathname.match(/^\\/status\\/([^/]+)$/);
        if (m) {
            const inst = await env.PIPELINE.get(m[1]);
            return Response.json(await inst.status());
        }
        m = url.pathname.match(/^\\/event\\/([^/]+)\\/([^/]+)$/);
        if (m) {
            const inst = await env.PIPELINE.get(m[1]);
            const payload = await req.json().catch(() => ({}));
            await inst.sendEvent({ type: m[2], payload });
            return Response.json({ ok: true });
        }
        return new Response("not found", { status: 404 });
    },
} satisfies ExportedHandler<Env>;
"""


def _write_test_overlay(out_dir: Path) -> None:
    """Replace stubs with echo mocks and overlay the test index.ts.

    The emitted ``extracted/*.ts`` and ``signatures/*.ts`` modules all
    throw NotImplementedError. For an integration test that exercises
    step orchestration (not real I/O), we replace each with a function
    that returns a serializable canned object so the workflow can
    progress through its waves. The mock echoes the input it received
    (``received``) so the test can assert that data-flow threading
    delivered real payloads between steps.
    """
    src = out_dir / "src"
    for sub in ("extracted", "signatures"):
        d = src / sub
        if not d.exists():
            continue
        for f in d.glob("*.ts"):
            node_id = f.stem
            fn = _to_camel_case(node_id)
            f.write_text(
                f"export async function {fn}("
                f"input?: unknown, _env?: unknown"
                f"): Promise<Record<string, unknown>> {{\n"
                f"  return {{ mocked: true, node: {json.dumps(node_id)}, "
                f"received: input ?? null }};\n"
                f"}}\n",
                encoding="utf-8",
            )
    (src / "index.ts").write_text(_TEST_INDEX_TS, encoding="utf-8")


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _http_post(url: str, body: dict | None = None, timeout: int = 10) -> dict:
    data = json.dumps(body or {}).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_get(url: str, timeout: int = 10) -> dict:
    with urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _wait_until_parked_or_complete(
    port: int,
    instance_id: str,
    *,
    min_step_count: int = 0,
    timeout: float = 60.0,
    poll_interval: float = 0.5,
    stability_polls: int = 3,
) -> dict:
    """Poll /status/<id> until the workflow either reaches a terminal
    state or parks on a ``waitForEvent``.

    Cloudflare's local-dev status JSON keeps the top-level
    ``status`` field as ``"running"`` while the instance is parked inside
    a ``step.waitForEvent`` — there's no literal ``"waiting"`` enum value
    we can poll for. We infer parking by tracking
    ``__LOCAL_DEV_STEP_OUTPUTS`` length: when it reaches ``min_step_count``
    *and* stops growing for ``stability_polls`` consecutive ticks, the
    instance is necessarily either at a HITL gate or done. Terminal
    states (``complete`` / ``errored``) short-circuit the wait.
    """
    deadline = time.time() + timeout
    last_state: dict = {}
    last_count = -1
    stable = 0
    while time.time() < deadline:
        try:
            last_state = _http_get(f"http://127.0.0.1:{port}/status/{instance_id}")
        except URLError:
            time.sleep(poll_interval)
            continue
        s = last_state.get("status")
        if s == "errored":
            raise AssertionError(
                f"workflow {instance_id!r} reached errored state: "
                f"{json.dumps(last_state, indent=2)}"
            )
        if s == "complete":
            return last_state
        outputs = last_state.get("__LOCAL_DEV_STEP_OUTPUTS", [])
        count = len(outputs)
        if count >= min_step_count and count == last_count:
            stable += 1
            if stable >= stability_polls:
                return last_state
        else:
            stable = 0
            last_count = count
        time.sleep(poll_interval)
    raise AssertionError(
        f"timeout waiting for parked-or-complete on {instance_id!r}; "
        f"last state: {json.dumps(last_state, indent=2)}"
    )


@pytest.fixture(scope="module")
def wrangler_dev_session(bdr_pipeline: Pipeline, tmp_path_factory: pytest.TempPathFactory):
    """Spin up `wrangler dev` against a mocked copy of the emitted output.

    Module-scoped: startup takes 5–10s, so we share one running instance
    across all execution tests. Each test creates a fresh workflow
    instance via POST / so they don't share durable state.
    """
    if not _node_available():
        pytest.skip("Node / npm not available")

    out = tmp_path_factory.mktemp("cf-workflow-e2e")
    CloudflareAdapter().emit(bdr_pipeline, out)
    _write_test_overlay(out)

    npm_proc = subprocess.run(
        ["npm", "install", "--no-audit", "--no-fund"],
        cwd=out,
        capture_output=True,
        text=True,
        timeout=180,
        env={**os.environ, "npm_config_progress": "false"},
    )
    if npm_proc.returncode != 0:
        pytest.fail(f"npm install failed in {out}:\n{npm_proc.stdout}\n{npm_proc.stderr}")

    port = _find_free_port()
    log_path = out / "wrangler-dev.log"
    log = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        ["npx", "wrangler", "dev", "--port", str(port), "--local"],
        cwd=out,
        stdout=log,
        stderr=subprocess.STDOUT,
        env={**os.environ, "WRANGLER_LOG": "info"},
    )

    # Poll the worker until it answers HTTP. Cold-start is ~5–10s; give
    # it 45s of headroom on slow CI.
    ready_deadline = time.time() + 45
    while time.time() < ready_deadline:
        if proc.poll() is not None:
            log.close()
            pytest.fail(
                f"wrangler dev exited prematurely (code {proc.returncode}); log:\n"
                f"{log_path.read_text()}"
            )
        try:
            with urlopen(f"http://127.0.0.1:{port}/", timeout=2) as resp:
                resp.read()
            break
        except (TimeoutError, URLError, ConnectionResetError):
            time.sleep(0.5)
    else:
        proc.terminate()
        log.close()
        pytest.fail(
            f"wrangler dev did not start in 45s on port {port}; log:\n{log_path.read_text()}"
        )

    try:
        yield port, out
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        log.close()


def test_workflow_executes_through_hitl_gates(
    wrangler_dev_session: tuple[int, Path], bdr_pipeline: Pipeline
) -> None:
    """Drive a real workflow instance through both BDR HITL gates.

    Concretely:

    1. POST / creates an instance.
    2. The workflow executes its non-HITL steps (mocked) until it
       reaches ``contact_review_gate`` and parks in ``waiting``.
    3. POST /event/<id>/contact_review_approved resumes it.
    4. The workflow runs through Phase 4–7 steps (mocked) until it
       reaches ``manual_enrollment_handoff`` and parks in ``waiting``.
    5. POST /event/<id>/bdr_enrollment_complete resumes it.
    6. Workflow reaches ``complete`` terminal status.

    This exercises the entire durable-execution surface of the emitted
    code: ``step.do`` calls, ``step.waitForEvent`` registration, real
    event delivery via ``inst.sendEvent``, and step output return.
    Failures here mean the emitted IR-to-TS mapping is broken in a way
    no static analysis would catch.
    """
    port, _ = wrangler_dev_session

    def _node_names(state: dict) -> set[str]:
        return {
            o.get("node")
            for o in state.get("__LOCAL_DEV_STEP_OUTPUTS", [])
            if isinstance(o, dict) and o.get("node")
        }

    def _step_output(state: dict, node: str) -> dict:
        return next(
            o
            for o in state.get("__LOCAL_DEV_STEP_OUTPUTS", [])
            if isinstance(o, dict) and o.get("node") == node
        )

    pre_gate_nodes = {"target_research", "taxonomy_lookup", "lead_generation_loop"}
    post_gate_nodes = pre_gate_nodes | {
        "hubspot_upsert",
        "hubspot_create_list",
        "exclusion_check_dnc",
        "exclusion_check_recent",
        "exclusion_check_sequence",
        "personalize_email",
        "create_sales_template",
        "pre_enrollment_report",
    }

    # Step 1: create instance with a complete campaign brief — the emitted
    # workflow threads real payloads now, so the input contract matters.
    # Values reuse the fictionalized examples from the IR's comments.
    brief = {
        "drug_brand": "Orladeyo",
        "drug_generic": "berotralstat",
        "condition_full": "hereditary angioedema",
        "condition_acronym": "HAE",
        "therapeutic_area": "rare disease, hematology",
        "manufacturer": "BioCryst Pharmaceuticals",
        "campaign_type": "drug-specific",
        "target_quota": 3,
    }
    create = _http_post(f"http://127.0.0.1:{port}/start", brief)
    instance_id = create["id"]
    assert re.fullmatch(r"[0-9a-f-]{36}", instance_id), (
        f"unexpected instance id format: {instance_id!r}"
    )

    # Step 2: wait until the workflow has run all pre-HITL steps and is
    # parked on contact_review_gate. Local-dev keeps top-level status as
    # "running" while inside waitForEvent — we infer parking via step
    # output stability.
    state = _wait_until_parked_or_complete(
        port, instance_id, min_step_count=len(pre_gate_nodes), timeout=30
    )
    assert state["status"] == "running"  # parked inside waitForEvent
    assert _node_names(state) == pre_gate_nodes, (
        f"expected pre-gate nodes {pre_gate_nodes}, got {_node_names(state)}"
    )

    # Step 3: send first event. Use the actual signal name from the IR
    # so we catch any name drift between adapter and IR.
    contact_gate = next(n for n in bdr_pipeline.nodes if n.id == "contact_review_gate")
    assert contact_gate.signal == "contact_review_approved"
    _http_post(
        f"http://127.0.0.1:{port}/event/{instance_id}/{contact_gate.signal}",
        {"approved_contacts": [{"id": "test"}]},
    )

    # Step 4: wait until the workflow has progressed past the gate and
    # is parked on manual_enrollment_handoff.
    state = _wait_until_parked_or_complete(
        port, instance_id, min_step_count=len(post_gate_nodes), timeout=30
    )
    assert state["status"] == "running"
    assert _node_names(state) == post_gate_nodes, (
        f"expected post-gate nodes {post_gate_nodes}, got {_node_names(state)}"
    )

    # Step 5: send second event.
    handoff_gate = next(n for n in bdr_pipeline.nodes if n.id == "manual_enrollment_handoff")
    assert handoff_gate.signal == "bdr_enrollment_complete"
    handoff_payload = {"enrolled": True, "enrolled_count": 1}
    _http_post(
        f"http://127.0.0.1:{port}/event/{instance_id}/{handoff_gate.signal}",
        handoff_payload,
    )

    # Step 6: workflow completes.
    final = _wait_until_parked_or_complete(
        port, instance_id, min_step_count=len(post_gate_nodes), timeout=30
    )
    assert final["status"] == "complete", (
        f"expected complete, got {final['status']}: {json.dumps(final, indent=2)}"
    )

    # Sanity: the final status payload includes the workflow's return
    # value (the IR's exit_nodes mapped to step results). The workflow
    # returns { "manual_enrollment_handoff": <event payload> } — the
    # second event's payload, not a mocked step result.
    output = final.get("output")
    assert output is not None, f"expected workflow output, got {final}"
    assert "manual_enrollment_handoff" in output, (
        f"expected exit node in output keys: {list(output.keys())}"
    )
    assert output["manual_enrollment_handoff"] == handoff_payload, (
        f"second event payload didn't survive into the workflow return: "
        f"got {output['manual_enrollment_handoff']!r}"
    )

    # ─── Data-flow threading assertions ───
    # The overlay mocks echo the payload they received, so the step
    # outputs prove real data moved between steps inside the actual
    # Cloudflare Workflows runtime.

    # Pipeline input reached both entry nodes intact.
    assert _step_output(final, "target_research")["received"] == {"brief": brief}
    assert _step_output(final, "taxonomy_lookup")["received"] == {"brief": brief}

    # The first gate's event payload flowed into hubspot_upsert via
    # `contacts: contact_review_gate.output.approved_contacts`.
    assert _step_output(final, "hubspot_upsert")["received"] == {"contacts": [{"id": "test"}]}

    # A whole-output binding: create_sales_template received
    # personalize_email's full step result.
    sales_template_received = _step_output(final, "create_sales_template")["received"]
    assert sales_template_received["personalizations"]["node"] == "personalize_email"
    assert sales_template_received["campaign_name"] == brief["drug_brand"]

    # Pipeline input field selection at the end of the chain.
    report_received = _step_output(final, "pre_enrollment_report")["received"]
    assert report_received["campaign_name"] == brief["drug_brand"]
