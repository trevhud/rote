"""Slow integration tests: emit → compile → run the BDR workflow on real DBOS TS.

This is the DBOS TypeScript analog of ``test_dbos_e2e.py`` (semantics)
and ``test_cloudflare_e2e.py`` (toolchain choreography). Three tiers:

1. ``test_emitted_typescript_compiles`` — real ``npm install`` +
   ``tsc --noEmit`` over the emitted output. Catches type errors and
   SDK surface drift against the actual published packages.
2. ``test_node_modules_contains_expected_packages`` — the emitted
   ``package.json`` resolves against the real npm registry.
3. ``test_workflow_executes_through_hitl_gates`` — runs the emitted app
   on the real DBOS TS runtime against a Docker Postgres (the TS SDK is
   Postgres-only; unlike DBOS Python there is no SQLite mode), driving
   a workflow through both BDR HITL gates:

   - stubs/judges replaced by recorder mocks that log (node, payload)
     to a JSONL file and return canned outputs, so the test asserts
     data-flow threading, not just completion;
   - a small HTTP driver (overlaid ``src/e2e.ts``) exposes
     start/send/status/steps/result so the Python test can drive the
     Node process;
   - parking is detected via recorded-step-set stability plus status
     ``PENDING`` (DBOS reports PENDING while parked on ``DBOS.recv`` —
     same behavior as DBOS Python);
   - durability is verified via the step log: one ``DBOS.recv``
     checkpoint per gate in the system database. (Each recv also logs
     a ``DBOS.sleep`` timer checkpoint — count recv entries, don't
     compare whole lists.)

Gated behind ``@pytest.mark.slow``: needs Node/npm, and tier 3 needs a
running Docker daemon (it boots a throwaway ``postgres:16-alpine``).
Missing toolchain → skip with a clear message.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import time
import uuid
from pathlib import Path

import pytest

from rote.adapters._common import _execution_waves, _to_camel_case
from rote.adapters.dbos_ts import DbosTsAdapter
from rote.ir import NodeKind, Pipeline, load_pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
BDR_PIPELINE_YAML = REPO_ROOT / "examples" / "bdr-outreach" / "expected" / "pipeline.yaml"

pytestmark = pytest.mark.slow


def _node_available() -> bool:
    return shutil.which("node") is not None and shutil.which("npm") is not None


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    proc = subprocess.run(["docker", "info"], capture_output=True, timeout=30)
    return proc.returncode == 0


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _npm_install(cwd: Path) -> None:
    proc = subprocess.run(
        ["npm", "install", "--no-audit", "--no-fund"],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "npm_config_progress": "false"},
    )
    if proc.returncode != 0:
        pytest.fail(f"npm install failed in {cwd}:\n{proc.stdout}\n{proc.stderr}")


# ───────── Tier 1 + 2: compile against the real packages ─────────


@pytest.fixture(scope="module")
def bdr_pipeline() -> Pipeline:
    return load_pipeline(BDR_PIPELINE_YAML)


@pytest.fixture(scope="module")
def emitted_dir(bdr_pipeline: Pipeline, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Emit the BDR pipeline + run npm install once per module."""
    if not _node_available():
        pytest.skip("Node / npm not available — skipping dbos-ts e2e tests")
    out = tmp_path_factory.mktemp("dbos-ts-e2e")
    DbosTsAdapter().emit(bdr_pipeline, out)
    _npm_install(out)
    return out


def test_emitted_typescript_compiles(emitted_dir: Path) -> None:
    """Run `tsc --noEmit` over the emitted output. Zero diagnostics expected.

    If this passes, the emitted code typechecks against the real
    @dbos-inc/dbos-sdk v4 definitions, the Anthropic SDK, and Zod.
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
            f"tsc --noEmit reported errors in emitted DBOS TS code:\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )


def test_node_modules_contains_expected_packages(emitted_dir: Path) -> None:
    node_modules = emitted_dir / "node_modules"
    assert (node_modules / "@dbos-inc" / "dbos-sdk").exists()
    assert (node_modules / "@anthropic-ai" / "sdk").exists()
    assert (node_modules / "zod").exists()
    assert (node_modules / "typescript").exists()


# ───────── Tier 3: live workflow execution ─────────
#
# The emitted extracted/ stubs throw and the signatures/ judges call
# real LLM APIs. For an integration test that exercises orchestration
# (not I/O), each is replaced by a mock that appends its node id *and
# the payload it received* to a JSONL file and returns a canned dict
# shaped like the node's declared output — same technique as the DBOS
# Python e2e.

_MOCK_OUTPUTS: dict[str, dict] = {
    "lead_generation_loop": {"vetted_contacts": [], "discarded_summary": {}},
    "hubspot_upsert": {"upserted": [{"vid": "hs-1"}]},
    "hubspot_create_list": {"list_id": "list-abc123"},
    "exclusion_check_dnc": {"passed": [], "excluded": []},
    "exclusion_check_recent": {"passed": [], "excluded": []},
    "exclusion_check_sequence": {"passed": [], "excluded": []},
    "create_sales_template": {"template_ids": ["t1", "t2"]},
}


def _mock_output(node_id: str) -> dict:
    return _MOCK_OUTPUTS.get(node_id, {"mocked": True, "node": node_id})


_MOCK_MODULE = """\
import {{ appendFileSync }} from "node:fs";

export async function {fn}(input: unknown{extra}): Promise<Record<string, unknown>> {{
    appendFileSync(
        process.env.E2E_RECORD_PATH!,
        JSON.stringify({{ node: {node_id}, payload: input ?? null }}) + "\\n",
    );
    return {output};
}}
"""

# HTTP driver overlaid as src/e2e.ts: the Python test drives the Node
# process over localhost. DBOS.send here is in-process (same bar as the
# DBOS Python e2e); cross-process delivery via DBOSClient is documented
# in the emitted README.
_DRIVER_TS = """\
import * as http from "node:http";
import { DBOS } from "@dbos-inc/dbos-sdk";
import { runPipeline } from "./main";

async function serve(): Promise<void> {
    DBOS.setConfig({
        name: "bdr-campaign",
        systemDatabaseUrl: process.env.DBOS_SYSTEM_DATABASE_URL,
        runAdminServer: false,
    });
    await DBOS.launch();
    const server = http.createServer((req, res) => {
        const chunks: Buffer[] = [];
        req.on("data", (c) => chunks.push(c));
        req.on("end", () => {
            void (async () => {
                const body: unknown = chunks.length
                    ? JSON.parse(Buffer.concat(chunks).toString())
                    : {};
                const url = req.url ?? "";
                const parts = url.split("/").filter((p) => p.length > 0);
                let out: unknown;
                if (req.method === "POST" && url === "/start") {
                    const handle = await DBOS.startWorkflow(runPipeline)(
                        body as Record<string, unknown>,
                    );
                    out = { id: handle.workflowID };
                } else if (req.method === "POST" && parts[0] === "send") {
                    await DBOS.send(parts[1], body, parts[2]);
                    out = { ok: true };
                } else if (parts[0] === "status") {
                    const status = await DBOS.retrieveWorkflow(parts[1]).getStatus();
                    out = { status: status?.status ?? null };
                } else if (parts[0] === "steps") {
                    const steps = (await DBOS.listWorkflowSteps(parts[1])) ?? [];
                    out = { steps: steps.map((s) => s.name) };
                } else if (parts[0] === "result") {
                    out = { result: await DBOS.retrieveWorkflow(parts[1]).getResult() };
                } else {
                    res.writeHead(404);
                    res.end();
                    return;
                }
                res.writeHead(200, { "content-type": "application/json" });
                res.end(JSON.stringify(out));
            })().catch((err: unknown) => {
                res.writeHead(500, { "content-type": "application/json" });
                res.end(JSON.stringify({ error: String(err) }));
            });
        });
    });
    server.listen(Number(process.env.E2E_PORT), "127.0.0.1", () => {
        console.log("e2e driver ready");
    });
}

serve().catch((err) => {
    console.error(err);
    process.exit(1);
});
"""


def _write_test_overlay(out_dir: Path, pipeline: Pipeline) -> None:
    for node in pipeline.nodes:
        if node.kind is NodeKind.HITL_GATE:
            continue
        sub = "signatures" if node.kind is NodeKind.LLM_JUDGE else "extracted"
        extra = ", _env?: unknown" if node.kind is NodeKind.LLM_JUDGE else ""
        src = _MOCK_MODULE.format(
            fn=_to_camel_case(node.id),
            extra=extra,
            node_id=json.dumps(node.id),
            output=json.dumps(_mock_output(node.id)),
        )
        (out_dir / "src" / sub / f"{node.id}.ts").write_text(src, encoding="utf-8")
    (out_dir / "src" / "e2e.ts").write_text(_DRIVER_TS, encoding="utf-8")


def _pre_and_post_gate_nodes(pipeline: Pipeline) -> tuple[set[str], set[str]]:
    """Split the top-level non-HITL nodes into before-first-gate and the
    rest, derived from the execution waves (robust to pipeline edits)."""
    pre: set[str] = set()
    post: set[str] = set()
    seen_gate = False
    for wave in _execution_waves(pipeline):
        for node in wave:
            if node.kind is NodeKind.HITL_GATE:
                seen_gate = True
            elif seen_gate:
                post.add(node.id)
            else:
                pre.add(node.id)
    return pre, post


# ───────── HTTP + polling helpers ─────────


def _http(method: str, url: str, body: dict | None = None, timeout: int = 30) -> dict:
    from urllib.request import Request, urlopen

    data = json.dumps(body).encode() if body is not None else None
    req = Request(url, data=data, headers={"Content-Type": "application/json"}, method=method)
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _recorded(record_path: Path) -> list[dict]:
    if not record_path.exists():
        return []
    return [json.loads(line) for line in record_path.read_text().splitlines()]


def _wait_for_node_set(
    record_path: Path,
    expected: set[str],
    *,
    timeout: float = 90.0,
    poll: float = 0.25,
) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if {r["node"] for r in _recorded(record_path)} >= expected:
            return
        time.sleep(poll)
    raise AssertionError(
        f"timeout waiting for nodes {sorted(expected)}; "
        f"recorded so far: {sorted({r['node'] for r in _recorded(record_path)})}"
    )


def _wait_for_status(port: int, wf_id: str, status: str, *, timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = _http("GET", f"http://127.0.0.1:{port}/status/{wf_id}")["status"]
        if last == status:
            return
        time.sleep(0.5)
    raise AssertionError(f"timeout waiting for status {status!r}; last seen: {last!r}")


# ───────── Fixtures: Postgres container + Node driver ─────────


@pytest.fixture(scope="module")
def postgres_url(tmp_path_factory: pytest.TempPathFactory):  # noqa: ANN201
    """Boot a throwaway Docker Postgres; yield a system database URL."""
    if not _docker_available():
        pytest.skip("Docker daemon not available — skipping dbos-ts live execution test")
    name = f"rote-dbos-ts-e2e-{uuid.uuid4().hex[:8]}"
    port = _find_free_port()
    run = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--rm",
            "--name",
            name,
            "-p",
            f"{port}:5432",
            "-e",
            "POSTGRES_PASSWORD=dbos",
            "postgres:16-alpine",
        ],
        capture_output=True,
        text=True,
        timeout=300,  # includes a possible first-time image pull
    )
    if run.returncode != 0:
        pytest.fail(f"docker run failed: {run.stdout}\n{run.stderr}")
    try:
        deadline = time.time() + 60
        while time.time() < deadline:
            ready = subprocess.run(
                ["docker", "exec", name, "pg_isready", "-U", "postgres"],
                capture_output=True,
                timeout=10,
            )
            if ready.returncode == 0:
                break
            time.sleep(1)
        else:
            pytest.fail("Postgres container did not become ready in 60s")
        yield f"postgresql://postgres:dbos@localhost:{port}/bdr_campaign_dbos_sys"
    finally:
        subprocess.run(["docker", "stop", name], capture_output=True, timeout=60)


@pytest.fixture(scope="module")
def dbos_ts_session(
    bdr_pipeline: Pipeline,
    postgres_url: str,
    tmp_path_factory: pytest.TempPathFactory,
):  # noqa: ANN201
    """Emit + overlay + compile the app, launch the Node driver process."""
    if not _node_available():
        pytest.skip("Node / npm not available")

    out = tmp_path_factory.mktemp("dbos-ts-live")
    record_path = out / "record.jsonl"
    DbosTsAdapter().emit(bdr_pipeline, out)
    _write_test_overlay(out, bdr_pipeline)
    _npm_install(out)

    build = subprocess.run(
        ["npx", "--no-install", "tsc"],
        cwd=out,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if build.returncode != 0:
        pytest.fail(f"tsc build failed:\n{build.stdout}\n{build.stderr}")

    port = _find_free_port()
    log_path = out / "driver.log"
    log = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        ["node", "dist/e2e.js"],
        cwd=out,
        stdout=log,
        stderr=subprocess.STDOUT,
        env={
            **os.environ,
            "DBOS_SYSTEM_DATABASE_URL": postgres_url,
            "E2E_RECORD_PATH": str(record_path),
            "E2E_PORT": str(port),
            # The mocks never call the vendor SDK, but the emitted step
            # wrappers resolve the key eagerly via requireEnv.
            "ANTHROPIC_API_KEY": "unused-by-mocks",
        },
    )

    # Wait for the driver to answer HTTP (launch runs system DB
    # migrations on first boot — allow generous headroom).
    ready_deadline = time.time() + 60
    while time.time() < ready_deadline:
        if proc.poll() is not None:
            log.close()
            pytest.fail(
                f"driver exited prematurely (code {proc.returncode}); log:\n{log_path.read_text()}"
            )
        try:
            _http("GET", f"http://127.0.0.1:{port}/status/nonexistent", timeout=2)
            break
        except Exception:
            time.sleep(0.5)
    else:
        proc.terminate()
        log.close()
        pytest.fail(f"driver did not start in 60s; log:\n{log_path.read_text()}")

    try:
        yield port, record_path
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        log.close()


# ───────── The end-to-end test ─────────


def test_workflow_executes_through_hitl_gates(
    dbos_ts_session,  # noqa: ANN001
    bdr_pipeline: Pipeline,
) -> None:
    port, record_path = dbos_ts_session
    base = f"http://127.0.0.1:{port}"
    pre_gate, post_gate = _pre_and_post_gate_nodes(bdr_pipeline)
    assert pre_gate == {"target_research", "taxonomy_lookup", "lead_generation_loop"}

    # A complete brief matching the pipeline's input contract — the
    # emitted workflow threads real payloads, so every referenced field
    # must exist. Values reuse the fictionalized examples from the IR.
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
    wf_id = _http("POST", f"{base}/start", brief)["id"]

    # ── Park at gate 1 ──
    _wait_for_node_set(record_path, pre_gate)
    # Stability window: nothing beyond the pre-gate set may run while
    # the workflow is parked on DBOS.recv. (DBOS reports PENDING while
    # parked — the recorded step set is the authoritative parking
    # signal; status is corroboration. Same behavior as DBOS Python.)
    time.sleep(1.5)
    recorded_names = {r["node"] for r in _recorded(record_path)}
    assert recorded_names == pre_gate, (
        f"workflow should be parked at contact_review_gate after {sorted(pre_gate)}; "
        f"recorded: {sorted(recorded_names)}"
    )
    assert _http("GET", f"{base}/status/{wf_id}")["status"] == "PENDING"

    # ── Wave order: the loop consumed both wave-1 results before starting ──
    order = [r["node"] for r in _recorded(record_path)]
    assert order.index("lead_generation_loop") > order.index("target_research")
    assert order.index("lead_generation_loop") > order.index("taxonomy_lookup")

    # ── Resume gate 1 with the IR's signal name (catches name drift) ──
    contact_gate = bdr_pipeline.node_by_id("contact_review_gate")
    assert contact_gate.signal == "contact_review_approved"
    _http(
        "POST",
        f"{base}/send/{wf_id}/{contact_gate.signal}",
        {"approved_contacts": [{"id": "c1"}]},
    )

    # ── Park at gate 2 ──
    _wait_for_node_set(record_path, pre_gate | post_gate)
    time.sleep(1.5)
    recorded_names = {r["node"] for r in _recorded(record_path)}
    assert recorded_names == pre_gate | post_gate
    assert _http("GET", f"{base}/status/{wf_id}")["status"] == "PENDING"

    # ── Resume gate 2 and complete ──
    handoff_gate = bdr_pipeline.node_by_id("manual_enrollment_handoff")
    assert handoff_gate.signal == "bdr_enrollment_complete"
    handoff_payload = {"enrolled": True, "enrolled_count": 1}
    _http("POST", f"{base}/send/{wf_id}/{handoff_gate.signal}", handoff_payload)

    _wait_for_status(port, wf_id, "SUCCESS")
    result = _http("GET", f"{base}/result/{wf_id}")["result"]
    assert result == {"manual_enrollment_handoff": handoff_payload}, (
        "the second gate's payload must flow through to the exit-node result"
    )

    # ── Data-flow threading: real payloads reached the steps ──
    payloads = {r["node"]: r["payload"] for r in _recorded(record_path)}

    # The pipeline input reached the entry node intact.
    assert payloads["target_research"] == {"brief": brief}

    # The fan-in loop consumed both wave-1 results plus an input field.
    loop_payload = payloads["lead_generation_loop"]
    assert loop_payload["brief"] == brief
    assert loop_payload["taxonomy"] == _mock_output("taxonomy_lookup")
    assert loop_payload["target_quota"] == brief["target_quota"]

    # The first HITL gate's resume payload flowed into hubspot_upsert via
    # `contacts: contact_review_gate.output.approved_contacts`.
    assert payloads["hubspot_upsert"] == {"contacts": [{"id": "c1"}]}

    # hubspot_upsert's mocked result flowed into the DNC check via
    # `contacts: hubspot_upsert.output.upserted`.
    assert payloads["exclusion_check_dnc"] == {"contacts": [{"vid": "hs-1"}]}

    # The report node received a fan-in of upstream results:
    # pipeline input field + two different upstream nodes.
    report_payload = payloads["pre_enrollment_report"]
    assert report_payload["campaign_name"] == brief["drug_brand"]
    assert report_payload["passed_contacts"] == []
    assert report_payload["template_ids"] == ["t1", "t2"]

    # ── Durability: the gates were checkpointed recv steps in the system
    # database, not in-memory waits. A crashed process would recover the
    # workflow and resume waiting at the same recv. (Each recv also logs
    # a DBOS.sleep timer checkpoint — count recv entries only.)
    steps = _http("GET", f"{base}/steps/{wf_id}")["steps"]
    assert steps.count("DBOS.recv") == 2, (
        f"expected 2 durable recv checkpoints, got step log: {steps}"
    )
    # Every non-HITL top-level node appears as a checkpointed step.
    assert set(steps) >= pre_gate | post_gate, steps
