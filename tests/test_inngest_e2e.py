"""Slow integration tests: emit → compile → run the BDR pipeline on real Inngest.

This is the Inngest analog of ``test_cloudflare_e2e.py`` /
``test_dbos_ts_e2e.py``. Three tiers:

1. ``test_emitted_typescript_compiles`` — real ``npm install`` +
   ``tsc --noEmit`` over the emitted output. Catches type errors and
   SDK surface drift against the actual published packages.
2. ``test_node_modules_contains_expected_packages`` — the emitted
   ``package.json`` resolves against the real npm registry.
3. ``test_workflow_executes_through_hitl_gates`` — runs the emitted app
   against the real Inngest dev server (``inngest-cli dev``, single
   binary, no account), driving a run through both BDR HITL gates:

   - stubs/judges replaced by recorder mocks that log (node, payload)
     to a JSONL file and return canned outputs, so the test asserts
     data-flow threading, not just completion;
   - the emitted ``src/index.ts`` is the driver — no overlay server
     needed (the serve entrypoint is part of the adapter's output);
   - app registration is triggered deterministically with a ``PUT`` to
     the app's serve handler (``--no-poll`` means the dev server's
     single startup sync can race the app boot — verified empirically);
   - events are sent through the dev server's event API
     (``POST /e/dev`` — any key works in dev);
   - run state is read from ``GET /v1/runs/{run_id}`` (**not**
     ``GET /v1/events/{id}/runs``, which reported a stale ``Completed``
     while the run was parked on ``waitForEvent`` during probing — use
     it only to discover the run id);
   - parking is detected via recorded-node-set stability plus run
     status ``Running`` (Inngest, like Cloudflare local dev, keeps the
     status ``Running`` while parked — there is no ``Waiting`` value);
   - the final return value is read from the dev server's GraphQL API
     (``POST /v0/gql``, ``run(runID:){output}``) — the v1 REST
     ``output`` field came back empty during probing; GraphQL returns
     the executor's ``RunComplete`` op JSON with the function's return
     value under ``data``.

Verified against ``inngest`` v4.11.0 + ``inngest-cli`` v1.34.0.

Gated behind ``@pytest.mark.slow``: needs Node/npm and network access
(the inngest-cli postinstall downloads a platform binary). Missing
toolchain → skip with a clear message.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import pytest

from rote.adapters._common import _execution_waves, _to_camel_case
from rote.adapters.inngest import InngestAdapter, gate_event_name, trigger_event_name
from rote.ir import NodeKind, Pipeline
from tests._helpers import FAN_OUT_ELEMENTS, fan_out_element, fan_out_source_keys

REPO_ROOT = Path(__file__).resolve().parent.parent
BDR_PIPELINE_YAML = REPO_ROOT / "examples" / "bdr-outreach" / "expected" / "pipeline.yaml"

pytestmark = pytest.mark.slow

INNGEST_CLI_SPEC = "inngest-cli@^1.34.0"


def _node_available() -> bool:
    return shutil.which("node") is not None and shutil.which("npm") is not None


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _npm_install(cwd: Path, *extra_packages: str) -> None:
    proc = subprocess.run(
        ["npm", "install", "--no-audit", "--no-fund", *extra_packages],
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
def emitted_dir(bdr_pipeline: Pipeline, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Emit the BDR pipeline + run npm install once per module."""
    if not _node_available():
        pytest.skip("Node / npm not available — skipping inngest e2e tests")
    out = tmp_path_factory.mktemp("inngest-e2e")
    InngestAdapter().emit(bdr_pipeline, out)
    _npm_install(out)
    return out


def test_emitted_typescript_compiles(emitted_dir: Path) -> None:
    """Run `tsc --noEmit` over the emitted output. Zero diagnostics expected.

    If this passes, the emitted code typechecks against the real
    inngest v4 definitions, the Anthropic SDK, and Zod.
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
            f"tsc --noEmit reported errors in emitted Inngest code:\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )


def test_node_modules_contains_expected_packages(emitted_dir: Path) -> None:
    node_modules = emitted_dir / "node_modules"
    assert (node_modules / "inngest").exists()
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
# TS e2e.

_MOCK_OUTPUTS: dict[str, dict] = {
    "lead_generation_loop": {"vetted_contacts": [], "discarded_summary": {}},
    "hubspot_upsert": {"upserted": [{"vid": "hs-1"}]},
    "hubspot_create_list": {"list_id": "list-abc123"},
    "exclusion_check_dnc": {"passed": [], "excluded": []},
    "exclusion_check_recent": {"passed": [], "excluded": []},
    "exclusion_check_sequence": {"passed": [], "excluded": []},
    "create_sales_template": {"template_ids": ["t1", "t2"]},
}


def _mock_output(node_id: str, pipeline: Pipeline) -> dict:
    """The canned output for a node, with any fanned list filled in.

    A node feeding a fan_out node MUST return that list non-empty:
    the hardcoded `passed: []` below made BDR's fan dispatch zero
    times, so personalize_email silently never ran and the live
    tests still passed.
    """
    out = dict(_MOCK_OUTPUTS.get(node_id, {"mocked": True, "node": node_id}))
    out.update(fan_out_source_keys(pipeline).get(node_id, {}))
    return out


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
            output=json.dumps(_mock_output(node.id, pipeline)),
        )
        (out_dir / "src" / sub / f"{node.id}.ts").write_text(src, encoding="utf-8")


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
    timeout: float = 120.0,
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


class _DevServer:
    """Typed handle on the dev server the test drives over HTTP."""

    def __init__(self, port: int) -> None:
        self.base = f"http://127.0.0.1:{port}"

    def send_event(self, name: str, data: dict) -> str:
        """POST an event to the dev event API; returns the event id.
        Any event key works against the dev server."""
        out = _http("POST", f"{self.base}/e/dev", {"name": name, "data": data})
        return out["ids"][0]

    def run_id_for_event(self, event_id: str, *, timeout: float = 60.0) -> str:
        """Discover the run a trigger event started. Only the run *id* is
        trusted from this endpoint — its status field reported a stale
        'Completed' while parked during probing."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            runs = _http("GET", f"{self.base}/v1/events/{event_id}/runs")["data"]
            if runs:
                return runs[0]["run_id"]
            time.sleep(0.5)
        raise AssertionError(f"no run appeared for event {event_id}")

    def run_status(self, run_id: str) -> str:
        return _http("GET", f"{self.base}/v1/runs/{run_id}")["data"]["status"]

    def wait_for_status(self, run_id: str, status: str, *, timeout: float = 60.0) -> None:
        deadline = time.time() + timeout
        last = None
        while time.time() < deadline:
            last = self.run_status(run_id)
            if last == status:
                return
            if last in ("Failed", "Cancelled") and status not in ("Failed", "Cancelled"):
                raise AssertionError(f"run {run_id} reached terminal {last!r} awaiting {status!r}")
            time.sleep(0.5)
        raise AssertionError(f"timeout waiting for status {status!r}; last seen: {last!r}")

    def run_output(self, run_id: str) -> dict:
        """The function's return value, via the dev server's GraphQL API.

        The v1 REST ``output`` field is empty in inngest-cli 1.34.0; the
        GraphQL ``run(runID:){output}`` returns the executor's op array —
        the ``RunComplete`` op carries the return value under ``data``.
        """
        query = f"{{ run(runID: {json.dumps(run_id)}) {{ status output }} }}"
        out = _http("POST", f"{self.base}/v0/gql", {"query": query})
        run = out["data"]["run"]
        ops = json.loads(run["output"])
        [complete] = [op for op in ops if op["op"] == "RunComplete"]
        result = complete["data"]
        assert isinstance(result, dict), f"unexpected RunComplete data shape: {result!r}"
        return result


# ───────── Fixtures: emitted app + dev server processes ─────────


def _wait_until_healthy(url: str, proc: subprocess.Popen, log_path: Path, what: str) -> None:
    deadline = time.time() + 90
    while time.time() < deadline:
        if proc.poll() is not None:
            pytest.fail(
                f"{what} exited prematurely (code {proc.returncode}); log:\n{log_path.read_text()}"
            )
        try:
            with urlopen(url, timeout=2) as resp:
                resp.read()
            return
        except (TimeoutError, URLError, ConnectionResetError):
            time.sleep(0.5)
    pytest.fail(f"{what} did not become healthy in 90s; log:\n{log_path.read_text()}")


@pytest.fixture(scope="module")
def inngest_session(
    bdr_pipeline: Pipeline,
    tmp_path_factory: pytest.TempPathFactory,
):  # noqa: ANN201
    """Emit + overlay + compile the app; launch it and the dev server."""
    if not _node_available():
        pytest.skip("Node / npm not available")

    out = tmp_path_factory.mktemp("inngest-live")
    record_path = out / "record.jsonl"
    InngestAdapter().emit(bdr_pipeline, out)
    _write_test_overlay(out, bdr_pipeline)
    # inngest-cli is test-only tooling (its postinstall downloads the
    # platform binary — hence --ignore-scripts=false in case the npm
    # config disables install scripts globally).
    _npm_install(out, "--ignore-scripts=false", INNGEST_CLI_SPEC)
    rebuild = subprocess.run(
        ["npm", "rebuild", "--ignore-scripts=false", "inngest-cli"],
        cwd=out,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if rebuild.returncode != 0:
        pytest.fail(f"npm rebuild inngest-cli failed:\n{rebuild.stdout}\n{rebuild.stderr}")
    cli = out / "node_modules" / ".bin" / "inngest-cli"
    if not cli.exists():
        pytest.fail("inngest-cli binary missing after install + rebuild")

    build = subprocess.run(
        ["npx", "--no-install", "tsc"],
        cwd=out,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if build.returncode != 0:
        pytest.fail(f"tsc build failed:\n{build.stdout}\n{build.stderr}")

    app_port = _find_free_port()
    dev_port = _find_free_port()

    app_log_path = out / "app.log"
    app_log = app_log_path.open("w", encoding="utf-8")
    app_proc = subprocess.Popen(
        ["node", "dist/index.js"],
        cwd=out,
        stdout=app_log,
        stderr=subprocess.STDOUT,
        env={
            **os.environ,
            # INNGEST_DEV takes the dev server URL so the SDK registers
            # against the test's port instead of the default 8288.
            "INNGEST_DEV": f"http://127.0.0.1:{dev_port}",
            "PORT": str(app_port),
            "E2E_RECORD_PATH": str(record_path),
            # The mocks never call the vendor SDK, but the emitted judge
            # steps resolve the key eagerly via requireEnv.
            "ANTHROPIC_API_KEY": "unused-by-mocks",
        },
    )

    dev_log_path = out / "dev-server.log"
    dev_log = dev_log_path.open("w", encoding="utf-8")
    dev_proc = subprocess.Popen(
        [
            str(cli),
            "dev",
            "-u",
            f"http://127.0.0.1:{app_port}",
            "--no-discovery",
            "--no-poll",
            "-p",
            str(dev_port),
        ],
        cwd=out,
        stdout=dev_log,
        stderr=subprocess.STDOUT,
    )

    try:
        _wait_until_healthy(
            f"http://127.0.0.1:{dev_port}/health", dev_proc, dev_log_path, "inngest dev server"
        )
        # The app answers GET on any path once listening (serve handler).
        _wait_until_healthy(f"http://127.0.0.1:{app_port}/", app_proc, app_log_path, "emitted app")

        # Deterministic registration: with --no-poll the dev server's
        # single startup sync can race the app boot (observed empirically
        # — the app stays unregistered forever). A PUT to the SDK's serve
        # handler pushes registration explicitly; retry until acknowledged.
        deadline = time.time() + 60
        registered = False
        while time.time() < deadline:
            try:
                ack = _http("PUT", f"http://127.0.0.1:{app_port}/")
                if "registered" in str(ack.get("message", "")).lower():
                    registered = True
                    break
            except (TimeoutError, URLError, ConnectionResetError):
                pass
            time.sleep(1)
        if not registered:
            pytest.fail(
                f"app never registered with the dev server;\n"
                f"app log:\n{app_log_path.read_text()}\n"
                f"dev log:\n{dev_log_path.read_text()}"
            )

        yield _DevServer(dev_port), record_path
    finally:
        for proc in (app_proc, dev_proc):
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        app_log.close()
        dev_log.close()


# ───────── The end-to-end test ─────────


def test_workflow_executes_through_hitl_gates(
    inngest_session,  # noqa: ANN001
    bdr_pipeline: Pipeline,
) -> None:
    dev, record_path = inngest_session
    pre_gate, post_gate = _pre_and_post_gate_nodes(bdr_pipeline)
    assert pre_gate == {"target_research", "taxonomy_lookup", "lead_generation_loop"}

    # A complete brief matching the pipeline's input contract — the
    # emitted function threads real payloads, so every referenced field
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
    event_id = dev.send_event(trigger_event_name(bdr_pipeline), brief)
    run_id = dev.run_id_for_event(event_id)

    # ── Park at gate 1 ──
    _wait_for_node_set(record_path, pre_gate)
    # Stability window: nothing beyond the pre-gate set may run while
    # the run is parked on waitForEvent. (Inngest keeps run status
    # "Running" while parked — the recorded node set is the
    # authoritative parking signal; status is corroboration.)
    time.sleep(2.0)
    recorded_names = {r["node"] for r in _recorded(record_path)}
    assert recorded_names == pre_gate, (
        f"run should be parked at contact_review_gate after {sorted(pre_gate)}; "
        f"recorded: {sorted(recorded_names)}"
    )
    assert dev.run_status(run_id) == "Running"

    # ── Wave order: the loop consumed both wave-1 results before starting ──
    order = [r["node"] for r in _recorded(record_path)]
    assert order.index("lead_generation_loop") > order.index("target_research")
    assert order.index("lead_generation_loop") > order.index("taxonomy_lookup")

    # ── Resume gate 1 with the namespaced IR signal (catches name drift) ──
    contact_gate = bdr_pipeline.node_by_id("contact_review_gate")
    assert gate_event_name(bdr_pipeline, contact_gate) == "bdr-campaign/contact_review_approved"
    dev.send_event(
        gate_event_name(bdr_pipeline, contact_gate),
        {"approved_contacts": [{"id": "c1"}]},
    )

    # ── Park at gate 2 ──
    _wait_for_node_set(record_path, pre_gate | post_gate)
    time.sleep(2.0)
    recorded_names = {r["node"] for r in _recorded(record_path)}
    assert recorded_names == pre_gate | post_gate
    assert dev.run_status(run_id) == "Running"

    # ── Resume gate 2 and complete ──
    handoff_gate = bdr_pipeline.node_by_id("manual_enrollment_handoff")
    assert gate_event_name(bdr_pipeline, handoff_gate) == "bdr-campaign/bdr_enrollment_complete"
    handoff_payload = {"enrolled": True, "enrolled_count": 1}
    dev.send_event(gate_event_name(bdr_pipeline, handoff_gate), handoff_payload)

    dev.wait_for_status(run_id, "Completed")
    result = dev.run_output(run_id)
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
    assert loop_payload["taxonomy"] == _mock_output("taxonomy_lookup", bdr_pipeline)
    assert loop_payload["target_quota"] == brief["target_quota"]

    # The first HITL gate's resume payload flowed into hubspot_upsert via
    # `contacts: contact_review_gate.output.approved_contacts`.
    assert payloads["hubspot_upsert"] == {"contacts": [{"id": "c1"}]}

    # hubspot_upsert's mocked result flowed into the DNC check via
    # `contacts: hubspot_upsert.output.upserted`.
    assert payloads["exclusion_check_dnc"] == {"contacts": [{"vid": "hs-1"}]}

    # ── fan_out: personalize_email ran once per surviving contact ──
    # `payloads` is keyed by node so it keeps only the last record; count
    # the raw records instead, and check each invocation got exactly ONE
    # element rather than the whole list.
    fan_records = [r for r in _recorded(record_path) if r["node"] == "personalize_email"]
    assert len(fan_records) == FAN_OUT_ELEMENTS, (
        f"fan_out node personalize_email must run once per element of "
        f"exclusion_check_sequence.passed; ran {len(fan_records)} time(s)"
    )
    assert sorted(
        (r["payload"]["contact"] for r in fan_records), key=lambda c: c["fanElement"]
    ) == [fan_out_element(i) for i in range(FAN_OUT_ELEMENTS)], (
        "each fan_out invocation must receive one element, not the batch"
    )

    # The report node received a fan-in of upstream results:
    # pipeline input field + two different upstream nodes.
    report_payload = payloads["pre_enrollment_report"]
    assert report_payload["campaign_name"] == brief["drug_brand"]
    # exclusion_check_sequence.passed is the list personalize_email
    # fans over, so the mock returns it non-empty (see _mock_output).
    assert report_payload["passed_contacts"] == [
        fan_out_element(i) for i in range(FAN_OUT_ELEMENTS)
    ]
    assert report_payload["template_ids"] == ["t1", "t2"]

    # Every non-HITL top-level node ran exactly once — memoized steps
    # are not re-executed across the executor's re-invocations of the
    # handler (the durable-execution contract). A fan_out node is the
    # one exception: it legitimately runs once per element, and its
    # per-element step ids are what keep those from memoizing onto each
    # other (a constant id would collapse them to a single execution).
    fan_out_ids = {n.id for n in bdr_pipeline.nodes if n.fan_out}
    node_counts: dict[str, int] = {}
    for r in _recorded(record_path):
        node_counts[r["node"]] = node_counts.get(r["node"], 0) + 1
    expected_counts = {nid: (FAN_OUT_ELEMENTS if nid in fan_out_ids else 1) for nid in node_counts}
    assert node_counts == expected_counts, (
        f"steps must execute exactly once (memoization), or once per element "
        f"for fan_out nodes; counts: {node_counts}"
    )
