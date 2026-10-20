"""A read-only live check: call the real vendor once and compare its shape with the spec.

``rote run <compile dir> --live-check --input <json>`` runs each entry
vendor read of an ``--backend api`` deliverable (an external_call node whose
inputs all come from the pipeline input or its constants) once, for real,
with the user's own credentials from the environment. The intercept
(:mod:`rote.vendor_specs.intercept`, live mode) lets only GET and HEAD reach
the vendor hosts in play and records each response. Each response is then
compared with what rote believed:

* the operation's response schema (the vendor's spec, or the compile's docs
  spec): a field the live API returned that the schema lacks, and a value
  of the wrong type;
* the stored fixtures for that operation (:mod:`.fixture_store`): fields
  the fixtures have that the live response did not return, and the
  reverse.

The report holds field paths and types, never values, and the recorded
responses are deleted when the check ends: nothing the user's account
returned is kept.

The deliverable is refused before anything runs when any request it makes
is not provably a read (a non-GET operation, or a request whose operation
cannot be resolved), since the check could then not promise read-only.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rote.ir import NodeKind, Pipeline, load_pipeline, parse_input_ref
from rote.vendor_specs import docs, google
from rote.vendor_specs.check import best_operation, check_deliverable
from rote.vendor_specs.deps import install_vendor_deps
from rote.vendor_specs.extract import extract
from rote.vendor_specs.fixture_store import FixtureStore
from rote.vendor_specs.model import ApiSpec, Operation, host_of, path_segments
from rote.vendor_specs.store import SpecStore

INTERCEPT_DIR = Path(__file__).resolve().parent / "intercept"
LIVE_TIMEOUT_SECONDS = 120.0

_DRIVER = """
import importlib, json, sys
impl, kwargs = json.loads(sys.argv[1])
module, symbol = impl.split(":")
fn = getattr(importlib.import_module(module.removesuffix(".py").replace("/", ".")), symbol)
try:
    fn(**kwargs)
    print(json.dumps({"error": None}))
except Exception as e:
    print(json.dumps({"error": f"{type(e).__name__}: {e}"[:300]}))
"""


class LiveCheckRefused(RuntimeError):
    """The deliverable cannot be live-checked without risking a write."""


@dataclass
class ResponseCheck:
    """One live response, compared with the spec and the stored fixtures."""

    node_id: str
    method: str
    host: str
    path: str
    status: int
    operation: str | None = None
    spec: str | None = None
    #: Field paths the code reads off this response that the live response did
    #: not have: the pipeline would break on them. The one that matters most.
    reads_missing: list[str] = field(default_factory=list)
    #: Field paths the live response had that the schema lacks, and wrong types.
    schema_drift: list[str] = field(default_factory=list)
    #: Field paths the schema has that the live response did not return.
    spec_only: list[str] = field(default_factory=list)
    #: Field paths the stored fixtures have that the live response did not.
    fixture_only: list[str] = field(default_factory=list)
    #: Field paths the live response had that no stored fixture has.
    live_only: list[str] = field(default_factory=list)


@dataclass
class LiveReport:
    checked: list[ResponseCheck] = field(default_factory=list)
    #: Nodes that were not called, with why (inputs from upstream nodes).
    skipped: dict[str, str] = field(default_factory=dict)
    #: Nodes whose call raised, with the error's first line.
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def drifted(self) -> bool:
        return any(c.schema_drift or c.reads_missing for c in self.checked)

    def as_dict(self) -> dict[str, Any]:
        return {
            "checked": [c.__dict__ for c in self.checked],
            "skipped": self.skipped,
            "errors": self.errors,
            "drifted": self.drifted,
        }


def entry_reads(pipeline: Pipeline) -> tuple[list[Any], dict[str, str]]:
    """The external_call nodes whose inputs all come from the pipeline input."""
    loop_body = {n for node in pipeline.nodes for n in (node.loop_body or [])}
    runnable: list[Any] = []
    skipped: dict[str, str] = {}
    for node in pipeline.nodes:
        if node.kind is not NodeKind.EXTERNAL_CALL or not node.impl or node.id in loop_body:
            continue
        upstream = sorted(
            {
                ref.node_id
                for ref in (parse_input_ref(r) for r in (node.inputs or {}).values())
                if ref.node_id is not None
            }
        )
        if upstream:
            skipped[node.id] = f"its inputs come from {', '.join(upstream)}"
        else:
            runnable.append(node)
    return runnable, skipped


def _kwargs(node: Any, payload: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    for param, ref in (node.inputs or {}).items():
        parsed = parse_input_ref(ref)
        kwargs[param] = payload if parsed.field is None else payload.get(parsed.field)
    kwargs.update(node.constants or {})
    return kwargs


def live_check(
    compiled_dir: Path,
    payload: dict[str, Any],
    specs: SpecStore | None = None,
    fixtures: FixtureStore | None = None,
    env: dict[str, str] | None = None,
) -> LiveReport:
    """Run each entry vendor read once against the real API; compare shapes."""
    specs = specs or SpecStore()
    fixtures = fixtures or FixtureStore()
    pipeline = load_pipeline(compiled_dir / "pipeline.yaml")
    checks = check_deliverable(pipeline, compiled_dir, specs)
    unsafe = [
        f"{c.path}:{c.line} ({c.operation or c.note or 'unresolved'})"
        for c in checks
        if c.status != "ok" or c.operation is None
    ]
    if unsafe:
        raise LiveCheckRefused(
            "every vendor request must resolve to a spec operation that passes the spec "
            f"check before a live call: {'; '.join(unsafe)}"
        )
    written = [s for _, s in docs.load(compiled_dir)]
    # The intercept compares hostnames, without a port.
    hosts = sorted({h.split(":")[0] for c in checks for h in _hosts_of(c, specs, written)})
    reads = {op.id: op for spec in _specs_of(checks, specs, written) for op in spec.operations}
    writes = [c for c in checks if c.operation and reads[c.operation].method not in ("GET", "HEAD")]
    if writes:
        raise LiveCheckRefused(
            "the live check is read-only, and this deliverable writes: "
            + ", ".join(f"{c.operation} ({c.path}:{c.line})" for c in writes)
        )
    runnable, skipped = entry_reads(pipeline)
    report = LiveReport(skipped=skipped)
    deps = install_vendor_deps(compiled_dir)
    path = [str(INTERCEPT_DIR), str(compiled_dir), *([str(deps)] if deps else [])]
    with tempfile.TemporaryDirectory(prefix="rote-live-") as tmp:
        for node in runnable:
            record = Path(tmp) / f"{node.id}.jsonl"
            child = {
                **(env if env is not None else os.environ),
                "PYTHONPATH": os.pathsep.join(path),
                "ROTE_HTTP_LIVE_HOSTS": ",".join(hosts),
                "ROTE_HTTP_LIVE_RECORD": str(record),
            }
            proc = subprocess.run(
                [sys.executable, "-c", _DRIVER, json.dumps([node.impl, _kwargs(node, payload)])],
                cwd=compiled_dir,
                env=child,
                capture_output=True,
                text=True,
                timeout=LIVE_TIMEOUT_SECONDS,
            )
            outcome = _last_json(proc.stdout) or {"error": proc.stderr.strip()[-300:]}
            if outcome.get("error"):
                report.errors[node.id] = str(outcome["error"])
            code_reads = _reads_by_operation(
                compiled_dir / node.impl.partition(":")[0], specs, written
            )
            for line in record.read_text(encoding="utf-8").splitlines() if record.is_file() else []:
                report.checked.append(
                    _compare(node.id, json.loads(line), specs, written, fixtures, code_reads)
                )
    return report


def _hosts_of(check: Any, specs: SpecStore, written: list[ApiSpec]) -> set[str]:
    return {
        h
        for spec in _specs_of([check], specs, written)
        for op in spec.operations
        if (h := host_of(op.url_template))
    }


def _specs_of(checks: list[Any], specs: SpecStore, written: list[ApiSpec]) -> list[ApiSpec]:
    out: dict[str, ApiSpec] = {}
    for c in checks:
        key = (c.spec or "").split(" ")[0]
        if not key:
            continue
        vendor, api, version = key.split("/")
        found = next((s for s in written if s.key == key), None)
        if found is None and vendor == "google":
            found = specs.google(api, version)
        if found is not None:
            out[key] = found
    return list(out.values())


def _spec_for_url(url: str, specs: SpecStore, written: list[ApiSpec]) -> ApiSpec | None:
    named = google.api_for_url(url)
    if named is not None:
        version = specs.resolve_google_url(named)
        return specs.google(named.api, version) if version is not None else None
    host = host_of(url)
    return next(
        (s for s in written if any(host_of(o.url_template) == host for o in s.operations)), None
    )


def _reads_by_operation(
    module: Path, specs: SpecStore, written: list[ApiSpec]
) -> dict[str, set[str]]:
    """Operation id -> the response field paths the module's code reads."""
    out: dict[str, set[str]] = {}
    for site in extract(module.read_text(encoding="utf-8")):
        op_id = site.google_method_id
        if site.kind == "http" and site.url and site.method:
            spec = _spec_for_url(site.url, specs, written)
            op = _operation(spec, site.method, site.url) if spec else None
            op_id = op.id if op else None
        if op_id:
            out.setdefault(op_id, set()).update(_dotted(p) for p, _ in site.response_reads)
    return out


def _dotted(path: tuple[str, ...]) -> str:
    return "".join("[]" if k == "[]" else f".{k}" for k in path).lstrip(".")


def _schema_paths(
    spec: ApiSpec, schema: dict[str, Any] | None, at: str = "", depth: int = 0
) -> set[str]:
    node = spec.resolve(schema)
    if node is None or depth > 8:
        return set()
    out: set[str] = set()
    for key, sub in (node.get("properties") or {}).items():
        here = f"{at}.{key}" if at else key
        out.add(here)
        out |= _schema_paths(spec, sub, here, depth + 1)
    if isinstance(node.get("items"), dict):
        out |= _schema_paths(spec, node["items"], f"{at}[]", depth + 1)
    return out


def _compare(
    node_id: str,
    rec: dict[str, Any],
    specs: SpecStore,
    written: list[ApiSpec],
    fixtures: FixtureStore,
    reads: dict[str, set[str]] | None = None,
) -> ResponseCheck:
    url = str(rec["url"])
    check = ResponseCheck(
        node_id=node_id,
        method=str(rec["method"]),
        host=host_of(url) or "",
        path="/" + "/".join(path_segments(url)),
        status=int(rec["status"]),
    )
    spec = _spec_for_url(url, specs, written)
    op = _operation(spec, check.method, url) if spec else None
    if spec is None or op is None:
        return check
    check.operation, check.spec = op.id, spec.key
    if check.status >= 400 or "json" not in str(rec.get("content_type", "")):
        return check
    try:
        body = json.loads(rec["body"])
    except json.JSONDecodeError:
        check.schema_drift.append("$: the response is not JSON")
        return check
    check.schema_drift = _drift(spec, op.response_schema, body)
    live = _paths(body)
    check.reads_missing = sorted(p for p in (reads or {}).get(op.id, set()) if p not in live)
    if spec.source == "docs":
        # A docs spec lists what the code needs, from an example: only a field
        # it claims and the live API lacks is a finding.
        check.schema_drift = [d for d in check.schema_drift if "returned live" not in d]
    check.spec_only = sorted(p for p in _schema_paths(spec, op.response_schema) if p not in live)
    stored: set[str] = set()
    for _, name in fixtures.list(spec.key):
        for response in fixtures.load(spec.key, name).operations.get(op.id, []):
            if response.status < 400:
                stored |= _paths(response.body)
    if stored:
        check.fixture_only = sorted(stored - live)
        check.live_only = sorted(live - stored)
    return check


def _operation(spec: ApiSpec, method: str, url: str) -> Operation | None:
    return best_operation(spec.operations, method.upper(), url)[0]


_TYPES: dict[str, tuple[type, ...]] = {
    "object": (dict,),
    "array": (list,),
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
}


def _drift(spec: ApiSpec, schema: dict[str, Any] | None, value: Any, at: str = "") -> list[str]:
    """Paths (never values) where ``value`` leaves ``schema``: unknown fields and wrong types."""
    node = spec.resolve(schema)
    if node is None:
        return []
    kind = node.get("type")
    if isinstance(kind, str) and kind in _TYPES and not isinstance(value, _TYPES[kind]):
        int64 = (
            kind == "string"
            and isinstance(value, int)
            and node.get("format") in ("int64", "uint64")
        )
        if not int64:
            return [f"{at or '$'}: expected {kind}, got {type(value).__name__}"]
    out: list[str] = []
    if isinstance(value, dict):
        props = node.get("properties")
        extra = node.get("additionalProperties")
        for key, item in value.items():
            here = f"{at}.{key}" if at else key
            if props is not None and key in props:
                out += _drift(spec, props[key], item, here)
            elif isinstance(extra, dict):
                out += _drift(spec, extra, item, here)
            elif props is not None and not extra:
                out.append(f"{here}: returned live, not in the spec")
    elif isinstance(value, list) and isinstance(node.get("items"), dict):
        seen: set[str] = set()
        for item in value:
            for problem in _drift(spec, node["items"], item, f"{at}[]"):
                if problem not in seen:
                    seen.add(problem)
                    out.append(problem)
    return out


def _paths(value: Any, at: str = "") -> set[str]:
    out: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            here = f"{at}.{key}" if at else key
            out.add(here)
            out |= _paths(item, here)
    elif isinstance(value, list):
        for item in value:
            out |= _paths(item, f"{at}[]")
    return out


def _last_json(text: str) -> dict[str, Any] | None:
    for line in reversed(text.strip().splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


__all__ = ["LiveCheckRefused", "LiveReport", "ResponseCheck", "entry_reads", "live_check"]
