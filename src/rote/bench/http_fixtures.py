"""Fidelity for ``--backend api`` pipelines: vendor HTTP answered from spec-checked fixtures.

An api pipeline makes direct vendor requests instead of MCP tool calls, so
the bench serves those requests from a local fake
(:mod:`rote.vendor_specs.fake`) built from the vendor's own spec and the
case's ``http:`` fixtures, and routes the pipeline process's requests to it
(:mod:`rote.vendor_specs.intercept`). Each request the fake logs becomes a
call record in the MCP trace's shape. A fixture's ``reads`` names the MCP
fixture response that serves the same data, which makes an API read and
the skill's tool read the same read when the two are scored.

The emitted app's vendor libraries (``google-api-python-client``,
``httpx``...) are not rote's dependencies, so they are installed into a
cached target directory (``uv pip install --target``), keyed by the
requirement set, and put on the pipeline's ``PYTHONPATH``.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from rote.vendor_specs import docs
from rote.vendor_specs.deps import VendorDepsError, install_vendor_deps
from rote.vendor_specs.fake import FakeApi, read_log, serving
from rote.vendor_specs.fixtures import FixtureError, FixtureSet
from rote.vendor_specs.model import ApiSpec
from rote.vendor_specs.store import SpecStore, SpecUnavailable

INTERCEPT_DIR = Path(__file__).resolve().parent.parent / "vendor_specs" / "intercept"

#: Host suffix per vendor: any host of a vendor in play that the fake does
#: not answer for is blocked, so a run can never reach the real API.
VENDOR_HOSTS = {"google": "googleapis.com"}

#: Values the bench gives every credential variable a module reads: the
#: fake accepts any token, and the real API never sees one.
FIXTURE_CREDENTIAL = "rote-fidelity-fixture"

_ENV_READ = re.compile(
    r"""os\.(?:environ\.get|getenv)\(\s*['"]([A-Z][A-Z0-9_]*)['"]|os\.environ\[\s*['"]([A-Z][A-Z0-9_]*)['"]\s*\]"""
)


class HttpFixtureError(RuntimeError):
    """The bench could not stand up the api side (environment, not pipeline)."""


def load_http(raw: dict[str, Any] | None) -> dict[str, FixtureSet]:
    """A case's ``http:`` section: ``{spec key: {trust?, sources?, operations}}``."""
    return {key: FixtureSet(spec=key, **value) for key, value in (raw or {}).items()}


def specs_for(
    fixtures: dict[str, FixtureSet], store: SpecStore, compiled_dir: Path | None = None
) -> list[ApiSpec]:
    """The spec each fixture set answers for: the vendor's own, else the
    compile's docs spec (``vendor_specs/``, lower trust)."""
    written = {s.key: s for _, s in docs.load(compiled_dir)} if compiled_dir else {}
    out: list[ApiSpec] = []
    for key in fixtures:
        vendor, api, version = key.split("/")
        if key in written:
            out.append(written[key])
            continue
        if vendor != "google":
            raise HttpFixtureError(
                f"no spec for {key}: neither a source rote reads nor a docs spec in the compile"
            )
        try:
            spec = store.google(api, version)
        except SpecUnavailable as e:
            # The bench's environment (Discovery unreachable), not the
            # pipeline's: it stops the bench instead of scoring a 0.0.
            raise HttpFixtureError(f"the spec for the fixtures {key} is unavailable: {e}") from e
        if spec is None:
            raise HttpFixtureError(f"Google lists no {api} {version} for the fixtures {key}")
        out.append(spec)
    return out


@contextmanager
def api_side(
    app: Path,
    fixtures: dict[str, FixtureSet],
    work: Path,
    store: SpecStore | None = None,
    compiled_dir: Path | None = None,
) -> Iterator[tuple[dict[str, str], str | None, Path]]:
    """Fake every vendor the case has fixtures for; yield ``(env, python, log)``."""
    store = store or SpecStore()
    specs = specs_for(fixtures, store, compiled_dir)
    work.mkdir(parents=True, exist_ok=True)
    log = work / "http-calls.jsonl"
    log.write_text("", encoding="utf-8")
    try:
        api = FakeApi([(spec, fixtures[spec.key]) for spec in specs], log)
    except FixtureError as e:
        # A ValueError: left alone, the bench scored it as a dead pipeline run.
        raise HttpFixtureError(f"a case's http: fixtures do not fit the spec: {e}") from e
    try:
        deps = install_vendor_deps(app)
    except VendorDepsError as e:
        raise HttpFixtureError(str(e)) from e
    with serving(api) as base:
        path = [str(INTERCEPT_DIR), *([str(deps)] if deps else [])]
        env = {
            "PYTHONPATH": ":".join(path),
            "ROTE_HTTP_FIXTURE_URL": base,
            "ROTE_HTTP_FIXTURE_HOSTS": ",".join(sorted(api.hosts)),
            "ROTE_HTTP_FIXTURE_BLOCK": ",".join(
                sorted({VENDOR_HOSTS[s.vendor] for s in specs if s.vendor in VENDOR_HOSTS})
            ),
            **{name: FIXTURE_CREDENTIAL for name in credential_variables(app)},
        }
        yield env, None, log


def credential_variables(app: Path) -> set[str]:
    """The environment variables the app's extracted modules read."""
    names: set[str] = set()
    for path in (app / "extracted").glob("*.py"):
        if path.name.startswith("_rote"):
            continue
        for match in _ENV_READ.finditer(path.read_text(encoding="utf-8", errors="replace")):
            names.add(match.group(1) or match.group(2))
    return {n for n in names if not n.startswith("ROTE_")}


def api_calls(log: Path) -> list[dict[str, Any]]:
    """The fake's request log as call records in the MCP trace's shape.

    A request answered by a fixture that ``reads`` an MCP fixture response
    is recorded as that tool call, so it keys the same as the skill's read.
    Every other request keys by its operation and parameters.
    """
    calls: list[dict[str, Any]] = []
    for entry in read_log(log):
        reads = entry.get("reads")
        side_effect = entry.get("method") not in ("GET", "HEAD")
        if reads and (m := re.fullmatch(r"([^.]+)\.([^\[]+)\[(\d+)\]", reads)):
            calls.append(
                {
                    "server": m.group(1),
                    "tool": m.group(2),
                    "arguments": entry.get("params") or {},
                    "side_effect": side_effect,
                    "response": int(m.group(3)),
                    "via": entry.get("operation"),
                }
            )
            continue
        calls.append(
            {
                "server": entry.get("spec") or entry.get("host"),
                "tool": entry.get("operation") or f"{entry.get('method')} {entry.get('path')}",
                "arguments": entry.get("params") or entry.get("query") or {},
                "side_effect": side_effect,
                "response": None,
                "status": entry.get("status"),
            }
        )
    return calls


__all__ = [
    "FIXTURE_CREDENTIAL",
    "HttpFixtureError",
    "api_calls",
    "api_side",
    "credential_variables",
    "load_http",
]
