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

import ast
import hashlib
import importlib.util
import re
import shutil
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from rote.vendor_specs.fake import FakeApi, read_log, serving
from rote.vendor_specs.fixtures import FixtureError, FixtureSet
from rote.vendor_specs.model import ApiSpec
from rote.vendor_specs.store import SpecStore, SpecUnavailable, default_root

INTERCEPT_DIR = Path(__file__).resolve().parent.parent / "vendor_specs" / "intercept"

#: Import name -> the distribution that provides it, for vendor libraries an
#: emitted module may use that rote itself does not install.
DISTRIBUTIONS = {
    "googleapiclient": "google-api-python-client",
    "google_auth_httplib2": "google-auth-httplib2",
    "google.auth": "google-auth",
    "google.oauth2": "google-auth",
    "httplib2": "httplib2",
    "httpx": "httpx",
    "requests": "requests",
}

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


def specs_for(fixtures: dict[str, FixtureSet], store: SpecStore) -> list[ApiSpec]:
    out: list[ApiSpec] = []
    for key in fixtures:
        vendor, api, version = key.split("/")
        if vendor != "google":
            raise HttpFixtureError(f"no spec source for vendor {vendor!r} ({key})")
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
    app: Path, fixtures: dict[str, FixtureSet], work: Path, store: SpecStore | None = None
) -> Iterator[tuple[dict[str, str], str | None, Path]]:
    """Fake every vendor the case has fixtures for; yield ``(env, python, log)``."""
    store = store or SpecStore()
    specs = specs_for(fixtures, store)
    work.mkdir(parents=True, exist_ok=True)
    log = work / "http-calls.jsonl"
    log.write_text("", encoding="utf-8")
    try:
        api = FakeApi([(spec, fixtures[spec.key]) for spec in specs], log)
    except FixtureError as e:
        # A ValueError: left alone, the bench scored it as a dead pipeline run.
        raise HttpFixtureError(f"a case's http: fixtures do not fit the spec: {e}") from e
    deps = install_vendor_deps(app)
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


def vendor_requirements(app: Path) -> list[str]:
    """Distributions the app imports that this interpreter cannot import yet.

    An import that is neither importable here nor in :data:`DISTRIBUTIONS`
    is an error: a guessed distribution name could install the wrong
    package.
    """
    needed: set[str] = set()
    unknown: set[str] = set()
    for path in sorted((app / "extracted").glob("*.py")) + sorted(
        (app / "signatures").glob("*.py")
    ):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module]
            for name in names:
                top = name.split(".")[0]
                if top in sys.stdlib_module_names or top in ("extracted", "signatures"):
                    continue
                dist = (
                    DISTRIBUTIONS.get(name)
                    or DISTRIBUTIONS.get(".".join(name.split(".")[:2]))
                    or DISTRIBUTIONS.get(top)
                )
                if _importable(name):
                    continue
                if dist is None:
                    unknown.add(name)
                else:
                    needed.add(dist)
    if unknown:
        raise HttpFixtureError(
            f"the app imports {', '.join(sorted(unknown))}, which is not installed here and "
            f"has no known distribution (rote.bench.http_fixtures.DISTRIBUTIONS)"
        )
    return sorted(needed)


def _importable(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def install_vendor_deps(app: Path) -> Path | None:
    """Install the app's missing vendor libraries into a cached target dir."""
    wanted = vendor_requirements(app)
    if not wanted:
        return None
    digest = hashlib.sha256("\n".join([sys.version, *wanted]).encode()).hexdigest()[:16]
    target = default_root().parent / "bench-deps" / digest
    if (target / ".complete").is_file():
        return target
    uv = shutil.which("uv")
    if uv is None:
        raise HttpFixtureError(f"installing {', '.join(wanted)} for the app needs uv on PATH")
    if target.exists():
        shutil.rmtree(target)
    result = subprocess.run(
        [
            uv,
            "pip",
            "install",
            "--quiet",
            "--python",
            sys.executable,
            "--target",
            str(target),
            *wanted,
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        raise HttpFixtureError(f"could not install {', '.join(wanted)}: {result.stderr[-2000:]}")
    (target / ".complete").write_text("\n".join(wanted), encoding="utf-8")
    return target


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
    "install_vendor_deps",
    "load_http",
    "vendor_requirements",
]
