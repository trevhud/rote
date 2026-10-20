"""A local fake of a vendor API, built from its spec and answered from fixtures.

The fidelity bench points an ``--backend api`` pipeline's vendor requests
here (:mod:`rote.vendor_specs.intercept`): a request to
``https://gmail.googleapis.com/gmail/v1/users/me/threads?q=...`` arrives as
``/gmail.googleapis.com/gmail/v1/users/me/threads?q=...``. The fake answers
the way the real API would for the case's data:

* an operation the spec does not have: 404;
* no credential (an ``Authorization`` header or a ``key`` parameter): 401;
* an undeclared, mistyped, out-of-enum or missing required parameter: 400,
  in Google's error shape;
* otherwise the first matching fixture's body, else the smallest
  spec-shaped answer (``{}`` for a list with no results; 404 for a file).

Every request is logged as JSONL, so the bench scores the pipeline's reads
the way it scores an MCP pipeline's tool calls. Nothing here reaches the
network.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from rote.vendor_specs import schema
from rote.vendor_specs.check import best_operation
from rote.vendor_specs.fixtures import FixtureSet, validate
from rote.vendor_specs.model import ApiSpec, Operation, ParamSpec, host_of


class FakeApi:
    """The routing and answering, separate from the HTTP server so it is testable."""

    def __init__(self, served: list[tuple[ApiSpec, FixtureSet | None]], log: Path) -> None:
        for spec, fixtures in served:
            if fixtures is not None:
                validate(fixtures, spec)
        self.served = served
        self.log = log
        self._lock = threading.Lock()

    @property
    def hosts(self) -> set[str]:
        return {h for spec, _ in self.served for op in spec.operations for h in _hosts(spec, op)}

    def answer(self, method: str, target: str, headers: dict[str, str]) -> tuple[int, str, bytes]:
        """``(status, content type, body)`` for ``method`` on ``/<host>/<path>?<query>``."""
        host, _, rest = target.lstrip("/").partition("/")
        split = urlsplit("/" + rest)
        query = {k: v if len(v) > 1 else v[0] for k, v in parse_qs(split.query).items()}
        found = self._route(method, host.lower(), split.path)
        entry: dict[str, Any] = {"host": host, "method": method, "path": split.path, "query": query}
        if found is None:
            return self._done(entry, 404, f"no operation {method} {split.path} on {host}")
        spec, fixtures, op, path_params = found
        entry |= {"spec": spec.key, "operation": op.id}
        params = {**path_params, **query}
        entry["params"] = params
        if "authorization" not in {k.lower() for k in headers} and "key" not in query:
            return self._done(entry, 401, "Request is missing required authentication credential.")
        bad = _param_problems(op, spec, path_params, query)
        if bad:
            return self._done(entry, 400, "; ".join(bad))
        # A file's bytes: always for an export, and on alt=media for a
        # method that supports media download (files.get).
        media = op.media_download or (op.supports_media and query.get("alt") == "media")
        responses = (fixtures.operations.get(op.id) if fixtures else None) or []
        for i, response in enumerate(responses):
            if response.matches(params):
                entry |= {"response": i, "reads": response.reads}
                if response.status >= 400:
                    return self._done(entry, response.status, str(response.body))
                if media and not isinstance(response.body, str):
                    return self._done(entry, 500, f"fixture {op.id}[{i}] has no file body")
                return self._ok(entry, media, response.body)
        entry["response"] = None
        if media:
            return self._done(entry, 404, "File not found.")
        return self._ok(entry, media, schema.minimal(spec, op.response_schema))

    def _route(
        self, method: str, host: str, path: str
    ) -> tuple[ApiSpec, FixtureSet | None, Operation, dict[str, str]] | None:
        # The check's own matcher: the most specific operation, upload URLs
        # included. The host is the fake's to filter (the check is
        # host-agnostic: Gmail also answers at www.googleapis.com/gmail/v1).
        for spec, fixtures in self.served:
            ops = [op for op in spec.operations if host in _hosts(spec, op)]
            op, bound = best_operation(ops, method, f"https://{host}{path}")
            if op is not None:
                return spec, fixtures, op, bound
        return None

    def _ok(self, entry: dict[str, Any], media: bool, body: Any) -> tuple[int, str, bytes]:
        entry["status"] = 200
        self._write(entry)
        if media:
            return 200, "text/plain; charset=utf-8", str(body).encode("utf-8")
        return 200, "application/json; charset=utf-8", json.dumps(body).encode("utf-8")

    def _done(self, entry: dict[str, Any], status: int, message: str) -> tuple[int, str, bytes]:
        entry |= {"status": status, "error": message}
        self._write(entry)
        error = {"error": {"code": status, "message": message, "status": _STATUS.get(status, "")}}
        return status, "application/json; charset=utf-8", json.dumps(error).encode("utf-8")

    def _write(self, entry: dict[str, Any]) -> None:
        with self._lock, self.log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")


_STATUS = {400: "INVALID_ARGUMENT", 401: "UNAUTHENTICATED", 404: "NOT_FOUND"}


def _param_problems(
    op: Operation, spec: ApiSpec, path_params: dict[str, str], query: dict[str, Any]
) -> list[str]:
    out: list[str] = []
    allowed = {
        **spec.global_params,
        **{n: p for n, p in op.params.items() if p.location == "query"},
    }
    for name, value in query.items():
        param = allowed.get(name)
        if param is None:
            out.append(
                f"Invalid JSON payload received. Unknown name {name!r}: "
                f"Cannot bind query parameter."
            )
            continue
        problem = _value_problem(param, value)
        if problem:
            out.append(f"Invalid value at {name!r} ({problem})")
    for name, param in op.params.items():
        if param.required and name not in query and name not in path_params:
            out.append(f"Required parameter: {name}")
    return out


def _value_problem(param: ParamSpec, value: Any) -> str | None:
    values = value if isinstance(value, list) else [value]
    if isinstance(value, list) and not param.repeated:
        return "repeated value for a single parameter"
    for v in values:
        if param.enum is not None and v not in param.enum:
            return f"{v!r} is not one of {', '.join(param.enum)}"
        if param.type == "integer" and not str(v).lstrip("-").isdigit():
            return f"{v!r} is not an integer"
        if param.type == "boolean" and str(v) not in ("true", "false", "True", "False"):
            return f"{v!r} is not a boolean"
    return None


#: Every Google API also answers on the shared host, at its own path
#: (www.googleapis.com/gmail/v1/... is Gmail), as the request check accepts.
_GOOGLE_SHARED_HOST = "www.googleapis.com"


def _hosts(spec: ApiSpec, op: Operation) -> set[str]:
    """The hosts ``op`` is answered on."""
    hosts = {h for t in op.templates() if (h := host_of(t))}
    if spec.vendor == "google":
        hosts.add(_GOOGLE_SHARED_HOST)
    return hosts


@contextmanager
def serving(api: FakeApi) -> Iterator[str]:
    """Serve ``api`` on a free local port for the block; yields its base URL."""

    class Handler(BaseHTTPRequestHandler):
        def _handle(self) -> None:
            status, ctype, body = api.answer(self.command, self.path, dict(self.headers))
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = _handle  # noqa: N815

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def read_log(log: Path) -> list[dict[str, Any]]:
    if not log.is_file():
        return []
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line]


__all__ = ["FakeApi", "read_log", "serving"]
