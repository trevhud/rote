"""A read-only live check: call the real vendor once and compare its shape with the spec.

``rote run <compile dir> --live-check --input <json>`` checks each entry
vendor read of an ``--backend api`` deliverable (an external_call node whose
inputs all come from the pipeline input or its constants) against the real
API, once, with the user's own credentials.

**The deliverable's code never runs.** A live run of generated code holds
the user's real credentials, and no patch over its HTTP clients can promise
it only reads: it can write through any client, any helper module, any host
alias or a subprocess. So the check builds each request itself from the
static extraction (:mod:`rote.vendor_specs.extract`): the operation the spec
check resolved, the URL template, the query parameters, and the headers,
filled from the node's inputs and the user's environment. It sends only GET
or HEAD, through one client it controls (no redirects followed). A request
it cannot build exactly from that (a value computed at run time, a helper
function, a request the extractor does not see) is reported as not
live-checked, with why, and is not sent.

Each response is compared with what rote believed:

* the operation's response schema (the vendor's spec, or the compile's docs
  spec): a field the live API returned that the schema lacks, and a value
  of the wrong type;
* the stored fixtures for that operation (:mod:`.fixture_store`): fields
  the fixtures have that the live response did not return, and the
  reverse.

The report holds schema paths and types, never values: a map's live keys
(user data in Drive ``appProperties``) are ``{key}``, and a live field name
that does not look like one is too. Errors are an exception type and an HTTP
status, never their text: a URL in an error message carries query
credentials. Nothing the user's account returned is kept.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rote.compiler import api_docs
from rote.ir import NodeKind, Pipeline, load_pipeline, parse_input_ref
from rote.vendor_specs import docs, google
from rote.vendor_specs.check import best_operation, check_deliverable
from rote.vendor_specs.extract import UNKNOWN, CallSite, Expr, extract
from rote.vendor_specs.fixture_store import FixtureStore
from rote.vendor_specs.model import ApiSpec, Operation, host_of, path_segments
from rote.vendor_specs.store import SpecStore

LIVE_TIMEOUT_SECONDS = 30.0
#: The most of one live response the check reads.
MAX_RESPONSE_BYTES = 5 * 1024 * 1024
_READ_METHODS = frozenset({"GET", "HEAD"})
#: A docs spec's method is whatever the agent wrote; an operation named like
#: a write is refused whatever its method says.
_WRITE_NAMES = re.compile(
    r"(^|[._-])(send|delete|remove|create|insert|update|patch|put|trash|purge|write|"
    r"modify|move|copy|post|upload|import|batch)",
    re.IGNORECASE,
)
#: Headers the check will not send: a method override turns a GET into a write.
_REFUSED_HEADERS = frozenset({"x-http-method-override", "x-method-override"})
_TEMPLATE_PART = re.compile(r"\{(\$env:)?([^}]*)\}")
_FIELD_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")


class LiveCheckRefused(RuntimeError):
    """The deliverable cannot be live-checked without risking a write."""


@dataclass
class ResponseCheck:
    """One live response, compared with the spec and the stored fixtures."""

    node_id: str
    method: str
    host: str
    #: The operation's path template, never the concrete path (ids are data).
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
    #: Requests that were not sent, keyed ``<node>:<path>:<line>``, with why.
    unchecked: dict[str, str] = field(default_factory=dict)
    #: Requests that were sent and failed: an exception type and status only.
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def drifted(self) -> bool:
        return any(c.schema_drift or c.reads_missing for c in self.checked)

    def as_dict(self) -> dict[str, Any]:
        return {
            "checked": [c.__dict__ for c in self.checked],
            "skipped": self.skipped,
            "unchecked": self.unchecked,
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


@dataclass(frozen=True)
class LiveRequest:
    """One request the check sends: always GET or HEAD."""

    method: str
    url: str
    headers: dict[str, str]


class _Unbuildable(Exception):
    """Why a request cannot be built exactly from the static extraction."""


@dataclass(frozen=True)
class PlannedRequest:
    """One request the check will send, as the user confirms it."""

    key: str
    node_id: str
    method: str
    host: str
    operation: str
    #: The environment variable (or Google's OAuth token) it carries, if any.
    credential: str | None


Confirm = Callable[[list[PlannedRequest]], bool]


def live_check(
    compiled_dir: Path,
    payload: dict[str, Any],
    specs: SpecStore | None = None,
    fixtures: FixtureStore | None = None,
    env: dict[str, str] | None = None,
    send: Any = None,
    *,
    confirm: Confirm,
    allow_hosts: frozenset[str] = frozenset(),
    allow_credentials: frozenset[str] = frozenset(),
) -> LiveReport:
    """Send each entry vendor read once to the real API, built statically; compare shapes.

    Every request is built before any is sent, and ``confirm`` sees the whole
    plan (each host, operation and credential); nothing is sent unless it
    returns True. ``env`` is where the declared credential is read (default:
    ``os.environ``); ``send`` replaces the HTTP client (tests count what it
    was asked to send). ``allow_hosts`` admits docs-spec hosts the compile's
    fetched pages do not vouch for.
    """
    specs = specs or SpecStore()
    fixtures = fixtures or FixtureStore()
    env = dict(os.environ) if env is None else env
    send = send or _send
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
    ops = {
        op.id: (spec, op) for spec in _specs_of(checks, specs, written) for op in spec.operations
    }
    writes = [
        c
        for c in checks
        if c.operation
        and (
            ops[c.operation][1].method not in _READ_METHODS
            or (ops[c.operation][0].source == "docs" and _WRITE_NAMES.search(c.operation))
        )
    ]
    if writes:
        raise LiveCheckRefused(
            "the live check is read-only, and this deliverable writes: "
            + ", ".join(f"{c.operation} ({c.path}:{c.line})" for c in writes)
        )
    _refuse_unvouched_hosts(
        [ops[c.operation or ""] for c in checks], api_docs.read_fetched(compiled_dir), allow_hosts
    )
    runnable, skipped = entry_reads(pipeline)
    report = LiveReport(skipped=skipped)
    # Build everything first: a credential refused anywhere sends nothing.
    planned: list[tuple[PlannedRequest, LiveRequest, ApiSpec, Operation, dict[str, set[str]]]] = []
    for node in runnable:
        rel, _, symbol = node.impl.partition(":")
        module = compiled_dir / rel
        kwargs = _kwargs(node, payload)
        sites = extract(module.read_text(encoding="utf-8"))
        own = [s for s in sites if s.function == symbol]
        if not own:
            report.unchecked[node.id] = (
                f"{symbol}() makes no request the check can see (a helper function or an "
                f"HTTP client the extractor does not read): nothing was sent"
            )
            continue
        code_reads = _reads_by_operation(module, specs, written)
        for site in own:
            key = f"{node.id}:{rel}:{site.line}"
            check = next(c for c in checks if c.path == rel and c.line == site.line)
            spec, op = ops[check.operation or ""]
            try:
                request, credential = _build(
                    site,
                    spec,
                    op,
                    kwargs,
                    env,
                    frozenset((h, c) for h in allow_hosts for c in allow_credentials),
                )
            except _Unbuildable as e:
                report.unchecked[key] = str(e)
                continue
            plan = PlannedRequest(
                key, node.id, request.method, host_of(request.url) or "", op.id, credential
            )
            planned.append((plan, request, spec, op, code_reads))
    if planned and not confirm([p for p, *_ in planned]):
        raise LiveCheckRefused("the live calls were not confirmed; nothing was sent")
    for plan, request, spec, op, code_reads in planned:
        assert request.method in _READ_METHODS  # the one invariant everything rests on
        try:
            status, content_type, body = send(request)
        except Exception as e:  # noqa: BLE001 (any failure is reported, never its text)
            report.errors[plan.key] = _redacted(e)
            continue
        report.checked.append(
            _compare(plan.node_id, spec, op, status, content_type, body, fixtures, code_reads)
        )
    return report


def registrable_domain(host: str) -> str:
    """``slack.com`` for ``api.slack.com``: the last two labels, three under a
    two-letter country code's ``co``/``com``/``org``/``net``/``ac``/``gov``/``edu``
    (``bbc.co.uk``). An IP literal or a one-label host is itself."""
    labels = host.lower().rstrip(".").split(".")
    if len(labels) < 2 or all(part.isdigit() for part in labels) or ":" in host:
        return host.lower()
    second = {"co", "com", "org", "net", "ac", "gov", "edu"}
    if len(labels) >= 3 and len(labels[-1]) == 2 and labels[-2] in second:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def _refuse_unvouched_hosts(
    used: list[tuple[ApiSpec, Operation]], fetched: set[str], allow_hosts: frozenset[str]
) -> None:
    """A docs spec is the agent's claim, and so is the host it declares. That host
    is called only when it equals, or shares the registrable domain of, a page the
    spec cites that rote itself recorded fetching, or when the user names it.

    "Every cited page was fetched" vouches for nothing: any fetched page would
    then vouch for any host (review6 reproduced a python.org page vouching for
    127.0.0.1)."""
    for spec, op in used:
        if spec.source != "docs":
            continue
        host = (host_of(op.url_template) or "").split(":")[0]
        if host in allow_hosts:
            continue
        pages = [s for s in spec.sources if s != "skill"]
        page_hosts = {
            h.split(":")[0]
            for p in pages
            if api_docs.normalize_url(p) in fetched and (h := host_of(p))
        }
        if host in page_hosts or registrable_domain(host) in {
            registrable_domain(h) for h in page_hosts
        }:
            continue
        if not pages:
            why = "cites only the skill"
        elif not page_hosts:
            why = "cites pages this compile has no record of fetching"
        else:
            why = f"cites fetched pages on {', '.join(sorted(page_hosts))}, another domain"
        raise LiveCheckRefused(
            f"{host} comes from the compile's own docs spec ({spec.key}), which {why}: "
            f"pass --allow-host {host} to call it anyway"
        )


class CredentialRefused(LiveCheckRefused):
    """The code would send an environment value the spec does not declare as its auth."""


def _build(
    site: CallSite,
    spec: ApiSpec,
    op: Operation,
    kwargs: dict[str, Any],
    env: dict[str, str],
    allowed: frozenset[tuple[str, str]] = frozenset(),
) -> tuple[LiveRequest, str | None]:
    """The exact GET/HEAD ``site`` makes, and the credential it carries.

    Only the credential the spec declares is ever sent: Google's OAuth token
    (the check's own, whatever the code builds), or a docs spec's ``auth``
    at its declared place. Any other environment read the code makes, in the
    URL, a parameter or a header, refuses the whole check: a skill can
    steer the code and the docs spec both, and the user's shell holds more
    secrets than this vendor's.
    """
    if not site.params_complete:
        raise _Unbuildable("the call spreads parameters the check cannot see (**kwargs)")
    if site.kind == "google_client":
        return _build_google(site, spec, op, kwargs, env), "Google OAuth token"
    if site.auth_kwarg:
        raise _Unbuildable("the call authenticates with an auth= object the check cannot rebuild")
    if site.headers is None:
        raise _Unbuildable("the call passes headers the check cannot read")
    assert site.url is not None and site.method is not None
    google_api = spec.vendor == "google"
    headers_in = dict(site.headers)
    if google_api:
        # The check sends Google's token itself; the code's own is not used.
        headers_in = {k: v for k, v in headers_in.items() if k.lower() != "authorization"}
    used = _check_credentials(site.url, site.params, headers_in, spec)
    params_in = dict(site.params)
    if used is not None and spec.auth is not None:
        host = (host_of(site.url) or "").split(":")[0]
        if (host, spec.auth.env) not in allowed:
            # A docs spec's credential is the agent's choice of secret: sent
            # only when the user names both the host and the variable.
            target = headers_in if spec.auth.location == "header" else params_in
            for name in [n for n in target if n.lower() == spec.auth.name.lower()]:
                del target[name]
            used = None
    url = _fill(site.url, kwargs, env, path=True)
    query = {name: _resolve(value, kwargs, env) for name, value in params_in.items()}
    headers = {name: str(_resolve(value, kwargs, env)) for name, value in headers_in.items()}
    _refuse_method_override(headers)
    credential = used
    if google_api:
        headers["Authorization"] = f"Bearer {_google_token(op, env)}"
        credential = "Google OAuth token"
    return LiveRequest(site.method, _with_query(url, query), headers), credential


def _env_names(value: object) -> set[str]:
    text = value.template if isinstance(value, Expr) else value if isinstance(value, str) else ""
    return {m.group(2) for m in _TEMPLATE_PART.finditer(text) if m.group(1)}


def _check_credentials(
    url: str, params: dict[str, object], headers: dict[str, object], spec: ApiSpec
) -> str | None:
    """The declared credential the request uses, or raise for any other env read."""
    auth = spec.auth
    used: set[str] = set()
    places = [("url", "", url)] + [("query", n, v) for n, v in params.items()]
    places += [("header", n, v) for n, v in headers.items()]
    for location, name, value in places:
        names = _env_names(value)
        if not names:
            continue
        allowed = (
            auth is not None
            and names == {auth.env}
            and location == auth.location
            and name.lower() == auth.name.lower()
        )
        if not allowed:
            where = "the URL" if location == "url" else f"{location} {name!r}"
            declared = (
                f"the spec's declared credential is {auth.env} in {auth.location} {auth.name!r}"
                if auth is not None
                else "the spec declares no credential"
            )
            raise CredentialRefused(
                f"the code sends {', '.join(sorted(names))} from the environment in {where}, "
                f"and {declared}: nothing was sent"
            )
        used |= names
    return next(iter(used), None)


def _build_google(
    site: CallSite, spec: ApiSpec, op: Operation, kwargs: dict[str, Any], env: dict[str, str]
) -> LiveRequest:
    """A client-library call as the HTTP request the library would send."""
    # Its credential is the check's own Google token: no environment value
    # rides along as a parameter.
    _check_credentials("", site.params, {}, spec)
    by_client = {google.client_param(n): n for n in {**spec.global_params, **op.params}}
    path: dict[str, str] = {}
    query: dict[str, Any] = {}
    for name, value in site.params.items():
        if name in ("body", "media_body", "media_mime_type"):
            raise _Unbuildable(f"the call passes {name}, which a read does not")
        discovery = by_client.get(name)
        if discovery is None:
            raise _Unbuildable(f"{name} is not a parameter of {op.id}")
        resolved = _resolve(value, kwargs, env)
        if discovery in op.params and op.params[discovery].location == "path":
            path[discovery] = str(resolved)
        else:
            query[discovery] = resolved
    url = op.url_template
    for part in re.findall(r"\{\+?([^}]+)\}", op.url_template):
        if part not in path:
            raise _Unbuildable(f"path parameter {part} is not passed")
        safe = "/" if "{+" + part + "}" in op.url_template else ""
        url = url.replace("{+" + part + "}", urllib.parse.quote(path[part], safe=safe))
        url = url.replace("{" + part + "}", urllib.parse.quote(path[part], safe=safe))
    token = _google_token(op, env)
    return LiveRequest(
        op.method if op.method in _READ_METHODS else "GET",
        _with_query(url, query),
        {"Authorization": f"Bearer {token}"},
    )


def _google_token(op: Operation, env: dict[str, str]) -> str:
    """An access token for the user's Google credential (application default)."""
    if env.get("GOOGLE_OAUTH_ACCESS_TOKEN"):
        return env["GOOGLE_OAUTH_ACCESS_TOKEN"]
    try:
        import google.auth
        import google.auth.transport.requests
    except ImportError as e:
        raise _Unbuildable(
            "a Google client-library call needs google-auth installed to get the user's "
            "credential, or GOOGLE_OAUTH_ACCESS_TOKEN set"
        ) from e
    try:
        found: Any = google.auth.default(scopes=list(op.scopes) or None)
        credentials = found[0]
        credentials.refresh(google.auth.transport.requests.Request())
    except Exception as e:  # noqa: BLE001 (the reason is the type, never its text)
        raise _Unbuildable(
            f"no usable Google credential ({type(e).__name__}): run `gcloud auth "
            f"application-default login` or set GOOGLE_OAUTH_ACCESS_TOKEN"
        ) from e
    return str(credentials.token)


def _resolve(value: object, kwargs: dict[str, Any], env: dict[str, str]) -> Any:
    if value is UNKNOWN:
        raise _Unbuildable("a value is computed at run time")
    if isinstance(value, Expr):
        whole = _TEMPLATE_PART.fullmatch(value.template)
        if whole is not None and not whole.group(1):
            # Exactly one reference: the value itself, its type kept (an
            # integer parameter from the input stays an integer).
            found = _lookup(kwargs, whole.group(2))
            if found is None or not _scalar_or_list(found):
                raise _Unbuildable("a value is computed at run time")
            return found
        return _fill(value.template, kwargs, env)
    return value


def _scalar_or_list(value: Any) -> bool:
    if isinstance(value, list):
        return all(isinstance(v, (str, int, float, bool)) for v in value)
    return isinstance(value, (str, int, float, bool))


def _lookup(kwargs: dict[str, Any], label: str) -> Any:
    """``request.season`` in the node's arguments, or ``None`` when it is not there."""
    head, *keys = label.split(".")
    if head == "?" or head not in kwargs:
        return None
    value = kwargs[head]
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _fill(template: str, kwargs: dict[str, Any], env: dict[str, str], path: bool = False) -> str:
    def part(match: re.Match[str]) -> str:
        is_env, name = match.group(1), match.group(2)
        if is_env:
            if name not in env:
                raise _Unbuildable(f"the code reads {name} from the environment, and it is not set")
            return env[name]
        value = _lookup(kwargs, name)
        if not isinstance(value, (str, int, float)) or isinstance(value, bool):
            raise _Unbuildable("a value is computed at run time")
        text = str(value)
        return urllib.parse.quote(text, safe="") if path else text

    return _TEMPLATE_PART.sub(part, template)


def _refuse_method_override(headers: dict[str, Any]) -> None:
    if any(name.lower() in _REFUSED_HEADERS for name in headers):
        raise _Unbuildable("the call sends a method-override header")


def _with_query(url: str, query: dict[str, Any]) -> str:
    if not query:
        return url
    pairs: list[tuple[str, str]] = []
    for name, value in query.items():
        for v in value if isinstance(value, (list, tuple)) else [value]:
            pairs.append((name, str(v).lower() if isinstance(v, bool) else str(v)))
    sep = "&" if "?" in url else "?"
    return url + sep + urllib.parse.urlencode(pairs)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None  # a 3xx is returned as the response, never followed


def _send(request: LiveRequest) -> tuple[int, str, bytes]:
    """The one HTTP client the live check uses: GET or HEAD, no redirects, bounded."""
    if request.method not in _READ_METHODS:
        raise LiveCheckRefused(f"refusing to send {request.method}")
    opener = urllib.request.build_opener(_NoRedirect)
    req = urllib.request.Request(request.url, method=request.method, headers=request.headers)
    try:
        with opener.open(req, timeout=LIVE_TIMEOUT_SECONDS) as response:  # noqa: S310
            return (
                response.status,
                response.headers.get("Content-Type", ""),
                response.read(MAX_RESPONSE_BYTES),
            )
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type", ""), b""


def _redacted(error: BaseException) -> str:
    """An error as its type (and HTTP status): never its text, which carries URLs."""
    status = getattr(error, "code", None) or getattr(error, "status", None)
    return f"{type(error).__name__}{f' {status}' if isinstance(status, int) else ''}"


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
    spec: ApiSpec,
    op: Operation,
    status: int,
    content_type: str,
    raw: bytes,
    fixtures: FixtureStore,
    reads: dict[str, set[str]] | None = None,
) -> ResponseCheck:
    check = ResponseCheck(
        node_id=node_id,
        method=op.method,
        host=host_of(op.url_template) or "",
        path="/" + "/".join(path_segments(op.url_template)),
        status=status,
        operation=op.id,
        spec=spec.key,
    )
    if status >= 400 or "json" not in content_type:
        return check
    try:
        body = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        check.schema_drift.append("$: the response is not JSON")
        return check
    check.schema_drift = _drift(spec, op.response_schema, body)
    live = _paths(spec, op.response_schema, body)
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
                stored |= _paths(spec, op.response_schema, response.body)
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
            if props is not None and key in props:
                out += _drift(spec, props[key], item, _join(at, key))
            elif isinstance(extra, dict):
                out += _drift(spec, extra, item, _join(at, "{key}"))
            elif props is not None and not extra:
                out.append(f"{_join(at, _shown(key))}: returned live, not in the spec")
    elif isinstance(value, list) and isinstance(node.get("items"), dict):
        seen: set[str] = set()
        for item in value:
            for problem in _drift(spec, node["items"], item, f"{at}[]"):
                if problem not in seen:
                    seen.add(problem)
                    out.append(problem)
    return out


def _paths(spec: ApiSpec, schema: dict[str, Any] | None, value: Any, at: str = "") -> set[str]:
    """``value``'s field paths, named as the schema names them: a map's keys are
    ``{key}``, and a field the schema does not know is shown only if it looks
    like a field name (never an email or an id)."""
    node = spec.resolve(schema) or {}
    out: set[str] = set()
    if isinstance(value, dict):
        props = node.get("properties") or {}
        extra = node.get("additionalProperties")
        for key, item in value.items():
            if key in props:
                here, sub = _join(at, key), props[key]
            elif isinstance(extra, dict):
                here, sub = _join(at, "{key}"), extra
            else:
                here, sub = _join(at, _shown(key)), None
            out.add(here)
            out |= _paths(spec, sub, item, here)
    elif isinstance(value, list):
        items = node.get("items") if isinstance(node.get("items"), dict) else None
        for item in value:
            out |= _paths(spec, items, item, f"{at}[]")
    return out


def _join(at: str, key: str) -> str:
    return f"{at}.{key}" if at else key


def _shown(key: str) -> str:
    return key if _FIELD_NAME.fullmatch(key) else "{key}"


__all__ = ["LiveCheckRefused", "LiveReport", "ResponseCheck", "entry_reads", "live_check"]
