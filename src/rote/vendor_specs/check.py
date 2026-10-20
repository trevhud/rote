"""Check every vendor request a deliverable makes against the vendor's own spec.

For each request :mod:`rote.vendor_specs.extract` finds in an external_call
module, the check:

1. resolves the spec (a Google API by its URL or ``build(...)`` name) and
   the operation (method + URL template, or the Discovery method id);
2. checks the parameters: every one the spec declares, every required one
   present, a literal value of the declared type and in the declared enum;
3. checks the request body's top-level keys against the request schema;
4. checks every response field the code reads against the response schema.

Each violation is a :class:`~rote.contracts.ContractFinding` error, sent back
to the agent with the repair loop like any other. A request the check cannot
resolve to a spec it can fetch is not an error: it is listed, with why, in
the report (``checks``), and an unreachable spec service never fails a paid
compile.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from rote.contracts import ContractFinding
from rote.ir import NodeKind, Pipeline
from rote.vendor_specs import google
from rote.vendor_specs.extract import UNKNOWN, CallSite, extract
from rote.vendor_specs.model import ApiSpec, Operation, ParamSpec, host_of, path_segments
from rote.vendor_specs.store import SpecStore, SpecUnavailable

CheckStatus = Literal["ok", "violations", "unresolved", "spec_unavailable"]


@dataclass
class RequestCheck:
    """One request, what it resolved to, and what was wrong with it."""

    node_ids: tuple[str, ...]
    path: str
    line: int
    status: CheckStatus
    operation: str | None = None
    spec: str | None = None
    #: Why a request is unresolved or its spec unavailable.
    note: str | None = None
    findings: list[ContractFinding] = field(default_factory=list)


def check_deliverable(pipeline: Pipeline, root: Path, store: SpecStore) -> list[RequestCheck]:
    """Every vendor request in the pipeline's external_call modules, checked."""
    modules: dict[str, list[str]] = {}
    for node in pipeline.nodes:
        if node.kind is NodeKind.EXTERNAL_CALL and node.impl:
            modules.setdefault(node.impl.partition(":")[0], []).append(node.id)
    checks: list[RequestCheck] = []
    for rel, node_ids in sorted(modules.items()):
        try:
            source = (root / rel).read_text(encoding="utf-8")
            sites = extract(source)
        except (OSError, UnicodeDecodeError, SyntaxError):
            continue  # check_contracts reports an unreadable module
        for site in sites:
            checks.append(_check(site, tuple(node_ids), rel, store))
    return checks


def spec_findings(checks: list[RequestCheck]) -> list[ContractFinding]:
    return [f for c in checks for f in c.findings]


def _check(site: CallSite, node_ids: tuple[str, ...], rel: str, store: SpecStore) -> RequestCheck:
    def done(status: CheckStatus, **kw: Any) -> RequestCheck:
        return RequestCheck(node_ids, rel, site.line, status, **kw)

    if site.kind == "google_client":
        assert site.google_api is not None
        api, version = site.google_api
    else:
        if site.url is None:
            return done("unresolved", note="the URL is built at run time")
        named = google.api_for_url(site.url)
        if named is None:
            host = host_of(site.url) or site.url
            return done("unresolved", note=f"no machine-readable spec source for {host} yet")
        try:
            listed = store.google_versions(named.api)
        except SpecUnavailable as e:
            return done("spec_unavailable", note=str(e))
        resolved = next((v for v in named.versions if v in listed), None)
        if resolved is None:
            if listed and named.versions and named.dedicated_host:
                # The API's own host, a version Google does not list: wrong.
                return done(
                    "violations",
                    findings=[
                        _finding(
                            node_ids[0],
                            rel,
                            site.line,
                            "spec_unknown_api",
                            f"line {site.line}: Google's Discovery directory lists "
                            f"{named.api} as {', '.join(listed)}, not "
                            f"{' or '.join(named.versions)} ({google.DIRECTORY_URL})",
                        )
                    ],
                )
            return done(
                "unresolved",
                note=(
                    f"Google's Discovery directory describes no {named.api} "
                    f"{' or '.join(named.versions) or 'version'} for this URL, so it has no "
                    f"spec to check against"
                ),
            )
        api, version = named.api, resolved
    try:
        spec = store.google(api, version)
    except SpecUnavailable as e:
        return done("spec_unavailable", note=str(e))
    if spec is None:
        return done(
            "violations",
            findings=[
                _finding(
                    node_ids[0],
                    rel,
                    site.line,
                    "spec_unknown_api",
                    f"line {site.line}: Google lists no API {api!r} version {version!r} in its "
                    f"Discovery directory ({google.DIRECTORY_URL})",
                )
            ],
        )
    label = f"{spec.key} rev {spec.revision}"
    op, matched_path = _operation(site, spec)
    if op is None:
        where = site.google_method_id or f"{site.method} {site.url}"
        return done(
            "violations",
            spec=label,
            findings=[
                _finding(
                    node_ids[0],
                    rel,
                    site.line,
                    "spec_unknown_operation",
                    f"line {site.line}: {where} is not an operation of {label} ({spec.source_url})",
                )
            ],
        )
    findings = [
        _finding(node_ids[0], rel, site.line, kind, f"line {site.line}: {op.id}: {text}")
        for kind, text in _violations(site, op, spec, matched_path)
    ]
    return done("violations" if findings else "ok", operation=op.id, spec=label, findings=findings)


def _operation(site: CallSite, spec: ApiSpec) -> tuple[Operation | None, dict[str, str]]:
    """The operation ``site`` calls, and the path parameters its URL supplies."""
    if site.kind == "google_client":
        op_id = site.google_method_id or ""
        op = spec.operation(op_id)
        if op is None:
            # The client library's own helpers on an operation: export_media()
            # is drive.files.export returning the file's bytes, and
            # list_next(previous_request, previous_response) is the next
            # page of the list call that request was.
            for suffix, applies in _CLIENT_HELPERS.items():
                base = spec.operation(op_id.removesuffix(suffix))
                if op_id.endswith(suffix) and base is not None and applies(base):
                    return base, {}
        return op, {}
    assert site.url is not None
    return best_operation(spec.operations, site.method, site.url)


#: Client-library method suffix -> which operations have it.
_CLIENT_HELPERS: dict[str, Callable[[Operation], bool]] = {
    "_media": lambda op: op.supports_media,
    "_next": lambda op: op.response_schema is not None,
}

#: Keyword arguments Google's client library takes on an operation that
#: supports media upload, which are not request parameters.
_UPLOAD_KWARGS = ("media_body", "media_mime_type")


def best_operation(
    operations: list[Operation], method: str | None, url: str
) -> tuple[Operation | None, dict[str, str]]:
    """The operation ``method`` ``url`` calls, and its path parameters.

    Of the operations whose URL (or upload URL) matches, the most specific
    one: People's ``v1/{+resourceName}/connections`` over
    ``v1/{+resourceName}``, which also matches that path.
    """
    call = path_segments(url)
    best: tuple[int, Operation, dict[str, str]] | None = None
    for op in operations:
        if method is not None and op.method != method:
            continue
        for template in op.templates():
            segments = path_segments(template)
            bound = match_path(segments, call)
            if bound is None:
                continue
            literal = len(re.sub(r"\{[^}]*\}", "", "/".join(segments)))
            if best is None or literal > best[0]:
                best = (literal, op, bound)
    return (best[1], best[2]) if best is not None else (None, {})


def match_path(template: list[str], call: list[str]) -> dict[str, str] | None:
    """Path parameter -> the call's value, or ``None`` when the paths differ.

    ``{name}`` is one segment, ``{+name}`` one or more (slashes included),
    and either may share a segment with literal text
    (``{spreadsheetId}:batchUpdate``, ``{+resourceName}:updateContact``):
    every parameter is bound, wherever it sits.
    """
    names: list[str] = []

    def group(match: re.Match[str]) -> str:
        names.append(match.group(2))
        return "(.+)" if match.group(1) else "([^/]+)"

    pattern = ""
    rest = "/".join(template)
    for i, piece in enumerate(re.split(r"(\{\+?[^}]+\})", rest)):
        if i % 2:
            pattern += re.sub(r"\{(\+?)([^}]+)\}", group, piece)
        else:
            pattern += re.escape(piece)
    found = re.fullmatch(pattern, "/".join(call))
    if found is None:
        return None
    return dict(zip(names, found.groups(), strict=True))


def _violations(
    site: CallSite, op: Operation, spec: ApiSpec, path_params: dict[str, str]
) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    sent: dict[str, object] = dict(site.params)
    complete = site.params_complete
    if site.kind == "google_client":
        if (site.google_method_id or "").endswith("_next") and site.google_method_id != op.id:
            # list_next(previous_request, previous_response) re-sends the
            # previous request's parameters with the next page token.
            sent, complete = {}, False
        for kwarg in _UPLOAD_KWARGS:
            if kwarg in sent:
                del sent[kwarg]
                if not op.supports_media_upload:
                    out.append(
                        (
                            "spec_unknown_param",
                            f"passes {kwarg}, but this operation takes no upload",
                        )
                    )
    if site.kind == "http" and site.url and "?" in site.url:
        for pair in site.url.split("?", 1)[1].split("&"):
            key, _, raw = pair.partition("=")
            if key:
                sent.setdefault(key, raw if "{" not in raw else UNKNOWN)
    allowed = {**spec.global_params, **op.params}
    for name, value in sent.items():
        param = allowed.get(name)
        if param is None or (site.kind == "http" and param.location == "path"):
            names = sorted(n for n, p in op.params.items() if p.location == "query")
            out.append(
                (
                    "spec_unknown_param",
                    f"sends parameter {name!r}, which the spec does not declare"
                    f"{' as a query parameter' if param is not None else ''}. "
                    f"Query parameters: {', '.join(names) or 'none'}",
                )
            )
            continue
        problem = _value_problem(param, value)
        if problem:
            out.append(("spec_param_value", f"parameter {name!r}: {problem}"))
    for name, param in op.params.items():
        if not param.required:
            continue
        given = (
            name in path_params
            if site.kind == "http" and param.location == "path"
            else (name in sent)
        )
        if not given and complete:
            out.append(("spec_missing_required", f"required parameter {name!r} is not sent"))
    if site.body_keys:
        request = spec.resolve(op.request_schema)
        props = (request or {}).get("properties")
        if props is not None and not (request or {}).get("additionalProperties"):
            for key in site.body_keys:
                if key not in props:
                    out.append(
                        (
                            "spec_unknown_body_field",
                            f"request body field {key!r} is not in the request schema",
                        )
                    )
    media = (
        op.media_download
        or (site.google_method_id or "").endswith("_media")
        or (op.supports_media and str(sent.get("alt")) == "media")
    )
    if op.response_schema is not None and not media:
        for path, line in site.response_reads:
            missing = _unknown_field(spec, op.response_schema, path)
            if missing is not None:
                out.append(
                    (
                        "spec_unknown_response_field",
                        f"line {line} reads {_dotted(path)}, but the response schema has no "
                        f"{missing!r} there",
                    )
                )
    elif media and site.response_reads:
        out.append(
            (
                "spec_unknown_response_field",
                f"reads fields off the response ({_dotted(site.response_reads[0][0])}), but this "
                f"operation returns the file's bytes, not JSON",
            )
        )
    return out


def _value_problem(param: ParamSpec, value: object) -> str | None:
    if value is UNKNOWN:
        return None
    values = value if isinstance(value, list | tuple) else [value]
    if isinstance(value, list | tuple) and not param.repeated:
        return "a list was sent, but the parameter is not repeated"
    for v in values:
        if param.enum is not None and isinstance(v, str) and v not in param.enum:
            return f"{v!r} is not one of {', '.join(param.enum)}"
        if param.type == "integer" and not (
            isinstance(v, int)
            and not isinstance(v, bool)
            or isinstance(v, str)
            and v.lstrip("-").isdigit()
        ):
            return f"{v!r} is not an integer"
        if param.type == "boolean" and not (
            isinstance(v, bool) or isinstance(v, str) and v in ("true", "false")
        ):
            return f"{v!r} is not a boolean"
    return None


def _unknown_field(spec: ApiSpec, schema: dict[str, Any], path: tuple[str, ...]) -> str | None:
    """The first key of ``path`` the schema does not have, or ``None`` if all are there."""
    node: dict[str, Any] | None = spec.resolve(schema)
    for key in path:
        if node is None:
            return None
        if key == "[]":
            node = spec.resolve(node.get("items"))
            continue
        props = node.get("properties")
        if props is None:
            extra = node.get("additionalProperties")
            node = spec.resolve(extra) if isinstance(extra, dict) else None
            continue
        if key not in props:
            return key
        node = spec.resolve(props[key])
    return None


def _dotted(path: tuple[str, ...]) -> str:
    return "".join("[]" if k == "[]" else f".{k}" for k in path).lstrip(".")


def _finding(node_id: str, rel: str, line: int, kind: str, message: str) -> ContractFinding:
    return ContractFinding(
        node_id=node_id,
        kind=kind,
        severity="error",
        message=f"{rel} {message} (check it against the vendor's spec, not memory)",
        path=rel,
    )


def report_section(checks: list[RequestCheck]) -> list[str]:
    """compile-report.md's "Vendor API spec checks" section."""
    if not checks:
        return ["No vendor request was found in the external_call modules."]
    lines = [
        "Each vendor request the code makes, checked in code against the vendor's "
        "machine-readable spec (method, path, parameters, request body and the "
        "response fields the code reads).",
        "",
        "| Where | Operation | Spec | Result |",
        "|---|---|---|---|",
    ]
    for c in checks:
        result = {
            "ok": "ok",
            "violations": f"**{len(c.findings)} violation(s)**",
            "unresolved": f"not checked: {c.note}",
            "spec_unavailable": f"not checked: {c.note}",
        }[c.status]
        lines.append(
            f"| `{c.path}:{c.line}` | {c.operation or ''} | {c.spec or ''} | "
            f"{result.replace('|', chr(92) + '|')} |"
        )
    for c in checks:
        for f in c.findings:
            lines.append(f"- {f.message}")
    return lines


__all__ = [
    "CheckStatus",
    "RequestCheck",
    "check_deliverable",
    "report_section",
    "spec_findings",
]
