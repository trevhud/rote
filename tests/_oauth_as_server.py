"""A switchable OAuth authorization server + protected MCP server, for tests.

Run as a script (the e2e suites start it as a subprocess)::

    python tests/_oauth_as_server.py <http_port> <tls_port> <cimd_document_path> <ca_pem_out>

One process serves three things:

- **Two authorization servers** at ``http://127.0.0.1:<http_port>/as-a`` and
  ``.../as-b`` (path issuers, so RFC 8414 path-aware discovery is real).
  Each advertises CIMD and/or DCR according to the current mode, and can
  answer the authorization redirect with a correct, mismatched, or missing
  RFC 9207 ``iss``.
- **A protected MCP server** at ``/mcp`` whose protected-resource metadata
  names the *active* authorization server. A request without a token that
  the active server issued gets a 401 with ``resource_metadata``.
- **A client metadata document host** over TLS at
  ``https://127.0.0.1:<tls_port>/oauth/rote-cli.json``, serving the file it
  was given (or a 404, or a tampered copy, per ``cimd_document``). The SDK
  only accepts an HTTPS ``client_metadata_url``, so the document is served
  over a local ``trustme`` CA, written to ``<ca_pem_out>`` so the test can
  make rote's own pre-flight fetch trust it, rather than relaxing that check.

The authorization servers are strict where a real one is: a CIMD client's
redirect URI must appear in the fetched document by exact string match
(CIMD draft section 4.2), a DCR or pre-registered client's must match its
registration exactly, and PKCE S256 is verified at the token endpoint.

Test control, plain JSON over HTTP:

- ``POST /_control`` sets any of ``cimd``, ``dcr`` (bools), ``iss``
  (``"correct"``, ``"mismatch"``, ``"omit"``), ``advertise_iss`` (bool),
  ``active`` (``"a"`` or ``"b"``), ``preregistered`` (a map of client id to
  ``{"redirect_uris": [...], "client_secret": ...}``), ``cimd_document``
  (``"served"``, ``"missing"``, ``"tampered"``, ``"dribble"``), the
  protected-resource metadata knobs ``prm_path`` (serve the path-specific
  document), ``prm_origin_resource`` and ``prm_issuer`` (override what the
  documents say), and ``reset`` (clears
  issued tokens, registrations, and the event log).
- ``GET /_events`` returns every authorization-server hit since the last
  reset, so a test can assert what did *not* happen (no token request after
  a bad ``iss``, no registration when CIMD was used).
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import secrets
import sys
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx
import trustme
import uvicorn
from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import (
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from starlette.types import Receive, Scope, Send

HTTP_PORT = int(sys.argv[1])
TLS_PORT = int(sys.argv[2])
CIMD_DOCUMENT = Path(sys.argv[3])
BASE = f"http://127.0.0.1:{HTTP_PORT}"
RESOURCE = f"{BASE}/mcp"

STATE: dict[str, Any] = {
    "cimd": True,
    "dcr": True,
    "iss": "correct",
    "advertise_iss": True,
    "active": "a",
    "preregistered": {},
    "cimd_document": "served",
    "prm_path": True,
    "prm_origin_resource": None,
    "prm_issuer": None,
}
REGISTERED: dict[str, dict[str, Any]] = {}  # "<as>:<client_id>" -> registration
CODES: dict[str, dict[str, Any]] = {}
TOKENS: dict[str, dict[str, Any]] = {}  # access or refresh token -> grant
EVENTS: list[dict[str, Any]] = []

_ca = trustme.CA()
_tmp = Path(tempfile.mkdtemp(prefix="rote-oauth-fixture-"))
CA_PEM = Path(sys.argv[4])
_ca.cert_pem.write_to_path(str(CA_PEM))
_leaf = _ca.issue_cert("127.0.0.1", "localhost")
LEAF_PEM = _tmp / "leaf.pem"
_leaf.private_key_and_cert_chain_pem.write_to_path(str(LEAF_PEM))


def issuer(name: str) -> str:
    return f"{BASE}/as-{name}"


def log(event: str, **fields: Any) -> None:
    EVENTS.append({"event": event, **fields})


def as_metadata(name: str) -> dict[str, Any]:
    iss = issuer(name)
    doc: dict[str, Any] = {
        "issuer": iss,
        "authorization_endpoint": f"{iss}/authorize",
        "token_endpoint": f"{iss}/token",
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code", "refresh_token"],
        "code_challenge_methods_supported": ["S256"],
        "token_endpoint_auth_methods_supported": ["none", "client_secret_post"],
    }
    if STATE["dcr"]:
        doc["registration_endpoint"] = f"{iss}/register"
    if STATE["cimd"]:
        doc["client_id_metadata_document_supported"] = True
    if STATE["advertise_iss"]:
        doc["authorization_response_iss_parameter_supported"] = True
    return doc


async def resolve_client(name: str, client_id: str) -> dict[str, Any] | None:
    """The client record the AS would trust for this client_id, or None."""
    pre = STATE["preregistered"].get(client_id)
    if pre is not None:
        return {"client_id": client_id, **pre}
    registered = REGISTERED.get(f"{name}:{client_id}")
    if registered is not None:
        return registered
    if STATE["cimd"] and client_id.startswith("https://"):
        async with httpx.AsyncClient(verify=str(CA_PEM)) as client:
            response = await client.get(client_id, headers={"x-fixture-as": "1"})
        log("cimd_fetch", issuer=issuer(name), url=client_id, status=response.status_code)
        if response.status_code != 200:
            return None
        doc = response.json()
        if doc.get("client_id") != client_id:
            return None
        return doc
    return None


def bad_request(error: str, description: str) -> JSONResponse:
    return JSONResponse({"error": error, "error_description": description}, status_code=400)


async def authorize(name: str, request: Request) -> Response:
    params = dict(request.query_params)
    client_id = params.get("client_id", "")
    redirect_uri = params.get("redirect_uri", "")
    log("authorize", issuer=issuer(name), client_id=client_id, redirect_uri=redirect_uri)
    client = await resolve_client(name, client_id)
    if client is None:
        return bad_request("invalid_client", f"unknown client {client_id!r}")
    if redirect_uri not in (client.get("redirect_uris") or []):
        return bad_request("invalid_request", f"redirect_uri {redirect_uri!r} is not registered")
    if params.get("code_challenge_method") != "S256" or not params.get("code_challenge"):
        return bad_request("invalid_request", "PKCE S256 is required")
    code = secrets.token_urlsafe(16)
    CODES[code] = {
        "as": name,
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "challenge": params["code_challenge"],
    }
    answer = {"code": code, "state": params.get("state", "")}
    if STATE["iss"] == "correct":
        answer["iss"] = issuer(name)
    elif STATE["iss"] == "mismatch":
        answer["iss"] = "https://attacker.example"
    return RedirectResponse(f"{redirect_uri}?{urlencode(answer)}", status_code=302)


def mint(name: str, client_id: str) -> dict[str, Any]:
    access = secrets.token_urlsafe(24)
    refresh = secrets.token_urlsafe(24)
    grant = {"as": name, "client_id": client_id}
    TOKENS[access] = {**grant, "kind": "access"}
    TOKENS[refresh] = {**grant, "kind": "refresh"}
    return {
        "access_token": access,
        "token_type": "Bearer",
        "expires_in": 3600,
        "refresh_token": refresh,
    }


async def token(name: str, request: Request) -> Response:
    form = dict(await request.form())
    grant_type = form.get("grant_type")
    client_id = str(form.get("client_id", ""))
    log("token", issuer=issuer(name), grant_type=grant_type, client_id=client_id)
    if grant_type == "authorization_code":
        record = CODES.pop(str(form.get("code", "")), None)
        if record is None or record["as"] != name or record["client_id"] != client_id:
            return bad_request("invalid_grant", "unknown code")
        if record["redirect_uri"] != form.get("redirect_uri"):
            return bad_request("invalid_grant", "redirect_uri mismatch")
        verifier = str(form.get("code_verifier", "")).encode()
        digest = base64.urlsafe_b64encode(hashlib.sha256(verifier).digest()).rstrip(b"=")
        if digest.decode() != record["challenge"]:
            return bad_request("invalid_grant", "PKCE verification failed")
        return JSONResponse(mint(name, client_id))
    if grant_type == "refresh_token":
        record = TOKENS.pop(str(form.get("refresh_token", "")), None)
        if record is None or record["kind"] != "refresh" or record["as"] != name:
            return bad_request("invalid_grant", "unknown refresh token")
        if record["client_id"] != client_id:
            return bad_request("invalid_grant", "refresh token belongs to another client")
        return JSONResponse(mint(name, client_id))
    return bad_request("unsupported_grant_type", str(grant_type))


async def register(name: str, request: Request) -> Response:
    body = await request.json()
    log("register", issuer=issuer(name), redirect_uris=body.get("redirect_uris"))
    if not STATE["dcr"]:
        return JSONResponse({"error": "registration disabled"}, status_code=404)
    client_id = f"dcr-{secrets.token_hex(6)}"
    record = {**body, "client_id": client_id, "client_id_issued_at": 0}
    REGISTERED[f"{name}:{client_id}"] = record
    return JSONResponse(record, status_code=201)


mcp = FastMCP("oauth-fixture")


@mcp.tool
def whoami() -> dict:
    return {"authenticated": True}


mcp_app = mcp.http_app(path="/mcp")


def unauthorized() -> Response:
    return Response(
        status_code=401,
        headers={
            "WWW-Authenticate": (
                f'Bearer resource_metadata="{BASE}/.well-known/oauth-protected-resource/mcp"'
            )
        },
    )


def bearer_is_valid(request: Request) -> bool:
    header = request.headers.get("authorization", "")
    if not header.lower().startswith("bearer "):
        return False
    grant = TOKENS.get(header[7:])
    return grant is not None and grant["kind"] == "access" and grant["as"] == STATE["active"]


async def control(request: Request) -> Response:
    body = await request.json()
    if body.pop("reset", False):
        REGISTERED.clear()
        CODES.clear()
        TOKENS.clear()
        EVENTS.clear()
    for key, value in body.items():
        if key not in STATE:
            return bad_request("invalid_request", f"unknown control key {key!r}")
        STATE[key] = value
    return JSONResponse(STATE)


async def http_app(scope: Scope, receive: Receive, send: Send) -> None:
    if scope["type"] == "lifespan":
        await mcp_app(scope, receive, send)
        return
    request = Request(scope, receive)
    path = scope["path"]
    response: Response | None = None
    if path == "/_control" and request.method == "POST":
        response = await control(request)
    elif path == "/_events":
        response = JSONResponse(EVENTS)
    elif path == "/.well-known/oauth-protected-resource/mcp" and STATE["prm_path"]:
        response = JSONResponse(
            {
                "resource": RESOURCE,
                "authorization_servers": [STATE["prm_issuer"] or issuer(STATE["active"])],
            }
        )
    elif path == "/.well-known/oauth-protected-resource":
        # The origin-level document. On a shared host it may describe a
        # different resource than /mcp (`prm_origin_resource`).
        response = JSONResponse(
            {
                "resource": STATE["prm_origin_resource"] or RESOURCE,
                "authorization_servers": [STATE["prm_issuer"] or issuer(STATE["active"])],
            }
        )
    elif path.startswith("/.well-known/oauth-authorization-server/as-"):
        response = JSONResponse(as_metadata(path.rsplit("-", 1)[1]))
    elif path.startswith("/as-"):
        name, _, endpoint = path[len("/as-") :].partition("/")
        if endpoint == "authorize":
            response = await authorize(name, request)
        elif endpoint == "token":
            response = await token(name, request)
        elif endpoint == "register":
            response = await register(name, request)
    elif path.startswith("/mcp"):
        if not bearer_is_valid(request):
            response = unauthorized()
        else:
            await mcp_app(scope, receive, send)
            return
    if response is None:
        # Logged so a test can prove a guessed endpoint (the SDK's
        # `<resource-origin>/register` fallback) was never requested.
        log("not_found", method=request.method, path=path)
        response = PlainTextResponse("not found", status_code=404)
    await response(scope, receive, send)


async def tls_app(scope: Scope, receive: Receive, send: Send) -> None:
    if scope["type"] != "http":
        return
    request = Request(scope, receive)
    mode = STATE["cimd_document"]
    if scope["path"] == "/oauth/rote-cli.json" and "x-fixture-as" not in request.headers:
        # The fetch rote makes itself before offering the document.
        log("document_check", mode=mode)
    if scope["path"] != "/oauth/rote-cli.json" or mode == "missing":
        response: Response = PlainTextResponse("not found", status_code=404)
    elif mode == "dribble":
        # Headers at once, then a byte every half second for a minute: every
        # read finishes inside any per-read timeout, the body never does.
        async def trickle() -> Any:
            for _ in range(120):
                yield b" "
                await asyncio.sleep(0.5)

        response = StreamingResponse(trickle(), media_type="application/json")
    elif mode == "tampered":
        doc = json.loads(CIMD_DOCUMENT.read_text(encoding="utf-8"))
        doc["redirect_uris"] = ["http://localhost:1/callback"]
        response = JSONResponse(doc)
    else:
        response = Response(CIMD_DOCUMENT.read_bytes(), media_type="application/json")
    await response(scope, receive, send)


async def main() -> None:
    servers = [
        uvicorn.Server(
            uvicorn.Config(http_app, host="127.0.0.1", port=HTTP_PORT, log_level="warning")
        ),
        uvicorn.Server(
            uvicorn.Config(
                tls_app,
                host="127.0.0.1",
                port=TLS_PORT,
                log_level="warning",
                lifespan="off",
                ssl_certfile=str(LEAF_PEM),
            )
        ),
    ]
    await asyncio.gather(*(s.serve() for s in servers))


if __name__ == "__main__":
    asyncio.run(main())
