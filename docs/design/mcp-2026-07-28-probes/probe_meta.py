"""Fetch public MCP auth discovery metadata (no authentication) and summarize."""

import json
import sys
import urllib.error
import urllib.request
from urllib.parse import urlparse

SERVERS = sys.argv[1:]


def get(url, method="GET", body=None, headers=None):
    req = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    req.add_header("User-Agent", "rote-metadata-probe/0.1")
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, dict(r.headers), r.read(200000)
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read(200000)
    except Exception as e:  # noqa: BLE001
        return None, {}, str(e).encode()


def jget(url):
    s, _, b = get(url)
    if s != 200:
        return None
    try:
        return json.loads(b)
    except Exception:  # noqa: BLE001
        return None


KEYS = [
    "issuer",
    "registration_endpoint",
    "client_id_metadata_document_supported",
    "authorization_response_iss_parameter_supported",
    "code_challenge_methods_supported",
    "token_endpoint_auth_methods_supported",
    "grant_types_supported",
]

for url in SERVERS:
    p = urlparse(url)
    base = f"{p.scheme}://{p.netloc}"
    out = {"server": url}
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}).encode()
    s, h, _ = get(
        url,
        "POST",
        body,
        {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": "2026-07-28",
        },
    )
    out["unauth_post_status"] = s
    out["www_authenticate"] = h.get("WWW-Authenticate") or h.get("www-authenticate")
    prm = None
    for cand in [
        f"{base}/.well-known/oauth-protected-resource{p.path.rstrip('/')}",
        f"{base}/.well-known/oauth-protected-resource",
    ]:
        prm = jget(cand)
        if prm:
            out["prm_url"] = cand
            break
    asurl = None
    if prm:
        out["prm_resource"] = prm.get("resource")
        out["prm_as"] = prm.get("authorization_servers")
        if prm.get("authorization_servers"):
            asurl = prm["authorization_servers"][0]
    if not asurl:
        asurl = base
    a = urlparse(asurl)
    asbase = f"{a.scheme}://{a.netloc}"
    apath = a.path.rstrip("/")
    meta = None
    for cand in [
        f"{asbase}/.well-known/oauth-authorization-server{apath}",
        f"{asbase}/.well-known/openid-configuration{apath}",
        f"{asurl.rstrip('/')}/.well-known/openid-configuration",
        f"{asbase}/.well-known/oauth-authorization-server",
    ]:
        meta = jget(cand)
        if meta:
            out["as_meta_url"] = cand
            break
    if meta:
        for k in KEYS:
            out[k] = meta.get(k, "<absent>")
    print(json.dumps(out, indent=1))
