"""The rote-cloud CLI credential (``~/.local/share/rote/cloud.json``).

``rote login`` stores one credential — the platform URL, a tenant
``rote_…`` API key, and a display label for ``rote whoami`` — and every
cloud-touching command (``rote deploy --target rote-cloud``, the
graduate auto-deploy default) resolves it with the same precedence:

    explicit flag  >  ROTE_CLOUD_URL / ROTE_CLOUD_TOKEN  >  stored login

The file is mode ``0600`` and written atomically (tempfile + rename),
matching the MCP token store (`rote.mcp.tokens`). Override the location
with ``ROTE_CLOUD_CRED_PATH`` (tests rely on this for isolation).

The login flow lives here too. Both UXes ride one grant path — the
OAuth 2.0 device-authorization flow (RFC 8628) served by the platform's
better-auth ``deviceAuthorization`` plugin:

1. ``POST /api/auth/device/code`` → user code + verification URLs.
2. When a browser is available, open ``verification_uri_complete``
   (code pre-filled, one Approve click — wrangler-style UX); headless
   sessions get the printed code + URL (gh-style). Same grant either
   way, so there is no localhost listener and no second server-side
   mechanism to secure.
3. Poll ``POST /api/auth/device/token`` per the RFC contract
   (``authorization_pending`` / ``slow_down`` / terminal errors).
4. Trade the short-lived session token for a durable tenant API key at
   ``POST /v1/cli/keys`` — the CLI only ever persists a ``rote_…`` key,
   which the platform's ``/v1/*`` gate already trusts.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import tempfile
import time
import webbrowser
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

#: The hosted platform. ``rote login --url`` targets a dev instance.
DEFAULT_CLOUD_URL = "https://app.roteskills.com"

#: OAuth client identifier the platform's device flow accepts.
CLIENT_ID = "rote-cli"

DEVICE_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:device_code"


class LoginError(RuntimeError):
    """Device-flow or key-mint failure; the message carries the fix."""


@dataclass(frozen=True)
class CloudCredential:
    """One rote-cloud login, as stored on disk."""

    url: str
    token: str
    user: str = ""
    """Display label for ``rote whoami`` (email or tenant name)."""
    created_at: str = ""
    """ISO-8601 stamp from the platform, informational only."""


def credential_path() -> Path:
    override = os.environ.get("ROTE_CLOUD_CRED_PATH")
    if override:
        return Path(override)
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg) if xdg else Path.home() / ".local" / "share"
    return base / "rote" / "cloud.json"


def load_credential() -> CloudCredential | None:
    path = credential_path()
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    url = doc.get("url")
    token = doc.get("token")
    if not url or not token:
        return None
    return CloudCredential(
        url=str(url).rstrip("/"),
        token=str(token),
        user=str(doc.get("user", "")),
        created_at=str(doc.get("created_at", "")),
    )


def save_credential(cred: CloudCredential) -> Path:
    path = credential_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=".cloud-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(asdict(cred), f, indent=2)
            f.write("\n")
        os.chmod(tmp_name, 0o600)
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise
    return path


def clear_credential() -> bool:
    path = credential_path()
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    return True


# ───────────────────────── device-flow login ─────────────────────────


def _request_json(
    method: str,
    url: str,
    body: dict[str, Any] | None = None,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> tuple[int, dict[str, Any]]:
    """(status, parsed body) — 4xx bodies are data here, not exceptions.

    The RFC 8628 polling contract delivers ``authorization_pending`` /
    ``slow_down`` as HTTP 400 JSON, so the transport layer must hand
    those back for interpretation rather than raising.
    """
    req = Request(
        url,
        data=json.dumps(body).encode("utf-8") if body is not None else None,
        headers={"Content-Type": "application/json", **(headers or {})},
        method=method,
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8") or "{}")
    except HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, {"error": f"http_{e.code}", "error_description": raw[:300]}
    except URLError as e:
        raise LoginError(f"could not reach {url}: {e.reason}") from e


def start_device_flow(base_url: str) -> dict[str, Any]:
    status, doc = _request_json(
        "POST", f"{base_url}/api/auth/device/code", {"client_id": CLIENT_ID}
    )
    if status != 200 or "device_code" not in doc:
        raise LoginError(
            f"device authorization rejected (HTTP {status}): "
            f"{doc.get('error_description') or doc.get('error') or doc}"
        )
    return doc


def poll_for_token(
    base_url: str,
    device_code: str,
    *,
    interval: float,
    expires_in: float,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """Poll the token endpoint until approval; returns the session token.

    ``slow_down`` raises the interval by 5s and holds the new floor, per
    the RFC. The loop budget is the code's own lifetime — when it runs
    out the grant is dead and only a fresh ``rote login`` can recover.
    """
    deadline = time.monotonic() + expires_in
    while time.monotonic() < deadline:
        sleep(interval)
        status, doc = _request_json(
            "POST",
            f"{base_url}/api/auth/device/token",
            {"grant_type": DEVICE_GRANT_TYPE, "device_code": device_code, "client_id": CLIENT_ID},
        )
        if status == 200 and doc.get("access_token"):
            return str(doc["access_token"])
        error = doc.get("error", "")
        if error == "authorization_pending":
            continue
        if error == "slow_down":
            interval += 5
            continue
        if error == "access_denied":
            raise LoginError("login denied in the browser")
        if error in ("expired_token", "invalid_grant"):
            raise LoginError("the device code expired — run `rote login` again")
        raise LoginError(
            f"token polling failed (HTTP {status}): {doc.get('error_description') or error or doc}"
        )
    raise LoginError("timed out waiting for browser approval — run `rote login` again")


def mint_key(base_url: str, access_token: str) -> dict[str, Any]:
    status, doc = _request_json(
        "POST",
        f"{base_url}/v1/cli/keys",
        {},
        headers={"Authorization": f"Bearer {access_token}"},
    )
    if status != 200 or not doc.get("key"):
        raise LoginError(
            f"the platform did not mint an API key (HTTP {status}): {doc.get('error') or doc}"
        )
    return doc


def login(
    base_url: str | None = None,
    *,
    open_browser: bool = True,
    echo: Callable[[str], None] | None = None,
) -> CloudCredential:
    """Run the full device-flow login and persist the credential."""
    url = (base_url or DEFAULT_CLOUD_URL).rstrip("/")
    say = echo or (lambda line: print(line, file=sys.stderr))
    grant = start_device_flow(url)
    verify_url = grant.get("verification_uri_complete") or grant.get("verification_uri") or url
    if not str(verify_url).startswith(("http://", "https://")):
        verify_url = f"{url}{verify_url}"
    say(f"rote login: confirm code {grant['user_code']} at:")
    say(f"  {verify_url}")
    if open_browser and webbrowser.open(str(verify_url)):
        say("  (opened in your browser — waiting for approval…)")
    else:
        say("  (waiting for approval…)")
    token = poll_for_token(
        url,
        str(grant["device_code"]),
        interval=float(grant.get("interval", 5)),
        expires_in=float(grant.get("expires_in", 1800)),
    )
    minted = mint_key(url, token)
    cred = CloudCredential(
        url=url,
        token=str(minted["key"]),
        user=str(minted.get("user", "")),
        created_at=str(minted.get("created_at", "")),
    )
    path = save_credential(cred)
    label = f" as {cred.user}" if cred.user else ""
    say(f"rote login: ✓ logged in{label} ({path})")
    return cred


def fetch_me(cred: CloudCredential) -> dict[str, Any]:
    """Live credential check against ``GET /v1/me``."""
    status, doc = _request_json(
        "GET",
        f"{cred.url}/v1/me",
        headers={"Authorization": f"Bearer {cred.token}"},
    )
    if status != 200:
        raise LoginError(
            f"credential rejected by {cred.url} (HTTP {status}) — run `rote login` again"
        )
    return doc


def revoke_key(cred: CloudCredential) -> bool:
    """Revoke the stored key server-side (``rote logout``)."""
    status, doc = _request_json(
        "DELETE",
        f"{cred.url}/v1/cli/keys",
        headers={"Authorization": f"Bearer {cred.token}"},
    )
    if status != 200:
        raise LoginError(f"could not revoke the key (HTTP {status}): {doc.get('error') or doc}")
    return bool(doc.get("revoked", True))
