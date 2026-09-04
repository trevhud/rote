"""Upload an emitted Cloudflare pipeline to a rote-cloud instance.

``rote deploy --target rote-cloud`` on an emitted cloudflare app:
bundle ``src/workflow.ts`` into a single ESM module (the platform's
Worker Loader executes JS, not the emitted TS), derive the pipeline's
identity from the emitted ``manifest.json``, and POST the deploy
payload to the platform's ``/v1/pipelines`` endpoint with a tenant
bearer token.

The endpoint contract (mirrors the platform's ``DeployPayload``):
``name`` + ``class_name`` + ``module_js`` required; ``version``,
``pipeline_hash``, ``node_ids``, ``input_schema``, ``mcp_servers`` ride along. The
platform stores the module in R2 keyed by tenant and registers the
pipeline row — the same code path its own in-cloud compilation deploy
uses.

Bundling uses esbuild's neutral ESM target with worker-first export
conditions, explicit ``module,main`` package entry fields, and
``cloudflare:*`` / ``node:*`` externals. esbuild is invoked via
``npx`` so the only requirement is a Node toolchain — no Python-side
JS dependency.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from rote.cloud_auth import USER_AGENT, load_credential
from rote.deploy import DeployError, DeployReport
from rote.runners import RunTarget

#: Same major the platform's reference bundler uses.
ESBUILD_SPEC = "esbuild@0.25"

BUNDLE_ARGS = [
    "--bundle",
    "--format=esm",
    "--platform=neutral",
    "--conditions=worker,browser,import",
    # Neutral has no default entry fields; packages without exports
    # (including @cloudflare/ai-utils) still need their module/main entry.
    "--main-fields=module,main",
    "--external:cloudflare:*",
    "--external:node:*",
    "--target=es2022",
]


def resolve_endpoint(url_flag: str | None) -> str:
    url = url_flag or os.environ.get("ROTE_CLOUD_URL")
    if not url:
        cred = load_credential()
        if cred is not None:
            return cred.url
        raise DeployError(
            "no rote-cloud endpoint: run `rote login`, pass --url, or set "
            "ROTE_CLOUD_URL (local dev: http://127.0.0.1:8787)"
        )
    return url.rstrip("/")


def resolve_token(token_flag: str | None) -> str:
    token = token_flag or os.environ.get("ROTE_CLOUD_TOKEN")
    if not token:
        cred = load_credential()
        if cred is not None:
            return cred.token
        raise DeployError("no rote-cloud credential: run `rote login` (or pass --token)")
    return token


def load_manifest(app_dir: Path) -> dict[str, Any]:
    manifest_path = app_dir / "manifest.json"
    if not manifest_path.is_file():
        raise DeployError(
            f"{app_dir} has no manifest.json — re-emit with a current rote "
            "(`rote emit --runtime cloudflare`) to get the deploy descriptor"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for field in ("name", "class_name"):
        if not manifest.get(field):
            raise DeployError(f"manifest.json is missing `{field}` — re-emit the app")
    return dict(manifest)


def bundle_workflow(app_dir: Path) -> str:
    """esbuild ``src/workflow.ts`` → one ESM module string.

    The app's npm dependencies must be present first: the platform's
    Worker Loader executes the bundle stand-alone, so judge deps (zod,
    vendor SDKs) get inlined — only ``cloudflare:*``/``node:*``
    stay external. A freshly emitted app (the compile auto-deploy
    path) has no ``node_modules`` yet, so install idempotently here.
    """
    if shutil.which("npx") is None:
        raise DeployError("`npx` not found — a Node toolchain is required to bundle the module")
    entry = app_dir / "src" / "workflow.ts"
    if not entry.is_file():
        raise DeployError(f"{entry} not found — pass an emitted cloudflare app dir")
    from rote.eval.empirical import EmpiricalError
    from rote.runners._node import ensure_npm_install

    try:
        ensure_npm_install(app_dir)
    except EmpiricalError as e:
        raise DeployError(str(e)) from e
    proc = subprocess.run(
        ["npx", "-y", ESBUILD_SPEC, str(entry), *BUNDLE_ARGS],
        cwd=app_dir,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if proc.returncode != 0:
        raise DeployError(f"esbuild bundle failed:\n{proc.stderr[:800]}")
    return proc.stdout


def deploy_rote_cloud(
    target: RunTarget,
    *,
    url: str | None = None,
    token: str | None = None,
    input_example: dict[str, Any] | None = None,
) -> DeployReport:
    """Bundle the emitted app and upload it to a rote-cloud instance."""
    endpoint = resolve_endpoint(url)
    bearer = resolve_token(token)
    manifest = load_manifest(target.path)
    module_js = bundle_workflow(target.path)

    input_schema: dict[str, Any] = dict(manifest.get("input_schema") or {})
    if input_example is not None:
        input_schema.setdefault("examples", []).append(input_example)

    payload = {
        "name": manifest["name"],
        "version": manifest.get("version", "0.0.0"),
        "pipeline_hash": manifest.get("pipeline_hash", ""),
        "class_name": manifest["class_name"],
        "node_ids": manifest.get("node_ids", []),
        "input_schema": input_schema,
        "mcp_servers": manifest.get("mcp_servers"),
        "module_js": module_js,
    }
    print(
        f"rote deploy: uploading {payload['name']} v{payload['version']} "
        f"(class {payload['class_name']}, {len(payload['node_ids'])} nodes, "
        f"{len(module_js) / 1024:.0f} KB) → {endpoint}",
        file=sys.stderr,
    )
    req = Request(
        f"{endpoint}/v1/pipelines",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {bearer}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        },
        method="POST",
    )
    try:
        with urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:500]
        raise DeployError(f"upload failed: HTTP {e.code}: {detail}") from e
    except URLError as e:
        raise DeployError(f"could not reach {endpoint}: {e.reason}") from e

    return DeployReport(
        target="rote-cloud",
        runtime=target.runtime or "cloudflare",
        app_dir=target.path,
        ok=True,
        action="deployed",
        detail=json.dumps(body),
    )
