"""Deploy emitted pipelines to their hosting targets (``rote deploy``).

Only some runtimes have a true push-deploy, and pretending otherwise
would be worse than useless:

* **cloudflare** → ``npx wrangler deploy`` (a Workflow is just a Worker
  with a ``workflows`` binding — no separate deploy path; verified
  against the July 2026 Workflows guide).
* **dbos / dbos-ts** → ``npx dbos-cloud app deploy`` (DBOS Cloud's CLI
  is a Node package even for Python apps; current docs have no separate
  ``app register`` step — ``app deploy`` handles first deploy and
  updates).
* **rote-cloud** → upload the bundled module to a rote-cloud instance
  (implemented with the rote-cloud client work).
* **temporal / inngest / python** → there is nothing to push. Temporal
  Cloud hosts only the server (workers are always self-hosted); Inngest
  reads your own hosted serve endpoint (deploy the host, then sync);
  the python target is a plain script. These print honest, current
  guidance with doc URLs instead of a fake action.

Wrapper philosophy: the vendor CLI owns auth, output, and errors — rote
adds target detection, preflights with actionable messages, and streams
the vendor CLI's output untouched. Gate on exit codes, never scrape
human-readable output (wrangler has no ``--json`` deploy mode).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from rote.runners import RunTarget

DEPLOY_TIMEOUT_SECONDS = 900.0


class DeployError(RuntimeError):
    """Preflight or invocation failure; the message carries the fix."""


@dataclass(frozen=True)
class DeployReport:
    """What happened, machine-readable for ``--json``."""

    target: str
    runtime: str
    app_dir: Path
    ok: bool
    action: str
    """``deployed`` / ``dry-run`` / ``guidance``."""
    detail: str = ""


#: runtime → push-deploy target (None = guidance only).
TARGET_BY_RUNTIME: dict[str, str | None] = {
    "cloudflare": "cloudflare",
    "dbos": "dbos-cloud",
    "dbos-ts": "dbos-cloud",
    "temporal": None,
    "inngest": None,
    "python": None,
}

_GUIDANCE = {
    "temporal": (
        "Temporal has no artifact push-deploy: Temporal Cloud (or your\n"
        "self-hosted cluster) runs only the *server* — workers are always\n"
        "self-hosted. Containerize the emitted worker and run it on any\n"
        "container platform (Kubernetes is the canonical path), with your\n"
        "Temporal address, namespace, and mTLS/API-key credentials in the\n"
        "environment. Use Worker Versioning for safe rollouts.\n"
        "Docs: https://docs.temporal.io/production-deployment/worker-deployments"
    ),
    "inngest": (
        "Inngest reads functions from a serve endpoint *you* host: deploy\n"
        "this app to your Node host (Vercel/Fly/anything), then sync so\n"
        "Inngest Cloud re-reads the function list:\n"
        '  curl -X POST "https://api.inngest.com/v2/apps/$APP_ID/syncs" \\\n'
        '    -H "Authorization: Bearer $INNGEST_API_KEY" \\\n'
        '    -d \'{"url": "https://<your-host>/api/inngest"}\'\n'
        "Set INNGEST_SIGNING_KEY and INNGEST_EVENT_KEY on the host. The\n"
        "Vercel integration syncs automatically on every deploy. Note: the\n"
        "new code must be live on the host *before* syncing.\n"
        "Docs: https://www.inngest.com/docs/platform/deployment"
    ),
    "python": (
        "The python target is a plain script — run it anywhere Python runs\n"
        "(cron, systemd, a container). It has no durable state: for crash\n"
        "recovery or HITL gates, re-emit onto a durable runtime\n"
        "(`rote emit --runtime dbos`)."
    ),
}


def resolve_deploy_target(target_flag: str | None, runtime: str) -> str | None:
    """The deploy target for an emitted runtime (explicit flag wins)."""
    if target_flag is not None and target_flag != "auto":
        if target_flag == "rote-cloud":
            # The platform's Worker Loader executes bundled cloudflare
            # workflow modules — other runtimes' output can't run there.
            if runtime != "cloudflare":
                raise DeployError(
                    "rote-cloud runs bundled Cloudflare workflow modules — emit "
                    "with `--runtime cloudflare` first"
                )
            return "rote-cloud"
        expected = TARGET_BY_RUNTIME.get(runtime)
        if expected != target_flag:
            raise DeployError(
                f"--target {target_flag} does not apply to an emitted `{runtime}` app"
                + (f" (its push target is {expected})" if expected else " (guidance-only runtime)")
            )
        return target_flag
    return TARGET_BY_RUNTIME.get(runtime)


def guidance_for(runtime: str) -> str:
    return _GUIDANCE.get(runtime, "see the emitted README.md")


def _require_npx() -> None:
    if shutil.which("npx") is None:
        raise DeployError("`npx` not found — a Node toolchain is required for this deploy target")


def _stream(args: list[str], cwd: Path) -> int:
    """Run a vendor CLI with output streaming straight to the terminal."""
    proc = subprocess.run(args, cwd=cwd, timeout=DEPLOY_TIMEOUT_SECONDS)
    return proc.returncode


def deploy_cloudflare(
    target: RunTarget, *, dry_run: bool = False, extra_args: list[str] | None = None
) -> DeployReport:
    """``npx wrangler deploy`` on the emitted app dir.

    Preflight: authenticated either by ``CLOUDFLARE_API_TOKEN`` (with
    ``CLOUDFLARE_ACCOUNT_ID`` when the token spans accounts) or a
    ``wrangler login`` session — ``wrangler whoami`` exits non-zero when
    neither is present. The account line is surfaced before deploying so
    a wrong-account deploy is visible before it happens.
    """
    _require_npx()
    if not os.environ.get("CLOUDFLARE_API_TOKEN"):
        whoami = subprocess.run(
            ["npx", "wrangler", "whoami"],
            cwd=target.path,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if whoami.returncode != 0:
            raise DeployError(
                "wrangler is not authenticated — run `npx wrangler login` or set "
                "CLOUDFLARE_API_TOKEN (+ CLOUDFLARE_ACCOUNT_ID)"
            )
        account = next(
            (line for line in whoami.stdout.splitlines() if "@" in line or "Account" in line),
            "",
        )
        if account:
            print(f"rote deploy: wrangler session: {account.strip()}", file=sys.stderr)
    args = ["npx", "wrangler", "deploy"]
    if dry_run:
        args.append("--dry-run")
    args += extra_args or []
    code = _stream(args, target.path)
    return DeployReport(
        target="cloudflare",
        runtime=target.runtime or "cloudflare",
        app_dir=target.path,
        ok=code == 0,
        action="dry-run" if dry_run else "deployed",
        detail="" if code == 0 else f"wrangler deploy exited {code}",
    )


def deploy_dbos_cloud(target: RunTarget, *, extra_args: list[str] | None = None) -> DeployReport:
    """``npx dbos-cloud app deploy`` on the emitted app dir.

    DBOS Cloud is Postgres-only and expects the app to listen on port
    8000; interactive login opens a browser portal (CI uses
    ``dbos-cloud login --with-refresh-token``). The CLI owns all of
    that — rote only preflights the obvious.
    """
    _require_npx()
    if target.runtime == "dbos" and not (target.path / "requirements.txt").is_file():
        print(
            "rote deploy: warning — no requirements.txt; DBOS Cloud needs one "
            "(pip freeze > requirements.txt)",
            file=sys.stderr,
        )
    args = ["npx", "dbos-cloud", "app", "deploy", *(extra_args or [])]
    code = _stream(args, target.path)
    return DeployReport(
        target="dbos-cloud",
        runtime=target.runtime or "dbos",
        app_dir=target.path,
        ok=code == 0,
        action="deployed",
        detail="" if code == 0 else f"dbos-cloud app deploy exited {code}",
    )
