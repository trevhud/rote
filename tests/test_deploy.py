"""Tests for ``rote deploy`` (target resolution, preflights, CLI wiring).

The vendor CLIs are faked at the subprocess layer — a real
``wrangler deploy --dry-run`` is exercised in the slow cloudflare
validation, not here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rote.cli import main as cli_main
from rote.deploy import (
    TARGET_BY_RUNTIME,
    DeployError,
    deploy_cloudflare,
    guidance_for,
    resolve_deploy_target,
)
from rote.runners import RunTarget

# ───────── Target resolution ─────────


def test_target_map_covers_every_adapter() -> None:
    from rote.adapters import ADAPTERS

    assert set(TARGET_BY_RUNTIME) == set(ADAPTERS)


@pytest.mark.parametrize(
    ("runtime", "expected"),
    [
        ("cloudflare", "cloudflare"),
        ("dbos", "dbos-cloud"),
        ("dbos-ts", "dbos-cloud"),
        ("temporal", None),
        ("inngest", None),
        ("python", None),
    ],
)
def test_auto_target_resolution(runtime: str, expected: str | None) -> None:
    assert resolve_deploy_target("auto", runtime) == expected
    assert resolve_deploy_target(None, runtime) == expected


def test_explicit_target_must_match_runtime() -> None:
    assert resolve_deploy_target("cloudflare", "cloudflare") == "cloudflare"
    with pytest.raises(DeployError, match="does not apply"):
        resolve_deploy_target("cloudflare", "dbos")
    with pytest.raises(DeployError, match="guidance-only"):
        resolve_deploy_target("dbos-cloud", "temporal")


def test_guidance_exists_for_every_guidance_runtime() -> None:
    for runtime, target in TARGET_BY_RUNTIME.items():
        if target is None:
            text = guidance_for(runtime)
            assert len(text) > 40, f"guidance for {runtime} should be substantive"
    # Hosted-platform runtimes must cite current doc URLs.
    assert "docs.temporal.io" in guidance_for("temporal")
    assert "inngest.com/docs" in guidance_for("inngest")


# ───────── Cloudflare wrapper preflight ─────────


def test_cloudflare_unauthenticated_is_actionable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import rote.deploy as dep

    monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)
    monkeypatch.setattr(dep.shutil, "which", lambda name: "/usr/bin/npx")

    class _Proc:
        returncode = 1
        stdout = ""
        stderr = "not authenticated"

    monkeypatch.setattr(dep.subprocess, "run", lambda *a, **k: _Proc())
    target = RunTarget(kind="pipeline", path=tmp_path, runtime="cloudflare")
    with pytest.raises(DeployError, match="wrangler login"):
        deploy_cloudflare(target)


def test_cloudflare_api_token_skips_whoami(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import rote.deploy as dep

    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "t")
    monkeypatch.setattr(dep.shutil, "which", lambda name: "/usr/bin/npx")
    calls: list[list[str]] = []

    class _Proc:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(args: list[str], **kwargs: Any) -> _Proc:
        calls.append(list(args))
        return _Proc()

    monkeypatch.setattr(dep.subprocess, "run", fake_run)
    target = RunTarget(kind="pipeline", path=tmp_path, runtime="cloudflare")
    report = deploy_cloudflare(target, dry_run=True)
    assert report.ok and report.action == "dry-run"
    assert calls == [["npx", "wrangler", "deploy", "--dry-run"]]


# ───────── CLI wiring ─────────


def _emitted(tmp_path: Path, *names: str) -> Path:
    app = tmp_path / "app"
    app.mkdir()
    for n in names:
        p = app / n
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("", encoding="utf-8")
    return app


def test_deploy_guidance_runtime_exits_0(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    app = _emitted(tmp_path, "workflow.py", "activities.py")
    rc = cli_main(["deploy", str(app), "--json"])
    captured = capsys.readouterr()
    assert rc == 0
    lines = captured.out.splitlines()
    start = max(i for i, line in enumerate(lines) if line == "{")
    payload = json.loads("\n".join(lines[start:]))
    assert payload["action"] == "guidance"
    assert "workers are always" in captured.out


def test_deploy_skill_dir_is_rejected(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("# s", encoding="utf-8")
    rc = cli_main(["deploy", str(skill)])
    assert rc == 2
    assert "compile it first" in capsys.readouterr().err


def test_deploy_dispatches_cloudflare_wrapper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from rote.deploy import DeployReport

    app = _emitted(tmp_path, "wrangler.jsonc", "package.json", "src/workflow.ts")
    captured: dict[str, Any] = {}

    def fake_deploy(target: Any, *, dry_run: bool = False, extra_args: Any = None) -> DeployReport:
        captured["dry_run"] = dry_run
        return DeployReport(
            target="cloudflare",
            runtime="cloudflare",
            app_dir=target.path,
            ok=True,
            action="dry-run" if dry_run else "deployed",
        )

    monkeypatch.setattr("rote.deploy.deploy_cloudflare", fake_deploy)
    rc = cli_main(["deploy", str(app), "--dry-run", "--json"])
    out = capsys.readouterr()
    assert rc == 0
    assert captured["dry_run"] is True
    payload = json.loads(out.out)
    assert payload["target"] == "cloudflare"
    assert payload["action"] == "dry-run"


def test_deploy_mismatched_target_exits_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    app = _emitted(tmp_path, "main.py", "dbos-config.yaml")
    rc = cli_main(["deploy", str(app), "--target", "cloudflare"])
    assert rc == 1
    assert "does not apply" in capsys.readouterr().err


# ───────── rote-cloud client ─────────


def test_rote_cloud_target_requires_cloudflare_runtime() -> None:
    assert resolve_deploy_target("rote-cloud", "cloudflare") == "rote-cloud"
    with pytest.raises(DeployError, match="runtime cloudflare"):
        resolve_deploy_target("rote-cloud", "dbos")


def test_rote_cloud_endpoint_and_token_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    from rote.deploy_rote_cloud import resolve_endpoint, resolve_token

    monkeypatch.delenv("ROTE_CLOUD_URL", raising=False)
    monkeypatch.delenv("ROTE_CLOUD_TOKEN", raising=False)
    with pytest.raises(DeployError, match="rote login"):
        resolve_endpoint(None)
    with pytest.raises(DeployError, match="rote login"):
        resolve_token(None)
    monkeypatch.setenv("ROTE_CLOUD_URL", "http://x:1/")
    monkeypatch.setenv("ROTE_CLOUD_TOKEN", "rote_t")
    assert resolve_endpoint(None) == "http://x:1"
    assert resolve_token(None) == "rote_t"
    assert resolve_endpoint("http://flag:2/") == "http://flag:2"


def test_stored_login_feeds_deploy_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    """flag > env > stored login — the stored credential is the quiet default."""
    from rote.cloud_auth import CloudCredential, save_credential
    from rote.deploy_rote_cloud import resolve_endpoint, resolve_token

    monkeypatch.delenv("ROTE_CLOUD_URL", raising=False)
    monkeypatch.delenv("ROTE_CLOUD_TOKEN", raising=False)
    save_credential(CloudCredential(url="http://stored:3/", token="rote_stored", user="t@x"))
    assert resolve_endpoint(None) == "http://stored:3"
    assert resolve_token(None) == "rote_stored"
    monkeypatch.setenv("ROTE_CLOUD_TOKEN", "rote_env")
    assert resolve_token(None) == "rote_env"
    assert resolve_token("rote_flag") == "rote_flag"


def test_rote_cloud_manifest_required(tmp_path: Path) -> None:
    from rote.deploy_rote_cloud import load_manifest

    with pytest.raises(DeployError, match="manifest.json"):
        load_manifest(tmp_path)
    (tmp_path / "manifest.json").write_text('{"name": "p"}', encoding="utf-8")
    with pytest.raises(DeployError, match="class_name"):
        load_manifest(tmp_path)


def test_bundle_installs_npm_deps_first(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A freshly emitted app (the compile auto-deploy path) has no
    node_modules; the bundle step must install so judge deps (zod,
    vendor SDKs) can be inlined — found live when esbuild failed to
    resolve @anthropic-ai/sdk on a clean emit."""
    import rote.deploy_rote_cloud as rc

    (tmp_path / "src").mkdir(parents=True)
    (tmp_path / "src" / "workflow.ts").write_text("export class X {}", encoding="utf-8")
    installed: list[Path] = []
    monkeypatch.setattr(
        "rote.runners._node.ensure_npm_install", lambda app_dir: installed.append(app_dir)
    )
    monkeypatch.setattr(rc.shutil, "which", lambda name: "/usr/bin/npx")

    class _Proc:
        returncode = 0
        stdout = "export class X {}"
        stderr = ""

    monkeypatch.setattr(rc.subprocess, "run", lambda *a, **k: _Proc())
    assert rc.bundle_workflow(tmp_path) == "export class X {}"
    assert installed == [tmp_path]


def test_rote_cloud_upload_payload_shape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import rote.deploy_rote_cloud as rc

    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "name": "word-lookup",
                "version": "0.1.0",
                "pipeline_hash": "abc",
                "class_name": "WordLookupWorkflow",
                "node_ids": ["lookup_word"],
                "input_schema": {},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(rc, "bundle_workflow", lambda app_dir: "export class X {}")
    sent: dict[str, Any] = {}

    class _Resp:
        def __enter__(self) -> _Resp:
            return self

        def __exit__(self, *a: Any) -> None:
            return None

        def read(self) -> bytes:
            return b'{"ok": true, "pipeline_id": "p1"}'

    def fake_urlopen(req: Any, timeout: float = 0) -> _Resp:
        sent["url"] = req.full_url
        sent["auth"] = req.headers.get("Authorization")
        sent["payload"] = json.loads(req.data.decode())
        return _Resp()

    monkeypatch.setattr(rc, "urlopen", fake_urlopen)
    target = RunTarget(kind="pipeline", path=tmp_path, runtime="cloudflare")
    report = rc.deploy_rote_cloud(
        target, url="http://127.0.0.1:8787", token="rote_tok", input_example={"word": "hi"}
    )
    assert report.ok and report.target == "rote-cloud"
    assert sent["url"] == "http://127.0.0.1:8787/v1/pipelines"
    assert sent["auth"] == "Bearer rote_tok"
    assert sent["payload"]["class_name"] == "WordLookupWorkflow"
    assert sent["payload"]["module_js"] == "export class X {}"
    assert sent["payload"]["input_schema"] == {"examples": [{"word": "hi"}]}
    assert "pipeline_id" in report.detail
