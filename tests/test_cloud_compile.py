"""Tests for server-side ("cloud") compilation (``rote compile`` on rote cloud).

HTTP is faked at the ``rote.cloud_compile.urlopen`` seam the same way
``tests/test_deploy.py`` fakes ``rote.deploy_rote_cloud.urlopen``: a small
router maps ``(method, path)`` to scripted ``(status, body)`` responses
(4xx/5xx become ``HTTPError``, a callable can raise ``URLError`` /
``KeyboardInterrupt``). The credential store is redirected to a tmp file by
the autouse ``_isolated_mcp_state`` fixture in ``conftest.py``.
"""

from __future__ import annotations

import dataclasses
import io
import json
import urllib.parse
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError

import pytest

import rote.cloud_compile as cg
from rote.cli import main as cli_main
from rote.cloud_auth import CloudCredential, save_credential
from rote.cloud_compile import (
    ActiveCompilationExists,
    CloudCompileError,
    CloudEndpoint,
    NoChanges,
    _event_from_wire,
    download_artifacts,
    poll_compilation,
    resolve_model,
    start_compilation,
    sync_skill,
)
from rote.compiler.events import CompilationEvent

EP = CloudEndpoint(url="http://cloud", token="rote_t")

# ───────────────────────── HTTP fake ─────────────────────────

Response = tuple[int, dict[str, Any]]


class _Resp:
    def __init__(self, status: int, body: dict[str, Any]) -> None:
        self.status = status
        self._data = json.dumps(body).encode("utf-8")

    def __enter__(self) -> _Resp:
        return self

    def __exit__(self, *a: Any) -> None:
        return None

    def read(self) -> bytes:
        return self._data


class _Router:
    """Scripted HTTP by ``(method, path)``; multi-response routes advance."""

    def __init__(self) -> None:
        self.routes: dict[tuple[str, str], list[Any]] = {}
        self.calls: list[tuple[str, str, dict[str, Any] | None]] = []

    def add(self, method: str, path: str, *responses: Any) -> _Router:
        self.routes.setdefault((method, path), []).extend(responses)
        return self

    def install(self, monkeypatch: pytest.MonkeyPatch) -> _Router:
        def fake_urlopen(req: Any, timeout: float = 0) -> _Resp:
            method = req.get_method()
            url = req.full_url
            body = json.loads(req.data.decode()) if req.data else None
            self.calls.append((method, url, body))
            path = urllib.parse.urlsplit(url).path
            queue = self.routes.get((method, path))
            if not queue:
                raise AssertionError(f"unexpected request {method} {url}")
            resp = queue.pop(0) if len(queue) > 1 else queue[0]
            if callable(resp):
                resp = resp(method, url, body)
            status, payload = resp
            if status >= 400:
                raise HTTPError(
                    url, status, "err", {}, io.BytesIO(json.dumps(payload).encode("utf-8"))
                )
            return _Resp(status, payload)

        monkeypatch.setattr(cg, "urlopen", fake_urlopen)
        return self

    def paths(self) -> list[tuple[str, str]]:
        return [(m, urllib.parse.urlsplit(u).path) for m, u, _ in self.calls]


def _skill_dir(tmp_path: Path, name: str = "word-lookup", extra: bool = True) -> Path:
    skill = tmp_path / "skill"
    (skill / "references").mkdir(parents=True)
    (skill / "SKILL.md").write_text(f"---\nname: {name}\n---\n# {name}\n", encoding="utf-8")
    if extra:
        (skill / "references" / "notes.md").write_text("hello", encoding="utf-8")
        (skill / ".hidden").write_text("ignored", encoding="utf-8")
        (skill / "__pycache__").mkdir()
        (skill / "__pycache__" / "x.pyc").write_text("nope", encoding="utf-8")
    return skill


# ───────────────────────── mode selection (CLI dispatch) ─────────────────────────


class _LocalReached(Exception):
    """Raised by the patched Compiler so a test can prove the local path ran."""


@pytest.fixture
def patch_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cloud dispatch returns sentinel 77; the local path raises _LocalReached."""

    class _FakeCompiler:
        def __init__(self, **_kw: Any) -> None:
            raise _LocalReached

    monkeypatch.setattr("rote.cli._cmd_compile_cloud", lambda _args: 77)
    monkeypatch.setattr("rote.cli.Compiler", _FakeCompiler)


def _login() -> None:
    save_credential(CloudCredential(url="http://cloud", token="rote_t"))


def test_logged_out_runs_local(patch_dispatch: None, tmp_path: Path) -> None:
    skill = _skill_dir(tmp_path)
    with pytest.raises(_LocalReached):
        cli_main(["compile", str(skill), "--out", str(tmp_path / "o")])


def test_logged_in_runs_cloud(patch_dispatch: None, tmp_path: Path) -> None:
    _login()
    skill = _skill_dir(tmp_path)
    assert cli_main(["compile", str(skill), "--out", str(tmp_path / "o")]) == 77


def test_local_flag_forces_local_when_logged_in(patch_dispatch: None, tmp_path: Path) -> None:
    _login()
    skill = _skill_dir(tmp_path)
    with pytest.raises(_LocalReached):
        cli_main(["compile", str(skill), "--out", str(tmp_path / "o"), "--local"])


def test_no_deploy_forces_local_when_logged_in(patch_dispatch: None, tmp_path: Path) -> None:
    _login()
    skill = _skill_dir(tmp_path)
    with pytest.raises(_LocalReached):
        cli_main(["compile", str(skill), "--out", str(tmp_path / "o"), "--no-deploy"])


def _user_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, body: str) -> None:
    cfg = tmp_path / "user-config.yaml"
    cfg.write_text(body, encoding="utf-8")
    monkeypatch.setenv("ROTE_CONFIG_PATH", str(cfg))


def test_config_noncloudflare_runtime_forces_local(
    patch_dispatch: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _login()
    _user_config(monkeypatch, tmp_path, "runtime: temporal\n")
    skill = _skill_dir(tmp_path)
    with pytest.raises(_LocalReached):  # config pin opts out of the cloud default
        cli_main(["compile", str(skill), "--out", str(tmp_path / "o")])


def test_config_deploy_none_forces_local(
    patch_dispatch: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _login()
    _user_config(monkeypatch, tmp_path, "deploy: none\n")
    skill = _skill_dir(tmp_path)
    with pytest.raises(_LocalReached):
        cli_main(["compile", str(skill), "--out", str(tmp_path / "o")])


def test_cloud_flag_overrides_config_local_pin(
    patch_dispatch: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _login()
    _user_config(monkeypatch, tmp_path, "runtime: temporal\n")
    skill = _skill_dir(tmp_path)
    # --cloud wins over a config runtime pin (the config runtime only applies
    # to local runs); the flag routes to cloud, exit sentinel 77.
    assert cli_main(["compile", str(skill), "--out", str(tmp_path / "o"), "--cloud"]) == 77


def test_config_cloudflare_runtime_stays_cloud(
    patch_dispatch: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _login()
    _user_config(monkeypatch, tmp_path, "runtime: cloudflare\n")
    skill = _skill_dir(tmp_path)
    # A cloudflare pin is not a local opt-out — the cloud default stands.
    assert cli_main(["compile", str(skill), "--out", str(tmp_path / "o")]) == 77


def test_cloud_flag_logged_out_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    skill = _skill_dir(tmp_path)
    rc = cli_main(["compile", str(skill), "--out", str(tmp_path / "o"), "--cloud"])
    assert rc == 2
    assert "rote login" in capsys.readouterr().err


def test_cloud_and_no_deploy_conflict_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    skill = _skill_dir(tmp_path)
    rc = cli_main(["compile", str(skill), "--out", str(tmp_path / "o"), "--cloud", "--no-deploy"])
    assert rc == 2
    assert "--no-deploy" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("extra", "needle"),
    [
        (["--runtime", "dbos"], "cloudflare runtime"),
        (["--agent", "claude"], "--agent runs locally"),
        (["--backend", "api"], "--backend runs locally"),
        (["--baseline"], "--baseline runs locally"),
        (["--baseline-input", "x.json"], "--baseline-input runs locally"),
        (["--yes"], "--yes runs locally"),
        (["--no-eval"], "--no-eval runs locally"),
    ],
)
def test_local_only_flags_rejected_in_cloud_mode(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], extra: list[str], needle: str
) -> None:
    skill = _skill_dir(tmp_path)
    rc = cli_main(["compile", str(skill), "--out", str(tmp_path / "o"), "--cloud", *extra])
    assert rc == 2
    assert needle in capsys.readouterr().err


# ───────────────────────── the wire boundary ─────────────────────────


def test_route_paths_still_say_graduations(monkeypatch: pytest.MonkeyPatch) -> None:
    """The network strings kept the pre-rename spelling, deliberately.

    Every other `graduate` in this repo became `compile`, which makes the
    handful of survivors look like a missed sweep. They are not: they name
    routes on a separately deployed platform. Renaming them here would 404
    against every released server, so this test exists to fail loudly if a
    future cleanup pass "finishes the job".
    """
    router = _Router().add("GET", "/v1/graduations/models", (200, {"models": [{"id": "a"}]}))
    router.install(monkeypatch)
    resolve_model(EP, "a")
    assert router.paths() == [("GET", "/v1/graduations/models")]


def test_active_compilation_reads_the_servers_graduation_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`active_graduation` / `graduation_id` are server response keys.

    The exception attribute they populate is `compilation_id` — the rename
    stops at the JSON boundary, and this pins both halves of that seam.
    """
    _Router().add(
        "GET",
        "/v1/skills",
        (
            200,
            {
                "skills": [
                    {
                        "id": "s1",
                        "name": "word-lookup",
                        "active_graduation": {"id": "g9"},
                    }
                ]
            },
        ),
    ).install(monkeypatch)
    with pytest.raises(ActiveCompilationExists) as ei:
        sync_skill(EP, _skill_dir(tmp_path))
    assert ei.value.compilation_id == "g9"


# ───────────────────────── resolve_model ─────────────────────────


def test_resolve_model_default(monkeypatch: pytest.MonkeyPatch) -> None:
    _Router().add(
        "GET",
        "/v1/graduations/models",
        (200, {"models": [{"id": "a"}, {"id": "b"}], "default": "b"}),
    ).install(monkeypatch)
    assert resolve_model(EP, None) == "b"


def test_resolve_model_unknown_lists_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    _Router().add(
        "GET", "/v1/graduations/models", (200, {"models": [{"id": "a"}, {"id": "b"}]})
    ).install(monkeypatch)
    with pytest.raises(CloudCompileError) as ei:
        resolve_model(EP, "zzz")
    assert ei.value.exit_code == 2
    assert "a, b" in str(ei.value)


# ───────────────────────── sync_skill ─────────────────────────


def test_sync_skill_creates_with_encoded_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    skill = _skill_dir(tmp_path)
    # Reserved root path (never uploaded) + a binary file (uploaded verbatim).
    (skill / "bundle.json").write_text("{}", encoding="utf-8")
    (skill / "logo.png").write_bytes(b"\x89PNG\r\n\x00\xff")
    router = _Router()
    router.add("GET", "/v1/skills", (200, {"skills": []}))
    router.add("POST", "/v1/skills", (200, {"id": "s1"}))
    router.install(monkeypatch)

    assert sync_skill(EP, skill) == "s1"
    post = next(body for m, _u, body in router.calls if m == "POST")
    assert post is not None
    assert post["name"] == "word-lookup"
    assert "name: word-lookup" in post["skill_md"]
    files = {f["path"]: f["content_b64"] for f in post["files"]}
    # SKILL.md, bundle.json, hidden, __pycache__ excluded; binary included.
    assert set(files) == {"references/notes.md", "logo.png"}
    import base64

    assert base64.b64decode(files["references/notes.md"]) == b"hello"
    assert base64.b64decode(files["logo.png"]) == b"\x89PNG\r\n\x00\xff"


def test_sync_skill_skips_upload_when_hashes_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    skill = _skill_dir(tmp_path)
    skill_md = (skill / "SKILL.md").read_text(encoding="utf-8")
    import hashlib

    local_md = hashlib.sha256(skill_md.encode("utf-8")).hexdigest()
    files = cg._collect_bundle_files(skill)
    local_bundle = cg._bundle_sha256(files)

    router = _Router()
    router.add(
        "GET",
        "/v1/skills",
        (200, {"skills": [{"id": "s1", "name": "word-lookup", "skill_md_sha256": local_md}]}),
    )
    router.add("GET", "/v1/skills/s1/bundle", (200, {"bundle_sha256": local_bundle, "files": []}))
    router.install(monkeypatch)

    assert sync_skill(EP, skill) == "s1"
    assert ("PUT", "/v1/skills/s1/bundle") not in router.paths()


def test_sync_skill_active_raises_before_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    skill = _skill_dir(tmp_path)
    router = _Router()
    router.add(
        "GET",
        "/v1/skills",
        (
            200,
            {
                "skills": [
                    {
                        "id": "s1",
                        "name": "word-lookup",
                        "active_graduation": {"id": "g9", "status": "running", "phase": 2},
                    }
                ]
            },
        ),
    )
    router.install(monkeypatch)

    with pytest.raises(ActiveCompilationExists) as ei:
        sync_skill(EP, skill)
    assert ei.value.compilation_id == "g9"
    assert ei.value.skill_id == "s1"
    # No PUT/POST was attempted — only the read happened.
    assert router.paths() == [("GET", "/v1/skills")]


def test_sync_skill_puts_when_bundle_differs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    skill = _skill_dir(tmp_path)
    router = _Router()
    router.add(
        "GET",
        "/v1/skills",
        (200, {"skills": [{"id": "s1", "name": "word-lookup", "skill_md_sha256": "stale"}]}),
    )
    router.add("GET", "/v1/skills/s1/bundle", (200, {"bundle_sha256": "stale", "files": []}))
    router.add("PUT", "/v1/skills/s1/bundle", (200, {"bundle_sha256": "new"}))
    router.install(monkeypatch)

    assert sync_skill(EP, skill) == "s1"
    assert ("PUT", "/v1/skills/s1/bundle") in router.paths()


def test_sync_skill_missing_skill_md(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    _Router().install(monkeypatch)
    with pytest.raises(CloudCompileError) as ei:
        sync_skill(EP, empty)
    assert ei.value.exit_code == 2


def test_bundle_caps_enforced(tmp_path: Path) -> None:
    big = [(f"f{i}", b"x") for i in range(cg.MAX_BUNDLE_FILES + 1)]
    with pytest.raises(CloudCompileError) as ei:
        cg._enforce_bundle_caps(big)
    assert ei.value.exit_code == 2
    with pytest.raises(CloudCompileError):
        cg._enforce_bundle_caps([("f", b"x" * (cg.MAX_FILE_BYTES + 1))])


# ───────────────────────── start_compilation ─────────────────────────


def _start_router(monkeypatch: pytest.MonkeyPatch, resp: Response) -> _Router:
    return _Router().add("POST", "/v1/skills/s1/graduations", resp).install(monkeypatch)


def test_start_compilation_success(monkeypatch: pytest.MonkeyPatch) -> None:
    _start_router(monkeypatch, (200, {"id": "g1", "status": "queued", "mode": "full"}))
    assert start_compilation(EP, "s1", mode="full", model="m")["id"] == "g1"


def test_start_compilation_insufficient_credits(monkeypatch: pytest.MonkeyPatch) -> None:
    _start_router(
        monkeypatch, (402, {"error": "insufficient_credits", "code": "insufficient_credits"})
    )
    with pytest.raises(CloudCompileError) as ei:
        start_compilation(EP, "s1", mode="full", model="m")
    assert "top up at http://cloud/billing" in str(ei.value)


def test_start_compilation_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    _start_router(monkeypatch, (503, {"error": "disabled"}))
    with pytest.raises(CloudCompileError) as ei:
        start_compilation(EP, "s1", mode="full", model="m")
    assert "--local" in str(ei.value)


def test_start_compilation_no_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    _start_router(monkeypatch, (409, {"error": "no changes", "code": "no_changes"}))
    with pytest.raises(NoChanges):
        start_compilation(EP, "s1", mode="update", model="m")


def test_start_compilation_active(monkeypatch: pytest.MonkeyPatch) -> None:
    _start_router(monkeypatch, (409, {"error": "active", "graduation_id": "g7"}))
    with pytest.raises(ActiveCompilationExists) as ei:
        start_compilation(EP, "s1", mode="full", model="m")
    assert ei.value.compilation_id == "g7"


# ───────────────────────── poll_compilation ─────────────────────────


def _ev(**kw: Any) -> dict[str, Any]:
    base = {"type": "log", "ts": 1.0, "message": "m"}
    base.update(kw)
    return base


def test_poll_replays_events_and_advances_cursor(monkeypatch: pytest.MonkeyPatch) -> None:
    router = _Router()
    router.add(
        "GET",
        "/v1/graduations/g1",
        (200, {"status": "running", "events": [_ev(message="a"), _ev(message="b")], "cursor": 2}),
        (200, {"status": "complete", "events": [_ev(message="c")], "cursor": 5}),
    )
    router.install(monkeypatch)

    seen: list[str] = []
    final = poll_compilation(EP, "g1", lambda e: seen.append(e.message), sleep=lambda _s: None)
    assert final["status"] == "complete"
    assert seen == ["a", "b", "c"]
    # The second request carried ?after=2.
    assert any("after=2" in u for m, u, _ in router.calls if u.endswith("after=2"))


def test_poll_tolerates_malformed_events(monkeypatch: pytest.MonkeyPatch) -> None:
    router = _Router()
    router.add(
        "GET",
        "/v1/graduations/g1",
        (200, {"status": "complete", "events": [{"type": "bogus"}, "garbage", _ev(message="ok")]}),
    )
    router.install(monkeypatch)
    seen: list[str] = []
    poll_compilation(EP, "g1", lambda e: seen.append(e.message), sleep=lambda _s: None)
    assert seen == ["ok"]


def test_poll_retries_then_gives_up(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a: Any) -> Response:
        raise URLError("down")

    router = _Router().add("GET", "/v1/graduations/g1", boom).install(monkeypatch)
    sleeps: list[float] = []
    with pytest.raises(CloudCompileError) as ei:
        poll_compilation(EP, "g1", None, sleep=sleeps.append)
    assert "lost contact" in str(ei.value)
    assert len(sleeps) == 4  # 5th consecutive failure gives up
    assert router  # silence unused


def test_poll_keyboardinterrupt_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    def interrupt(*_a: Any) -> Response:
        raise KeyboardInterrupt

    _Router().add("GET", "/v1/graduations/g1", interrupt).install(monkeypatch)
    with pytest.raises(KeyboardInterrupt):
        poll_compilation(EP, "g1", None, sleep=lambda _s: None)


# ───────────────────────── _event_from_wire round-trip ─────────────────────────


@pytest.mark.parametrize(
    "event",
    [
        CompilationEvent(type="phase", ts=1.0, phase=2, phase_name="Classify", message="phase 2"),
        CompilationEvent(
            type="turn", ts=2.0, turn=5, tokens={"input": 40, "output": 8}, message=""
        ),
        CompilationEvent(type="tool", ts=3.0, turn=5, tool_name="Write", path="p.py", message="w"),
        CompilationEvent(type="artifact", ts=4.0, path="a", message="art"),
        CompilationEvent(type="log", ts=5.0, message="log"),
        CompilationEvent(type="warning", ts=6.0, message="warn"),
        CompilationEvent(type="complete", ts=7.0, message="done"),
        CompilationEvent(type="error", ts=8.0, message="boom"),
    ],
)
def test_event_from_wire_round_trip(event: CompilationEvent) -> None:
    wire = dataclasses.asdict(event)
    assert _event_from_wire(wire) == event


def test_event_from_wire_rejects_garbage() -> None:
    assert _event_from_wire("nope") is None
    assert _event_from_wire({"type": "unknown"}) is None


# ───────────────────────── download_artifacts ─────────────────────────


@pytest.mark.parametrize("bad", ["/etc/passwd", "../escape", "a/../../b"])
def test_download_rejects_traversal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad: str
) -> None:
    _Router().add("GET", "/v1/graduations/g1/files", (200, {"files": [{"path": bad}]})).install(
        monkeypatch
    )
    with pytest.raises(CloudCompileError, match="unsafe"):
        download_artifacts(EP, "g1", tmp_path / "out")


def test_download_writes_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    router = _Router()
    router.add(
        "GET",
        "/v1/graduations/g1/files",
        (
            200,
            {
                "files": [
                    {"path": "compiled/pipeline.yaml"},
                    {"path": "runtime/cloudflare/manifest.json"},
                ]
            },
        ),
    )
    router.add(
        "GET", "/v1/graduations/g1/files/compiled/pipeline.yaml", (200, {"content": "name: p\n"})
    )
    router.add(
        "GET",
        "/v1/graduations/g1/files/runtime/cloudflare/manifest.json",
        (200, {"content": '{"name": "p"}'}),
    )
    router.install(monkeypatch)

    out = tmp_path / "out"
    written = download_artifacts(EP, "g1", out)
    assert (out / "compiled" / "pipeline.yaml").read_text() == "name: p\n"
    assert (out / "runtime" / "cloudflare" / "manifest.json").read_text() == '{"name": "p"}'
    assert set(written) == {"compiled/pipeline.yaml", "runtime/cloudflare/manifest.json"}


# ───────────────────────── end-to-end CLI cloud run ─────────────────────────


def _full_run_router() -> _Router:
    router = _Router()
    router.add("GET", "/v1/graduations/models", (200, {"models": [{"id": "m1"}], "default": "m1"}))
    router.add("GET", "/v1/skills", (200, {"skills": []}))
    router.add("POST", "/v1/skills", (200, {"id": "s1"}))
    router.add(
        "POST", "/v1/skills/s1/graduations", (200, {"id": "g1", "status": "queued", "mode": "full"})
    )
    router.add(
        "GET",
        "/v1/graduations/g1",
        (
            200,
            {
                "status": "complete",
                "mode": "full",
                "model": "m1",
                "tokens": {"input": 100, "output": 50},
                "cost_usd": 0.12,
                "pipeline_id": "p1",
                "deployed": True,
                "events": [_ev(type="complete", message="done")],
                "cursor": 1,
            },
        ),
    )
    router.add(
        "GET",
        "/v1/graduations/g1/files",
        (
            200,
            {
                "files": [
                    {"path": "runtime/cloudflare/manifest.json"},
                    {"path": "compiled/scorecard.md"},
                ]
            },
        ),
    )
    router.add(
        "GET",
        "/v1/graduations/g1/files/runtime/cloudflare/manifest.json",
        (200, {"content": json.dumps({"name": "word-lookup", "version": "0.1.0"})}),
    )
    router.add(
        "GET",
        "/v1/graduations/g1/files/compiled/scorecard.md",
        (200, {"content": "# scorecard\n"}),
    )
    return router


def test_cli_cloud_run_json_and_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _login()
    _full_run_router().install(monkeypatch)
    # Sink pricing must be hermetic (no network).
    from rote.eval.pricing import PricingError

    def _raise(*_a: object, **_k: object) -> object:
        raise PricingError("offline (test)")

    monkeypatch.setattr("rote.eval.pricing.fetch_catalog", _raise)

    skill = _skill_dir(tmp_path)
    out = tmp_path / "out"
    progress = tmp_path / "run.ndjson"
    rc = cli_main(
        [
            "compile",
            str(skill),
            "--out",
            str(out),
            "--cloud",
            "--json",
            "--progress-file",
            str(progress),
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["pipeline"] == {"name": "word-lookup", "version": "0.1.0"}
    assert payload["runtime"] == "cloudflare"
    # The CLI's own --json key renamed with everything else. Only the strings
    # that cross the network to rote cloud kept the old spelling; this one is
    # the CLI's contract with whoever pipes its output, not the server's.
    assert payload["cloud"]["compilation_id"] == "g1"
    assert payload["cloud"]["skill_id"] == "s1"
    assert payload["deploy"] == {"target": "rote-cloud", "ok": True, "pipeline_id": "p1"}
    assert Path(payload["scorecard"]).name == "scorecard.md"

    # Artifacts landed on disk.
    assert (out / "runtime" / "cloudflare" / "manifest.json").is_file()
    assert (out / "compiled" / "scorecard.md").is_file()

    # The JSONL sink's last line is the summary digest.
    lines = progress.read_text(encoding="utf-8").splitlines()
    summary = json.loads(lines[-1])
    assert summary["type"] == "summary"
    assert summary["cloud"]["compilation_id"] == "g1"


def test_config_agent_does_not_leak_into_cloud(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A config-file `agent:` is a local pref — the logged-in default still
    runs on cloud, with no spurious `--agent runs locally` rejection."""
    _login()
    _user_config(monkeypatch, tmp_path, "agent: claude\n")
    _full_run_router().install(monkeypatch)
    skill = _skill_dir(tmp_path)
    rc = cli_main(["compile", str(skill), "--out", str(tmp_path / "out")])
    assert rc == 0
    assert "runs locally" not in capsys.readouterr().err


def test_config_model_does_not_leak_into_cloud(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A config-file `model:` the server doesn't offer must NOT reach
    resolve_model — the cloud run takes the server default, exit 0."""
    _login()
    _user_config(monkeypatch, tmp_path, "model: claude-sonnet-4-6\n")
    router = _full_run_router().install(monkeypatch)
    skill = _skill_dir(tmp_path)
    rc = cli_main(["compile", str(skill), "--out", str(tmp_path / "out"), "--json"])
    assert rc == 0  # config model ignored; no "unknown model" exit 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["cloud"]["model"] == "m1"  # server default
    # The start-compilation POST carried the server default, not the config model.
    start = next(body for m, u, body in router.calls if m == "POST" and u.endswith("/graduations"))
    assert start["model"] == "m1"


def test_cli_cloud_run_server_error_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _login()
    router = _Router()
    router.add("GET", "/v1/graduations/models", (200, {"models": [{"id": "m1"}], "default": "m1"}))
    router.add("GET", "/v1/skills", (200, {"skills": []}))
    router.add("POST", "/v1/skills", (200, {"id": "s1"}))
    router.add(
        "POST", "/v1/skills/s1/graduations", (200, {"id": "g1", "status": "queued", "mode": "full"})
    )
    router.add(
        "GET",
        "/v1/graduations/g1",
        (200, {"status": "error", "error": "compiler crashed", "events": []}),
    )
    router.add("GET", "/v1/graduations/g1/files", (200, {"files": []}))
    router.install(monkeypatch)

    skill = _skill_dir(tmp_path)
    rc = cli_main(["compile", str(skill), "--out", str(tmp_path / "out"), "--cloud"])
    assert rc == 1
    assert "compiler crashed" in capsys.readouterr().err
