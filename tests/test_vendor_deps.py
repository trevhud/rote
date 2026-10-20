"""An --backend api deliverable's tests run with the vendor libraries its code imports.

A real compile (ops-report, --system1) had every generated test for its
Google client code fail on ``ModuleNotFoundError: google.auth``: the tests
ran in rote's environment, which does not ship the vendor's SDK.
"""

from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

import pytest

from rote.compiler import sandbox
from rote.compiler.verify import EmitTarget, run_generated_tests, verify_deliverable

PIPELINE = """\
name: p
version: "0.1.0"
description: d
input: {type: In}
nodes:
  - id: fetch
    kind: external_call
    description: Fetch.
    impl: extracted/vendor.py:fetch
edges: []
entry_nodes: [fetch]
exit_nodes: [fetch]
"""


def _deliverable(tmp_path: Path) -> Path:
    root = tmp_path / "d"
    (root / "extracted").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "pipeline.yaml").write_text(PIPELINE, encoding="utf-8")
    (root / "extracted" / "vendor.py").write_text(
        textwrap.dedent(
            """
            # docs: skill
            import fakevendorsdk


            def fetch():
                return {"v": fakevendorsdk.VALUE}
            """
        ),
        encoding="utf-8",
    )
    (root / "tests" / "test_vendor.py").write_text(
        "from extracted.vendor import fetch\n\n\n"
        "def test_fetch():\n    assert fetch() == {'v': 42}\n",
        encoding="utf-8",
    )
    return root


def _library(tmp_path: Path) -> Path:
    deps = tmp_path / "vendor-deps" / "abc"
    (deps / "fakevendorsdk").mkdir(parents=True)
    (deps / "fakevendorsdk" / "__init__.py").write_text("VALUE = 42\n", encoding="utf-8")
    return deps


@pytest.mark.parametrize("trust_skill", [True, False])
def test_the_tests_import_the_vendor_library_from_its_install_dir(
    tmp_path: Path, trust_skill: bool
) -> None:
    if not trust_skill and sandbox.detect()[0] is None:
        pytest.skip("no OS sandbox on this machine")
    root = _deliverable(tmp_path)
    deps = _library(tmp_path)
    without = run_generated_tests(root, trust_skill=trust_skill)
    with_deps = run_generated_tests(root, trust_skill=trust_skill, extra_paths=(deps,))
    assert without is not None and without.status == "failed"
    assert "ModuleNotFoundError" in without.output
    assert with_deps is not None and with_deps.status == "passed", with_deps.output


def test_an_install_dir_that_would_expose_the_home_is_not_confined(tmp_path: Path) -> None:
    if sandbox.detect()[0] is None:
        pytest.skip("no OS sandbox on this machine")
    root = _deliverable(tmp_path)
    result = run_generated_tests(root, extra_paths=(Path.home(),))
    assert result is not None and result.status == "unverified"
    assert "vendor libraries cannot be confined" in result.summary


def test_verification_installs_the_apps_vendor_libraries_on_the_api_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _deliverable(tmp_path)
    deps = _library(tmp_path)
    asked: list[Path] = []

    def install(app: Path) -> Path:
        asked.append(app)
        return deps

    monkeypatch.setattr("rote.compiler.verify.install_vendor_deps", install)
    api = verify_deliverable(root, trust_skill=True, emit_target=EmitTarget("dbos", "api"))
    assert asked == [root.resolve()] or asked == [root]
    assert api.tests is not None and api.tests.status == "passed", api.tests.output
    # The mcp backend never calls this module, so nothing is installed for it.
    asked.clear()
    mcp = verify_deliverable(root, trust_skill=True, emit_target=EmitTarget("dbos", "mcp"))
    assert asked == []
    assert mcp.tests is not None and mcp.tests.status == "failed"


def test_an_install_that_fails_is_listed_not_hidden(tmp_path: Path) -> None:
    """conftest refuses every real install, as an offline machine would."""
    root = _deliverable(tmp_path)
    (root / "extracted" / "vendor.py").write_text(
        "import httplib2_not_mapped_x\n", encoding="utf-8"
    )
    result = verify_deliverable(root, trust_skill=True, emit_target=EmitTarget("dbos", "api"))
    assert any("ran without the app's vendor libraries" in q.text for q in result.questions)


def test_two_installs_of_one_set_at_once_leave_one_complete_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bench runs two compiles, often of one skill. In place, B deleted A's
    half-written install and A marked complete a directory B was filling."""
    import threading
    import time

    from rote.vendor_specs import deps

    monkeypatch.setenv("ROTE_SPEC_CACHE_DIR", str(tmp_path / "cache" / "specs"))
    monkeypatch.setattr(deps, "vendor_requirements", lambda app: ["pkg-a"])
    installs: list[Path] = []

    def slow_uv(args: list[str]) -> subprocess.CompletedProcess[str]:
        # What `uv pip install --target T` does: fill T, over time.
        target = Path(args[args.index("--target") + 1])
        installs.append(target)
        target.mkdir(parents=True, exist_ok=True)
        mine = target / f"run-{threading.get_ident()}"
        mine.write_text("x")
        time.sleep(0.3)  # the other run is installing meanwhile
        assert mine.is_file(), "another run deleted this install while it was written"
        mine.unlink()
        (target / "first.py").write_text("x")
        (target / "second.py").write_text("y")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(deps, "_run_uv", slow_uv)
    monkeypatch.setattr(deps.shutil, "which", lambda name: "/usr/bin/uv")
    results: list[Path | None] = []
    errors: list[BaseException] = []

    barrier = threading.Barrier(2)

    def run(delay: float) -> None:
        try:
            barrier.wait()
            time.sleep(delay)  # the second run starts while the first is installing
            results.append(deps.install_vendor_deps(tmp_path / "app"))
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=run, args=(delay,)) for delay in (0.0, 0.1)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    [target] = set(results)
    assert target is not None
    assert sorted(p.name for p in target.iterdir()) == [".complete", "first.py", "second.py"]
    # No staging directory is left behind, and a third run installs nothing.
    assert [p for p in target.parent.iterdir() if p != target] == []
    count = len(installs)
    assert deps.install_vendor_deps(tmp_path / "app") == target and len(installs) == count


def test_the_cache_key_changes_with_the_date(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The newest compatible releases are installed again the next day."""
    import datetime

    from rote.vendor_specs import deps

    monkeypatch.setenv("ROTE_SPEC_CACHE_DIR", str(tmp_path / "cache" / "specs"))
    monkeypatch.setattr(deps, "vendor_requirements", lambda app: ["pkg-a"])
    monkeypatch.setattr(
        deps,
        "_run_uv",
        lambda args: subprocess.CompletedProcess(args, 0, "", ""),
    )
    monkeypatch.setattr(deps.shutil, "which", lambda name: "/usr/bin/uv")

    class Day(datetime.date):
        value = datetime.date(2026, 10, 20)

        @classmethod
        def today(cls) -> datetime.date:
            return cls.value

    monkeypatch.setattr(deps, "date", Day)
    first = deps.install_vendor_deps(tmp_path / "app")
    assert first is not None and first.is_dir()
    # Another set's install from the same day is not this set's to prune.
    monkeypatch.setattr(deps, "vendor_requirements", lambda app: ["pkg-b"])
    other = deps.install_vendor_deps(tmp_path / "app")
    monkeypatch.setattr(deps, "vendor_requirements", lambda app: ["pkg-a"])
    Day.value = datetime.date(2026, 10, 21)
    second = deps.install_vendor_deps(tmp_path / "app")
    assert first != second
    # Yesterday's install stays: a run that began before midnight may still
    # import an unloaded submodule from it.
    assert second is not None and second.is_dir() and first.is_dir()
    Day.value = datetime.date(2026, 10, 22)
    third = deps.install_vendor_deps(tmp_path / "app")
    # Two days on, the oldest goes; yesterday's (second) is kept.
    assert third is not None and third.is_dir() and second.is_dir() and not first.exists()
    assert other is not None and other.is_dir()


def test_a_failed_move_into_place_is_an_error_not_a_lost_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rote.vendor_specs import deps

    monkeypatch.setenv("ROTE_SPEC_CACHE_DIR", str(tmp_path / "cache" / "specs"))
    monkeypatch.setattr(deps, "vendor_requirements", lambda app: ["pkg-a"])
    monkeypatch.setattr(deps, "_run_uv", lambda args: subprocess.CompletedProcess(args, 0, "", ""))
    monkeypatch.setattr(deps.shutil, "which", lambda name: "/usr/bin/uv")

    def broken(src: object, dst: object) -> None:
        raise OSError("read-only file system")

    monkeypatch.setattr(deps.os, "replace", broken)
    with pytest.raises(OSError, match="read-only"):
        deps.install_vendor_deps(tmp_path / "app")
    # The staging copy is cleaned up either way.
    assert list((tmp_path / "cache" / "vendor-deps").iterdir()) == []
