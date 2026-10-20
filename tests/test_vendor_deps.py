"""An --backend api deliverable's tests run with the vendor libraries its code imports.

A real compile (ops-report, --system1) had every generated test for its
Google client code fail on ``ModuleNotFoundError: google.auth``: the tests
ran in rote's environment, which does not ship the vendor's SDK.
"""

from __future__ import annotations

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
