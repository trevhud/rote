"""Every module the emitted code imports ships with it.

Two real --system1 deliverables passed verification and crashed on their
first import at run time: impl modules imported ``extracted._common`` (a
helper the agent wrote) and ``extracted.rule_tables`` (System-1's tables),
and emission copied only the modules a node's ``impl:`` names. The default
path has the same exposure whenever the agent writes a shared helper.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from rote.adapters import get_adapter
from rote.compiler.verify import EmitTarget, emit_findings, unshipped_imports
from rote.ir import load_pipeline

PIPELINE = {
    "name": "helpers",
    "input": {"type": "In"},
    "nodes": [
        {
            "id": "total_lines",
            "kind": "pure_function",
            "description": "Total the lines.",
            "impl": "extracted/total_lines.py:total_lines",
            "inputs": {"request": "pipeline.input"},
        }
    ],
    "edges": [],
    "entry_nodes": ["total_lines"],
    "exit_nodes": ["total_lines"],
}
IMPL = (
    "from extracted._common import to_cents\n"
    "from extracted.rule_tables import CAPS\n\n\n"
    "def total_lines(request):\n"
    "    return {'total': to_cents(sum(CAPS.values()))}\n"
)


def _artifact(tmp_path: Path, *, helpers: bool) -> Path:
    art = tmp_path / "compiled"
    (art / "extracted").mkdir(parents=True)
    (art / "pipeline.yaml").write_text(yaml.safe_dump(PIPELINE, sort_keys=False))
    (art / "extracted" / "total_lines.py").write_text(IMPL)
    if helpers:
        (art / "extracted" / "_common.py").write_text("def to_cents(x):\n    return round(x, 2)\n")
        (art / "extracted" / "rule_tables.py").write_text("CAPS = {'meals': 75}\n")
    return art


@pytest.mark.parametrize("runtime", ["dbos", "python"])
def test_helper_modules_ship_with_the_emitted_runtime(tmp_path: Path, runtime: str) -> None:
    art = _artifact(tmp_path, helpers=True)
    out = tmp_path / "runtime"
    get_adapter(runtime, extracted_source_dir=art).emit(load_pipeline(art / "pipeline.yaml"), out)
    assert (out / "extracted" / "_common.py").read_text() == (
        art / "extracted" / "_common.py"
    ).read_text()
    assert (out / "extracted" / "rule_tables.py").is_file()
    assert unshipped_imports(out) == []


def test_verification_catches_an_import_the_runtime_cannot_satisfy(tmp_path: Path) -> None:
    """The agent imports a helper it never wrote: verification, not run time, says so."""
    art = _artifact(tmp_path, helpers=False)
    findings = emit_findings(load_pipeline(art / "pipeline.yaml"), art, EmitTarget("dbos"))
    assert sorted(f.message.split(" imports ")[1].split(",")[0] for f in findings) == [
        "extracted._common",
        "extracted.rule_tables",
    ]
    assert all(f.kind == "unshipped_module" and f.severity == "error" for f in findings)


def test_every_local_import_form_is_resolved(tmp_path: Path) -> None:
    ext = tmp_path / "extracted"
    ext.mkdir()
    (ext / "a.py").write_text(
        "import extracted.b\n"
        "from extracted import c\n"
        "from . import d\n"
        "from .e import f\n"
        "import json\n"
    )
    assert sorted(m for _, m in unshipped_imports(tmp_path)) == ["b", "c", "d", "e"]
    for name in "bcde":
        (ext / f"{name}.py").write_text("")
    assert unshipped_imports(tmp_path) == []
