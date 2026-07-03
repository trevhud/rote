"""Tests for hash-guarded re-emission (the .rote-manifest.json mechanism).

An emitted output directory is a working directory: users fill in the
``extracted/`` stubs and (on Temporal) judge classes. Before the
manifest existed, a re-emit clobbered that work unconditionally. These
tests pin the guard's contract at both levels:

1. **EmitWriter unit behavior** — the four-way decision (missing /
   identical / pristine / user-edited) and manifest bookkeeping.
2. **Adapter integration** — every registered adapter routes its writes
   through the guard, so a user-edited stub survives a real re-emit.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rote.adapters import get_adapter
from rote.adapters._common import MANIFEST_NAME, EmitWriter
from rote.ir import Pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
BDR_PIPELINE_YAML = REPO_ROOT / "examples" / "bdr-outreach" / "expected" / "pipeline.yaml"


# ───────── EmitWriter unit behavior ─────────


def _emit_one(out: Path, content: str) -> Path:
    writer = EmitWriter(out)
    path = writer.write("mod.py", content=content)
    writer.finalize()
    return path


def test_fresh_write_creates_file_and_manifest(tmp_path: Path) -> None:
    path = _emit_one(tmp_path, "v1\n")

    assert path == tmp_path / "mod.py"
    assert path.read_text() == "v1\n"
    manifest = json.loads((tmp_path / MANIFEST_NAME).read_text())
    assert manifest["version"] == 1
    assert "mod.py" in manifest["files"]


def test_nested_write_creates_parents_and_posix_manifest_key(tmp_path: Path) -> None:
    writer = EmitWriter(tmp_path)
    path = writer.write("src", "signatures", "judge.ts", content="x\n")
    writer.finalize()

    assert path == tmp_path / "src" / "signatures" / "judge.ts"
    manifest = json.loads((tmp_path / MANIFEST_NAME).read_text())
    assert "src/signatures/judge.ts" in manifest["files"]


def test_pristine_file_is_overwritten(tmp_path: Path) -> None:
    _emit_one(tmp_path, "v1\n")
    path = _emit_one(tmp_path, "v2\n")

    assert path == tmp_path / "mod.py"
    assert path.read_text() == "v2\n"
    assert not (tmp_path / "mod.py.new").exists()


def test_identical_content_is_a_noop_even_when_user_owned(tmp_path: Path) -> None:
    # No prior manifest, file already has exactly the fresh content
    # (e.g. a committed snapshot): adopt it, don't spray a .new sibling.
    (tmp_path / "mod.py").write_text("v1\n")
    path = _emit_one(tmp_path, "v1\n")

    assert path == tmp_path / "mod.py"
    assert not (tmp_path / "mod.py.new").exists()
    manifest = json.loads((tmp_path / MANIFEST_NAME).read_text())
    assert "mod.py" in manifest["files"]


def test_user_edited_file_is_preserved_with_new_sibling(tmp_path: Path) -> None:
    _emit_one(tmp_path, "v1\n")
    (tmp_path / "mod.py").write_text("user's edit\n")

    writer = EmitWriter(tmp_path)
    path = writer.write("mod.py", content="v2\n")
    writer.finalize()

    assert path == tmp_path / "mod.py.new"
    assert (tmp_path / "mod.py").read_text() == "user's edit\n"
    assert path.read_text() == "v2\n"
    assert writer.preserved == {"mod.py": path}


def test_preservation_does_not_adopt_the_users_hash(tmp_path: Path) -> None:
    # After a preservation event, the manifest must still hold the hash
    # of what *rote* last wrote — so a user who reverts their edit gets
    # normal overwrite behavior back on the following emit.
    _emit_one(tmp_path, "v1\n")
    (tmp_path / "mod.py").write_text("user's edit\n")
    _emit_one(tmp_path, "v2\n")  # preserved → mod.py.new

    (tmp_path / "mod.py").write_text("v1\n")  # user reverts
    path = _emit_one(tmp_path, "v3\n")

    assert path == tmp_path / "mod.py"
    assert path.read_text() == "v3\n"


def test_pre_manifest_file_that_differs_is_preserved(tmp_path: Path) -> None:
    # A directory emitted before manifests existed has files but no
    # manifest: rote cannot prove it wrote them, so it must not clobber.
    (tmp_path / "mod.py").write_text("hand-written\n")
    path = _emit_one(tmp_path, "v1\n")

    assert path == tmp_path / "mod.py.new"
    assert (tmp_path / "mod.py").read_text() == "hand-written\n"
    # And the unproven file's hash is never adopted as rote's own:
    manifest = json.loads((tmp_path / MANIFEST_NAME).read_text())
    assert "mod.py" not in manifest["files"]


def test_corrupt_manifest_raises(tmp_path: Path) -> None:
    (tmp_path / MANIFEST_NAME).write_text("{not json")
    with pytest.raises(ValueError, match="Corrupt emit manifest"):
        EmitWriter(tmp_path)


def test_manifest_carries_forward_entries_for_files_not_reemitted(tmp_path: Path) -> None:
    writer = EmitWriter(tmp_path)
    writer.write("a.py", content="a\n")
    writer.write("b.py", content="b\n")
    writer.finalize()

    writer = EmitWriter(tmp_path)
    writer.write("a.py", content="a\n")  # b.py not emitted this time
    writer.finalize()

    manifest = json.loads((tmp_path / MANIFEST_NAME).read_text())
    assert "b.py" in manifest["files"]


# ───────── Adapter integration ─────────

ADAPTER_NAMES = ["dbos", "temporal", "cloudflare", "dbos-ts", "inngest"]


@pytest.mark.parametrize("runtime", ADAPTER_NAMES)
def test_double_emit_is_clean_and_manifest_exists(
    runtime: str, bdr_pipeline: Pipeline, tmp_path: Path
) -> None:
    adapter = get_adapter(runtime)
    adapter.emit(bdr_pipeline, tmp_path)
    written = adapter.emit(bdr_pipeline, tmp_path)

    assert (tmp_path / MANIFEST_NAME).is_file()
    assert not [p for p in written.values() if p.name.endswith(".new")]


def test_python_adapter_routes_through_the_guard(tmp_path: Path) -> None:
    """The python adapter refuses HITL pipelines (so it can't run the BDR
    parametrization above) — cover it with a gate-free mini pipeline."""
    from rote.ir import Node, NodeKind

    pipeline = Pipeline(
        name="mini",
        input={"type": "In", "required": [], "optional": []},
        nodes=[
            Node(
                id="only",
                kind=NodeKind.PURE_FUNCTION,
                description="x",
                impl="extracted/only.py:only",
            )
        ],
        edges=[],
        entry_nodes=["only"],
        exit_nodes=["only"],
    )
    adapter = get_adapter("python")
    first = adapter.emit(pipeline, tmp_path)
    assert (tmp_path / MANIFEST_NAME).is_file()

    target = first["extracted/only"]
    original = target.read_text(encoding="utf-8")
    target.write_text(original + "\n# user implementation\n", encoding="utf-8")

    second = adapter.emit(pipeline, tmp_path)
    assert target.read_text(encoding="utf-8").endswith("# user implementation\n")
    assert second["extracted/only"] == target.with_name(target.name + ".new")


@pytest.mark.parametrize("runtime", ADAPTER_NAMES)
def test_reemit_preserves_user_edited_file(
    runtime: str, bdr_pipeline: Pipeline, tmp_path: Path
) -> None:
    adapter = get_adapter(runtime)
    first = adapter.emit(bdr_pipeline, tmp_path)

    # Edit every emitted file the way a user filling in stubs would.
    label, target = next(iter(first.items()))
    original = target.read_text(encoding="utf-8")
    target.write_text(original + "\n# user implementation\n", encoding="utf-8")

    second = adapter.emit(bdr_pipeline, tmp_path)

    assert target.read_text(encoding="utf-8").endswith("# user implementation\n")
    assert second[label] == target.with_name(target.name + ".new")
    assert second[label].read_text(encoding="utf-8") == original
