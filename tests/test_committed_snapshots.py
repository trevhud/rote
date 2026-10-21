"""The committed BDR runtime snapshots match what the adapters emit today.

examples/bdr-outreach/expected/runtimes/<runtime>/ is the reviewable
record of each adapter's output. The adapter suites used to keep it fresh
by emitting straight into it, which also meant running pytest rewrote
committed files (and, without the example's extracted/ sources, replaced
its real implementations with stubs). Now the suites emit into temp dirs,
and this test does the comparison: a fresh emission, with the same
extracted/ sources `rote emit` uses, must reproduce the snapshot byte for
byte. When an adapter change is intended, regenerate with the command in
the failure message and commit the result.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rote.adapters import get_adapter
from rote.ir import Pipeline

EXPECTED = Path(__file__).resolve().parent.parent / "examples" / "bdr-outreach" / "expected"
RUNTIMES = sorted(p.name for p in (EXPECTED / "runtimes").iterdir() if p.is_dir())

#: Install and import artifacts a developer may have locally; never committed.
_IGNORED_PARTS = frozenset({"__pycache__", "node_modules"})


def _files(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file() and not _IGNORED_PARTS.intersection(p.relative_to(root).parts)
    }


def test_every_adapter_suite_has_a_snapshot() -> None:
    assert RUNTIMES == ["cloudflare", "dbos", "dbos-ts", "inngest", "temporal"]


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_fresh_emission_reproduces_the_committed_snapshot(
    runtime: str, bdr_pipeline: Pipeline, tmp_path: Path
) -> None:
    # The same extracted/ sources `rote emit <pipeline.yaml>` reads: the
    # dir holding the pipeline.
    get_adapter(runtime, extracted_source_dir=EXPECTED).emit(bdr_pipeline, tmp_path)
    emitted = _files(tmp_path)
    committed = _files(EXPECTED / "runtimes" / runtime)

    # The app registry records every emitted dir for park-on-auth release; a
    # committed snapshot is not an app, so the command keeps it out of the
    # user's real ~/.local/share/rote/apps.json.
    regenerate = (
        'ROTE_APPS_PATH="$(mktemp -d)/apps.json" '
        f"rote emit examples/bdr-outreach/expected/pipeline.yaml --runtime {runtime} "
        f"--out examples/bdr-outreach/expected/runtimes/{runtime}"
    )
    assert sorted(emitted) == sorted(committed), f"file set differs; regenerate: {regenerate}"
    stale = sorted(rel for rel in emitted if emitted[rel] != committed[rel])
    assert not stale, f"stale snapshot files {stale}; regenerate: {regenerate}"


# ───────── The session guard in conftest ─────────


def test_the_examples_guard_sees_adds_removes_and_edits(tmp_path: Path) -> None:
    """The fingerprint the conftest guard compares before and after the run.

    Each kind of write has to register, including an edit that keeps the
    file size (a stub replacing a same-length body), and tooling
    artifacts must not.
    """
    from tests.conftest import examples_fingerprint, fingerprint_changes

    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "b.py").write_text("y = 2\n")
    before = examples_fingerprint(tmp_path)

    (tmp_path / "a.py").write_text("x = 9\n")  # same size, new content
    (tmp_path / "b.py").unlink()
    (tmp_path / "c.py").write_text("z = 3\n")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "a.cpython-311.pyc").write_bytes(b"\0")
    (tmp_path / "node_modules" / "pkg").mkdir(parents=True)
    (tmp_path / "node_modules" / "pkg" / "index.js").write_text("")

    assert fingerprint_changes(before, examples_fingerprint(tmp_path)) == [
        "added: c.py",
        "removed: b.py",
        "modified: a.py",
    ]
    assert fingerprint_changes(before, before) == []
