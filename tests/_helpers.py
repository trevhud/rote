"""Shared helpers for adapter tests.

Everything here exists in 3+ adapter test files otherwise — and the
copies had already drifted (the Cloudflare MCP scan was missing the
backtick template-literal alternative, so a legitimate "mcp" mention
inside a template string failed only that adapter's test).
"""

from __future__ import annotations

import re
from pathlib import Path

from rote.ir import Node, Pipeline

# ───────── MCP invariant scan (TS) ─────────
#
# Comments and JSDoc may *mention* MCP to explain the compilation
# history; only executable code is forbidden from referencing it. So
# strip /* ... */ blocks, // line comments, and string literals —
# including backtick template literals — before scanning.

_JS_STRING = re.compile(r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|`(?:[^`\\]|\\.)*`')
_BLOCK_COMMENT = re.compile(r"/\*[\s\S]*?\*/")
_LINE_COMMENT = re.compile(r"//[^\n]*")


def assert_no_mcp_in_ts(
    emit_result: dict[str, Path], *, min_files: int, expected: set[str] | None = None
) -> None:
    """Assert no emitted .ts file references MCP in executable code.

    ``expected`` names the labels that legitimately DO — the MCP helper
    itself, and any agent_loop module, whose ``tools:`` are MCP tool
    names by the IR's own definition. Naming them (rather than skipping
    every file that happens to mention MCP) keeps the scan a real
    constraint: an unexpected MCP reference still fails, and an expected
    site that stops referencing MCP fails too.

    ``min_files`` guards against the scan silently checking nothing
    (e.g. a refactor that changes emitted file suffixes).
    """
    expected = expected or set()
    checked = 0
    for label, path in emit_result.items():
        if not str(path).endswith(".ts"):
            continue
        if label in expected:
            assert "mcp" in path.read_text(encoding="utf-8").lower(), (
                f"{label} was declared an expected MCP site but references no MCP"
            )
            continue
        src = path.read_text(encoding="utf-8")
        cleaned = _BLOCK_COMMENT.sub(" ", src)
        cleaned = _LINE_COMMENT.sub(" ", cleaned)
        cleaned = _JS_STRING.sub('""', cleaned)
        assert "mcp" not in cleaned.lower(), (
            f"{label} ({path.name}) contains forbidden substring 'mcp' in executable code"
        )
        checked += 1
    assert checked >= min_files, f"MCP scan only saw {checked} .ts files (expected >= {min_files})"


# ───────── Minimal pipeline factory ─────────


def mini_pipeline(node: Node) -> Pipeline:
    """A single-node pipeline for focused emission tests."""
    return Pipeline(
        name="mini",
        input={"type": "In", "required": [], "optional": []},
        nodes=[node],
        edges=[],
        entry_nodes=[node.id],
        exit_nodes=[node.id],
    )
