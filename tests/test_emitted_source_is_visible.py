"""Untrusted IR text never lands invisible in emitted source.

A ``pipeline.yaml`` is untrusted (LLM-written, sometimes third-party), and
its strings reach emitted ``.py`` and ``.ts`` files: literals, constants,
prompts, docstrings and comments. A bidi override (U+202E), a zero-width
space (U+200B) or a BOM (U+FEFF) written raw makes the code read
differently to a reviewer than it runs, the Trojan Source shape
(CVE-2021-42574). And a lone surrogate (PyYAML turns ``"\\ud800"`` into
one) crashed emit with UnicodeEncodeError partway through writing files.

Every such character is now an escape. Literals still round-trip the
exact value, and visible characters, emoji included, stay raw.
"""

from __future__ import annotations

import ast
import json
import re
import shutil
import subprocess
import unicodedata
from pathlib import Path
from typing import Any

import pytest

from rote.adapters import get_adapter
from rote.ir import Pipeline

INVISIBLE = {
    "rlo": "admin‮//gnp.exe",
    "zwsp": "zero​width",
    "bom": "﻿bom",
    "lone_surrogate": "half\ud800pair",
}

_CATEGORIES = {"Cc", "Cf", "Zl", "Zp", "Cs"}


def _pipeline(value: str, *, mcp: bool) -> Pipeline:
    node: dict[str, Any] = {
        "id": "step",
        "kind": "external_call",
        "description": f"Look {value} up.\nSecond {value} line.",
        "inputs": {"query": "pipeline.input.query"},
    }
    if mcp:
        node["mcp"] = {"server": "vendor", "tool": "lookup", "args": {"q": {"literal": value}}}
    else:
        node["impl"] = "extracted/vendor.py:lookup"
        node["constants"] = {"label": value}
    return Pipeline.model_validate(
        {
            "name": "visible",
            "description": f"Pipeline {value}.",
            "input": {"type": "In", "required": ["query"], "optional": []},
            "nodes": [node],
            "edges": [],
            "entry_nodes": ["step"],
            "exit_nodes": ["step"],
        }
    )


def _invisible_in(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return [
        f"{path.name}: U+{ord(ch):04X}"
        for ch in text
        if unicodedata.category(ch) in _CATEGORIES and ch not in "\n\t"
    ]


@pytest.mark.parametrize("value", list(INVISIBLE.values()), ids=list(INVISIBLE))
@pytest.mark.parametrize(
    ("runtime", "mcp"),
    [
        ("dbos", True),
        ("dbos-ts", True),
        ("cloudflare", True),
        ("inngest", True),
        ("dbos", False),
        ("python", False),
        ("temporal", False),
        ("dbos-ts", False),
    ],
)
def test_no_invisible_character_reaches_emitted_source(
    runtime: str, mcp: bool, value: str, tmp_path: Path
) -> None:
    out = tmp_path / runtime
    get_adapter(runtime).emit(_pipeline(value, mcp=mcp), out)
    sources = [p for p in out.rglob("*") if p.suffix in (".py", ".ts") and p.is_file()]
    assert sources
    found = [hit for path in sources for hit in _invisible_in(path)]
    assert found == []
    for path in sources:
        if path.suffix == ".py":
            ast.parse(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("value", list(INVISIBLE.values()), ids=list(INVISIBLE))
def test_python_literals_round_trip_the_exact_value(value: str, tmp_path: Path) -> None:
    """The escaped literal still evaluates to the value, lone surrogate
    included: for an MCP argument and for a constant."""
    out = tmp_path / "dbos"
    get_adapter("dbos").emit(_pipeline(value, mcp=True), out)
    main = (out / "main.py").read_text(encoding="utf-8")
    (line,) = [ln.strip() for ln in main.split("\n") if ln.strip().startswith("arguments = ")]
    scope: dict[str, Any] = {"payload": {"query": "x"}}
    exec(line, scope)  # noqa: S102 — our own emitted literal
    assert scope["arguments"] == {"q": value}

    out = tmp_path / "python"
    get_adapter("python").emit(_pipeline(value, mcp=False), out)
    main = (out / "main.py").read_text(encoding="utf-8")
    match = re.search(r'"label": (".*?")\}', main)
    assert match is not None, main
    assert ast.literal_eval(match.group(1)) == value


@pytest.mark.parametrize("value", list(INVISIBLE.values()), ids=list(INVISIBLE))
def test_ts_literals_round_trip_the_exact_value(value: str, tmp_path: Path) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    out = tmp_path / "dbos-ts"
    get_adapter("dbos-ts").emit(_pipeline(value, mcp=True), out)
    module = next(out.rglob("step.ts")).read_text(encoding="utf-8")
    match = re.search(r"^\s*const args = (.*);(?:\s*//.*)?$", module, re.MULTILINE)
    assert match is not None, module
    script = f"const payload = {{}}; console.log(JSON.stringify({match.group(1)}));"
    printed = subprocess.run(
        [node, "-e", script], capture_output=True, text=True, check=True
    ).stdout
    assert json.loads(printed) == {"q": value}
