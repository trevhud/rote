"""A judge input validates the same way on Python and TypeScript runtimes.

The Pydantic emitter forbids unknown keys only under
``additionalProperties: false`` and otherwise drops them; the Zod emitter
used to put ``.strict()`` on every object. So an upstream node whose
objects carried one undeclared field ran on dbos and python and failed
every judge call on Cloudflare, Inngest and DBOS-TS.

This runs one judge input schema through both real validators: the
Pydantic model the python adapter emits, imported and called, and the
Zod schema the Cloudflare adapter emits, evaluated by real ``zod`` under
Node. Each payload must get the same verdict and the same parsed value.
No inference: nothing here calls a model.
"""

from __future__ import annotations

import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from rote.adapters._ts_common import json_schema_to_zod
from rote.adapters.cloudflare import CloudflareAdapter
from rote.adapters.python import PythonAdapter
from rote.ir import Pipeline

pytestmark = pytest.mark.slow

NODE_ID = "parity_judge"
PASCAL = "ParityJudge"

INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"thread": {"$ref": "#/$defs/Thread"}},
    "required": ["thread"],
    "additionalProperties": False,
    "$defs": {
        # Open: an upstream object may carry fields the judge doesn't read.
        "Thread": {
            "type": "object",
            "properties": {
                "thread_id": {"type": "string"},
                "meta": {"type": "object"},  # free-form: keeps every key
                "sender": {"$ref": "#/$defs/Sender"},
            },
            "required": ["thread_id", "meta", "sender"],
        },
        # Closed: an extra key here is a contract violation on every runtime.
        "Sender": {
            "type": "object",
            "properties": {"email": {"type": "string"}},
            "required": ["email"],
            "additionalProperties": False,
        },
    },
}

_BASE = {"thread_id": "t-1", "meta": {"labels": ["quote"], "n": 2}, "sender": {"email": "a@x"}}

#: (case, payload, expected verdict). The verdict is asserted on both
#: runtimes, and the parsed values are compared to each other.
CASES: list[tuple[str, dict[str, Any], bool]] = [
    ("exact", {"thread": _BASE}, True),
    ("extra field on an open nested object", {"thread": {**_BASE, "account": "Acme"}}, True),
    (
        "extra field on a closed nested object",
        {"thread": {**_BASE, "sender": {"email": "a@x", "cc": "b"}}},
        False,
    ),
    ("extra top-level key under a closed root", {"thread": _BASE, "rubric": "x"}, False),
    ("missing required nested field", {"thread": {"meta": {}, "sender": {"email": "a@x"}}}, False),
]


def _pipeline() -> Pipeline:
    return Pipeline.model_validate(
        {
            "name": "schema-parity",
            "input": {"type": "In"},
            "nodes": [
                {
                    "id": "source",
                    "kind": "pure_function",
                    "description": "Produce threads.",
                    "impl": "extracted/source.py:threads",
                },
                {
                    "id": NODE_ID,
                    "kind": "llm_judge",
                    "description": "Classify a thread.",
                    "fan_out": True,
                    "inputs": {"thread": "source.output.threads"},
                    "signature_spec": {
                        "input_schema": INPUT_SCHEMA,
                        "output_schema": {
                            "type": "object",
                            "properties": {"step": {"type": "integer"}},
                            "required": ["step"],
                        },
                        "prompt": "Classify {{ thread }}.",
                    },
                },
            ],
            "edges": [{"from": "source", "to": NODE_ID, "fan_out": True}],
        }
    )


def _python_verdicts(out: Path) -> list[tuple[bool, Any]]:
    """Validate each payload with the emitted Pydantic Input model."""
    PythonAdapter().emit(_pipeline(), out)
    sys.path.insert(0, str(out))
    try:
        module = importlib.import_module(f"signatures.{NODE_ID}")
        model = getattr(module, f"{PASCAL}Input")
        results: list[tuple[bool, Any]] = []
        for _, payload, _ in CASES:
            try:
                results.append((True, model(**payload).model_dump(by_alias=True)))
            except Exception:  # pydantic.ValidationError
                results.append((False, None))
        return results
    finally:
        sys.path.remove(str(out))
        for name in [m for m in sys.modules if m == "signatures" or m.startswith("signatures.")]:
            del sys.modules[name]


def _zod_verdicts(out: Path, tmp: Path) -> list[tuple[bool, Any]]:
    """Parse each payload with the Zod schema the TS judge module exports."""
    CloudflareAdapter().emit(_pipeline(), out)
    zod_expr = json_schema_to_zod(INPUT_SCHEMA, indent=0)
    # The expression evaluated below is the one the emitted judge parses with.
    emitted = (out / "src" / "signatures" / f"{NODE_ID}.ts").read_text(encoding="utf-8")
    assert f"export const {PASCAL}Input = {zod_expr};" in emitted

    zod_range = json.loads((out / "package.json").read_text(encoding="utf-8"))["dependencies"][
        "zod"
    ]
    proj = tmp / "zod-parity"
    proj.mkdir()
    (proj / "package.json").write_text(
        json.dumps({"name": "zod-parity", "private": True, "type": "module"}), encoding="utf-8"
    )
    subprocess.run(
        ["npm", "install", "--no-audit", "--no-fund", f"zod@{zod_range}"],
        cwd=proj,
        check=True,
        capture_output=True,
        text=True,
    )
    payloads = json.dumps([payload for _, payload, _ in CASES])
    # The module's JSON_VALUE (JSON_VALUE_ZOD_TS), minus its TS types: this
    # harness is plain JS.
    json_value = "const JSON_VALUE = z.custom(() => true);\n" if "JSON_VALUE" in zod_expr else ""
    assert not json_value or "const JSON_VALUE = z.custom<JsonValue>(() => true);" in emitted
    (proj / "check.mjs").write_text(
        'import { z } from "zod";\n'
        f"{json_value}"
        f"const Input = {zod_expr};\n"
        f"const payloads = {payloads};\n"
        "console.log(JSON.stringify(payloads.map((p) => {\n"
        "  const r = Input.safeParse(p);\n"
        "  return r.success ? [true, r.data] : [false, null];\n"
        "})));\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        ["node", "check.mjs"], cwd=proj, check=True, capture_output=True, text=True
    )
    return [tuple(r) for r in json.loads(proc.stdout)]


def test_judge_input_validates_identically_on_python_and_typescript(tmp_path: Path) -> None:
    if shutil.which("node") is None or shutil.which("npm") is None:
        pytest.skip("Node / npm not available")
    python = _python_verdicts(tmp_path / "py")
    zod = _zod_verdicts(tmp_path / "cf", tmp_path)

    for (case, _, expected), py, ts in zip(CASES, python, zod, strict=True):
        assert py[0] is expected, f"python, {case}: accepted={py[0]}"
        assert ts[0] is expected, f"zod, {case}: accepted={ts[0]}"
        # Same parsed value, so downstream code sees the same object: an
        # extra field on an open object is dropped on both, and a
        # free-form object keeps every key on both.
        assert py[1] == ts[1], f"{case}: python={py[1]!r} zod={ts[1]!r}"

    accepted_extra = python[1][1]
    assert "account" not in accepted_extra["thread"]
    assert accepted_extra["thread"]["meta"] == {"labels": ["quote"], "n": 2}
