"""Replay the preserved self-compilation through Python runtime emitters.

Only generated schema models execute here; no inference or workflow runs.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from rote.adapters import get_adapter
from rote.ir import load_pipeline

SELF_COMPILE = (
    Path(__file__).resolve().parent.parent
    / "examples/rote-compile/runs/2026-09-04-selfcompile/compiled/pipeline.yaml"
)


@pytest.mark.parametrize("runtime", ["python", "dbos", "temporal"])
def test_selfcompile_emits_independent_signature_schemas(runtime: str, tmp_path: Path) -> None:
    pipeline = load_pipeline(SELF_COMPILE)
    node = next(node for node in pipeline.nodes if node.id == "classify_step")
    assert node.signature_spec is not None
    original_schema = json.loads(json.dumps(node.signature_spec.output_schema))
    written = get_adapter(runtime).emit(pipeline, tmp_path / runtime)

    # Use a fresh interpreter so Pydantic resolves deferred annotations against
    # this actual generated module, without another test's signatures package.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.util
import json
import sys

from pydantic import ValidationError

spec = importlib.util.spec_from_file_location("selfcompile_classify", sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

step = {
    "step_id": "read_skill_bundle",
    "name": "Read skill bundle",
    "description": "Read SKILL.md and its reference files",
    "source_section": "Phase 1: Skill Analysis",
    "phase": "1",
}
input_payload = {
    "step": step,
    "skill_md": "Read the source skill and reference files.",
    "reference_files": {"implementation.md": "Read files as UTF-8."},
}
output_payload = {
    "step": step,
    "kind": "pure_function",
    "justification": "Reading fixed local paths requires no model judgment.",
    "suggest_mandatory": True,
}
input_model = module.ClassifyStepInput.model_validate(input_payload)
output_model = module.ClassifyStepOutput.model_validate(output_payload)
assert input_model.model_dump() == input_payload
assert output_model.model_dump() == output_payload
assert type(input_model.step) is not type(output_model.step)

# This is the actual difference in the preserved input/output definitions:
# the input forbids extra fields; the output does not forbid them.
extra_step = {**step, "extra_context": "retained by the classifier upstream"}
try:
    module.ClassifyStepInput.model_validate({**input_payload, "step": extra_step})
except ValidationError as error:
    assert any(
        item["loc"] == ("step", "extra_context") and item["type"] == "extra_forbidden"
        for item in error.errors()
    )
else:
    raise AssertionError("Input StepDescription must reject extra fields")
module.ClassifyStepOutput.model_validate({**output_payload, "step": extra_step})

assert module.OUTPUT_JSON_SCHEMA == json.load(sys.stdin)
""",
            str(written["signatures/classify_step"]),
        ],
        input=json.dumps(original_schema),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert node.signature_spec.output_schema == original_schema
