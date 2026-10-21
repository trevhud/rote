"""A judge whose skill step asks it to quote its evidence checks each quote.

The rule lives twice, once per language: ``quote_problems`` in the Python
inference helper and ``quoteProblems`` in ``_ts_common.QUOTE_HELPER_TS``.
One table of cases runs through both, so a judge checks its evidence the
same way on every runtime. No inference: the judge call is a fake.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from rote.adapters._py_common import emit_signature_module
from rote.adapters._ts_common import QUOTE_HELPER_TS, emit_ts_signature_module
from rote.inference import _runtime_helper as helper
from rote.ir import LLMSignature, Node

INPUTS = {
    "product": "Queues",
    "results": [
        {
            "title": "Pull consumers",
            "text": "Pull consumers can now read  5,000 messages\nper second",
        },
        {"title": "Pricing", "text": "billed at $0.40 / million operations"},
    ],
    "count": 3,
}

#: (answer text, the problems both languages must report).
CASES: list[tuple[str, list[str]]] = [
    ('low: "read 5,000 messages per second"', []),
    # Whitespace differs on both sides: the answer's double space, the input's line break.
    ('low: "read 5,000  messages per second"', []),
    ("low: “billed at $0.40 / million operations”", []),
    (
        'high: "requires the Workers Paid plan"',
        ['the quote "requires the Workers Paid plan" is not in its inputs'],
    ),
    (
        'mixed: "Pull consumers" and "not on the Free plan"',
        ['the quote "not on the Free plan" is not in its inputs'],
    ),
    (
        "no quotes at all",
        [
            "it quotes nothing from its inputs: quote the evidence it rests on in double "
            "quotes, exactly as the inputs write it"
        ],
    ),
    # Under four characters is not a quote: the answer still quotes nothing.
    (
        'tiny "R2"',
        [
            "it quotes nothing from its inputs: quote the evidence it rests on in double "
            "quotes, exactly as the inputs write it"
        ],
    ),
]


@pytest.mark.parametrize(("text", "expected"), CASES)
def test_python_rule(text: str, expected: list[str]) -> None:
    assert helper.quote_problems(text, INPUTS) == expected


def test_typescript_rule_matches_python(tmp_path: Path) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    script = tmp_path / "check.ts"
    script.write_text(
        QUOTE_HELPER_TS
        + f"\nconst inputs = {json.dumps(INPUTS)};\n"
        + f"const cases: string[] = {json.dumps([t for t, _ in CASES])};\n"
        + "console.log(JSON.stringify(cases.map((t) => quoteProblems(t, inputs))));\n"
        + 'console.log(JSON.stringify(quoteRetryNote(["a", "b"])));\n'
    )
    out = subprocess.run(
        [node, str(script)], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    assert json.loads(out[0]) == [expected for _, expected in CASES]
    assert json.loads(out[1]) == helper.quote_retry_note(["a", "b"])


def _fake_calls(monkeypatch: pytest.MonkeyPatch, answers: list[dict[str, Any]]) -> list[str]:
    prompts: list[str] = []

    def structured_call(**kw: Any) -> tuple[dict[str, Any], list[Any]]:
        prompts.append(kw["prompt"])
        return answers.pop(0), []

    monkeypatch.setattr(helper, "structured_call", structured_call)
    monkeypatch.setattr(helper, "select_provider", lambda *a, **k: "api")
    return prompts


def _judge(**kw: Any) -> dict[str, Any]:
    return helper.call_judge(
        node_id="rate", client="anthropic", model="m", prompt="P", output_schema={}, **kw
    )


def test_a_failing_quote_is_re_asked_once_naming_it(monkeypatch: pytest.MonkeyPatch) -> None:
    bad = {"risk": "high", "reason": '"requires the Workers Paid plan"'}
    good = {"risk": "low", "reason": '"read 5,000 messages per second"'}
    prompts = _fake_calls(monkeypatch, [bad, good])
    assert _judge(evidence_field="reason", inputs=INPUTS) == good
    assert prompts == [
        "P",
        "P"
        + helper.quote_retry_note(
            ['the quote "requires the Workers Paid plan" is not in its inputs']
        ),
    ]


def test_a_second_failure_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    bad = {"reason": '"requires the Workers Paid plan"'}
    _fake_calls(monkeypatch, [dict(bad), dict(bad)])
    with pytest.raises(RuntimeError, match="judge rate: its quoted evidence is not in its inputs"):
        _judge(evidence_field="reason", inputs=INPUTS)


def test_without_an_evidence_field_nothing_is_checked(monkeypatch: pytest.MonkeyPatch) -> None:
    prompts = _fake_calls(monkeypatch, [{"reason": "no quotes"}])
    assert _judge() == {"reason": "no quotes"}
    assert prompts == ["P"]


def _node(evidence: str | None) -> Node:
    spec = LLMSignature(
        input_schema={
            "type": "object",
            "properties": {"results": {"type": "array"}},
            "required": ["results"],
        },
        output_schema={
            "type": "object",
            "properties": {"risk": {"type": "string"}, "reason": {"type": "string"}},
            "required": ["risk", "reason"],
        },
        prompt="Rate {{ results }}",
        evidence_field=evidence,
    )
    return Node(
        id="rate_risk",
        kind="llm_judge",
        description="rate",
        inputs={"results": "pipeline.input.results"},
        signature_spec=spec,
    )


def test_python_judges_pass_the_evidence_field_only_when_set() -> None:
    def emit(node: Node) -> str:
        return emit_signature_module(
            node,
            anthropic_default_model="m",
            openai_default_model="m",
            generated_by="t",
            regen_command="t",
            context_note="t",
        )

    on = emit(_node("reason"))
    assert '            evidence_field="reason",\n' in on
    assert "            inputs=inputs.model_dump(by_alias=True),\n" in on
    off = emit(_node(None))
    assert "evidence_field" not in off


@pytest.mark.parametrize("client", ["anthropic", "openai", "workers-ai"])
def test_ts_judges_wrap_the_call_with_the_check(client: str) -> None:
    node = _node("reason")
    assert node.signature_spec is not None
    node.signature_spec.client = client
    module = emit_ts_signature_module(
        node, anthropic_default_model="m", openai_default_model="m", generated_by="t"
    )
    assert "async function rateRiskOnce(\n    rawInput: unknown,\n    prompt: string,\n" in module
    assert "export async function rateRisk(" in module
    assert module.count("export async function rateRisk(") == 1
    assert 'const field = "reason";' in module
    assert "quoteRetryNote(problems)" in module
    assert "interpolate(PROMPT, input as Record<string, unknown>)" in module
    off = emit_ts_signature_module(
        _node(None), anthropic_default_model="m", openai_default_model="m", generated_by="t"
    )
    assert "quoteProblems" not in off and "rateRiskOnce" not in off


def test_the_ir_keeps_evidence_on_a_string_output() -> None:
    with pytest.raises(ValueError, match="must be a string property of output_schema"):
        _node("risk_level")
    with pytest.raises(ValueError, match="must be an identifier"):
        _node("reason; drop")


def test_the_ts_wrapper_re_asks_once_then_throws(tmp_path: Path) -> None:
    """Run the emitted wrapper against a stub call: the same behavior as Python's."""
    node_bin = shutil.which("node")
    if node_bin is None:
        pytest.skip("node is not installed")
    module = emit_ts_signature_module(
        _node("reason"), anthropic_default_model="m", openai_default_model="m", generated_by="t"
    )
    wrapper = module[module.index("const QUOTE_RE") :]
    stub = """
const PROMPT = "Rate {{ results }}";
const RateRiskInput = { parse: (x: unknown) => x as Record<string, unknown> };
type RateRiskOutput = Record<string, unknown>;
function interpolate(t: string, v: Record<string, unknown>): string {
    return t.replace("{{ results }}", JSON.stringify(v.results));
}
const answers: RateRiskOutput[] = JSON.parse(process.argv[2]);
const prompts: string[] = [];
async function rateRiskOnce(rawInput: unknown, prompt: string, env: { K: string }) {
    prompts.push(prompt);
    return answers.shift() as RateRiskOutput;
}
"""
    run = """
rateRisk(INPUTS, { K: "k" }).then(
    (out) => console.log(JSON.stringify({ out, prompts })),
    (err) => console.log(JSON.stringify({ error: String(err.message), prompts })),
);
""".replace("INPUTS", json.dumps(INPUTS))
    script = tmp_path / "wrap.ts"
    script.write_text(stub + wrapper.replace("export async function", "async function") + run)

    def go(answers: list[dict[str, Any]]) -> dict[str, Any]:
        out = subprocess.run(
            [node_bin, str(script), json.dumps(answers)], capture_output=True, text=True, check=True
        )
        return json.loads(out.stdout)

    bad = {"reason": '"requires the Workers Paid plan"'}
    good = {"reason": '"read 5,000 messages per second"'}
    first = go([bad, good])
    prompt = "Rate " + json.dumps(INPUTS["results"], separators=(",", ":"))
    assert first["out"] == good
    assert first["prompts"] == [
        prompt,
        prompt
        + helper.quote_retry_note(
            ['the quote "requires the Workers Paid plan" is not in its inputs']
        ),
    ]
    twice = go([bad, bad])
    assert twice["error"].startswith(
        "judge rate_risk: its quoted evidence is not in its inputs after a re-ask"
    )
    assert go([good])["prompts"] == [prompt]
