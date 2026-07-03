"""Gate-less fixture pipeline shared by the raw-Python adapter tests.

Compact but complete: covers every node kind the python adapter can emit
— ``pure_function``, ``external_call`` (with a retry policy and a
timeout), ``llm_judge`` (via ``signature_spec``), and ``agent_loop``
(with a ``loop_body`` sub-node) — plus a parallel entry wave and a
fan-in exit node. Deliberately no ``hitl_gate``: the python adapter
refuses those, and the refusal is tested against the committed BDR
pipeline (which has two gates).

DAG shape (waves as the adapter computes them)::

    wave 1:  fetch_profile ∥ normalize_brief      (parallel entry wave)
    wave 2:  research_loop                        (fan-in of wave 1)
    wave 3:  grade                                (llm_judge)
    wave 4:  final_report                         (fan-in: grade + fetch_profile
                                                   + pipeline input field)

    score_item is a loop_body sub-node of research_loop — it exists as a
    top-level function (testable in isolation) but the pipeline body
    never dispatches it.
"""

from __future__ import annotations

from rote.ir import Pipeline


def build_gateless_pipeline() -> Pipeline:
    return Pipeline.model_validate(
        {
            "name": "research-brief",
            "version": "0.1.0",
            "source_skill": "tests/fixtures/research-brief",
            "description": "Research a topic and grade the findings.",
            "input": {
                "type": "ResearchBrief",
                "required": ["topic", "depth"],
                "optional": [],
            },
            "nodes": [
                {
                    "id": "normalize_brief",
                    "kind": "pure_function",
                    "description": "Normalize the research topic string.",
                    "impl": "extracted/brief.py:normalize_brief",
                    "mandatory": True,
                    "inputs": {"topic": "pipeline.input.topic"},
                },
                {
                    "id": "fetch_profile",
                    "kind": "external_call",
                    "description": "Fetch the topic profile from the vendor API.",
                    "impl": "extracted/profile.py:fetch_profile",
                    "timeout": "5m",
                    "retry": {
                        "max": 2,
                        "backoff": "exponential",
                        "retry_on": ["rate_limit", "network"],
                    },
                    "inputs": {"brief": "pipeline.input"},
                },
                {
                    "id": "score_item",
                    "kind": "pure_function",
                    "description": "Score a single finding (loop-body sub-node).",
                    "impl": "extracted/brief.py:score_item",
                },
                {
                    "id": "research_loop",
                    "kind": "agent_loop",
                    "description": "Iteratively research the topic until enough findings.",
                    "tools": ["web_search", "fetch_page"],
                    "loop_body": ["score_item"],
                    "termination": {
                        "condition": "enough findings collected",
                        "max_iterations": 5,
                    },
                    "inputs": {
                        "profile": "fetch_profile.output",
                        "topic": "normalize_brief.output.topic",
                        "depth": "pipeline.input.depth",
                    },
                },
                {
                    "id": "grade",
                    "kind": "llm_judge",
                    "description": "Grade the research findings for quality.",
                    "signature_spec": {
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "findings": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                }
                            },
                            "required": ["findings"],
                        },
                        "output_schema": {
                            "type": "object",
                            "properties": {
                                "grade": {"type": "integer"},
                                "rationale": {"type": "string"},
                            },
                            "required": ["grade", "rationale"],
                        },
                        "prompt": "Grade these findings: {{ findings }}",
                        "client": "anthropic",
                    },
                    "inputs": {"findings": "research_loop.output.findings"},
                },
                {
                    "id": "final_report",
                    "kind": "pure_function",
                    "description": "Assemble the final report from all upstream results.",
                    "impl": "extracted/report.py:build_report",
                    "inputs": {
                        "grade": "grade.output.grade",
                        "profile": "fetch_profile.output.profile",
                        "topic": "pipeline.input.topic",
                    },
                },
            ],
            "edges": [
                {"from": "normalize_brief", "to": "research_loop"},
                {"from": "fetch_profile", "to": "research_loop"},
                {"from": "research_loop", "to": "grade"},
                {"from": "grade", "to": "final_report"},
                {"from": "fetch_profile", "to": "final_report"},
            ],
            "entry_nodes": ["normalize_brief", "fetch_profile"],
            "exit_nodes": ["final_report"],
        }
    )
