# Held-out benchmark skills

These skills exist to catch overfitting. When you tune the compiler's
rubric or prompts against the five top-level examples, do not read these
skills or their expected pipelines, and do not tune against their scores.
A change that helps the five examples and hurts these is fitting the
examples, not getting better at compiling.

| Skill | Style | What it exercises |
| --- | --- | --- |
| `customer-onboarding` | terse checklist | MCP reads and writes with `mcp.args` literals, a mandatory screen, a numeric terms check beside an `llm_judge` over a free-text note, a CS-lead `hitl_gate` whose payload changes downstream values, a non-MCP `fan_out` |
| `saas-spend-audit` | rules table | three parallel pulls, an MCP-bound `fan_out` (one Okta call per app id, a constant passed through `mcp.args`), a fanned `llm_judge` with shared context, a six-rule `pure_function`, a finance-lead gate |
| `incident-triage` | verbose runbook | parallel entry calls, catalog and matrix lookups as constants, an `llm_judge` hypothesis, the one `agent_loop` (20 minute hard stop, `tool_servers` across two servers), MANDATORY steps, an incident-commander gate |

Bench them on their own, never mixed into the default run:

```sh
rote bench --examples ~/rote/examples/holdout --out /tmp/bench/holdout
```

`--examples examples/` never picks these up: discovery only looks one
level down, and `holdout/` is not itself a skill. The gold labels'
judgment calls are recorded in the PR that added them,
[#149](https://github.com/trevhud/rote/pull/149). `tests/test_examples.py` emits every
`expected/pipeline.yaml` here on every adapter, so the labels stay valid
IR as the schema moves.
