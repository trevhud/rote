# Example: Expense Audit

The **free first run**. A month-end expense audit skill whose compiled
pipeline ships with *working* extracted code, so `rote emit` + `rote run`
complete end to end with no API key, no subscription, and no tokens. The
other examples show what compilation produces against real vendor APIs.
Their extracted modules are `NotImplementedError` stubs until someone
writes the HubSpot or Gmail client. This one shows the result actually
running.

The work is all arithmetic and rules: convert sixteen expense lines from
five currencies at the finance month-end rates, find duplicates and late
submissions, check every line against the policy's caps and receipt
rules, auto-approve clean reports, and file the audit for payroll. Run
as an agent, every month re-reads the policy and re-derives the same
rules from prose. Compiled, it is a few milliseconds of Python and one
human decision.

## What it demonstrates

- **A parallel entry wave.** `convert_to_usd`, `find_duplicates`, and
  `find_late_submissions` all read the raw export and nothing else, so
  they run concurrently (a thread pool on python, `asyncio.gather` on
  temporal, steps enqueued on a DBOS queue on dbos).
- **`fan_out`.** `audit_report` runs once per employee report, five
  times on the sample batch, and the results come back as a list in
  input order.
- **`mandatory: true`.** The duplicate and late checks, which SKILL.md
  marks MANDATORY, are unconditional code. They can't be skipped on a
  month that "looks small enough to eyeball".
- **Rules lifted out of prose.** Category caps, the $25.00 receipt
  threshold, the $2,500.00 report limit, the 60-day window, and the FX
  table live as constants in `expected/extracted/`, not prompt text.
- **`hitl_gate`.** `controller_review` parks the durable workflow until
  the controller approves or rejects each review item, then the report
  is filed as final.
- **No `llm_judge`.** Nothing in the skill needs judgment once the
  policy is written down, so the compiled pipeline makes zero model
  calls. Roteness 1.0.

## Two IRs, one pipeline

| File | Gate | Runs on |
| --- | --- | --- |
| `expected/pipeline.yaml` | `controller_review` | `dbos`, `temporal` (and the TS runtimes for emission) |
| `expected/pipeline-draft.yaml` | none | `python` too |

The `python` runtime refuses pipelines with a `hitl_gate`: a plain script
can't durably park for days waiting on a human. `pipeline-draft.yaml` is
the same pipeline with the gate removed, so `file_report` files a
**draft** with every review item pending. A test
(`tests/test_expense_audit_example.py`) pins that the two IRs differ by
exactly the gate and its edges.

## Run it

From the repo root, after `uv tool install rote-cli` (or `pip install rote-cli` into a venv).
The DBOS commands need the dbos extra; the first one offers to install it.

```sh
# Plain Python, no gate: a draft report.
rote emit examples/expense-audit/expected/pipeline-draft.yaml --runtime python --out /tmp/expense-audit
rote run /tmp/expense-audit --input examples/expense-audit/input.json

# DBOS, with the controller's decision delivered to the parked gate.
rote emit examples/expense-audit/expected/pipeline.yaml --out /tmp/expense-audit-dbos
rote run /tmp/expense-audit-dbos --input examples/expense-audit/input.json \
  --signal controller_review='{"reviewer": "Priya Shah", "approved_ids": ["EXP-2403", "EXP-2413", "EXP-2420"], "rejected_ids": ["EXP-2410"]}'

# Temporal: needs the temporal extra and a release newer than 0.14.0. The first run downloads the dev server.
rote emit examples/expense-audit/expected/pipeline.yaml --runtime temporal --out /tmp/expense-audit-temporal
rote run /tmp/expense-audit-temporal --input examples/expense-audit/input.json \
  --signal controller_review='{"reviewer": "Priya Shah", "approved_ids": ["EXP-2403", "EXP-2413", "EXP-2420"], "rejected_ids": ["EXP-2410"]}'

# The before/after scorecard (fetches live model prices, no inference).
rote eval examples/expense-audit/expected/pipeline.yaml
```

The run prints the totals as JSON and writes the markdown report to
`reports/2026-09-expense-audit.md` under the app dir on every runtime.
Set `EXPENSE_AUDIT_REPORT_DIR` to put it somewhere else.

Expected results on `input.json`:

| | Draft (python) | Final (dbos, temporal) |
| --- | --- | --- |
| Reimbursable | $2,395.59 | $2,414.34 |
| Pending review | $18.75 | $0.00 |
| Returned to employee | $889.81 | $889.81 |
| Batch total | $3,304.15 | $3,304.15 |

The review queue holds one review item (a late line) and five blocks (a
meal over the cap, software over the cap, entertainment without
pre-approval, a $25.97 ride with no receipt, a duplicate Lyft charge
filed as "lyft"). Blocks go back to the employee whatever the controller
says, so the decision above approving two of them and rejecting a third
changes nothing for those lines. A decision that names an id not in the
queue fails the step with the ids to fix.

The policy gives no severity for a cap breach or a missing pre-approval.
They are blocks because the skill, run as an agent, blocked them in 10
and 11 of 12 recorded runs, and the compiled pipeline should do what the
skill does. `expected/extracted/audit.py` marks both as assumptions.

## Layout

```
examples/expense-audit/
├── README.md
├── input.json                    # the sample month export (2026-09)
├── eval-input.json               # input.json plus the gate payload, for `rote eval --run`
├── scorecard.md                  # measured before/after (rote eval --run)
├── skill/
│   ├── SKILL.md                  # the source skill, as an agent would run it
│   ├── sample-input.json         # the sample run `rote compile` grounds its contracts in
│   └── references/
│       ├── expense-policy.md     # caps, receipts, severities
│       ├── fx-rates.md           # finance month-end rates
│       └── report-format.md      # the layout payroll reads
└── expected/
    ├── pipeline.yaml             # hand-compiled IR, with the controller gate
    ├── pipeline-draft.yaml       # the same IR without the gate
    ├── eval.yaml                 # per-step agent-turn estimates for `rote eval`
    └── extracted/                # working implementations, no stubs
        ├── fx.py
        ├── checks.py
        ├── audit.py
        └── report.py
└── runs/
    └── 2026-10-18-first-agent-run/   # a real `rote compile` of skill/, for comparison
```

`expected/` is hand-authored, like the other examples' baselines.

## Measured, not estimated

`scorecard.md` is `rote eval --run --trials 2` on this example, with
`eval-input.json` (the batch plus the controller's decision). The agent
runs took 42.9 s and $0.22 each, and they disagreed on `report_path`.
The pipeline runs took 9.8 s, most of it DBOS startup, cost $0, and
produced identical output. The static estimate at the top of that file
(3.2 to 6.5 minutes before) is only a model of agent turns. Trust the
Measured section over it. To reproduce, emit the dbos app as above and
run this (it spends subscription tokens on the agent side):

```sh
rote eval examples/expense-audit/expected/pipeline.yaml --run --trials 2 \
  --input examples/expense-audit/eval-input.json --runtime-dir /tmp/expense-audit-dbos
```

## What the real compiler produced

`runs/2026-10-18-first-agent-run/` is a real `rote compile` of this
skill (claude driver, Sonnet, 46 turns, 12 minutes, $2.56). It agrees on
node kinds and on the three-way parallel entry wave, and it puts the
controller gate in the same place. It differs in three ways:

- It folded grouping and auditing into one `apply_policy_rules` node
  instead of fanning `audit_report` out per employee.
- It marked 7 nodes mandatory where the hand-written IR marks 5.
- It left every extracted function as a `NotImplementedError` stub, even
  though this skill's logic is plain Python.

The hand-written IR is the intended shape. The compiler gaps are being
fixed in a separate PR.
