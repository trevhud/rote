# Example: Expense Audit

The **free first run**. A month-end expense audit skill whose compiled
pipeline ships with *working* extracted code, so `rote emit` + `rote run`
complete end to end with no API key, no subscription, and no tokens. The
other examples show what compilation produces against real vendor APIs;
their extracted modules are `NotImplementedError` stubs until someone
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

From the repo root, after `pip install 'rote-cli[dbos]'`:

```sh
# Plain Python, no gate: a draft report.
rote emit examples/expense-audit/expected/pipeline-draft.yaml --runtime python --out /tmp/expense-audit
rote run /tmp/expense-audit --input examples/expense-audit/input.json

# DBOS, with the controller's decision delivered to the parked gate.
rote emit examples/expense-audit/expected/pipeline.yaml --out /tmp/expense-audit-dbos
rote run /tmp/expense-audit-dbos --input examples/expense-audit/input.json \
  --signal controller_review='{"reviewer": "Priya Shah", "approved_ids": ["EXP-2403", "EXP-2413", "EXP-2420"], "rejected_ids": ["EXP-2410"]}'

# Temporal (needs `rote-cli[temporal]`; first run downloads the dev server).
rote emit examples/expense-audit/expected/pipeline.yaml --runtime temporal --out /tmp/expense-audit-temporal
rote run /tmp/expense-audit-temporal --input examples/expense-audit/input.json \
  --signal controller_review='{"reviewer": "Priya Shah", "approved_ids": ["EXP-2403", "EXP-2413", "EXP-2420"], "rejected_ids": ["EXP-2410"]}'

# The before/after scorecard (fetches live model prices, no inference).
rote eval examples/expense-audit/expected/pipeline.yaml
```

The run prints the totals as JSON and writes the markdown report to
`reports/2026-09-expense-audit.md` under the process's working
directory (the app dir for python and dbos, your shell's directory for
temporal, which runs in-process). Set `EXPENSE_AUDIT_REPORT_DIR` to put
it somewhere else.

Expected results on `input.json`:

| | Draft (python) | Final (dbos, temporal) |
| --- | --- | --- |
| Reimbursable | $2,395.59 | $2,990.98 |
| Pending review | $844.39 | $0.00 |
| Returned to employee | $64.17 | $313.17 |
| Batch total | $3,304.15 | $3,304.15 |

The review queue holds four review items (a meal over the cap, software
over the cap, a late line, entertainment without pre-approval) and two
blocks (a $25.97 ride with no receipt, a duplicate Lyft charge filed as
"lyft"). Blocks go back to the employee whatever the controller says; a
decision that names a block item fails the step with the ids to fix.

## Layout

```
examples/expense-audit/
├── README.md
├── input.json                    # the sample month export (2026-09)
├── skill/
│   ├── SKILL.md                  # the source skill, as an agent would run it
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
```

`expected/` is hand-authored, like the other examples' baselines. There
is no `runs/` snapshot of a real `rote compile` on this skill yet; that
costs inference and is the next thing to add.
