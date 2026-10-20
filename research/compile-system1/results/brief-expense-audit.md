# rote System-1 brief (machine-generated; NOT part of the skill)

rote produced this file deterministically (plus one small-model call) before
your run. It is a pre-analysis of the skill, not skill text: never cite it in
a `# skill:` comment, and never compile it as a step. Use it to skip
re-deriving what it already settles; where it is wrong, the skill wins.

Confidence below 0.8 means the classifier was unsure: decide those yourself.

## Steps and proposed nodes

| # | SKILL.md section | step | proposed nodes (kind, confidence) | MANDATORY | MCP |
|---|---|---|---|---|---|
| 0 | Step 1: Convert every line to USD | Step 1: Convert every line to USD | `convert_lines_to_usd` pure_function (0.85) |  |  |
| 1 | Step 2: Check for duplicate submissions (MANDATORY) | Step 2: Check for duplicate submissions (MANDATORY) | `flag_duplicate_submissions` pure_function (0.95) | yes |  |
| 2 | Step 3: Check for late submissions (MANDATORY) | Step 3: Check for late submissions (MANDATORY) | `flag_late_submissions` pure_function (0.95) | yes |  |
| 3 | Step 4: Audit each employee's report | Step 4: Audit each employee's report | `audit_employee_reports` pure_function (0.80) |  |  |
| 4 | Step 5: Summarize the batch | Step 5: Summarize the batch | `summarize_batch_and_review_queue` pure_function (0.95) |  |  |
| 5 | Step 6: Controller review | Step 6: Controller review | `controller_review_gate` hitl_gate (0.95) |  |  |
| 6 | Step 7: File the audit report | Step 7: File the audit report | `build_audit_report` pure_function (0.85); `save_audit_report_to_finance_folder` external_call (0.80) |  |  |

## Literal thresholds and constants the prose states

- SKILL.md:47: `more than 60`
- SKILL.md:47: `60 days`
- SKILL.md:59: `$25.00 or more`
- references/expense-policy.md:20: `id (`preapproval_id``
- references/expense-policy.md:27: `$25.00 or more`
- references/expense-policy.md:29: `under $25.00`
- references/expense-policy.md:35: `more than 60`
- references/expense-policy.md:35: `60 days`

## Rule tables, already lifted to Python (verify, then reuse)

```python
"""Rule tables lifted from the skill bundle by rote System-1 (no inference)."""

from decimal import Decimal

# skill: references/expense-policy.md:12 "Category caps" | Category | Expense tool code | Cap | Notes
# skill: references/fx-rates.md:11 "2026-09" | Currency | USD per unit
# skill: references/fx-rates.md:21 "2026-08" | Currency | USD per unit
CAP = {'airfare': Decimal('900.00'), 'lodging': Decimal('325.00'), 'meals': Decimal('75.00'), 'ground_transport': Decimal('150.00'), 'software': Decimal('100.00'), 'office_supplies': Decimal('250.00')}

USD_PER_UNIT = {'2026-09': {'USD': Decimal('1.000000'), 'EUR': Decimal('1.105000'), 'GBP': Decimal('1.312500'), 'CAD': Decimal('0.724000'), 'JPY': Decimal('0.006780')}, '2026-08': {'USD': Decimal('1.000000'), 'EUR': Decimal('1.098000'), 'GBP': Decimal('1.301000'), 'CAD': Decimal('0.731000'), 'JPY': Decimal('0.006850')}}

```

{'cost_usd': 0.00084791, 'wall_s': 7.8}
