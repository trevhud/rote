# Audit Report Format

Payroll reads this file, so keep the layout exactly as below. Money is
always USD with two decimals and a `$` sign.

```markdown
# Expense audit: <period>

Status: **final** (reviewed by <controller>)    <- or **draft** (awaiting controller review)

| | USD |
| --- | --- |
| Reimbursable | $x |
| Pending review | $x |
| Returned to employee | $x |
| Batch total | $x |

## Employees

| Employee | Department | Lines | Total | Outcome |
| --- | --- | --- | --- | --- |
| ... | ... | ... | $x | auto-approved / reviewed / pending review |

## Review queue

| Expense | Employee | Rule | Severity | Detail | Decision |
| --- | --- | --- | --- | --- | --- |
| ... | ... | ... | review | ... | approved / rejected / pending |
| ... | ... | ... | block | ... | returned |
```

Rules for the decision column:

- a **block** row always reads `returned`
- a **review** row reads the controller's decision, or `pending` when the
  report is still a draft

A line with several findings appears once per finding. It is reimbursable
only when every one of its findings was approved.

An employee's outcome is `auto-approved` when their report had no
findings, `reviewed` once the controller decided every review item on
it, and `pending review` otherwise.
