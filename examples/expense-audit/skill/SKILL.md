---
name: expense-audit
description: >-
  Month-end audit of employee expense reports. Takes the month's export
  from the expense tool, converts every line to USD at the finance
  month-end rates, checks each employee's report against the travel and
  expense policy (category caps, receipts, pre-approvals, duplicates,
  late submissions), auto-approves clean reports, routes everything else
  to the controller for a decision, and files the audit report for
  payroll. Use when someone asks to "run the expense audit", "close out
  expenses for <month>", or "check this month's expense reports".
---

# Month-End Expense Audit

Run this after the expense tool's submission window closes. The input is
the month's export: one row per expense line, plus the reporting period
(`YYYY-MM`). Payroll reimburses from the filed report, so every number in
it has to be traceable to a rule in
[references/expense-policy.md](references/expense-policy.md).

Work through the steps in order. Steps 1 to 3 do not depend on each
other, so do them together.

## Step 1: Convert every line to USD

Look up the period's month-end rates in
[references/fx-rates.md](references/fx-rates.md) and convert each line's
`amount` from its `currency` to USD. Use the month-end rate of the
reporting period for every line, even lines dated in an earlier month.
Round to the cent only after converting. If a line uses a currency that
isn't in the table, stop and ask finance for the rate. Do not look one
up online. Payroll must match the general ledger.

## Step 2: Check for duplicate submissions (MANDATORY)

Two lines are duplicates when they have the same employee, merchant
(ignore case), amount, currency, and date. Keep the first one submitted
(lowest expense id) and flag every later copy. Duplicates are a **block**:
they go back to the employee and the controller cannot approve them.

This check is mandatory. Run it on every line, every month, even when the
batch looks small enough to eyeball.

## Step 3: Check for late submissions (MANDATORY)

The policy window is 60 days. Flag any line dated more than 60 days
before the last day of the reporting period, and any line dated after the
period closed. Late lines are **review**, not block: the controller can
still approve them.

## Step 4: Audit each employee's report

Group the converted lines by employee. For each employee, apply every
rule in [references/expense-policy.md](references/expense-policy.md) to
every line:

- category cap exceeded (lodging is capped per night)
- receipt missing on a line of $25.00 or more
- entertainment without a pre-approval id
- category not in the policy table
- the duplicate and late flags from steps 2 and 3

Then check the report total against the per-report limit. A report with
no findings and a total within the limit is **auto-approved**. Do not ask
the controller about it.

## Step 5: Summarize the batch

Total the batch in USD, list the auto-approved employees, and build the
review queue: every finding from step 4, one row per finding, with the
expense id, employee, rule, severity, and a one-line detail. Sort the
queue by employee, then expense id.

## Step 6: Controller review

Post the review queue to the controller and wait for a decision on every
**review** item: approve or reject. Block items are listed for
information only. They go back to the employee no matter what. If the
controller names an expense id that isn't in the queue, ask again rather
than guessing what they meant.

Approvals usually arrive within a day. Hold the audit open for up to
three days before escalating to the finance manager.

## Step 7: File the audit report

Write the report in the layout in
[references/report-format.md](references/report-format.md) and save it
to the finance shared folder as `<period>-expense-audit.md`. A line is
reimbursable when it has no findings, or when all of its findings are
review items the controller approved. Blocked and rejected lines are
returned to the employee. Payroll only pays from a report marked
**final**.
