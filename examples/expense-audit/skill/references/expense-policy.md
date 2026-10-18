# Travel and Expense Policy (finance audit rules)

This is the version of the policy finance audits against. The employee
handbook has the friendlier wording. When the two disagree, this page
wins.

All amounts are USD, after conversion at the month-end rate
(see [fx-rates.md](fx-rates.md)).

## Category caps

| Category | Expense tool code | Cap | Notes |
| --- | --- | --- | --- |
| Airfare | `airfare` | $900.00 per ticket | Economy only. Anything above the cap needs a written exception. |
| Lodging | `lodging` | $325.00 per night | The line carries `nights`, and the cap is $325.00 times nights. |
| Meals | `meals` | $75.00 per line | Per person. Team meals are filed by the most senior attendee. |
| Ground transport | `ground_transport` | $150.00 per line | Taxis, rideshare, rail, parking. |
| Software | `software` | $100.00 per line | Above the cap goes through IT procurement instead. |
| Office supplies | `office_supplies` | $250.00 per line | |
| Entertainment | `entertainment` | no cap | Always needs a pre-approval id (`preapproval_id`). |

A line whose category isn't in this table is a **block**: the employee
has to recategorize it.

## Receipts

A receipt is required on every line of **$25.00 or more**. The expense
tool sets `receipt: true` when an image is attached. A missing receipt
is a **block**. Lines under $25.00 don't need one.

## Duplicates and late submissions

- **Duplicate** (same employee, merchant ignoring case, amount, currency,
  and date): every copy after the first is a **block**.
- **Late** (dated more than 60 days before the last day of the period,
  or after the period closed): **review**.

## Report total

A single employee's report over **$2,500.00** for the month always goes
to the controller (severity **review**), even when every line is clean.

## Severity

- **block**: returned to the employee. The controller cannot approve it.
- **review**: the controller approves or rejects it.

A report with no findings at all is auto-approved and paid as filed.
