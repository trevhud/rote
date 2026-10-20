---
name: saas-spend-audit
description: >-
  Monthly SaaS spend audit for finance ops. Pulls the month's software card
  spend from Ramp, the vendor register from Google Sheets and per-app seat
  usage from Okta, works out which vendor every charge belongs to, runs
  the audit rules, gets the finance lead's keep / cancel / renegotiate
  call on every flagged vendor, and logs the actions in the Procurement
  Requests sheet. Use for "run the SaaS audit", "software spend review
  for <month>", or "who are we paying for tools we don't use".
---

# Monthly SaaS spend audit

Run on the 3rd business day of the month, for the month that just closed.
Inputs: the first and last day of that month (`YYYY-MM-DD`).

## Data

Pull these three at the same time. None of them needs the others.

| Source | Tool | Details |
| --- | --- | --- |
| Card spend | `list_transactions` (Ramp MCP) | `from_date` / `to_date` = the month, `category` = `software` |
| Vendor register | `read_sheet_values` (Google Sheets MCP) | spreadsheet `1VendorRegisterExampleSheetId000000000000`, range `Register!A:J` |
| Seat usage | `get_app_usage` (Okta MCP) | once per Okta app id listed in the register, `days` = 30 |

The register has one row per approved vendor: vendor, card descriptor,
category, budget owner, contracted monthly price (USD), paid seats, Okta
app id (blank if the tool isn't behind Okta), renewal date, auto-renew
(Y/N), notes. Skip the header row and any row with a blank vendor.

Seat usage needs the register's Okta app ids, so read the register first,
then fan out over the ids.

## Match charges to vendors

Card merchant strings are messy (`SQ *FIGMA`, `ATLASSIAN* JIRA SOFTWARE`,
`PADDLE.NET* LINEAR`). Group the month's transactions by merchant
descriptor first, then work out the vendor for each descriptor:

1. If the descriptor matches a register row's card descriptor exactly
   (ignoring case), that's the vendor. Done.
2. Otherwise, use judgment: payment processor prefixes (`SQ *`, `PADDLE.NET*`,
   `FS *`, `PAYPAL *`) hide the real vendor after the prefix, and product
   names can map to a parent vendor (Jira is Atlassian). Match to a
   register vendor if you're confident.
3. If it's clearly not any register vendor, name the vendor and give it a
   category from the register's list. Mark it unregistered.

## Rules

Every rule runs on every vendor, every month. MANDATORY: don't skip a rule
because the vendor is small or "we already know about it". The finance
lead uses the full list to track what's been fixed.

| ID | Rule | Condition | Severity | Suggested action |
| --- | --- | --- | --- | --- |
| R1 | Unregistered vendor | Charges this month, vendor not in the register | high | Find an owner, security review |
| R2 | Over contract | Month's charges more than 10% above contracted monthly price | medium | Renegotiate or correct the register |
| R3 | Idle seats | Okta active users in the last 30 days below 80% of paid seats | medium | Right-size at renewal |
| R4 | Overlapping tools | Two or more vendors with spend this month in the same category | low | Consolidate |
| R5 | Renewal coming | Renewal date within 45 days of the month end and auto-renew = Y | info | Decide before it renews |
| R6 | Zombie contract | Register vendor with no charges in the last month and no Okta app activity | low | Confirm cancelled, remove from register |

Savings estimate per finding: R2 = charges minus contracted price, R3 =
idle seats times (contracted price / paid seats). Other rules have no
estimate.

## Report

Write the audit as markdown, in this layout:

```
# SaaS spend audit: {month}

Total software spend: ${total_spend}  ({vendor_count} vendors, {unregistered_count} unregistered)
Estimated monthly savings if every finding is fixed: ${savings}

## Findings
| Vendor | Owner | Rule | Severity | Detail | Est. savings |
|---|---|---|---|---|---|
{one row per finding, sorted by severity (high first) then vendor}

## Renewals in the next 45 days
{vendor, renewal date, contracted price}
```

## Finance lead review

Send the report to the finance lead in #finance-ops and wait. For every
vendor with a finding they answer keep, cancel, or renegotiate, with an
optional note. Nothing gets cancelled without their call.

## Log the actions

Append one row per cancel or renegotiate decision to the Procurement
Requests sheet (`append_sheet_values`, spreadsheet
`1ProcurementRequestsExampleSheetId00000000`, range `Requests!A:F`):
date, vendor, decision, budget owner, estimated savings, note. Keep
decisions are not logged. Procurement only works from that sheet, so
every cancel and renegotiate has to land there.
