---
name: invoice-push
description: >-
  Automates bulk "Push to Procurement" from a fulfillment platform's
  Cloud Imports screen using a browser-automation plugin. Use this skill
  whenever the user says "push invoices to procurement", "run the
  invoice push", or "process imports for procurement". Verifies
  prerequisites, calculates a 7-day rolling date window ending
  yesterday, clears stale filters, applies Date Imported + Sent To
  Procurement = No filters, sets page size to 90, pushes each qualifying
  invoice via the ⋮ menu, refreshes after each push to capture the Sent
  Date, paginates through all pages, and generates a three-tab report
  (Invoice Detail, Run Summary, Failure Code Reference) saved to the
  Invoice Push Reports Archive in Google Drive. Do NOT trigger on the
  bare word "Run" alone.
---
# invoice-push

**Owner:** Parcel Audit Team
**Tool:** Claude + browser-automation plugin

---

## Trigger Phrases

Activate this skill when the user says any of the following:

- `Push invoices to procurement` ← recommended
- `Push to procurement`
- `Send imports to procurement`
- `Run the invoice push`
- `Process imports for procurement`

⚠️ Do NOT trigger on just `Run` — too generic, may conflict with other skills.

---

## Prerequisites Check

Before any automation begins, verify both:

1. **Browser-automation plugin** is installed and active (Settings → Plugins → browser plugin = enabled, browser open with extension running).
2. **User is logged in** to `https://ops.example.com/app/audit/import` in the browser.

If the plugin is unavailable, respond:

> "I cannot run the invoice push — the browser-automation plugin is not connected. Please go to Settings → Plugins, confirm the plugin is installed and enabled, and that the browser is open with the extension active. Then try again."

Do not proceed until both prerequisites are confirmed.

---

## Date Window Calculation

Calculate dynamically before opening the browser:

- **End date (yesterday):** today − 1 day
- **Start date:** end date − 7 days (7-day rolling window)
- **Format:** `M/D/YYYY` (e.g., `4/22/2026`)

**Example — run date = May 5, 2026:**
- End date = May 4, 2026 → `5/4/2026`
- Start date = April 27, 2026 → `4/27/2026`

Invoices imported today are **never** pushed — log as ⚠️ Skipped: `Imported within 24 hours of run — not eligible.`

---

## Step-by-Step Automation

### Step 1 — Navigate to the Imports Screen

Navigate to: `https://ops.example.com/app/audit/import`

Wait for the table to fully load (tab title = "Imports", rows visible).

---

### Step 2 — Clear All Existing Filter Pills

Check for active filter pills below the search bar.

- Click `×` on each pill to remove, **or** click **Filters** → **Clear all** → close panel.
- Confirm the table resets to unfiltered state before proceeding.

⚠️ Stale filters silently exclude qualifying invoices. Always start clean.

---

### Step 3 — Apply Qualifying Filters

1. Click the **Filters** button (toolbar, top left). Panel slides in from the right.
2. Set **Date Imported**: enter `[start date] - [end date]` using the 7-day window.
3. Set **Sent To Procurement**: select **No**.
4. Leave all other fields empty.
5. Close the panel (× top right).
6. Confirm pills show:
   - `Date Imported: [start] - [end] ×`
   - `Sent To Procurement: No ×`

---

### Step 4 — Set Page Size to 90

Click the **"25 per page"** dropdown (bottom left) → select **90**.
Wait for the table to reload.

---

### Step 5 — Record Run Start Time

Note the current timestamp (e.g., `May 5, 2026 9:15 AM MDT`).
This goes in the Run Summary tab of the final report.

---

### Step 6 — Read Total Record Count

Check the pagination label (e.g., `1-22 of 22 imports`).
Record the total qualifying invoice count for the Run Summary.

---

### Step 7 — Process Each Invoice Row

For each visible row:

#### 7a — Read All Column Values

Read every column exactly as displayed. Do not infer or assume. Columns:

| Column | Notes |
|---|---|
| Batch ID | Unique identifier — use to re-identify rows after refresh |
| Status | Must be `IMPORTED` to qualify |
| Sent Date | Should be `-` (not yet pushed) |
| Invoice Number | Carrier invoice number |
| Carrier | e.g., FedEx, UPS, DHL |
| Invoice Type | e.g., FedEx, UPS_W2G, DHL ecommerce |
| Import File | Source filename as shown |
| Date Imported | Full timestamp with timezone |
| Imported By | Full name of importing user |
| Record Count | Number of records in batch |
| Charges | Number of charge line items |
| Packages | Package count |
| Ship Packages | Shipped package count |
| Earliest Ship Date | Earliest shipment date or `-` |
| Latest Ship Date | Latest shipment date or `-` |

#### 7b — Check Status Eligibility

| Status | Action |
|---|---|
| `IMPORTED` | ✅ Eligible — proceed to 7c |
| `FAILED` | ⚠️ Skip — log: `Status is FAILED — not eligible for push.` |
| `IN PROGRESS` | ⚠️ Skip — log: `Status is IN PROGRESS — not eligible for push.` |
| Any other | ⚠️ Skip — log: `Status is [value] — not eligible for push.` |

⚠️ Never click ⋮ on a FAILED or IN PROGRESS row.

#### 7c — Apply the 24-Hour Rule

If **Date Imported** is within 24 hours of the current run time:
- Do **not** push.
- Log as: **⚠️ Skipped** | Failure Detail: `Imported within 24 hours of run — not eligible.`
- Move to next row.

#### 7d — Open the ⋮ Menu

The ⋮ icon is in the 3rd column (between Status badge and Sent Date).
Click it. Dropdown shows:
- **Push to Procurement**
- View Payables
- Delete

#### 7e — Click "Push to Procurement"

Click **Push to Procurement**.

#### 7f — Wait for Toast Notification (up to 5 seconds)

| Toast Message | Color | Result | Action |
|---|---|---|---|
| *(any success message)* | Green | ✅ Success | Record; proceed to 7g |
| `Procurement is not enabled for this invoice type` | Red | ❌ Failed | ERR-INVOICE-TYPE; skip refresh; next row |
| `Invoice already exists in procurement` | Red | ❌ Failed | ERR-DUPLICATE; skip refresh; next row |
| `Procurement request timed out` | Red | ❌ Failed | ERR-TIMEOUT; check the portal manually |
| `Authorization denied` | Red | ❌ Failed | ERR-AUTH; **stop run; alert user** |
| `Cannot connect to procurement` | Red | ❌ Failed | ERR-CONN; **stop run; alert user** |
| No toast after 5 seconds | — | ❌ Failed | ERR-NO-CONFIRM; check the portal manually |

#### 7g — Refresh and Capture Sent Date (Success Only)

1. Click the **↻ refresh** icon (top right of table toolbar).
2. Wait for table to reload.
3. Re-locate the row by **Batch ID** — never by position.
4. Read and record the **Sent Date** now populated in the row.

⚠️ Always re-identify rows by Batch ID after refresh. Row order may shift.

---

### Step 8 — Pagination

After all rows on the current page are processed:
- Check pagination label (e.g., `1-90 of 150 imports`).
- If more pages exist, click **Next →** and repeat Step 7.
- Continue until the last page is fully processed.

---

### Step 9 — Generate and Save the Google Sheet Report

**Archive folder ID:** `1ExampleInvoicePushArchiveFolderId0000000000`
**File name format:** `Invoice_Push_Report_YYYY-MM-DD`
**Folder path:** Archive → `YYYY` → `MM`

1. Check if `YYYY` and `MM` subfolders exist in the archive. Create if missing.
2. Create the Google Sheet in the correct `MM` subfolder with three tabs:

---

#### Tab 1 — Invoice Detail

One row per processed invoice. Columns in exact order:

| # | Column | Source |
|---|---|---|
| 1 | Batch ID | Screen |
| 2 | Status | Screen |
| 3 | Invoice Number | Screen |
| 4 | Carrier | Screen |
| 5 | Invoice Type | Screen |
| 6 | Import File | Screen |
| 7 | Date Imported | Screen |
| 8 | Imported By | Screen |
| 9 | Record Count | Screen |
| 10 | Charges | Screen |
| 11 | Packages | Screen |
| 12 | Ship Packages | Screen |
| 13 | Earliest Ship Date | Screen |
| 14 | Latest Ship Date | Screen |
| 15 | Sent Date | Screen (populated after success; `-` otherwise) |
| 16 | Push Status | Added: ✅ Success / ❌ Failed / ⚠️ Skipped |
| 17 | Failure Code | Added: ERR code from §7f, or blank on success |
| 18 | Failure Detail | Added: human-readable reason, or blank on success |

#### Tab 2 — Run Summary

One row per metric:

| Metric | Value |
|---|---|
| Run start time | From Step 5 |
| Date window | `[start] - [end]` from the calculation |
| Total qualifying invoices | From Step 6 |
| Pushed successfully | Count of ✅ |
| Failed | Count of ❌, with per-ERR-code breakdown |
| Skipped | Count of ⚠️ |
| Run stopped early | Yes/No — Yes only on ERR-AUTH or ERR-CONN |

#### Tab 3 — Failure Code Reference

Static reference table, one row per code from §7f: code, toast message,
meaning, and the operator action (retry tomorrow, check the portal
manually, or escalate to an administrator).

---

3. After the sheet is created, reply to the user with the sheet link, the
   success/failed/skipped counts, and — if the run stopped early — which
   error stopped it and which invoices were never attempted.
