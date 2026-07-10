---
name: ops-report
description: >-
  Scheduled daily executive operations report for a parcel fulfillment
  network. Auto-pulls shipment-container and dock/dwell spreadsheets from
  Google Drive and dock-activity emails from Gmail, asks the duty manager
  for the handful of numbers that live behind interactive dashboards, then
  assembles a three-section executive summary with a KPI header and
  color-coded dwell alerts.
---

# Daily Executive Operations Report

Generate the daily executive operations report for the parcel fulfillment
network. This report covers three sections.

## Data sources to auto-pull:

**1. Shipment Containers (Google Drive MCP — read_file_content)**
File ID: 1ExampleShipmentContainersFileId00000000000
Parse the facility summary table with COMPLETION % column. Report active, loaded, staged, shipped counts and completion % by site. Flag any site below 75% completion. Show week-over-week trend if available.

**2. Dock Pending Log (Google Drive MCP — read_file_content)**
File ID: 1ExampleDockPendingLogFileId0000000000000000
Count all rows where Resolved = FALSE. Break down by site (EAST1, WEST2, etc.) and issue type (PO Number, Pallet/Piece Count Discrepancy). Report total open count.

**3. Dock Activity Emails (Gmail MCP — search_threads)**
Search query: label:dock-activity newer_than:1d
Count Approved vs. Requested appointments. Flag any outstanding requests not yet approved.

**4. Open Dwell Tickets (Google Drive MCP — read_file_content)**
File ID: 1ExampleDwellTicketLogFileId0000000000000000
Tab: Dwell Log - Form Responses. Filter rows where Open/Closed column = "Open". Report each open ticket with: date, facility, # packages, client impact, carrier impact, issue description. Flag packages >= 25 (Missort Warning), >= 41 (Missort Alert), carrier >= 80 (Possible Misload Alert).

## Manual data to request from user:

After pulling the above, ask the duty manager to provide:
1. **Dock Appointment Completion** (from the dock scheduler's appointment view for the prior day): count of Green (Complete), Red (No Call/No Show), Grey (Canceled), and any other color (Non-Compliant). Non-Compliant must be escalated.
2. **Sorter Metrics** (for sites with automated divert lines): volume output and fail rate % per site. Flag if fail rate > 5% (action needed) or if volume dropped >15% day-over-day (divert adjustment may be needed).
3. **Current Dwell** (from the BI dashboard, filtered by site): shipper/brand name, carrier, site, # orders/packages dwelling. Apply thresholds: Brand 25-40 = Missort Warning (light red), 41+ = Missort Alert (bright red). Carrier 80+ missing = Possible Misload Alert (darkest red).

## Report format:

Produce an executive-level summary with:
- **KPI header**: Active sites, Dock Pending count, Avg Completion %, Open Dwell Tickets
- **Section 1 — Dock Compliance**: Appointment tiles + Gmail summary + Pending Log table
- **Section 2 — Missort & Loss**: Sorter metrics + container flow table by site with completion %
- **Section 3 — Dwell**: BI-dashboard dwell alerts (color-coded) + Open Dwell Impact Log tickets

Also remind the duty manager to open the live artifact "ops-report" for the interactive version with charts. The artifact auto-refreshes Sheets and Gmail data on every open.
