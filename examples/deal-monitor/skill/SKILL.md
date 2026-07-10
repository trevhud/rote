---
name: deal-monitor
description: >-
  Scheduled morning dashboard for a fulfillment brokerage's deal pipeline.
  Pulls the #deal-intake Slack channel and Gmail quoting threads, classifies
  each warehouse-quoting email thread into a 6-step pipeline, scores new
  (unquoted) opportunities against the deal-scoring rubric, and renders a
  self-contained three-tab HTML dashboard (New Opportunities | Quoting |
  Closed).
---

# Deal Monitoring Dashboard

Run the deal monitoring dashboard for the account owner (operator@example.com).

## Objective
Build a live deal monitoring dashboard by pulling from the #deal-intake Slack channel and Gmail, then present the output.

## Steps

### Step 1: Pull #deal-intake Slack messages
Read the 50 most recent messages from #deal-intake (channel ID: `C0EXAMPLE000`) using the Slack MCP tool.

Extract for each: Account Name, Submitter, ARR, Monthly DTC Orders, SKU Count, Requested Location, Sales Channels, Current Situation, Pallets, Sq Ft, International Shipping, Go-Live date, product/item context.

**Include:**
- **EU/UK**: ALL opps — any UK, EU, or non-US location, plus anything submitted by Alex Rivers or Sam Patel (always UK).
- **North America**: Only if SKU count < 2,000. Unknown SKU = include, flag as risk. Confirmed ≥ 2,000 = exclude.

### Step 2: Search Gmail for outbound quoting emails
Run multiple searches to maximize coverage. Search all of the following:
1. Subject contains "Quote Request"
2. Subject contains "New Business"
3. Subject contains "RFP"
4. Any thread where the sender or recipient is a known warehouse contact domain
5. Also search by known account names from #deal-intake

Use the last 60 days as the window. For each thread found, extract: account name, every warehouse emailed, per warehouse sent date/replies/state, and the Gmail thread URL (format: `https://mail.google.com/mail/u/0/#inbox/[threadId]`).

Match threads to #deal-intake opps by account name (fuzzy match fine). Any opp with at least one matched email thread goes to the Quoting tab.

### Step 3: Classify each warehouse thread into a pipeline step
Read the entire thread before classifying. Steps: 1=RFP Sent, 2=Response Needed, 3=Follow-Up Needed, 4=Pricing Returned, 5=Pricing Sent, 6=Declined.

### Step 4: Score New Opportunities
Read the deal-scoring skill at `~/.claude/skills/deal-scoring/SKILL.md` and apply its scoring rules. Only score opps with NO matched email threads.

### Step 5: Generate the HTML Dashboard
Generate a self-contained HTML file with three tabs: New Opportunities | Quoting | Closed. Use a clean, self-contained layout — no external assets — with a summary tile row above the tabs and color-coded needs-attention rows.

Save the HTML file to `./outputs/deal-monitor.html`.

### Step 6: Output
1. Brief text summary in chat: new opps count, active quoting count (with how many need attention), closed count.
2. Present the HTML file for download.
