---
name: incident-triage
description: >-
  First-responder triage for a PagerDuty page. Acknowledges the incident,
  figures out which service is hurting and who owns it, pulls recent
  deploys and error logs, assigns a severity from the severity matrix,
  forms a first hypothesis, digs until the picture is clear, posts the
  triage summary to the owning team's incident channel, and hands off to
  the incident commander. Use when you get paged, when someone says
  "triage incident <id>", or "what's going on with this alert".
---

# On-call incident triage

This is the runbook we follow as the first responder when PagerDuty pages
us. The goal of triage is NOT to fix the incident. The goal is to get the
right people looking at the right thing within the first 15 minutes, with
a severity everyone agrees on and a written summary that the incident
commander (IC) can work from. Fixing comes after.

We wrote this down after a bad quarter where three incidents sat
unacknowledged for 20+ minutes and two more were run at the wrong
severity for an hour because the responder "had a feeling." So some of
the steps below are marked MANDATORY. Please don't skip them, even when
the incident looks trivial or you already know what's wrong.

The only input is the PagerDuty incident id (it's in the page, e.g.
`Q1ABCD2EFGH3IJ`).

## Step 1: Acknowledge the page (MANDATORY)

Acknowledge the incident in PagerDuty right away with the
`acknowledge_incident` tool on the PagerDuty MCP. Do this before you read
anything else. An unacknowledged page escalates to the secondary after 5
minutes and to the engineering manager after 10, and nobody wants that.

At the same time, pull the full incident with `get_incident` (also on the
PagerDuty MCP). You need the service name, the alert title, the monitor
priority (P1 to P5), and when it triggered. These two calls don't depend
on each other, so fire them together.

## Step 2: Figure out what service this is and who owns it

PagerDuty service names don't always match what we call things
internally, so look the service up in
[references/service-catalog.md](references/service-catalog.md). From the
catalog row you get:

- the GitHub repo the service deploys from
- the owning team and their incident Slack channel
- the service tier (1, 2 or 3)
- whether it's customer-facing
- the Datadog log query to use for it

If the service isn't in the catalog, that's a real problem (it means
something got deployed without being registered). Stop and ask in
#platform-oncall rather than guessing an owner.

## Step 3: Pull recent deploys and error logs

Nine times out of ten an incident is caused by something we shipped. So
pull two things, in parallel since they're independent:

1. **Recent deploys.** Use `list_deployments` on the GitHub MCP for the
   service's repo, environment `production`, and look at everything in
   the last 2 hours before the incident triggered. Two hours is our
   standard lookback. Don't widen it at this stage. The investigation
   step can go further back if it needs to.
2. **Error logs.** Use `search_logs` on the Datadog MCP with the catalog's
   log query for the service, over the 30 minutes before the trigger
   time, capped at 200 lines. More than that and you're just reading
   noise.

## Step 4: Assign the severity (MANDATORY)

Severity comes from the matrix in
[references/severity-matrix.md](references/severity-matrix.md) and
nowhere else. Don't go with your gut, don't go with how loud the Slack
channel is. The matrix only looks at the monitor priority, the service
tier and whether the service is customer-facing, which you already have
from steps 1 and 2. If two rows match, the higher severity wins.

The IC can change the severity later (step 8). You don't.

## Step 5: Form a first hypothesis

Now read the error log lines and the deploy list together and make a
first call on the likely cause. Pick exactly one of:

- **recent_deploy**: the errors started after a deploy and look related
  to what changed (new exception types, a stack trace in code that just
  shipped). Name the deploy (its SHA).
- **upstream_dependency**: the errors are timeouts or connection failures
  to something we don't own (a vendor API, a managed database).
- **infrastructure**: node, network, DNS, disk, or OOM kills.
- **traffic_spike**: the service is healthy but overwhelmed (rate limits,
  queue depth, 429s/503s with no new error types).
- **unknown**: you honestly can't tell yet. That's a fine answer. Don't
  force a story.

Write one or two sentences of evidence, quoting log lines where you can,
and say how confident you are (low, medium, high).

## Step 6: Investigate

This is the part that can't be scripted. Starting from your hypothesis,
go find out what's actually happening. Some things that usually help:

- search the logs of the services that call this one, or that it calls
- check the service's Datadog monitors and metrics (latency, error rate,
  saturation) and compare to the same time yesterday
- look at the diff of a suspect deploy on GitHub
- read the service's runbook (linked from the catalog) and run its checks

Use your judgment about what to look at next based on what you find.
Stop when you can explain what's going on in a paragraph, or when you've
been at it for 20 minutes, whichever comes first. Twenty minutes is a
hard stop: at that point the IC needs your summary more than you need
another dashboard. Keep a list of what you checked and what you ruled
out, the IC will ask.

## Step 7: Post the triage summary (MANDATORY)

Always post the summary to the owning team's incident channel (from the
catalog), even if the investigation came up empty. A summary that says
"unknown, here's what I ruled out" is still useful. Silence is not.

Use exactly this format, so people can scan it on their phone:

```
:rotating_light: [{severity}] {alert_title}
Service: {service} (tier {tier}, owner {owning_team})
Triggered: {triggered_at}
Likely cause: {cause} ({confidence}) {suspect_sha}
What we know: {findings}
Checked / ruled out: {checked}
Recent deploys (2h): {deploy_count}
PagerDuty: {incident_url}
```

Post it with `chat_post_message` on the Slack MCP.

## Step 8: Hand off to the incident commander

Page in the IC (they're tagged automatically from the channel) and wait
for them to take over. The IC confirms or changes the severity, and
decides whether to roll back. Do NOT roll back yourself from triage, even
if you're sure it's the deploy. Rollbacks are the IC's call, they run
them, and they need to know the system state when they do.

Wait for the IC to reply with the confirmed severity. If they don't
respond within 15 minutes, PagerDuty's own escalation takes over.

## Step 9: Record the confirmed severity (MANDATORY)

Once the IC has confirmed, set the incident's priority in PagerDuty to
the confirmed severity with `update_incident`. Our incident metrics and
the postmortem schedule are driven off this field, so it has to be right
even when the IC kept your original call.

That's the end of triage. From here the IC runs the incident.
