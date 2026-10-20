---
name: customer-onboarding
description: >-
  Onboard a new customer once their deal is closed-won in HubSpot. Pulls
  the deal, screens the company, checks the terms, gets the CS lead's
  sign-off, then provisions the workspace, invites the customer's admins,
  opens the Linear onboarding project and announces it in #cs-onboarding.
  Use when someone says "onboard <company>" or "deal <id> closed".
---

# Customer onboarding

Input: the HubSpot deal id.

## 1. Pull the deal

`hubspot_get_deal` (HubSpot MCP) with the deal id. It returns company
name, plan, seats, discount %, payment terms (days), the special terms
note, and the admin contacts.

## 2. Sanctions screen (MANDATORY)

Run the company name through `screen_entity` on the compliance MCP. Every
customer, no exceptions, before anything is provisioned. A hit doesn't
stop onboarding by itself, but the CS lead has to see it in step 5.

## 3. Commercial terms

Standard terms:

- discount 20% or less
- payment terms net 30 or shorter
- seats within the plan cap: Starter 25, Growth 200, Enterprise no cap

Anything outside that is non-standard. List each one.

## 4. Special terms note

Read the deal's special terms note. Flag anything that changes our MSA:
custom SLA, liability cap changes, data residency, termination for
convenience, MFN pricing. Nothing in the note, or only scheduling notes
(kickoff dates, contact preferences), means standard.

## 5. CS lead sign-off

Send the CS lead the deal summary, the screen result, and the
non-standard items from 3 and 4. Wait for approval. The CS lead can
change the plan or seat count when approving. Don't provision anything
until they approve.

## 6. Provision

`POST https://admin.internal.example.com/v1/workspaces` with the company
name and the approved plan and seats. Region is `us-east-1` unless the
CS lead says otherwise. Retry on 5xx, the admin API is flaky.

At the same time, create the Linear project with `create_project` on the
Linear MCP, from the template `Customer Onboarding v3`, named after the
company.

## 7. Invite admins

One invite per admin contact:
`POST /v1/workspaces/{workspace_id}/invites` with their email and role
`admin`.

## 8. Announce

Post to #cs-onboarding (`C0ONBOARD01`) with `slack_post_message`:

```
:tada: {company} is live on {plan} ({seats} seats)
Workspace: {workspace_url}
Onboarding project: {project_url}
Approved by {approver}
```
