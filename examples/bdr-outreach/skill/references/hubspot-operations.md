# HubSpot CRM Operations

## Contact Management

| Tool | Purpose |
|---|---|
| `hubspot_batch_upsert_contacts` | Upload up to 100 contacts at once (create or update by email) |
| `hubspot_create_contact` | Create a single contact |
| `hubspot_get_contact` | Look up contact by ID or email |
| `hubspot_search_contacts` | Search contacts with filters |
| `hubspot_update_contact` | Update contact properties |

## List Management

| Tool | Purpose |
|---|---|
| `hubspot_search_lists` | Search lists by name (e.g. "BDR do not contact"). **Use this to find list IDs.** |
| `hubspot_create_list` | Create a MANUAL (static) or DYNAMIC (auto-updating) list |
| `hubspot_get_list` | Get list details and size by ID |
| `hubspot_get_contact_list_memberships` | Get all lists a contact belongs to (for do-not-contact checks) |
| `hubspot_add_contacts_to_list` | Add contacts to a static list (max 250 per call) |
| `hubspot_remove_contacts_from_list` | Remove contacts from a static list |

### Naming Conventions

Name lists after the campaign strategy for easy tracking:
* `Denver Conference 2026 - Pharma Speakers`
* `Rare Disease Campaign - Q1 2026`
* `Specialty Pharmacy Outreach - March`

## Sequence Info (READ-ONLY)

| Tool | Purpose |
|---|---|
| `hubspot_list_sequences` | List available sequences (requires userId) |
| `hubspot_get_sequence` | Get sequence details — steps, delays, template refs |
| `hubspot_get_contact_enrollment` | Check if contact is already in a sequence |

> `hubspot_get_sequence` returns step metadata including `emailTemplateId` for email steps. Use `hubspot_get_sales_template` with that ID to read/edit the actual email content.

## Engagement History

| Tool | Purpose |
|---|---|
| `hubspot_get_contact_emails` | Check if a contact was emailed recently (last N days) |

## Exclusion Checks (MANDATORY)

Before recommending contacts for enrollment, **always** run these checks:

### 1. "Do Not Contact" List Check

```text
Setup (once per campaign):
  1. Call hubspot_search_lists with query="BDR do not contact"
  2. Note the listId from the result

For each contact:
  1. Call hubspot_get_contact_list_memberships with the contact's ID
  2. Check if the do-not-contact listId appears in their memberships
  3. Skip any matches
```

### 2. Recently Emailed Check (30-day window)

```text
For each contact:
  1. Call hubspot_get_contact_emails with daysBack=30, direction=OUTBOUND
  2. If wasEmailedInPeriod is true -> skip this contact
  3. Log which contacts were skipped and why
```

**IMPORTANT: Search exhaustively for email addresses before guessing patterns.** Use ZoomInfo, LinkedIn, and other enrichment sources first. Guessing email patterns leads to bounces which hurts sender reputation.

### 3. Active Sequence Check

A contact can only be in **one sequence at a time**.

```text
For each contact:
  1. Call hubspot_get_contact_enrollment
  2. If isEnrolled is true -> skip (they're in another campaign)
```

## Pre-Enrollment Report

Generate a summary for the BDR to review before they manually enroll in HubSpot:

```text
Campaign: Denver Conference 2026
Total contacts enriched: 47
Already in HubSpot: 12 (will update)
New contacts: 35

Exclusions:
  - On "do not contact" list: 3
  - Emailed in last 30 days: 7
  - Already in active sequence: 2

Ready to enroll: 35
-> BDR should enroll these contacts manually in HubSpot UI
```

## Enrollment Warning

**Sequence enrollment has been REMOVED from the API tools.** Enrolling a contact in an active sequence via API immediately sends emails with no verification step. There is no way to:
- Check if a sequence is paused or active via API
- Pause or unpause a sequence via API
- Enroll contacts without triggering sends
- Create or edit sequences themselves via API (only their templates)

**Enrollment must be done manually in the HubSpot UI.**
