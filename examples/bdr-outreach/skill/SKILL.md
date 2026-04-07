---
name: bdr-outreach
description: >-
  End-to-end BDR outreach campaign workflow. Use when running pharma/biotech
  outreach campaigns: finding leads via ZoomInfo, enriching conference attendee
  lists, uploading contacts to HubSpot CRM, managing lists, running exclusion
  checks (do-not-contact, recently emailed, active sequence), creating/editing
  sales email templates, and generating pre-enrollment reports. Trigger phrases:
  "outreach campaign", "BDR campaign", "find contacts for [drug]", "pull leads
  for [condition]", "who should we reach out to at [company]", "run a lead
  search", "find RWE contacts", "filter conference list to pharma", "enrich
  conference attendees", "add contacts to hubspot", "upload contacts", "campaign
  enrollment", "hubspot sequence", "do not contact list", "check if recently
  emailed", "edit sequence template", "sales template", "sequence email",
  "hubspot outreach", "sales sequence", "build outreach list from conference".
plugin: commercial
---

# BDR Outreach Campaign

You are a BDR representative. Campaign ideas, theories, and hypotheses go in; a built-out HubSpot sequence comes out.

This skill orchestrates the full outreach pipeline: from identifying target contacts through ZoomInfo or conference lists, to uploading them into HubSpot, running mandatory exclusion checks, crafting personalized email templates, and producing a pre-enrollment report for the BDR to act on.

## Phase Routing

Not every invocation requires all phases. Match the user's entry point:

| Entry Point | Start At | Example Triggers |
|---|---|---|
| Full campaign | Phase 1 → 1.5 → 2 | "run an outreach campaign for Orladeyo" |
| Conference list | Phase 2-alt → 4 | "filter this conference list to pharma", "enrich conference attendees" |
| Contacts ready | Phase 4 | "upload contacts to hubspot", "add contacts to list" |
| Exclusion check | Phase 5 | "run exclusion checks", "check do not contact list" |
| Template editing | Phase 6 | "edit sequence template", "create sales template" |

## Pipeline

### Phase 1: Campaign Brief

Confirm with the user: drug name (brand + generic), condition (full name + acronym), therapeutic area, manufacturer, campaign type (`drug-specific`, `condition-specific`, or `general-capabilities`), and optional job focus.

### Phase 1.5: Target Company Research

Read `references/target-research.md` before starting this phase.

Run external research (Bright Data web search, ClinicalTrials.gov) and internal research (Airweave enterprise search, Salesforce account history, Acme publications) in parallel. Compile a short intel brief covering the company's pipeline in this indication, any existing RWE/HEOR activity, key program names and teams, prior Acme interactions, and validated Acme experience in the TA.

This brief is not a deliverable -- it's working context that directly informs:
- **Phase 2**: Which titles and program names to search for, and which contacts to keep or discard during vetting
- **Phase 6**: Specific references to their programs, recent data, and Acme's relevant experience for email personalization

### Phase 2: Lead Generation (Iterative)

Read `references/lead-generation.md` and `references/quality-and-vetting.md` before starting this phase.

**This phase is an autonomous loop, not a single pass.** Start with three parallel ZoomInfo searches, then enrich and vet in batches. Discard contacts that fail the vetting criteria (red flags, core test, no valid email, low accuracy). Backfill with additional targeted searches until the user's requested contact quota is met with fully vetted, high-quality contacts.

**Do not present contacts to the user until the vetting loop is complete.** Every contact in the final table should already pass the "Would this person commission an RWE study?" test. The user should not need to catch bad contacts -- that's the model's job.

### Phase 2-alt: Conference List Enrichment

Read `references/conference-enrichment.md` before starting this phase.

Filter a provided attendee list to pharma/biotech contacts, enrich via ZoomInfo in batches of 10, and optionally produce a formatted XLSX.

### Phase 3: Contact Review Gate

Present the final vetted contact table to the user for approval. Include a short summary of how many contacts were searched, how many were discarded during vetting (with aggregate reasons), and the tier breakdown. The user may add, remove, or reprioritize contacts before proceeding to CRM upload.

### Phase 4: CRM Upload + List Creation

Read `references/hubspot-operations.md` before starting this phase.

Batch upsert contacts to HubSpot (100 per call), create a named campaign list, and add contacts to it.

### Phase 5: Exclusion Checks (MANDATORY)

This phase is required before any enrollment recommendation. Using the workflows in `references/hubspot-operations.md`:

1. **Do-not-contact list** — search for "BDR do not contact" list, check each contact's memberships
2. **Recently emailed** — check each contact for outbound emails in last 30 days
3. **Active sequence** — check if contact is already enrolled in a sequence

Skip any contact that fails any check. Generate a pre-enrollment report.

### Phase 6: Email Templates

Read `references/email-templates.md` before starting this phase.

Create or edit sales email templates for the campaign's sequence steps. Personalize using enrichment data (title, company, career history).

### Phase 7: Handoff

Generate the final pre-enrollment report and present it to the BDR. **Sequence enrollment must be done manually in the HubSpot UI** — there is no safe API for this (API enrollment sends emails immediately with no verification).

## Available Tools

### ZoomInfo (Lead Generation)

| Tool | Purpose |
|---|---|
| `zoominfo_search_contacts` | Search contacts by title, company, industry, management level |
| `zoominfo_search_companies` | Find company ZoomInfo IDs |
| `zoominfo_enrich_contacts` | Batch enrich with email, LinkedIn, employment history |
| `zoominfo_lookup` | Taxonomy lookups (management levels, industries, departments) |

### HubSpot CRM

| Tool | Purpose |
|---|---|
| `hubspot_batch_upsert_contacts` | Upload up to 100 contacts (create or update by email) |
| `hubspot_create_contact` | Create a single contact |
| `hubspot_get_contact` | Look up contact by ID or email |
| `hubspot_search_contacts` | Search contacts with filters |
| `hubspot_update_contact` | Update contact properties |

### HubSpot Lists

| Tool | Purpose |
|---|---|
| `hubspot_search_lists` | Search lists by name (e.g. "BDR do not contact") |
| `hubspot_create_list` | Create a static or dynamic list |
| `hubspot_get_list` | Get list details and size |
| `hubspot_get_contact_list_memberships` | Get all lists a contact belongs to |
| `hubspot_add_contacts_to_list` | Add contacts to a static list (max 250/call) |
| `hubspot_remove_contacts_from_list` | Remove contacts from a static list |

### HubSpot Sales Templates (Internal API)

| Tool | Purpose |
|---|---|
| `hubspot_search_sales_templates` | Search/list sales templates by name |
| `hubspot_get_sales_template` | Get template with subject, body HTML, metadata |
| `hubspot_create_sales_template` | Create a new template |
| `hubspot_update_sales_template` | Update an existing template |

### HubSpot Sequences & Engagement (Read-Only)

| Tool | Purpose |
|---|---|
| `hubspot_list_sequences` | List available sequences |
| `hubspot_get_sequence` | Get sequence details (steps, delays, template refs) |
| `hubspot_get_contact_enrollment` | Check if contact is in an active sequence |
| `hubspot_get_contact_emails` | Check if contact was emailed recently |

## Limits & Gotchas

| Constraint | Limit |
|---|---|
| Batch upsert | 100 contacts per call |
| Add to list | 250 contacts per call |
| ZoomInfo enrich | 10 contacts per call |
| Sequence enrollment | UI only (API sends immediately, no verification) |
| Sequence creation/editing | UI only (no API) |
| Sales template CRUD | Internal API, requires browser cookies |
| Cookie expiration | Periodic rotation via your own `rotate-hubspot-cookie` script |

## Related Skills

- **pharma-research** — async company research for pitch preparation (upstream context)
- **hubspot-marketing** — CMS design templates and marketing emails (different domain, not sales templates)
- **competitive-intelligence** — competitor monitoring feed
