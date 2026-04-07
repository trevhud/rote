# Target Company Research

Run this research phase immediately after the campaign brief is confirmed. The output -- a short intel brief -- feeds into both lead generation (better search terms, smarter vetting) and email personalization (specific references to their programs and priorities).

## Why This Matters

Without upfront research, the lead generation phase operates blind. You end up casting broad nets ("immunology at Roche") that pull in MSLs, biomarker scientists, and translational researchers who have nothing to do with RWE. Knowing that Roche has a specific lupus nephritis program, or that Genentech just presented Phase 3 data at ACR, changes which titles you search for and how you evaluate each contact.

## Research Sources

Run these in parallel:

### 1. External: Bright Data Web Search

Use `search_engine` and `scrape_as_markdown` to gather recent public information about the target company + indication:

- **Company pipeline news**: Search `"[company] [condition] pipeline 2025 2026"` to find recent press releases, data readouts, and regulatory updates
- **Conference presentations**: Search `"[company] [condition] ACR OR EULAR OR ASN OR ISPOR"` (swap in relevant conferences for the TA) to find recent scientific presentations
- **Clinical trials**: Search `"[company] [condition] clinical trial Phase"` for active or recently completed trials
- **RWE/HEOR activity**: Search `"[company] real world evidence [condition]"` or `"[company] HEOR [condition]"` to see if they've already published RWE in this space

Focus on the last 12-18 months. Older news is less useful for outreach.

### 2. External: ClinicalTrials.gov

Use `search_clinical_trials` or `search_by_sponsor` to find active and recently completed trials:

- Search by sponsor + condition to see their active development programs
- Note the specific drug names, phases, and study designs -- these help you identify which internal teams are involved
- Look for observational studies or registries -- these are the closest analog to what Acme does and signal existing RWE interest

### 3. Internal: Acme Knowledge Base

Use `search_corpus` (Airweave enterprise search) to check for prior Acme interactions with this account:

- Previous proposals or pitch decks sent to this company
- Slack conversations mentioning the company
- Existing contacts or relationships in CRM
- Any prior campaigns targeting this account (to avoid duplicate outreach)

Also check the validated internal sources for Acme's own experience in the
indication. **Replace this list with real sources for your organization** —
e.g. an internal research studies database and your public publications page.

### 4. Internal: Salesforce

Use `salesforce_query` or `salesforce_search` to check account history:

- Open or closed opportunities with this company
- Account owner and recent activity
- Any notes from previous BD interactions

## Output: Intel Brief

Compile findings into a short brief (not a document -- just structured notes for use in later phases):

### [Company] -- [Condition] Intel Brief

**Pipeline**: What drugs do they have in this space? What phase? Any recent data readouts?

**RWE signals**: Have they published RWE studies in this indication? Are they running observational studies or registries? Have they presented HEOR/outcomes data at conferences?

**Key programs/teams**: Based on clinical trials and press releases, which internal teams or program names are involved? (These become search terms for Phase 2.)

**Acme history**: Have we pitched this account before? Who owns the relationship? Any existing proposals or contacts?

**Acme experience**: What validated experience do we have in this indication or adjacent TAs? (From Notion database and publications only.)

**Messaging angles**: Based on the above, what 2-3 specific things would resonate in outreach? (e.g., "Your Phase 3 AURORA trial just read out -- we can help with the post-marketing RWE strategy" or "We see you're running a lupus nephritis registry -- our patient-consented longitudinal approach could complement that.")

This brief is not shared with the user as a deliverable. It's working context that informs search strategy, contact vetting, and email personalization in later phases.
