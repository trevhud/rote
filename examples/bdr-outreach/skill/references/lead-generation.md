# Lead Generation via ZoomInfo

## Campaign Brief

Before searching, confirm with the user:

* **Drug brand name** — e.g., Orladeyo
* **Drug generic name** — e.g., berotralstat
* **Condition (full name + acronym)** — e.g., hereditary angioedema (HAE)
* **Therapeutic area(s)** — e.g., Rare Disease, Hematology, Genetics
* **Manufacturer** — company that markets the drug, e.g., BioCryst Pharmaceuticals
* **Campaign type** — `drug-specific`, `condition-specific`, or `general-capabilities`
* **Job focus** (optional, defaults to all) — `medical`, `rwe`, `heor`, `clinical-development`

If any required fields are missing, ask before proceeding. You can usually infer the manufacturer from the drug name if you're confident.

## Taxonomy Lookups

Before searching, run these lookups in parallel to get exact ZoomInfo IDs:

* `management-levels` — confirm IDs for `VP Level Exec` and `Director`
* `industries` with fuzzyMatch `"pharmaceutical"` — get pharma industry ID
* `industries` with fuzzyMatch `"biotechnology"` — get biotech industry ID
* `departments` with fuzzyMatch `"medical"` — get Medical & Health department ID

This prevents search failures from guessed field values.

## Iterative Search-Enrich-Vet Loop

This is the core of lead generation. The model should autonomously iterate until the user's target quota is met with high-quality contacts. **Do not present contacts to the user until the full vetting loop is complete.**

### How It Works

1. **Overshoot on search volume.** If the user asks for 50 contacts, search for 80-100. Many will be discarded during vetting. That's expected.
2. **Enrich in batches of 10.** After each enrichment batch, immediately vet each contact against `references/quality-and-vetting.md`. Apply the red flags list and the core test ("Would this person commission an RWE study?").
3. **Discard contacts that fail vetting.** Don't hold onto borderline contacts hoping the user will accept them. Remove them and keep searching.
4. **Backfill with new searches.** When discards create a gap, run additional targeted searches. Vary the search terms -- try different title keywords, departments, or seniority levels to find contacts the initial searches missed.
5. **Repeat until quota is met.** The loop ends when you have enough vetted, enriched contacts with valid emails and accuracy scores 85+ to meet the user's requested count.

### Initial Searches (3 in parallel)

Start with three parallel searches to build the initial pool.

#### Search 1 -- Ideal Persona (Condition or Drug Name in Job Title)
The highest-signal contacts have the drug name, generic name, or condition name explicitly in their job title. These are indication specialists who own the franchise.

```text
search_contacts:
  jobTitle: "[drug brand] OR [drug generic] OR [condition full name] OR [condition acronym]"
  managementLevel: "VP Level Exec,Director"
  industryCodes: [pharma ID, biotech ID]
  pageSize: 25
```

#### Search 2 -- Manufacturer's Internal Team
The drug's manufacturer employs Medical Affairs, HEOR, and RWE teams who directly commission real-world evidence studies.

```text
search_companies:
  companyName: "[manufacturer]"
  pageSize: 3

search_contacts:
  companyId: [manufacturer ZoomInfo ID]
  managementLevel: "VP Level Exec,Director"
  department: [Medical & Health ID]
  pageSize: 25
```

#### Search 3 -- Therapeutic Area Broad Net
Catches Directors/VPs working in the TA who don't have the specific drug in their title.

```text
search_contacts:
  jobTitle: "[condition] OR [condition acronym] OR [therapeutic area keywords]"
  managementLevel: "VP Level Exec,Director"
  industryCodes: [pharma ID, biotech ID]
  pageSize: 25
```

### Backfill Searches (as needed)

When the initial three searches don't yield enough qualified contacts after vetting, try variations:

- Narrow by specific function: `"HEOR" OR "health economics" OR "outcomes research"` at the target company
- Try RWE-specific titles: `"real world evidence" OR "real world data" OR "evidence generation"`
- Try adjacent titles that often commission RWE: `"population health" OR "epidemiology" OR "patient centered outcomes"`
- Broaden seniority: include `"Senior Manager"` if Director+ pool is thin
- Search subsidiary/parent companies (e.g., Genentech if targeting Roche, or vice versa)

### Enrich and Vet

Enrich using `enrich_contacts` (batch up to 10 at a time). Always request `employmentHistory` -- this is essential for vetting.

```text
enrich_contacts:
  outputFields: ["firstName", "lastName", "jobTitle", "email", "phone", "mobilePhone",
                 "contactAccuracyScore", "companyName", "externalUrls", "employmentHistory",
                 "directPhoneDoNotCall", "mobilePhoneDoNotCall", "validDate"]
```

After each enrichment batch, apply the full vetting criteria from `references/quality-and-vetting.md`:

- Check each contact against every red flag
- Apply the core test: "Would this person commission, design, or approve an RWE study?"
- Discard contacts without valid email addresses
- Discard contacts with accuracy scores below 85
- Keep a running tally: `[qualified so far] / [target quota]`

**Do not wait for all contacts to be enriched before starting to vet.** Vet after each batch so you know how many more you need to find.

### Tier Assignment

Assign each qualified contact a priority tier:

1. **Ideal** -- Job title explicitly mentions the drug name, generic name, or condition/acronym, OR career is entirely in RWE/HEOR for this TA
2. **Strong** -- RWE / HEOR / Evidence Generation / Medical Affairs (scientific, not ops) Director+ with relevant domain history
3. **Good** -- TA-relevant Director or VP with a credible path to research commissioning, but domain is broader

## Output

Present a ranked table followed by a short narrative. **Every contact in this table should have already passed the full vetting criteria.** The user should not need to second-guess whether any contact belongs.

### Priority Contacts -- [Drug Name] / [Condition] Campaign

| # | Name | Title | Company | Email | LinkedIn | Accuracy | Priority | Relevance Evidence |
| - | ---- | ----- | ------- | ----- | -------- | -------- | -------- | ------------------ |
| 1 | ...  | ...   | ...     | ...   | [link]   | 98       | Ideal    | Sr. Director RWE CoE; prior Director Real World Data |
| 2 | ...  | ...   | ...     | ...   | [link]   | 93       | Strong   | 10 yrs HEOR at AZ; ISPOR editorial board |

Follow with a **brief narrative** (3-5 sentences) covering: total contacts searched, how many were discarded during vetting (and why), tier breakdown of the final list, and the 2-3 contacts to prioritize first.

Also include a **Discarded Contacts Summary** with aggregate counts by discard reason (e.g., "8 discarded: 3 MSL roles, 2 biomarker/translational, 2 no valid email, 1 accuracy below 85"). This gives the BD owner visibility into the vetting process without cluttering the main table.

## Email Drafts (Optional)

If the user requests email drafts, tailor them to the campaign type:

* **Drug-specific**: Reference the drug directly. Highlight Acme's patient recruitment via specialty pharmacies or RWE study capabilities for that indication.
* **Condition-specific**: Reference the condition and Acme's experience in that TA. Connect to what's in their portfolio without naming a specific drug.
* **General-capabilities**: Broader intro to Acme's RWE/registry platform, calling out their therapy area(s) specifically.

For each contact, personalize using: their title, the company's known pipeline, and career history from enrichment (e.g., previous HEOR roles at competitor companies, ISPOR involvement, or academic affiliations).

Target 150-200 words. Lead with a specific observation about their work, not a generic intro. End with a clear, low-friction CTA (e.g., "Would a 20-minute call make sense?").
