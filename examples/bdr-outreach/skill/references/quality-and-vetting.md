# Contact Vetting & Quality Standards

**This step is required before presenting results.** Job title alone is not sufficient signal. Use the enriched employment history to verify that each contact is genuinely involved in the type of research Acme does (RWE, HEOR, registry studies, long-term safety/effectiveness).

## High-Signal Indicators (boosts priority)

- Current or past roles explicitly in RWE, HEOR, Real World Data, Outcomes Research, or Health Economics
- Title contains "Center of Excellence" (CoE) — these people often *own* the RWE methodology function cross-company
- Career history spanning RWE/HEOR roles at multiple pharma companies (e.g., AZ -> BMS -> GSK) — indicates a specialist, not a generalist
- Membership or editorial board roles at ISPOR, AMCP, or similar outcomes research bodies
- Academic affiliations (adjunct professor, visiting researcher) in relevant fields
- Background in epidemiology, biostatistics, or health policy — these translate directly to RWE study design

## Red Flags (discard or flag for BD owner)

These contacts are **not research buyers** and should be removed from the list during the vetting loop. Do not include them in the final output unless the BD owner specifically requests it.

- **Indication mismatch**: The contact's domain (past titles, TA focus) doesn't match the drug-indication combo being targeted. Example: a VP Medical Affairs with an oncology career targeted for a respiratory program. Always check if the contact owns the *specific franchise*, not just the company.
- **MSL / Medical Science Liaison roles**: MSLs are field-based educators, not research commissioners. They don't buy RWE studies or sign contracts. Titles containing "MSL", "Medical Science Liaison", or "Field Medical" should be discarded.
- **Biomarkers / Discovery Science**: Biomarker researchers work on assay development, companion diagnostics, and discovery science. They don't commission real-world evidence studies. Titles containing "Biomarker", "Biomarkers", or "Discovery" should be discarded.
- **Translational Research**: Translational scientists bridge bench-to-bedside science (preclinical models, target validation, early pharmacology). This is fundamentally different from RWE/HEOR. Titles containing "Translational" should be discarded unless paired with clear RWE/outcomes signals in their employment history.
- **Sales/commercial background**: VPs and Directors with primarily sales, brand management, or marketing histories are not research buyers. Discard as primary outreach targets.
- **Operations/strategy MAs**: Medical Affairs Strategy & Operations roles focus on process and governance, not research commissioning. Lower priority than scientific/clinical MA roles.
- **Program management / project management**: Titles focused on "Program Management", "Project Management", or "Portfolio Management" indicate operational coordination roles, not research decision-makers.
- **US commercial team, global clinical program**: For smaller biotechs (Mid-Cap, Emerging), the US commercial org may own a marketed product while the pipeline/clinical program sits with a global (often European) team. Flag this -- the right contact may not be in the US.
- **Low accuracy score**: Flag contacts below 85 -- data may be stale. Contacts validated within the last 90 days are most reliable.

### The Core Test

When evaluating any contact, ask: **"Would this person commission, design, or approve a real-world evidence study?"** If the answer is no -- even if they're senior, even if they're in the right TA -- they don't belong on the list. Acme sells to people who buy RWE, not people who are adjacent to research but doing something else (field education, biomarker assays, translational biology, program ops).

After vetting, add a **"Relevance Evidence"** note for each contact summarizing the 1-2 most compelling pieces of evidence from their history.

## Quality Standards

* **Quality over volume**: The goal is 20-50 highly qualified contacts per campaign, not hundreds. A mismatch damages relationships — large pharma accounts have historically pushed back on over-contacting.
* **Employment history is the ground truth**: Job titles are often generic. Employment history reveals whether a contact has spent their career in research-buying roles or in commercial/sales functions. Always check before outreaching.
* **Indication alignment matters at the franchise level**: At large pharma (Top 25), Medical Affairs and HEOR are organized by franchise or TA. A VP of Medical Affairs whose career is entirely in oncology is effectively not the right contact for a respiratory program, even though they're a VP at the right company. Verify franchise alignment.
* **Small/mid-size biotech org structure caveat**: For Biotech, Mid-Cap 15, and Emerging tier companies, the US team often owns a single marketed product while pipeline programs are run from global (often European) HQ. If the target drug is in early/mid development, the decision-maker may be overseas. Flag this to the BD owner rather than outreaching to the US commercial team.
* **LinkedIn is in the enrichment**: ZoomInfo `enrich_contacts` returns LinkedIn URLs in the `externalUrls` field. No separate lookup step is needed. Always include them in the output table.
* **Deduplication matters**: Flag contacts who appear in multiple searches so the team can cross-check against past campaigns before sending.
* **Accuracy thresholds**: Flag any contacts with accuracy score below 85 — these may have stale data. Contacts validated within the last 90 days are most reliable.
* **Do Not Call**: ZoomInfo returns `directPhoneDoNotCall` and `mobilePhoneDoNotCall` flags. Surface these in the output so the team knows which channels are available.
* **CoE titles are high-signal**: "Center of Excellence" in a title (e.g., "Sr. Director, Real World Evidence CoE") indicates the person owns the RWE methodology or platform function company-wide — not just for one franchise. These are often the best first contacts at large pharma.
* **ISPOR/outcomes research affiliations**: Membership on ISPOR editorial boards, advisory committees, or academic adjunct roles in pharmacoeconomics/outcomes research are strong signals that a contact is deeply embedded in the research community Acme serves.
