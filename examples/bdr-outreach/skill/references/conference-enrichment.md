# Conference List Enrichment

Use this workflow when the user provides a conference attendee or speaker list and wants to:
* Remove non-pharma/biotech companies (consultants, CROs, tech vendors, academic, payers, etc.)
* Enrich remaining pharma contacts with LinkedIn URLs and email addresses
* Output a clean, formatted XLSX for outreach

## Step 1: Load and Inspect the Input File

The user will provide a CSV, XLSX, or Google Sheet. Read it and identify the relevant columns. You need at minimum:
* **Company** (or Organization)
* **First Name** and **Last Name** (or Full Name)
* **Job Title** (optional but helpful)

If the file is an XLSX or Google Sheet, convert or read it into a Python list of dicts for processing.

## Step 2: Categorize Companies (Pharma vs Non-Pharma)

Write a Python script that classifies each contact's company as pharma/biotech or not. Use keyword matching on the company name.

**Pharma/Biotech keywords to INCLUDE** (case-insensitive):
```text
pharma, biopharm, biotech, therapeutics, biopharma, biologics,
oncology, genomics, sciences, medicines, medical, health (when part of pharma co names),
abbvie, pfizer, roche, novartis, merck, lilly, astrazeneca, genentech,
amgen, regeneron, biogen, gilead, bristol myers, bms, sanofi, gsk,
glaxosmithkline, boehringer, novo nordisk, takeda, bayer, janssen,
astellas, eisai, daiichi, otsuka, lundbeck, ucb, vertex, moderna,
abbvie, alexion, shire, allergan, ipsen
```

**Non-Pharma keywords to EXCLUDE**:
```text
consulting, consultancy, consulting group, advisory, advisors, partners,
university, college, institute, foundation, hospital, clinic, health system,
insurance, payer, cro, contract research, iqvia, parexel, covance, icon plc,
medidata, veeva, oracle, microsoft, google, salesforce, sas institute,
mckinsey, deloitte, accenture, pwc, bain, bcg, ey, kpmg,
law firm, legal, government, agency, fda, nih, cms, who,
device, medical device, diagnostics (when not pharma-adjacent)
```

**Ambiguous cases** — manually review companies that:
* Have "health" or "life sciences" in their name but are clearly IT/consulting
* Are contract manufacturers (CMOs) — typically exclude unless clearly pharma-focused
* Are smaller biotechs you don't recognize — err on the side of including and note them for review

Output two lists: `pharma_contacts` (keep) and `non_pharma_contacts` (excluded). Log counts.

**Example Python classification function:**

```python
def is_pharma(company_name):
    if not company_name:
        return False
    name = company_name.lower()

    exclude_keywords = [
        'consulting', 'consultancy', 'advisory', 'advisors', 'partners llp',
        'university', 'college', 'institute', 'hospital', 'health system',
        'insurance', 'iqvia', 'parexel', 'covance', 'icon plc', 'medidata',
        'veeva', 'oracle', 'microsoft', 'google', 'salesforce', 'sas institute',
        'mckinsey', 'deloitte', 'accenture', 'pwc', 'bain & company', 'bcg',
        'ernst & young', 'kpmg', 'government', 'agency', 'foundation'
    ]
    for kw in exclude_keywords:
        if kw in name:
            return False

    include_keywords = [
        'pharma', 'biopharm', 'biotech', 'therapeutics', 'biopharma',
        'biologics', 'oncology', 'genomics', 'sciences', 'medicines',
        'abbvie', 'pfizer', 'roche', 'novartis', 'merck', 'lilly',
        'astrazeneca', 'genentech', 'amgen', 'regeneron', 'biogen',
        'gilead', 'bristol', 'sanofi', 'gsk', 'glaxo', 'boehringer',
        'novo nordisk', 'takeda', 'bayer', 'janssen', 'astellas',
        'eisai', 'daiichi', 'otsuka', 'lundbeck', 'ucb', 'vertex',
        'moderna', 'alexion', 'shire', 'allergan', 'ipsen'
    ]
    for kw in include_keywords:
        if kw in name:
            return True

    return False  # default: exclude if unclear
```

After running, **always show the user** the excluded list and ask if any companies should be moved back to pharma before proceeding to enrichment.

## Step 3: Enrich with ZoomInfo (Batches of 10)

Use the `enrich_contacts` ZoomInfo tool to look up LinkedIn URLs and email addresses. The API accepts **maximum 10 contacts per call**, so batch accordingly.

**Input format for each contact:**
```json
{
  "firstName": "Jane",
  "lastName": "Doe",
  "companyName": "Pfizer"
}
```

**Request these output fields:**
```text
firstName, lastName, email, externalUrls, companyName, jobTitle
```

**LinkedIn URL extraction** — LinkedIn URLs are nested inside `externalUrls`. Parse them like this:
```python
def extract_linkedin(external_urls):
    if not external_urls:
        return ""
    for url_obj in external_urls:
        url = url_obj.get("url", "")
        if "linkedin.com" in url.lower():
            return url
    return ""
```

**Email handling** — Use the `email` field directly. Some contacts won't have emails available; leave blank rather than guessing.

**Batching loop:**
```python
BATCH_SIZE = 10
all_results = {}

for i in range(0, len(pharma_contacts), BATCH_SIZE):
    batch = pharma_contacts[i:i+BATCH_SIZE]
    contacts_input = [
        {"firstName": c["First Name"], "lastName": c["Last Name"], "companyName": c["Company"]}
        for c in batch
    ]
    # Call enrich_contacts tool with contacts=contacts_input and outputFields=[...]
    # Store results indexed back to original contacts
```

After enrichment, store results in a dict keyed by contact index. Track coverage:
* How many contacts have email
* How many contacts have LinkedIn URL

## Step 4: Build the XLSX Output

Use `openpyxl` to create a professionally formatted spreadsheet.

**Columns:**
1. Company
2. First Name
3. Last Name
4. Job Title
5. LinkedIn
6. Email

**Formatting standards:**
* **Font**: Arial, size 11 for data; size 12 bold white for headers
* **Header fill**: Dark navy `#1F4E79`
* **Alternating row fill**: Light blue `#EBF3FB` for even rows, white for odd
* **Frozen pane**: Row 1 (freeze at `A2`)
* **Auto-filter**: Applied to header row
* **Column widths**: Company=30, First Name=18, Last Name=18, Job Title=35, LinkedIn=45, Email=35

**Example openpyxl setup:**
```python
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

header_fill = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
alt_fill = PatternFill(start_color="EBF3FB", end_color="EBF3FB", fill_type="solid")
header_font = Font(name="Arial", size=12, bold=True, color="FFFFFF")
data_font = Font(name="Arial", size=11)

ws.freeze_panes = 'A2'
ws.auto_filter.ref = f"A1:{get_column_letter(len(headers))}1"
```

**Output file naming convention:** `YYYY_[ConferenceName]_Pharma_Contacts.xlsx`

## Step 5: Deliver and Summarize

Present the final XLSX file. Include a summary covering:
* Total contacts after pharma filtering (and how many were removed)
* Email coverage (X/total, X%)
* LinkedIn coverage (X/total, X%)
* Any notable gaps or contacts worth manual follow-up

## Notes

* **Always confirm the excluded list** with the user before enriching — it's faster to add back a mistakenly excluded company than to re-run enrichment.
* ZoomInfo enrichment works best with **First Name + Last Name + Company**. Do not use job title as an input field.
* Some contacts at large pharmas (Pfizer, Roche, J&J) may have LinkedIn but no corporate email in ZoomInfo — this is normal.
* For very large lists (300+ contacts), consider enriching in parallel sessions or across multiple tool invocations to avoid timeouts.
* The `contactAccuracyScoreMin` parameter can be set to `"85"` for higher-confidence results at the cost of lower coverage.
