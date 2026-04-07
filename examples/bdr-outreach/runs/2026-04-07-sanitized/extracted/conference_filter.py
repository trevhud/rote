"""
Extracted from: examples/bdr-outreach/skill/references/conference-enrichment.md
Lines 52–115 — pharma classifier and LinkedIn extractor.

These functions were previously executed mentally by the LLM on every run.
Lifting them to code makes them deterministic, testable, and impossible to
accidentally skip during a conference enrichment campaign.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXCLUDE_KEYWORDS: list[str] = [
    "consulting",
    "consultancy",
    "advisory",
    "advisors",
    "partners llp",
    "university",
    "college",
    "institute",
    "hospital",
    "health system",
    "insurance",
    "iqvia",
    "parexel",
    "covance",
    "icon plc",
    "medidata",
    "veeva",
    "oracle",
    "microsoft",
    "google",
    "salesforce",
    "sas institute",
    "mckinsey",
    "deloitte",
    "accenture",
    "pwc",
    "bain & company",
    "bcg",
    "ernst & young",
    "kpmg",
    "government",
    "agency",
    "foundation",
]

INCLUDE_KEYWORDS: list[str] = [
    "pharma",
    "biopharm",
    "biotech",
    "therapeutics",
    "biopharma",
    "biologics",
    "oncology",
    "genomics",
    "sciences",
    "medicines",
    "abbvie",
    "pfizer",
    "roche",
    "novartis",
    "merck",
    "lilly",
    "astrazeneca",
    "genentech",
    "amgen",
    "regeneron",
    "biogen",
    "gilead",
    "bristol",
    "sanofi",
    "gsk",
    "glaxo",
    "boehringer",
    "novo nordisk",
    "takeda",
    "bayer",
    "janssen",
    "astellas",
    "eisai",
    "daiichi",
    "otsuka",
    "lundbeck",
    "ucb",
    "vertex",
    "moderna",
    "alexion",
    "shire",
    "allergan",
    "ipsen",
]


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def is_pharma(company_name: str | None) -> bool:
    """Return True if company_name matches pharma/biotech heuristics.

    Source: conference-enrichment.md lines 52–84 (verbatim extraction).
    Ambiguous companies (no keyword match) default to False — the excluded list
    is shown to the user before enrichment so they can override.
    """
    if not company_name:
        return False
    name = company_name.lower()

    for kw in EXCLUDE_KEYWORDS:
        if kw in name:
            return False

    for kw in INCLUDE_KEYWORDS:
        if kw in name:
            return True

    return False  # default: exclude if unclear


def extract_linkedin(external_urls: list[dict] | None) -> str:
    """Extract LinkedIn profile URL from ZoomInfo externalUrls array.

    Source: conference-enrichment.md lines 109–115 (verbatim extraction).
    Returns empty string if no LinkedIn URL is found.
    """
    if not external_urls:
        return ""
    for url_obj in external_urls:
        url = url_obj.get("url", "")
        if "linkedin.com" in url.lower():
            return url
    return ""


def partition_contacts(
    contacts: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Split a raw attendee list into pharma and non-pharma contacts.

    Returns (pharma_contacts, excluded_contacts).
    The excluded list is surfaced to the user at conf_exclusion_review_gate
    before enrichment starts.
    """
    pharma: list[dict] = []
    excluded: list[dict] = []
    for contact in contacts:
        company = contact.get("Company") or contact.get("company") or ""
        if is_pharma(company):
            pharma.append(contact)
        else:
            excluded.append(contact)
    return pharma, excluded
