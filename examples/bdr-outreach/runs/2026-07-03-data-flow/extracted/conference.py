"""Conference list enrichment functions for the BDR Phase 2-alt path.

Three deterministic functions extracted from references/conference-enrichment.md:

1. load_attendee_file    — parse CSV/XLSX into normalized contact dicts
2. filter_pharma_contacts — apply is_pharma() keyword classifier
3. build_conference_xlsx — produce formatted XLSX using openpyxl

The is_pharma() function is lifted verbatim from conference-enrichment.md
lines 52–85. All include/exclude keyword lists and formatting constants
(colors, column widths, etc.) are defined here as module-level constants.
"""

from __future__ import annotations

from typing import Any


# ── Constants (from references/conference-enrichment.md) ──────────────────────

PHARMA_INCLUDE_KEYWORDS: tuple[str, ...] = (
    "pharma", "biopharm", "biotech", "therapeutics", "biopharma",
    "biologics", "oncology", "genomics", "sciences", "medicines",
    "abbvie", "pfizer", "roche", "novartis", "merck", "lilly",
    "astrazeneca", "genentech", "amgen", "regeneron", "biogen",
    "gilead", "bristol", "sanofi", "gsk", "glaxo", "boehringer",
    "novo nordisk", "takeda", "bayer", "janssen", "astellas",
    "eisai", "daiichi", "otsuka", "lundbeck", "ucb", "vertex",
    "moderna", "alexion", "shire", "allergan", "ipsen",
)

PHARMA_EXCLUDE_KEYWORDS: tuple[str, ...] = (
    "consulting", "consultancy", "advisory", "advisors", "partners llp",
    "university", "college", "institute", "hospital", "health system",
    "insurance", "iqvia", "parexel", "covance", "icon plc", "medidata",
    "veeva", "oracle", "microsoft", "google", "salesforce", "sas institute",
    "mckinsey", "deloitte", "accenture", "pwc", "bain & company", "bcg",
    "ernst & young", "kpmg", "government", "agency", "foundation",
)

# Formatting constants for build_conference_xlsx
HEADER_FILL_COLOR: str = "1F4E79"   # dark navy
ALT_ROW_FILL_COLOR: str = "EBF3FB"  # light blue
HEADER_FONT_NAME: str = "Arial"
HEADER_FONT_SIZE: int = 12
DATA_FONT_SIZE: int = 11

COLUMN_WIDTHS: dict[str, int] = {
    "Company": 30,
    "First Name": 18,
    "Last Name": 18,
    "Job Title": 35,
    "LinkedIn": 45,
    "Email": 35,
}


def is_pharma(company_name: str) -> bool:
    """Classify a company as pharma/biotech or not using keyword matching.

    Lifted verbatim from references/conference-enrichment.md lines 52–85.
    Exclude keywords are checked first (they take precedence over include).
    Ambiguous names (no keyword match) default to False (excluded).

    Args:
        company_name: Company name string from the attendee list.

    Returns:
        True if the company is pharma/biotech, False if not or ambiguous.
    """
    if not company_name:
        return False
    name = company_name.lower()

    for kw in PHARMA_EXCLUDE_KEYWORDS:
        if kw in name:
            return False

    for kw in PHARMA_INCLUDE_KEYWORDS:
        if kw in name:
            return True

    return False  # default: exclude if unclear


def load_attendee_file(file_path: str) -> list[dict[str, Any]]:
    """Load a conference attendee CSV or XLSX into normalized contact dicts.

    Minimum required columns: Company (or Organization), First Name,
    Last Name. Job Title is optional but used downstream by enrichment.

    Args:
        file_path: Path to CSV, XLSX, or Google Sheet export.

    Returns:
        List of dicts with normalized keys:
          Company, First Name, Last Name, Job Title (empty string if absent).

    Raises:
        NotImplementedError: stub. Production uses pandas or openpyxl to
            read the file and normalize column names.
    """
    raise NotImplementedError(
        f"Replace with real file parsing for: {file_path}\n"
        "Use pandas.read_csv or openpyxl.load_workbook depending on extension.\n"
        "Normalize column names to: Company, First Name, Last Name, Job Title."
    )


def filter_pharma_contacts(
    raw_contacts: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Partition the attendee list into pharma/biotech and non-pharma contacts.

    Applies is_pharma() to each contact's Company field. Runs deterministically
    — no LLM, no API call.

    Args:
        raw_contacts: Normalized contact dicts from load_attendee_file.

    Returns:
        Tuple of (pharma_contacts, excluded_contacts).
        The user confirms the excluded list before enrichment proceeds
        (see conf_exclusion_review_gate in pipeline.yaml).
    """
    pharma: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for contact in raw_contacts:
        company = contact.get("Company", "")
        if is_pharma(company):
            pharma.append(contact)
        else:
            excluded.append(contact)
    return pharma, excluded


def build_conference_xlsx(
    enriched_contacts: list[dict[str, Any]],
    conference_name: str,
) -> dict[str, Any]:
    """Build a formatted XLSX from enriched conference contacts.

    Formatting standards (from references/conference-enrichment.md lines 148–174):
    - Font: Arial 11 for data, 12 bold white for headers
    - Header fill: dark navy #1F4E79
    - Alternating row fill: light blue #EBF3FB for even rows, white for odd
    - Frozen pane: row 1 (freeze at A2)
    - Auto-filter: applied to header row
    - Column widths: see COLUMN_WIDTHS constant

    Output file naming: YYYY_ConferenceName_Pharma_Contacts.xlsx

    Args:
        enriched_contacts: ZoomInfo-enriched contacts with email and LinkedIn.
        conference_name: Conference identifier for the filename (e.g., "ISPOR2026").

    Returns:
        Dict with keys: xlsx_path: str, coverage_summary: dict.

    Raises:
        NotImplementedError: stub. Production uses openpyxl.
    """
    raise NotImplementedError(
        "Replace with real openpyxl implementation.\n"
        f"Columns: {list(COLUMN_WIDTHS.keys())}\n"
        f"Header fill: #{HEADER_FILL_COLOR}, alt-row fill: #{ALT_ROW_FILL_COLOR}\n"
        f"Output filename: YYYY_{conference_name}_Pharma_Contacts.xlsx"
    )
