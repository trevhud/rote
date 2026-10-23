import json

import pytest

from extracted.dock_pending_summary import dock_pending_summary

REQ = {"report_date": "2026-09-14"}
ROWS = [
    {"Site": "EAST1", "Issue Type": "PO Number", "Resolved": "FALSE"},
    {"Site": "EAST1", "Issue Type": "Pallet/Piece Count Discrepancy", "Resolved": "FALSE"},
    {"Site": "WEST2", "Issue Type": "PO Number", "Resolved": "TRUE"},
    {"Site": "WEST2", "Issue Type": "PO Number", "Resolved": "false"},
]
EXPECTED = {
    "dock_pending_total": 3,
    "dock_pending_by_site": {"EAST1": 2, "WEST2": 1},
    "dock_pending_by_issue_type": {"PO Number": 2, "Pallet/Piece Count Discrepancy": 1},
}
CSV = (
    "Site,Issue Type,Resolved\n"
    "EAST1,PO Number,FALSE\n"
    'EAST1,Pallet/Piece Count Discrepancy,FALSE\n'
    "WEST2,PO Number,TRUE\n"
    "WEST2,PO Number,FALSE\n"
)


def test_list():
    assert dock_pending_summary(REQ, ROWS) == EXPECTED


def test_wrapped():
    assert dock_pending_summary(REQ, {"rows": ROWS}) == EXPECTED


def test_json_text():
    assert dock_pending_summary(REQ, json.dumps(ROWS)) == EXPECTED
    assert dock_pending_summary(REQ, json.dumps({"data": ROWS})) == EXPECTED


def test_csv_text():
    assert dock_pending_summary(REQ, CSV) == EXPECTED


def test_mcp_content():
    assert dock_pending_summary(REQ, [{"type": "text", "text": CSV}]) == EXPECTED


def test_bool_and_case_insensitive_columns():
    rows = [{" site ": "A", "ISSUE TYPE": "X", "resolved": False},
            {" site ": "A", "ISSUE TYPE": "X", "resolved": True}]
    assert dock_pending_summary(REQ, rows)["dock_pending_total"] == 1


def test_empty():
    assert dock_pending_summary(REQ, []) == {
        "dock_pending_total": 0,
        "dock_pending_by_site": {},
        "dock_pending_by_issue_type": {},
    }


def test_malformed():
    with pytest.raises(ValueError):
        dock_pending_summary(REQ, "Site,Issue Type\nEAST1,PO Number\n")
    with pytest.raises(ValueError):
        dock_pending_summary(REQ, [{"Site": "A"}])
    with pytest.raises(ValueError):
        dock_pending_summary(REQ, 42)
