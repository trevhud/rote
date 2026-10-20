import pytest
from extracted.parse_dock_pending_log import parse_dock_pending_log

HDR = "Resolved,Site,Issue Type\n"


def csv_rows(*rows):
    return HDR + "\n".join(",".join(str(v) for v in r) for r in rows)


# --- core filtering rule ---

def test_counts_false_rows():
    # skill: "Count all rows where Resolved = FALSE"
    data = csv_rows(("FALSE", "EAST1", "PO Number"), ("FALSE", "WEST2", "PO Number"))
    result = parse_dock_pending_log(data)
    assert result["total_open"] == 2


def test_excludes_true_rows():
    data = csv_rows(
        ("FALSE", "EAST1", "PO Number"),
        ("TRUE", "EAST1", "PO Number"),
    )
    result = parse_dock_pending_log(data)
    assert result["total_open"] == 1


def test_case_insensitive_false():
    data = csv_rows(("false", "EAST1", "PO Number"))
    result = parse_dock_pending_log(data)
    assert result["total_open"] == 1


def test_no_rows_as_resolved():
    data = csv_rows(("NO", "EAST1", "Pallet/Piece Count Discrepancy"))
    result = parse_dock_pending_log(data)
    assert result["total_open"] == 1


# --- breakdown by site ---

def test_by_site_breakdown():
    data = csv_rows(
        ("FALSE", "EAST1", "PO Number"),
        ("FALSE", "EAST1", "PO Number"),
        ("FALSE", "WEST2", "PO Number"),
    )
    result = parse_dock_pending_log(data)
    assert result["by_site"]["EAST1"] == 2
    assert result["by_site"]["WEST2"] == 1


# --- breakdown by issue type ---

def test_by_issue_type_breakdown():
    data = csv_rows(
        ("FALSE", "EAST1", "PO Number"),
        ("FALSE", "EAST1", "Pallet/Piece Count Discrepancy"),
        ("FALSE", "WEST2", "PO Number"),
    )
    result = parse_dock_pending_log(data)
    assert result["by_issue_type"]["PO Number"] == 2
    assert result["by_issue_type"]["Pallet/Piece Count Discrepancy"] == 1


# --- empty inputs ---

def test_empty_string_returns_zeros():
    result = parse_dock_pending_log("")
    assert result == {"total_open": 0, "by_site": {}, "by_issue_type": {}}


def test_header_only_returns_zeros():
    result = parse_dock_pending_log(HDR)
    assert result["total_open"] == 0


def test_all_resolved_returns_zero_open():
    data = csv_rows(("TRUE", "EAST1", "PO Number"), ("TRUE", "WEST2", "PO Number"))
    result = parse_dock_pending_log(data)
    assert result["total_open"] == 0
    assert result["by_site"] == {}


# --- missing column error ---

def test_missing_resolved_column_raises():
    bad = "Site,Issue Type\nEAST1,PO Number\n"
    with pytest.raises(ValueError, match="Resolved"):
        parse_dock_pending_log(bad)


def test_missing_site_column_raises():
    bad = "Resolved,Issue Type\nFALSE,PO Number\n"
    with pytest.raises(ValueError, match="Site"):
        parse_dock_pending_log(bad)
