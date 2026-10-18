"""Tests for extracted/parsers.py — derived from skill rules and examples."""

import pytest
from extracted.parsers import (
    COMPLETION_THRESHOLD_PCT,
    MISSORT_WARNING_PACKAGES,
    MISSORT_ALERT_PACKAGES,
    MISLOAD_ALERT_CARRIER,
    parse_shipment_containers,
    parse_dock_pending_log,
    parse_dwell_tickets,
)


# ════════════════════════════════════════════════════════════════════════════
# Constants pinned
# ════════════════════════════════════════════════════════════════════════════

def test_completion_threshold_constant():
    assert COMPLETION_THRESHOLD_PCT == 75.0

def test_missort_warning_constant():
    assert MISSORT_WARNING_PACKAGES == 25

def test_missort_alert_constant():
    assert MISSORT_ALERT_PACKAGES == 41

def test_misload_alert_constant():
    assert MISLOAD_ALERT_CARRIER == 80


# ════════════════════════════════════════════════════════════════════════════
# parse_shipment_containers
# ════════════════════════════════════════════════════════════════════════════

_CONTAINERS_CSV = (
    "Site,Active,Loaded,Staged,Shipped,Completion %\n"
    "EAST1,10,5,3,2,80.0\n"      # above 75 — not flagged
    "WEST2,8,4,2,1,74.9\n"       # just below 75 — flagged
    "NORTH3,12,6,4,3,75.0\n"     # exactly 75 — NOT flagged (must be strictly below)
)

def test_containers_basic_counts():
    result = parse_shipment_containers(_CONTAINERS_CSV)
    assert result["active_site_count"] == 3
    assert result["total_active"] == 30

def test_containers_flagged_below_75():
    result = parse_shipment_containers(_CONTAINERS_CSV)
    assert "WEST2" in result["flagged_sites"]

def test_containers_not_flagged_at_75():
    result = parse_shipment_containers(_CONTAINERS_CSV)
    assert "NORTH3" not in result["flagged_sites"]

def test_containers_not_flagged_above_75():
    result = parse_shipment_containers(_CONTAINERS_CSV)
    assert "EAST1" not in result["flagged_sites"]

def test_containers_avg_completion():
    result = parse_shipment_containers(_CONTAINERS_CSV)
    expected_avg = round((80.0 + 74.9 + 75.0) / 3, 1)
    assert result["avg_completion_pct"] == expected_avg

def test_containers_site_data():
    result = parse_shipment_containers(_CONTAINERS_CSV)
    east = next(s for s in result["sites"] if s["site"] == "EAST1")
    assert east["active"] == 10
    assert east["loaded"] == 5
    assert east["staged"] == 3
    assert east["shipped"] == 2
    assert east["completion_pct"] == 80.0
    assert east["flagged"] is False

def test_containers_wow_absent():
    result = parse_shipment_containers(_CONTAINERS_CSV)
    for site in result["sites"]:
        assert site["wow_trend_pct"] is None

def test_containers_wow_present():
    csv = (
        "Site,Active,Loaded,Staged,Shipped,Completion %,Prior Week Completion %\n"
        "EAST1,10,5,3,2,80.0,77.0\n"
    )
    result = parse_shipment_containers(csv)
    assert result["sites"][0]["wow_trend_pct"] == 77.0

def test_containers_empty_input():
    result = parse_shipment_containers("")
    assert result["sites"] == []
    assert result["flagged_sites"] == []
    assert result["active_site_count"] == 0

def test_containers_header_only():
    result = parse_shipment_containers("Site,Active,Loaded,Staged,Shipped,Completion %\n")
    assert result["sites"] == []

def test_containers_missing_column_raises():
    csv = "Site,Active,Loaded,Staged,Shipped\n"  # missing Completion %
    with pytest.raises(ValueError, match="(?i)completion"):
        parse_shipment_containers(csv)

def test_containers_tsv():
    tsv = "Site\tActive\tLoaded\tStaged\tShipped\tCompletion %\nEAST1\t10\t5\t3\t2\t80.0\n"
    result = parse_shipment_containers(tsv)
    assert result["active_site_count"] == 1


# ════════════════════════════════════════════════════════════════════════════
# parse_dock_pending_log
# ════════════════════════════════════════════════════════════════════════════

_PENDING_CSV = (
    "Site,Issue Type,Resolved\n"
    "EAST1,PO Number,FALSE\n"
    "EAST1,Pallet/Piece Count Discrepancy,FALSE\n"
    "WEST2,PO Number,TRUE\n"       # resolved — should NOT be counted
    "WEST2,PO Number,FALSE\n"
    "NORTH3,Pallet/Piece Count Discrepancy,FALSE\n"
)

def test_pending_total_open():
    result = parse_dock_pending_log(_PENDING_CSV)
    assert result["total_open"] == 4  # 3 EAST1+WEST2+NORTH3 unresolved

def test_pending_excludes_resolved():
    result = parse_dock_pending_log(_PENDING_CSV)
    # WEST2 TRUE row should not be counted
    assert result["by_site"].get("WEST2", 0) == 1

def test_pending_by_site():
    result = parse_dock_pending_log(_PENDING_CSV)
    assert result["by_site"]["EAST1"] == 2
    assert result["by_site"]["NORTH3"] == 1

def test_pending_by_issue_type():
    result = parse_dock_pending_log(_PENDING_CSV)
    assert result["by_issue_type"]["PO Number"] == 2
    assert result["by_issue_type"]["Pallet/Piece Count Discrepancy"] == 2

def test_pending_empty_input():
    result = parse_dock_pending_log("")
    assert result["total_open"] == 0
    assert result["by_site"] == {}
    assert result["by_issue_type"] == {}

def test_pending_all_resolved():
    csv = "Site,Issue Type,Resolved\nEAST1,PO Number,TRUE\n"
    result = parse_dock_pending_log(csv)
    assert result["total_open"] == 0

def test_pending_missing_column_raises():
    csv = "Site,Issue Type\nEAST1,PO Number\n"
    with pytest.raises(ValueError):
        parse_dock_pending_log(csv)


# ════════════════════════════════════════════════════════════════════════════
# parse_dwell_tickets
# ════════════════════════════════════════════════════════════════════════════

_DWELL_CSV = (
    "Open/Closed,Date,Facility,# Packages,Client Impact,Carrier Impact,Issue Description\n"
    "Open,2026-10-17,EAST1,24,ShipperA,CarrierA,Missort suspected\n"   # below 25 — no alert
    "Open,2026-10-17,EAST1,25,ShipperB,CarrierB,Possible missort\n"    # exactly 25 — warning
    "Open,2026-10-17,WEST2,40,ShipperC,CarrierC,Dwell issue\n"         # 40 — warning (below 41)
    "Open,2026-10-17,WEST2,41,ShipperD,CarrierD,Missort alert\n"       # exactly 41 — alert
    "Open,2026-10-17,NORTH3,80,ShipperE,CarrierE,Carrier dwell\n"      # 80 — misload alert
    "Closed,2026-10-16,EAST1,50,ShipperF,CarrierF,Resolved issue\n"    # closed — excluded
)

def test_dwell_total_open():
    result = parse_dwell_tickets(_DWELL_CSV)
    assert result["total_open"] == 5  # 5 Open rows

def test_dwell_excludes_closed():
    result = parse_dwell_tickets(_DWELL_CSV)
    # ShipperF's ticket is closed
    facilities = [t["facility"] for t in result["open_tickets"]]
    assert facilities.count("NORTH3") == 1

def test_dwell_below_25_no_alert():
    result = parse_dwell_tickets(_DWELL_CSV)
    ticket = next(t for t in result["open_tickets"] if t["packages"] == 24)
    assert ticket["alert_level"] == "none"

def test_dwell_exactly_25_warning():
    result = parse_dwell_tickets(_DWELL_CSV)
    ticket = next(t for t in result["open_tickets"] if t["packages"] == 25)
    assert ticket["alert_level"] == "missort_warning"
    assert ticket in result["missort_warnings"]

def test_dwell_40_still_warning():
    result = parse_dwell_tickets(_DWELL_CSV)
    ticket = next(t for t in result["open_tickets"] if t["packages"] == 40)
    assert ticket["alert_level"] == "missort_warning"

def test_dwell_exactly_41_alert():
    result = parse_dwell_tickets(_DWELL_CSV)
    ticket = next(t for t in result["open_tickets"] if t["packages"] == 41)
    assert ticket["alert_level"] == "missort_alert"
    assert ticket in result["missort_alerts"]
    assert ticket not in result["missort_warnings"]

def test_dwell_misload_at_80():
    result = parse_dwell_tickets(_DWELL_CSV)
    ticket = next(t for t in result["open_tickets"] if t["packages"] == 80)
    assert ticket["misload_alert"] is True
    assert ticket in result["misload_alerts"]

def test_dwell_below_80_no_misload():
    result = parse_dwell_tickets(_DWELL_CSV)
    ticket = next(t for t in result["open_tickets"] if t["packages"] == 41)
    assert ticket["misload_alert"] is False

def test_dwell_empty_input():
    result = parse_dwell_tickets("")
    assert result["total_open"] == 0
    assert result["open_tickets"] == []

def test_dwell_header_only():
    csv = "Open/Closed,Date,Facility,# Packages,Client Impact,Carrier Impact,Issue Description\n"
    result = parse_dwell_tickets(csv)
    assert result["total_open"] == 0

def test_dwell_missing_column_raises():
    csv = "Open/Closed,Date,Facility,Client Impact,Carrier Impact,Issue Description\n"
    with pytest.raises(ValueError):
        parse_dwell_tickets(csv)
