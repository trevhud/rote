"""
Tests for extracted/parsers.py.

Derived from the ops-report SKILL.md rules — every stated threshold is
tested on both sides of its boundary. Offline, stdlib only, no credentials.
"""

import json
import pytest

from extracted.parsers import (
    COMPLETION_THRESHOLD,
    DWELL_PACKAGES_WARNING,
    DWELL_PACKAGES_ALERT,
    DWELL_CARRIER_ALERT,
    parse_shipment_containers,
    parse_dock_pending_log,
    parse_dock_activity_emails,
    parse_dwell_tickets,
)


# ─────────────────────────────────────────────────────────────
# Constant-pinning tests  (Pattern 2 — these fail if values drift)
# ─────────────────────────────────────────────────────────────

def test_completion_threshold_constant():
    assert COMPLETION_THRESHOLD == 75.0


def test_dwell_warning_constant():
    assert DWELL_PACKAGES_WARNING == 25


def test_dwell_alert_constant():
    assert DWELL_PACKAGES_ALERT == 41


def test_dwell_carrier_alert_constant():
    assert DWELL_CARRIER_ALERT == 80


# ─────────────────────────────────────────────────────────────
# parse_shipment_containers
# ─────────────────────────────────────────────────────────────

SHIPMENT_CSV_BASIC = (
    "SITE,ACTIVE,LOADED,STAGED,SHIPPED,COMPLETION %\n"
    "EAST1,5,10,3,20,80.0\n"
    "WEST2,2,8,1,15,74.9\n"
    "NORTH3,0,12,4,30,75.0\n"
)

SHIPMENT_CSV_WITH_TREND = (
    "SITE,ACTIVE,LOADED,STAGED,SHIPPED,COMPLETION %,PREV COMPLETION %\n"
    "EAST1,5,10,3,20,80.0,70.0\n"
    "WEST2,2,8,1,15,60.0,65.0\n"
)


def test_parse_shipment_containers_counts():
    result = parse_shipment_containers(raw=SHIPMENT_CSV_BASIC)
    assert result["active_site_count"] == 3
    assert result["avg_completion_pct"] == round((80.0 + 74.9 + 75.0) / 3, 2)


def test_parse_shipment_containers_site_fields():
    result = parse_shipment_containers(raw=SHIPMENT_CSV_BASIC)
    east1 = next(s for s in result["sites"] if s["site"] == "EAST1")
    assert east1["active"] == 5
    assert east1["loaded"] == 10
    assert east1["staged"] == 3
    assert east1["shipped"] == 20
    assert east1["completion_pct"] == 80.0


def test_parse_shipment_containers_flag_below_75():
    # skill: "Flag any site below 75% completion"
    result = parse_shipment_containers(raw=SHIPMENT_CSV_BASIC)
    assert "WEST2" in result["flagged_sites"]   # 74.9 < 75.0
    assert "EAST1" not in result["flagged_sites"]


def test_parse_shipment_containers_exactly_75_not_flagged():
    # skill: "Flag any site below 75% completion" — boundary: 75.0 is NOT flagged
    result = parse_shipment_containers(raw=SHIPMENT_CSV_BASIC)
    north3 = next(s for s in result["sites"] if s["site"] == "NORTH3")
    assert north3["flagged"] is False


def test_parse_shipment_containers_wow_trend():
    result = parse_shipment_containers(raw=SHIPMENT_CSV_WITH_TREND)
    east1 = next(s for s in result["sites"] if s["site"] == "EAST1")
    assert east1["wow_trend"] == 10.0   # 80.0 - 70.0
    west2 = next(s for s in result["sites"] if s["site"] == "WEST2")
    assert west2["wow_trend"] == -5.0   # 60.0 - 65.0


def test_parse_shipment_containers_no_trend_without_prev_col():
    result = parse_shipment_containers(raw=SHIPMENT_CSV_BASIC)
    for site in result["sites"]:
        assert site["wow_trend"] is None


def test_parse_shipment_containers_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        parse_shipment_containers(raw="")


def test_parse_shipment_containers_missing_column_raises():
    bad_csv = "SITE,ACTIVE\nEAST1,5\n"
    with pytest.raises(ValueError):
        parse_shipment_containers(raw=bad_csv)


def test_parse_shipment_containers_skips_blank_rows():
    csv_with_blank = (
        "SITE,ACTIVE,LOADED,STAGED,SHIPPED,COMPLETION %\n"
        "EAST1,5,10,3,20,80.0\n"
        "\n"
        "WEST2,2,8,1,15,70.0\n"
    )
    result = parse_shipment_containers(raw=csv_with_blank)
    assert result["active_site_count"] == 2


# ─────────────────────────────────────────────────────────────
# parse_dock_pending_log
# ─────────────────────────────────────────────────────────────

DOCK_LOG_CSV = (
    "SITE,ISSUE TYPE,RESOLVED\n"
    "EAST1,PO Number,FALSE\n"
    "EAST1,Pallet/Piece Count Discrepancy,FALSE\n"
    "WEST2,PO Number,TRUE\n"
    "WEST2,PO Number,FALSE\n"
    "NORTH3,Pallet/Piece Count Discrepancy,FALSE\n"
)


def test_parse_dock_pending_log_total_open():
    # skill: "Count all rows where Resolved = FALSE"
    result = parse_dock_pending_log(raw=DOCK_LOG_CSV)
    assert result["total_open"] == 4


def test_parse_dock_pending_log_excludes_true():
    result = parse_dock_pending_log(raw=DOCK_LOG_CSV)
    assert result["total_open"] == 4   # the TRUE row for WEST2/PO Number is excluded


def test_parse_dock_pending_log_by_site():
    # skill: "Break down by site (EAST1, WEST2, etc.)"
    result = parse_dock_pending_log(raw=DOCK_LOG_CSV)
    assert result["by_site"]["EAST1"] == 2
    assert result["by_site"]["WEST2"] == 1
    assert result["by_site"]["NORTH3"] == 1


def test_parse_dock_pending_log_by_issue_type():
    # skill: "issue type (PO Number, Pallet/Piece Count Discrepancy)"
    result = parse_dock_pending_log(raw=DOCK_LOG_CSV)
    assert result["by_issue_type"]["PO Number"] == 2
    assert result["by_issue_type"]["Pallet/Piece Count Discrepancy"] == 2


def test_parse_dock_pending_log_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        parse_dock_pending_log(raw="")


def test_parse_dock_pending_log_missing_column_raises():
    with pytest.raises(ValueError):
        parse_dock_pending_log(raw="SITE,ISSUE TYPE\nEAST1,PO Number\n")


def test_parse_dock_pending_log_all_resolved_zero_open():
    all_resolved = "SITE,ISSUE TYPE,RESOLVED\nEAST1,PO Number,TRUE\n"
    result = parse_dock_pending_log(raw=all_resolved)
    assert result["total_open"] == 0
    assert result["by_site"] == {}


# ─────────────────────────────────────────────────────────────
# parse_dock_activity_emails
# ─────────────────────────────────────────────────────────────

def _make_gmail_payload(threads):
    return {"threads": threads}


def test_parse_dock_activity_emails_approved_count():
    # skill: "Count Approved vs. Requested appointments"
    payload = _make_gmail_payload([
        {"subject": "Dock Appointment Approved - EAST1 08:00"},
        {"subject": "Dock Appointment Approved - WEST2 10:00"},
        {"subject": "Dock Appointment Requested - NORTH3 14:00"},
    ])
    result = parse_dock_activity_emails(raw=payload)
    assert result["approved_count"] == 2
    assert result["requested_count"] == 1


def test_parse_dock_activity_emails_unapproved_requests():
    # skill: "Flag any outstanding requests not yet approved"
    payload = _make_gmail_payload([
        {"subject": "Dock Appointment Approved - EAST1"},
        {"subject": "Dock Appointment Requested - WEST2"},
        {"subject": "Dock Appointment Requested - NORTH3"},
    ])
    result = parse_dock_activity_emails(raw=payload)
    assert len(result["unapproved_requests"]) == 2
    assert "WEST2" in result["unapproved_requests"][0]


def test_parse_dock_activity_emails_accepts_json_string():
    payload = json.dumps({"threads": [{"subject": "Approved - EAST1"}]})
    result = parse_dock_activity_emails(raw=payload)
    assert result["approved_count"] == 1


def test_parse_dock_activity_emails_snippet_fallback():
    payload = _make_gmail_payload([{"snippet": "Dock appointment requested for EAST1"}])
    result = parse_dock_activity_emails(raw=payload)
    assert result["requested_count"] == 1


def test_parse_dock_activity_emails_empty_threads():
    result = parse_dock_activity_emails(raw={"threads": []})
    assert result["approved_count"] == 0
    assert result["requested_count"] == 0
    assert result["total_threads"] == 0


def test_parse_dock_activity_emails_invalid_json_raises():
    with pytest.raises(ValueError):
        parse_dock_activity_emails(raw="not json at all")


# ─────────────────────────────────────────────────────────────
# parse_dwell_tickets
# ─────────────────────────────────────────────────────────────

DWELL_CSV = (
    "Open/Closed,Date,Facility,# Packages,Client Impact,Carrier Impact,Issue Description\n"
    "Open,2026-10-17,EAST1,24,Client A,Carrier X,Delay in processing\n"
    "Open,2026-10-17,EAST1,25,Client B,Carrier Y,Package scan issue\n"
    "Open,2026-10-17,WEST2,40,Client C,Carrier Z,Missing labels\n"
    "Open,2026-10-17,WEST2,41,Client D,Carrier W,Sort error\n"
    "Open,2026-10-17,NORTH3,80,Client E,Carrier V,Missort bulk\n"
    "Closed,2026-10-16,EAST1,50,Client F,Carrier U,Resolved\n"
)


def test_parse_dwell_tickets_filters_open_only():
    # skill: "Filter rows where Open/Closed column = 'Open'"
    result = parse_dwell_tickets(raw=DWELL_CSV)
    assert result["total_open"] == 5  # 5 Open rows, 1 Closed excluded


def test_parse_dwell_tickets_below_threshold_no_alert():
    result = parse_dwell_tickets(raw=DWELL_CSV)
    t = next(t for t in result["open_tickets"] if t["packages"] == 24)
    assert t["alert_level"] == ""   # 24 < 25, no alert


def test_parse_dwell_tickets_warning_boundary():
    # skill: "packages >= 25 (Missort Warning)" — boundary: 25 triggers warning
    result = parse_dwell_tickets(raw=DWELL_CSV)
    t = next(t for t in result["open_tickets"] if t["packages"] == 25)
    assert t["alert_level"] == "Missort Warning"


def test_parse_dwell_tickets_warning_upper_boundary():
    # boundary: 40 is still warning (< 41)
    result = parse_dwell_tickets(raw=DWELL_CSV)
    t = next(t for t in result["open_tickets"] if t["packages"] == 40)
    assert t["alert_level"] == "Missort Warning"


def test_parse_dwell_tickets_alert_boundary():
    # skill: "packages >= 41 (Missort Alert)" — boundary: 41 triggers alert
    result = parse_dwell_tickets(raw=DWELL_CSV)
    t = next(t for t in result["open_tickets"] if t["packages"] == 41)
    assert t["alert_level"] == "Missort Alert"


def test_parse_dwell_tickets_carrier_alert_boundary():
    # skill: "carrier >= 80 (Possible Misload Alert)" — boundary: 80 triggers alert
    result = parse_dwell_tickets(raw=DWELL_CSV)
    t = next(t for t in result["open_tickets"] if t["packages"] == 80)
    assert t["alert_level"] == "Possible Misload Alert"


def test_parse_dwell_tickets_fields():
    result = parse_dwell_tickets(raw=DWELL_CSV)
    t = result["open_tickets"][0]
    assert "date" in t
    assert "facility" in t
    assert "packages" in t
    assert "client_impact" in t
    assert "carrier_impact" in t
    assert "issue_description" in t
    assert "alert_level" in t


def test_parse_dwell_tickets_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        parse_dwell_tickets(raw="")


def test_parse_dwell_tickets_missing_column_raises():
    bad = "Open/Closed,Date,Facility\nOpen,2026-10-17,EAST1\n"
    with pytest.raises(ValueError):
        parse_dwell_tickets(raw=bad)


def test_parse_dwell_tickets_all_closed_returns_zero():
    all_closed = (
        "Open/Closed,Date,Facility,# Packages,Client Impact,Carrier Impact,Issue Description\n"
        "Closed,2026-10-17,EAST1,50,C,C,Resolved\n"
    )
    result = parse_dwell_tickets(raw=all_closed)
    assert result["total_open"] == 0
    assert result["open_tickets"] == []
