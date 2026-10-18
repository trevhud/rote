"""Tests for extracted/report.py and extracted/email_aggregator.py."""

import pytest
from extracted.report import render_ops_report
from extracted.email_aggregator import aggregate_dock_emails


# ════════════════════════════════════════════════════════════════════════════
# aggregate_dock_emails
# ════════════════════════════════════════════════════════════════════════════

_CLASSIFIED = [
    {"classification": "appointment_approved", "appointment_ref": "APT-001", "thread_id": "t1"},
    {"classification": "appointment_requested", "appointment_ref": "APT-002", "thread_id": "t2"},
    {"classification": "appointment_requested", "appointment_ref": "APT-001", "thread_id": "t3"},  # matched to approval
    {"classification": "other", "appointment_ref": None, "thread_id": "t4"},
]

def test_aggregate_approved_count():
    result = aggregate_dock_emails(_CLASSIFIED)
    assert result["approved_count"] == 1

def test_aggregate_requested_count():
    result = aggregate_dock_emails(_CLASSIFIED)
    assert result["requested_count"] == 2

def test_aggregate_outstanding_unmatched():
    result = aggregate_dock_emails(_CLASSIFIED)
    # APT-002 was requested but not approved; APT-001 was requested and approved
    assert "APT-002" in result["outstanding_requests"]
    assert "APT-001" not in result["outstanding_requests"]

def test_aggregate_empty():
    result = aggregate_dock_emails([])
    assert result["approved_count"] == 0
    assert result["requested_count"] == 0
    assert result["outstanding_requests"] == []


# ════════════════════════════════════════════════════════════════════════════
# render_ops_report — fixtures
# ════════════════════════════════════════════════════════════════════════════

_SHIPMENT = {
    "sites": [
        {"site": "EAST1", "active": 10, "loaded": 5, "staged": 3, "shipped": 2,
         "completion_pct": 80.0, "wow_trend_pct": None, "flagged": False},
        {"site": "WEST2", "active": 8, "loaded": 4, "staged": 2, "shipped": 1,
         "completion_pct": 70.0, "wow_trend_pct": None, "flagged": True},
    ],
    "flagged_sites": ["WEST2"],
    "avg_completion_pct": 75.0,
    "total_active": 18,
    "active_site_count": 2,
}

_DOCK_PENDING = {
    "total_open": 3,
    "by_site": {"EAST1": 2, "WEST2": 1},
    "by_issue_type": {"PO Number": 2, "Pallet/Piece Count Discrepancy": 1},
}

_EMAIL_DATA = {
    "approved_count": 5,
    "requested_count": 6,
    "outstanding_requests": ["Dock Appt Request: WEST2 1400"],
}

_DWELL_TICKETS = {
    "open_tickets": [
        {"date": "2026-10-17", "facility": "EAST1", "packages": 41,
         "client_impact": "BrandX", "carrier_impact": "CarrierX",
         "issue_description": "Packages dwelling over 24h",
         "alert_level": "missort_alert", "misload_alert": False},
    ],
    "missort_warnings": [],
    "missort_alerts": [{"date": "2026-10-17", "facility": "EAST1", "packages": 41,
                        "client_impact": "BrandX", "carrier_impact": "CarrierX",
                        "issue_description": "Packages dwelling over 24h",
                        "alert_level": "missort_alert", "misload_alert": False}],
    "misload_alerts": [],
    "total_open": 1,
}

_MANUAL = {
    "appointment_summary": {
        "green": 10, "red": 2, "grey": 1, "other": 1,
        "total": 14, "non_compliant_count": 1, "non_compliant_escalated": True,
    },
    "sorter_sites": [
        {"site": "EAST1", "volume_output": 1200, "fail_rate_pct": 6.0,
         "prev_day_volume": 1200, "fail_rate_flagged": True,
         "volume_drop_flagged": False, "volume_drop_pct": 0.0},
    ],
    "flagged_sorter_sites": ["EAST1"],
    "dwell_alerts": [
        {"shipper": "BrandY", "carrier": "CarrierY", "site": "WEST2",
         "packages": 41, "alert_level": "missort_alert", "misload_alert": False},
    ],
    "dwell_warning_entries": [],
    "dwell_alert_entries": [
        {"shipper": "BrandY", "carrier": "CarrierY", "site": "WEST2",
         "packages": 41, "alert_level": "missort_alert", "misload_alert": False},
    ],
    "misload_alert_entries": [],
    "non_compliant_appointments": 1,
}


def _render():
    return render_ops_report(_SHIPMENT, _DOCK_PENDING, _EMAIL_DATA, _DWELL_TICKETS, _MANUAL)


# ── structural checks ─────────────────────────────────────────────────────────

def test_render_returns_report_markdown_key():
    result = _render()
    assert "report_markdown" in result

def test_render_title():
    result = _render()
    assert "Daily Executive Operations Report" in result["report_markdown"]

def test_render_kpi_header_present():
    md = _render()["report_markdown"]
    assert "KPI Summary" in md

def test_render_kpi_active_sites():
    md = _render()["report_markdown"]
    assert "Active Sites" in md
    assert "2" in md

def test_render_kpi_dock_pending():
    md = _render()["report_markdown"]
    assert "Dock Pending" in md
    assert "3" in md

def test_render_kpi_avg_completion():
    md = _render()["report_markdown"]
    assert "75.0%" in md

def test_render_kpi_open_dwell_tickets():
    md = _render()["report_markdown"]
    assert "Open Dwell Tickets" in md
    assert "1" in md

# ── Section 1 ─────────────────────────────────────────────────────────────────

def test_render_section1_present():
    md = _render()["report_markdown"]
    assert "Section 1" in md
    assert "Dock Compliance" in md

def test_render_section1_escalation_note():
    md = _render()["report_markdown"]
    assert "ESCALATION" in md or "escalat" in md.lower()

def test_render_section1_gmail_summary():
    md = _render()["report_markdown"]
    assert "Gmail" in md or "dock-activity" in md.lower() or "Dock Activity" in md

def test_render_section1_pending_table():
    md = _render()["report_markdown"]
    assert "EAST1" in md
    assert "WEST2" in md

# ── Section 2 ─────────────────────────────────────────────────────────────────

def test_render_section2_present():
    md = _render()["report_markdown"]
    assert "Section 2" in md
    assert "Missort" in md

def test_render_section2_sorter_fail_rate_flag():
    md = _render()["report_markdown"]
    assert "Fail Rate" in md or "fail rate" in md.lower()

def test_render_section2_container_table():
    md = _render()["report_markdown"]
    assert "Completion %" in md
    assert "Below 75%" in md or "below 75" in md.lower() or "⚠" in md

# ── Section 3 ─────────────────────────────────────────────────────────────────

def test_render_section3_present():
    md = _render()["report_markdown"]
    assert "Section 3" in md
    assert "Dwell" in md

def test_render_section3_missort_alert_label():
    md = _render()["report_markdown"]
    assert "Missort Alert" in md

def test_render_section3_open_ticket():
    md = _render()["report_markdown"]
    assert "2026-10-17" in md or "EAST1" in md

# ── Closing reminder ──────────────────────────────────────────────────────────

def test_render_includes_artifact_reminder():
    md = _render()["report_markdown"]
    assert "ops-report" in md
    assert "auto-refreshes" in md or "interactive" in md

# ── No-data smoke test ────────────────────────────────────────────────────────

def test_render_empty_data_no_crash():
    empty_shipment = {"sites": [], "flagged_sites": [], "avg_completion_pct": 0.0,
                      "total_active": 0, "active_site_count": 0}
    empty_pending = {"total_open": 0, "by_site": {}, "by_issue_type": {}}
    empty_email = {"approved_count": 0, "requested_count": 0, "outstanding_requests": []}
    empty_dwell = {"open_tickets": [], "missort_warnings": [], "missort_alerts": [],
                   "misload_alerts": [], "total_open": 0}
    empty_manual = {
        "appointment_summary": {"green": 0, "red": 0, "grey": 0, "other": 0,
                                "total": 0, "non_compliant_count": 0, "non_compliant_escalated": False},
        "sorter_sites": [], "flagged_sorter_sites": [], "dwell_alerts": [],
        "dwell_warning_entries": [], "dwell_alert_entries": [], "misload_alert_entries": [],
        "non_compliant_appointments": 0,
    }
    result = render_ops_report(empty_shipment, empty_pending, empty_email,
                               empty_dwell, empty_manual)
    assert "report_markdown" in result
    assert len(result["report_markdown"]) > 100
