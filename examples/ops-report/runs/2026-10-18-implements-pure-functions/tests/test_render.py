"""
Tests for extracted/render.py.

Verifies that every section heading, KPI, and alert label the skill names
appears in the rendered output for a fixture that should produce it, and is
absent when it should not be.
"""

import pytest

from extracted.render import render_report

# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

SHIPMENT_DATA_BASE = {
    "sites": [
        {
            "site": "EAST1",
            "active": 5, "loaded": 10, "staged": 3, "shipped": 20,
            "completion_pct": 80.0, "flagged": False, "wow_trend": 2.0,
        },
        {
            "site": "WEST2",
            "active": 2, "loaded": 8, "staged": 1, "shipped": 15,
            "completion_pct": 70.0, "flagged": True, "wow_trend": None,
        },
    ],
    "avg_completion_pct": 75.0,
    "active_site_count": 2,
    "flagged_sites": ["WEST2"],
}

DOCK_PENDING_BASE = {
    "total_open": 4,
    "by_site": {"EAST1": 2, "WEST2": 2},
    "by_issue_type": {"PO Number": 3, "Pallet/Piece Count Discrepancy": 1},
}

DOCK_EMAIL_BASE = {
    "approved_count": 5,
    "requested_count": 2,
    "unapproved_requests": ["Dock Appointment Requested - NORTH3 14:00"],
    "total_threads": 7,
}

DWELL_TICKETS_BASE = {
    "open_tickets": [
        {
            "date": "2026-10-17",
            "facility": "EAST1",
            "packages": 45,
            "client_impact": "Client A",
            "carrier_impact": "Carrier X",
            "issue_description": "Sort error",
            "alert_level": "Missort Alert",
        }
    ],
    "total_open": 1,
}

APPOINTMENT_DATA_BASE = {
    "green": 10, "red": 2, "grey": 1, "other": 0,
    "non_compliant_count": 0, "escalation_required": False,
}

SORTER_DATA_BASE = {
    "sites": [
        {"site": "EAST1", "volume": 1200, "fail_rate_pct": 3.0, "prev_volume": 1000, "flags": []},
    ],
    "action_required_sites": [],
    "divert_adjustment_sites": [],
}

DWELL_DATA_BASE = {
    "all_rows": [
        {"brand": "ACME", "carrier": "FedEx", "site": "EAST1", "packages": 30, "alert_level": "Missort Warning"}
    ],
    "flagged_rows": [
        {"brand": "ACME", "carrier": "FedEx", "site": "EAST1", "packages": 30, "alert_level": "Missort Warning"}
    ],
}


def _render(**overrides):
    kwargs = dict(
        shipment_data=SHIPMENT_DATA_BASE,
        dock_pending_data=DOCK_PENDING_BASE,
        dock_email_data=DOCK_EMAIL_BASE,
        dwell_tickets_data=DWELL_TICKETS_BASE,
        appointment_data=APPOINTMENT_DATA_BASE,
        sorter_data=SORTER_DATA_BASE,
        dwell_data=DWELL_DATA_BASE,
    )
    kwargs.update(overrides)
    return render_report(**kwargs)


# ─────────────────────────────────────────────────────────────
# Structure: section headings always present
# ─────────────────────────────────────────────────────────────

def test_render_kpi_header_section():
    # skill: "KPI header: Active sites, Dock Pending count, Avg Completion %, Open Dwell Tickets"
    output = _render()
    assert "KPI" in output
    assert "Active" in output
    assert "Dock Pending" in output


def test_render_section_1_heading():
    # skill: "Section 1 — Dock Compliance"
    output = _render()
    assert "Dock Compliance" in output


def test_render_section_2_heading():
    # skill: "Section 2 — Missort & Loss"
    output = _render()
    assert "Missort & Loss" in output


def test_render_section_3_heading():
    # skill: "Section 3 — Dwell"
    output = _render()
    assert "Dwell" in output


# ─────────────────────────────────────────────────────────────
# KPI values appear in output
# ─────────────────────────────────────────────────────────────

def test_render_active_site_count_in_kpi():
    output = _render()
    assert "2" in output  # active_site_count


def test_render_dock_pending_count_in_kpi():
    output = _render()
    assert "4" in output  # total_open dock pending


def test_render_avg_completion_in_kpi():
    output = _render()
    assert "75.0%" in output


def test_render_open_dwell_tickets_in_kpi():
    output = _render()
    assert "1" in output  # open dwell tickets


# ─────────────────────────────────────────────────────────────
# Section 1: escalation flag
# ─────────────────────────────────────────────────────────────

def test_render_escalation_shown_when_non_compliant():
    # skill: "Non-Compliant must be escalated"
    appt_with_nc = dict(APPOINTMENT_DATA_BASE, other=3, non_compliant_count=3, escalation_required=True)
    output = _render(appointment_data=appt_with_nc)
    assert "ESCALATION" in output


def test_render_no_escalation_when_compliant():
    output = _render()
    assert "ESCALATION" not in output


# ─────────────────────────────────────────────────────────────
# Section 1: unapproved requests flag
# ─────────────────────────────────────────────────────────────

def test_render_unapproved_requests_flagged():
    # skill: "Flag any outstanding requests not yet approved"
    output = _render()
    assert "Outstanding requests not yet approved" in output
    assert "NORTH3" in output


def test_render_no_unapproved_flag_when_none():
    email_no_pending = dict(DOCK_EMAIL_BASE, requested_count=0, unapproved_requests=[])
    output = _render(dock_email_data=email_no_pending)
    assert "Outstanding requests not yet approved" not in output


# ─────────────────────────────────────────────────────────────
# Section 2: completion flag
# ─────────────────────────────────────────────────────────────

def test_render_completion_flag_present_for_flagged_site():
    # skill: "Flag any site below 75% completion"
    output = _render()
    assert "BELOW 75%" in output


def test_render_no_completion_flag_for_passing_sites():
    passing_shipment = dict(SHIPMENT_DATA_BASE)
    # Override with no flagged sites
    passing_shipment = {
        "sites": [
            {"site": "EAST1", "active": 5, "loaded": 10, "staged": 3, "shipped": 20,
             "completion_pct": 80.0, "flagged": False, "wow_trend": None},
        ],
        "avg_completion_pct": 80.0,
        "active_site_count": 1,
        "flagged_sites": [],
    }
    output = _render(shipment_data=passing_shipment)
    assert "BELOW 75%" not in output


# ─────────────────────────────────────────────────────────────
# Section 3: alert labels
# ─────────────────────────────────────────────────────────────

def test_render_missort_warning_label():
    # skill: "Brand 25-40 = Missort Warning (light red)"
    output = _render()
    assert "Missort Warning" in output


def test_render_missort_alert_in_ticket_log():
    # skill: "41+ = Missort Alert (bright red)"
    output = _render()
    assert "Missort Alert" in output


def test_render_no_dwell_alerts_message():
    empty_dwell = {"all_rows": [], "flagged_rows": []}
    output = _render(dwell_data=empty_dwell)
    assert "No dwell alerts" in output


def test_render_no_open_tickets_message():
    empty_tickets = {"open_tickets": [], "total_open": 0}
    output = _render(dwell_tickets_data=empty_tickets)
    assert "No open dwell tickets" in output


# ─────────────────────────────────────────────────────────────
# Reminder line
# ─────────────────────────────────────────────────────────────

def test_render_artifact_reminder_present():
    # skill: "Also remind the duty manager to open the live artifact 'ops-report'..."
    output = _render()
    assert "ops-report" in output
    assert "interactive" in output


# ─────────────────────────────────────────────────────────────
# Sorter flags surface in output
# ─────────────────────────────────────────────────────────────

def test_render_sorter_action_needed_flag():
    # skill: "Flag if fail rate > 5% (action needed)"
    sorter_with_flag = {
        "sites": [
            {"site": "EAST1", "volume": 1000, "fail_rate_pct": 6.5,
             "prev_volume": None, "flags": ["ACTION NEEDED"]},
        ],
        "action_required_sites": ["EAST1"],
        "divert_adjustment_sites": [],
    }
    output = _render(sorter_data=sorter_with_flag)
    assert "ACTION NEEDED" in output


def test_render_returns_string():
    output = _render()
    assert isinstance(output, str)
    assert len(output) > 0
