from extracted.render_ops_report import render_ops_report, ALERT_LABELS


SHIPMENT_DATA = {
    "sites": [
        {
            "site": "EAST1", "active": 10, "loaded": 5, "staged": 3, "shipped": 2,
            "completion_pct": 80.0, "flagged": False, "wow_trend": 2.0,
        },
        {
            "site": "WEST2", "active": 6, "loaded": 2, "staged": 1, "shipped": 1,
            "completion_pct": 70.0, "flagged": True, "wow_trend": None,
        },
    ],
    "flagged_sites": ["WEST2"],
    "total_active": 16,
    "avg_completion_pct": 75.0,
}

DOCK_PENDING_DATA = {
    "total_open": 5,
    "by_site": {"EAST1": 3, "WEST2": 2},
    "by_issue_type": {"PO Number": 3, "Pallet/Piece Count Discrepancy": 2},
}

EMAIL_DATA = {
    "approved_count": 8,
    "requested_count": 2,
    "outstanding_requests": [{"snippet": "appointment pending"}],
    "total_threads": 10,
}

DWELL_DATA = {
    "open_tickets": [
        {
            "date": "2026-10-18", "facility": "EAST1", "packages": 45,
            "client_impact": "BrandX", "carrier_impact": "FedEx",
            "issue_description": "Delayed scan", "alert_level": "Missort Alert",
        },
    ],
    "total_open": 1,
    "alerts": {"missort_warnings": 0, "missort_alerts": 1, "possible_misload_alerts": 0},
}

MANUAL_DATA_ESCALATION = {
    "dock_appointments": {
        "green": 10, "red": 2, "grey": 1, "non_compliant": 1, "escalation_required": True,
    },
    "sorter_metrics": [
        {
            "site": "EAST1", "volume_output": 840, "fail_rate_pct": 6.2,
            "previous_volume": 1000, "volume_drop_pct": 16.0,
            "fail_rate_flag": True, "volume_drop_flag": True,
        },
    ],
    "current_dwell": [
        {
            "brand": "BrandX", "carrier": "FedEx", "site": "EAST1",
            "packages": 30, "carrier_packages": 0, "alert_level": "Missort Warning",
        },
    ],
}

MANUAL_DATA_CLEAN = {
    "dock_appointments": {
        "green": 10, "red": 0, "grey": 0, "non_compliant": 0, "escalation_required": False,
    },
    "sorter_metrics": [],
    "current_dwell": [],
}


def render_full():
    return render_ops_report(
        SHIPMENT_DATA, DOCK_PENDING_DATA, EMAIL_DATA, DWELL_DATA, MANUAL_DATA_ESCALATION
    )


# --- required section headings ---

def test_report_contains_kpi_summary():
    # skill: "KPI header: Active sites, Dock Pending count, Avg Completion %, Open Dwell Tickets"
    assert "KPI Summary" in render_full()


def test_report_contains_section_1():
    # skill: "Section 1 — Dock Compliance"
    assert "Section 1" in render_full()
    assert "Dock Compliance" in render_full()


def test_report_contains_section_2():
    # skill: "Section 2 — Missort & Loss"
    assert "Section 2" in render_full()
    assert "Missort" in render_full()


def test_report_contains_section_3():
    # skill: "Section 3 — Dwell"
    assert "Section 3" in render_full()
    assert "Dwell" in render_full()


# --- KPI values appear in output ---

def test_kpi_active_sites_count():
    output = render_full()
    assert "| Active Sites | 2 |" in output


def test_kpi_dock_pending_count():
    output = render_full()
    assert "| Dock Pending (Open) | 5 |" in output


def test_kpi_open_dwell_tickets():
    output = render_full()
    assert "| Open Dwell Tickets | 1 |" in output


# --- site names appear ---

def test_east1_site_appears():
    assert "EAST1" in render_full()


def test_west2_site_appears():
    assert "WEST2" in render_full()


# --- escalation flag ---

def test_escalation_required_appears_in_output():
    # skill: "Non-Compliant must be escalated"
    assert "ESCALATION REQUIRED" in render_full()


def test_no_escalation_when_zero_non_compliant():
    output = render_ops_report(
        SHIPMENT_DATA, DOCK_PENDING_DATA, EMAIL_DATA, DWELL_DATA, MANUAL_DATA_CLEAN
    )
    assert "ESCALATION REQUIRED" not in output


# --- sorter flags ---

def test_sorter_fail_rate_flag_appears():
    assert "ACTION: Fail rate > 5%" in render_full()


def test_sorter_volume_drop_flag_appears():
    assert "Divert adj. needed" in render_full()


# --- dwell alert labels ---

def test_missort_alert_label_appears_in_dwell_log():
    assert ALERT_LABELS["Missort Alert"] in render_full()
    assert "[MISSORT ALERT]" in render_full()


def test_missort_warning_label_appears_in_current_dwell():
    assert ALERT_LABELS["Missort Warning"] in render_full()
    assert "[MISSORT WARNING]" in render_full()


# --- flagged site notice ---

def test_below_75_notice_for_west2():
    assert "WEST2" in render_full()
    assert "75%" in render_full()


# --- outstanding email requests ---

def test_outstanding_requests_mention():
    assert "Outstanding" in render_full()


# --- live artifact reminder ---

def test_reminder_contains_ops_report():
    # skill: "remind the duty manager to open the live artifact 'ops-report'"
    assert "ops-report" in render_full()


# --- empty data graceful handling ---

def test_empty_shipment_data_does_not_crash():
    empty_shipment = {"sites": [], "flagged_sites": [], "total_active": 0, "avg_completion_pct": 0.0}
    output = render_ops_report(
        empty_shipment, DOCK_PENDING_DATA, EMAIL_DATA, DWELL_DATA, MANUAL_DATA_CLEAN
    )
    assert "KPI Summary" in output
    assert "No container data available" in output


def test_empty_dwell_tickets_shows_no_open_message():
    empty_dwell = {"open_tickets": [], "total_open": 0, "alerts": {}}
    output = render_ops_report(
        SHIPMENT_DATA, DOCK_PENDING_DATA, EMAIL_DATA, empty_dwell, MANUAL_DATA_CLEAN
    )
    assert "No open dwell tickets" in output
