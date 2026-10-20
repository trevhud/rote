import pytest
from extracted.parse_dwell_tickets import (
    parse_dwell_tickets,
    BRAND_WARNING_THRESHOLD,
    BRAND_ALERT_THRESHOLD,
    CARRIER_ALERT_THRESHOLD,
)

HDR = "Open/Closed,Date,Facility,# Packages,Client Impact,Carrier Impact,Issue Description\n"


def csv_rows(*rows):
    return HDR + "\n".join(",".join(str(v) for v in r) for r in rows)


# --- constant pins ---

def test_brand_warning_threshold_constant():
    assert BRAND_WARNING_THRESHOLD == 25


def test_brand_alert_threshold_constant():
    assert BRAND_ALERT_THRESHOLD == 41


def test_carrier_alert_threshold_constant():
    assert CARRIER_ALERT_THRESHOLD == 80


# --- open/closed filter ---

def test_only_open_rows_included():
    # skill: "Filter rows where Open/Closed column = 'Open'"
    data = csv_rows(
        ("Open", "2026-10-18", "EAST1", 10, "BrandX", "0", "Delayed"),
        ("Closed", "2026-10-18", "EAST1", 50, "BrandY", "0", "Resolved"),
    )
    result = parse_dwell_tickets(data)
    assert result["total_open"] == 1
    assert result["open_tickets"][0]["facility"] == "EAST1"
    assert result["open_tickets"][0]["packages"] == 10


# --- alert thresholds: brand ---

def test_packages_below_25_no_alert():
    data = csv_rows(("Open", "2026-10-18", "EAST1", 24, "", "0", ""))
    result = parse_dwell_tickets(data)
    assert result["open_tickets"][0]["alert_level"] is None


def test_packages_at_25_missort_warning():
    # skill: "packages >= 25 (Missort Warning)"
    data = csv_rows(("Open", "2026-10-18", "EAST1", 25, "", "0", ""))
    result = parse_dwell_tickets(data)
    assert result["open_tickets"][0]["alert_level"] == "Missort Warning"
    assert result["alerts"]["missort_warnings"] == 1


def test_packages_at_40_missort_warning():
    data = csv_rows(("Open", "2026-10-18", "EAST1", 40, "", "0", ""))
    result = parse_dwell_tickets(data)
    assert result["open_tickets"][0]["alert_level"] == "Missort Warning"


def test_packages_at_41_missort_alert():
    # skill: "packages >= 41 (Missort Alert)"
    data = csv_rows(("Open", "2026-10-18", "EAST1", 41, "", "0", ""))
    result = parse_dwell_tickets(data)
    assert result["open_tickets"][0]["alert_level"] == "Missort Alert"
    assert result["alerts"]["missort_alerts"] == 1


def test_packages_above_41_missort_alert():
    data = csv_rows(("Open", "2026-10-18", "EAST1", 100, "", "0", ""))
    result = parse_dwell_tickets(data)
    assert result["open_tickets"][0]["alert_level"] == "Missort Alert"


# --- alert thresholds: carrier ---

def test_carrier_at_79_no_carrier_alert():
    data = csv_rows(("Open", "2026-10-18", "EAST1", 10, "", "79", ""))
    result = parse_dwell_tickets(data)
    assert result["open_tickets"][0]["alert_level"] is None


def test_carrier_at_80_possible_misload_alert():
    # skill: "carrier >= 80 (Possible Misload Alert)"
    data = csv_rows(("Open", "2026-10-18", "EAST1", 10, "", "80", ""))
    result = parse_dwell_tickets(data)
    assert result["open_tickets"][0]["alert_level"] == "Possible Misload Alert"
    assert result["alerts"]["possible_misload_alerts"] == 1


def test_carrier_priority_over_brand_alert():
    # rote: assumption: Possible Misload Alert (darkest red) takes priority
    data = csv_rows(("Open", "2026-10-18", "EAST1", 50, "", "80", ""))
    result = parse_dwell_tickets(data)
    assert result["open_tickets"][0]["alert_level"] == "Possible Misload Alert"


# --- empty / error cases ---

def test_empty_string_returns_empty_result():
    result = parse_dwell_tickets("")
    assert result["open_tickets"] == []
    assert result["total_open"] == 0
    assert result["alerts"] == {"missort_warnings": 0, "missort_alerts": 0, "possible_misload_alerts": 0}


def test_header_only_returns_empty():
    result = parse_dwell_tickets(HDR)
    assert result["total_open"] == 0


def test_missing_facility_column_raises():
    bad = "Open/Closed,Date,# Packages\nOpen,2026-10-18,30\n"
    with pytest.raises(ValueError, match="Facility"):
        parse_dwell_tickets(bad)


def test_missing_status_column_raises():
    bad = "Date,Facility,# Packages\n2026-10-18,EAST1,30\n"
    with pytest.raises(ValueError, match="Open/Closed"):
        parse_dwell_tickets(bad)


# --- alert summary counts ---

def test_alert_summary_counts_correctly():
    data = csv_rows(
        ("Open", "2026-10-18", "EAST1", 25, "", "0", ""),
        ("Open", "2026-10-18", "EAST1", 41, "", "0", ""),
        ("Open", "2026-10-18", "WEST2", 10, "", "80", ""),
        ("Open", "2026-10-18", "WEST2", 5, "", "0", ""),
    )
    result = parse_dwell_tickets(data)
    assert result["total_open"] == 4
    assert result["alerts"]["missort_warnings"] == 1
    assert result["alerts"]["missort_alerts"] == 1
    assert result["alerts"]["possible_misload_alerts"] == 1
