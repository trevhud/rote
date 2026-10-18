"""Tests for extracted/thresholds.py — derived from skill rules."""

import pytest
from extracted.thresholds import (
    SORTER_FAIL_RATE_THRESHOLD_PCT,
    SORTER_VOLUME_DROP_THRESHOLD_PCT,
    DWELL_WARNING_PACKAGES,
    DWELL_ALERT_PACKAGES,
    DWELL_MISLOAD_CARRIER,
    apply_manual_thresholds,
)


# ════════════════════════════════════════════════════════════════════════════
# Constants pinned
# ════════════════════════════════════════════════════════════════════════════

def test_sorter_fail_rate_threshold():
    assert SORTER_FAIL_RATE_THRESHOLD_PCT == 5.0

def test_sorter_volume_drop_threshold():
    assert SORTER_VOLUME_DROP_THRESHOLD_PCT == 15.0

def test_dwell_warning_threshold():
    assert DWELL_WARNING_PACKAGES == 25

def test_dwell_alert_threshold():
    assert DWELL_ALERT_PACKAGES == 41

def test_dwell_misload_threshold():
    assert DWELL_MISLOAD_CARRIER == 80


# ════════════════════════════════════════════════════════════════════════════
# Dock appointment compliance
# ════════════════════════════════════════════════════════════════════════════

_CLEAN_APPT = {"green": 10, "red": 2, "grey": 1, "other": 0}
_NON_COMPLIANT_APPT = {"green": 8, "red": 3, "grey": 2, "other": 2}

def test_appt_no_non_compliant():
    result = apply_manual_thresholds(_CLEAN_APPT, [], [])
    appt = result["appointment_summary"]
    assert appt["non_compliant_count"] == 0
    assert appt["non_compliant_escalated"] is False

def test_appt_non_compliant_triggers_escalation():
    result = apply_manual_thresholds(_NON_COMPLIANT_APPT, [], [])
    appt = result["appointment_summary"]
    assert appt["non_compliant_count"] == 2
    assert appt["non_compliant_escalated"] is True

def test_appt_total_count():
    result = apply_manual_thresholds(_NON_COMPLIANT_APPT, [], [])
    appt = result["appointment_summary"]
    assert appt["total"] == 15  # 8+3+2+2

def test_non_compliant_appointments_field():
    result = apply_manual_thresholds(_NON_COMPLIANT_APPT, [], [])
    assert result["non_compliant_appointments"] == 2


# ════════════════════════════════════════════════════════════════════════════
# Sorter metrics
# ════════════════════════════════════════════════════════════════════════════

_SORTER_OK = [{"site": "EAST1", "volume_output": 1000, "fail_rate_pct": 4.9, "prev_day_volume": 1010}]
_SORTER_FAIL_RATE = [{"site": "WEST2", "volume_output": 900, "fail_rate_pct": 5.1, "prev_day_volume": 900}]
_SORTER_VOLUME_DROP = [{"site": "NORTH3", "volume_output": 840, "fail_rate_pct": 3.0, "prev_day_volume": 1000}]
_SORTER_NO_PREV = [{"site": "SOUTH4", "volume_output": 800, "fail_rate_pct": 2.0, "prev_day_volume": None}]

def test_sorter_fail_rate_below_threshold_not_flagged():
    result = apply_manual_thresholds(_CLEAN_APPT, _SORTER_OK, [])
    assert result["flagged_sorter_sites"] == []
    assert result["sorter_sites"][0]["fail_rate_flagged"] is False

def test_sorter_fail_rate_above_5_flagged():
    result = apply_manual_thresholds(_CLEAN_APPT, _SORTER_FAIL_RATE, [])
    assert "WEST2" in result["flagged_sorter_sites"]
    assert result["sorter_sites"][0]["fail_rate_flagged"] is True

def test_sorter_volume_drop_above_15_flagged():
    # 1000 -> 840 = 16% drop
    result = apply_manual_thresholds(_CLEAN_APPT, _SORTER_VOLUME_DROP, [])
    assert "NORTH3" in result["flagged_sorter_sites"]
    assert result["sorter_sites"][0]["volume_drop_flagged"] is True

def test_sorter_volume_drop_at_15_not_flagged():
    # Exactly 15% drop — should NOT be flagged (rule is > 15%)
    sorter = [{"site": "SITE5", "volume_output": 850, "fail_rate_pct": 2.0, "prev_day_volume": 1000}]
    result = apply_manual_thresholds(_CLEAN_APPT, sorter, [])
    assert result["sorter_sites"][0]["volume_drop_flagged"] is False

def test_sorter_no_prev_day_volume_no_drop_flag():
    result = apply_manual_thresholds(_CLEAN_APPT, _SORTER_NO_PREV, [])
    assert result["sorter_sites"][0]["volume_drop_pct"] is None
    assert result["sorter_sites"][0]["volume_drop_flagged"] is False


# ════════════════════════════════════════════════════════════════════════════
# Current dwell thresholds
# ════════════════════════════════════════════════════════════════════════════

_DWELL_ENTRIES = [
    {"shipper": "BrandA", "carrier": "CarrierA", "site": "EAST1", "packages": 24},
    {"shipper": "BrandB", "carrier": "CarrierB", "site": "EAST1", "packages": 25},
    {"shipper": "BrandC", "carrier": "CarrierC", "site": "WEST2", "packages": 40},
    {"shipper": "BrandD", "carrier": "CarrierD", "site": "WEST2", "packages": 41},
    {"shipper": "BrandE", "carrier": "CarrierE", "site": "NORTH3", "packages": 80},
]

def test_dwell_below_25_no_alert():
    result = apply_manual_thresholds(_CLEAN_APPT, [], _DWELL_ENTRIES)
    entry = next(e for e in result["dwell_alerts"] if e["packages"] == 24)
    assert entry["alert_level"] == "none"

def test_dwell_exactly_25_warning():
    result = apply_manual_thresholds(_CLEAN_APPT, [], _DWELL_ENTRIES)
    entry = next(e for e in result["dwell_alerts"] if e["packages"] == 25)
    assert entry["alert_level"] == "missort_warning"
    assert entry in result["dwell_warning_entries"]

def test_dwell_40_warning_not_alert():
    result = apply_manual_thresholds(_CLEAN_APPT, [], _DWELL_ENTRIES)
    entry = next(e for e in result["dwell_alerts"] if e["packages"] == 40)
    assert entry["alert_level"] == "missort_warning"

def test_dwell_exactly_41_alert():
    result = apply_manual_thresholds(_CLEAN_APPT, [], _DWELL_ENTRIES)
    entry = next(e for e in result["dwell_alerts"] if e["packages"] == 41)
    assert entry["alert_level"] == "missort_alert"
    assert entry in result["dwell_alert_entries"]

def test_dwell_80_misload_alert():
    result = apply_manual_thresholds(_CLEAN_APPT, [], _DWELL_ENTRIES)
    entry = next(e for e in result["dwell_alerts"] if e["packages"] == 80)
    assert entry["misload_alert"] is True
    assert entry in result["misload_alert_entries"]

def test_dwell_79_no_misload():
    entries = [{"shipper": "BrandX", "carrier": "CarrierX", "site": "S1", "packages": 79}]
    result = apply_manual_thresholds(_CLEAN_APPT, [], entries)
    assert result["dwell_alerts"][0]["misload_alert"] is False
