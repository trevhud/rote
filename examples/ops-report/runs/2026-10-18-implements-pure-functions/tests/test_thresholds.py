"""
Tests for extracted/thresholds.py.

Every threshold boundary from the skill is tested on both sides. Offline only.
"""

import pytest

from extracted.thresholds import (
    SORTER_FAIL_RATE_THRESHOLD,
    SORTER_VOLUME_DROP_THRESHOLD,
    DWELL_BRAND_WARNING_THRESHOLD,
    DWELL_BRAND_ALERT_THRESHOLD,
    DWELL_CARRIER_ALERT_THRESHOLD,
    apply_appointment_flags,
    apply_sorter_flags,
    apply_dwell_thresholds,
)


# ─────────────────────────────────────────────────────────────
# Constant-pinning
# ─────────────────────────────────────────────────────────────

def test_sorter_fail_rate_constant():
    assert SORTER_FAIL_RATE_THRESHOLD == 5.0


def test_sorter_volume_drop_constant():
    assert SORTER_VOLUME_DROP_THRESHOLD == 15.0


def test_dwell_brand_warning_constant():
    assert DWELL_BRAND_WARNING_THRESHOLD == 25


def test_dwell_brand_alert_constant():
    assert DWELL_BRAND_ALERT_THRESHOLD == 41


def test_dwell_carrier_alert_constant():
    assert DWELL_CARRIER_ALERT_THRESHOLD == 80


# ─────────────────────────────────────────────────────────────
# apply_appointment_flags
# ─────────────────────────────────────────────────────────────

def test_apply_appointment_flags_no_non_compliant():
    result = apply_appointment_flags(
        appointment_completion={"green": 10, "red": 2, "grey": 1, "other": 0}
    )
    assert result["escalation_required"] is False
    assert result["non_compliant_count"] == 0


def test_apply_appointment_flags_with_non_compliant():
    # skill: "Non-Compliant must be escalated"
    result = apply_appointment_flags(
        appointment_completion={"green": 8, "red": 1, "grey": 2, "other": 3}
    )
    assert result["escalation_required"] is True
    assert result["non_compliant_count"] == 3


def test_apply_appointment_flags_preserves_counts():
    result = apply_appointment_flags(
        appointment_completion={"green": 5, "red": 0, "grey": 3, "other": 1}
    )
    assert result["green"] == 5
    assert result["red"] == 0
    assert result["grey"] == 3
    assert result["other"] == 1


def test_apply_appointment_flags_missing_keys_default_zero():
    result = apply_appointment_flags(appointment_completion={})
    assert result["non_compliant_count"] == 0
    assert result["escalation_required"] is False


# ─────────────────────────────────────────────────────────────
# apply_sorter_flags
# ─────────────────────────────────────────────────────────────

def test_apply_sorter_flags_no_flags():
    metrics = [{"site": "EAST1", "volume": 1000, "fail_rate_pct": 5.0, "prev_volume": None}]
    result = apply_sorter_flags(sorter_metrics=metrics)
    assert result["sites"][0]["flags"] == []
    # skill: "fail rate > 5%" — exactly 5.0 is NOT flagged (strict >)
    assert "EAST1" not in result["action_required_sites"]


def test_apply_sorter_flags_fail_rate_above_threshold():
    # skill: "Flag if fail rate > 5% (action needed)"
    metrics = [{"site": "WEST2", "volume": 800, "fail_rate_pct": 5.1, "prev_volume": None}]
    result = apply_sorter_flags(sorter_metrics=metrics)
    assert "ACTION NEEDED" in result["sites"][0]["flags"]
    assert "WEST2" in result["action_required_sites"]


def test_apply_sorter_flags_volume_drop_above_threshold():
    # skill: "volume dropped >15% day-over-day (divert adjustment may be needed)"
    # 1000 → 840 = 16% drop (> 15%)
    metrics = [{"site": "EAST1", "volume": 840, "fail_rate_pct": 2.0, "prev_volume": 1000}]
    result = apply_sorter_flags(sorter_metrics=metrics)
    assert "DIVERT ADJUSTMENT MAY BE NEEDED" in result["sites"][0]["flags"]
    assert "EAST1" in result["divert_adjustment_sites"]


def test_apply_sorter_flags_volume_drop_exactly_15_not_flagged():
    # skill: "dropped >15%" — 15.0% is NOT flagged (strict >)
    # 1000 → 850 = exactly 15% drop
    metrics = [{"site": "EAST1", "volume": 850, "fail_rate_pct": 2.0, "prev_volume": 1000}]
    result = apply_sorter_flags(sorter_metrics=metrics)
    assert "DIVERT ADJUSTMENT MAY BE NEEDED" not in result["sites"][0]["flags"]


def test_apply_sorter_flags_no_prev_volume_skips_drop_check():
    metrics = [{"site": "EAST1", "volume": 500, "fail_rate_pct": 2.0, "prev_volume": None}]
    result = apply_sorter_flags(sorter_metrics=metrics)
    assert "DIVERT ADJUSTMENT MAY BE NEEDED" not in result["sites"][0]["flags"]


def test_apply_sorter_flags_empty_list():
    result = apply_sorter_flags(sorter_metrics=[])
    assert result["sites"] == []
    assert result["action_required_sites"] == []


def test_apply_sorter_flags_multiple_sites():
    metrics = [
        {"site": "EAST1", "volume": 1000, "fail_rate_pct": 6.0, "prev_volume": None},
        {"site": "WEST2", "volume": 500, "fail_rate_pct": 2.0, "prev_volume": None},
    ]
    result = apply_sorter_flags(sorter_metrics=metrics)
    assert "EAST1" in result["action_required_sites"]
    assert "WEST2" not in result["action_required_sites"]


# ─────────────────────────────────────────────────────────────
# apply_dwell_thresholds
# ─────────────────────────────────────────────────────────────

def test_apply_dwell_thresholds_no_alert():
    rows = [{"brand": "ACME", "carrier": "FedEx", "site": "EAST1", "packages": 24}]
    result = apply_dwell_thresholds(dwell_data=rows)
    assert result["flagged_rows"] == []
    assert result["all_rows"][0]["alert_level"] == ""


def test_apply_dwell_thresholds_warning_at_25():
    # skill: "Brand 25-40 = Missort Warning (light red)"
    rows = [{"brand": "ACME", "carrier": "FedEx", "site": "EAST1", "packages": 25}]
    result = apply_dwell_thresholds(dwell_data=rows)
    assert result["all_rows"][0]["alert_level"] == "Missort Warning"
    assert len(result["flagged_rows"]) == 1


def test_apply_dwell_thresholds_warning_at_40():
    rows = [{"brand": "ACME", "carrier": "FedEx", "site": "EAST1", "packages": 40}]
    result = apply_dwell_thresholds(dwell_data=rows)
    assert result["all_rows"][0]["alert_level"] == "Missort Warning"


def test_apply_dwell_thresholds_alert_at_41():
    # skill: "41+ = Missort Alert (bright red)"
    rows = [{"brand": "ACME", "carrier": "FedEx", "site": "EAST1", "packages": 41}]
    result = apply_dwell_thresholds(dwell_data=rows)
    assert result["all_rows"][0]["alert_level"] == "Missort Alert"


def test_apply_dwell_thresholds_carrier_alert_at_80():
    # skill: "Carrier 80+ missing = Possible Misload Alert (darkest red)"
    rows = [{"brand": "ACME", "carrier": "FedEx", "site": "EAST1", "packages": 80}]
    result = apply_dwell_thresholds(dwell_data=rows)
    assert result["all_rows"][0]["alert_level"] == "Possible Misload Alert"


def test_apply_dwell_thresholds_carrier_alert_at_79_is_missort_alert():
    rows = [{"brand": "ACME", "carrier": "FedEx", "site": "EAST1", "packages": 79}]
    result = apply_dwell_thresholds(dwell_data=rows)
    assert result["all_rows"][0]["alert_level"] == "Missort Alert"


def test_apply_dwell_thresholds_mixed_rows():
    rows = [
        {"brand": "A", "carrier": "C1", "site": "S1", "packages": 10},
        {"brand": "B", "carrier": "C2", "site": "S1", "packages": 30},
        {"brand": "C", "carrier": "C3", "site": "S2", "packages": 50},
        {"brand": "D", "carrier": "C4", "site": "S2", "packages": 90},
    ]
    result = apply_dwell_thresholds(dwell_data=rows)
    levels = [r["alert_level"] for r in result["all_rows"]]
    assert levels == ["", "Missort Warning", "Missort Alert", "Possible Misload Alert"]
    assert len(result["flagged_rows"]) == 3


def test_apply_dwell_thresholds_empty_input():
    result = apply_dwell_thresholds(dwell_data=[])
    assert result["all_rows"] == []
    assert result["flagged_rows"] == []
