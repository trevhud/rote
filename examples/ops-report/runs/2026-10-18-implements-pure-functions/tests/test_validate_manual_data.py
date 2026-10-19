import pytest
from extracted.validate_manual_data import (
    validate_manual_data,
    SORTER_FAIL_RATE_THRESHOLD,
    SORTER_VOLUME_DROP_THRESHOLD,
    BRAND_WARNING_THRESHOLD,
    BRAND_ALERT_THRESHOLD,
    CARRIER_ALERT_THRESHOLD,
)


# --- constant pins ---

def test_sorter_fail_rate_threshold_constant():
    assert SORTER_FAIL_RATE_THRESHOLD == 5.0


def test_sorter_volume_drop_threshold_constant():
    assert SORTER_VOLUME_DROP_THRESHOLD == 15.0


def test_brand_warning_threshold_constant():
    assert BRAND_WARNING_THRESHOLD == 25


def test_brand_alert_threshold_constant():
    assert BRAND_ALERT_THRESHOLD == 41


def test_carrier_alert_threshold_constant():
    assert CARRIER_ALERT_THRESHOLD == 80


# --- dock appointment escalation ---

def test_non_compliant_triggers_escalation():
    # skill: "Non-Compliant must be escalated"
    payload = {
        "dock_appointments": {"green": 10, "red": 2, "grey": 1, "non_compliant": 1},
        "sorter_metrics": [],
        "current_dwell": [],
    }
    result = validate_manual_data(payload)
    assert result["dock_appointments"]["escalation_required"] is True
    assert result["dock_appointments"]["non_compliant"] == 1


def test_zero_non_compliant_no_escalation():
    payload = {
        "dock_appointments": {"green": 10, "red": 2, "grey": 1, "non_compliant": 0},
        "sorter_metrics": [],
        "current_dwell": [],
    }
    result = validate_manual_data(payload)
    assert result["dock_appointments"]["escalation_required"] is False


def test_dock_appointment_counts_preserved():
    payload = {
        "dock_appointments": {"green": 5, "red": 3, "grey": 2, "non_compliant": 0},
        "sorter_metrics": [],
        "current_dwell": [],
    }
    result = validate_manual_data(payload)
    appts = result["dock_appointments"]
    assert appts["green"] == 5
    assert appts["red"] == 3
    assert appts["grey"] == 2


# --- sorter fail rate flag ---

def test_fail_rate_above_5_flagged():
    # skill: "Flag if fail rate > 5% (action needed)"
    payload = {
        "dock_appointments": {"green": 0, "red": 0, "grey": 0, "non_compliant": 0},
        "sorter_metrics": [{"site": "EAST1", "volume_output": 1000, "fail_rate_pct": 5.1}],
        "current_dwell": [],
    }
    result = validate_manual_data(payload)
    assert result["sorter_metrics"][0]["fail_rate_flag"] is True


def test_fail_rate_at_5_not_flagged():
    # skill: "Flag if fail rate > 5%" — exactly 5.0 is NOT flagged
    payload = {
        "dock_appointments": {"green": 0, "red": 0, "grey": 0, "non_compliant": 0},
        "sorter_metrics": [{"site": "EAST1", "volume_output": 1000, "fail_rate_pct": 5.0}],
        "current_dwell": [],
    }
    result = validate_manual_data(payload)
    assert result["sorter_metrics"][0]["fail_rate_flag"] is False


def test_fail_rate_below_5_not_flagged():
    payload = {
        "dock_appointments": {"green": 0, "red": 0, "grey": 0, "non_compliant": 0},
        "sorter_metrics": [{"site": "EAST1", "volume_output": 1000, "fail_rate_pct": 4.9}],
        "current_dwell": [],
    }
    result = validate_manual_data(payload)
    assert result["sorter_metrics"][0]["fail_rate_flag"] is False


# --- sorter volume drop flag ---

def test_volume_drop_above_15pct_flagged():
    # skill: "volume dropped >15% day-over-day (divert adjustment may be needed)"
    payload = {
        "dock_appointments": {"green": 0, "red": 0, "grey": 0, "non_compliant": 0},
        "sorter_metrics": [{
            "site": "EAST1", "volume_output": 840, "fail_rate_pct": 1.0,
            "previous_volume": 1000,
        }],
        "current_dwell": [],
    }
    result = validate_manual_data(payload)
    m = result["sorter_metrics"][0]
    assert m["volume_drop_flag"] is True
    assert m["volume_drop_pct"] == 16.0


def test_volume_drop_at_15pct_not_flagged():
    payload = {
        "dock_appointments": {"green": 0, "red": 0, "grey": 0, "non_compliant": 0},
        "sorter_metrics": [{
            "site": "EAST1", "volume_output": 850, "fail_rate_pct": 1.0,
            "previous_volume": 1000,
        }],
        "current_dwell": [],
    }
    result = validate_manual_data(payload)
    assert result["sorter_metrics"][0]["volume_drop_flag"] is False


def test_no_previous_volume_no_drop_flag():
    payload = {
        "dock_appointments": {"green": 0, "red": 0, "grey": 0, "non_compliant": 0},
        "sorter_metrics": [{"site": "EAST1", "volume_output": 1000, "fail_rate_pct": 1.0}],
        "current_dwell": [],
    }
    result = validate_manual_data(payload)
    assert result["sorter_metrics"][0]["volume_drop_flag"] is False
    assert result["sorter_metrics"][0]["volume_drop_pct"] is None


# --- current dwell alert thresholds ---

def test_dwell_brand_warning_at_25():
    payload = {
        "dock_appointments": {"green": 0, "red": 0, "grey": 0, "non_compliant": 0},
        "sorter_metrics": [],
        "current_dwell": [{"brand": "BrandX", "carrier": "FedEx", "site": "EAST1", "packages": 25}],
    }
    result = validate_manual_data(payload)
    assert result["current_dwell"][0]["alert_level"] == "Missort Warning"


def test_dwell_brand_alert_at_41():
    payload = {
        "dock_appointments": {"green": 0, "red": 0, "grey": 0, "non_compliant": 0},
        "sorter_metrics": [],
        "current_dwell": [{"brand": "BrandX", "carrier": "FedEx", "site": "EAST1", "packages": 41}],
    }
    result = validate_manual_data(payload)
    assert result["current_dwell"][0]["alert_level"] == "Missort Alert"


def test_dwell_carrier_alert_at_80():
    payload = {
        "dock_appointments": {"green": 0, "red": 0, "grey": 0, "non_compliant": 0},
        "sorter_metrics": [],
        "current_dwell": [{
            "brand": "BrandX", "carrier": "FedEx", "site": "EAST1",
            "packages": 10, "carrier_packages": 80,
        }],
    }
    result = validate_manual_data(payload)
    assert result["current_dwell"][0]["alert_level"] == "Possible Misload Alert"


def test_dwell_below_25_no_alert():
    payload = {
        "dock_appointments": {"green": 0, "red": 0, "grey": 0, "non_compliant": 0},
        "sorter_metrics": [],
        "current_dwell": [{"brand": "BrandX", "carrier": "FedEx", "site": "EAST1", "packages": 24}],
    }
    result = validate_manual_data(payload)
    assert result["current_dwell"][0]["alert_level"] is None


# --- constants are passable as keyword arguments (IR contract) ---

def test_custom_fail_rate_threshold_passed_as_kwarg():
    payload = {
        "dock_appointments": {"green": 0, "red": 0, "grey": 0, "non_compliant": 0},
        "sorter_metrics": [{"site": "EAST1", "volume_output": 1000, "fail_rate_pct": 3.0}],
        "current_dwell": [],
    }
    result_default = validate_manual_data(payload)
    assert result_default["sorter_metrics"][0]["fail_rate_flag"] is False

    result_lower = validate_manual_data(payload, sorter_fail_rate_threshold=2.0)
    assert result_lower["sorter_metrics"][0]["fail_rate_flag"] is True


def test_custom_brand_alert_threshold_passed_as_kwarg():
    payload = {
        "dock_appointments": {"green": 0, "red": 0, "grey": 0, "non_compliant": 0},
        "sorter_metrics": [],
        "current_dwell": [{"brand": "X", "carrier": "Y", "site": "Z", "packages": 35}],
    }
    result_default = validate_manual_data(payload)
    assert result_default["current_dwell"][0]["alert_level"] == "Missort Warning"

    result_lower = validate_manual_data(payload, brand_alert_threshold=30)
    assert result_lower["current_dwell"][0]["alert_level"] == "Missort Alert"


# --- missing required keys ---

def test_missing_dock_appointments_raises():
    with pytest.raises(ValueError, match="dock_appointments"):
        validate_manual_data({"sorter_metrics": [], "current_dwell": []})


def test_missing_sorter_metrics_raises():
    with pytest.raises(ValueError, match="sorter_metrics"):
        validate_manual_data({
            "dock_appointments": {"green": 0, "red": 0, "grey": 0, "non_compliant": 0},
            "current_dwell": [],
        })
