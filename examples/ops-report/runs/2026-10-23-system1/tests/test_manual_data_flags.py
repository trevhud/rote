import inspect

import pytest

from extracted.manual_data_flags import manual_data_flags


def gate(appt=None, sorter=None, dwell=None):
    return {
        "dock_appointment_completion": appt or {"green": 1, "red": 0, "grey": 0, "other": 0},
        "sorter_metrics": sorter or [],
        "current_dwell": dwell or [],
    }


REQ = {"report_date": "2026-09-14"}


def dw(n):
    return [{"shipper": "S", "carrier": "C", "site": "X", "packages": n}]


def test_constants():
    p = inspect.signature(manual_data_flags).parameters
    assert p["sorter_fail_rate_threshold"].default == "5%"
    assert p["sorter_volume_drop_threshold"].default == "15%"
    assert p["brand_warning_range"].default == "25-40"
    assert p["brand_alert_threshold"].default == "41+"
    assert p["carrier_misload_threshold"].default == "80+"


def test_sample():
    g = {
        "dock_appointment_completion": {"green": 19, "red": 7, "grey": 15, "other": 11},
        "sorter_metrics": [
            {"site": "CENTRAL5", "volume": 41500, "prior_day_volume": 47300, "fail_rate_pct": 5.7},
            {"site": "MTN6", "volume": 22800, "prior_day_volume": 23100, "fail_rate_pct": 2.3},
        ],
        "current_dwell": [
            {"shipper": "Harbor Goods", "carrier": "OnTrac", "site": "CENTRAL5", "packages": 47},
            {"shipper": "Kestrel Outdoor", "carrier": "DHL", "site": "MTN6", "packages": 27},
            {"shipper": "Juniper Baby", "carrier": "LaserShip", "site": "CENTRAL5", "packages": 13},
        ],
    }
    r = manual_data_flags(REQ, g)
    assert r["appointment_completion"] == {"green": 19, "red": 7, "grey": 15, "non_compliant": 11}
    assert r["appointment_escalations"] == [{"category": "Non-Compliant", "count": 11, "escalate": True}]
    assert r["sorter_flags"][0]["fail_rate_flag"] is True
    assert r["sorter_flags"][0]["volume_drop_flag"] is False
    assert r["sorter_flags"][1]["fail_rate_flag"] is False
    a = r["current_dwell_alerts"]
    assert a[0]["brand_alert"] == "Missort Alert" and a[0]["brand_color"] == "bright red"
    assert a[1]["brand_alert"] == "Missort Warning" and a[1]["brand_color"] == "light red"
    assert a[2]["brand_alert"] is None


def test_no_escalation():
    assert manual_data_flags(REQ, gate())["appointment_escalations"] == []


def test_fail_rate_boundary():
    s = [{"site": "A", "volume": 10, "prior_day_volume": 10, "fail_rate_pct": 5.0}]
    assert manual_data_flags(REQ, gate(sorter=s))["sorter_flags"][0]["fail_rate_flag"] is False
    s[0]["fail_rate_pct"] = 5.1
    assert manual_data_flags(REQ, gate(sorter=s))["sorter_flags"][0]["fail_rate_flag"] is True


def test_volume_drop_boundary():
    s = [{"site": "A", "volume": 85, "prior_day_volume": 100, "fail_rate_pct": 1}]
    assert manual_data_flags(REQ, gate(sorter=s))["sorter_flags"][0]["volume_drop_flag"] is False
    s[0]["volume"] = 84
    f = manual_data_flags(REQ, gate(sorter=s))["sorter_flags"][0]
    assert f["volume_drop_flag"] is True
    assert len(f["flags"]) == 1


def test_brand_boundaries():
    def alert(n):
        return manual_data_flags(REQ, gate(dwell=dw(n)))["current_dwell_alerts"][0]["brand_alert"]
    assert alert(24) is None
    assert alert(25) == "Missort Warning"
    assert alert(40) == "Missort Warning"
    assert alert(41) == "Missort Alert"


def test_carrier_boundary():
    def c(n):
        return manual_data_flags(REQ, gate(dwell=dw(n)))["current_dwell_alerts"][0]
    assert c(79)["carrier_alert"] is None
    assert c(80)["carrier_alert"] == "Possible Misload Alert"
    assert c(80)["carrier_color"] == "darkest red"


def test_empty():
    r = manual_data_flags(REQ, gate())
    assert r["sorter_flags"] == [] and r["current_dwell_alerts"] == []


def test_malformed():
    with pytest.raises(ValueError):
        manual_data_flags(REQ, {"sorter_metrics": []})
    with pytest.raises(ValueError):
        manual_data_flags(REQ, gate(appt={"green": 1}))
