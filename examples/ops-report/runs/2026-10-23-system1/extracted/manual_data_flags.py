def _num(text):
    s = str(text).strip().replace("%", "").replace("+", "")
    try:
        return float(s)
    except ValueError:
        raise ValueError("bad threshold: " + repr(text))


def _range(text):
    parts = str(text).split("-")
    if len(parts) != 2:
        raise ValueError("bad range: " + repr(text))
    return _num(parts[0]), _num(parts[1])


def manual_data_flags(request: dict, manual_data_gate: dict, sorter_fail_rate_threshold: str = '5%', sorter_volume_drop_threshold: str = '15%', brand_warning_range: str = '25-40', brand_alert_threshold: str = '41+', carrier_misload_threshold: str = '80+') -> dict:
    if not isinstance(manual_data_gate, dict):
        raise ValueError("manual_data_gate must be an object")
    for key in ("current_dwell", "dock_appointment_completion", "sorter_metrics"):
        if key not in manual_data_gate:
            raise ValueError("missing key: " + key)
    fail_limit = _num(sorter_fail_rate_threshold)
    drop_limit = _num(sorter_volume_drop_threshold)
    warn_lo, warn_hi = _range(brand_warning_range)
    alert_min = _num(brand_alert_threshold)
    misload_min = _num(carrier_misload_threshold)

    appt = manual_data_gate["dock_appointment_completion"]
    for key in ("green", "red", "grey", "other"):
        if key not in appt:
            raise ValueError("missing appointment column: " + key)
    completion = {
        "green": int(appt["green"]),
        "red": int(appt["red"]),
        "grey": int(appt["grey"]),
        "non_compliant": int(appt["other"]),
    }

    escalations = []
    if completion["non_compliant"] > 0:
        # skill: "and any other color (Non-Compliant). Non-Compliant must be escalated."
        escalations.append({
            "category": "Non-Compliant",
            "count": completion["non_compliant"],
            "escalate": True,
        })

    sorter = []
    for row in manual_data_gate["sorter_metrics"]:
        for key in ("site", "volume", "prior_day_volume", "fail_rate_pct"):
            if key not in row:
                raise ValueError("missing sorter column: " + key)
        volume = row["volume"]
        prior = row["prior_day_volume"]
        fail = row["fail_rate_pct"]
        change = (volume - prior) / prior * 100.0 if prior else 0.0
        # skill: "Flag if fail rate > 5% (action needed) or if volume dropped >15% day-over-day"
        fail_flag = fail > fail_limit
        drop_flag = change < -drop_limit
        flags = []
        if fail_flag:
            # skill: "Flag if fail rate > 5% (action needed)"
            flags.append("Fail rate above threshold (action needed)")
        if drop_flag:
            # skill: "or if volume dropped >15% day-over-day (divert adjustment may be needed)"
            flags.append("Volume dropped day-over-day (divert adjustment may be needed)")
        sorter.append({
            "site": row["site"],
            "volume": volume,
            "prior_day_volume": prior,
            "fail_rate_pct": fail,
            "volume_change_pct": change,
            "fail_rate_flag": fail_flag,
            "volume_drop_flag": drop_flag,
            "flags": flags,
        })

    alerts = []
    for row in manual_data_gate["current_dwell"]:
        for key in ("shipper", "carrier", "site", "packages"):
            if key not in row:
                raise ValueError("missing dwell column: " + key)
        pk = row["packages"]
        brand_alert = None
        brand_color = None
        carrier_alert = None
        carrier_color = None
        # skill: "Brand 25-40 = Missort Warning (light red), 41+ = Missort Alert (bright red)."
        if pk >= alert_min:
            brand_alert = "Missort Alert"
            brand_color = "bright red"
        elif warn_lo <= pk <= warn_hi:
            brand_alert = "Missort Warning"
            brand_color = "light red"
        # skill: "Carrier 80+ missing = Possible Misload Alert (darkest red)."
        if pk >= misload_min:
            carrier_alert = "Possible Misload Alert"
            carrier_color = "darkest red"
        alerts.append({
            "shipper": row["shipper"],
            "carrier": row["carrier"],
            "site": row["site"],
            "packages": pk,
            "brand_alert": brand_alert,
            "brand_color": brand_color,
            "carrier_alert": carrier_alert,
            "carrier_color": carrier_color,
        })

    return {
        "appointment_completion": completion,
        "appointment_escalations": escalations,
        "sorter_flags": sorter,
        "current_dwell_alerts": alerts,
    }
