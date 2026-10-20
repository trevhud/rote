"""
Validate and structure the duty manager's manually-provided operations data.

The duty manager supplies three sections via the HITL gate signal payload:
  1. dock_appointments: Green/Red/Grey/Non-Compliant counts
  2. sorter_metrics: per-site volume output and fail rate %
  3. current_dwell: BI-dashboard dwell snapshot (shipper, carrier, site, packages)
"""


SORTER_FAIL_RATE_THRESHOLD = 5.0
SORTER_VOLUME_DROP_THRESHOLD = 15.0
BRAND_WARNING_THRESHOLD = 25
BRAND_ALERT_THRESHOLD = 41
CARRIER_ALERT_THRESHOLD = 80


def _classify_dwell_alert(
    packages: int,
    carrier_packages: int,
    *,
    brand_warning_threshold: int,
    brand_alert_threshold: int,
    carrier_alert_threshold: int,
) -> str | None:
    # skill: "Brand 25-40 = Missort Warning (light red), 41+ = Missort Alert (bright red). Carrier 80+ missing = Possible Misload Alert (darkest red)"
    # rote: assumption: Possible Misload Alert (darkest red / most severe) checked first
    # when both carrier >= 80 and packages >= 41 conditions hold simultaneously.
    if carrier_packages >= carrier_alert_threshold:
        return "Possible Misload Alert"
    if packages >= brand_alert_threshold:
        return "Missort Alert"
    if packages >= brand_warning_threshold:
        return "Missort Warning"
    return None


def validate_manual_data(
    duty_manager_input: dict,
    *,
    sorter_fail_rate_threshold: float = SORTER_FAIL_RATE_THRESHOLD,
    sorter_volume_drop_threshold: float = SORTER_VOLUME_DROP_THRESHOLD,
    brand_warning_threshold: int = BRAND_WARNING_THRESHOLD,
    brand_alert_threshold: int = BRAND_ALERT_THRESHOLD,
    carrier_alert_threshold: int = CARRIER_ALERT_THRESHOLD,
) -> dict:
    """Validate and structure the duty manager's signal payload.

    Args:
        duty_manager_input: Dict from the HITL gate signal with keys:
            dock_appointments: {green, red, grey, non_compliant} (int values)
            sorter_metrics: list of {site, volume_output, fail_rate_pct,
                previous_volume (optional)} dicts
            current_dwell: list of {brand, carrier, site, packages,
                carrier_packages (optional)} dicts
        sorter_fail_rate_threshold: Flag fail rates strictly above this value.
        sorter_volume_drop_threshold: Flag DoD volume drops strictly above this %.
        brand_warning_threshold: Brand packages >= this → Missort Warning.
        brand_alert_threshold: Brand packages >= this → Missort Alert.
        carrier_alert_threshold: Carrier packages >= this → Possible Misload Alert.

    Returns:
        Structured dict with validated fields and computed alert flags:
        {
            "dock_appointments": {green, red, grey, non_compliant,
                                  escalation_required: bool},
            "sorter_metrics": list of per-site dicts with flag fields,
            "current_dwell": list of dwell items with alert_level field,
        }

    Raises:
        ValueError: If required top-level keys are missing.
    """
    required_keys = ["dock_appointments", "sorter_metrics", "current_dwell"]
    missing = [k for k in required_keys if k not in duty_manager_input]
    if missing:
        raise ValueError(f"duty_manager_input missing required keys: {missing}")

    appts = duty_manager_input["dock_appointments"]
    non_compliant = int(appts.get("non_compliant", 0))
    # skill: "Non-Compliant must be escalated"
    escalation_required = non_compliant > 0

    dock_appointments = {
        "green": int(appts.get("green", 0)),
        "red": int(appts.get("red", 0)),
        "grey": int(appts.get("grey", 0)),
        "non_compliant": non_compliant,
        "escalation_required": escalation_required,
    }

    sorter_metrics = []
    for entry in duty_manager_input.get("sorter_metrics") or []:
        volume = int(entry.get("volume_output", 0))
        fail_rate = float(entry.get("fail_rate_pct", 0.0))
        prev_volume = entry.get("previous_volume")

        volume_drop_pct = None
        volume_drop_flag = False
        if prev_volume is not None:
            prev_v = int(prev_volume)
            if prev_v > 0:
                drop = (prev_v - volume) / prev_v * 100
                volume_drop_pct = round(drop, 2)
                # skill: "volume dropped >15% day-over-day (divert adjustment may be needed)"
                volume_drop_flag = drop > sorter_volume_drop_threshold

        # skill: "Flag if fail rate > 5% (action needed)"
        fail_rate_flag = fail_rate > sorter_fail_rate_threshold

        sorter_metrics.append({
            "site": entry.get("site", ""),
            "volume_output": volume,
            "fail_rate_pct": fail_rate,
            "previous_volume": int(prev_volume) if prev_volume is not None else None,
            "volume_drop_pct": volume_drop_pct,
            "fail_rate_flag": fail_rate_flag,
            "volume_drop_flag": volume_drop_flag,
        })

    current_dwell = []
    for item in duty_manager_input.get("current_dwell") or []:
        packages = int(item.get("packages", 0))
        carrier_packages = int(item.get("carrier_packages", 0))
        alert_level = _classify_dwell_alert(
            packages, carrier_packages,
            brand_warning_threshold=brand_warning_threshold,
            brand_alert_threshold=brand_alert_threshold,
            carrier_alert_threshold=carrier_alert_threshold,
        )
        current_dwell.append({
            "brand": item.get("brand", ""),
            "carrier": item.get("carrier", ""),
            "site": item.get("site", ""),
            "packages": packages,
            "carrier_packages": carrier_packages,
            "alert_level": alert_level,
        })

    return {
        "dock_appointments": dock_appointments,
        "sorter_metrics": sorter_metrics,
        "current_dwell": current_dwell,
    }
