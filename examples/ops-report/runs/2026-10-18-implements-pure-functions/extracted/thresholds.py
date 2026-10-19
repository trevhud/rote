"""
Pure-function threshold classifiers for manually-supplied ops-report data.

These three functions process the duty-manager's HITL signal payload after
the request_manual_data_gate fires. Stdlib only.
"""

# ---------------------------------------------------------------------------
# Constants (lifted from skill prose — Pattern 2)
# ---------------------------------------------------------------------------

# skill: "Flag if fail rate > 5% (action needed)"
SORTER_FAIL_RATE_THRESHOLD: float = 5.0

# skill: "volume dropped >15% day-over-day (divert adjustment may be needed)"
SORTER_VOLUME_DROP_THRESHOLD: float = 15.0

# skill: "Brand 25-40 = Missort Warning (light red), 41+ = Missort Alert (bright red)"
DWELL_BRAND_WARNING_THRESHOLD: int = 25
DWELL_BRAND_ALERT_THRESHOLD: int = 41

# skill: "Carrier 80+ missing = Possible Misload Alert (darkest red)"
DWELL_CARRIER_ALERT_THRESHOLD: int = 80


# ---------------------------------------------------------------------------
# apply_appointment_flags
# ---------------------------------------------------------------------------

def apply_appointment_flags(*, appointment_completion: dict) -> dict:
    """
    Apply compliance flags to duty-manager-supplied dock appointment counts.

    Input shape: {
        "green":  int,   # Complete
        "red":    int,   # No Call/No Show
        "grey":   int,   # Canceled
        "other":  int    # Non-Compliant (any other color)
    }

    Returns: {
        "green":              int,
        "red":                int,
        "grey":               int,
        "other":              int,
        "non_compliant_count": int,
        "escalation_required": bool
    }
    """
    green = int(appointment_completion.get("green", 0))
    red = int(appointment_completion.get("red", 0))
    grey = int(appointment_completion.get("grey", 0))
    # skill: "any other color (Non-Compliant). Non-Compliant must be escalated."
    other = int(appointment_completion.get("other", 0))
    non_compliant_count = other

    return {
        "green": green,
        "red": red,
        "grey": grey,
        "other": other,
        "non_compliant_count": non_compliant_count,
        # skill: "Non-Compliant must be escalated"
        "escalation_required": non_compliant_count > 0,
    }


# ---------------------------------------------------------------------------
# apply_sorter_flags
# ---------------------------------------------------------------------------

def apply_sorter_flags(
    *,
    sorter_metrics: list,
    fail_rate_threshold: float = SORTER_FAIL_RATE_THRESHOLD,
    volume_drop_threshold: float = SORTER_VOLUME_DROP_THRESHOLD,
) -> dict:
    """
    Apply action flags to duty-manager-supplied sorter metrics.

    Input shape: list of {
        "site":          str,
        "volume":        int,
        "fail_rate_pct": float,
        "prev_volume":   int | None   # previous day's volume, for WoW calc
    }

    Returns: {
        "sites": list[dict],           # input rows enriched with "flags" list
        "action_required_sites": list[str],
        "divert_adjustment_sites": list[str]
    }

    Each enriched site dict gains a "flags" list:
      - "ACTION NEEDED" when fail_rate > fail_rate_threshold
      - "DIVERT ADJUSTMENT MAY BE NEEDED" when prev_volume known and
         volume dropped > volume_drop_threshold % day-over-day
    """
    enriched = []
    action_required_sites: list = []
    divert_adjustment_sites: list = []

    for row in sorter_metrics:
        site = row.get("site", "")
        volume = int(row.get("volume", 0))
        fail_rate_pct = float(row.get("fail_rate_pct", 0.0))
        prev_volume = row.get("prev_volume")

        flags: list = []

        # skill: "Flag if fail rate > 5% (action needed)"
        if fail_rate_pct > fail_rate_threshold:
            flags.append("ACTION NEEDED")
            action_required_sites.append(site)

        # skill: "volume dropped >15% day-over-day (divert adjustment may be needed)"
        if prev_volume is not None and prev_volume > 0:
            drop_pct = (prev_volume - volume) / prev_volume * 100.0
            if drop_pct > volume_drop_threshold:
                flags.append("DIVERT ADJUSTMENT MAY BE NEEDED")
                divert_adjustment_sites.append(site)

        enriched.append({
            "site": site,
            "volume": volume,
            "fail_rate_pct": fail_rate_pct,
            "prev_volume": prev_volume,
            "flags": flags,
        })

    return {
        "sites": enriched,
        "action_required_sites": action_required_sites,
        "divert_adjustment_sites": divert_adjustment_sites,
    }


# ---------------------------------------------------------------------------
# apply_dwell_thresholds
# ---------------------------------------------------------------------------

def _classify_dwell_row(
    packages: int,
    *,
    brand_warning_threshold: int,
    brand_alert_threshold: int,
    carrier_alert_threshold: int,
) -> str:
    """
    Return alert level for a single BI-dashboard dwell row.

    # rote: assumption: "Brand 25-40 = Missort Warning, 41+ = Missort Alert,
    # Carrier 80+ missing = Possible Misload Alert" is applied as a single
    # tiered threshold on the packages column (highest threshold wins).
    # The skill names Brand/Carrier as separate dimensions but does not provide
    # separate package counts per dimension; a single per-row packages column
    # is used. See compile-report.md Open questions.
    """
    # skill: "Carrier 80+ missing = Possible Misload Alert (darkest red)"
    if packages >= carrier_alert_threshold:
        return "Possible Misload Alert"
    # skill: "41+ = Missort Alert (bright red)"
    if packages >= brand_alert_threshold:
        return "Missort Alert"
    # skill: "Brand 25-40 = Missort Warning (light red)"
    if packages >= brand_warning_threshold:
        return "Missort Warning"
    return ""


def apply_dwell_thresholds(
    *,
    dwell_data: list,
    brand_warning_threshold: int = DWELL_BRAND_WARNING_THRESHOLD,
    brand_alert_threshold: int = DWELL_BRAND_ALERT_THRESHOLD,
    carrier_alert_threshold: int = DWELL_CARRIER_ALERT_THRESHOLD,
) -> dict:
    """
    Apply color-coded dwell alert thresholds to BI-dashboard dwell data.

    Input shape: list of {
        "brand":    str,
        "carrier":  str,
        "site":     str,
        "packages": int
    }

    Returns: {
        "all_rows":    list[dict],   # every row enriched with alert_level
        "flagged_rows": list[dict]   # rows where alert_level != ''
    }
    """
    all_rows: list = []
    flagged_rows: list = []

    for row in dwell_data:
        brand = row.get("brand", "")
        carrier = row.get("carrier", "")
        site = row.get("site", "")
        packages = int(row.get("packages", 0))

        alert_level = _classify_dwell_row(
            packages,
            brand_warning_threshold=brand_warning_threshold,
            brand_alert_threshold=brand_alert_threshold,
            carrier_alert_threshold=carrier_alert_threshold,
        )

        enriched = {
            "brand": brand,
            "carrier": carrier,
            "site": site,
            "packages": packages,
            "alert_level": alert_level,
        }
        all_rows.append(enriched)
        if alert_level:
            flagged_rows.append(enriched)

    return {
        "all_rows": all_rows,
        "flagged_rows": flagged_rows,
    }
