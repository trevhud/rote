"""
Deterministic threshold enforcement for manually-supplied ops-report metrics.

Applies the numeric rules from the skill to duty-manager-provided data:
  - Sorter fail rate > 5%  → action needed
  - Sorter volume drop > 15% day-over-day → divert adjustment may be needed
  - Dock Non-Compliant appointments  → must be escalated
  - Dwell brand 25-40 → Missort Warning (light red)
  - Dwell brand 41+   → Missort Alert (bright red)
  - Dwell carrier 80+ → Possible Misload Alert (darkest red)

No third-party imports — stdlib only.
"""

from typing import Any

# ── constants lifted from skill prose ────────────────────────────────────────

SORTER_FAIL_RATE_THRESHOLD_PCT: float = 5.0     # fail rate > 5% → action needed
SORTER_VOLUME_DROP_THRESHOLD_PCT: float = 15.0  # volume drop > 15% → divert adjustment
DWELL_WARNING_PACKAGES: int = 25                # brand 25-40 → Missort Warning
DWELL_ALERT_PACKAGES: int = 41                  # brand 41+   → Missort Alert
DWELL_MISLOAD_CARRIER: int = 80                 # carrier 80+ → Possible Misload Alert


def apply_manual_thresholds(
    dock_appointment_completion: dict[str, Any],
    sorter_metrics: list[dict[str, Any]],
    current_dwell: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Apply all skill-defined thresholds to manually-supplied duty-manager data.

    dock_appointment_completion expected shape:
        {
            "green": int,   # Complete
            "red": int,     # No Call/No Show
            "grey": int,    # Canceled
            "other": int,   # Non-Compliant (any other colour)
        }

    sorter_metrics expected shape (list of per-site dicts):
        [{
            "site": str,
            "volume_output": int,
            "fail_rate_pct": float,
            "prev_day_volume": int | None,  # None if unavailable
        }, ...]

    current_dwell expected shape (list of dwell entries):
        [{
            "shipper": str,
            "carrier": str,
            "site": str,
            "packages": int,
        }, ...]

    Returns:
        {
            "appointment_summary": {
                "green": int, "red": int, "grey": int, "other": int,
                "total": int,
                "non_compliant_count": int,
                "non_compliant_escalated": bool,
            },
            "sorter_sites": [{site, volume_output, fail_rate_pct,
                               prev_day_volume, fail_rate_flagged,
                               volume_drop_flagged, volume_drop_pct}],
            "flagged_sorter_sites": [str, ...],
            "dwell_alerts": [{shipper, carrier, site, packages, alert_level}, ...],
            "dwell_warning_entries": [...],
            "dwell_alert_entries": [...],
            "misload_alert_entries": [...],
            "non_compliant_appointments": int,
        }
    """
    # ── dock appointment completion ───────────────────────────────────────────
    green = int(dock_appointment_completion.get("green", 0))
    red = int(dock_appointment_completion.get("red", 0))
    grey = int(dock_appointment_completion.get("grey", 0))
    other = int(dock_appointment_completion.get("other", 0))
    total = green + red + grey + other
    non_compliant = other  # Non-Compliant = any color that isn't Green/Red/Grey

    appointment_summary = {
        "green": green,
        "red": red,
        "grey": grey,
        "other": other,
        "total": total,
        "non_compliant_count": non_compliant,
        "non_compliant_escalated": non_compliant > 0,  # MUST be escalated per skill
    }

    # ── sorter metrics ────────────────────────────────────────────────────────
    sorter_sites = []
    flagged_sorter_sites = []

    for entry in sorter_metrics:
        site = str(entry.get("site", ""))
        volume = int(entry.get("volume_output", 0))
        fail_rate = float(entry.get("fail_rate_pct", 0.0))
        prev_volume_raw = entry.get("prev_day_volume")
        prev_volume: int | None = int(prev_volume_raw) if prev_volume_raw is not None else None

        fail_rate_flagged = fail_rate > SORTER_FAIL_RATE_THRESHOLD_PCT

        volume_drop_pct: float | None = None
        volume_drop_flagged = False
        if prev_volume is not None and prev_volume > 0:
            volume_drop_pct = round((prev_volume - volume) / prev_volume * 100, 1)
            volume_drop_flagged = volume_drop_pct > SORTER_VOLUME_DROP_THRESHOLD_PCT

        site_result = {
            "site": site,
            "volume_output": volume,
            "fail_rate_pct": fail_rate,
            "prev_day_volume": prev_volume,
            "fail_rate_flagged": fail_rate_flagged,
            "volume_drop_flagged": volume_drop_flagged,
            "volume_drop_pct": volume_drop_pct,
        }
        sorter_sites.append(site_result)

        if fail_rate_flagged or volume_drop_flagged:
            flagged_sorter_sites.append(site)

    # ── current dwell alerts ──────────────────────────────────────────────────
    dwell_alerts = []
    dwell_warning_entries = []
    dwell_alert_entries = []
    misload_alert_entries = []

    for entry in current_dwell:
        shipper = str(entry.get("shipper", ""))
        carrier = str(entry.get("carrier", ""))
        site = str(entry.get("site", ""))
        packages = int(entry.get("packages", 0))

        # Brand thresholds
        if packages >= DWELL_ALERT_PACKAGES:
            alert_level = "missort_alert"
        elif packages >= DWELL_WARNING_PACKAGES:
            alert_level = "missort_warning"
        else:
            alert_level = "none"

        misload = packages >= DWELL_MISLOAD_CARRIER

        alert_entry = {
            "shipper": shipper,
            "carrier": carrier,
            "site": site,
            "packages": packages,
            "alert_level": alert_level,
            "misload_alert": misload,
        }
        dwell_alerts.append(alert_entry)

        if alert_level == "missort_alert":
            dwell_alert_entries.append(alert_entry)
        elif alert_level == "missort_warning":
            dwell_warning_entries.append(alert_entry)
        if misload:
            misload_alert_entries.append(alert_entry)

    return {
        "appointment_summary": appointment_summary,
        "sorter_sites": sorter_sites,
        "flagged_sorter_sites": flagged_sorter_sites,
        "dwell_alerts": dwell_alerts,
        "dwell_warning_entries": dwell_warning_entries,
        "dwell_alert_entries": dwell_alert_entries,
        "misload_alert_entries": misload_alert_entries,
        "non_compliant_appointments": non_compliant,
    }
