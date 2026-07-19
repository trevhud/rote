"""Validate duty-manager-provided manual data and apply alert thresholds.

This module is MANDATORY in the pipeline. It MUST run before report assembly
because it enforces the Non-Compliant escalation check (explicitly required by
the source skill) and applies color-coded threshold rules to BI dashboard data
that cannot be checked without the manager's numbers.

All thresholds are module-level constants lifted from the source skill so they
cannot drift through prompt edits.
"""
from __future__ import annotations

# ── Thresholds lifted from SKILL.md ──────────────────────────────────────────

SORTER_FAIL_RATE_THRESHOLD = 0.05         # > 5%  → action needed
SORTER_VOLUME_DROP_THRESHOLD = 0.15       # > 15% DoD drop → divert adjustment

DWELL_MISSORT_WARNING_PACKAGES = 25       # brand packages: light red
DWELL_MISSORT_ALERT_PACKAGES = 41         # brand packages: bright red
DWELL_MISLOAD_ALERT_CARRIER_PACKAGES = 80 # carrier packages missing: darkest red


# ── Input / output shape docs ─────────────────────────────────────────────────
#
# ManagerInputData (from the HITL gate signal payload):
#   appointment_completion: {
#     green_count: int,        # Complete
#     red_count: int,          # No Call / No Show
#     grey_count: int,         # Canceled
#     non_compliant_count: int # Any other color — MUST escalate
#   }
#   sorter_metrics: list of {
#     site: str, volume_output: int, fail_rate_pct: float,
#     prior_day_volume: int | None
#   }
#   current_dwell: list of {
#     shipper_brand: str, carrier: str, site: str,
#     order_count: int, package_count: int
#   }
#
# ValidatedManagerData (output):
#   appointment_completion: same as input
#   non_compliant_escalation_required: bool
#   sorter_metrics: same list with added fields:
#     action_needed: bool          (fail_rate_pct > SORTER_FAIL_RATE_THRESHOLD)
#     divert_adjustment_needed: bool (DoD volume drop > SORTER_VOLUME_DROP_THRESHOLD)
#   current_dwell: same list with added field:
#     alert_level: str  # "missort_warning" | "missort_alert" | "misload_alert" | "normal"


def validate_manager_data(data: dict) -> dict:
    """Apply alert thresholds to duty-manager-provided data. MANDATORY.

    Checks performed:
    - Non-Compliant count > 0 → non_compliant_escalation_required = True
      (source skill: "Non-Compliant must be escalated")
    - Sorter fail_rate_pct > SORTER_FAIL_RATE_THRESHOLD → action_needed = True
    - Sorter prior_day_volume drop > SORTER_VOLUME_DROP_THRESHOLD → divert_adjustment_needed = True
    - Dwell package_count >= DWELL_MISSORT_WARNING_PACKAGES → alert_level = "missort_warning"
    - Dwell package_count >= DWELL_MISSORT_ALERT_PACKAGES → alert_level = "missort_alert"
    - Dwell carrier package_count >= DWELL_MISLOAD_ALERT_CARRIER_PACKAGES → alert_level = "misload_alert"

    Returns: ValidatedManagerData dict (see shape above).

    Raises:
        ValueError: if required fields are missing from data (input validation).
    """
    raise NotImplementedError(
        "Apply threshold rules to the ManagerInputData dict. "
        "Set non_compliant_escalation_required=True if "
        "data['appointment_completion']['non_compliant_count'] > 0. "
        "For each sorter site, set action_needed=True if "
        f"fail_rate_pct > {SORTER_FAIL_RATE_THRESHOLD} (SORTER_FAIL_RATE_THRESHOLD). "
        "Compute DoD drop as (prior - current) / prior if prior_day_volume is set; "
        f"set divert_adjustment_needed=True if drop > {SORTER_VOLUME_DROP_THRESHOLD}. "
        "For each dwell entry, apply: "
        f">= {DWELL_MISSORT_ALERT_PACKAGES} → 'missort_alert', "
        f">= {DWELL_MISSORT_WARNING_PACKAGES} → 'missort_warning', "
        f"(carrier) >= {DWELL_MISLOAD_ALERT_CARRIER_PACKAGES} → 'misload_alert', "
        "else 'normal'. Return ValidatedManagerData dict."
    )
