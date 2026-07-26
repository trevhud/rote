"""Apply threshold rules to duty-manager-provided manual metrics.

Pure functions. All three process sub-fields of the duty_manager_data_gate
HITL signal payload. Thresholds are extracted from SKILL.md prose constants
in the 'Manual data to request from user' section.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

# Sorter thresholds — SKILL.md: 'Flag if fail rate > 5% (action needed)'
SORTER_FAIL_RATE_THRESHOLD_PCT: float = 5.0
# SKILL.md: 'volume dropped >15% day-over-day (divert adjustment may be needed)'
SORTER_VOLUME_DROP_THRESHOLD_PCT: float = 15.0

# Dwell thresholds (consistent with dwell_ticket_parser.py)
# SKILL.md: 'Brand 25-40 = Missort Warning, 41+ = Missort Alert, Carrier 80+ = Possible Misload'
PKG_WARNING_THRESHOLD: int = 25
PKG_ALERT_THRESHOLD: int = 41
CARRIER_ALERT_THRESHOLD: int = 80


# ── Dock Appointment ──────────────────────────────────────────────────────────

@dataclass
class DockAppointmentCounts:
    green: int          # Complete
    red: int            # No Call / No Show
    grey: int           # Canceled
    non_compliant: int  # any other color — triggers MANDATORY escalation


@dataclass
class DockAppointmentAnalysis:
    counts: DockAppointmentCounts
    total: int
    completion_rate_pct: float
    non_compliant_count: int
    escalation_required: bool  # MANDATORY: True when non_compliant > 0


def apply_dock_appointment_rules(
    dock_appointments: DockAppointmentCounts,
) -> DockAppointmentAnalysis:
    """Apply dock appointment rules.

    MANDATORY rule (SKILL.md): 'Non-Compliant must be escalated.'
    This node is marked mandatory=true in the IR — it cannot be skipped or
    made conditional.

    Green = Complete; Red = No Call/No Show; Grey = Canceled;
    non_compliant = any other color → escalation_required = True.
    """
    total = (
        dock_appointments.green
        + dock_appointments.red
        + dock_appointments.grey
        + dock_appointments.non_compliant
    )
    completion_rate = (dock_appointments.green / total * 100) if total > 0 else 0.0

    return DockAppointmentAnalysis(
        counts=dock_appointments,
        total=total,
        completion_rate_pct=round(completion_rate, 1),
        non_compliant_count=dock_appointments.non_compliant,
        escalation_required=dock_appointments.non_compliant > 0,
    )


# ── Sorter Metrics ────────────────────────────────────────────────────────────

@dataclass
class SorterSiteData:
    site: str
    volume_output: int
    fail_rate_pct: float
    prior_volume: int | None = None  # previous day's volume for DoD comparison


@dataclass
class SorterSiteAnalysis:
    site: str
    volume_output: int
    fail_rate_pct: float
    fail_rate_alert: bool     # fail_rate_pct > SORTER_FAIL_RATE_THRESHOLD_PCT
    volume_drop_alert: bool   # DoD drop > SORTER_VOLUME_DROP_THRESHOLD_PCT
    volume_drop_pct: float | None = None  # None when prior_volume unavailable


@dataclass
class SorterAnalysis:
    sites: list[SorterSiteAnalysis] = field(default_factory=list)
    sites_requiring_action: list[str] = field(default_factory=list)


def apply_sorter_rules(sorter_metrics: list[SorterSiteData]) -> SorterAnalysis:
    """Apply sorter metric thresholds.

    Flags:
      fail_rate_pct > 5%  → action needed   (SORTER_FAIL_RATE_THRESHOLD_PCT)
      volume drop > 15% DoD → divert adjustment may be needed
                            (SORTER_VOLUME_DROP_THRESHOLD_PCT)

    Day-over-day comparison requires prior_volume on each SorterSiteData.
    When prior_volume is None the DoD check is skipped (see open question in
    compile-report.md about tracking prior-day volumes).
    """
    site_analyses: list[SorterSiteAnalysis] = []
    sites_requiring_action: list[str] = []

    for site in sorter_metrics:
        fail_rate_alert = site.fail_rate_pct > SORTER_FAIL_RATE_THRESHOLD_PCT

        volume_drop_pct: float | None = None
        volume_drop_alert = False
        if site.prior_volume is not None and site.prior_volume > 0:
            volume_drop_pct = (
                (site.prior_volume - site.volume_output) / site.prior_volume * 100
            )
            volume_drop_alert = volume_drop_pct > SORTER_VOLUME_DROP_THRESHOLD_PCT

        analysis = SorterSiteAnalysis(
            site=site.site,
            volume_output=site.volume_output,
            fail_rate_pct=site.fail_rate_pct,
            fail_rate_alert=fail_rate_alert,
            volume_drop_alert=volume_drop_alert,
            volume_drop_pct=volume_drop_pct,
        )
        site_analyses.append(analysis)
        if fail_rate_alert or volume_drop_alert:
            sites_requiring_action.append(site.site)

    return SorterAnalysis(sites=site_analyses, sites_requiring_action=sites_requiring_action)


# ── Dwell Thresholds (BI Dashboard) ──────────────────────────────────────────

class DwellSeverity(str, Enum):
    OK = "ok"
    PKG_WARNING = "pkg_warning"     # Brand 25-40 packages (light red)
    PKG_ALERT = "pkg_alert"         # Brand 41+ packages   (bright red)
    CARRIER_ALERT = "carrier_alert"  # Carrier 80+ missing  (darkest red)


@dataclass
class DwellRecord:
    shipper: str   # brand / shipper name
    carrier: str
    site: str
    orders: int
    packages: int
    brand_severity: DwellSeverity
    carrier_severity: DwellSeverity


@dataclass
class DwellAnalysis:
    records: list[DwellRecord] = field(default_factory=list)
    brand_warnings: list[DwellRecord] = field(default_factory=list)   # 25-40 packages
    brand_alerts: list[DwellRecord] = field(default_factory=list)     # 41+ packages
    carrier_alerts: list[DwellRecord] = field(default_factory=list)   # 80+ packages


@dataclass
class DwellInputRecord:
    """Raw record from the duty manager's BI dashboard export."""
    shipper: str
    carrier: str
    site: str
    orders: int
    packages: int


def apply_dwell_thresholds(dwell_data: list[DwellInputRecord]) -> DwellAnalysis:
    """Apply color-coded dwell thresholds to BI dashboard data.

    Brand (shipper) thresholds (SKILL.md manual-data section):
      25-40 packages → Missort Warning  (light red)
      41+   packages → Missort Alert    (bright red)

    Carrier threshold:
      80+ packages   → Possible Misload Alert (darkest red)

    Both brand and carrier severity are classified independently per record
    because a single record can trigger both brand and carrier alerts.
    """
    classified: list[DwellRecord] = []
    brand_warnings: list[DwellRecord] = []
    brand_alerts: list[DwellRecord] = []
    carrier_alerts: list[DwellRecord] = []

    for raw in dwell_data:
        brand_sev = _brand_severity(raw.packages)
        carrier_sev = _carrier_severity(raw.packages)
        record = DwellRecord(
            shipper=raw.shipper,
            carrier=raw.carrier,
            site=raw.site,
            orders=raw.orders,
            packages=raw.packages,
            brand_severity=brand_sev,
            carrier_severity=carrier_sev,
        )
        classified.append(record)
        if brand_sev == DwellSeverity.PKG_ALERT:
            brand_alerts.append(record)
        elif brand_sev == DwellSeverity.PKG_WARNING:
            brand_warnings.append(record)
        if carrier_sev == DwellSeverity.CARRIER_ALERT:
            carrier_alerts.append(record)

    return DwellAnalysis(
        records=classified,
        brand_warnings=brand_warnings,
        brand_alerts=brand_alerts,
        carrier_alerts=carrier_alerts,
    )


def _brand_severity(packages: int) -> DwellSeverity:
    if packages >= PKG_ALERT_THRESHOLD:
        return DwellSeverity.PKG_ALERT
    if packages >= PKG_WARNING_THRESHOLD:
        return DwellSeverity.PKG_WARNING
    return DwellSeverity.OK


def _carrier_severity(packages: int) -> DwellSeverity:
    if packages >= CARRIER_ALERT_THRESHOLD:
        return DwellSeverity.CARRIER_ALERT
    return DwellSeverity.OK
