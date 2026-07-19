"""Assemble the daily executive operations report from parsed data.

Pure function. Extracted from SKILL.md report-format section:
  KPI header + Section 1 (Dock Compliance) + Section 2 (Missort & Loss)
  + Section 3 (Dwell) + artifact reminder.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dock_email_parser import DockEmailSummary
    from dock_pending_parser import DockPendingSummary
    from dwell_ticket_parser import DwellTicketSummary
    from manual_data_rules import DockAppointmentAnalysis, DwellAnalysis, SorterAnalysis
    from shipment_parser import ShipmentContainersSummary

# Mirror thresholds locally so this module is self-contained for tests
_COMPLETION_THRESHOLD_PCT = 75
_SORTER_FAIL_RATE_THRESHOLD_PCT = 5
_SORTER_VOLUME_DROP_THRESHOLD_PCT = 15
_PKG_WARNING_THRESHOLD = 25
_PKG_ALERT_THRESHOLD = 41
_CARRIER_ALERT_THRESHOLD = 80


def assemble_report(
    shipment_summary: "ShipmentContainersSummary",
    dock_pending_summary: "DockPendingSummary",
    dock_email_summary: "DockEmailSummary",
    dwell_ticket_summary: "DwellTicketSummary",
    dock_appointment_analysis: "DockAppointmentAnalysis",
    sorter_analysis: "SorterAnalysis",
    dwell_analysis: "DwellAnalysis",
    report_date: str = "",
) -> str:
    """Render the daily executive operations report as Markdown.

    Sections:
      KPI Header
      Section 1 — Dock Compliance
        Appointment tiles + Gmail summary + Pending Log table
      Section 2 — Missort & Loss
        Sorter metrics + container flow table by site with completion %
      Section 3 — Dwell
        BI-dashboard dwell alerts (color-coded) + Open Dwell Impact Log tickets
      Artifact reminder
    """
    lines: list[str] = []
    date_label = f" — {report_date}" if report_date else ""

    # ── KPI Header ────────────────────────────────────────────────────────────
    lines.append(f"# Daily Operations Report{date_label}\n\n")
    lines.append("## KPI Summary\n\n")
    lines.append(
        "| KPI | Value |\n"
        "|-----|-------|\n"
        f"| Active Sites | {shipment_summary.active_site_count} |\n"
        f"| Dock Pending (Open) | {dock_pending_summary.total_open} |\n"
        f"| Avg Completion % | {shipment_summary.avg_completion_pct:.1f}% |\n"
        f"| Open Dwell Tickets | {dwell_ticket_summary.total_open} |\n"
        "\n"
    )

    # ── Section 1 — Dock Compliance ───────────────────────────────────────────
    lines.append("## Section 1 — Dock Compliance\n\n")

    # Appointment tiles
    c = dock_appointment_analysis.counts
    lines.append("### Dock Appointment Completion (Prior Day)\n\n")
    lines.append(
        "| 🟢 Complete | 🔴 No Call/No Show | ⬜ Canceled | ⚠️ Non-Compliant |\n"
        "|------------|-------------------|------------|------------------|\n"
        f"| {c.green} | {c.red} | {c.grey} | {c.non_compliant} |\n\n"
        f"**Completion Rate:** {dock_appointment_analysis.completion_rate_pct:.1f}%\n\n"
    )
    if dock_appointment_analysis.escalation_required:
        lines.append(
            f"> 🚨 **ESCALATION REQUIRED** — "
            f"{dock_appointment_analysis.non_compliant_count} Non-Compliant "
            f"appointment(s) detected. Escalate immediately per SOP.\n\n"
        )

    # Gmail dock-activity summary
    lines.append("### Dock Activity Emails\n\n")
    lines.append(
        f"- **Approved:** {dock_email_summary.approved_count}\n"
        f"- **Requested (pending):** {dock_email_summary.requested_count}\n"
    )
    if dock_email_summary.outstanding_requests:
        lines.append("\n**Outstanding (not yet approved):**\n\n")
        for req in dock_email_summary.outstanding_requests:
            lines.append(f"- {req}\n")
    lines.append("\n")

    # Dock Pending Log table
    lines.append("### Dock Pending Log\n\n")
    lines.append(f"**Total Open:** {dock_pending_summary.total_open}\n\n")
    if dock_pending_summary.by_site:
        lines.append("| Site | Open Count | Issue Breakdown |\n")
        lines.append("|------|-----------|------------------|\n")
        for s in dock_pending_summary.by_site:
            breakdown = "; ".join(f"{k}: {v}" for k, v in s.by_issue_type.items())
            lines.append(f"| {s.site} | {s.total} | {breakdown} |\n")
    lines.append("\n")

    # ── Section 2 — Missort & Loss ────────────────────────────────────────────
    lines.append("## Section 2 — Missort & Loss\n\n")

    # Sorter metrics
    lines.append("### Sorter Metrics\n\n")
    if sorter_analysis.sites:
        lines.append("| Site | Volume Output | Fail Rate % | Status |\n")
        lines.append("|------|--------------|-------------|--------|\n")
        for site in sorter_analysis.sites:
            alerts: list[str] = []
            if site.fail_rate_alert:
                alerts.append(f"⚠️ Fail rate > {_SORTER_FAIL_RATE_THRESHOLD_PCT}% — ACTION NEEDED")
            if site.volume_drop_alert and site.volume_drop_pct is not None:
                alerts.append(
                    f"📉 Volume drop {site.volume_drop_pct:.1f}% DoD "
                    f"— divert adjustment may be needed"
                )
            status = "; ".join(alerts) if alerts else "✅ OK"
            lines.append(
                f"| {site.site} | {site.volume_output:,} | {site.fail_rate_pct:.1f}% | {status} |\n"
            )
    else:
        lines.append("_No sorter sites reported._\n")
    lines.append("\n")

    # Container flow table by site
    lines.append("### Container Flow by Site\n\n")
    if shipment_summary.flagged_sites:
        sites_str = ", ".join(shipment_summary.flagged_sites)
        lines.append(
            f"> ⚠️ Sites below {_COMPLETION_THRESHOLD_PCT}% completion: **{sites_str}**\n\n"
        )
    if shipment_summary.facilities:
        lines.append("| Site | Active | Loaded | Staged | Shipped | Completion % | WoW Trend |\n")
        lines.append("|------|--------|--------|--------|---------|-------------|----------|\n")
        for f in shipment_summary.facilities:
            flag = " ⚠️" if f.below_threshold else ""
            trend = (
                f"{f.week_over_week_delta:+.1f}pp"
                if f.week_over_week_delta is not None
                else "—"
            )
            lines.append(
                f"| {f.site} | {f.active} | {f.loaded} | {f.staged} | {f.shipped} "
                f"| {f.completion_pct:.1f}%{flag} | {trend} |\n"
            )
    lines.append("\n")

    # ── Section 3 — Dwell ─────────────────────────────────────────────────────
    lines.append("## Section 3 — Dwell\n\n")

    # BI dashboard dwell alerts (color-coded)
    lines.append("### Dwell Alerts (BI Dashboard)\n\n")

    if dwell_analysis.carrier_alerts:
        lines.append(
            f"#### 🔴🔴 Possible Misload Alerts — Carrier ≥ {_CARRIER_ALERT_THRESHOLD} packages\n\n"
        )
        for r in dwell_analysis.carrier_alerts:
            lines.append(
                f"- **{r.shipper}** | Carrier: {r.carrier} | "
                f"Site: {r.site} | Orders: {r.orders} | Packages: **{r.packages}**\n"
            )
        lines.append("\n")

    if dwell_analysis.brand_alerts:
        lines.append(
            f"#### 🔴 Missort Alerts — ≥ {_PKG_ALERT_THRESHOLD} packages\n\n"
        )
        for r in dwell_analysis.brand_alerts:
            lines.append(
                f"- **{r.shipper}** | Carrier: {r.carrier} | "
                f"Site: {r.site} | Packages: **{r.packages}**\n"
            )
        lines.append("\n")

    if dwell_analysis.brand_warnings:
        lines.append(
            f"#### 🟡 Missort Warnings — "
            f"{_PKG_WARNING_THRESHOLD}–{_PKG_ALERT_THRESHOLD - 1} packages\n\n"
        )
        for r in dwell_analysis.brand_warnings:
            lines.append(
                f"- **{r.shipper}** | Carrier: {r.carrier} | "
                f"Site: {r.site} | Packages: {r.packages}\n"
            )
        lines.append("\n")

    if not (dwell_analysis.carrier_alerts or dwell_analysis.brand_alerts or dwell_analysis.brand_warnings):
        lines.append("_No active dwell alerts._\n\n")

    # Open Dwell Impact Log tickets
    lines.append("### Open Dwell Impact Log Tickets\n\n")
    if dwell_ticket_summary.open_tickets:
        lines.append(f"**Total Open:** {dwell_ticket_summary.total_open}\n\n")
        lines.append(
            "| Date | Facility | Packages | Severity | Client Impact | Issue Description |\n"
            "|------|----------|----------|----------|---------------|-------------------|\n"
        )
        sev_icons = {
            "ok": "—",
            "pkg_warning": "🟡",
            "pkg_alert": "🔴",
            "carrier_alert": "🔴🔴",
        }
        for t in dwell_ticket_summary.open_tickets:
            icon = sev_icons.get(t.severity.value, "")
            lines.append(
                f"| {t.date} | {t.facility} | {t.packages} | {icon} {t.severity.value} "
                f"| {t.client_impact[:35]} | {t.issue_description[:45]} |\n"
            )
    else:
        lines.append("_No open dwell tickets._\n")
    lines.append("\n")

    # ── Artifact reminder ─────────────────────────────────────────────────────
    lines.append(
        "---\n\n"
        "> 📊 **Reminder:** Open the live artifact **ops-report** for the interactive "
        "version with charts. The artifact auto-refreshes Sheets and Gmail data on every open.\n"
    )

    return "".join(lines)
