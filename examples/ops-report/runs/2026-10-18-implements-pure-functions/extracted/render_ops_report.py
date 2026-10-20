"""
Render the daily executive operations report as a Markdown string.

Pure string-formatting function — no LLM, no I/O.
Inputs are the structured dicts produced by all five upstream parse/validate nodes.
"""

import datetime


# skill: "Brand 25-40 = Missort Warning (light red), 41+ = Missort Alert (bright red). Carrier 80+ missing = Possible Misload Alert (darkest red)."
ALERT_LABELS = {
    "Missort Warning": "[MISSORT WARNING]",
    "Missort Alert": "[MISSORT ALERT]",
    "Possible Misload Alert": "[POSSIBLE MISLOAD ALERT]",
}


def render_ops_report(
    shipment_data: dict,
    dock_pending_data: dict,
    email_data: dict,
    dwell_data: dict,
    manual_data: dict,
) -> str:
    """Render the three-section executive operations report as Markdown.

    Args:
        shipment_data: Output of parse_shipment_containers.
        dock_pending_data: Output of parse_dock_pending_log.
        email_data: Output of parse_dock_activity_emails.
        dwell_data: Output of parse_dwell_tickets.
        manual_data: Output of validate_manual_data.

    Returns:
        Multi-line Markdown string for the complete report.
    """
    today = datetime.date.today().strftime("%B %d, %Y")
    lines = []

    # skill: "KPI header: Active sites, Dock Pending count, Avg Completion %, Open Dwell Tickets"
    active_sites = len(shipment_data.get("sites") or [])
    dock_pending = dock_pending_data.get("total_open", 0)
    avg_completion = shipment_data.get("avg_completion_pct", 0.0)
    open_dwell = dwell_data.get("total_open", 0)

    lines.append(f"# Daily Operations Report — {today}")
    lines.append("")
    lines.append("## KPI Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Active Sites | {active_sites} |")
    lines.append(f"| Dock Pending (Open) | {dock_pending} |")
    avg_str = f"{avg_completion:.1f}%"
    lines.append(f"| Avg Completion % | {avg_str} |")
    lines.append(f"| Open Dwell Tickets | {open_dwell} |")
    lines.append("")

    # skill: "Section 1 — Dock Compliance: Appointment tiles + Gmail summary + Pending Log table"
    lines.append("## Section 1 — Dock Compliance")
    lines.append("")

    appts = manual_data.get("dock_appointments") or {}
    lines.append("### Dock Appointment Completion (Prior Day)")
    lines.append("")
    lines.append("| Status | Count |")
    lines.append("|---|---|")
    lines.append(f"| Green (Complete) | {appts.get('green', 0)} |")
    lines.append(f"| Red (No Call/No Show) | {appts.get('red', 0)} |")
    lines.append(f"| Grey (Canceled) | {appts.get('grey', 0)} |")
    nc = appts.get("non_compliant", 0)
    nc_suffix = " **[ESCALATION REQUIRED]**" if appts.get("escalation_required") else ""
    lines.append(f"| Non-Compliant | {nc}{nc_suffix} |")
    lines.append("")

    lines.append("### Dock Activity Emails (Last 24h)")
    lines.append("")
    lines.append(f"- Approved appointments: **{email_data.get('approved_count', 0)}**")
    lines.append(f"- Requested appointments: **{email_data.get('requested_count', 0)}**")
    outstanding = email_data.get("outstanding_requests") or []
    if outstanding:
        lines.append(f"- **Outstanding (not yet approved): {len(outstanding)}** — review required")
    else:
        lines.append("- No outstanding requests")
    lines.append("")

    lines.append("### Dock Pending Log (Open Issues)")
    lines.append("")
    lines.append(f"**Total open: {dock_pending}**")
    lines.append("")
    by_site = dock_pending_data.get("by_site") or {}
    if by_site:
        lines.append("**By Site:**")
        for site_name, count in sorted(by_site.items()):
            lines.append(f"- {site_name}: {count}")
        lines.append("")
    by_issue = dock_pending_data.get("by_issue_type") or {}
    if by_issue:
        lines.append("**By Issue Type:**")
        for issue, count in sorted(by_issue.items()):
            lines.append(f"- {issue}: {count}")
        lines.append("")

    # skill: "Section 2 — Missort & Loss: Sorter metrics + container flow table by site with completion %"
    lines.append("## Section 2 — Missort & Loss")
    lines.append("")

    sorter_metrics = manual_data.get("sorter_metrics") or []
    if sorter_metrics:
        lines.append("### Sorter Metrics")
        lines.append("")
        lines.append("| Site | Volume Output | Fail Rate % | DoD Change | Flags |")
        lines.append("|---|---|---|---|---|")
        for m in sorter_metrics:
            flags = []
            if m.get("fail_rate_flag"):
                flags.append("**ACTION: Fail rate > 5%**")
            if m.get("volume_drop_flag"):
                drop = m.get("volume_drop_pct") or 0
                drop_str = f"{drop:.1f}"
                flags.append(f"**Divert adj. needed (volume -{drop_str}%)**")
            dod = ""
            vdp = m.get("volume_drop_pct")
            if vdp is not None:
                dod = f"{vdp:+.1f}%"
            # rote: assumption: display "OK" when no sorter flags are raised; the skill names the flags but not the passing-state label
            flag_str = "; ".join(flags) if flags else "OK"
            fr_str = f"{m['fail_rate_pct']:.1f}%"
            lines.append(
                f"| {m['site']} | {m['volume_output']} | {fr_str} | {dod} | {flag_str} |"
            )
        lines.append("")

    lines.append("### Container Flow by Site")
    lines.append("")
    sites = shipment_data.get("sites") or []
    if sites:
        lines.append("| Site | Active | Loaded | Staged | Shipped | Completion % | WoW | Status |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for s in sites:
            wow = f"{s['wow_trend']:+.1f}%" if s.get("wow_trend") is not None else "—"
            # skill: "Flag any site below 75% completion"
            # rote: assumption: "OK" is the passing-state status label; the skill names the flag but not the passing label
            status = "**BELOW 75%**" if s.get("flagged") else "OK"
            comp_str = f"{s['completion_pct']:.1f}%"
            lines.append(
                f"| {s['site']} | {s['active']} | {s['loaded']} | {s['staged']}"
                f" | {s['shipped']} | {comp_str} | {wow} | {status} |"
            )
        lines.append("")
        flagged = shipment_data.get("flagged_sites") or []
        if flagged:
            flagged_str = ", ".join(flagged)
            lines.append(f"> **Sites below 75% completion: {flagged_str}**")
            lines.append("")
    else:
        lines.append("_No container data available._")
        lines.append("")

    # skill: "Section 3 — Dwell: BI-dashboard dwell alerts (color-coded) + Open Dwell Impact Log tickets"
    lines.append("## Section 3 — Dwell")
    lines.append("")

    current_dwell = manual_data.get("current_dwell") or []
    if current_dwell:
        lines.append("### Current Dwell Alerts (BI Dashboard)")
        lines.append("")
        lines.append("| Brand | Carrier | Site | Packages | Alert |")
        lines.append("|---|---|---|---|---|")
        for item in current_dwell:
            alert = item.get("alert_level")
            # rote: assumption: "—" is the no-alert placeholder; the skill describes alerts but not the no-alert display value
            alert_str = ALERT_LABELS.get(alert, "—") if alert else "—"
            lines.append(
                f"| {item['brand']} | {item['carrier']} | {item['site']}"
                f" | {item['packages']} | {alert_str} |"
            )
        lines.append("")

    open_tickets = dwell_data.get("open_tickets") or []
    lines.append("### Open Dwell Impact Log")
    lines.append("")
    if open_tickets:
        lines.append("| Date | Facility | Packages | Alert | Client Impact | Carrier Impact | Issue |")
        lines.append("|---|---|---|---|---|---|---|")
        for t in open_tickets:
            alert = t.get("alert_level")
            # rote: assumption: "—" is the no-alert placeholder; the skill describes alerts but not the no-alert display value
            alert_str = ALERT_LABELS.get(alert, "—") if alert else "—"
            lines.append(
                f"| {t['date']} | {t['facility']} | {t['packages']} | {alert_str}"
                f" | {t['client_impact']} | {t['carrier_impact']} | {t['issue_description']} |"
            )
    else:
        lines.append("_No open dwell tickets._")
    lines.append("")

    # skill: "Also remind the duty manager to open the live artifact "ops-report" for the interactive version with charts."
    lines.append("---")
    lines.append("")
    lines.append(
        "> **Reminder:** Open the live artifact **ops-report** for the interactive "
        "version with charts. The artifact auto-refreshes Sheets and Gmail data on every open."
    )
    lines.append("")

    return "\n".join(lines)
