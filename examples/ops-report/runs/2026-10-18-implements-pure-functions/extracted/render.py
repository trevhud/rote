"""
Pure-function report renderer for the ops-report pipeline.

Takes all processed data dicts and returns the full Markdown executive
operations report. Stdlib only.
"""


def render_report(
    *,
    shipment_data: dict,
    dock_pending_data: dict,
    dock_email_data: dict,
    dwell_tickets_data: dict,
    appointment_data: dict,
    sorter_data: dict,
    dwell_data: dict,
) -> str:
    """
    Render the daily executive operations report as Markdown.

    Returns: str — full Markdown text of the report.

    Output structure (per skill):
      - KPI header: Active sites, Dock Pending count, Avg Completion %, Open Dwell Tickets
      - Section 1 — Dock Compliance
      - Section 2 — Missort & Loss
      - Section 3 — Dwell
      - Reminder to open live artifact
    """
    lines: list = []

    # skill: "Produce an executive-level summary"
    lines.append("# Daily Executive Operations Report")
    lines.append("")

    # skill: "KPI header: Active sites, Dock Pending count, Avg Completion %, Open Dwell Tickets"
    lines.append("## KPI Summary")
    lines.append("")
    lines.append("| Active Sites | Dock Pending | Avg Completion % | Open Dwell Tickets |")
    lines.append("|---|---|---|---|")
    avg_pct = shipment_data.get("avg_completion_pct", 0.0)
    lines.append(
        "| "
        + str(shipment_data.get("active_site_count", 0))
        + " | "
        + str(dock_pending_data.get("total_open", 0))
        + " | "
        + "{:.1f}%".format(avg_pct)
        + " | "
        + str(dwell_tickets_data.get("total_open", 0))
        + " |"
    )
    lines.append("")

    # skill: "Section 1 — Dock Compliance: Appointment tiles + Gmail summary + Pending Log table"
    lines.append("## Section 1 — Dock Compliance")
    lines.append("")

    # Appointment tiles
    # skill: "count of Green (Complete), Red (No Call/No Show), Grey (Canceled),
    #         and any other color (Non-Compliant). Non-Compliant must be escalated."
    lines.append("### Dock Appointments (Prior Day)")
    lines.append("")
    lines.append("| Status | Count |")
    lines.append("|---|---|")
    lines.append("| Green (Complete) | " + str(appointment_data.get("green", 0)) + " |")
    lines.append("| Red (No Call/No Show) | " + str(appointment_data.get("red", 0)) + " |")
    lines.append("| Grey (Canceled) | " + str(appointment_data.get("grey", 0)) + " |")
    lines.append("| Non-Compliant (Other) | " + str(appointment_data.get("other", 0)) + " |")
    lines.append("")
    if appointment_data.get("escalation_required"):
        nc = appointment_data.get("non_compliant_count", 0)
        # skill: "Non-Compliant must be escalated"
        lines.append(
            "**ESCALATION REQUIRED: "
            + str(nc)
            + " Non-Compliant appointment(s) must be escalated.**"
        )
        lines.append("")

    # Gmail summary
    # skill: "Count Approved vs. Requested appointments.
    #         Flag any outstanding requests not yet approved."
    lines.append("### Dock Activity Emails (Gmail — label:dock-activity last 24h)")
    lines.append("")
    lines.append("- Approved: " + str(dock_email_data.get("approved_count", 0)))
    lines.append("- Requested: " + str(dock_email_data.get("requested_count", 0)))
    unapproved = dock_email_data.get("unapproved_requests", [])
    if unapproved:
        lines.append("")
        lines.append(
            "**Outstanding requests not yet approved ("
            + str(len(unapproved))
            + "):**"
        )
        for req in unapproved:
            lines.append("  - " + req)
    lines.append("")

    # Pending Log table
    # skill: "Count all rows where Resolved = FALSE"
    lines.append("### Dock Pending Log")
    lines.append("")
    lines.append("**Total Open: " + str(dock_pending_data.get("total_open", 0)) + "**")
    lines.append("")

    by_site = dock_pending_data.get("by_site", {})
    if by_site:
        lines.append("**By Site:**")
        lines.append("")
        lines.append("| Site | Open Count |")
        lines.append("|---|---|")
        for site, count in sorted(by_site.items()):
            lines.append("| " + site + " | " + str(count) + " |")
        lines.append("")

    by_issue = dock_pending_data.get("by_issue_type", {})
    if by_issue:
        lines.append("**By Issue Type:**")
        lines.append("")
        lines.append("| Issue Type | Count |")
        lines.append("|---|---|")
        for issue, count in sorted(by_issue.items()):
            lines.append("| " + issue + " | " + str(count) + " |")
        lines.append("")

    # skill: "Section 2 — Missort & Loss: Sorter metrics + container flow table by site with completion %"
    lines.append("## Section 2 — Missort & Loss")
    lines.append("")

    # Sorter metrics
    lines.append("### Sorter Metrics")
    lines.append("")
    sorter_sites = sorter_data.get("sites", [])
    if sorter_sites:
        lines.append("| Site | Volume | Fail Rate | Flags |")
        lines.append("|---|---|---|---|")
        for sm in sorter_sites:
            flags_str = " ".join(sm.get("flags", []))
            fail_pct = sm.get("fail_rate_pct", 0.0)
            lines.append(
                "| "
                + str(sm.get("site", ""))
                + " | "
                + str(sm.get("volume", 0))
                + " | "
                + "{:.1f}%".format(fail_pct)
                + " | "
                + flags_str
                + " |"
            )
    else:
        lines.append("No sorter metrics provided.")
    lines.append("")

    # Container flow table
    # skill: "container flow table by site with completion %"
    # skill: "Flag any site below 75% completion"
    lines.append("### Container Flow by Site")
    lines.append("")
    lines.append("| Site | Active | Loaded | Staged | Shipped | Completion % | WoW Trend | Flag |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for sd in shipment_data.get("sites", []):
        wow = sd.get("wow_trend")
        trend_str = ""
        if wow is not None:
            trend_str = ("+" if wow >= 0 else "") + "{:.1f}%".format(wow)
        flag_str = "BELOW 75%" if sd.get("flagged") else ""
        comp_pct = sd.get("completion_pct", 0.0)
        lines.append(
            "| "
            + str(sd.get("site", ""))
            + " | "
            + str(sd.get("active", 0))
            + " | "
            + str(sd.get("loaded", 0))
            + " | "
            + str(sd.get("staged", 0))
            + " | "
            + str(sd.get("shipped", 0))
            + " | "
            + "{:.1f}%".format(comp_pct)
            + " | "
            + trend_str
            + " | "
            + flag_str
            + " |"
        )
    lines.append("")

    # skill: "Section 3 — Dwell: BI-dashboard dwell alerts (color-coded) + Open Dwell Impact Log tickets"
    lines.append("## Section 3 — Dwell")
    lines.append("")

    # BI dashboard dwell alerts (color-coded)
    lines.append("### BI Dashboard Dwell Alerts")
    lines.append("")
    flagged_rows = dwell_data.get("flagged_rows", [])
    if flagged_rows:
        lines.append("| Brand | Carrier | Site | Packages | Alert Level |")
        lines.append("|---|---|---|---|---|")
        for row in flagged_rows:
            lines.append(
                "| "
                + str(row.get("brand", ""))
                + " | "
                + str(row.get("carrier", ""))
                + " | "
                + str(row.get("site", ""))
                + " | "
                + str(row.get("packages", 0))
                + " | **"
                + str(row.get("alert_level", ""))
                + "** |"
            )
    else:
        lines.append("No dwell alerts.")
    lines.append("")

    # Open Dwell Impact Log
    lines.append("### Open Dwell Impact Log")
    lines.append("")
    open_tickets = dwell_tickets_data.get("open_tickets", [])
    if open_tickets:
        lines.append("| Date | Facility | Packages | Client Impact | Carrier Impact | Issue | Alert |")
        lines.append("|---|---|---|---|---|---|---|")
        for ticket in open_tickets:
            alert = ticket.get("alert_level", "")
            alert_str = ("**" + alert + "**") if alert else ""
            lines.append(
                "| "
                + str(ticket.get("date", ""))
                + " | "
                + str(ticket.get("facility", ""))
                + " | "
                + str(ticket.get("packages", ""))
                + " | "
                + str(ticket.get("client_impact", ""))
                + " | "
                + str(ticket.get("carrier_impact", ""))
                + " | "
                + str(ticket.get("issue_description", ""))
                + " | "
                + alert_str
                + " |"
            )
    else:
        lines.append("No open dwell tickets.")
    lines.append("")

    # skill: 'Also remind the duty manager to open the live artifact "ops-report" for the interactive version with charts.'
    lines.append("---")
    lines.append(
        "*Open the live artifact **ops-report** for the interactive version"
        " with charts. The artifact auto-refreshes Sheets and Gmail data on every open.*"
    )

    return "\n".join(lines)
