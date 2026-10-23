def executive_report(request: dict, shipment_containers_summary: dict, dock_pending_summary: dict, dock_email_summary: dict, open_dwell_filter: dict, manual_data_flags: dict) -> dict:
    """Assemble the executive operations report."""
    if not isinstance(request, dict) or "report_date" not in request:
        raise ValueError("request must contain report_date")
    flags = open_dwell_filter.get("dwell_ticket_flags", [])
    log = []
    for t in open_dwell_filter.get("open_dwell_tickets", []):
        item = dict(t)
        item["flags"] = [
            f["flag"]
            for f in flags
            if f.get("facility") == t.get("facility") and f.get("date") == t.get("date")
        ]
        log.append(item)
    return {
        "executive_report": {
            "report_date": request["report_date"],
            # skill: "KPI header: Active sites, Dock Pending count, Avg Completion %, Open Dwell Tickets"
            "kpi_header": {
                "active_sites": shipment_containers_summary["active_site_count"],
                "dock_pending_count": dock_pending_summary["dock_pending_total"],
                "avg_completion_pct": shipment_containers_summary["average_completion_pct"],
                "open_dwell_tickets": open_dwell_filter["open_ticket_count"],
            },
            "section_1_dock_compliance": {
                "appointment_tiles": dict(manual_data_flags["appointment_completion"]),
                "gmail_summary": dict(dock_email_summary),
                "pending_log_table": dict(dock_pending_summary),
            },
            "section_2_missort_loss": {
                "sorter_metrics": list(manual_data_flags["sorter_flags"]),
                "container_flow_table": list(shipment_containers_summary["site_container_flow"]),
            },
            "section_3_dwell": {
                "dwell_alerts": list(manual_data_flags["current_dwell_alerts"]),
                "open_dwell_impact_log": log,
            },
            # skill: "Also remind the duty manager to open the live artifact"
            "artifact_reminder": 'Open the live artifact "ops-report" for the interactive version with charts. It auto-refreshes Sheets and Gmail data on every open.',
        }
    }
