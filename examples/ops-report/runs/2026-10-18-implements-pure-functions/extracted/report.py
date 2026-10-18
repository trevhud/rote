"""
Render the Daily Executive Operations Report as Markdown.

Pure string-formatting — no LLM, no external calls. Takes structured data
from all upstream parse/threshold nodes and produces the executive report
defined by the skill's report format section.

Report structure (from skill):
  - KPI header: Active sites, Dock Pending count, Avg Completion %, Open Dwell Tickets
  - Section 1 — Dock Compliance
  - Section 2 — Missort & Loss
  - Section 3 — Dwell
  - Closing reminder about live artifact

No third-party imports — stdlib only.
"""

from typing import Any


# ── alert label helpers ───────────────────────────────────────────────────────

_ALERT_LABEL = {
    "missort_warning": "⚠ Missort Warning",
    "missort_alert": "🔴 Missort Alert",
    "none": "",
}

_MISLOAD_LABEL = "🟥 Possible Misload Alert"


def _alert_suffix(alert_level: str, misload: bool) -> str:
    parts = []
    label = _ALERT_LABEL.get(alert_level, "")
    if label:
        parts.append(label)
    if misload:
        parts.append(_MISLOAD_LABEL)
    return " | ".join(parts)


# ── section renderers ─────────────────────────────────────────────────────────

def _render_kpi_header(
    shipment_data: dict[str, Any],
    dock_pending_data: dict[str, Any],
    dwell_tickets: dict[str, Any],
) -> str:
    active_sites = shipment_data.get("active_site_count", 0)
    dock_pending = dock_pending_data.get("total_open", 0)
    avg_completion = shipment_data.get("avg_completion_pct", 0.0)
    open_dwell = dwell_tickets.get("total_open", 0)

    return (
        "## KPI Summary\n\n"
        "| Metric | Value |\n"
        "|--------|-------|\n"
        f"| Active Sites | {active_sites} |\n"
        f"| Dock Pending (Open) | {dock_pending} |\n"
        f"| Avg Completion % | {avg_completion:.1f}% |\n"
        f"| Open Dwell Tickets | {open_dwell} |\n"
    )


def _render_section1_dock_compliance(
    manual_thresholds: dict[str, Any],
    email_data: dict[str, Any],
    dock_pending_data: dict[str, Any],
) -> str:
    appt = manual_thresholds.get("appointment_summary", {})
    non_compliant = appt.get("non_compliant_count", 0)
    escalation_note = ""
    if non_compliant > 0:
        escalation_note = (
            f"\n> **ESCALATION REQUIRED**: {non_compliant} Non-Compliant "
            "appointment(s) — notify dispatch supervisor immediately.\n"
        )

    appt_block = (
        "### Dock Appointment Completion\n\n"
        "| Status | Count |\n"
        "|--------|-------|\n"
        f"| ✅ Complete (Green) | {appt.get('green', 0)} |\n"
        f"| ❌ No Call/No Show (Red) | {appt.get('red', 0)} |\n"
        f"| ⬜ Canceled (Grey) | {appt.get('grey', 0)} |\n"
        f"| 🚨 Non-Compliant (Other) | {non_compliant} |\n"
        f"| **Total** | **{appt.get('total', 0)}** |\n"
        + escalation_note
    )

    approved = email_data.get("approved_count", 0)
    requested = email_data.get("requested_count", 0)
    outstanding = email_data.get("outstanding_requests", [])
    outstanding_block = ""
    if outstanding:
        items = "\n".join(f"  - {r}" for r in outstanding)
        outstanding_block = f"\n**Outstanding (not yet approved):**\n{items}\n"

    gmail_block = (
        "### Gmail Dock Activity\n\n"
        f"- Appointment Requests: **{requested}**\n"
        f"- Approved: **{approved}**\n"
        + outstanding_block
    )

    by_site = dock_pending_data.get("by_site", {})
    by_issue = dock_pending_data.get("by_issue_type", {})
    total_open = dock_pending_data.get("total_open", 0)

    site_rows = "\n".join(f"| {site} | {cnt} |" for site, cnt in sorted(by_site.items()))
    issue_rows = "\n".join(f"| {issue} | {cnt} |" for issue, cnt in sorted(by_issue.items()))

    pending_block = (
        "### Dock Pending Log\n\n"
        f"**Total Open: {total_open}**\n\n"
        "**By Site:**\n\n"
        "| Site | Open Count |\n"
        "|------|------------|\n"
        + (site_rows if site_rows else "| — | 0 |")
        + "\n\n**By Issue Type:**\n\n"
        "| Issue Type | Count |\n"
        "|------------|-------|\n"
        + (issue_rows if issue_rows else "| — | 0 |")
    )

    return "## Section 1 — Dock Compliance\n\n" + appt_block + "\n" + gmail_block + "\n" + pending_block


def _render_section2_missort_loss(
    manual_thresholds: dict[str, Any],
    shipment_data: dict[str, Any],
) -> str:
    sorter_sites = manual_thresholds.get("sorter_sites", [])

    sorter_rows = []
    for s in sorter_sites:
        flags = []
        if s.get("fail_rate_flagged"):
            flags.append("⚠ Fail Rate > 5%")
        if s.get("volume_drop_flagged"):
            drop = s.get("volume_drop_pct")
            flags.append(f"⚠ Volume Drop {drop:.1f}%")
        flag_str = " | ".join(flags) if flags else "—"
        vol_drop = f"{s.get('volume_drop_pct', 0):.1f}%" if s.get("volume_drop_pct") is not None else "N/A"
        sorter_rows.append(
            f"| {s['site']} | {s['volume_output']:,} | {s['fail_rate_pct']:.1f}% | {vol_drop} | {flag_str} |"
        )

    sorter_table = (
        "### Sorter Metrics\n\n"
        "| Site | Volume Output | Fail Rate | DoD Volume Drop | Flags |\n"
        "|------|--------------|-----------|-----------------|-------|\n"
        + ("\n".join(sorter_rows) if sorter_rows else "| — | — | — | — | No data |")
    )

    sites = shipment_data.get("sites", [])
    container_rows = []
    for s in sites:
        wow = f"{s['wow_trend_pct']:.1f}%" if s.get("wow_trend_pct") is not None else "N/A"
        flag = "⚠ Below 75%" if s.get("flagged") else "—"
        container_rows.append(
            f"| {s['site']} | {s['active']} | {s['loaded']} | "
            f"{s['staged']} | {s['shipped']} | {s['completion_pct']:.1f}% | {wow} | {flag} |"
        )

    container_table = (
        "### Container Flow by Site\n\n"
        "| Site | Active | Loaded | Staged | Shipped | Completion % | WoW | Flag |\n"
        "|------|--------|--------|--------|---------|-------------|-----|------|\n"
        + ("\n".join(container_rows) if container_rows else "| — | — | — | — | — | — | — | — |")
    )

    return "## Section 2 — Missort & Loss\n\n" + sorter_table + "\n\n" + container_table


def _render_section3_dwell(
    manual_thresholds: dict[str, Any],
    dwell_tickets: dict[str, Any],
) -> str:
    dwell_alerts = manual_thresholds.get("dwell_alerts", [])

    alert_rows = []
    for entry in dwell_alerts:
        suffix = _alert_suffix(entry.get("alert_level", "none"), entry.get("misload_alert", False))
        alert_rows.append(
            f"| {entry['site']} | {entry['shipper']} | {entry['carrier']} "
            f"| {entry['packages']} | {suffix or '—'} |"
        )

    dwell_table = (
        "### Current Dwell (BI Dashboard)\n\n"
        "| Site | Shipper/Brand | Carrier | Packages | Alert |\n"
        "|------|--------------|---------|----------|-------|\n"
        + ("\n".join(alert_rows) if alert_rows else "| — | — | — | — | No dwell entries |")
    )

    open_tickets = dwell_tickets.get("open_tickets", [])
    ticket_rows = []
    for t in open_tickets:
        suffix = _alert_suffix(t.get("alert_level", "none"), t.get("misload_alert", False))
        ticket_rows.append(
            f"| {t['date']} | {t['facility']} | {t['packages']} "
            f"| {t['client_impact']} | {t['carrier_impact']} "
            f"| {t['issue_description'][:60]} | {suffix or '—'} |"
        )

    impact_table = (
        "### Open Dwell Impact Log\n\n"
        "| Date | Facility | Pkgs | Client Impact | Carrier Impact | Issue | Alert |\n"
        "|------|----------|------|--------------|----------------|-------|-------|\n"
        + ("\n".join(ticket_rows) if ticket_rows else "| — | — | — | — | — | — | No open tickets |")
    )

    return "## Section 3 — Dwell\n\n" + dwell_table + "\n\n" + impact_table


# ── main render function ──────────────────────────────────────────────────────

def render_ops_report(
    shipment_data: dict[str, Any],
    dock_pending_data: dict[str, Any],
    email_data: dict[str, Any],
    dwell_tickets: dict[str, Any],
    manual_thresholds: dict[str, Any],
) -> dict[str, Any]:
    """
    Assemble the full Daily Executive Operations Report as Markdown.

    Inputs map to upstream node outputs:
      shipment_data     ← parse_shipment_containers.output
      dock_pending_data ← parse_dock_pending_log.output
      email_data        ← aggregate_dock_emails.output
      dwell_tickets     ← parse_dwell_tickets.output
      manual_thresholds ← apply_manual_thresholds.output

    Returns:
        {"report_markdown": str}
    """
    sections = [
        "# Daily Executive Operations Report\n",
        _render_kpi_header(shipment_data, dock_pending_data, dwell_tickets),
        _render_section1_dock_compliance(manual_thresholds, email_data, dock_pending_data),
        _render_section2_missort_loss(manual_thresholds, shipment_data),
        _render_section3_dwell(manual_thresholds, dwell_tickets),
        (
            "---\n\n"
            "> **Reminder:** Open the live artifact **ops-report** for the interactive "
            "version with charts. The artifact auto-refreshes Sheets and Gmail data on every open.\n"
        ),
    ]

    report_markdown = "\n\n".join(sections)
    return {"report_markdown": report_markdown}
