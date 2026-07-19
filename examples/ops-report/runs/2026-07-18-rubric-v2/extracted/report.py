"""Render the daily executive operations report.

This module produces the fixed three-section Markdown report described in
SKILL.md's "Report format" section. The output format is fully specified
in the source skill — no LLM reasoning is needed to fill in the template.

All formatting logic is deterministic given the structured inputs from
upstream parse and validation nodes.
"""
from __future__ import annotations


def assemble_report(
    *,
    run_date: str | None,
    shipment_data: dict,
    dock_pending_data: dict,
    dock_activity: dict,
    dwell_ticket_data: dict,
    validated_manager_data: dict,
) -> str:
    """Render the complete daily executive operations report as Markdown.

    Structure (from SKILL.md "Report format"):

      # Daily Operations Report — {run_date}

      ## KPI Header
      | Active Sites | Dock Pending | Avg Completion | Open Dwell Tickets |
      | {n}          | {count}      | {pct}%         | {count}            |

      ## Section 1 — Dock Compliance
      - Appointment completion tiles (Green/Red/Grey/Non-Compliant)
      - Gmail dock-activity summary (Approved / Requested / Outstanding)
      - Dock Pending Log table by site and issue type

      ## Section 2 — Missort & Loss
      - Sorter metrics per site (volume, fail rate; action-needed flags)
      - Container flow table by site (active/loaded/staged/shipped/completion%)
        Sites below 75% completion are flagged.

      ## Section 3 — Dwell
      - BI dashboard dwell alerts (color-coded: missort_warning=light red,
        missort_alert=bright red, misload_alert=darkest red)
      - Open Dwell Impact Log tickets from the spreadsheet

      ---
      _Reminder: open the live artifact "ops-report" for the interactive
      version with auto-refreshing charts._

    Returns:
        str: The fully rendered Markdown report.
    """
    raise NotImplementedError(
        "Render the executive summary from the six typed inputs. "
        "Follow the exact three-section structure in SKILL.md 'Report format'. "
        "KPI header must show: len(shipment_data['sites']) active sites, "
        "dock_pending_data['total_open'] dock pending, "
        "shipment_data['avg_completion_pct'] avg completion, "
        "dwell_ticket_data['total_open'] open dwell tickets. "
        "Section 1: appointment completion tiles from "
        "validated_manager_data['appointment_completion']; if "
        "non_compliant_escalation_required, include an escalation alert. "
        "Section 2: sorter metrics with action_needed / divert flags; "
        "container flow table (flag sites with below_threshold=True). "
        "Section 3: color-coded dwell alerts from "
        "validated_manager_data['current_dwell'] (alert_level field); "
        "open tickets from dwell_ticket_data['open_tickets']. "
        "Append the live-artifact reminder at the end."
    )
