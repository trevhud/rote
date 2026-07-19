"""Parse the Open Dwell Ticket Log into structured alert records.

Pure function. Extracted from SKILL.md data-source section:
'Filter rows where Open/Closed column = "Open". Report each open ticket.
Flag packages >= 25 (Missort Warning), >= 41 (Missort Alert),
carrier >= 80 (Possible Misload Alert).'
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

# Thresholds extracted from SKILL.md data-source section (lines ~33)
PKG_WARNING_THRESHOLD: int = 25   # Missort Warning  (light red)
PKG_ALERT_THRESHOLD: int = 41     # Missort Alert    (bright red)
CARRIER_ALERT_THRESHOLD: int = 80  # Possible Misload Alert (darkest red)

DWELL_TICKET_STATUS_OPEN = "Open"


class DwellSeverity(str, Enum):
    OK = "ok"
    PKG_WARNING = "pkg_warning"     # packages >= 25
    PKG_ALERT = "pkg_alert"         # packages >= 41
    CARRIER_ALERT = "carrier_alert"  # packages >= 80 (Possible Misload)


@dataclass
class DwellTicket:
    date: str
    facility: str
    packages: int
    client_impact: str
    carrier_impact: str
    issue_description: str
    severity: DwellSeverity


@dataclass
class DwellTicketSummary:
    open_tickets: list[DwellTicket] = field(default_factory=list)
    total_open: int = 0
    pkg_warning_tickets: list[DwellTicket] = field(default_factory=list)
    pkg_alert_tickets: list[DwellTicket] = field(default_factory=list)
    carrier_alert_tickets: list[DwellTicket] = field(default_factory=list)


def parse_dwell_tickets(rows: list[dict]) -> DwellTicketSummary:
    """Filter Open dwell tickets and apply package alert thresholds.

    Expects each row dict from the 'Dwell Log - Form Responses' tab to have:
      Open/Closed (or status): "Open" or "Closed"
      Date (or date): submission date string
      Facility (or facility): site identifier
      # packages (or packages): integer package count
      Client Impact (or client_impact): impact description text
      Carrier Impact (or carrier_impact): carrier impact description text
      Issue Description (or issue_description): issue detail text
    """
    open_rows = [
        r for r in rows
        if str(r.get("Open/Closed") or r.get("status") or "").strip() == DWELL_TICKET_STATUS_OPEN
    ]

    tickets: list[DwellTicket] = []
    for row in open_rows:
        pkgs = int(row.get("# packages") or row.get("packages") or 0)
        tickets.append(
            DwellTicket(
                date=str(row.get("Date") or row.get("date") or ""),
                facility=str(row.get("Facility") or row.get("facility") or ""),
                packages=pkgs,
                client_impact=str(row.get("Client Impact") or row.get("client_impact") or ""),
                carrier_impact=str(row.get("Carrier Impact") or row.get("carrier_impact") or ""),
                issue_description=str(
                    row.get("Issue Description") or row.get("issue_description") or ""
                ),
                severity=_classify_severity(pkgs),
            )
        )

    return DwellTicketSummary(
        open_tickets=tickets,
        total_open=len(tickets),
        pkg_warning_tickets=[t for t in tickets if t.severity == DwellSeverity.PKG_WARNING],
        pkg_alert_tickets=[t for t in tickets if t.severity == DwellSeverity.PKG_ALERT],
        carrier_alert_tickets=[t for t in tickets if t.severity == DwellSeverity.CARRIER_ALERT],
    )


def _classify_severity(packages: int) -> DwellSeverity:
    if packages >= CARRIER_ALERT_THRESHOLD:
        return DwellSeverity.CARRIER_ALERT
    if packages >= PKG_ALERT_THRESHOLD:
        return DwellSeverity.PKG_ALERT
    if packages >= PKG_WARNING_THRESHOLD:
        return DwellSeverity.PKG_WARNING
    return DwellSeverity.OK
