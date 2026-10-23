import csv
import io
import json
import re
from typing import Any


def _norm(s: Any) -> str:
    return re.sub(r"[^a-z0-9/]", "", str(s).lower())


def _csv_rows(text: str) -> list:
    reader = csv.reader(io.StringIO(text))
    table = [r for r in reader if any(c.strip() for c in r)]
    if not table:
        return []
    header = table[0]
    return [dict(zip(header, r)) for r in table[1:]]


def _rows(v: Any) -> list:
    if v is None:
        raise ValueError("open dwell data is missing")
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return []
        try:
            return _rows(json.loads(s))
        except ValueError:
            return _csv_rows(s)
    if isinstance(v, list):
        if not v:
            return []
        if all(isinstance(x, dict) for x in v):
            if all("text" in x and x.get("type") == "text" for x in v):
                return _rows("\n".join(str(x["text"]) for x in v))
            return list(v)
        if all(isinstance(x, (list, tuple)) for x in v):
            header = v[0]
            return [dict(zip(header, r)) for r in v[1:]]
        raise ValueError("unrecognized open dwell data")
    if isinstance(v, dict):
        for val in v.values():
            if isinstance(val, (list, dict)):
                try:
                    return _rows(val)
                except ValueError:
                    continue
        for val in v.values():
            if isinstance(val, str):
                try:
                    return _rows(val)
                except ValueError:
                    continue
        raise ValueError("no rows found in open dwell data")
    raise ValueError("unrecognized open dwell data")


def _num(x: Any):
    if x is None or isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return int(x)
    m = re.search(r"-?\d[\d,]*(?:\.\d+)?", str(x))
    if not m:
        return None
    return int(float(m.group(0).replace(",", "")))


def _text(x: Any):
    if x is None:
        return None
    s = str(x).strip()
    return s if s else None


def open_dwell_filter(request: dict, open_dwell_read: Any, missort_warning_packages: int = 25, missort_alert_packages: int = 41, possible_misload_carrier: int = 80) -> dict:
    rows = _rows(open_dwell_read)
    tickets = []
    flags = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("row is not an object")
        cols = {_norm(k): v for k, v in row.items()}
        if "open/closed" not in cols:
            raise ValueError("missing column: Open/Closed")
        status = str(cols["open/closed"] or "").strip()
        # skill: "Filter rows where Open/Closed column = "Open"."
        if status != "Open":
            continue

        def get(*names):
            for n in names:
                if n in cols:
                    return cols[n]
            return None

        packages = _num(get("packages", "#packages", "ofpackages", "package"))
        ticket = {
            "date": _text(get("date")),
            "facility": _text(get("facility")),
            "packages": packages,
            "client_impact": _text(get("clientimpact")),
            "carrier_impact": _text(get("carrierimpact")),
            "issue_description": _text(get("issuedescription", "issue")),
        }
        tickets.append(ticket)
        # skill: "Flag packages >= 25 (Missort Warning), >= 41 (Missort Alert), carrier >= 80 (Possible Misload Alert)."
        if packages is not None:
            if packages >= missort_alert_packages:
                name = "Missort Alert"
                flags.append({"date": ticket["date"], "facility": ticket["facility"], "flag": name, "value": packages})
            elif packages >= missort_warning_packages:
                name = "Missort Warning"
                flags.append({"date": ticket["date"], "facility": ticket["facility"], "flag": name, "value": packages})
        carrier = _num(ticket["carrier_impact"])
        if carrier is not None and carrier >= possible_misload_carrier:
            name = "Possible Misload Alert"
            flags.append({"date": ticket["date"], "facility": ticket["facility"], "flag": name, "value": carrier})
    return {
        "open_dwell_tickets": tickets,
        "open_ticket_count": len(tickets),
        "dwell_ticket_flags": flags,
    }
