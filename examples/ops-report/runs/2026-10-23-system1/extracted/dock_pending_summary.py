import csv
import io
import json
from typing import Any


def _norm(name: Any) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _parse_text(text: str) -> list:
    stripped = text.strip()
    if not stripped:
        return []
    if stripped[0] in "[{":
        try:
            return _find_rows(json.loads(stripped))
        except json.JSONDecodeError:
            pass
    reader = csv.DictReader(io.StringIO(stripped))
    rows = [r for r in reader if r and any((v or "").strip() for v in r.values() if isinstance(v, str))]
    if reader.fieldnames is None:
        raise ValueError("could not parse dock pending log")
    heads = {_norm(h) for h in reader.fieldnames}
    if "resolved" not in heads:
        raise ValueError("missing column: Resolved")
    return rows


def _find_rows(obj: Any) -> list:
    if isinstance(obj, str):
        return _parse_text(obj)
    if isinstance(obj, list):
        if all(isinstance(x, dict) for x in obj):
            if obj and all(x.get("type") == "text" and isinstance(x.get("text"), str) for x in obj):
                return _parse_text("\n".join(x["text"] for x in obj))
            return obj
        raise ValueError("unrecognised dock pending log rows")
    if isinstance(obj, dict):
        for value in obj.values():
            if isinstance(value, (list, dict, str)):
                try:
                    return _find_rows(value)
                except ValueError:
                    continue
        raise ValueError("no rows found in dock pending log")
    raise ValueError("could not parse dock pending log")


def _get(row: dict, key: str) -> Any:
    for k, v in row.items():
        if _norm(k) == key:
            return v
    return None


def _has(row: dict, key: str) -> bool:
    return any(_norm(k) == key for k in row)


def dock_pending_summary(request: dict, dock_pending_read: Any) -> dict:
    rows = _find_rows(dock_pending_read)
    if rows and not _has(rows[0], "resolved"):
        raise ValueError("missing column: Resolved")
    total = 0
    by_site: dict = {}
    by_issue: dict = {}
    for row in rows:
        val = _get(row, "resolved")
        if isinstance(val, bool):
            is_false = val is False
        else:
            is_false = str(val).strip().lower() == "false"
        # skill: "Count all rows where Resolved = FALSE"
        if not is_false:
            continue
        total += 1
        # rote: assumption: blank site or issue type is grouped under "Unknown"
        site = str(_get(row, "site") or "").strip() or "Unknown"
        issue = str(_get(row, "issuetype") or "").strip() or "Unknown"
        by_site[site] = by_site.get(site, 0) + 1
        by_issue[issue] = by_issue.get(issue, 0) + 1
    return {
        "dock_pending_total": total,
        "dock_pending_by_site": by_site,
        "dock_pending_by_issue_type": by_issue,
    }
