import json
from typing import Any


def _threads(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            return _threads(json.loads(text))
        except ValueError:
            return [ln.strip() for ln in text.splitlines() if ln.strip()]
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for v in value.values():
            if isinstance(v, list):
                return v
        for v in value.values():
            if isinstance(v, str):
                try:
                    parsed = json.loads(v)
                except ValueError:
                    continue
                if isinstance(parsed, (list, dict)):
                    return _threads(parsed)
        if "subject" in value or "snippet" in value:
            return [value]
        return []
    raise ValueError("unsupported dock_emails_search shape")


def _opt(d: dict, key: str):
    v = d.get(key)
    return None if v is None else str(v)


def dock_email_summary(request: dict, dock_emails_search: Any) -> dict:
    """Count Approved vs Requested dock appointments from email threads."""
    approved = 0
    outstanding = []
    for t in _threads(dock_emails_search):
        if isinstance(t, str):
            t = {"subject": t}
        if not isinstance(t, dict):
            raise ValueError("thread must be an object or text")
        subject = str(t.get("subject") or "")
        snippet = _opt(t, "snippet")
        text = (subject + " " + (snippet or "")).lower()
        # skill: "Count Approved vs. Requested appointments."
        if "approved" in text:
            approved += 1
        elif "request" in text:
            # skill: "Flag any outstanding requests not yet approved."
            tid = t.get("thread_id", t.get("id"))
            outstanding.append({
                "subject": subject,
                "sender": _opt(t, "sender"),
                "date": _opt(t, "date"),
                "snippet": snippet,
                "thread_id": None if tid is None else str(tid),
            })
    return {
        "approved_count": approved,
        "requested_count": len(outstanding),
        "outstanding_requests": outstanding,
        "outstanding_count": len(outstanding),
    }
