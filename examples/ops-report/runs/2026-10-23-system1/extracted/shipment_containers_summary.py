import csv
import io
import json
from typing import Any


def _norm(s):
    return "".join(str(s).lower().split())


def _num(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace(",", "").replace("%", "")
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _find_rows(data):
    if isinstance(data, list):
        if all(isinstance(x, dict) for x in data):
            return data
        if all(isinstance(x, list) for x in data) and data:
            hdr = [str(h) for h in data[0]]
            return [dict(zip(hdr, r)) for r in data[1:]]
        return []
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, (list, dict, str)):
                try:
                    r = _find_rows(v) if not isinstance(v, str) else _parse(v)
                except ValueError:
                    continue
                if r:
                    return r
    return []


def _parse_csv(text):
    lines = [ln for ln in text.splitlines() if ln.strip()]
    # find header line containing completion
    start = None
    for i, ln in enumerate(lines):
        if "completion" in ln.lower():
            start = i
            break
    if start is None:
        raise ValueError("missing column: COMPLETION %")
    hdr_line = lines[start]
    body = lines[start + 1:]
    if "|" in hdr_line and "," not in hdr_line:
        def split(ln):
            return [c.strip() for c in ln.strip().strip("|").split("|")]
        rows = [split(ln) for ln in body if not set(ln.strip()) <= set("|-: ")]
        hdr = split(hdr_line)
    else:
        parsed = list(csv.reader(io.StringIO("\n".join([hdr_line] + body))))
        hdr, rows = parsed[0], parsed[1:]
    return [dict(zip(hdr, r)) for r in rows]


def _parse(text):
    t = text.strip()
    if not t:
        return []
    if t[0] in "[{":
        try:
            return _find_rows(json.loads(t))
        except json.JSONDecodeError:
            pass
    return _parse_csv(t)


def shipment_containers_summary(request: dict, shipment_containers_read: Any, completion_flag_threshold: str = '75%') -> dict:
    thr = _num(completion_flag_threshold)
    if thr is None:
        raise ValueError("invalid completion_flag_threshold")
    data = shipment_containers_read
    if isinstance(data, str):
        rows = _parse(data)
    elif data is None:
        raise ValueError("no shipment container data")
    else:
        rows = _find_rows(data)
    flow = []
    for r in rows:
        m = {_norm(k): v for k, v in r.items()}
        site = None
        for k in ("site", "facility", "sitename"):
            if k in m and str(m[k]).strip():
                site = str(m[k]).strip()
                break
        if site is None:
            continue
        if "completion%" not in m and "completion" not in m:
            raise ValueError("missing column: COMPLETION %")
        comp = _num(m.get("completion%", m.get("completion")))
        if comp is None:
            continue
        vals = {}
        for k in ("active", "loaded", "staged", "shipped"):
            if k not in m:
                raise ValueError("missing column: " + k)
            n = _num(m[k])
            vals[k] = int(n) if n is not None else 0
        wow = None
        for k, v in m.items():
            if "wow" in k or "weekoverweek" in k or k == "trend":
                wow = _num(v)
                break
        flow.append({
            "site": site,
            **vals,
            "completion_pct": comp,
            # skill: "Flag any site below 75% completion."
            "below_threshold": comp < thr,
            "wow_trend": wow,
        })
    if not flow and rows:
        raise ValueError("no parsable site rows")
    low = [{"site": f["site"], "completion_pct": f["completion_pct"]} for f in flow if f["below_threshold"]]
    avg = sum(f["completion_pct"] for f in flow) / len(flow) if flow else None
    return {
        "site_container_flow": flow,
        "low_completion_sites": low,
        "average_completion_pct": avg,
        "active_site_count": len(flow),
    }
