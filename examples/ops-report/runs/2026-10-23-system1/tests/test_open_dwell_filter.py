import json
import pytest
from extracted.open_dwell_filter import open_dwell_filter

REQ = {"report_date": "2026-09-14"}


def row(status="Open", pk="10", carrier="5", fac="EAST1"):
    return {"Date": "2026-09-13", "Facility": fac, "# Packages": pk,
            "Client Impact": "low", "Carrier Impact": carrier,
            "Issue Description": "x", "Open/Closed": status}


def flags(rows):
    return [(f["flag"], f["value"]) for f in open_dwell_filter(REQ, rows)["dwell_ticket_flags"]]


def test_filters_open_only():
    r = open_dwell_filter(REQ, [row(), row("Closed")])
    assert r["open_ticket_count"] == 1
    assert r["open_dwell_tickets"][0]["packages"] == 10
    assert r["open_dwell_tickets"][0]["facility"] == "EAST1"


def test_package_boundaries():
    assert flags([row(pk="24")]) == []
    assert flags([row(pk="25")]) == [("Missort Warning", 25)]
    assert flags([row(pk="40")]) == [("Missort Warning", 40)]
    assert flags([row(pk="41")]) == [("Missort Alert", 41)]


def test_carrier_boundaries():
    assert flags([row(carrier="79")]) == []
    assert flags([row(carrier="80")]) == [("Possible Misload Alert", 80)]


def test_constants():
    assert flags([row(pk="30")]) == [("Missort Warning", 30)]
    r = open_dwell_filter(REQ, [row(pk="30")], missort_warning_packages=31)
    assert r["dwell_ticket_flags"] == []
    r = open_dwell_filter(REQ, [row(pk="30")], missort_alert_packages=30)
    assert r["dwell_ticket_flags"][0]["flag"] == "Missort Alert"
    r = open_dwell_filter(REQ, [row(carrier="50")], possible_misload_carrier=50)
    assert r["dwell_ticket_flags"][0]["flag"] == "Possible Misload Alert"


def test_shapes():
    rows = [row(pk="45")]
    assert open_dwell_filter(REQ, {"rows": rows})["open_ticket_count"] == 1
    assert open_dwell_filter(REQ, json.dumps(rows))["open_ticket_count"] == 1
    csv_text = "Date,Facility,# Packages,Client Impact,Carrier Impact,Issue Description,Open/Closed\n2026-09-13,EAST1,45,low,5,x,Open\n"
    r = open_dwell_filter(REQ, csv_text)
    assert r["dwell_ticket_flags"][0]["flag"] == "Missort Alert"
    wrapped = [{"type": "text", "text": csv_text}]
    assert open_dwell_filter(REQ, wrapped)["open_ticket_count"] == 1


def test_empty():
    r = open_dwell_filter(REQ, [])
    assert r == {"open_dwell_tickets": [], "open_ticket_count": 0, "dwell_ticket_flags": []}
    assert open_dwell_filter(REQ, "")["open_ticket_count"] == 0


def test_malformed():
    with pytest.raises(ValueError):
        open_dwell_filter(REQ, [{"Date": "x"}])
    with pytest.raises(ValueError):
        open_dwell_filter(REQ, None)
