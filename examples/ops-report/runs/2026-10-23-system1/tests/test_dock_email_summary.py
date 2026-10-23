import json
import pytest
from extracted.dock_email_summary import dock_email_summary

REQ = {"report_date": "2026-09-14"}
T = [
    {"id": "a", "subject": "Appointment Approved", "sender": "x", "date": "d", "snippet": "ok"},
    {"id": "b", "subject": "Appointment Requested", "sender": "y", "date": "d2", "snippet": "pls"},
]


def check(r):
    assert r["approved_count"] == 1
    assert r["requested_count"] == 1
    assert r["outstanding_count"] == 1
    assert r["outstanding_requests"][0]["thread_id"] == "b"
    assert r["outstanding_requests"][0]["subject"] == "Appointment Requested"


def test_list():
    check(dock_email_summary(REQ, T))


def test_wrapped():
    check(dock_email_summary(REQ, {"threads": T}))


def test_json_text():
    check(dock_email_summary(REQ, json.dumps(T)))


def test_plain_text():
    r = dock_email_summary(REQ, "Appointment Approved\nAppointment Requested")
    assert r["approved_count"] == 1 and r["outstanding_count"] == 1


def test_empty():
    r = dock_email_summary(REQ, [])
    assert r == {"approved_count": 0, "requested_count": 0,
                 "outstanding_requests": [], "outstanding_count": 0}


def test_malformed():
    with pytest.raises(ValueError):
        dock_email_summary(REQ, 5)
