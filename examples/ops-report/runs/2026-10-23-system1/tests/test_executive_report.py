import pytest
from extracted.executive_report import executive_report


def _args():
    return dict(
        request={"report_date": "2026-09-14"},
        shipment_containers_summary={"active_site_count": 2, "average_completion_pct": 80.0,
                                     "low_completion_sites": [], "site_container_flow": [{"site": "A"}]},
        dock_pending_summary={"dock_pending_total": 3, "dock_pending_by_site": {}, "dock_pending_by_issue_type": {}},
        dock_email_summary={"approved_count": 1, "requested_count": 2, "outstanding_requests": [], "outstanding_count": 1},
        open_dwell_filter={"open_dwell_tickets": [{"date": "d", "facility": "F", "packages": 30,
                                                   "client_impact": None, "carrier_impact": None,
                                                   "issue_description": None}],
                           "open_ticket_count": 1,
                           "dwell_ticket_flags": [{"date": "d", "facility": "F", "flag": "Missort Warning", "value": 30}]},
        manual_data_flags={"appointment_completion": {"green": 1, "red": 2, "grey": 3, "non_compliant": 4},
                           "appointment_escalations": [], "sorter_flags": [{"site": "A"}],
                           "current_dwell_alerts": [{"shipper": "s"}]},
    )


def test_report():
    r = executive_report(**_args())["executive_report"]
    assert r["report_date"] == "2026-09-14"
    assert r["kpi_header"] == {"active_sites": 2, "dock_pending_count": 3,
                               "avg_completion_pct": 80.0, "open_dwell_tickets": 1}
    assert r["section_1_dock_compliance"]["appointment_tiles"]["non_compliant"] == 4
    assert r["section_2_missort_loss"]["container_flow_table"] == [{"site": "A"}]
    assert r["section_3_dwell"]["open_dwell_impact_log"][0]["flags"] == ["Missort Warning"]
    assert "ops-report" in r["artifact_reminder"]


def test_empty():
    a = _args()
    a["open_dwell_filter"] = {"open_dwell_tickets": [], "open_ticket_count": 0, "dwell_ticket_flags": []}
    r = executive_report(**a)["executive_report"]
    assert r["section_3_dwell"]["open_dwell_impact_log"] == []


def test_malformed():
    a = _args()
    a["request"] = {}
    with pytest.raises(ValueError):
        executive_report(**a)
