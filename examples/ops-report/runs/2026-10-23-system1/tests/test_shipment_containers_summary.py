import pytest
from extracted.shipment_containers_summary import shipment_containers_summary as f

REQ = {"report_date": "2026-09-14"}
CSV = "Site,Active,Loaded,Staged,Shipped,COMPLETION %\nEAST1,5,10,3,20,74.9\nWEST2,1,2,3,4,75%\n"


def test_csv():
    r = f(REQ, CSV)
    assert r["active_site_count"] == 2
    assert [x["below_threshold"] for x in r["site_container_flow"]] == [True, False]
    assert r["low_completion_sites"] == [{"site": "EAST1", "completion_pct": 74.9}]
    assert r["average_completion_pct"] == pytest.approx(74.95)
    assert r["site_container_flow"][0]["active"] == 5


def test_list_wrapped_json():
    rows = [{"site": "A", "active": 1, "loaded": 2, "staged": 3, "shipped": 4, "completion %": 80}]
    import json
    for d in (rows, {"rows": rows}, json.dumps(rows)):
        assert f(REQ, d)["site_container_flow"][0]["site"] == "A"


def test_empty():
    r = f(REQ, [])
    assert r["average_completion_pct"] is None
    assert r["active_site_count"] == 0


def test_malformed():
    with pytest.raises(ValueError):
        f(REQ, "foo,bar\n1,2")


def test_threshold_default():
    import inspect
    assert inspect.signature(f).parameters["completion_flag_threshold"].default == "75%"
    assert f(REQ, CSV, "80%")["low_completion_sites"][-1]["site"] == "WEST2"
