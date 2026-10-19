import pytest
from extracted.parse_shipment_containers import parse_shipment_containers, COMPLETION_THRESHOLD

HDR = "Site,Active,Loaded,Staged,Shipped,COMPLETION %\n"


def csv_rows(*rows):
    return HDR + "\n".join(",".join(str(v) for v in r) for r in rows)


# --- constant pin ---

def test_completion_threshold_constant():
    assert COMPLETION_THRESHOLD == 75.0


# --- flagging rules ---

def test_flags_site_strictly_below_75():
    # skill: "Flag any site below 75% completion" — 74.9 is below
    data = csv_rows(("EAST1", 10, 5, 3, 2, 74.9))
    result = parse_shipment_containers(data)
    assert "EAST1" in result["flagged_sites"]
    assert result["sites"][0]["flagged"] is True


def test_does_not_flag_site_at_exactly_75():
    # skill: "Flag any site below 75% completion" — 75.0 is not below
    data = csv_rows(("WEST2", 8, 4, 2, 2, 75.0))
    result = parse_shipment_containers(data)
    assert result["flagged_sites"] == []
    assert result["sites"][0]["flagged"] is False


def test_does_not_flag_site_above_75():
    data = csv_rows(("NORTH3", 6, 3, 2, 1, 90.0))
    result = parse_shipment_containers(data)
    assert result["flagged_sites"] == []


def test_boundary_at_74():
    data = csv_rows(("X", 1, 0, 0, 0, 74.0))
    result = parse_shipment_containers(data, completion_threshold=75.0)
    assert result["sites"][0]["flagged"] is True


def test_boundary_at_75():
    data = csv_rows(("X", 1, 0, 0, 0, 75.0))
    result = parse_shipment_containers(data, completion_threshold=75.0)
    assert result["sites"][0]["flagged"] is False


# --- multi-site and aggregates ---

def test_multiple_sites_correct_flagging():
    data = csv_rows(("EAST1", 10, 5, 3, 2, 80.0), ("WEST2", 6, 2, 1, 1, 70.0))
    result = parse_shipment_containers(data)
    assert len(result["sites"]) == 2
    assert result["flagged_sites"] == ["WEST2"]
    assert result["total_active"] == 16


def test_avg_completion_pct():
    data = csv_rows(("EAST1", 10, 5, 3, 2, 80.0), ("WEST2", 6, 2, 1, 1, 70.0))
    result = parse_shipment_containers(data)
    assert result["avg_completion_pct"] == 75.0


# --- week-over-week trend ---

def test_wow_trend_computed_when_prev_column_present():
    hdr = "Site,Active,Loaded,Staged,Shipped,COMPLETION %,Prev Completion %\n"
    data = hdr + "EAST1,10,5,3,2,80.0,75.0\n"
    result = parse_shipment_containers(data)
    assert result["sites"][0]["wow_trend"] == 5.0


def test_wow_trend_none_when_prev_column_absent():
    data = csv_rows(("EAST1", 10, 5, 3, 2, 80.0))
    result = parse_shipment_containers(data)
    assert result["sites"][0]["wow_trend"] is None


def test_wow_trend_negative():
    hdr = "Site,Active,Loaded,Staged,Shipped,COMPLETION %,Prev Completion %\n"
    data = hdr + "EAST1,10,5,3,2,70.0,80.0\n"
    result = parse_shipment_containers(data)
    assert result["sites"][0]["wow_trend"] == -10.0


# --- empty and edge inputs ---

def test_empty_string_returns_zero_results():
    result = parse_shipment_containers("")
    assert result == {
        "sites": [],
        "flagged_sites": [],
        "total_active": 0,
        "avg_completion_pct": 0.0,
    }


def test_header_only_returns_zero_results():
    result = parse_shipment_containers(HDR)
    assert result["sites"] == []
    assert result["total_active"] == 0


# --- missing column errors ---

def test_missing_loaded_column_raises_value_error():
    bad = "Site,Active,Staged,Shipped,COMPLETION %\nEAST1,10,3,2,80.0\n"
    with pytest.raises(ValueError, match="Loaded"):
        parse_shipment_containers(bad)


def test_missing_completion_column_raises_value_error():
    bad = "Site,Active,Loaded,Staged,Shipped\nEAST1,10,5,3,2\n"
    with pytest.raises(ValueError, match="COMPLETION"):
        parse_shipment_containers(bad)


# --- TSV input ---

def test_tsv_input_parsed_correctly():
    tsv = "Site\tActive\tLoaded\tStaged\tShipped\tCOMPLETION %\nEAST1\t10\t5\t3\t2\t80.0\n"
    result = parse_shipment_containers(tsv)
    assert len(result["sites"]) == 1
    assert result["sites"][0]["site"] == "EAST1"
