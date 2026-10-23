import pytest
from extracted.pick_reading_list import pick_reading_list

REQ = {"repo": "a/b", "topic": "t"}


def _res(n):
    return {"results": [{"title": f"T{i}", "contentUrl": f"http://x/{i}", "content": "c"} for i in range(n)]}


def test_constant_default():
    import inspect
    assert inspect.signature(pick_reading_list).parameters["result_count"].default == 5


def test_six_results_keeps_five():
    out = pick_reading_list(REQ, _res(6))
    assert len(out["reading_list"]) == 5
    assert out["reading_list"][0] == {"title": "T0", "url": "http://x/0"}
    assert out["reading_list"][4]["title"] == "T4"


def test_exactly_five():
    assert len(pick_reading_list(REQ, _res(5))["reading_list"]) == 5


def test_four_results():
    assert len(pick_reading_list(REQ, _res(4))["reading_list"]) == 4


def test_empty():
    assert pick_reading_list(REQ, {"results": []}) == {"reading_list": []}


def test_custom_count():
    assert len(pick_reading_list(REQ, _res(6), result_count=2)["reading_list"]) == 2


def test_malformed():
    with pytest.raises(ValueError):
        pick_reading_list(REQ, "bad")
    with pytest.raises(ValueError):
        pick_reading_list(REQ, {"results": "x"})
