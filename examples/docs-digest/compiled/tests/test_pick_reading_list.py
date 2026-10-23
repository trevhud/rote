import pytest
from extracted.pick_reading_list import pick_reading_list, RESULT_LIMIT

REQ = {"repo": "a/b", "topic": "t"}


def mk(n):
    return {"results": [{"title": f"T{i}", "contentUrl": f"http://x/{i}", "content": "c"} for i in range(n)]}


def test_constant():
    assert RESULT_LIMIT == 5


def test_six_gives_five():
    out = pick_reading_list(REQ, mk(6))
    assert out["reading_list"] == [{"title": f"T{i}", "url": f"http://x/{i}"} for i in range(5)]


def test_exactly_five():
    assert len(pick_reading_list(REQ, mk(5))["reading_list"]) == 5


def test_four():
    assert len(pick_reading_list(REQ, mk(4))["reading_list"]) == 4


def test_empty():
    assert pick_reading_list(REQ, {"results": []}) == {"reading_list": []}


def test_malformed():
    with pytest.raises(ValueError):
        pick_reading_list(REQ, {})
