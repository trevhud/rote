import pytest
from extracted.write_digest import write_digest

REQ = {"repo": "a/b", "topic": "t"}


def test_basic():
    pl = {"reading_list": [{"title": "A", "url": "http://a"}, {"title": "B", "url": "http://b"}]}
    d = write_digest(REQ, pl, "- Topic 1\n- Topic 2")["digest"]
    assert "## Reading list" in d
    assert "- [A](http://a)" in d
    assert "- [B](http://b)" in d
    assert "## Repo outline" in d
    assert d.index("## Reading list") < d.index("## Repo outline")
    assert "- Topic 1\n- Topic 2" in d


def test_outline_verbatim():
    out = "  weird *outline*  \n# x"
    assert out in write_digest(REQ, {"reading_list": []}, out)["digest"]


def test_empty_list():
    d = write_digest(REQ, {"reading_list": []}, "")["digest"]
    assert "## Reading list" in d and "## Repo outline" in d
    assert "- [" not in d


def test_malformed():
    with pytest.raises(ValueError):
        write_digest(REQ, {}, "x")
    with pytest.raises(ValueError):
        write_digest(REQ, {"reading_list": [{"title": "A"}]}, "x")
    with pytest.raises(ValueError):
        write_digest(REQ, {"reading_list": []}, None)
