import pytest
from extracted.write_digest import write_digest, FIRST_SECTION, SECOND_SECTION

REQ = {"repo": "a/b", "topic": "t"}


def test_constants():
    assert FIRST_SECTION == "Reading list"
    assert SECOND_SECTION == "Repo outline"


def test_digest():
    pl = {"reading_list": [{"title": "A", "url": "http://a"}, {"title": "B", "url": "http://b"}]}
    d = write_digest(REQ, pl, "- Overview\n- Setup")["digest"]
    assert "## Reading list" in d
    assert "- [A](http://a)" in d
    assert "- [B](http://b)" in d
    assert "## Repo outline" in d
    assert d.index("Reading list") < d.index("Repo outline")
    assert "- Overview\n- Setup" in d


def test_empty():
    d = write_digest(REQ, {"reading_list": []}, "")["digest"]
    assert "## Reading list" in d and "## Repo outline" in d


def test_malformed():
    with pytest.raises(ValueError):
        write_digest(REQ, {}, "x")
    with pytest.raises(ValueError):
        write_digest(REQ, {"reading_list": [{"title": "A"}]}, "x")
