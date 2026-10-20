"""The fixture store: fixture sets keyed by vendor/api/version, in the shareable format."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rote.bench.http_fixtures import load_http
from rote.vendor_specs import SpecStore
from rote.vendor_specs.fixture_store import FixtureStore
from rote.vendor_specs.fixtures import FixtureError
from tests.test_http_fixtures import GMAIL_FIXTURES
from tests.test_vendor_specs import _Fetch


@pytest.fixture
def gmail(tmp_path: Path):  # noqa: ANN201
    spec = SpecStore(root=tmp_path / "specs", fetch=_Fetch()).google("gmail", "v1")
    assert spec is not None
    return spec


def test_a_set_is_validated_stamped_and_keyed_by_spec(tmp_path: Path, gmail) -> None:  # noqa: ANN001
    store = FixtureStore(tmp_path / "fx")
    fs = load_http({"google/gmail/v1": GMAIL_FIXTURES})["google/gmail/v1"]
    path = store.save(fs, gmail, name="ops-report.busy_day", origin="cases.yaml#busy_day")
    assert path == tmp_path / "fx" / "google" / "gmail" / "v1" / "ops-report.busy_day.json"
    raw = json.loads(path.read_text())
    assert (raw["spec"], raw["spec_revision"], raw["origin"], raw["trust"]) == (
        "google/gmail/v1",
        "20261012",
        "cases.yaml#busy_day",
        "spec",
    )
    again = store.load("google/gmail/v1", "ops-report.busy_day", gmail)
    assert again.operations == fs.operations
    assert store.list() == [("google/gmail/v1", "ops-report.busy_day")]
    assert store.list("google/drive/v3") == []


def test_a_set_that_does_not_fit_is_refused_in_and_out(tmp_path: Path, gmail) -> None:  # noqa: ANN001
    store = FixtureStore(tmp_path / "fx")
    bad = load_http(
        {"google/gmail/v1": {"operations": {"gmail.users.threads.list": [{"body": {"x": 1}}]}}}
    )["google/gmail/v1"]
    with pytest.raises(FixtureError, match="field 'x' is not in the schema"):
        store.save(bad, gmail, name="bad", origin="t")
    assert store.list() == []
    # A stored set that a newer revision no longer fits fails on the way out.
    good = load_http({"google/gmail/v1": GMAIL_FIXTURES})["google/gmail/v1"]
    store.save(good, gmail, name="good", origin="t")
    newer = gmail.model_copy(
        update={"operations": [o for o in gmail.operations if o.id != "gmail.users.threads.get"]}
    )
    with pytest.raises(FixtureError, match="gmail.users.threads.get: not an operation"):
        store.load("google/gmail/v1", "good", newer)


@pytest.mark.parametrize(
    ("key", "name"),
    [
        ("google/gmail", "n"),
        ("google/../v1/x", "n"),
        ("google/gmail/v1", "../escape"),
        ("a/b/c", ""),
    ],
)
def test_keys_and_names_cannot_leave_the_store(tmp_path: Path, key: str, name: str) -> None:
    with pytest.raises(FixtureError):
        FixtureStore(tmp_path / "fx").path(key, name)


def test_a_case_can_use_a_published_set_by_name(tmp_path: Path, gmail) -> None:  # noqa: ANN001
    store = FixtureStore(tmp_path / "fx")
    fs = load_http({"google/gmail/v1": GMAIL_FIXTURES})["google/gmail/v1"]
    store.save(fs, gmail, name="shared", origin="t")
    used = load_http({"google/gmail/v1": {"from_store": "shared"}}, store)
    assert used["google/gmail/v1"].operations == fs.operations
    with pytest.raises(FixtureError, match="no fixture set 'missing'"):
        load_http({"google/gmail/v1": {"from_store": "missing"}}, store)


def test_bench_save_fixtures_writes_every_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import yaml

    from rote.bench.runner import main

    monkeypatch.setattr("rote.vendor_specs.store.http_get_json", _Fetch())
    example = tmp_path / "examples" / "dock"
    (example / "fidelity").mkdir(parents=True)
    (example / "skill").mkdir()
    (example / "skill" / "SKILL.md").write_text("---\nname: dock\n---\n", encoding="utf-8")
    case = {"input": {}, "fixtures": {}, "http": {"google/gmail/v1": GMAIL_FIXTURES}}
    (example / "fidelity" / "cases.yaml").write_text(
        yaml.safe_dump(
            {"output_fields": {"x": "x"}, "cases": [{"id": "a", **case}, {"id": "b", **case}]}
        ),
        encoding="utf-8",
    )
    rc = main(["--examples", str(tmp_path / "examples"), "--skills", "dock", "--save-fixtures"])
    assert rc == 0
    assert FixtureStore().list() == [("google/gmail/v1", "dock.a"), ("google/gmail/v1", "dock.b")]
