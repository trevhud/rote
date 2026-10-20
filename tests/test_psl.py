"""Registrable domains by the bundled, pinned Public Suffix List."""

from __future__ import annotations

import hashlib

import pytest

from rote.vendor_specs import psl
from rote.vendor_specs.psl import public_suffix, registrable_domain


def test_the_bundled_list_is_the_pinned_upstream_file() -> None:
    text = psl.list_text()
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == psl.SHA256
    assert "===BEGIN PRIVATE DOMAINS===" in text


@pytest.mark.parametrize(
    ("host", "domain"),
    [
        ("api.slack.com", "slack.com"),
        ("slack.com", "slack.com"),
        ("api.bbc.co.uk", "bbc.co.uk"),
        # googleapis.com is on the private list: each subdomain is its own.
        ("www.googleapis.com", "www.googleapis.com"),
        ("API.Slack.com.", "slack.com"),
        # Private section: each is its own owner.
        ("a.github.io", "a.github.io"),
        ("docs.a.github.io", "a.github.io"),
        ("myapp.herokuapp.com", "myapp.herokuapp.com"),
        ("bucket.s3.amazonaws.com", "bucket.s3.amazonaws.com"),
        # A wildcard rule: *.compute.amazonaws.com.
        (
            "ec2-1-2-3-4.us-east-1.compute.amazonaws.com",
            "ec2-1-2-3-4.us-east-1.compute.amazonaws.com",
        ),
        (
            "x.ec2-1-2-3-4.us-east-1.compute.amazonaws.com",
            "ec2-1-2-3-4.us-east-1.compute.amazonaws.com",
        ),
        ("example.or.jp", "example.or.jp"),
        ("api.example.or.jp", "example.or.jp"),
        # An exception rule: !city.kobe.jp under *.kobe.jp.
        ("www.city.kobe.jp", "city.kobe.jp"),
        ("x.y.kobe.jp", "x.y.kobe.jp"),
        # A host that is itself a public suffix, an IP, one label: itself.
        ("github.io", "github.io"),
        ("or.jp", "or.jp"),
        ("127.0.0.1", "127.0.0.1"),
        ("localhost", "localhost"),
        # Not on the list: the default rule (the last label) applies.
        ("api.items.example", "items.example"),
    ],
)
def test_registrable_domain(host: str, domain: str) -> None:
    assert registrable_domain(host) == domain


def test_two_github_pages_sites_do_not_share_a_domain() -> None:
    assert registrable_domain("a.github.io") != registrable_domain("b.github.io")
    assert public_suffix("a.github.io") == "github.io"
