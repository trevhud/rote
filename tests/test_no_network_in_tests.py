"""The suite-wide network guard itself (tests/conftest.py::_no_network_in_fast_tests).

Its failure mode is silent: a test that reaches a real service still
passes, slowly, and spends or leaks whatever it sends. So the guard is
tested like the inference guard is.
"""

from __future__ import annotations

import socket
import threading

import pytest

from tests.conftest import NetworkBlocked, _loopback


def test_a_non_loopback_connect_is_blocked() -> None:
    with pytest.raises(NetworkBlocked, match="fast tests may not connect"):
        socket.create_connection(("192.0.2.10", 443), timeout=1)  # TEST-NET-1, never routed
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(NetworkBlocked):
            s.connect_ex(("203.0.113.5", 80))
    finally:
        s.close()


def test_loopback_still_works() -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    accepted: list[socket.socket] = []
    t = threading.Thread(target=lambda: accepted.append(server.accept()[0]))
    t.start()
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=5):
            pass
        t.join(5)
        assert accepted
    finally:
        for c in accepted:
            c.close()
        server.close()


@pytest.mark.parametrize(
    ("address", "allowed"),
    [
        (("127.0.0.1", 80), True),
        (("::1", 80, 0, 0), True),
        (("localhost", 80), True),
        (("0.0.0.0", 80), True),
        ("/tmp/some.sock", True),
        (("192.0.2.10", 443), False),
        (("2001:db8::1", 443, 0, 0), False),
        (("api.cloudflare.com", 443), False),
    ],
)
def test_what_counts_as_this_machine(address: object, allowed: bool) -> None:
    assert _loopback(address) is allowed
