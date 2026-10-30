"""Suite-wide offline/no-key guard.

The whole unit suite is contractually offline: it must run with **no**
``ANTHROPIC_API_KEY`` and **no** network access, exercising the pipeline through
injected fakes (a ``FakeTransport`` for the LLM, a fake ``getter`` for HTTP). The
autouse fixture below enforces that contract so a regression that secretly reaches
the real Anthropic SDK or opens a socket fails loudly *here*, locally, instead of
silently passing in CI (or worse, only failing when a key happens to be present).

For every test that is **not** marked ``smoke`` it:

* deletes ``ANTHROPIC_API_KEY`` from the environment, so any code path that builds
  the real client / reads the key raises instead of quietly succeeding;
* replaces ``socket.socket`` with a stub that raises on construction, so any
  accidental real outbound connection (requests, the anthropic SDK, trafilatura's
  fallback fetch, ...) blows up with a clear message; and
* answers ``socket.getaddrinfo`` offline (an IP literal maps to itself, any host
  name to one fixed public IPv4) and blocks the single-answer
  ``gethostbyname`` lookups, so no real DNS query is ever made.

The ``smoke`` tests (``pytest -m smoke``) are the *opt-in* live tests that legitimately
need a real key and the network, so they are exempt. All patches are applied via
the function-scoped ``monkeypatch`` fixture, so they are reverted automatically
after each test and never leak into collection/teardown.
"""

from __future__ import annotations

import ipaddress
import socket

import pytest

# The address every host name "resolves" to offline (a public IP, so URL-path
# tests pass the fetcher's SSRF guard).
OFFLINE_PUBLIC_IP = "93.184.216.34"


class _BlockedNetworkError(RuntimeError):
    """Raised when offline-guarded test code attempts to open a socket."""


def _offline_getaddrinfo(host, port, *args, **kwargs):
    """``socket.getaddrinfo`` without the network: one TCP answer per lookup."""
    try:
        ip = ipaddress.ip_address(host)
    except (TypeError, ValueError):
        ip = ipaddress.ip_address(OFFLINE_PUBLIC_IP)
    if ip.version == 6:
        return [(socket.AF_INET6, socket.SOCK_STREAM, socket.IPPROTO_TCP, "",
                 (str(ip), port or 0, 0, 0))]
    return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "",
             (str(ip), port or 0))]


@pytest.fixture(autouse=True)
def _offline_no_key_guard(request, monkeypatch):
    """Enforce the offline/no-key contract for every non-smoke test."""
    # The live smoke tests legitimately use a real key + network; leave them be.
    if request.node.get_closest_marker("smoke") is not None:
        yield
        return

    # 1) No key: any code that reads ANTHROPIC_API_KEY must fail, not succeed.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    # 2) No network: any attempt to open a socket raises loudly.
    def _blocked_socket(*args, **kwargs):
        raise _BlockedNetworkError(
            "network access is blocked in offline unit tests; inject a fake "
            "transport/getter instead of making a real connection"
        )

    monkeypatch.setattr(socket, "socket", _blocked_socket)

    # 3) No real DNS: the fetcher's SSRF guard enumerates every A/AAAA answer via
    #    socket.getaddrinfo. Answer it offline with a fixed PUBLIC address so
    #    URL-path tests exercise that default resolver AND pass the guard; the
    #    SSRF-specific tests inject (or patch in) their own answers to exercise
    #    the blocked cases. The single-answer lookups are blocked outright, so no
    #    code path can quietly resolve around the guard's complete answer set.
    def _blocked_lookup(*args, **kwargs):
        raise _BlockedNetworkError(
            "single-answer DNS lookups are blocked in offline unit tests; the "
            "fetcher must enumerate every answer via socket.getaddrinfo"
        )

    monkeypatch.setattr(socket, "getaddrinfo", _offline_getaddrinfo)
    monkeypatch.setattr(socket, "gethostbyname", _blocked_lookup)
    monkeypatch.setattr(socket, "gethostbyname_ex", _blocked_lookup)

    yield
