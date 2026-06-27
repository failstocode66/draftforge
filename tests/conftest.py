"""Suite-wide offline/no-key guard.

The whole unit suite is contractually offline: it must run with **no**
``ANTHROPIC_API_KEY`` and **no** network access, exercising the pipeline through
injected fakes (a ``FakeTransport`` for the LLM, a fake ``getter`` for HTTP). The
autouse fixture below enforces that contract so a regression that secretly reaches
the real Anthropic SDK or opens a socket fails loudly *here*, locally, instead of
silently passing in CI (or worse, only failing when a key happens to be present).

For every test that is **not** marked ``smoke`` it:

* deletes ``ANTHROPIC_API_KEY`` from the environment, so any code path that builds
  the real client / reads the key raises instead of quietly succeeding; and
* replaces ``socket.socket`` with a stub that raises on construction, so any
  accidental real outbound connection (requests, the anthropic SDK, trafilatura's
  fallback fetch, ...) blows up with a clear message.

The ``smoke`` tests (``pytest -m smoke``) are the *opt-in* live tests that legitimately
need a real key and the network, so they are exempt. Both patches are applied via
the function-scoped ``monkeypatch`` fixture, so they are reverted automatically
after each test and never leak into collection/teardown.
"""

from __future__ import annotations

import socket

import pytest


class _BlockedNetworkError(RuntimeError):
    """Raised when offline-guarded test code attempts to open a socket."""


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

    # 3) No real DNS: the fetcher's SSRF guard resolves hosts via
    #    socket.gethostbyname. Stub it to a fixed PUBLIC ip so URL-path tests stay
    #    offline AND pass the guard; the SSRF-specific tests inject their own
    #    resolver to exercise the blocked cases.
    monkeypatch.setattr(socket, "gethostbyname", lambda *_a, **_k: "93.184.216.34")

    yield
