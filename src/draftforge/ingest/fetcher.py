"""URL readable-text fetcher.

``fetch_url`` retrieves a web page and extracts its main article text, stripping
navigation, ads, and other boilerplate via ``trafilatura.extract``.

The HTTP getter is *injected* (default ``requests.get``) so tests can pass a
fake that returns canned HTML or raises — no real network, no flakiness. Every
failure path (transport error, non-200 status, empty/non-article content)
raises :class:`FetchError` rather than returning empty text, so callers never
silently ingest nothing.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

import requests
import trafilatura

# Only real web schemes are fetchable — block file://, gopher://, etc. that could
# read local files or reach odd services.
_ALLOWED_SCHEMES = ("http", "https")


class FetchError(Exception):
    """Raised when a URL cannot be fetched or yields no readable article text."""


def _assert_public_url(url: str, *, resolver) -> None:
    """Reject a URL whose host resolves to a non-public address (SSRF guard).

    Validates the scheme, resolves the host, and refuses loopback / private /
    link-local (incl. the ``169.254.169.254`` cloud-metadata endpoint) / reserved
    / multicast / unspecified addresses. ``resolver`` (a ``host -> ip-string``
    callable, default :func:`socket.gethostbyname`) is injected so the guard is
    testable offline.

    Raises:
        FetchError: on a disallowed scheme, an unresolvable host, or a non-public
            resolved address.
    """
    parsed = urlparse(url)
    if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
        raise FetchError(
            f"refusing non-http(s) URL scheme {parsed.scheme!r}: {url}"
        )
    host = parsed.hostname
    if not host:
        raise FetchError(f"URL has no host: {url}")
    try:
        ip = ipaddress.ip_address(resolver(host))
    except (OSError, ValueError) as exc:
        raise FetchError(f"cannot resolve host {host!r} for {url}: {exc}") from exc
    if (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    ):
        raise FetchError(
            f"refusing to fetch non-public address {ip} for {url} (SSRF guard)"
        )


# A browser-like User-Agent so ordinary article sites (many 403 the bare
# python-requests UA — e.g. Wikipedia) actually serve the page. The per-run
# source URLs are public articles the operator chose.
_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    )
}


def fetch_url(
    url: str, *, getter=requests.get, max_chars: int = 200_000, resolver=None
) -> str:
    """Fetch ``url`` and return its main article text.

    Args:
        url: The page URL to fetch.
        getter: Callable ``(url, ...) -> response`` with ``.status_code`` and
            ``.text`` attributes (default :func:`requests.get`). Injected for
            deterministic, offline testing.
        max_chars: Upper bound on the returned text length. A giant scraped page
            is truncated to this many characters (mirrors the document loader's
            cap) so a single source can't blow the downstream prompt/token
            budget. Defaults to 200,000.
        resolver: ``host -> ip-string`` callable for the SSRF guard (default
            :func:`socket.gethostbyname`, resolved at call time so it is
            patchable/injectable in tests).

    Returns:
        The extracted, boilerplate-free article text, capped at ``max_chars``.

    Raises:
        FetchError: if the URL is non-public (SSRF guard), the request fails,
            returns a non-200 status, redirects to a non-public address, or
            yields no extractable article text.
    """
    resolve = resolver or socket.gethostbyname

    # SSRF guard: validate BEFORE any request so an internal/metadata host is
    # never even contacted.
    _assert_public_url(url, resolver=resolve)

    try:
        response = getter(url, timeout=15, headers=_REQUEST_HEADERS)
    except Exception as exc:  # connection error, timeout, anything the getter raises
        raise FetchError(f"failed to fetch {url}: {exc}") from exc

    # Defense in depth: if the server redirected, re-validate the final URL so a
    # public host can't bounce us to an internal one.
    final_url = getattr(response, "url", None)
    if final_url and final_url != url:
        _assert_public_url(final_url, resolver=resolve)

    status = getattr(response, "status_code", None)
    if status != 200:
        raise FetchError(f"unexpected status {status} fetching {url}")

    html = response.text or ""
    if not html.strip():
        raise FetchError(f"empty response body for {url}")

    extracted = trafilatura.extract(html)
    if not extracted or not extracted.strip():
        raise FetchError(f"no readable article text extracted from {url}")

    return extracted.strip()[:max_chars]
