"""URL readable-text fetcher.

``fetch_url`` retrieves a web page and extracts its main article text, stripping
navigation, ads, and other boilerplate via ``trafilatura.extract``.

**Destination guard (SSRF).** Every hop — the original URL and each redirect
target — is validated *before* anything is sent to it: ``http``/``https`` only,
no userinfo, ports 80/443 only, and every resolved address must be a public
unicast address (no loopback / private / link-local incl. the cloud-metadata
endpoint / shared / reserved / multicast / unspecified, also checked for IPv4
embedded in IPv6). The default resolver enumerates EVERY A and AAAA answer the
system resolver returns (one ``socket.getaddrinfo`` call); a single non-public,
scoped or malformed answer rejects the whole set. Redirects are never followed
automatically: a ``Location`` is resolved against the current URL, validated,
and only then requested, for at most ``max_redirects`` hops; loops are refused.

The default transport (:class:`PinnedTransport`) opens its sockets directly to
the numeric addresses that passed validation — it never looks the host name up
again — while TLS still verifies the certificate against the original host name
(SNI + hostname check) and the ``Host`` header carries the original host. It
does not use proxy environment variables, so no proxy can re-resolve the
destination. Each request has ONE absolute deadline: every blocking step
(connect, TLS handshake, send, status line, headers, chunk framing, body,
trailers) waits at most the time left until it, so a server that drips bytes
cannot stretch a response past it.

Offline-test seams: ``resolver`` (host -> address or addresses), a
:class:`PinnedTransport` with a fake ``connect`` / TLS context / ``clock``, or
any requests-style ``getter`` callable (called once per hop with
``allow_redirects=False``; such a caller-owned transport is NOT bound to the
validated addresses or the deadline). Every failure path raises
:class:`FetchError` rather than returning empty text, so callers never silently
ingest nothing; messages never include URL userinfo or query strings.
"""

from __future__ import annotations

import http.client
import io
import ipaddress
import socket
import ssl
import time
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit

import requests
import trafilatura
from requests.utils import requote_uri

# Only real web schemes on their standard ports are fetchable — block file://,
# gopher://, etc. and odd services on other ports.
_ALLOWED_SCHEMES = ("http", "https")
_DEFAULT_PORTS = {"http": 80, "https": 443}
_ALLOWED_PORTS = (80, 443)

_REDIRECT_STATUSES = (301, 302, 303, 307, 308)
_DEFAULT_MAX_REDIRECTS = 5
_MAX_URL_CHARS = 8192

_TIMEOUT_SECONDS = 15  # longest single wait (connect / send / receive)
_REQUEST_DEADLINE_SECONDS = 30  # one whole request: connect through the last byte
_MAX_BODY_BYTES = 10 * 1024 * 1024
_READ_CHUNK_BYTES = 64 * 1024

# NAT64 well-known prefix (RFC 6052): the low 32 bits are an IPv4 destination.
_NAT64_PREFIX = ipaddress.ip_network("64:ff9b::/96")


class FetchError(Exception):
    """Raised when a URL cannot be fetched or yields no readable article text."""


# A browser-like User-Agent so ordinary article sites (many 403 the bare
# python-requests UA — e.g. Wikipedia) actually serve the page. The per-run
# source URLs are public articles the operator chose.
_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    ),
    "Accept-Encoding": "identity",
    "Connection": "close",
}


def _redact(url: str) -> str:
    """``url`` for messages: no userinfo, query, or fragment (``?…`` marks a query)."""
    try:
        parts = urlsplit(url)
        host = parts.hostname or ""
        port = f":{parts.port}" if parts.port is not None else ""
    except ValueError:
        return "<malformed URL>"
    if ":" in host:
        host = f"[{host}]"
    path = parts.path[:200]
    return f"{parts.scheme}://{host}{port}{path}{'?…' if parts.query else ''}"


@dataclass(frozen=True)
class _Hop:
    """One validated request target."""

    url: str  # normalized, fragment-free URL that is requested
    scheme: str
    host: str  # ASCII host name or IP literal (no brackets)
    port: int
    target: str  # path + query for the request line


def _parse_hop(url: str) -> _Hop:
    """Validate a URL's shape (scheme, userinfo, host, port) without resolving it."""
    if not isinstance(url, str) or not url or len(url) > _MAX_URL_CHARS:
        raise FetchError("refusing malformed URL (empty, non-text, or too long)")
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in url):
        raise FetchError("refusing malformed URL (control characters)")
    try:
        parts = urlsplit(url)
        explicit_port = parts.port  # raises on a non-numeric / out-of-range port
    except ValueError:
        raise FetchError(f"refusing malformed URL {_redact(url)}") from None

    scheme = parts.scheme.lower()
    if scheme not in _ALLOWED_SCHEMES:
        raise FetchError(f"refusing non-http(s) URL scheme {parts.scheme!r}")
    if "@" in parts.netloc:
        raise FetchError(
            "refusing URL with embedded credentials (userinfo); they are never sent"
        )
    host = parts.hostname
    if not host:
        raise FetchError(f"URL has no host: {_redact(url)}")
    if any(ch.isspace() for ch in host):
        raise FetchError("refusing malformed URL (whitespace in host)")
    port = _DEFAULT_PORTS[scheme] if explicit_port is None else explicit_port
    if port not in _ALLOWED_PORTS:
        raise FetchError(
            f"refusing port {port} for {_redact(url)} (only 80 and 443 are fetched)"
        )
    try:
        ascii_host = host.encode("idna").decode("ascii")
    except UnicodeError:
        raise FetchError(f"refusing invalid host name in {_redact(url)}") from None

    path = parts.path or "/"
    target = requote_uri(f"{path}?{parts.query}" if parts.query else path)
    netloc = f"[{ascii_host}]" if ":" in ascii_host else ascii_host
    if explicit_port is not None:
        netloc = f"{netloc}:{explicit_port}"
    return _Hop(
        url=f"{scheme}://{netloc}{target}",
        scheme=scheme,
        host=ascii_host,
        port=port,
        target=target,
    )


def _non_public_reason(ip) -> str | None:
    """Why ``ip`` is not a public unicast destination, or ``None`` if it is."""
    candidates = [ip]
    if isinstance(ip, ipaddress.IPv6Address):
        # IPv4 carried inside IPv6 (mapped, 6to4, Teredo, NAT64) must pass too.
        candidates += [ip.ipv4_mapped, ip.sixtofour, *(ip.teredo or ())]
        if ip in _NAT64_PREFIX:
            candidates.append(ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF))
    for addr in candidates:
        if addr is None:
            continue
        if (
            addr.is_loopback
            or addr.is_private
            or addr.is_link_local
            or addr.is_reserved
            or addr.is_multicast
            or addr.is_unspecified
            or not addr.is_global
        ):
            return f"non-public address {addr}"
    return None


def _system_addresses(host: str) -> list:
    """Every A and AAAA answer the system resolver returns for ``host``.

    One ``socket.getaddrinfo`` call for both address families (looked up at call
    time so tests can patch it), each answer reduced to its numeric address. An
    answer of any other family or shape makes the whole reply unusable.
    """
    answers = []
    for entry in socket.getaddrinfo(host, None, socket.AF_UNSPEC, socket.SOCK_STREAM):
        try:
            family, sockaddr = entry[0], entry[4]
            address = sockaddr[0]
            version = ipaddress.ip_address(address).version
        except (TypeError, IndexError, ValueError):
            raise ValueError("malformed resolver answer") from None
        if not isinstance(address, str) or (family, version) not in (
            (socket.AF_INET, 4), (socket.AF_INET6, 6),
        ):
            raise ValueError("unexpected resolver answer family")
        answers.append(address)
    return answers


def _validated_addresses(host: str, *, resolver) -> tuple:
    """Resolve ``host`` and return its addresses, refusing if ANY is non-public.

    An IP-literal host is used as-is. ``resolver`` may return one address string
    or an iterable of them (e.g. every A/AAAA answer); a single non-public answer
    rejects the host, because an attacker who controls DNS can mix answers.
    Duplicates collapse to one, in first-seen order.
    """
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        addresses = [literal]
    else:
        try:
            answer = resolver(host)
        except (OSError, UnicodeError, ValueError) as exc:
            raise FetchError(
                f"cannot resolve host {host!r} ({type(exc).__name__})"
            ) from None
        raw = [answer] if isinstance(answer, str) else list(answer or ())
        if not raw:
            raise FetchError(f"cannot resolve host {host!r} (no addresses)")
        try:
            addresses = [ipaddress.ip_address(item) for item in raw]
        except (TypeError, ValueError):
            raise FetchError(
                f"cannot resolve host {host!r} (resolver returned a malformed address)"
            ) from None

    for ip in addresses:
        if getattr(ip, "scope_id", None) is not None:
            # A zone index (``%eth0``) is looked up by name at connect time and
            # is never a public destination.
            raise FetchError(f"refusing scoped IPv6 address for host {host!r} (SSRF guard)")
        reason = _non_public_reason(ip)
        if reason is not None:
            raise FetchError(
                f"refusing to fetch {reason} for host {host!r} (SSRF guard)"
            )
    return tuple(dict.fromkeys(addresses))


def _default_tls_context() -> ssl.SSLContext:
    """Certificate + hostname verification against the system trust store."""
    return ssl.create_default_context()


def _connect_numeric(target, timeout):
    """Open a TCP socket to a numeric ``(address, port)`` — no name lookup at all.

    The family comes from the address itself and IPv6 uses a zone-free 4-tuple.
    (``socket.create_connection`` would pass even a numeric address back through
    ``getaddrinfo``.)
    """
    address, port = target
    ip = ipaddress.ip_address(address)
    if ip.version == 6:
        family, sockaddr = socket.AF_INET6, (str(ip), port, 0, 0)
    else:
        family, sockaddr = socket.AF_INET, (str(ip), port)
    sock = socket.socket(family, socket.SOCK_STREAM)
    try:
        sock.settimeout(timeout)
        sock.connect(sockaddr)
    except BaseException:
        sock.close()
        raise
    return sock


class _Deadline:
    """One absolute deadline shared by every blocking step of a request."""

    def __init__(self, seconds: float, *, step_timeout: float, clock):
        self._clock = clock
        self._at = clock() + seconds
        self._step_timeout = step_timeout

    def remaining(self) -> float:
        """Seconds left; raises :class:`TimeoutError` once the deadline has passed."""
        left = self._at - self._clock()
        if left <= 0:
            raise TimeoutError("request deadline exceeded")
        return left

    def timeout(self) -> float:
        """Timeout for the next blocking call: the step limit, cut to the time left."""
        return min(self._step_timeout, self.remaining())


class _DeadlineSocket:
    """A connected socket whose every send/receive is bounded by a :class:`_Deadline`.

    The timeout is re-armed to the time left *before each* underlying call, so
    http.client's buffered status-line / header / chunk-size / body / trailer
    reads can never block past the deadline (an inactivity timeout alone would
    restart with every byte a slow server drips). Close semantics mirror
    :class:`socket.socket`: the real socket closes once the connection and every
    file made by :meth:`makefile` are closed.
    """

    def __init__(self, sock, deadline: _Deadline):
        self._sock = sock
        self._deadline = deadline
        self._io_refs = 0
        self._closed = False

    def _arm(self):
        self._sock.settimeout(self._deadline.timeout())

    def sendall(self, data):
        self._arm()
        return self._sock.sendall(data)

    def send(self, data):
        self._arm()
        return self._sock.send(data)

    def recv_into(self, buffer, *args):
        self._arm()
        return self._sock.recv_into(buffer, *args)

    def makefile(self, mode="r", buffering=None, **_kwargs):
        if mode != "rb":  # http.client only ever reads its response as "rb"
            raise ValueError(f"unsupported makefile mode {mode!r}")
        self._io_refs += 1
        return io.BufferedReader(socket.SocketIO(self, "rb"))

    def fileno(self):
        return self._sock.fileno()

    def _decref_socketios(self):
        if self._io_refs > 0:
            self._io_refs -= 1
        if self._closed and self._io_refs <= 0:
            self._sock.close()

    def close(self):
        self._closed = True
        if self._io_refs <= 0:
            self._sock.close()


class _PinnedHTTPConnection(http.client.HTTPConnection):
    """HTTPConnection whose socket goes to a pre-validated address, not a lookup."""

    def __init__(self, host, port, *, address, deadline, connect):
        super().__init__(host, port)
        self._pinned_address = address
        self._pinned_connect = connect
        self._pinned_deadline = deadline

    def connect(self):
        sock = self._pinned_connect(
            (self._pinned_address, self.port), self._pinned_deadline.timeout()
        )
        self.sock = _DeadlineSocket(sock, self._pinned_deadline)


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPS to a pre-validated address; SNI + certificate check use the host name."""

    def __init__(self, host, port, *, address, deadline, connect, context):
        super().__init__(host, port, context=context)
        self._pinned_address = address
        self._pinned_connect = connect
        self._pinned_context = context
        self._pinned_deadline = deadline

    def connect(self):
        sock = self._pinned_connect(
            (self._pinned_address, self.port), self._pinned_deadline.timeout()
        )
        try:
            sock.settimeout(self._pinned_deadline.timeout())  # the handshake shares it
            sock = self._pinned_context.wrap_socket(sock, server_hostname=self.host)
        except BaseException:
            sock.close()
            raise
        self.sock = _DeadlineSocket(sock, self._pinned_deadline)


@dataclass
class _Response:
    status_code: int
    headers: object
    text: str
    url: str


class PinnedTransport:
    """The default transport: one GET to a validated address, no redirects.

    ``connect`` (default: a numeric-only TCP connect, no name lookup) receives
    ``((address, port), timeout)`` — the address is always one the destination
    guard approved, so DNS cannot change between validation and connection.
    HTTPS wraps that socket with ``tls_context`` (default
    :func:`ssl.create_default_context`: certificate + hostname verification)
    using the original host for SNI and the hostname check. Proxy environment
    variables are not consulted. Each request — one hop, across every validated
    address tried — has a single absolute ``deadline`` (seconds on ``clock``,
    default :func:`time.monotonic`): every blocking step waits at most
    ``min(timeout, time left)``. The body is also capped at ``max_body_bytes``.
    """

    def __init__(
        self,
        *,
        connect=None,
        tls_context=None,
        timeout: float = _TIMEOUT_SECONDS,
        max_body_bytes: int = _MAX_BODY_BYTES,
        deadline: float = _REQUEST_DEADLINE_SECONDS,
        clock=None,
    ):
        self._connect = connect
        self._tls_context = tls_context
        self._timeout = timeout
        self._max_body_bytes = max_body_bytes
        self._deadline = deadline
        self._clock = clock

    def send(self, hop: _Hop, addresses) -> _Response:
        connect = self._connect or _connect_numeric
        deadline = _Deadline(
            self._deadline, step_timeout=self._timeout, clock=self._clock or time.monotonic
        )
        failure = None
        for address in addresses:
            conn = None
            try:
                if hop.scheme == "https":
                    conn = _PinnedHTTPSConnection(
                        hop.host, hop.port, address=str(address), deadline=deadline,
                        connect=connect, context=self._tls_context or _default_tls_context(),
                    )
                else:
                    conn = _PinnedHTTPConnection(
                        hop.host, hop.port, address=str(address), deadline=deadline,
                        connect=connect,
                    )
                conn.request("GET", hop.target, headers=_REQUEST_HEADERS)
                response = conn.getresponse()
                try:
                    return self._read(response, hop, deadline)
                finally:
                    response.close()
            except (OSError, http.client.HTTPException) as exc:
                failure = exc  # try the next validated address, if any time is left
            finally:
                if conn is not None:
                    conn.close()
        if isinstance(failure, TimeoutError):
            raise FetchError(f"timed out fetching {_redact(hop.url)}")
        raise FetchError(f"failed to fetch {_redact(hop.url)} ({type(failure).__name__})")

    def _read(self, response, hop: _Hop, deadline: _Deadline) -> _Response:
        status = response.status
        if status != 200:
            return _Response(status, response.msg, "", hop.url)
        encoding = (response.getheader("Content-Encoding") or "identity").strip().lower()
        if encoding != "identity":
            raise FetchError(
                f"unsupported content-encoding {encoding!r} from {_redact(hop.url)}"
            )
        chunks, total = [], 0
        while chunk := response.read(_READ_CHUNK_BYTES):
            total += len(chunk)
            if total > self._max_body_bytes:
                raise FetchError(
                    f"response from {_redact(hop.url)} exceeds {self._max_body_bytes} bytes"
                )
            chunks.append(chunk)
        deadline.remaining()  # a body that only completed at/after the deadline fails
        body = b"".join(chunks)
        charset = response.msg.get_content_charset() or "utf-8"
        try:
            text = body.decode(charset, errors="replace")
        except LookupError:
            text = body.decode("utf-8", errors="replace")
        return _Response(status, response.msg, text, hop.url)


def _sender(getter):
    """Map the ``getter`` argument to a ``send(hop, addresses)`` callable."""
    # ``requests.get`` (the historical default, still passed by draftforge.cli)
    # re-resolves DNS and honours proxy env vars, so it cannot enforce the
    # validated destination: it means "the real network" -> the pinned transport.
    if getter is None or getter is requests.get:
        return PinnedTransport().send
    if isinstance(getter, PinnedTransport):
        return getter.send

    def send(hop: _Hop, addresses):
        try:
            response = getter(
                hop.url, timeout=_TIMEOUT_SECONDS, headers=_REQUEST_HEADERS,
                allow_redirects=False,
            )
        except Exception as exc:  # connection error, timeout, anything the getter raises
            raise FetchError(
                f"failed to fetch {_redact(hop.url)} ({type(exc).__name__})"
            ) from None
        reported = getattr(response, "url", None)
        if reported and reported != hop.url:
            raise FetchError(
                f"refusing response for {_redact(hop.url)}: the transport followed a "
                "redirect itself (SSRF guard validates every hop before contact)"
            )
        return response

    return send


def _location(response) -> str | None:
    headers = getattr(response, "headers", None) or {}
    return headers.get("Location") or headers.get("location")


def fetch_url(
    url: str,
    *,
    getter=None,
    max_chars: int = 200_000,
    resolver=None,
    max_redirects: int = _DEFAULT_MAX_REDIRECTS,
) -> str:
    """Fetch ``url`` and return its main article text.

    Args:
        url: The page URL to fetch.
        getter: The transport. ``None`` (or ``requests.get``) uses
            :class:`PinnedTransport`; a :class:`PinnedTransport` instance (e.g.
            with a fake ``connect``) is used as-is; any other callable is a
            requests-style getter ``(url, ...) -> response`` with
            ``.status_code``, ``.text`` and, for redirects, ``.headers`` —
            injected for deterministic, offline testing.
        max_chars: Upper bound on the returned text length. A giant scraped page
            is truncated to this many characters (mirrors the document loader's
            cap) so a single source can't blow the downstream prompt/token
            budget. Defaults to 200,000.
        resolver: ``host -> address(es)`` callable for the destination guard.
            The default enumerates every A/AAAA answer through
            :func:`socket.getaddrinfo` (looked up at call time so it is
            patchable in tests).
        max_redirects: Most redirects followed before giving up (default 5).

    Returns:
        The extracted, boilerplate-free article text, capped at ``max_chars``.

    Raises:
        FetchError: if any hop is refused by the destination guard (before it is
            contacted), the request fails or misses its deadline, redirects are
            malformed / looping / too many, the final status is not 200, or no
            article text is extracted.
    """
    resolve = resolver or _system_addresses
    send = _sender(getter)

    seen = set()
    current = url
    for _ in range(max_redirects + 1):
        hop = _parse_hop(current)
        if hop.url in seen:
            raise FetchError(f"redirect loop at {_redact(hop.url)}")
        seen.add(hop.url)
        # SSRF guard: validate BEFORE any request so a refused host is never
        # contacted — including every redirect target.
        addresses = _validated_addresses(hop.host, resolver=resolve)
        response = send(hop, addresses)

        status = getattr(response, "status_code", None)
        if status not in _REDIRECT_STATUSES:
            break
        location = _location(response)
        if not isinstance(location, str) or not location.strip():
            raise FetchError(f"redirect {status} without a Location from {_redact(hop.url)}")
        if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in location):
            raise FetchError(f"refusing malformed redirect from {_redact(hop.url)}")
        try:
            current = urljoin(hop.url, location.strip())
        except ValueError:  # e.g. an unbalanced IPv6 bracket
            raise FetchError(f"refusing malformed redirect from {_redact(hop.url)}") from None
    else:
        raise FetchError(
            f"too many redirects (more than {max_redirects}) from {_redact(url)}"
        )

    if status != 200:
        raise FetchError(f"unexpected status {status} fetching {_redact(hop.url)}")

    html = response.text or ""
    if not html.strip():
        raise FetchError(f"empty response body for {_redact(hop.url)}")

    extracted = trafilatura.extract(html)
    if not extracted or not extracted.strip():
        raise FetchError(f"no readable article text extracted from {_redact(hop.url)}")

    return extracted.strip()[:max_chars]
