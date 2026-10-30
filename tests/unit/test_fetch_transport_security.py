"""Destination-guard tests at the SEND boundary (P2 SSRF remediation).

Everything is offline. Host names resolve through an injected fake resolver (or
a patched ``socket.getaddrinfo`` on the DEFAULT resolver path), and "contact" is
recorded by either

* ``FakeWeb`` — a ``requests.get`` stand-in that, like requests, follows
  redirects ITSELF unless called with ``allow_redirects=False``. A fetcher that
  relies on automatic redirects therefore contacts every hop here, and the
  assertions on ``web.contacted`` catch it (the baseline control proves it does);
* ``FakeConnector`` / ``FakeTLSContext`` — stand-ins for the transport's
  ``connect`` seam and TLS context, recording the exact ``(address, port)`` each
  socket is opened to, the SNI host name, and the request bytes; or
* ``FakeNet`` — a ``socket.socket`` stand-in under the DEFAULT numeric connector,
  recording each socket's address family and the exact sockaddr it connects to.

Fake sockets deliver scripted bytes on a fake clock and honour ``settimeout``
like a blocking socket, while the transport's real ``socket.SocketIO`` and
``http.client`` do all the HTTP parsing — so the deadline tests exercise the
stdlib's own status-line / header / chunk / trailer reads.

Assertions are about who was CONTACTED, not only about what fetch_url returned.
All host names are reserved/example names and all addresses are synthetic.
"""

from __future__ import annotations

import socket
import ssl
from urllib.parse import urljoin, urlsplit

import pytest
import requests

from draftforge.ingest import fetcher
from draftforge.ingest.fetcher import FetchError, PinnedTransport, fetch_url

PUBLIC_IP = "93.184.216.34"
PUBLIC_IP_2 = "93.184.216.35"
PUBLIC_V6 = "2606:2800:220:1:248:1893:25c8:1946"
REDIRECTS = (301, 302, 303, 307, 308)

ARTICLE_HTML = (
    "<html><head><title>Float notes</title></head><body><nav>Home | About</nav>"
    "<article><h1>The Benefits of Float Therapy</h1>"
    "<p>Floating in an Epsom-salt tank promotes deep relaxation and reduces "
    "stress for many clients.</p>"
    "<p>People often report sleeping better after a single sixty-minute float "
    "session in the quiet, dark tank.</p></article></body></html>"
)


# --- fakes ----------------------------------------------------------------------------


class FakeResolver:
    """host -> scripted answer; records every lookup. Unknown hosts fail loudly."""

    def __init__(self, answers):
        self.answers = answers
        self.calls = []

    def __call__(self, host):
        self.calls.append(host)
        answer = self.answers[host]
        return answer() if callable(answer) else answer


class FakeAddrInfo:
    """``socket.getaddrinfo`` stand-in: host -> list of 5-tuples, or an exception."""

    def __init__(self, answers):
        self.answers = answers
        self.calls = []

    def __call__(self, host, port, family=0, type=0, proto=0, flags=0):
        self.calls.append((host, family))
        answer = self.answers[host]
        if isinstance(answer, Exception):
            raise answer
        return list(answer)


def a(address):
    """One getaddrinfo A answer."""
    return (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (address, 0))


def aaaa(address):
    """One getaddrinfo AAAA answer."""
    return (socket.AF_INET6, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (address, 0, 0, 0))


class FakeResponse:
    def __init__(self, status_code, *, text="", headers=None, url=None):
        self.status_code = status_code
        self.text = text
        self.headers = headers or {}
        self.url = url


class FakeWeb:
    """requests.get stand-in over scripted routes ``url -> (status, payload)``.

    For a redirect status the payload is the Location. Like requests, it follows
    redirects itself unless ``allow_redirects=False``; every URL it touches is
    appended to ``contacted``.
    """

    def __init__(self, routes):
        self.routes = routes
        self.contacted = []

    def __call__(self, url, *args, allow_redirects=True, **kwargs):
        for _ in range(20):
            self.contacted.append(url)
            status, payload = self.routes.get(url, (404, ""))
            if status in REDIRECTS:
                if not allow_redirects:
                    return FakeResponse(status, headers={"Location": payload})
                url = urljoin(url, payload)
                continue
            return FakeResponse(status, text=payload, url=url)
        raise AssertionError("FakeWeb redirect limit")


def http_response(status=200, reason="OK", headers=(), body=b""):
    head = [f"HTTP/1.1 {status} {reason}", *(f"{k}: {v}" for k, v in headers),
            f"Content-Length: {len(body)}", "", ""]
    return "\r\n".join(head).encode("latin-1") + body


def html_ok(html=ARTICLE_HTML):
    return http_response(
        headers=[("Content-Type", "text/html; charset=utf-8")], body=html.encode()
    )


class FakeClock:
    """A monotonic clock that only moves when a fake socket makes it wait."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class Drip:
    """A response delivered as ``(delay_seconds, bytes)`` steps on a FakeClock."""

    def __init__(self, *deliveries):
        self.deliveries = deliveries


def _deliveries(planned):
    if isinstance(planned, Drip):
        return list(planned.deliveries)
    return [(0, planned)] if planned else []


class FakeSocket:
    """A connected socket fed by scripted ``(delay_seconds, bytes)`` deliveries.

    It honours ``settimeout`` the way a blocking socket does, against a fake
    clock: a delivery due after the armed timeout instead advances the clock by
    exactly that timeout and raises ``TimeoutError``. Every armed timeout is
    recorded in ``timeouts``.
    """

    def __init__(self, planned=b"", *, clock=None, net=None, family=None):
        self._pending = _deliveries(planned)
        self.clock = clock or FakeClock()
        self.net = net
        self.family = family
        self.sent = b""
        self.timeouts = []
        self.timeout = None
        self.closed = False

    def settimeout(self, value):
        self.timeout = value
        self.timeouts.append(value)

    def connect(self, sockaddr):  # only reached through FakeNet (the default connector)
        self.net.opened.append((self.family, sockaddr))
        planned = self.net.script[sockaddr[0]]
        if isinstance(planned, Exception):
            raise planned
        self._pending = _deliveries(planned)

    def sendall(self, data):
        assert not self.closed
        self.sent += bytes(data)

    def recv_into(self, buffer, *args):
        assert not self.closed
        if not self._pending:
            return 0  # EOF
        delay, data = self._pending[0]
        if self.timeout is not None and delay > self.timeout:
            self.clock.now += self.timeout
            self._pending[0] = (delay - self.timeout, data)
            raise TimeoutError("timed out")
        self.clock.now += delay
        n = min(len(buffer), len(data))
        buffer[:n] = data[:n]
        if data[n:]:
            self._pending[0] = (0, data[n:])
        else:
            self._pending.pop(0)
        return n

    def close(self):
        self.closed = True


class FakeConnector:
    """The transport's ``connect`` seam: ``(address, port) -> bytes | Drip | Exception``.

    Each scripted value is a list consumed in order (or a single value reused).
    Records every attempted ``(address, port)`` and connect timeout; unscripted
    targets fail loudly.
    """

    def __init__(self, script, clock=None):
        self.script = {k: (list(v) if isinstance(v, list) else v) for k, v in script.items()}
        self.clock = clock or FakeClock()
        self.calls = []
        self.timeouts = []
        self.sockets = []

    def __call__(self, address, timeout=None, *args, **kwargs):
        self.calls.append(address)
        self.timeouts.append(timeout)
        planned = self.script[address]
        item = planned.pop(0) if isinstance(planned, list) else planned
        if isinstance(item, Exception):
            raise item
        sock = FakeSocket(item, clock=self.clock)
        self.sockets.append(sock)
        return sock


class FakeNet:
    """``socket.socket`` stand-in under the DEFAULT numeric connector.

    Records ``(family, sockaddr)`` for every connect; ``script`` maps a numeric
    address to response bytes or an exception. A name target fails loudly.
    """

    def __init__(self, script):
        self.script = script
        self.opened = []
        self.sockets = []

    def __call__(self, family=-1, type=-1, *args, **kwargs):
        sock = FakeSocket(net=self, family=family)
        self.sockets.append(sock)
        return sock


class FakeTLSContext:
    def __init__(self):
        self.server_hostnames = []

    def wrap_socket(self, sock, server_hostname=None, **kwargs):
        self.server_hostnames.append(server_hostname)
        return sock


# --- redirect matrix at the getter boundary (no rejected hop is contacted) ----------

# name -> (start URL, routes). NON_PUBLIC_HOSTS is the independent judge of which
# contacts are leaks; EXPECTED_BASELINE_LEAKS is exactly the scenarios in which a
# transport that follows redirects itself reaches one of them.
REDIRECT_SCENARIOS = {
    "public-to-metadata": ("http://pub.example/start", {
        "http://pub.example/start": (302, "http://metadata.example/latest/meta-data"),
        "http://metadata.example/latest/meta-data": (200, ARTICLE_HTML),
    }),
    "public-public-private": ("http://pub.example/1", {
        "http://pub.example/1": (301, "http://pub2.example/2"),
        "http://pub2.example/2": (302, "http://10.1.2.3/3"),
        "http://10.1.2.3/3": (200, ARTICLE_HTML),
    }),
    "private-intermediate": ("http://pub.example/start", {
        "http://pub.example/start": (307, "http://192.168.7.7/bounce"),
        "http://192.168.7.7/bounce": (302, "http://pub2.example/final"),
        "http://pub2.example/final": (200, ARTICLE_HTML),
    }),
    "scheme-relative-loopback": ("http://pub.example/start", {
        "http://pub.example/start": (302, "//loopback.example/x"),
        "http://loopback.example/x": (200, ARTICLE_HTML),
    }),
    "public-relative": ("http://pub.example/start", {
        "http://pub.example/start": (302, "/article?id=7"),
        "http://pub.example/article?id=7": (200, ARTICLE_HTML),
    }),
    "public-to-public": ("http://pub.example/start", {
        "http://pub.example/start": (308, "http://pub2.example/final"),
        "http://pub2.example/final": (200, ARTICLE_HTML),
    }),
}
SCENARIO_DNS = {
    "pub.example": PUBLIC_IP, "pub2.example": PUBLIC_IP_2,
    "metadata.example": "169.254.169.254", "loopback.example": "127.0.0.1",
}
NON_PUBLIC_HOSTS = {"metadata.example", "10.1.2.3", "192.168.7.7", "loopback.example"}
EXPECTED_BASELINE_LEAKS = {
    "public-to-metadata", "public-public-private", "private-intermediate",
    "scheme-relative-loopback",
}


def _leaks(contacted):
    return [url for url in contacted if urlsplit(url).hostname in NON_PUBLIC_HOSTS]


def test_redirect_baseline_control_blind_following_leaks_exactly_the_expected_scenarios():
    # Positive control for the FakeWeb instrument: following redirects blindly
    # (requests' default) contacts a non-public host in EXACTLY these scenarios —
    # no fewer (the instrument sees leaks) and no more (the table is as intended).
    leaking = set()
    for name, (start, routes) in REDIRECT_SCENARIOS.items():
        web = FakeWeb(routes)
        web(start)  # allow_redirects defaults to True
        if _leaks(web.contacted):
            leaking.add(name)

    assert leaking == EXPECTED_BASELINE_LEAKS


@pytest.mark.parametrize("name", sorted(REDIRECT_SCENARIOS))
def test_guarded_fetch_contacts_no_non_public_host_in_any_baseline_scenario(name):
    start, routes = REDIRECT_SCENARIOS[name]
    web = FakeWeb(routes)

    try:
        fetch_url(start, getter=web, resolver=FakeResolver(SCENARIO_DNS))
        outcome = "fetched"
    except FetchError:
        outcome = "refused"

    assert _leaks(web.contacted) == []
    assert outcome == ("refused" if name in EXPECTED_BASELINE_LEAKS else "fetched")


def test_public_to_internal_redirect_never_contacts_internal_host():
    web = FakeWeb({
        "http://pub.example/start": (302, "http://metadata.example/latest/meta-data"),
        "http://metadata.example/latest/meta-data": (200, ARTICLE_HTML),
    })
    resolver = FakeResolver({"pub.example": PUBLIC_IP, "metadata.example": "169.254.169.254"})

    with pytest.raises(FetchError, match="SSRF"):
        fetch_url("http://pub.example/start", getter=web, resolver=resolver)

    assert web.contacted == ["http://pub.example/start"]


def test_public_public_internal_chain_stops_before_internal_hop():
    web = FakeWeb({
        "http://pub.example/1": (301, "http://pub2.example/2"),
        "http://pub2.example/2": (302, "http://10.1.2.3/3"),
        "http://10.1.2.3/3": (200, ARTICLE_HTML),
    })
    resolver = FakeResolver({"pub.example": PUBLIC_IP, "pub2.example": PUBLIC_IP_2})

    with pytest.raises(FetchError, match="SSRF"):
        fetch_url("http://pub.example/1", getter=web, resolver=resolver)

    assert web.contacted == ["http://pub.example/1", "http://pub2.example/2"]


def test_internal_intermediate_is_blocked_even_when_chain_ends_public():
    web = FakeWeb({
        "http://pub.example/start": (307, "http://192.168.7.7/bounce"),
        "http://192.168.7.7/bounce": (302, "http://pub2.example/final"),
        "http://pub2.example/final": (200, ARTICLE_HTML),
    })
    resolver = FakeResolver({"pub.example": PUBLIC_IP, "pub2.example": PUBLIC_IP_2})

    with pytest.raises(FetchError, match="SSRF"):
        fetch_url("http://pub.example/start", getter=web, resolver=resolver)

    assert web.contacted == ["http://pub.example/start"]


def test_public_relative_redirect_to_public_succeeds():
    web = FakeWeb({
        "http://pub.example/start": (302, "/article?id=7"),
        "http://pub.example/article?id=7": (200, ARTICLE_HTML),
    })
    resolver = FakeResolver({"pub.example": PUBLIC_IP})

    text = fetch_url("http://pub.example/start", getter=web, resolver=resolver)

    assert "deep relaxation" in text
    assert web.contacted == ["http://pub.example/start", "http://pub.example/article?id=7"]


def test_redirect_loop_is_refused_without_recontacting():
    web = FakeWeb({
        "http://pub.example/a": (302, "/b"),
        "http://pub.example/b": (302, "/a"),
    })
    resolver = FakeResolver({"pub.example": PUBLIC_IP})

    with pytest.raises(FetchError, match="loop"):
        fetch_url("http://pub.example/a", getter=web, resolver=resolver)

    assert web.contacted == ["http://pub.example/a", "http://pub.example/b"]


def test_redirect_bound_is_enforced():
    web = FakeWeb({f"http://pub.example/{i}": (302, f"/{i + 1}") for i in range(10)})
    resolver = FakeResolver({"pub.example": PUBLIC_IP})

    with pytest.raises(FetchError, match="too many redirects"):
        fetch_url("http://pub.example/0", getter=web, resolver=resolver, max_redirects=2)

    assert web.contacted == [f"http://pub.example/{i}" for i in range(3)]


@pytest.mark.parametrize(
    "location",
    ["http://[::1", "http://pub2.example:99999/x", "/x\r\nX-Injected: 1", "", None],
    ids=["bad-ipv6-bracket", "bad-port", "crlf", "empty", "missing"],
)
def test_malformed_redirects_are_refused_before_any_further_contact(location):
    web = FakeWeb({"http://pub.example/start": (302, location)})
    resolver = FakeResolver({"pub.example": PUBLIC_IP, "pub2.example": PUBLIC_IP_2})

    with pytest.raises(FetchError):
        fetch_url("http://pub.example/start", getter=web, resolver=resolver)

    assert web.contacted == ["http://pub.example/start"]


@pytest.mark.parametrize(
    "location",
    [
        "file:///etc/hosts",
        "gopher://pub2.example/x",
        "http://fixture-user:fixture-pw@pub2.example/x",
        "http://pub2.example:22/x",
        "https://pub2.example:8443/x",
        "//loopback.example/x",
    ],
    ids=["file", "gopher", "userinfo", "port-22", "port-8443", "scheme-relative-loopback"],
)
def test_disallowed_redirect_targets_are_never_contacted(location):
    web = FakeWeb({"http://pub.example/start": (302, location)})
    resolver = FakeResolver({
        "pub.example": PUBLIC_IP, "pub2.example": PUBLIC_IP_2, "loopback.example": "127.0.0.1",
    })

    with pytest.raises(FetchError) as excinfo:
        fetch_url("http://pub.example/start", getter=web, resolver=resolver)

    assert web.contacted == ["http://pub.example/start"]
    assert "fixture-pw" not in str(excinfo.value)


@pytest.mark.parametrize(
    "url",
    [
        "http://fixture-user:fixture-pw@pub.example/a",
        "ftp://pub.example/a",
        "http://pub.example:8080/a",
        "http://pub.example:0/a",
        "http://pub.example:notaport/a",
        "http:///no-host",
    ],
)
def test_initial_url_shape_is_refused_before_resolving_or_contact(url):
    web = FakeWeb({})
    resolver = FakeResolver({"pub.example": PUBLIC_IP})

    with pytest.raises(FetchError) as excinfo:
        fetch_url(url, getter=web, resolver=resolver)

    assert resolver.calls == []
    assert web.contacted == []
    assert "fixture-pw" not in str(excinfo.value)
    assert "fixture-user" not in str(excinfo.value)


@pytest.mark.parametrize(
    "host",
    [
        "127.0.0.1", "[::1]", "[::ffff:127.0.0.1]", "[::ffff:a9fe:a9fe]",
        "[fe80::1]", "[fc00::1]", "[2002:7f00:1::1]", "[64:ff9b::a00:1]",
        "100.64.0.1", "224.0.0.1", "255.255.255.255", "0.0.0.0", "169.254.169.254",
    ],
)
def test_non_public_ip_literals_are_refused_without_contact(host):
    web = FakeWeb({})
    resolver = FakeResolver({})

    with pytest.raises(FetchError, match="SSRF"):
        fetch_url(f"http://{host}/a", getter=web, resolver=resolver)

    assert web.contacted == []
    assert resolver.calls == []  # literals are checked as-is, never "resolved" away


@pytest.mark.parametrize("host", ["2130706433", "0x7f000001", "017700000001", "127.1"])
def test_alternate_ipv4_notations_are_judged_by_resolved_address(host):
    # A system resolver (inet_aton rules) maps these spellings to 127.0.0.1; the
    # guard judges the resolved address, not the string.
    web = FakeWeb({})
    resolver = FakeResolver({host: "127.0.0.1"})

    with pytest.raises(FetchError, match="SSRF"):
        fetch_url(f"http://{host}/a", getter=web, resolver=resolver)

    assert web.contacted == []


@pytest.mark.parametrize(
    "answers",
    [
        [PUBLIC_IP, "10.0.0.8"],
        [PUBLIC_V6, "::1"],
        [PUBLIC_IP, "fd00::1"],
        ["::ffff:192.168.0.1", PUBLIC_IP],
    ],
)
def test_any_non_public_answer_in_a_mixed_dns_reply_rejects_the_host(answers):
    web = FakeWeb({"http://mixed.example/a": (200, ARTICLE_HTML)})
    resolver = FakeResolver({"mixed.example": answers})

    with pytest.raises(FetchError, match="SSRF"):
        fetch_url("http://mixed.example/a", getter=web, resolver=resolver)

    assert web.contacted == []


def test_malformed_resolver_answer_is_refused():
    web = FakeWeb({})
    resolver = FakeResolver({"odd.example": ["not-an-address"]})

    with pytest.raises(FetchError, match="resolve"):
        fetch_url("http://odd.example/a", getter=web, resolver=resolver)

    assert web.contacted == []


def test_transport_that_follows_redirects_itself_is_refused():
    def follows_itself(url, *args, **kwargs):
        return FakeResponse(200, text=ARTICLE_HTML, url="http://elsewhere.example/x")

    with pytest.raises(FetchError, match="SSRF"):
        fetch_url(
            "http://pub.example/a", getter=follows_itself,
            resolver=FakeResolver({"pub.example": PUBLIC_IP}),
        )


# --- the DEFAULT resolver: every A/AAAA answer, then numeric-only connects ----------


def test_offline_fixture_serves_dns_through_getaddrinfo_and_blocks_other_lookups():
    # Control for tests/conftest.py: the default resolver path is live (answered
    # offline), while single-answer lookups and real sockets are blocked.
    assert fetcher._system_addresses("any.example") == [PUBLIC_IP]
    with pytest.raises(RuntimeError, match="blocked"):
        socket.gethostbyname("any.example")
    with pytest.raises(RuntimeError, match="blocked"):
        socket.socket()


@pytest.mark.parametrize(
    "answers",
    [
        [a(PUBLIC_IP), aaaa("::1")],
        [aaaa(PUBLIC_V6), a("10.0.0.8")],
        [a(PUBLIC_IP), a(PUBLIC_IP_2), aaaa("fd00::1")],
        [aaaa("fe80::1"), aaaa(PUBLIC_V6), a(PUBLIC_IP)],
        [a(PUBLIC_IP), aaaa("::ffff:169.254.169.254")],
        [aaaa(PUBLIC_V6), aaaa("64:ff9b::a00:1")],
        [a(PUBLIC_IP), a("100.64.0.1")],
    ],
    ids=[
        "A-public+AAAA-loopback", "AAAA-public+A-private", "A-public-x2+AAAA-ula",
        "AAAA-linklocal-first", "AAAA-mapped-metadata", "AAAA-nat64-private", "A-shared",
    ],
)
def test_default_resolver_rejects_the_whole_set_if_any_a_or_aaaa_is_non_public(
    monkeypatch, answers
):
    lookups = FakeAddrInfo({"mixed.example": answers})
    monkeypatch.setattr(socket, "getaddrinfo", lookups)
    net = FakeNet({})
    monkeypatch.setattr(socket, "socket", net)

    with pytest.raises(FetchError, match="SSRF"):
        fetch_url("http://mixed.example/a")

    assert lookups.calls == [("mixed.example", socket.AF_UNSPEC)]  # both families asked
    assert net.opened == []


def test_default_path_resolves_once_then_connects_only_to_numeric_validated_addresses(
    monkeypatch,
):
    lookups = FakeAddrInfo({"dual.example": [aaaa(PUBLIC_V6), a(PUBLIC_IP), a(PUBLIC_IP)]})
    monkeypatch.setattr(socket, "getaddrinfo", lookups)
    net = FakeNet({PUBLIC_V6: ConnectionRefusedError("refused"), PUBLIC_IP: html_ok()})
    monkeypatch.setattr(socket, "socket", net)

    text = fetch_url("http://dual.example/a")

    assert "deep relaxation" in text
    # One lookup of the name; the connects never resolve again (a numeric address
    # handed to socket.create_connection would reach getaddrinfo a second time).
    assert lookups.calls == [("dual.example", socket.AF_UNSPEC)]
    # The duplicate A answer collapses; the refused AAAA fails over to the A.
    assert net.opened == [
        (socket.AF_INET6, (PUBLIC_V6, 80, 0, 0)),
        (socket.AF_INET, (PUBLIC_IP, 80)),
    ]
    assert b"\r\nHost: dual.example\r\n" in net.sockets[1].sent
    assert all(sock.closed for sock in net.sockets)


def test_default_resolver_supports_ipv6_only_hosts_over_https(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", FakeAddrInfo({"v6only.example": [aaaa(PUBLIC_V6)]}))
    net = FakeNet({PUBLIC_V6: html_ok()})
    monkeypatch.setattr(socket, "socket", net)
    tls = FakeTLSContext()

    text = fetch_url("https://v6only.example/a", getter=PinnedTransport(tls_context=tls))

    assert "deep relaxation" in text
    assert net.opened == [(socket.AF_INET6, (PUBLIC_V6, 443, 0, 0))]
    assert tls.server_hostnames == ["v6only.example"]
    assert b"\r\nHost: v6only.example\r\n" in net.sockets[0].sent


def test_default_resolver_duplicates_collapse_to_one_connection_attempt(monkeypatch):
    monkeypatch.setattr(
        socket, "getaddrinfo",
        FakeAddrInfo({"dup.example": [a(PUBLIC_IP), a(PUBLIC_IP), a(PUBLIC_IP)]}),
    )
    net = FakeNet({PUBLIC_IP: ConnectionRefusedError("refused")})
    monkeypatch.setattr(socket, "socket", net)

    with pytest.raises(FetchError, match="ConnectionRefusedError"):
        fetch_url("http://dup.example/a")

    assert net.opened == [(socket.AF_INET, (PUBLIC_IP, 80))]


@pytest.mark.parametrize(
    "answer",
    [
        [],
        [a("not-an-address")],
        [a(PUBLIC_V6)],
        [a(PUBLIC_IP), (9999, socket.SOCK_STREAM, 0, "", (PUBLIC_IP_2, 0))],
        [a(PUBLIC_IP), ("truncated",)],
        [aaaa(PUBLIC_V6 + "%7")],
        socket.gaierror(8, "fixture: no such name"),
    ],
    ids=[
        "empty", "not-an-address", "family-mismatch", "unknown-family", "truncated",
        "scoped", "lookup-error",
    ],
)
def test_default_resolver_refuses_empty_failed_or_malformed_answers(monkeypatch, answer):
    monkeypatch.setattr(socket, "getaddrinfo", FakeAddrInfo({"odd.example": answer}))
    net = FakeNet({})
    monkeypatch.setattr(socket, "socket", net)

    with pytest.raises(FetchError, match="resolve|scoped"):
        fetch_url("http://odd.example/a")

    assert net.opened == []


@pytest.mark.parametrize("getter", [None, requests.get], ids=["default", "requests.get"])
def test_default_and_requests_get_route_through_the_pinned_transport(monkeypatch, getter):
    # Nothing injected: the conftest's offline getaddrinfo answers the DEFAULT
    # resolver, and the DEFAULT connector opens a numeric socket to that answer.
    net = FakeNet({PUBLIC_IP: html_ok()})
    monkeypatch.setattr(socket, "socket", net)

    text = fetch_url("http://pub.example/a", getter=getter)

    assert "deep relaxation" in text
    assert net.opened == [(socket.AF_INET, (PUBLIC_IP, 80))]


# --- PinnedTransport: the socket goes to the validated address ----------------------


def test_pinned_http_connects_to_validated_address_with_original_host_header():
    connector = FakeConnector({(PUBLIC_IP, 80): html_ok()})
    resolver = FakeResolver({"pub.example": PUBLIC_IP})

    text = fetch_url(
        "http://pub.example/a?x=1", getter=PinnedTransport(connect=connector), resolver=resolver
    )

    assert "deep relaxation" in text
    assert connector.calls == [(PUBLIC_IP, 80)]
    sent = connector.sockets[0].sent
    assert sent.startswith(b"GET /a?x=1 HTTP/1.1\r\n")
    assert b"\r\nHost: pub.example\r\n" in sent
    assert connector.sockets[0].closed


def test_dns_change_between_validation_and_send_cannot_move_the_connection():
    answers = iter([PUBLIC_IP, "127.0.0.1", "127.0.0.1"])
    resolver = FakeResolver({"rebind.example": lambda: next(answers)})
    connector = FakeConnector({(PUBLIC_IP, 80): html_ok()})

    text = fetch_url(
        "http://rebind.example/a", getter=PinnedTransport(connect=connector), resolver=resolver
    )

    assert "deep relaxation" in text
    assert resolver.calls == ["rebind.example"]  # resolved once, then pinned
    assert connector.calls == [(PUBLIC_IP, 80)]


def test_pinned_redirect_to_internal_host_never_opens_a_socket_to_it():
    connector = FakeConnector({
        (PUBLIC_IP, 80): http_response(302, "Found", [("Location", "http://internal.example/x")]),
    })
    resolver = FakeResolver({"pub.example": PUBLIC_IP, "internal.example": "10.9.9.9"})

    with pytest.raises(FetchError, match="SSRF"):
        fetch_url("http://pub.example/a", getter=PinnedTransport(connect=connector), resolver=resolver)

    assert connector.calls == [(PUBLIC_IP, 80)]


def test_pinned_relative_redirect_reconnects_only_to_validated_addresses():
    connector = FakeConnector({
        (PUBLIC_IP, 80): [http_response(301, "Moved", [("Location", "/b")]), html_ok()],
    })
    resolver = FakeResolver({"pub.example": PUBLIC_IP})

    text = fetch_url("http://pub.example/a", getter=PinnedTransport(connect=connector), resolver=resolver)

    assert "deep relaxation" in text
    assert connector.calls == [(PUBLIC_IP, 80), (PUBLIC_IP, 80)]
    assert connector.sockets[1].sent.startswith(b"GET /b HTTP/1.1\r\n")


def test_https_keeps_original_host_for_sni_and_host_header():
    connector = FakeConnector({(PUBLIC_IP, 443): html_ok()})
    tls = FakeTLSContext()
    resolver = FakeResolver({"secure.example": PUBLIC_IP})

    fetch_url(
        "https://secure.example/a",
        getter=PinnedTransport(connect=connector, tls_context=tls), resolver=resolver,
    )

    assert connector.calls == [(PUBLIC_IP, 443)]
    assert tls.server_hostnames == ["secure.example"]
    assert b"\r\nHost: secure.example\r\n" in connector.sockets[0].sent


def test_default_tls_context_verifies_certificate_and_hostname():
    ctx = fetcher._default_tls_context()

    assert ctx.verify_mode == ssl.CERT_REQUIRED
    assert ctx.check_hostname is True


def test_pinned_https_uses_the_verifying_default_context_when_none_injected(monkeypatch):
    tls = FakeTLSContext()
    monkeypatch.setattr(fetcher, "_default_tls_context", lambda: tls)
    connector = FakeConnector({(PUBLIC_IP, 443): html_ok()})

    fetch_url(
        "https://secure.example/a", getter=PinnedTransport(connect=connector),
        resolver=FakeResolver({"secure.example": PUBLIC_IP}),
    )

    assert tls.server_hostnames == ["secure.example"]


def test_proxy_environment_variables_are_not_used(monkeypatch):
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy"):
        monkeypatch.setenv(name, "http://proxy.example:3128")
    connector = FakeConnector({(PUBLIC_IP, 80): html_ok()})

    fetch_url(
        "http://pub.example/a", getter=PinnedTransport(connect=connector),
        resolver=FakeResolver({"pub.example": PUBLIC_IP}),
    )

    assert connector.calls == [(PUBLIC_IP, 80)]


def test_ipv6_answer_is_the_connection_target_and_host_header_is_the_name():
    connector = FakeConnector({(PUBLIC_V6, 80): html_ok()})

    fetch_url(
        "http://v6.example/a", getter=PinnedTransport(connect=connector),
        resolver=FakeResolver({"v6.example": [PUBLIC_V6]}),
    )

    assert connector.calls == [(PUBLIC_V6, 80)]
    assert b"\r\nHost: v6.example\r\n" in connector.sockets[0].sent


def test_ipv6_literal_url_connects_to_literal_with_bracketed_host_header():
    connector = FakeConnector({(PUBLIC_V6, 80): html_ok()})

    fetch_url(f"http://[{PUBLIC_V6}]/a", getter=PinnedTransport(connect=connector),
              resolver=FakeResolver({}))

    assert connector.calls == [(PUBLIC_V6, 80)]
    assert f"\r\nHost: [{PUBLIC_V6}]\r\n".encode() in connector.sockets[0].sent


def test_connection_failover_only_tries_validated_addresses():
    connector = FakeConnector({
        (PUBLIC_IP, 80): ConnectionRefusedError("refused"),
        (PUBLIC_IP_2, 80): html_ok(),
    })

    text = fetch_url(
        "http://multi.example/a", getter=PinnedTransport(connect=connector),
        resolver=FakeResolver({"multi.example": [PUBLIC_IP, PUBLIC_IP_2]}),
    )

    assert "deep relaxation" in text
    assert connector.calls == [(PUBLIC_IP, 80), (PUBLIC_IP_2, 80)]


def test_pinned_body_is_size_capped():
    connector = FakeConnector({(PUBLIC_IP, 80): html_ok("<p>" + "x" * 5000 + "</p>")})

    with pytest.raises(FetchError, match="exceeds"):
        fetch_url(
            "http://pub.example/a",
            getter=PinnedTransport(connect=connector, max_body_bytes=1024),
            resolver=FakeResolver({"pub.example": PUBLIC_IP}),
        )

    assert connector.sockets[0].closed


def test_pinned_refuses_compressed_bodies_it_did_not_ask_for():
    connector = FakeConnector({
        (PUBLIC_IP, 80): http_response(headers=[("Content-Encoding", "gzip")], body=b"\x1f\x8b"),
    })

    with pytest.raises(FetchError, match="content-encoding"):
        fetch_url(
            "http://pub.example/a", getter=PinnedTransport(connect=connector),
            resolver=FakeResolver({"pub.example": PUBLIC_IP}),
        )


@pytest.mark.parametrize("transport", ["pinned", "getter"])
def test_failure_messages_omit_query_strings(transport):
    if transport == "pinned":
        getter = PinnedTransport(connect=FakeConnector({(PUBLIC_IP, 80): ConnectionRefusedError()}))
    else:
        def getter(url, *args, **kwargs):
            raise TimeoutError(f"timed out talking to {url}")

    with pytest.raises(FetchError) as excinfo:
        fetch_url(
            "http://pub.example/a?session=fixture-token-value", getter=getter,
            resolver=FakeResolver({"pub.example": PUBLIC_IP}),
        )

    message = str(excinfo.value)
    assert "pub.example" in message
    assert "fixture-token-value" not in message
    # No chained exception (whose text could carry the URL) is shown in tracebacks.
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None or excinfo.value.__suppress_context__


# --- one absolute deadline through every blocking read (stdlib HTTP parsing) --------

HEAD_CL5 = b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: 5\r\n\r\n"
CHUNKED_HEAD = (
    b"HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\n"
    b"Transfer-Encoding: chunked\r\n\r\n"
)

# Every step arrives inside the 15 s step timeout, so an inactivity timeout that
# restarts with each drip would accept all of these — each only after 30 s.
SLOW_DRIPS = {
    "status-line": Drip((12, b"HTTP/1.1 "), (12, b"200 OK"), (12, b"\r\nContent-Length: 1\r\n\r\nx")),
    "headers": Drip(
        (0, b"HTTP/1.1 200 OK\r\n"), (12, b"Content-Type: text/html\r\n"),
        (12, b"X-Slow: 1\r\n"), (12, b"Content-Length: 1\r\n\r\nx"),
    ),
    "body": Drip((0, HEAD_CL5), (12, b"a"), (12, b"b"), (12, b"c"), (12, b"d"), (12, b"e")),
    "chunk-size": Drip(
        (0, CHUNKED_HEAD), (12, b"1"), (12, b"0"), (12, b"\r\n" + b"x" * 16 + b"\r\n0\r\n\r\n"),
    ),
    "chunk-data": Drip(
        (0, CHUNKED_HEAD + b"a\r\n"), (12, b"0123"), (12, b"4567"), (12, b"89\r\n0\r\n\r\n"),
    ),
    "trailer": Drip(
        (0, CHUNKED_HEAD + b"5\r\nhello\r\n0\r\n"), (12, b"X-T1: 1\r\n"),
        (12, b"X-T2: 2\r\n"), (12, b"\r\n"),
    ),
}


def deadline_transport(connector, clock, **kwargs):
    return PinnedTransport(connect=connector, timeout=15, deadline=30, clock=clock, **kwargs)


@pytest.mark.parametrize("phase", sorted(SLOW_DRIPS))
def test_slow_drip_cannot_stretch_a_request_past_its_fixed_deadline(phase):
    clock = FakeClock()
    connector = FakeConnector({(PUBLIC_IP, 80): SLOW_DRIPS[phase]}, clock=clock)

    with pytest.raises(FetchError, match="timed out") as excinfo:
        fetch_url(
            "http://pub.example/a?session=fixture-token-value",
            getter=deadline_transport(connector, clock),
            resolver=FakeResolver({"pub.example": PUBLIC_IP}),
        )

    # Gave up exactly at the 30 s deadline: the last wait was cut to the 6 s left.
    assert clock.now == 30
    sock = connector.sockets[0]
    assert sock.timeouts[-1] == 6
    assert max(sock.timeouts) <= 15
    assert sock.closed
    message = str(excinfo.value)
    assert "pub.example" in message and "fixture-token-value" not in message
    assert excinfo.value.__cause__ is None


def test_chunked_response_inside_the_deadline_is_parsed_by_the_stdlib_and_succeeds():
    # Control for the drip instrument: the same fake clock/socket and the real
    # http.client chunked parser deliver the article when the steps fit.
    body = ARTICLE_HTML.encode()
    half = len(body) // 2
    clock = FakeClock()
    connector = FakeConnector({(PUBLIC_IP, 80): Drip(
        (0, CHUNKED_HEAD),
        (8, b"%x\r\n" % half + body[:half] + b"\r\n"),
        (8, b"%x\r\n" % (len(body) - half) + body[half:] + b"\r\n"),
        (8, b"0\r\nX-Trailer: done\r\n\r\n"),
    )}, clock=clock)

    text = fetch_url(
        "http://pub.example/a", getter=deadline_transport(connector, clock),
        resolver=FakeResolver({"pub.example": PUBLIC_IP}),
    )

    assert "deep relaxation" in text
    assert clock.now == 24
    assert connector.sockets[0].closed


def test_a_body_that_completes_only_at_the_deadline_is_not_a_success():
    clock = FakeClock()
    connector = FakeConnector({(PUBLIC_IP, 80): Drip(
        (0, HEAD_CL5), (12, b"ab"), (12, b"cd"), (6, b"e"),
    )}, clock=clock)

    with pytest.raises(FetchError, match="timed out"):
        fetch_url(
            "http://pub.example/a", getter=deadline_transport(connector, clock),
            resolver=FakeResolver({"pub.example": PUBLIC_IP}),
        )

    assert clock.now == 30


def test_the_deadline_spans_failover_so_no_address_is_contacted_after_it():
    clock = FakeClock()
    connector = FakeConnector({
        (PUBLIC_IP, 80): Drip((0, HEAD_CL5), (12, b"a"), (12, b"b"), (12, b"c")),
        (PUBLIC_IP_2, 80): html_ok(),
    }, clock=clock)

    with pytest.raises(FetchError, match="timed out"):
        fetch_url(
            "http://multi.example/a", getter=deadline_transport(connector, clock),
            resolver=FakeResolver({"multi.example": [PUBLIC_IP, PUBLIC_IP_2]}),
        )

    assert connector.calls == [(PUBLIC_IP, 80)]
    assert clock.now == 30


def test_connect_and_tls_handshake_waits_are_capped_by_the_time_left():
    clock = FakeClock()
    connector = FakeConnector({(PUBLIC_IP, 443): html_ok()}, clock=clock)
    tls = FakeTLSContext()

    fetch_url(
        "https://secure.example/a",
        getter=PinnedTransport(connect=connector, tls_context=tls, timeout=15, deadline=10,
                               clock=clock),
        resolver=FakeResolver({"secure.example": PUBLIC_IP}),
    )

    assert connector.timeouts == [10]
    assert connector.sockets[0].timeouts[0] == 10  # armed before the TLS handshake
    assert tls.server_hostnames == ["secure.example"]
