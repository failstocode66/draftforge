import pytest

from draftforge.ingest.fetcher import FetchError, fetch_url


class FakeResponse:
    """Minimal stand-in for a requests.Response."""

    def __init__(self, *, status_code=200, text="", url="http://example.com"):
        self.status_code = status_code
        self.text = text
        self.url = url


ARTICLE_HTML = """
<html><head><title>Float Therapy Guide</title></head>
<body>
  <nav>Home | About | Contact</nav>
  <article>
    <h1>The Benefits of Float Therapy</h1>
    <p>Floating in an Epsom-salt tank promotes deep relaxation and reduces
       stress for many clients.</p>
    <p>People often report sleeping better after a single sixty-minute float
       session in the quiet, dark tank.</p>
  </article>
  <footer>Copyright 2026 Example Floats</footer>
</body></html>
"""


def make_getter(response_or_exc):
    """Return a fake getter that yields a response or raises an exception."""
    calls = []

    def getter(url, *args, **kwargs):
        calls.append(url)
        if isinstance(response_or_exc, Exception):
            raise response_or_exc
        return response_or_exc

    getter.calls = calls
    return getter


def test_good_html_returns_clean_readable_text():
    getter = make_getter(FakeResponse(text=ARTICLE_HTML))

    text = fetch_url("http://example.com/float", getter=getter)

    assert "deep relaxation" in text
    assert "sleeping better" in text
    # Boilerplate nav/footer is stripped by the readability extractor.
    assert "Home | About | Contact" not in text
    assert "Copyright 2026" not in text
    assert getter.calls == ["http://example.com/float"]


def test_getter_raising_timeout_becomes_fetch_error():
    getter = make_getter(TimeoutError("timed out"))

    with pytest.raises(FetchError):
        fetch_url("http://example.com", getter=getter)


def test_non_200_status_becomes_fetch_error():
    getter = make_getter(FakeResponse(status_code=404, text="Not Found"))

    with pytest.raises(FetchError):
        fetch_url("http://example.com/missing", getter=getter)


def test_non_article_extraction_becomes_fetch_error():
    # A page whose markup yields no extractable text at all -> FetchError.
    # (trafilatura returns None when there is no content to extract.)
    getter = make_getter(
        FakeResponse(text="<html><head></head><body></body></html>")
    )

    with pytest.raises(FetchError):
        fetch_url("http://example.com/empty", getter=getter)


def test_empty_body_becomes_fetch_error():
    getter = make_getter(FakeResponse(text=""))

    with pytest.raises(FetchError):
        fetch_url("http://example.com/blank", getter=getter)


def _public(_host):
    return "93.184.216.34"  # a public IP, for the SSRF guard in offline tests


def test_rejects_non_http_scheme():
    with pytest.raises(FetchError, match="scheme"):
        fetch_url("file:///etc/passwd", getter=make_getter(FakeResponse()), resolver=_public)


@pytest.mark.parametrize(
    "internal_ip",
    [
        "127.0.0.1",        # loopback
        "169.254.169.254",  # cloud metadata (link-local)
        "10.0.0.5",         # private
        "192.168.1.10",     # private
        "172.16.0.9",       # private
        "0.0.0.0",          # unspecified
    ],
)
def test_rejects_internal_addresses_ssrf_guard(internal_ip):
    getter = make_getter(FakeResponse(text=ARTICLE_HTML))
    with pytest.raises(FetchError, match="SSRF|non-public"):
        fetch_url("http://internal.example", getter=getter, resolver=lambda _h: internal_ip)
    # the SSRF guard blocks BEFORE any request is made
    assert getter.calls == []


def test_allows_public_address():
    getter = make_getter(FakeResponse(text=ARTICLE_HTML))
    text = fetch_url("http://example.com/float", getter=getter, resolver=_public)
    assert "deep relaxation" in text


def test_blocks_redirect_to_internal_address():
    # The request resolves public, but the final (post-redirect) URL points at an
    # internal host — defense in depth catches it after the fetch.
    resolved = {"example.com": "93.184.216.34", "metadata.internal": "169.254.169.254"}
    getter = make_getter(FakeResponse(text=ARTICLE_HTML, url="http://metadata.internal/x"))
    with pytest.raises(FetchError, match="SSRF|non-public"):
        fetch_url("http://example.com/start", getter=getter, resolver=lambda h: resolved[h])


def test_unresolvable_host_becomes_fetch_error():
    def boom(_host):
        raise OSError("name resolution failed")

    with pytest.raises(FetchError, match="resolve"):
        fetch_url("http://nope.invalid", getter=make_getter(FakeResponse()), resolver=boom)


def test_extracted_text_is_capped_to_max_chars():
    # An extraction longer than a small max_chars is capped to that length
    # (mirrors the loader's slice, so a giant scraped page can't blow the
    # downstream prompt/token budget). The body has plenty of article text.
    long_html = (
        "<html><body><article>"
        + ("Floating in the quiet dark tank is deeply calming. " * 200)
        + "</article></body></html>"
    )
    getter = make_getter(FakeResponse(text=long_html))

    text = fetch_url("http://example.com/long", getter=getter, max_chars=100)

    assert len(text) == 100
