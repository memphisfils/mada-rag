"""Offline contract tests for the source-bounded MediaWiki client."""

from collections.abc import Callable

import httpx
import pytest

from mada_rag.ingestion import (
    ALLOWED_API_URL,
    MediaWikiClient,
    MediaWikiError,
    MediaWikiResponseError,
    SourceBoundaryError,
)

REVISION_ID = 123
PAGE_ID = 42


def revision_payload() -> dict[str, object]:
    return {
        "query": {
            "pages": [
                {
                    "pageid": PAGE_ID,
                    "title": "Madagascar",
                    "revisions": [
                        {
                            "revid": REVISION_ID,
                            "parentid": 122,
                            "timestamp": "2026-07-28T08:00:00Z",
                        }
                    ],
                }
            ]
        }
    }


def parse_payload(**overrides: object) -> dict[str, object]:
    parsed: dict[str, object] = {
        "title": "Madagascar",
        "pageid": PAGE_ID,
        "revid": REVISION_ID,
        "text": '<div class="mw-parser-output"><p>Synthetic HTML.</p></div>',
    }
    parsed.update(overrides)
    return {"parse": parsed}


def client_with_handler(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    follow_redirects: bool = False,
) -> tuple[MediaWikiClient, httpx.Client]:
    transport = httpx.MockTransport(handler)
    raw_client = httpx.Client(
        transport=transport,
        follow_redirects=follow_redirects,
    )
    return MediaWikiClient(client=raw_client), raw_client


@pytest.mark.parametrize(
    "constructor_kwargs",
    [
        {"api_url": "https://example.invalid/w/api.php"},
        {"page_title": "Another page"},
    ],
)
def test_allowlist_is_checked_before_any_request(
    constructor_kwargs: dict[str, str],
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(500)

    raw_client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(SourceBoundaryError):
            MediaWikiClient(client=raw_client, **constructor_kwargs)
        assert requests == []
    finally:
        raw_client.close()


def test_fetch_snapshot_uses_exactly_two_calls_and_pins_oldid() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        action = request.url.params["action"]
        payload = revision_payload() if action == "query" else parse_payload()
        return httpx.Response(200, json=payload, request=request)

    client, raw_client = client_with_handler(handler)
    try:
        revision, html = client.fetch_snapshot()
    finally:
        raw_client.close()

    assert len(requests) == 2
    assert [request.url.params["action"] for request in requests] == ["query", "parse"]
    assert requests[0].url.params["titles"] == "Madagascar"
    assert "redirects" not in requests[0].url.params
    assert requests[1].url.params["oldid"] == str(revision.revision_id)
    assert revision.revision_id == REVISION_ID
    assert html == '<div class="mw-parser-output"><p>Synthetic HTML.</p></div>'


def test_default_user_agent_contains_contact_url_and_is_sent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []
    real_client = httpx.Client

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=revision_payload(), request=request)

    def client_factory(**kwargs: object) -> httpx.Client:
        return real_client(
            transport=httpx.MockTransport(handler),
            headers=kwargs["headers"],
            timeout=kwargs["timeout"],
            follow_redirects=kwargs["follow_redirects"],
        )

    monkeypatch.setattr("mada_rag.ingestion.mediawiki.httpx.Client", client_factory)
    with MediaWikiClient() as client:
        client.resolve_revision()

    assert len(requests) == 1
    user_agent = requests[0].headers["User-Agent"]
    assert user_agent.startswith("mada-rag/")
    assert "https://github.com/memphisfils/mada-rag" in user_agent


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"query": {"pages": []}},
        {
            "query": {
                "pages": [
                    {
                        "pageid": PAGE_ID,
                        "title": "Another page",
                        "revisions": [
                            {
                                "revid": REVISION_ID,
                                "timestamp": "2026-07-28T08:00:00Z",
                            }
                        ],
                    }
                ]
            }
        },
        {
            "query": {
                "pages": [
                    {
                        "pageid": PAGE_ID,
                        "title": "Madagascar",
                        "revisions": [
                            {
                                "revid": REVISION_ID,
                                "timestamp": "not-a-timestamp",
                            }
                        ],
                    }
                ]
            }
        },
    ],
)
def test_invalid_revision_payloads_are_rejected(payload: object) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload, request=request)

    client, raw_client = client_with_handler(handler)
    try:
        with pytest.raises(MediaWikiResponseError):
            client.resolve_revision()
    finally:
        raw_client.close()


@pytest.mark.parametrize(
    "overrides",
    [
        {"title": "Another page"},
        {"pageid": PAGE_ID + 1},
        {"revid": REVISION_ID + 1},
        {"text": ""},
        {"text": None},
    ],
)
def test_invalid_parse_payloads_are_rejected(overrides: dict[str, object]) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        payload = revision_payload() if calls == 1 else parse_payload(**overrides)
        return httpx.Response(200, json=payload, request=request)

    client, raw_client = client_with_handler(handler)
    try:
        with pytest.raises(MediaWikiResponseError):
            client.fetch_snapshot()
    finally:
        raw_client.close()

    assert calls == 2


def test_mediawiki_redirect_payload_is_rejected() -> None:
    payload = revision_payload()
    query = payload["query"]
    assert isinstance(query, dict)
    query["redirects"] = [{"from": "Madagascar", "to": "Another page"}]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload, request=request)

    client, raw_client = client_with_handler(handler)
    try:
        with pytest.raises(MediaWikiResponseError, match="redirect"):
            client.resolve_revision()
    finally:
        raw_client.close()


def test_http_redirect_cannot_trigger_request_to_another_host() -> None:
    requested_hosts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_hosts.append(request.url.host)
        if request.url.host == "en.wikipedia.org":
            return httpx.Response(
                302,
                headers={"Location": "https://example.invalid/redirected"},
                request=request,
            )
        return httpx.Response(200, json=revision_payload(), request=request)

    client, raw_client = client_with_handler(handler, follow_redirects=True)
    try:
        with pytest.raises((MediaWikiError, SourceBoundaryError)):
            client.resolve_revision()
    finally:
        raw_client.close()

    assert requested_hosts == ["en.wikipedia.org"]


def test_http_error_is_wrapped_without_network() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, request=request)

    client, raw_client = client_with_handler(handler)
    try:
        with pytest.raises(MediaWikiError, match="request failed"):
            client.resolve_revision()
    finally:
        raw_client.close()

    assert client.api_url == ALLOWED_API_URL
