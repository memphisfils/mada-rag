"""Source-bounded MediaWiki clients for the Madagascar article only."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any, Self
from urllib.parse import urlsplit

import httpx
from pydantic import field_validator

from mada_rag.models import NonEmptyStr, PositiveInt, StrictModel

ALLOWED_API_URL = "https://en.wikipedia.org/w/api.php"
ALLOWED_PAGE_TITLE = "Madagascar"
DEFAULT_USER_AGENT = "mada-rag/0.1.0 (https://github.com/memphisfils/mada-rag)"


class MediaWikiError(RuntimeError):
    """Base error for source resolution and download failures."""


class SourceBoundaryError(MediaWikiError):
    """Raised before a request could leave the single allowed source."""


class MediaWikiResponseError(MediaWikiError):
    """Raised when MediaWiki returns an unexpected or inconsistent payload."""


class ResolvedRevision(StrictModel):
    """Exact MediaWiki revision selected before downloading parsed HTML."""

    page_title: NonEmptyStr
    page_id: PositiveInt
    revision_id: PositiveInt
    parent_revision_id: PositiveInt | None = None
    revision_timestamp: datetime

    @field_validator("page_title")
    @classmethod
    def require_madagascar(cls, value: str) -> str:
        if value != ALLOWED_PAGE_TITLE:
            raise ValueError(f"only {ALLOWED_PAGE_TITLE!r} is allowed")
        return value

    @field_validator("revision_timestamp")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("revision timestamp must be timezone-aware")
        return value


def validate_api_url(api_url: str) -> str:
    """Return the canonical endpoint or reject any host/path variation."""

    parsed = urlsplit(api_url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "en.wikipedia.org"
        or parsed.port not in {None, 443}
        or parsed.path != "/w/api.php"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise SourceBoundaryError(f"MediaWiki endpoint is not allowlisted: {api_url!r}")
    return ALLOWED_API_URL


def _positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MediaWikiResponseError(f"{field_name} must be a positive integer")
    return value


def _optional_positive_int(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, field_name)


def _mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise MediaWikiResponseError(f"{field_name} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise MediaWikiResponseError(f"{field_name} contains a non-string key")
    return value


def _sequence(value: object, field_name: str) -> list[object]:
    if not isinstance(value, list):
        raise MediaWikiResponseError(f"{field_name} must be a list")
    return value


def _parse_timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise MediaWikiResponseError("revision timestamp must be a string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise MediaWikiResponseError("revision timestamp is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise MediaWikiResponseError("revision timestamp must be timezone-aware")
    return parsed


def _parse_revision_payload(payload: object) -> ResolvedRevision:
    root = _mapping(payload, "response")
    query = _mapping(root.get("query"), "query")
    if "redirects" in query:
        raise MediaWikiResponseError("MediaWiki redirects are forbidden")
    pages = _sequence(query.get("pages"), "query.pages")
    if len(pages) != 1:
        raise MediaWikiResponseError("query must return exactly one page")
    page = _mapping(pages[0], "query.pages[0]")
    if page.get("missing") is not None:
        raise MediaWikiResponseError("the Madagascar page is missing")
    if page.get("title") != ALLOWED_PAGE_TITLE:
        raise MediaWikiResponseError("MediaWiki resolved a page outside the allowlist")

    revisions = _sequence(page.get("revisions"), "query.pages[0].revisions")
    if len(revisions) != 1:
        raise MediaWikiResponseError("query must return exactly one current revision")
    revision = _mapping(revisions[0], "query.pages[0].revisions[0]")
    return ResolvedRevision(
        page_title=ALLOWED_PAGE_TITLE,
        page_id=_positive_int(page.get("pageid"), "pageid"),
        revision_id=_positive_int(revision.get("revid"), "revid"),
        parent_revision_id=_optional_positive_int(revision.get("parentid"), "parentid"),
        revision_timestamp=_parse_timestamp(revision.get("timestamp")),
    )


def _parse_html_payload(payload: object, revision: ResolvedRevision) -> str:
    root = _mapping(payload, "response")
    parsed = _mapping(root.get("parse"), "parse")
    if parsed.get("title") != ALLOWED_PAGE_TITLE:
        raise MediaWikiResponseError("parse response title is outside the allowlist")
    if _positive_int(parsed.get("pageid"), "parse.pageid") != revision.page_id:
        raise MediaWikiResponseError("parse response page ID changed after revision resolution")
    if _positive_int(parsed.get("revid"), "parse.revid") != revision.revision_id:
        raise MediaWikiResponseError("parse response revision differs from requested oldid")
    html = parsed.get("text")
    if not isinstance(html, str) or not html.strip():
        raise MediaWikiResponseError("parse response contains no HTML")
    return html


class _ClientConfiguration:
    def __init__(self, api_url: str, page_title: str) -> None:
        self.api_url = validate_api_url(api_url)
        if page_title != ALLOWED_PAGE_TITLE:
            raise SourceBoundaryError(f"only page {ALLOWED_PAGE_TITLE!r} is allowlisted")
        self.page_title = page_title

    @property
    def revision_params(self) -> dict[str, str]:
        return {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "prop": "revisions",
            "rvlimit": "1",
            "rvprop": "ids|timestamp",
            "titles": ALLOWED_PAGE_TITLE,
        }

    def parse_params(self, revision_id: int) -> dict[str, str]:
        return {
            "action": "parse",
            "disableeditsection": "1",
            "disabletoc": "1",
            "format": "json",
            "formatversion": "2",
            "oldid": str(revision_id),
            "prop": "text",
        }

    def validate_response_url(self, url: httpx.URL) -> None:
        parsed = urlsplit(str(url))
        response_base = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        validate_api_url(response_base)


class MediaWikiClient:
    """Synchronous two-step client: resolve a revision, then parse that oldid."""

    def __init__(
        self,
        *,
        api_url: str = ALLOWED_API_URL,
        page_title: str = ALLOWED_PAGE_TITLE,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout_seconds: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._configuration = _ClientConfiguration(api_url, page_title)
        self._owns_client = client is None
        self._client = client or httpx.Client(
            headers={"User-Agent": user_agent},
            timeout=timeout_seconds,
            follow_redirects=False,
        )

    @property
    def api_url(self) -> str:
        return self._configuration.api_url

    def _get_json(self, params: Mapping[str, str]) -> object:
        try:
            response = self._client.get(self.api_url, params=params, follow_redirects=False)
            self._configuration.validate_response_url(response.url)
            response.raise_for_status()
            payload: Any = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise MediaWikiError("MediaWiki request failed") from exc
        return payload

    def resolve_revision(self) -> ResolvedRevision:
        return _parse_revision_payload(self._get_json(self._configuration.revision_params))

    def fetch_parsed_html(self, revision: ResolvedRevision) -> str:
        payload = self._get_json(self._configuration.parse_params(revision.revision_id))
        return _parse_html_payload(payload, revision)

    def fetch_snapshot(self) -> tuple[ResolvedRevision, str]:
        revision = self.resolve_revision()
        return revision, self.fetch_parsed_html(revision)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class AsyncMediaWikiClient:
    """Asynchronous equivalent with the same source-boundary guarantees."""

    def __init__(
        self,
        *,
        api_url: str = ALLOWED_API_URL,
        page_title: str = ALLOWED_PAGE_TITLE,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout_seconds: float = 30.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._configuration = _ClientConfiguration(api_url, page_title)
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            headers={"User-Agent": user_agent},
            timeout=timeout_seconds,
            follow_redirects=False,
        )

    @property
    def api_url(self) -> str:
        return self._configuration.api_url

    async def _get_json(self, params: Mapping[str, str]) -> object:
        try:
            response = await self._client.get(
                self.api_url,
                params=params,
                follow_redirects=False,
            )
            self._configuration.validate_response_url(response.url)
            response.raise_for_status()
            payload: Any = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise MediaWikiError("MediaWiki request failed") from exc
        return payload

    async def resolve_revision(self) -> ResolvedRevision:
        payload = await self._get_json(self._configuration.revision_params)
        return _parse_revision_payload(payload)

    async def fetch_parsed_html(self, revision: ResolvedRevision) -> str:
        payload = await self._get_json(self._configuration.parse_params(revision.revision_id))
        return _parse_html_payload(payload, revision)

    async def fetch_snapshot(self) -> tuple[ResolvedRevision, str]:
        revision = await self.resolve_revision()
        return revision, await self.fetch_parsed_html(revision)

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.close()
