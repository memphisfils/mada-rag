"""Revision-pinned MediaWiki snapshot acquisition."""

from mada_rag.ingestion.mediawiki import (
    ALLOWED_API_URL,
    ALLOWED_PAGE_TITLE,
    DEFAULT_USER_AGENT,
    AsyncMediaWikiClient,
    MediaWikiClient,
    MediaWikiError,
    MediaWikiResponseError,
    ResolvedRevision,
    SourceBoundaryError,
    validate_api_url,
)
from mada_rag.ingestion.snapshot import (
    SnapshotConflictError,
    SnapshotError,
    SnapshotIntegrityError,
    SnapshotStore,
    ingest_snapshot,
    ingest_snapshot_async,
    sha256_hex,
)

__all__ = [
    "ALLOWED_API_URL",
    "ALLOWED_PAGE_TITLE",
    "DEFAULT_USER_AGENT",
    "AsyncMediaWikiClient",
    "MediaWikiClient",
    "MediaWikiError",
    "MediaWikiResponseError",
    "ResolvedRevision",
    "SnapshotConflictError",
    "SnapshotError",
    "SnapshotIntegrityError",
    "SnapshotStore",
    "SourceBoundaryError",
    "ingest_snapshot",
    "ingest_snapshot_async",
    "sha256_hex",
    "validate_api_url",
]
