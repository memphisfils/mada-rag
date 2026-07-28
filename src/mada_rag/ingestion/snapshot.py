"""Atomic persistence and offline integrity checks for the sole snapshot."""

from __future__ import annotations

import hashlib
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from pydantic import AnyHttpUrl

from mada_rag.ingestion.mediawiki import (
    ALLOWED_API_URL,
    ALLOWED_PAGE_TITLE,
    AsyncMediaWikiClient,
    MediaWikiClient,
    ResolvedRevision,
    validate_api_url,
)
from mada_rag.models import SnapshotManifest

CANONICAL_URL = "https://en.wikipedia.org/wiki/Madagascar"


class SnapshotError(RuntimeError):
    """Base error for local snapshot persistence."""


class SnapshotConflictError(SnapshotError):
    """Raised when existing files would be overwritten or silently changed."""


class SnapshotIntegrityError(SnapshotError):
    """Raised when the manifest and immutable HTML no longer agree."""


def sha256_hex(content: bytes) -> str:
    """Return a lowercase SHA-256 digest."""

    return hashlib.sha256(content).hexdigest()


class SnapshotStore:
    """Store an HTML/manifest pair and treat the manifest as the commit marker."""

    def __init__(
        self,
        directory: Path,
        *,
        html_filename: str = "madagascar.html",
        manifest_filename: str = "manifest.json",
    ) -> None:
        if Path(html_filename).name != html_filename:
            raise ValueError("html_filename must be a plain filename")
        if Path(manifest_filename).name != manifest_filename:
            raise ValueError("manifest_filename must be a plain filename")
        self.directory = directory
        self.html_filename = html_filename
        self.manifest_filename = manifest_filename

    @property
    def html_path(self) -> Path:
        return self.directory / self.html_filename

    @property
    def manifest_path(self) -> Path:
        return self.directory / self.manifest_filename

    def save(
        self,
        revision: ResolvedRevision,
        html: str,
        *,
        api_url: str = ALLOWED_API_URL,
        canonical_url: str = CANONICAL_URL,
        parser_version: str = "1.0",
        fetched_at: datetime | None = None,
    ) -> SnapshotManifest:
        """Persist once; an identical existing pair is idempotent, any drift fails."""

        validate_api_url(api_url)
        if canonical_url != CANONICAL_URL:
            raise SnapshotIntegrityError("canonical URL is outside the single-source boundary")
        if revision.page_title != ALLOWED_PAGE_TITLE:
            raise SnapshotIntegrityError("revision title is outside the single-source boundary")
        if not html.strip():
            raise SnapshotIntegrityError("snapshot HTML cannot be empty")

        html_bytes = html.encode("utf-8")
        html_digest = sha256_hex(html_bytes)
        timestamp = fetched_at or datetime.now(UTC)
        manifest = SnapshotManifest(
            page_id=revision.page_id,
            revision_id=revision.revision_id,
            parent_revision_id=revision.parent_revision_id,
            revision_timestamp=revision.revision_timestamp,
            fetched_at=timestamp,
            canonical_url=AnyHttpUrl(canonical_url),
            api_url=AnyHttpUrl(api_url),
            raw_html_path=Path(self.html_filename),
            html_sha256=html_digest,
            parser_version=parser_version,
        )

        if self.html_path.exists() or self.manifest_path.exists():
            if self.html_path.exists() and self.manifest_path.exists():
                existing_manifest, existing_html = self.load_verified()
                same_manifest = existing_manifest.model_dump(exclude={"fetched_at"}) == (
                    manifest.model_dump(exclude={"fetched_at"})
                )
                same_timestamp = fetched_at is None or existing_manifest.fetched_at == fetched_at
                if same_manifest and same_timestamp and existing_html == html:
                    return existing_manifest
            raise SnapshotConflictError("snapshot files already exist and will not be overwritten")

        self.directory.mkdir(parents=True, exist_ok=True)
        manifest_bytes = (manifest.model_dump_json(indent=2) + "\n").encode("utf-8")
        created_html = False
        try:
            self._atomic_create(self.html_path, html_bytes)
            created_html = True
            self._atomic_create(self.manifest_path, manifest_bytes)
        except Exception:
            if created_html and not self.manifest_path.exists():
                self.html_path.unlink(missing_ok=True)
            raise
        return manifest

    def load_verified(self) -> tuple[SnapshotManifest, str]:
        """Load the local pair without network and verify all boundary metadata."""

        if not self.manifest_path.is_file() or not self.html_path.is_file():
            raise SnapshotIntegrityError("snapshot HTML and manifest must both exist")
        try:
            manifest = SnapshotManifest.model_validate_json(
                self.manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as exc:
            raise SnapshotIntegrityError("snapshot manifest is invalid") from exc

        if manifest.page_title != ALLOWED_PAGE_TITLE or manifest.source_count != 1:
            raise SnapshotIntegrityError("manifest violates the single-source boundary")
        if str(manifest.canonical_url) != CANONICAL_URL:
            raise SnapshotIntegrityError("manifest canonical URL is not allowlisted")
        try:
            validate_api_url(str(manifest.api_url))
        except RuntimeError as exc:
            raise SnapshotIntegrityError("manifest API URL is not allowlisted") from exc
        if manifest.raw_html_path.is_absolute() or manifest.raw_html_path.parts != (
            self.html_filename,
        ):
            raise SnapshotIntegrityError("manifest HTML path escapes its snapshot directory")

        try:
            html_bytes = self.html_path.read_bytes()
            html = html_bytes.decode("utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise SnapshotIntegrityError("snapshot HTML is unreadable UTF-8") from exc
        if sha256_hex(html_bytes) != manifest.html_sha256:
            raise SnapshotIntegrityError("snapshot HTML SHA-256 differs from the manifest")
        return manifest, html

    def _atomic_create(self, target: Path, content: bytes) -> None:
        """Atomically link a fully flushed temporary file without replacing target."""

        file_descriptor, temporary_name = tempfile.mkstemp(
            dir=self.directory,
            prefix=f".{target.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(file_descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary_path, target)
            except FileExistsError as exc:
                raise SnapshotConflictError(f"{target.name} already exists") from exc
        finally:
            temporary_path.unlink(missing_ok=True)


def ingest_snapshot(
    client: MediaWikiClient,
    store: SnapshotStore,
    *,
    parser_version: str = "1.0",
) -> SnapshotManifest:
    """Resolve and download exactly one revision, then persist it locally."""

    revision, html = client.fetch_snapshot()
    return store.save(
        revision,
        html,
        api_url=client.api_url,
        parser_version=parser_version,
    )


async def ingest_snapshot_async(
    client: AsyncMediaWikiClient,
    store: SnapshotStore,
    *,
    parser_version: str = "1.0",
) -> SnapshotManifest:
    """Asynchronous network acquisition with synchronous atomic persistence."""

    revision, html = await client.fetch_snapshot()
    return store.save(
        revision,
        html,
        api_url=client.api_url,
        parser_version=parser_version,
    )
