"""Filesystem integrity tests for immutable snapshot persistence."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from mada_rag.ingestion import (
    ResolvedRevision,
    SnapshotConflictError,
    SnapshotIntegrityError,
    SnapshotStore,
    sha256_hex,
)

REVISION_ID = 123
HTML = '<div class="mw-parser-output"><p>Synthetic snapshot.</p></div>'
FETCHED_AT = datetime(2026, 7, 28, 8, 1, tzinfo=UTC)


def make_revision(**overrides: object) -> ResolvedRevision:
    data: dict[str, object] = {
        "page_title": "Madagascar",
        "page_id": 42,
        "revision_id": REVISION_ID,
        "parent_revision_id": 122,
        "revision_timestamp": datetime(2026, 7, 28, 8, 0, tzinfo=UTC),
    }
    data.update(overrides)
    return ResolvedRevision.model_validate(data)


def test_save_load_and_hash_round_trip(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)
    saved = store.save(make_revision(), HTML, fetched_at=FETCHED_AT)

    loaded, html = store.load_verified()

    assert loaded == saved
    assert html == HTML
    assert loaded.raw_html_path == Path("madagascar.html")
    assert loaded.html_sha256 == sha256_hex(HTML.encode())
    assert store.html_path.read_text(encoding="utf-8") == HTML
    assert store.manifest_path.is_file()


def test_tampered_html_is_detected(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)
    store.save(make_revision(), HTML, fetched_at=FETCHED_AT)
    store.html_path.write_text(f"{HTML}tampered", encoding="utf-8")

    with pytest.raises(SnapshotIntegrityError, match="SHA-256"):
        store.load_verified()


@pytest.mark.parametrize("present_file", ["html", "manifest"])
def test_partial_snapshot_is_rejected(tmp_path: Path, present_file: str) -> None:
    store = SnapshotStore(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    target = store.html_path if present_file == "html" else store.manifest_path
    target.write_text(HTML if present_file == "html" else "{}", encoding="utf-8")

    with pytest.raises(SnapshotIntegrityError, match="must both exist"):
        store.load_verified()


def test_different_existing_snapshot_is_a_conflict(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)
    store.save(make_revision(), HTML, fetched_at=FETCHED_AT)

    with pytest.raises(SnapshotConflictError, match="will not be overwritten"):
        store.save(
            make_revision(),
            f"{HTML}\n<!-- different -->",
            fetched_at=FETCHED_AT,
        )


def test_same_snapshot_save_is_idempotent_with_explicit_timestamp(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)
    first = store.save(make_revision(), HTML, fetched_at=FETCHED_AT)
    second = store.save(make_revision(), HTML, fetched_at=FETCHED_AT)

    assert second == first
    assert store.load_verified() == (first, HTML)


def test_same_snapshot_save_is_idempotent_with_default_timestamp(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)
    first = store.save(make_revision(), HTML)
    second = store.save(make_revision(), HTML)

    assert second == first
    assert store.load_verified() == (first, HTML)


def test_existing_partial_snapshot_blocks_save_without_overwrite(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    store.html_path.write_text("partial", encoding="utf-8")

    with pytest.raises(SnapshotConflictError, match="will not be overwritten"):
        store.save(make_revision(), HTML, fetched_at=FETCHED_AT)

    assert store.html_path.read_text(encoding="utf-8") == "partial"
    assert not store.manifest_path.exists()
