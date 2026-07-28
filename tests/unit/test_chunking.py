"""Deterministic section-aware and table-aware chunking tests."""

from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

from mada_rag.chunking import ArticleChunker, WhitespaceTokenizer
from mada_rag.models import (
    ChunkType,
    ParsedArticle,
    SectionRecord,
    SnapshotManifest,
    TableRecord,
)
from mada_rag.parsing import parse_article

FIXTURE_PATH = Path("tests/fixtures/madagascar_synthetic.html")
REVISION_ID = 123
SOURCE_URL = "https://en.wikipedia.org/wiki/Madagascar"


def make_manifest() -> SnapshotManifest:
    return SnapshotManifest(
        page_id=42,
        revision_id=REVISION_ID,
        parent_revision_id=122,
        revision_timestamp=datetime(2026, 7, 28, 8, 0, tzinfo=UTC),
        fetched_at=datetime(2026, 7, 28, 8, 1, tzinfo=UTC),
        canonical_url=SOURCE_URL,
        api_url="https://en.wikipedia.org/w/api.php",
        raw_html_path=Path("madagascar.html"),
        html_sha256="a" * 64,
        parser_version="test",
    )


def parse_fixture() -> ParsedArticle:
    return parse_article(FIXTURE_PATH.read_text(encoding="utf-8"), make_manifest())


def test_chunk_ids_and_payloads_are_deterministic() -> None:
    article = parse_fixture()
    chunker = ArticleChunker()

    first = chunker.chunk(article)
    second = chunker.chunk(article)

    assert first == second
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert len({chunk.chunk_id for chunk in first}) == len(first)


def test_text_chunks_never_cross_section_boundaries() -> None:
    chunks = ArticleChunker().chunk(parse_fixture())
    text_chunks = [chunk for chunk in chunks if chunk.chunk_type is ChunkType.TEXT]

    history = next(chunk for chunk in text_chunks if "fictional Alpha event" in chunk.text)
    nested = next(chunk for chunk in text_chunks if "synthetic Beta transition" in chunk.text)
    units = next(chunk for chunk in text_chunks if "12.5 fixture km" in chunk.text)

    assert history.section_path == ("Fixture history",)
    assert nested.section_path == ("Fixture history", "Nested fixture period")
    assert units.section_path == ("Synthetic regions", "Units and missing values")
    assert all(
        not ("fictional Alpha event" in chunk.text and "synthetic Beta transition" in chunk.text)
        for chunk in text_chunks
    )


def test_text_budget_and_overlap_are_enforced() -> None:
    words = tuple(f"word-{index}" for index in range(20))
    text = " ".join(words)
    section = SectionRecord(
        section_id="section-1",
        revision_id=REVISION_ID,
        title="Synthetic",
        level=1,
        path=("Synthetic",),
        ordinal=0,
        text=text,
        paragraphs=(text,),
    )
    article = ParsedArticle(
        revision_id=REVISION_ID,
        source_url=SOURCE_URL,
        sections=(section,),
    )
    tokenizer = WhitespaceTokenizer()
    chunks = ArticleChunker(
        tokenizer=tokenizer,
        target_tokens=8,
        max_tokens=8,
        overlap_tokens=2,
    ).chunk(article)

    assert len(chunks) > 1
    assert all(chunk.token_count <= 8 for chunk in chunks)
    assert all(tokenizer.count_tokens(chunk.text) == chunk.token_count for chunk in chunks)
    for left, right in pairwise(chunks):
        assert left.text.split()[-2:] == right.text.split()[:2]


def test_each_table_gets_one_full_chunk_and_one_chunk_per_row() -> None:
    article = parse_fixture()
    chunks = ArticleChunker().chunk(article)

    for table in article.tables:
        table_chunks = [chunk for chunk in chunks if chunk.table_id == table.table_id]
        full = [chunk for chunk in table_chunks if chunk.chunk_type is ChunkType.TABLE_FULL]
        rows = [chunk for chunk in table_chunks if chunk.chunk_type is ChunkType.TABLE_ROW]

        assert len(full) == 1
        assert len(rows) == len(table.rows)
        assert {chunk.row_index for chunk in rows} == set(range(len(table.rows)))
        assert all(chunk.section_id == table.section_id for chunk in table_chunks)


def test_oversized_table_is_partitioned_and_still_has_every_row_chunk() -> None:
    section = SectionRecord(
        section_id="section-1",
        revision_id=REVISION_ID,
        title="Synthetic table",
        level=1,
        path=("Synthetic table",),
        ordinal=0,
        text="",
    )
    table = TableRecord(
        table_id="table-1",
        revision_id=REVISION_ID,
        section_id=section.section_id,
        section_path=section.path,
        ordinal=0,
        caption="Synthetic table",
        headers=("Key", "Value"),
        rows=(("A", "one"), ("B", "two"), ("C", "three")),
    )
    article = ParsedArticle(
        revision_id=REVISION_ID,
        source_url=SOURCE_URL,
        sections=(section,),
        tables=(table,),
    )
    chunker = ArticleChunker(
        target_tokens=8,
        max_tokens=8,
        overlap_tokens=2,
        table_max_tokens=18,
    )
    chunks = chunker.chunk(article)
    parts = [chunk for chunk in chunks if chunk.chunk_type is ChunkType.TABLE_PART]
    rows = [chunk for chunk in chunks if chunk.chunk_type is ChunkType.TABLE_ROW]

    assert not any(chunk.chunk_type is ChunkType.TABLE_FULL for chunk in chunks)
    assert len(parts) >= 2
    assert [chunk.table_part_index for chunk in parts] == list(range(len(parts)))
    assert all(chunk.token_count <= 18 for chunk in parts)
    assert len(rows) == len(table.rows)
    assert {chunk.row_index for chunk in rows} == {0, 1, 2}
