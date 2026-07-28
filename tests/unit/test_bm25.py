"""Deterministic lexical retrieval tests."""

import hashlib

import pytest

from mada_rag.generation.relevance import normalized_tokens
from mada_rag.indexing.bm25 import BM25Index, BM25IndexError
from mada_rag.models import Chunk, ChunkType, RetrievalMethod
from mada_rag.retrieval.bm25 import BM25Retriever


def make_chunk(chunk_id: str, ordinal: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        revision_id=123,
        chunk_type=ChunkType.TEXT,
        section_id="section-test",
        section_path=("Test",),
        ordinal=ordinal,
        text=text,
        token_count=len(text.split()),
        content_sha256=hashlib.sha256(text.encode()).hexdigest(),
        source_url="https://en.wikipedia.org/wiki/Madagascar",
    )


@pytest.fixture
def chunks() -> tuple[Chunk, ...]:
    return (
        make_chunk(
            "analamanga",
            0,
            "Region name Analamanga, population density per km 2: 198.0.",
        ),
        make_chunk("sava", 1, "Region name Sava, population density per km 2: 38.4."),
        make_chunk("president", 2, "President Michael Randrianirina."),
    )


def test_normalization_preserves_names_numbers_decimals_and_units() -> None:
    tokens = normalized_tokens("Analamanga: 198.0 people per km 2")

    assert {"analamanga", "198.0", "people", "per", "km", "2"} <= set(tokens)


def test_bm25_names_and_exact_numbers_rank_the_matching_row_first(
    chunks: tuple[Chunk, ...],
) -> None:
    results = BM25Index(chunks).search("Analamanga 198.0", top_k=3)

    assert results
    assert results[0][1] == 0
    assert results[0][0] > 0


def test_bm25_retriever_exposes_lexical_rank_score_and_limit(
    chunks: tuple[Chunk, ...],
) -> None:
    results = BM25Retriever(BM25Index(chunks)).retrieve("Michael Randrianirina", top_k=1)

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "president"
    assert results[0].method is RetrievalMethod.BM25
    assert results[0].rank == results[0].bm25_rank == 1
    assert results[0].score == results[0].bm25_score


def test_bm25_rejects_invalid_corpus_and_query(chunks: tuple[Chunk, ...]) -> None:
    with pytest.raises(BM25IndexError, match="empty"):
        BM25Index(())
    with pytest.raises(BM25IndexError, match="unique"):
        BM25Index((chunks[0], chunks[0]))

    index = BM25Index(chunks)
    with pytest.raises(ValueError, match="empty"):
        index.search("  ", top_k=1)
    with pytest.raises(ValueError, match="positive"):
        index.search("Analamanga", top_k=0)
