"""Exact Reciprocal Rank Fusion and reranker behavior."""

import hashlib
from dataclasses import dataclass, field

import pytest

from mada_rag.models import Chunk, ChunkType, RetrievalMethod, RetrievedChunk
from mada_rag.retrieval.hybrid import HybridRetriever
from mada_rag.retrieval.rerank import CrossEncoderReranker, RerankerUnavailableError


def make_chunk(chunk_id: str, ordinal: int) -> Chunk:
    text = f"Evidence for {chunk_id}."
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


def dense(chunk: Chunk, rank: int, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=chunk,
        method=RetrievalMethod.DENSE,
        rank=rank,
        score=score,
        dense_rank=rank,
        dense_score=score,
    )


def bm25(chunk: Chunk, rank: int, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=chunk,
        method=RetrievalMethod.BM25,
        rank=rank,
        score=score,
        bm25_rank=rank,
        bm25_score=score,
    )


@dataclass
class StubRetriever:
    chunks: tuple[Chunk, ...]
    results: tuple[RetrievedChunk, ...]
    revision_id: int = 123
    calls: list[tuple[str, int | None]] = field(default_factory=list)

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
    ) -> tuple[RetrievedChunk, ...]:
        self.calls.append((query, top_k))
        return self.results[:top_k]


@pytest.fixture
def corpus() -> tuple[Chunk, ...]:
    return tuple(make_chunk(chunk_id, ordinal) for ordinal, chunk_id in enumerate(("a", "b", "c")))


def test_rrf_exact_scores_deduplication_and_provenance(corpus: tuple[Chunk, ...]) -> None:
    dense_retriever = StubRetriever(corpus, (dense(corpus[0], 1, 0.91), dense(corpus[1], 2, 0.82)))
    lexical_retriever = StubRetriever(
        corpus,
        (bm25(corpus[1], 1, 8.0), bm25(corpus[2], 2, 4.0)),
    )
    retriever = HybridRetriever(
        dense_retriever,
        lexical_retriever,
        rrf_k=60,
        dense_candidates=2,
        lexical_candidates=2,
    )

    results = retriever.retrieve("query", top_k=3)

    assert [item.chunk.chunk_id for item in results] == ["b", "a", "c"]
    assert len({item.chunk.chunk_id for item in results}) == 3
    assert results[0].rrf_score == pytest.approx(1 / 62 + 1 / 61)
    assert results[1].rrf_score == pytest.approx(1 / 61)
    assert results[2].rrf_score == pytest.approx(1 / 62)
    assert results[0].method is RetrievalMethod.HYBRID_RRF
    assert results[0].dense_rank == 2
    assert results[0].dense_score == 0.82
    assert results[0].bm25_rank == 1
    assert results[0].bm25_score == 8.0
    assert [item.rrf_rank for item in results] == [1, 2, 3]


def test_hybrid_rejects_mismatched_revision_and_ordered_corpus(
    corpus: tuple[Chunk, ...],
) -> None:
    base = StubRetriever(corpus, ())
    with pytest.raises(ValueError, match="revision"):
        HybridRetriever(base, StubRetriever(corpus, (), revision_id=999))
    with pytest.raises(ValueError, match="ordered corpus"):
        HybridRetriever(base, StubRetriever(tuple(reversed(corpus)), ()))


class PredictingModel:
    def __init__(self, scores: object) -> None:
        self.scores = scores
        self.calls: list[tuple[list[tuple[str, str]], dict[str, object]]] = []

    def predict(self, sentences: list[tuple[str, str]], **kwargs: object) -> object:
        self.calls.append((sentences, kwargs))
        return self.scores


def test_reranker_is_lazy_cached_and_preserves_rrf_provenance(
    corpus: tuple[Chunk, ...],
) -> None:
    source = (
        RetrievedChunk(
            chunk=corpus[0],
            method=RetrievalMethod.HYBRID_RRF,
            rank=1,
            score=0.03,
            dense_rank=1,
            dense_score=0.8,
            rrf_rank=1,
            rrf_score=0.03,
        ),
        RetrievedChunk(
            chunk=corpus[1],
            method=RetrievalMethod.HYBRID_RRF,
            rank=2,
            score=0.02,
            bm25_rank=1,
            bm25_score=5.0,
            rrf_rank=2,
            rrf_score=0.02,
        ),
    )
    model = PredictingModel([0.1, 0.9])
    factory_calls: list[str] = []

    def factory(model_name: str) -> PredictingModel:
        factory_calls.append(model_name)
        return model

    reranker = CrossEncoderReranker("local-test-model", model_factory=factory)
    assert factory_calls == []
    assert reranker.rerank("query", (), top_k=2) == ()
    assert factory_calls == []

    results = reranker.rerank("query", source, top_k=2)
    reranker.rerank("second query", source, top_k=1)

    assert factory_calls == ["local-test-model"]
    assert [item.chunk.chunk_id for item in results] == ["b", "a"]
    assert results[0].method is RetrievalMethod.HYBRID_RERANK
    assert results[0].reranker_rank == 1
    assert results[0].reranker_score == 0.9
    assert results[0].bm25_rank == 1
    assert results[0].rrf_rank == 2
    assert model.calls[0][0] == [
        ("query", corpus[0].text),
        ("query", corpus[1].text),
    ]
    assert model.calls[0][1] == {"show_progress_bar": False}


@pytest.mark.parametrize("scores", [[0.1], [float("nan"), 0.2]])
def test_reranker_fails_closed_on_invalid_scores(
    corpus: tuple[Chunk, ...],
    scores: list[float],
) -> None:
    candidates = (
        RetrievedChunk(
            chunk=corpus[0],
            method=RetrievalMethod.HYBRID_RRF,
            rank=1,
            score=0.03,
            dense_rank=1,
            dense_score=0.8,
            rrf_rank=1,
            rrf_score=0.03,
        ),
        RetrievedChunk(
            chunk=corpus[1],
            method=RetrievalMethod.HYBRID_RRF,
            rank=2,
            score=0.02,
            bm25_rank=1,
            bm25_score=5.0,
            rrf_rank=2,
            rrf_score=0.02,
        ),
    )
    reranker = CrossEncoderReranker(
        "local-test-model",
        model_factory=lambda _name: PredictingModel(scores),
    )

    with pytest.raises(RerankerUnavailableError, match="invalid scores"):
        reranker.rerank("query", candidates, top_k=2)


def test_reranker_wraps_model_load_failure() -> None:
    def failing_factory(_name: str) -> PredictingModel:
        raise OSError("offline")

    reranker = CrossEncoderReranker("missing", model_factory=failing_factory)
    dummy = dense(make_chunk("only", 0), 1, 0.8).model_copy(
        update={
            "method": RetrievalMethod.HYBRID_RRF,
            "rrf_rank": 1,
            "rrf_score": 0.03,
        }
    )

    with pytest.raises(RerankerUnavailableError, match="'missing' is unavailable"):
        reranker.rerank("query", (dummy,), top_k=1)
