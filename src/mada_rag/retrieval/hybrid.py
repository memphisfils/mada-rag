"""Reciprocal Rank Fusion of dense and BM25 candidates."""

from __future__ import annotations

from dataclasses import dataclass

from mada_rag.models import Chunk, RetrievalMethod, RetrievedChunk
from mada_rag.retrieval.base import Retriever


@dataclass
class _CandidateRanks:
    chunk: Chunk
    dense_rank: int | None = None
    dense_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None

    def rrf_score(self, rrf_k: int) -> float:
        score = 0.0
        if self.dense_rank is not None:
            score += 1.0 / (rrf_k + self.dense_rank)
        if self.bm25_rank is not None:
            score += 1.0 / (rrf_k + self.bm25_rank)
        return score


class HybridRetriever:
    """Fuse ranks, never incomparable raw dense and lexical scores."""

    def __init__(
        self,
        dense_retriever: Retriever,
        bm25_retriever: Retriever,
        *,
        rrf_k: int = 60,
        dense_candidates: int = 20,
        lexical_candidates: int = 20,
        default_top_k: int = 10,
    ) -> None:
        if min(rrf_k, dense_candidates, lexical_candidates, default_top_k) <= 0:
            raise ValueError("RRF and candidate limits must be positive")
        if dense_retriever.revision_id != bm25_retriever.revision_id:
            raise ValueError("dense and BM25 retrievers must share one revision")
        if tuple(chunk.chunk_id for chunk in dense_retriever.chunks) != tuple(
            chunk.chunk_id for chunk in bm25_retriever.chunks
        ):
            raise ValueError("dense and BM25 retrievers must share the same ordered corpus")
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.rrf_k = rrf_k
        self.dense_candidates = dense_candidates
        self.lexical_candidates = lexical_candidates
        self.default_top_k = default_top_k

    @property
    def revision_id(self) -> int:
        return self.dense_retriever.revision_id

    @property
    def chunks(self) -> tuple[Chunk, ...]:
        return self.dense_retriever.chunks

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
    ) -> tuple[RetrievedChunk, ...]:
        if not query.strip():
            raise ValueError("query cannot be empty")
        limit = top_k or self.default_top_k
        if limit <= 0:
            raise ValueError("top_k must be positive")

        candidates: dict[str, _CandidateRanks] = {}
        for dense_item in self.dense_retriever.retrieve(query, top_k=self.dense_candidates):
            candidates[dense_item.chunk.chunk_id] = _CandidateRanks(
                chunk=dense_item.chunk,
                dense_rank=dense_item.dense_rank,
                dense_score=dense_item.dense_score,
            )
        for lexical_item in self.bm25_retriever.retrieve(
            query,
            top_k=self.lexical_candidates,
        ):
            candidate = candidates.setdefault(
                lexical_item.chunk.chunk_id,
                _CandidateRanks(chunk=lexical_item.chunk),
            )
            candidate.bm25_rank = lexical_item.bm25_rank
            candidate.bm25_score = lexical_item.bm25_score

        ordered = sorted(
            candidates.values(),
            key=lambda item: (
                -item.rrf_score(self.rrf_k),
                min(item.dense_rank or 10**9, item.bm25_rank or 10**9),
                item.chunk.ordinal,
            ),
        )
        results: list[RetrievedChunk] = []
        for rank, fused_item in enumerate(ordered[:limit], start=1):
            rrf_score = fused_item.rrf_score(self.rrf_k)
            results.append(
                RetrievedChunk(
                    chunk=fused_item.chunk,
                    method=RetrievalMethod.HYBRID_RRF,
                    rank=rank,
                    score=rrf_score,
                    dense_rank=fused_item.dense_rank,
                    dense_score=fused_item.dense_score,
                    bm25_rank=fused_item.bm25_rank,
                    bm25_score=fused_item.bm25_score,
                    rrf_rank=rank,
                    rrf_score=rrf_score,
                )
            )
        return tuple(results)
