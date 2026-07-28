"""Explainable lexical retrieval over the local chunk corpus."""

from __future__ import annotations

from mada_rag.indexing.bm25 import BM25Index
from mada_rag.models import Chunk, RetrievalMethod, RetrievedChunk


class BM25Retriever:
    def __init__(self, index: BM25Index, *, default_top_k: int = 20) -> None:
        if default_top_k <= 0:
            raise ValueError("default_top_k must be positive")
        self.index = index
        self.default_top_k = default_top_k

    @property
    def revision_id(self) -> int:
        return self.index.revision_id

    @property
    def chunks(self) -> tuple[Chunk, ...]:
        return self.index.chunks

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
    ) -> tuple[RetrievedChunk, ...]:
        requested_top_k = top_k or self.default_top_k
        matches = self.index.search(query, top_k=requested_top_k)
        return tuple(
            RetrievedChunk(
                chunk=self.index.chunks[position],
                method=RetrievalMethod.BM25,
                rank=rank,
                score=score,
                bm25_rank=rank,
                bm25_score=score,
            )
            for rank, (score, position) in enumerate(matches, start=1)
        )
