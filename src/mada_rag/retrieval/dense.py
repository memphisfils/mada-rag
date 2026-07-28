"""Explainable dense retrieval over a validated local FAISS index."""

from __future__ import annotations

from mada_rag.indexing import DenseIndex, DenseIndexIntegrityError, EmbeddingBackend
from mada_rag.models import Chunk, RetrievalMethod, RetrievedChunk


class DenseRetriever:
    """Embed one query and expose FAISS ranks and cosine-equivalent scores."""

    def __init__(
        self,
        index: DenseIndex,
        embedder: EmbeddingBackend,
        *,
        default_top_k: int = 10,
    ) -> None:
        if default_top_k <= 0:
            raise ValueError("default_top_k must be positive")
        if embedder.model_name != index.embedding_model:
            raise DenseIndexIntegrityError(
                "retriever embedding model differs from the loaded dense index"
            )
        self.index = index
        self.embedder = embedder
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
        if not query.strip():
            raise ValueError("query cannot be empty")
        requested_top_k = top_k or self.default_top_k
        if requested_top_k <= 0:
            raise ValueError("top_k must be positive")
        query_vector = self.embedder.embed_query(query)
        matches = self.index.search(query_vector, top_k=requested_top_k)
        return tuple(
            RetrievedChunk(
                chunk=self.index.chunks[position],
                method=RetrievalMethod.DENSE,
                rank=rank,
                score=score,
                dense_rank=rank,
                dense_score=score,
            )
            for rank, (score, position) in enumerate(matches, start=1)
        )
