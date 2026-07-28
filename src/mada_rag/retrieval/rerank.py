"""Lazy injectable multilingual cross-encoder reranking."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, cast

import numpy as np

from mada_rag.models import Chunk, RetrievalMethod, RetrievedChunk
from mada_rag.retrieval.base import Retriever

DEFAULT_RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


class RerankerUnavailableError(RuntimeError):
    """Raised clearly when the optional reranker cannot be loaded."""


class _CrossEncoderModel(Protocol):
    def predict(self, sentences: Sequence[tuple[str, str]], **kwargs: object) -> object: ...


RerankerFactory = Callable[[str], _CrossEncoderModel]


def _default_factory(model_name: str) -> _CrossEncoderModel:
    from sentence_transformers import CrossEncoder

    return cast(_CrossEncoderModel, CrossEncoder(model_name))


class CrossEncoderReranker:
    """Load the cross-encoder only when a rerank request is executed."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        *,
        model_factory: RerankerFactory | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("reranker model_name cannot be empty")
        self.model_name = model_name
        self._model_factory = model_factory or _default_factory
        self._model: _CrossEncoderModel | None = None

    def rerank(
        self,
        query: str,
        candidates: tuple[RetrievedChunk, ...],
        *,
        top_k: int,
    ) -> tuple[RetrievedChunk, ...]:
        if not candidates:
            return ()
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        pairs = [(query, candidate.chunk.text) for candidate in candidates]
        try:
            raw_scores = self._get_model().predict(
                pairs,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise RerankerUnavailableError(
                f"reranker model {self.model_name!r} is unavailable"
            ) from exc
        scores = np.asarray(raw_scores, dtype=np.float64).reshape(-1)
        if len(scores) != len(candidates) or not np.all(np.isfinite(scores)):
            raise RerankerUnavailableError("reranker returned invalid scores")
        order = sorted(
            range(len(candidates)),
            key=lambda position: (-float(scores[position]), candidates[position].rank),
        )
        results: list[RetrievedChunk] = []
        for rank, position in enumerate(order[:top_k], start=1):
            source = candidates[position]
            score = float(scores[position])
            results.append(
                RetrievedChunk(
                    chunk=source.chunk,
                    method=RetrievalMethod.HYBRID_RERANK,
                    rank=rank,
                    score=score,
                    dense_rank=source.dense_rank,
                    dense_score=source.dense_score,
                    bm25_rank=source.bm25_rank,
                    bm25_score=source.bm25_score,
                    rrf_rank=source.rrf_rank,
                    rrf_score=source.rrf_score,
                    reranker_rank=rank,
                    reranker_score=score,
                )
            )
        return tuple(results)

    def _get_model(self) -> _CrossEncoderModel:
        if self._model is None:
            try:
                self._model = self._model_factory(self.model_name)
            except Exception as exc:
                raise RerankerUnavailableError(
                    f"reranker model {self.model_name!r} is unavailable"
                ) from exc
        return self._model


class RerankedRetriever:
    """Retrieve an RRF pool, then order it with a cross-encoder."""

    def __init__(
        self,
        hybrid_retriever: Retriever,
        reranker: CrossEncoderReranker,
        *,
        candidate_top_k: int = 20,
        default_top_k: int = 10,
    ) -> None:
        if candidate_top_k <= 0 or default_top_k <= 0:
            raise ValueError("reranking limits must be positive")
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.candidate_top_k = candidate_top_k
        self.default_top_k = default_top_k

    @property
    def revision_id(self) -> int:
        return self.hybrid_retriever.revision_id

    @property
    def chunks(self) -> tuple[Chunk, ...]:
        return self.hybrid_retriever.chunks

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
    ) -> tuple[RetrievedChunk, ...]:
        limit = top_k or self.default_top_k
        pool = self.hybrid_retriever.retrieve(query, top_k=self.candidate_top_k)
        return self.reranker.rerank(query, pool, top_k=limit)
