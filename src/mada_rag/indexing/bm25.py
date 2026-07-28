"""Lexical BM25 over normalized chunks with names, numbers, and units retained."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from mada_rag.generation.relevance import normalized_tokens
from mada_rag.models import Chunk


class BM25IndexError(RuntimeError):
    """Raised when a lexical index or query is invalid."""


class BM25Index:
    """Small in-memory BM25 index rebuilt deterministically from local chunks."""

    def __init__(self, chunks: tuple[Chunk, ...]) -> None:
        if not chunks:
            raise BM25IndexError("BM25 corpus cannot be empty")
        revision_id = chunks[0].revision_id
        if any(chunk.revision_id != revision_id for chunk in chunks):
            raise BM25IndexError("BM25 chunks must share one revision")
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise BM25IndexError("BM25 chunk IDs must be unique")
        tokenized = [list(normalized_tokens(chunk.text)) for chunk in chunks]
        if any(not tokens for tokens in tokenized):
            raise BM25IndexError("BM25 chunks must contain at least one lexical token")

        from rank_bm25 import BM25Okapi

        self.chunks = chunks
        self.revision_id = revision_id
        self._index: Any = BM25Okapi(tokenized)

    def search(self, query: str, *, top_k: int) -> tuple[tuple[float, int], ...]:
        if not query.strip():
            raise ValueError("query cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query_tokens = list(normalized_tokens(query))
        if not query_tokens:
            return ()
        raw_scores: Sequence[float] = self._index.get_scores(query_tokens)
        scores = np.asarray(raw_scores, dtype=np.float64)
        positions = sorted(
            range(len(self.chunks)),
            key=lambda position: (-float(scores[position]), position),
        )
        return tuple(
            (float(scores[position]), position)
            for position in positions[: min(top_k, len(positions))]
            if float(scores[position]) > 0.0
        )
