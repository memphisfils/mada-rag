"""Shared retriever protocol."""

from __future__ import annotations

from typing import Protocol

from mada_rag.models import Chunk, RetrievedChunk


class Retriever(Protocol):
    @property
    def revision_id(self) -> int: ...

    @property
    def chunks(self) -> tuple[Chunk, ...]: ...

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
    ) -> tuple[RetrievedChunk, ...]: ...
