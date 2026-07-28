"""Corpus-only context expansion for structured tables."""

from __future__ import annotations

from mada_rag.models import Chunk, RetrievedChunk


class ContextExpander:
    """Expose all representations of a retrieved table without external data."""

    def __init__(self, chunks: tuple[Chunk, ...], *, max_expanded_chunks: int = 100) -> None:
        if max_expanded_chunks <= 0:
            raise ValueError("max_expanded_chunks must be positive")
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("context corpus chunk IDs must be unique")
        self.chunks = chunks
        self.max_expanded_chunks = max_expanded_chunks
        self._table_chunks: dict[str, tuple[Chunk, ...]] = {}
        table_groups: dict[str, list[Chunk]] = {}
        for chunk in chunks:
            if chunk.table_id is not None:
                table_groups.setdefault(chunk.table_id, []).append(chunk)
        for table_id, group in table_groups.items():
            self._table_chunks[table_id] = tuple(sorted(group, key=lambda chunk: chunk.ordinal))

    def expand(self, candidates: tuple[RetrievedChunk, ...]) -> tuple[RetrievedChunk, ...]:
        if not candidates:
            return ()
        output = list(candidates)
        seen = {candidate.chunk.chunk_id for candidate in candidates}
        for source in candidates:
            if source.chunk.table_id is None:
                continue
            for related in self._table_chunks.get(source.chunk.table_id, ()):
                if related.chunk_id in seen:
                    continue
                if len(output) >= self.max_expanded_chunks:
                    return tuple(output)
                seen.add(related.chunk_id)
                output.append(
                    source.model_copy(
                        update={
                            "chunk": related,
                            "rank": len(output) + 1,
                            "expanded_from_chunk_id": source.chunk.chunk_id,
                        }
                    )
                )
        return tuple(output)
