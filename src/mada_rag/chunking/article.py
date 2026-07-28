"""Deterministic section-aware and table-aware chunk construction."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from typing import Protocol, Self

from mada_rag.config import Settings
from mada_rag.models import Chunk, ChunkType, ParsedArticle, SectionRecord, TableRecord

_TOKEN_RE = re.compile(r"\S+")


class ChunkingError(RuntimeError):
    """Raised when evidence cannot fit the configured token budget safely."""


class Tokenizer(Protocol):
    """Minimal injectable tokenizer contract used by the chunker."""

    def count_tokens(self, text: str) -> int:
        """Return the exact token count used to enforce budgets."""


class WhitespaceTokenizer:
    """Deterministic dependency-free tokenizer intended for tests and smoke runs."""

    def count_tokens(self, text: str) -> int:
        return len(_TOKEN_RE.findall(text))


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stable_chunk_id(
    *,
    revision_id: int,
    chunk_type: ChunkType,
    section_id: str,
    ordinal: int,
    content_sha256: str,
    table_id: str | None,
    row_index: int | None,
    part_index: int | None,
) -> str:
    material = "\x1f".join(
        (
            str(revision_id),
            chunk_type.value,
            section_id,
            str(ordinal),
            table_id or "",
            "" if row_index is None else str(row_index),
            "" if part_index is None else str(part_index),
            content_sha256,
        )
    )
    return f"chunk-{hashlib.sha256(material.encode('utf-8')).hexdigest()[:32]}"


def _join_words(words: Sequence[str]) -> str:
    return " ".join(words)


def _furthest_fitting_word(
    words: Sequence[str],
    *,
    start: int,
    token_limit: int,
    tokenizer: Tokenizer,
) -> int:
    low = start + 1
    high = len(words)
    best = start
    while low <= high:
        middle = (low + high) // 2
        candidate = _join_words(words[start:middle])
        if tokenizer.count_tokens(candidate) <= token_limit:
            best = middle
            low = middle + 1
        else:
            high = middle - 1
    return best


def _split_to_budget(
    text: str,
    *,
    token_limit: int,
    overlap_tokens: int,
    tokenizer: Tokenizer,
) -> tuple[str, ...]:
    words = text.split()
    if not words:
        return ()

    pieces: list[str] = []
    start = 0
    while start < len(words):
        end = _furthest_fitting_word(
            words,
            start=start,
            token_limit=token_limit,
            tokenizer=tokenizer,
        )
        if end == start:
            raise ChunkingError("a single source token exceeds the configured chunk budget")
        pieces.append(_join_words(words[start:end]))
        if end == len(words):
            break

        next_start = end
        if overlap_tokens:
            cursor = end - 1
            while cursor > start:
                overlap = _join_words(words[cursor:end])
                if tokenizer.count_tokens(overlap) > overlap_tokens:
                    break
                next_start = cursor
                cursor -= 1
        start = next_start if next_start > start else end
    return tuple(pieces)


def _render_table_header(table: TableRecord) -> tuple[str, ...]:
    lines: list[str] = []
    if table.caption:
        lines.append(f"Table: {table.caption}")
    lines.append(f"Columns: {' | '.join(table.headers)}")
    return tuple(lines)


def _render_table_row(table: TableRecord, row_index: int) -> str:
    row = table.rows[row_index]
    values = " | ".join(
        f"{header}={value or 'N/A'}" for header, value in zip(table.headers, row, strict=True)
    )
    prefix = f"Table: {table.caption}\n" if table.caption else ""
    return f"{prefix}Row {row_index + 1}: {values}"


class ArticleChunker:
    """Create stable chunks without crossing section or table boundaries."""

    def __init__(
        self,
        *,
        tokenizer: Tokenizer | None = None,
        target_tokens: int = 350,
        max_tokens: int = 450,
        overlap_tokens: int = 50,
        table_max_tokens: int = 450,
    ) -> None:
        if target_tokens <= 0 or max_tokens <= 0 or table_max_tokens <= 0:
            raise ValueError("token budgets must be positive")
        if target_tokens > max_tokens:
            raise ValueError("target_tokens cannot exceed max_tokens")
        if overlap_tokens < 0 or overlap_tokens >= target_tokens:
            raise ValueError("overlap_tokens must be non-negative and smaller than target_tokens")
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.table_max_tokens = table_max_tokens

    @classmethod
    def from_settings(cls, settings: Settings, *, tokenizer: Tokenizer | None = None) -> Self:
        return cls(
            tokenizer=tokenizer,
            target_tokens=settings.chunk_target_tokens,
            max_tokens=settings.chunk_max_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
            table_max_tokens=settings.table_chunk_max_tokens,
        )

    def chunk(self, article: ParsedArticle) -> tuple[Chunk, ...]:
        """Return all text, table-global/table-part, and table-row chunks."""

        chunks: list[Chunk] = []
        tables_by_section: dict[str, list[TableRecord]] = {}
        for table in article.tables:
            tables_by_section.setdefault(table.section_id, []).append(table)

        ordinal = 0
        for section in sorted(article.sections, key=lambda record: record.ordinal):
            for text in self._section_pieces(section):
                chunks.append(
                    self._make_chunk(
                        article=article,
                        section=section,
                        text=text,
                        chunk_type=ChunkType.TEXT,
                        ordinal=ordinal,
                    )
                )
                ordinal += 1

            for table in sorted(
                tables_by_section.get(section.section_id, []),
                key=lambda record: record.ordinal,
            ):
                table_chunks = self._table_chunks(
                    article=article,
                    section=section,
                    table=table,
                    starting_ordinal=ordinal,
                )
                chunks.extend(table_chunks)
                ordinal += len(table_chunks)
        return tuple(chunks)

    def _section_pieces(self, section: SectionRecord) -> tuple[str, ...]:
        paragraphs = section.paragraphs or tuple(
            paragraph.strip() for paragraph in section.text.split("\n\n") if paragraph.strip()
        )
        if not paragraphs:
            return ()
        text = "\n\n".join(paragraphs)
        pieces = _split_to_budget(
            text,
            token_limit=self.target_tokens,
            overlap_tokens=self.overlap_tokens,
            tokenizer=self.tokenizer,
        )
        if any(self.tokenizer.count_tokens(piece) > self.max_tokens for piece in pieces):
            raise ChunkingError("text chunk exceeds max_tokens")
        return pieces

    def _table_chunks(
        self,
        *,
        article: ParsedArticle,
        section: SectionRecord,
        table: TableRecord,
        starting_ordinal: int,
    ) -> tuple[Chunk, ...]:
        chunks: list[Chunk] = []
        header = _render_table_header(table)
        row_lines = tuple(_render_table_row(table, index) for index in range(len(table.rows)))
        full_text = "\n".join((*header, *row_lines))
        full_count = self.tokenizer.count_tokens(full_text)

        ordinal = starting_ordinal
        if full_count <= self.table_max_tokens:
            chunks.append(
                self._make_chunk(
                    article=article,
                    section=section,
                    text=full_text,
                    chunk_type=ChunkType.TABLE_FULL,
                    ordinal=ordinal,
                    table=table,
                )
            )
            ordinal += 1
        else:
            for part_index, part_text in enumerate(self._partition_table(header, row_lines)):
                chunks.append(
                    self._make_chunk(
                        article=article,
                        section=section,
                        text=part_text,
                        chunk_type=ChunkType.TABLE_PART,
                        ordinal=ordinal,
                        table=table,
                        table_part_index=part_index,
                    )
                )
                ordinal += 1

        for row_index, row_text in enumerate(row_lines):
            if self.tokenizer.count_tokens(row_text) > self.table_max_tokens:
                raise ChunkingError(
                    f"row {row_index} of table {table.table_id} exceeds the table token budget"
                )
            chunks.append(
                self._make_chunk(
                    article=article,
                    section=section,
                    text=row_text,
                    chunk_type=ChunkType.TABLE_ROW,
                    ordinal=ordinal,
                    table=table,
                    row_index=row_index,
                )
            )
            ordinal += 1
        return tuple(chunks)

    def _partition_table(
        self,
        header: tuple[str, ...],
        row_lines: tuple[str, ...],
    ) -> tuple[str, ...]:
        parts: list[str] = []
        current_rows: list[str] = []
        for row in row_lines:
            candidate = "\n".join((*header, *current_rows, row))
            if self.tokenizer.count_tokens(candidate) <= self.table_max_tokens:
                current_rows.append(row)
                continue
            if not current_rows:
                raise ChunkingError("one table row cannot fit with its repeated headers")
            parts.append("\n".join((*header, *current_rows)))
            current_rows = [row]
            if self.tokenizer.count_tokens("\n".join((*header, row))) > self.table_max_tokens:
                raise ChunkingError("one table row cannot fit with its repeated headers")
        if current_rows:
            parts.append("\n".join((*header, *current_rows)))
        return tuple(parts)

    def _make_chunk(
        self,
        *,
        article: ParsedArticle,
        section: SectionRecord,
        text: str,
        chunk_type: ChunkType,
        ordinal: int,
        table: TableRecord | None = None,
        row_index: int | None = None,
        table_part_index: int | None = None,
    ) -> Chunk:
        token_count = self.tokenizer.count_tokens(text)
        if token_count <= 0:
            raise ChunkingError("empty chunks are forbidden")
        content_sha256 = _sha256(text)
        table_id = table.table_id if table else None
        return Chunk(
            chunk_id=_stable_chunk_id(
                revision_id=article.revision_id,
                chunk_type=chunk_type,
                section_id=section.section_id,
                ordinal=ordinal,
                content_sha256=content_sha256,
                table_id=table_id,
                row_index=row_index,
                part_index=table_part_index,
            ),
            revision_id=article.revision_id,
            chunk_type=chunk_type,
            section_id=section.section_id,
            section_path=section.path,
            ordinal=ordinal,
            text=text,
            token_count=token_count,
            content_sha256=content_sha256,
            source_url=article.source_url,
            source_anchor=table.source_anchor if table else section.source_anchor,
            table_id=table_id,
            row_index=row_index,
            table_part_index=table_part_index,
        )


def chunk_article(
    article: ParsedArticle,
    *,
    settings: Settings | None = None,
    tokenizer: Tokenizer | None = None,
) -> tuple[Chunk, ...]:
    """Convenience entry point using validated settings when supplied."""

    chunker = (
        ArticleChunker.from_settings(settings, tokenizer=tokenizer)
        if settings is not None
        else ArticleChunker(tokenizer=tokenizer)
    )
    return chunker.chunk(article)
