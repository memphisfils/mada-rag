"""Strict domain contracts shared by ingestion, retrieval, and generation."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

NonEmptyStr = Annotated[str, Field(min_length=1)]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]
Rank = Annotated[int, Field(ge=1)]
FiniteScore = Annotated[float, Field(allow_inf_nan=False)]


class Language(StrEnum):
    """Languages accepted for evaluation questions and generated answers."""

    EN = "en"
    FR = "fr"


class ChunkType(StrEnum):
    """Representations indexed by the retrieval layer."""

    TEXT = "text"
    TABLE_FULL = "table-full"
    TABLE_PART = "table-part"
    TABLE_ROW = "table-row"


class RetrievalMethod(StrEnum):
    """Pipeline responsible for the final candidate ordering."""

    DENSE = "dense"
    BM25 = "bm25"
    HYBRID_RRF = "hybrid-rrf"
    HYBRID_RERANK = "hybrid-rerank"


class AnswerStatus(StrEnum):
    """Only evidence-backed answers and safe abstentions are valid outcomes."""

    ANSWERED = "answered"
    ABSTAINED = "abstained"


class EvalCategory(StrEnum):
    """Required categories from the technical assessment."""

    SIMPLE_FACT = "simple-fact"
    PRECISE_NUMBER = "precise-number"
    TABLE_LOOKUP = "table-lookup"
    MULTI_PASSAGE = "multi-passage"
    TEMPORAL_AMBIGUITY = "temporal-ambiguity"
    OUT_OF_SCOPE = "out-of-scope"
    PARTIAL_COVERAGE = "partial-coverage"


class StrictModel(BaseModel):
    """Immutable Pydantic base class that rejects implicit coercion."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
        validate_default=True,
        str_strip_whitespace=True,
    )


class SnapshotManifest(StrictModel):
    """Provenance and integrity metadata for the sole knowledge snapshot."""

    schema_version: Literal["1.0"] = "1.0"
    source_count: Literal[1] = 1
    page_title: Literal["Madagascar"] = "Madagascar"
    page_id: PositiveInt
    revision_id: PositiveInt
    parent_revision_id: PositiveInt | None = None
    revision_timestamp: datetime
    fetched_at: datetime
    canonical_url: AnyHttpUrl
    api_url: AnyHttpUrl
    raw_html_path: Path
    html_sha256: Sha256
    parser_version: NonEmptyStr
    license_name: NonEmptyStr = "Creative Commons Attribution-ShareAlike 4.0 International"
    license_url: AnyHttpUrl = AnyHttpUrl("https://creativecommons.org/licenses/by-sa/4.0/")

    @field_validator("revision_timestamp", "fetched_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        """A timestamp without timezone cannot prove snapshot chronology."""

        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("snapshot timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_snapshot_chronology(self) -> Self:
        """A snapshot cannot be fetched before its source revision exists."""

        if self.fetched_at < self.revision_timestamp:
            raise ValueError("fetched_at cannot be earlier than revision_timestamp")
        return self


class SectionRecord(StrictModel):
    """Normalized article section in document order."""

    section_id: NonEmptyStr
    revision_id: PositiveInt
    title: NonEmptyStr
    level: Annotated[int, Field(ge=1, le=6)]
    path: tuple[NonEmptyStr, ...]
    ordinal: NonNegativeInt
    text: str
    paragraphs: tuple[NonEmptyStr, ...] = ()
    source_anchor: NonEmptyStr | None = None
    parent_section_id: NonEmptyStr | None = None

    @field_validator("path")
    @classmethod
    def require_section_path(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("section path cannot be empty")
        return value


class TableRecord(StrictModel):
    """Rectangular table normalized from one section of the article."""

    table_id: NonEmptyStr
    revision_id: PositiveInt
    section_id: NonEmptyStr
    section_path: tuple[NonEmptyStr, ...]
    ordinal: NonNegativeInt
    caption: str | None = None
    headers: tuple[NonEmptyStr, ...]
    rows: tuple[tuple[str, ...], ...]
    source_anchor: NonEmptyStr | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> Self:
        """Require a non-empty rectangular table after rowspan normalization."""

        if not self.section_path:
            raise ValueError("table section_path cannot be empty")
        if not self.headers:
            raise ValueError("table headers cannot be empty")
        if not self.rows:
            raise ValueError("table rows cannot be empty")
        expected_width = len(self.headers)
        if any(len(row) != expected_width for row in self.rows):
            raise ValueError("every table row must have the same width as headers")
        return self


class ParsedArticle(StrictModel):
    """Normalized content of the sole article revision, ready for chunking."""

    page_title: Literal["Madagascar"] = "Madagascar"
    schema_version: Literal["1.0"] = "1.0"
    revision_id: PositiveInt
    source_url: AnyHttpUrl
    sections: tuple[SectionRecord, ...]
    tables: tuple[TableRecord, ...] = ()

    @model_validator(mode="after")
    def validate_article_graph(self) -> Self:
        if not self.sections:
            raise ValueError("parsed article requires at least one section")

        section_ids = [section.section_id for section in self.sections]
        if len(section_ids) != len(set(section_ids)):
            raise ValueError("section IDs must be unique")
        if any(section.revision_id != self.revision_id for section in self.sections):
            raise ValueError("all sections must reference the article revision")

        known_sections = set(section_ids)
        table_ids = [table.table_id for table in self.tables]
        if len(table_ids) != len(set(table_ids)):
            raise ValueError("table IDs must be unique")
        if any(table.revision_id != self.revision_id for table in self.tables):
            raise ValueError("all tables must reference the article revision")
        if any(table.section_id not in known_sections for table in self.tables):
            raise ValueError("all tables must reference a parsed section")
        return self


class DenseIndexManifest(StrictModel):
    """Integrity metadata for one saved FAISS IndexFlatIP artifact."""

    schema_version: Literal["1.0"] = "1.0"
    revision_id: PositiveInt
    embedding_model: NonEmptyStr
    dimension: PositiveInt
    chunk_count: PositiveInt
    chunk_ids: tuple[NonEmptyStr, ...]
    metric: Literal["inner-product"] = "inner-product"
    index_type: Literal["IndexFlatIP"] = "IndexFlatIP"
    normalized: Literal[True] = True
    index_sha256: Sha256
    chunks_sha256: Sha256
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def require_created_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("index creation timestamp must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_index_manifest(self) -> Self:
        if self.chunk_count != len(self.chunk_ids):
            raise ValueError("chunk_count must match chunk_ids")
        if len(self.chunk_ids) != len(set(self.chunk_ids)):
            raise ValueError("index chunk IDs must be unique")
        return self


class Chunk(StrictModel):
    """Atomic evidence unit stored in dense and lexical indexes."""

    chunk_id: NonEmptyStr
    revision_id: PositiveInt
    chunk_type: ChunkType
    section_id: NonEmptyStr
    section_path: tuple[NonEmptyStr, ...]
    ordinal: NonNegativeInt
    text: NonEmptyStr
    token_count: PositiveInt
    content_sha256: Sha256
    source_url: AnyHttpUrl
    source_anchor: NonEmptyStr | None = None
    table_id: NonEmptyStr | None = None
    row_index: NonNegativeInt | None = None
    table_part_index: NonNegativeInt | None = None

    @model_validator(mode="after")
    def validate_table_metadata(self) -> Self:
        """Keep table-specific provenance consistent with the chunk type."""

        if not self.section_path:
            raise ValueError("chunk section_path cannot be empty")
        is_table = self.chunk_type is not ChunkType.TEXT
        if is_table and self.table_id is None:
            raise ValueError("table chunks require table_id")
        if not is_table and any(
            value is not None for value in (self.table_id, self.row_index, self.table_part_index)
        ):
            raise ValueError("text chunks cannot carry table metadata")
        if self.chunk_type is ChunkType.TABLE_ROW and self.row_index is None:
            raise ValueError("table-row chunks require row_index")
        if self.chunk_type is not ChunkType.TABLE_ROW and self.row_index is not None:
            raise ValueError("row_index is only valid for table-row chunks")
        if self.chunk_type is ChunkType.TABLE_PART and self.table_part_index is None:
            raise ValueError("table-part chunks require table_part_index")
        if self.chunk_type is not ChunkType.TABLE_PART and self.table_part_index is not None:
            raise ValueError("table_part_index is only valid for table-part chunks")
        return self


class ChunkCorpus(StrictModel):
    """Versioned serialization envelope for chunks from one article revision."""

    schema_version: Literal["1.0"] = "1.0"
    revision_id: PositiveInt
    chunks: tuple[Chunk, ...]

    @model_validator(mode="after")
    def validate_corpus(self) -> Self:
        if not self.chunks:
            raise ValueError("chunk corpus cannot be empty")
        chunk_ids = [chunk.chunk_id for chunk in self.chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("chunk IDs must be unique")
        if any(chunk.revision_id != self.revision_id for chunk in self.chunks):
            raise ValueError("all chunks must reference the corpus revision")
        return self


class RetrievedChunk(StrictModel):
    """A retrieved chunk with explainable ranks and scores."""

    chunk: Chunk
    method: RetrievalMethod
    rank: Rank
    score: FiniteScore
    dense_rank: Rank | None = None
    dense_score: FiniteScore | None = None
    bm25_rank: Rank | None = None
    bm25_score: FiniteScore | None = None
    rrf_rank: Rank | None = None
    rrf_score: FiniteScore | None = None
    reranker_rank: Rank | None = None
    reranker_score: FiniteScore | None = None
    expanded_from_chunk_id: NonEmptyStr | None = None

    @model_validator(mode="after")
    def validate_provenance(self) -> Self:
        """Ensure each retrieval mode exposes the ranking evidence it used."""

        rank_score_pairs = (
            ("dense", self.dense_rank, self.dense_score),
            ("BM25", self.bm25_rank, self.bm25_score),
            ("RRF", self.rrf_rank, self.rrf_score),
            ("reranker", self.reranker_rank, self.reranker_score),
        )
        for label, rank, score in rank_score_pairs:
            if (rank is None) != (score is None):
                raise ValueError(f"{label} rank and score must be both set or both omitted")

        if self.method is RetrievalMethod.DENSE and self.dense_rank is None:
            raise ValueError("dense retrieval requires dense_rank")
        if self.method is RetrievalMethod.BM25 and self.bm25_rank is None:
            raise ValueError("BM25 retrieval requires bm25_rank")
        if self.method in {RetrievalMethod.HYBRID_RRF, RetrievalMethod.HYBRID_RERANK}:
            if self.rrf_rank is None:
                raise ValueError("hybrid retrieval requires rrf_rank")
            if self.dense_rank is None and self.bm25_rank is None:
                raise ValueError("hybrid retrieval requires a dense or BM25 candidate rank")
        if self.method is RetrievalMethod.HYBRID_RERANK and self.reranker_rank is None:
            raise ValueError("reranked retrieval requires reranker_rank")
        return self


class Citation(StrictModel):
    """Exact evidence excerpt tied to a retrieved chunk and snapshot."""

    citation_id: NonEmptyStr
    chunk_id: NonEmptyStr
    revision_id: PositiveInt
    section_path: tuple[NonEmptyStr, ...]
    excerpt: NonEmptyStr
    source_url: AnyHttpUrl
    source_anchor: NonEmptyStr | None = None
    table_id: NonEmptyStr | None = None
    row_index: NonNegativeInt | None = None
    start_char: NonNegativeInt | None = None
    end_char: PositiveInt | None = None

    @model_validator(mode="after")
    def validate_offsets(self) -> Self:
        if not self.section_path:
            raise ValueError("citation section_path cannot be empty")
        if (self.start_char is None) != (self.end_char is None):
            raise ValueError("citation offsets must be both set or both omitted")
        if (
            self.start_char is not None
            and self.end_char is not None
            and self.end_char <= self.start_char
        ):
            raise ValueError("end_char must be greater than start_char")
        return self


class Claim(StrictModel):
    """One independently verifiable factual assertion."""

    claim_id: NonEmptyStr
    text: NonEmptyStr
    citation_ids: tuple[NonEmptyStr, ...] = ()
    supported: bool

    @model_validator(mode="after")
    def require_evidence_for_supported_claim(self) -> Self:
        if self.supported and not self.citation_ids:
            raise ValueError("supported claims require at least one citation")
        return self


class Answer(StrictModel):
    """Validated answer payload returned identically by CLI and API."""

    question: NonEmptyStr
    language: Language
    status: AnswerStatus
    text: str
    revision_id: PositiveInt
    claims: tuple[Claim, ...] = ()
    citations: tuple[Citation, ...] = ()
    retrieved_chunk_ids: tuple[NonEmptyStr, ...] = ()
    refusal_reason: str | None = None
    provider: str | None = None
    model: str | None = None
    latency_ms: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None

    @model_validator(mode="after")
    def validate_evidence_graph(self) -> Self:
        """Fail closed when claims, citations, and retrieved context disagree."""

        if self.status is AnswerStatus.ABSTAINED:
            if not self.refusal_reason:
                raise ValueError("abstained answers require a refusal_reason")
            if self.claims or self.citations:
                raise ValueError("abstained answers cannot contain factual claims or citations")
            return self

        if not self.text or not self.claims or not self.citations:
            raise ValueError("answered responses require text, claims, and citations")
        if self.refusal_reason is not None:
            raise ValueError("answered responses cannot contain a refusal_reason")
        if any(not claim.supported for claim in self.claims):
            raise ValueError("answered responses cannot contain unsupported claims")

        citation_ids = [citation.citation_id for citation in self.citations]
        if len(citation_ids) != len(set(citation_ids)):
            raise ValueError("citation IDs must be unique")
        known_citations = set(citation_ids)
        used_citations = {
            citation_id for claim in self.claims for citation_id in claim.citation_ids
        }
        if not used_citations <= known_citations:
            raise ValueError("claims reference citations absent from the answer")
        if known_citations != used_citations:
            raise ValueError("every answer citation must support at least one claim")

        retrieved_ids = set(self.retrieved_chunk_ids)
        if any(citation.chunk_id not in retrieved_ids for citation in self.citations):
            raise ValueError("citations must reference chunks retrieved for this answer")
        if any(citation.revision_id != self.revision_id for citation in self.citations):
            raise ValueError("citations must reference the same revision as the answer")
        return self


class EvalCase(StrictModel):
    """Versioned evaluation question with source-grounded expected evidence."""

    case_id: NonEmptyStr
    question: NonEmptyStr
    language: Language
    category: EvalCategory
    revision_id: PositiveInt
    answerable: bool
    expected_answer: str | None = None
    expected_section_paths: tuple[tuple[NonEmptyStr, ...], ...] = ()
    expected_chunk_ids: tuple[NonEmptyStr, ...] = ()
    evidence_excerpts: tuple[NonEmptyStr, ...] = ()
    tags: tuple[NonEmptyStr, ...] = ()
    notes: str | None = None

    @model_validator(mode="after")
    def validate_expected_evidence(self) -> Self:
        """Prevent unsourced positive cases and invented answers for traps."""

        if self.answerable:
            if not self.expected_answer:
                raise ValueError("answerable evaluation cases require expected_answer")
            if not (
                self.expected_section_paths or self.expected_chunk_ids or self.evidence_excerpts
            ):
                raise ValueError("answerable evaluation cases require expected source evidence")
        elif self.expected_answer is not None:
            raise ValueError("unanswerable evaluation cases cannot define expected_answer")
        return self
