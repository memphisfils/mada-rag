"""Contract tests for immutable source, retrieval, and answer models."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from mada_rag.models import (
    Answer,
    AnswerStatus,
    Chunk,
    ChunkType,
    Citation,
    Claim,
    EvalCase,
    EvalCategory,
    Language,
    RetrievalMethod,
    RetrievedChunk,
    SectionRecord,
    SnapshotManifest,
    TableRecord,
)

SOURCE_URL = "https://en.wikipedia.org/wiki/Madagascar"
API_URL = "https://en.wikipedia.org/w/api.php"
REVISION_ID = 123
SHA256 = "a" * 64


def make_manifest(**overrides: object) -> SnapshotManifest:
    data: dict[str, object] = {
        "page_id": 42,
        "revision_id": REVISION_ID,
        "parent_revision_id": 122,
        "revision_timestamp": datetime(2026, 7, 28, 8, 0, tzinfo=UTC),
        "fetched_at": datetime(2026, 7, 28, 8, 1, tzinfo=UTC),
        "canonical_url": SOURCE_URL,
        "api_url": API_URL,
        "raw_html_path": Path("data/raw/madagascar.html"),
        "html_sha256": SHA256,
        "parser_version": "1.0",
    }
    data.update(overrides)
    return SnapshotManifest.model_validate(data)


def make_chunk(**overrides: object) -> Chunk:
    data: dict[str, object] = {
        "chunk_id": "chunk-1",
        "revision_id": REVISION_ID,
        "chunk_type": ChunkType.TEXT,
        "section_id": "lead",
        "section_path": ("Lead",),
        "ordinal": 0,
        "text": "Synthetic evidence for a model contract test.",
        "token_count": 8,
        "content_sha256": SHA256,
        "source_url": SOURCE_URL,
    }
    data.update(overrides)
    return Chunk.model_validate(data)


def make_citation(**overrides: object) -> Citation:
    data: dict[str, object] = {
        "citation_id": "citation-1",
        "chunk_id": "chunk-1",
        "revision_id": REVISION_ID,
        "section_path": ("Lead",),
        "excerpt": "Synthetic evidence",
        "source_url": SOURCE_URL,
    }
    data.update(overrides)
    return Citation.model_validate(data)


def make_answer(**overrides: object) -> Answer:
    data: dict[str, object] = {
        "question": "What does the synthetic evidence say?",
        "language": Language.EN,
        "status": AnswerStatus.ANSWERED,
        "text": "It provides synthetic evidence.",
        "revision_id": REVISION_ID,
        "claims": (
            Claim(
                claim_id="claim-1",
                text="It provides synthetic evidence.",
                citation_ids=("citation-1",),
                supported=True,
            ),
        ),
        "citations": (make_citation(),),
        "retrieved_chunk_ids": ("chunk-1",),
    }
    data.update(overrides)
    return Answer.model_validate(data)


def test_strict_models_forbid_extra_reject_coercion_and_are_frozen() -> None:
    manifest = make_manifest()

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SnapshotManifest.model_validate(
            {**manifest.model_dump(), "unexpected_field": "not allowed"}
        )

    with pytest.raises(ValidationError):
        SectionRecord.model_validate(
            {
                "section_id": "lead",
                "revision_id": "123",
                "title": "Lead",
                "level": 1,
                "path": ("Lead",),
                "ordinal": 0,
                "text": "Synthetic.",
            }
        )

    with pytest.raises(ValidationError, match="Instance is frozen"):
        manifest.page_id = 43


@pytest.mark.parametrize("field", ["revision_timestamp", "fetched_at"])
def test_snapshot_timestamps_must_be_timezone_aware(field: str) -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        make_manifest(**{field: datetime(2026, 7, 28, 8, 0)})


def test_snapshot_accepts_timezone_aware_timestamps() -> None:
    manifest = make_manifest()

    assert manifest.revision_timestamp.utcoffset() is not None
    assert manifest.fetched_at.utcoffset() is not None


def test_snapshot_rejects_fetch_before_revision_timestamp() -> None:
    with pytest.raises(ValidationError, match="fetched_at"):
        make_manifest(
            revision_timestamp=datetime(2026, 7, 28, 8, 1, tzinfo=UTC),
            fetched_at=datetime(2026, 7, 28, 8, 0, tzinfo=UTC),
        )


def test_table_requires_non_empty_rectangular_rows() -> None:
    table = TableRecord(
        table_id="table-1",
        revision_id=REVISION_ID,
        section_id="regions",
        section_path=("Synthetic regions",),
        ordinal=0,
        caption="Synthetic table",
        headers=("Region", "Area"),
        rows=(("Alpha", "101"), ("Beta", "202")),
    )

    assert len(table.rows) == 2
    assert all(len(row) == len(table.headers) for row in table.rows)

    with pytest.raises(ValidationError, match="same width as headers"):
        TableRecord(
            table_id="table-1",
            revision_id=REVISION_ID,
            section_id="regions",
            section_path=("Synthetic regions",),
            ordinal=0,
            headers=("Region", "Area"),
            rows=(("Alpha",),),
        )

    with pytest.raises(ValidationError, match="table rows cannot be empty"):
        TableRecord(
            table_id="table-1",
            revision_id=REVISION_ID,
            section_id="regions",
            section_path=("Synthetic regions",),
            ordinal=0,
            headers=("Region", "Area"),
            rows=(),
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"chunk_type": ChunkType.TABLE_FULL},
            "table chunks require table_id",
        ),
        (
            {"chunk_type": ChunkType.TEXT, "table_id": "table-1"},
            "text chunks cannot carry table metadata",
        ),
        (
            {"chunk_type": ChunkType.TABLE_ROW, "table_id": "table-1"},
            "table-row chunks require row_index",
        ),
        (
            {
                "chunk_type": ChunkType.TABLE_FULL,
                "table_id": "table-1",
                "row_index": 0,
            },
            "row_index is only valid for table-row chunks",
        ),
        (
            {"chunk_type": ChunkType.TABLE_PART, "table_id": "table-1"},
            "table-part chunks require table_part_index",
        ),
    ],
)
def test_chunk_rejects_incoherent_table_metadata(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        make_chunk(**overrides)


def test_chunk_accepts_complete_table_row_metadata() -> None:
    chunk = make_chunk(
        chunk_type=ChunkType.TABLE_ROW,
        table_id="table-1",
        row_index=0,
    )

    assert chunk.table_id == "table-1"
    assert chunk.row_index == 0


@pytest.mark.parametrize(
    ("method", "extra", "message"),
    [
        (RetrievalMethod.DENSE, {}, "dense retrieval requires dense_rank"),
        (RetrievalMethod.BM25, {}, "BM25 retrieval requires bm25_rank"),
        (
            RetrievalMethod.HYBRID_RRF,
            {"dense_rank": 1, "dense_score": 0.8},
            "hybrid retrieval requires rrf_rank",
        ),
        (
            RetrievalMethod.HYBRID_RERANK,
            {
                "dense_rank": 1,
                "dense_score": 0.8,
                "rrf_rank": 1,
                "rrf_score": 0.03,
            },
            "reranked retrieval requires reranker_rank",
        ),
    ],
)
def test_retrieved_chunk_requires_method_provenance(
    method: RetrievalMethod, extra: dict[str, object], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        RetrievedChunk(
            chunk=make_chunk(),
            method=method,
            rank=1,
            score=0.5,
            **extra,
        )


def test_retrieved_chunk_accepts_complete_reranked_provenance() -> None:
    retrieved = RetrievedChunk(
        chunk=make_chunk(),
        method=RetrievalMethod.HYBRID_RERANK,
        rank=1,
        score=0.9,
        dense_rank=2,
        dense_score=0.7,
        bm25_rank=1,
        bm25_score=4.2,
        rrf_rank=1,
        rrf_score=0.03,
        reranker_rank=1,
        reranker_score=0.9,
    )

    assert retrieved.rrf_rank == 1
    assert retrieved.reranker_rank == 1


@pytest.mark.parametrize(
    "missing_field",
    [
        "dense_rank",
        "dense_score",
        "bm25_rank",
        "bm25_score",
        "rrf_rank",
        "rrf_score",
        "reranker_rank",
        "reranker_score",
    ],
)
def test_retrieved_chunk_rejects_incomplete_rank_score_pairs(
    missing_field: str,
) -> None:
    provenance: dict[str, object] = {
        "dense_rank": 2,
        "dense_score": 0.7,
        "bm25_rank": 1,
        "bm25_score": 4.2,
        "rrf_rank": 1,
        "rrf_score": 0.03,
        "reranker_rank": 1,
        "reranker_score": 0.9,
    }
    provenance[missing_field] = None

    with pytest.raises(ValidationError, match=r"rank.*score|score.*rank"):
        RetrievedChunk(
            chunk=make_chunk(),
            method=RetrievalMethod.HYBRID_RERANK,
            rank=1,
            score=0.9,
            **provenance,
        )


def test_answer_accepts_closed_evidence_graph() -> None:
    answer = make_answer()

    assert answer.status is AnswerStatus.ANSWERED
    assert answer.claims[0].citation_ids == ("citation-1",)
    assert answer.citations[0].chunk_id in answer.retrieved_chunk_ids


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {
                "claims": (
                    Claim(
                        claim_id="claim-1",
                        text="Synthetic claim.",
                        citation_ids=("missing-citation",),
                        supported=True,
                    ),
                )
            },
            "claims reference citations absent from the answer",
        ),
        (
            {
                "citations": (
                    make_citation(),
                    make_citation(
                        citation_id="citation-2",
                        chunk_id="chunk-2",
                    ),
                ),
                "retrieved_chunk_ids": ("chunk-1", "chunk-2"),
            },
            "every answer citation must support at least one claim",
        ),
        (
            {"retrieved_chunk_ids": ("another-chunk",)},
            "citations must reference chunks retrieved for this answer",
        ),
    ],
)
def test_answer_rejects_open_evidence_graph(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        make_answer(**overrides)


def test_answer_abstention_has_no_factual_graph() -> None:
    answer = Answer(
        question="What is absent?",
        language=Language.EN,
        status=AnswerStatus.ABSTAINED,
        text="I do not know from the supplied snapshot.",
        revision_id=REVISION_ID,
        refusal_reason="insufficient evidence",
    )

    assert not answer.claims
    assert not answer.citations

    with pytest.raises(ValidationError, match="cannot contain factual claims or citations"):
        make_answer(
            status=AnswerStatus.ABSTAINED,
            refusal_reason="insufficient evidence",
        )


def test_answer_rejects_citation_from_another_revision() -> None:
    with pytest.raises(ValidationError, match="revision"):
        make_answer(citations=(make_citation(revision_id=REVISION_ID + 1),))


def test_eval_case_accepts_sourced_answerable_case() -> None:
    case = EvalCase(
        case_id="simple-en-1",
        question="What does the synthetic fixture contain?",
        language=Language.EN,
        category=EvalCategory.SIMPLE_FACT,
        revision_id=REVISION_ID,
        answerable=True,
        expected_answer="Synthetic evidence.",
        expected_chunk_ids=("chunk-1",),
    )

    assert case.answerable
    assert case.expected_chunk_ids == ("chunk-1",)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"expected_answer": None, "expected_chunk_ids": ("chunk-1",)},
            "answerable evaluation cases require expected_answer",
        ),
        (
            {"expected_answer": "Synthetic evidence.", "expected_chunk_ids": ()},
            "answerable evaluation cases require expected source evidence",
        ),
    ],
)
def test_eval_case_rejects_incomplete_answerable_case(
    overrides: dict[str, object], message: str
) -> None:
    data: dict[str, object] = {
        "case_id": "simple-en-1",
        "question": "What does the synthetic fixture contain?",
        "language": Language.EN,
        "category": EvalCategory.SIMPLE_FACT,
        "revision_id": REVISION_ID,
        "answerable": True,
        "expected_answer": "Synthetic evidence.",
        "expected_chunk_ids": ("chunk-1",),
    }
    data.update(overrides)

    with pytest.raises(ValidationError, match=message):
        EvalCase.model_validate(data)


def test_eval_case_accepts_unanswerable_case_without_expected_answer() -> None:
    case = EvalCase(
        case_id="trap-fr-1",
        question="Quel fait est absent de la fixture ?",
        language=Language.FR,
        category=EvalCategory.OUT_OF_SCOPE,
        revision_id=REVISION_ID,
        answerable=False,
    )

    assert case.expected_answer is None

    with pytest.raises(
        ValidationError,
        match="unanswerable evaluation cases cannot define expected_answer",
    ):
        EvalCase(
            case_id="trap-fr-1",
            question="Quel fait est absent de la fixture ?",
            language=Language.FR,
            category=EvalCategory.OUT_OF_SCOPE,
            revision_id=REVISION_ID,
            answerable=False,
            expected_answer="An invented answer.",
        )
