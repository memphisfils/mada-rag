"""Fail-closed sufficiency, extractive generation, and citation attack tests."""

import hashlib

import pytest

from mada_rag.generation import (
    CitationValidationError,
    CitationValidator,
    ExtractiveGenerator,
    SufficiencyPolicy,
)
from mada_rag.models import (
    AnswerStatus,
    Chunk,
    ChunkType,
    Language,
    RetrievalMethod,
    RetrievedChunk,
)

REVISION_ID = 123
SOURCE_URL = "https://en.wikipedia.org/wiki/Madagascar"


def make_chunk(
    index: int = 0,
    *,
    text: str = "  Exact evidence sentence. A second exact sentence.",
    revision_id: int = REVISION_ID,
) -> Chunk:
    return Chunk(
        chunk_id=f"chunk-{index}",
        revision_id=revision_id,
        chunk_type=ChunkType.TEXT,
        section_id=f"section-{index}",
        section_path=(f"Section {index}",),
        ordinal=index,
        text=text,
        token_count=len(text.split()),
        content_sha256=hashlib.sha256(text.encode()).hexdigest(),
        source_url=SOURCE_URL,
        source_anchor=f"anchor-{index}",
    )


def make_retrieved(
    index: int = 0,
    *,
    score: float = 0.9,
    text: str = "  Exact evidence sentence. A second exact sentence.",
    revision_id: int = REVISION_ID,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=make_chunk(index, text=text, revision_id=revision_id),
        method=RetrievalMethod.DENSE,
        rank=index + 1,
        score=score,
        dense_rank=index + 1,
        dense_score=score,
    )


@pytest.mark.parametrize(
    ("candidates", "minimum_candidates", "minimum_score", "reason"),
    [
        ((), 1, 0.4, "no evidence"),
        ((make_retrieved(),), 2, 0.4, "too few"),
        ((make_retrieved(score=0.3),), 1, 0.4, "below the threshold"),
        (
            (
                make_retrieved(),
                make_retrieved(1, revision_id=REVISION_ID + 1),
            ),
            1,
            0.4,
            "mixes snapshot revisions",
        ),
    ],
)
def test_sufficiency_policy_rejects_unsafe_contexts(
    candidates: tuple[RetrievedChunk, ...],
    minimum_candidates: int,
    minimum_score: float,
    reason: str,
) -> None:
    decision = SufficiencyPolicy(
        minimum_score=minimum_score,
        minimum_candidates=minimum_candidates,
    ).assess(candidates)

    assert not decision.sufficient
    assert reason in decision.reason


def test_sufficiency_policy_accepts_enough_consistent_evidence() -> None:
    decision = SufficiencyPolicy(minimum_score=0.5, minimum_candidates=2).assess(
        (make_retrieved(), make_retrieved(1, score=0.8))
    )

    assert decision.sufficient


def test_extractive_generator_returns_exact_offsets_claims_and_provenance() -> None:
    candidates = (
        make_retrieved(),
        make_retrieved(1, text="Independent second evidence sentence."),
    )
    answer = ExtractiveGenerator(max_claims=2, max_excerpt_chars=40).generate(
        "Question?",
        Language.EN,
        candidates,
    )

    assert answer.status is AnswerStatus.ANSWERED
    assert len(answer.claims) == len(answer.citations) == 2
    assert answer.retrieved_chunk_ids == ("chunk-0", "chunk-1")
    for claim, citation in zip(answer.claims, answer.citations, strict=True):
        chunk = next(item.chunk for item in candidates if item.chunk.chunk_id == citation.chunk_id)
        assert citation.start_char is not None
        assert citation.end_char is not None
        assert chunk.text[citation.start_char : citation.end_char] == citation.excerpt
        assert claim.text == citation.excerpt
        assert claim.citation_ids == (citation.citation_id,)
    CitationValidator().validate(answer, candidates)


def test_extractive_generator_skips_duplicate_excerpts_and_honors_max_claims() -> None:
    duplicate_text = "Same exact evidence text long enough for extraction."
    candidates = (
        make_retrieved(0, text=duplicate_text),
        make_retrieved(1, text=duplicate_text),
        make_retrieved(2, text="Distinct exact evidence text long enough for extraction."),
    )

    answer = ExtractiveGenerator(max_claims=2).generate(
        "Question?",
        Language.FR,
        candidates,
    )

    assert len(answer.claims) == 2
    assert len({claim.text for claim in answer.claims}) == 2


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("chunk_id", "not-retrieved", "not retrieved"),
        ("revision_id", REVISION_ID + 1, "revision"),
        ("start_char", None, "exact offsets"),
        ("end_char", 10_000, "offsets exceed"),
        ("excerpt", "forged evidence", "not exact"),
        ("section_path", ("Forged",), "section differs"),
        ("source_anchor", "forged-anchor", "anchor differs"),
        ("table_id", "forged-table", "table provenance"),
    ],
)
def test_citation_validator_rejects_tampered_citation_fields(
    field: str, value: object, message: str
) -> None:
    candidates = (make_retrieved(),)
    answer = ExtractiveGenerator().generate("Question?", Language.EN, candidates)
    citation = answer.citations[0].model_copy(update={field: value})
    attacked = answer.model_copy(update={"citations": (citation,)})

    with pytest.raises(CitationValidationError, match=message):
        CitationValidator().validate(attacked, candidates)


def test_citation_validator_rejects_forged_claim_text() -> None:
    candidates = (make_retrieved(),)
    answer = ExtractiveGenerator().generate("Question?", Language.EN, candidates)
    claim = answer.claims[0].model_copy(update={"text": "forged claim"})
    attacked = answer.model_copy(update={"claims": (claim,)})

    with pytest.raises(CitationValidationError, match="claim"):
        CitationValidator().validate(attacked, candidates)


def test_citation_validator_rejects_duplicate_retrieved_chunk_ids() -> None:
    candidate = make_retrieved()
    answer = ExtractiveGenerator().generate("Question?", Language.EN, (candidate,))

    with pytest.raises(CitationValidationError, match="not unique"):
        CitationValidator().validate(answer, (candidate, candidate))


def test_citation_validator_rejects_answer_retrieved_id_tampering() -> None:
    candidates = (make_retrieved(),)
    answer = ExtractiveGenerator().generate("Question?", Language.EN, candidates)
    attacked = answer.model_copy(update={"retrieved_chunk_ids": ("forged-chunk",)})

    with pytest.raises(CitationValidationError, match="retrieved"):
        CitationValidator().validate(attacked, candidates)
