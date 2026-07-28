"""Application-service tests with real snapshot evidence and deterministic fakes."""

from dataclasses import dataclass, field
from pathlib import Path

from mada_rag.generation import (
    CitationValidationError,
    CitationValidator,
    ExtractiveGenerator,
    SufficiencyPolicy,
)
from mada_rag.models import Answer, AnswerStatus, Language, RetrievalMethod, RetrievedChunk
from mada_rag.service import RagService
from mada_rag.storage import load_chunks

LIFE_EXPECTANCY_CHUNK_ID = "chunk-0e65f76953781b67a8443ddd710e1b79"
QUESTION_EN = "What was adult life expectancy as of 2009?"
G3_CHUNKS_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "g3_chunks.json"


def real_chunk(chunk_id: str = LIFE_EXPECTANCY_CHUNK_ID) -> RetrievedChunk:
    chunks = load_chunks(G3_CHUNKS_FIXTURE)
    chunk = next(item for item in chunks if item.chunk_id == chunk_id)
    return RetrievedChunk(
        chunk=chunk,
        method=RetrievalMethod.DENSE,
        rank=1,
        score=0.9,
        dense_rank=1,
        dense_score=0.9,
    )


@dataclass
class FakeRetriever:
    candidates: tuple[RetrievedChunk, ...]
    revision_id: int
    calls: list[tuple[str, int | None]] = field(default_factory=list)

    @property
    def chunks(self) -> tuple:
        return tuple(candidate.chunk for candidate in self.candidates)

    def retrieve(
        self,
        question: str,
        *,
        top_k: int | None = None,
    ) -> tuple[RetrievedChunk, ...]:
        self.calls.append((question, top_k))
        return self.candidates[:top_k]


class RejectingValidator(CitationValidator):
    def validate(
        self,
        answer: Answer,
        retrieved: tuple[RetrievedChunk, ...],
    ) -> None:
        raise CitationValidationError("deliberate validation failure")


class FailingGenerator(ExtractiveGenerator):
    def generate(
        self,
        question: str,
        language: Language,
        candidates: tuple[RetrievedChunk, ...],
    ) -> Answer:
        raise ValueError("deliberate generation failure")


def make_service(
    retriever: FakeRetriever,
    *,
    generator: ExtractiveGenerator | None = None,
    validator: CitationValidator | None = None,
    minimum_score: float = 0.5,
    minimum_concept_coverage: float = 0.8,
) -> RagService:
    return RagService(
        retriever=retriever,
        generator=generator or ExtractiveGenerator(),
        sufficiency_policy=SufficiencyPolicy(
            minimum_score=minimum_score,
            minimum_concept_coverage=minimum_concept_coverage,
        ),
        citation_validator=validator,
        context_top_k=2,
    )


def test_service_returns_validated_answer_from_matching_snapshot_evidence() -> None:
    candidate = real_chunk()
    retriever = FakeRetriever((candidate,), revision_id=candidate.chunk.revision_id)

    answer = make_service(retriever).ask(QUESTION_EN, language=Language.EN)

    assert answer.status is AnswerStatus.ANSWERED
    assert "Adult life expectancy as of 2009" in answer.text
    assert all(citation.excerpt in candidate.chunk.text for citation in answer.citations)
    assert answer.latency_ms is not None
    assert answer.latency_ms >= 0
    assert answer.retrieved_chunk_ids == (LIFE_EXPECTANCY_CHUNK_ID,)
    assert retriever.calls == [(QUESTION_EN, 2)]


def test_service_abstains_in_requested_language_when_evidence_is_insufficient() -> None:
    candidate = real_chunk().model_copy(
        update={"score": 0.2, "dense_score": 0.2},
    )
    retriever = FakeRetriever((candidate,), revision_id=candidate.chunk.revision_id)

    answer = make_service(retriever).ask(QUESTION_EN, language=Language.FR)

    assert answer.status is AnswerStatus.ABSTAINED
    assert answer.text == "Je ne sais pas à partir du snapshot fourni."
    assert "below the threshold" in (answer.refusal_reason or "")
    assert not answer.claims
    assert not answer.citations


def test_service_fails_closed_on_citation_validation_error() -> None:
    candidate = real_chunk()
    retriever = FakeRetriever((candidate,), revision_id=candidate.chunk.revision_id)

    answer = make_service(
        retriever,
        validator=RejectingValidator(),
        minimum_concept_coverage=0.0,
    ).ask(QUESTION_EN)

    assert answer.status is AnswerStatus.ABSTAINED
    assert answer.refusal_reason == "generated evidence failed exact citation validation"


def test_service_fails_closed_on_generation_value_error() -> None:
    candidate = real_chunk()
    retriever = FakeRetriever((candidate,), revision_id=candidate.chunk.revision_id)

    answer = make_service(
        retriever,
        generator=FailingGenerator(),
        minimum_concept_coverage=0.0,
    ).ask(QUESTION_EN)

    assert answer.status is AnswerStatus.ABSTAINED
    assert answer.refusal_reason == "generated evidence failed exact citation validation"
