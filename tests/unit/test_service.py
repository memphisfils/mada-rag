"""Application-service tests with deterministic retriever and generator fakes."""

import hashlib
from dataclasses import dataclass, field

from mada_rag.generation import (
    CitationValidationError,
    CitationValidator,
    ExtractiveGenerator,
    SufficiencyPolicy,
)
from mada_rag.models import (
    Answer,
    AnswerStatus,
    Chunk,
    ChunkType,
    Language,
    RetrievalMethod,
    RetrievedChunk,
)
from mada_rag.service import RagService


def make_retrieved(index: int = 0, *, score: float = 0.9) -> RetrievedChunk:
    text = f"Exact service evidence sentence number {index}."
    chunk = Chunk(
        chunk_id=f"chunk-{index}",
        revision_id=123,
        chunk_type=ChunkType.TEXT,
        section_id=f"section-{index}",
        section_path=(f"Section {index}",),
        ordinal=index,
        text=text,
        token_count=len(text.split()),
        content_sha256=hashlib.sha256(text.encode()).hexdigest(),
        source_url="https://en.wikipedia.org/wiki/Madagascar",
    )
    return RetrievedChunk(
        chunk=chunk,
        method=RetrievalMethod.DENSE,
        rank=index + 1,
        score=score,
        dense_rank=index + 1,
        dense_score=score,
    )


@dataclass
class FakeRetriever:
    candidates: tuple[RetrievedChunk, ...]
    revision_id: int = 123
    calls: list[tuple[str, int | None]] = field(default_factory=list)

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
) -> RagService:
    return RagService(
        retriever=retriever,
        generator=generator or ExtractiveGenerator(),
        sufficiency_policy=SufficiencyPolicy(minimum_score=minimum_score),
        citation_validator=validator,
        context_top_k=2,
    )


def test_service_returns_validated_answer_and_limits_context() -> None:
    retriever = FakeRetriever((make_retrieved(), make_retrieved(1), make_retrieved(2)))

    answer = make_service(retriever).ask("Question?", language=Language.EN)

    assert answer.status is AnswerStatus.ANSWERED
    assert answer.latency_ms is not None
    assert answer.latency_ms >= 0
    assert answer.retrieved_chunk_ids == ("chunk-0", "chunk-1")
    assert retriever.calls == [("Question?", 2)]


def test_service_abstains_in_requested_language_when_evidence_is_insufficient() -> None:
    retriever = FakeRetriever((make_retrieved(score=0.2),))

    answer = make_service(retriever).ask("Question ?", language=Language.FR)

    assert answer.status is AnswerStatus.ABSTAINED
    assert answer.text == "Je ne sais pas à partir du snapshot fourni."
    assert "below the threshold" in (answer.refusal_reason or "")
    assert not answer.claims
    assert not answer.citations


def test_service_fails_closed_on_citation_validation_error() -> None:
    retriever = FakeRetriever((make_retrieved(),))

    answer = make_service(retriever, validator=RejectingValidator()).ask("Question?")

    assert answer.status is AnswerStatus.ABSTAINED
    assert answer.refusal_reason == "generated evidence failed exact citation validation"


def test_service_fails_closed_on_generation_value_error() -> None:
    retriever = FakeRetriever((make_retrieved(),))

    answer = make_service(retriever, generator=FailingGenerator()).ask("Question?")

    assert answer.status is AnswerStatus.ABSTAINED
    assert answer.refusal_reason == "generated evidence failed exact citation validation"
