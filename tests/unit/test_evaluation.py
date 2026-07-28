"""Offline tests for reproducible retrieval and answer evaluation."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path

import pytest

from mada_rag.evaluation import (
    EvaluationDataError,
    evaluate,
    load_evaluation_cases,
    write_evaluation_report,
)
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
)


def chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        revision_id=7,
        chunk_type=ChunkType.TEXT,
        section_id="section",
        section_path=("Section",),
        ordinal=0,
        text=text,
        token_count=len(text.split()),
        content_sha256=sha256(text.encode()).hexdigest(),
        source_url="https://en.wikipedia.org/wiki/Madagascar",
    )


def retrieved(value: Chunk, rank: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=value,
        method=RetrievalMethod.DENSE,
        rank=rank,
        score=1.0 / rank,
        dense_rank=rank,
        dense_score=1.0 / rank,
    )


@dataclass
class FakeRetriever:
    chunks: tuple[Chunk, ...]
    results: dict[str, tuple[RetrievedChunk, ...]]
    revision_id: int = 7
    error: Exception | None = None

    def retrieve(self, query: str, *, top_k: int | None = None) -> tuple[RetrievedChunk, ...]:
        if self.error is not None:
            raise self.error
        values = self.results[query]
        return values if top_k is None else values[:top_k]


@dataclass
class FakeService:
    answers: dict[str, Answer]

    def ask(self, question: str, *, language: Language = Language.EN) -> Answer:
        assert language is self.answers[question].language
        return self.answers[question]


def answer(case: EvalCase, evidence: Chunk) -> Answer:
    citation = Citation(
        citation_id="citation-1",
        chunk_id=evidence.chunk_id,
        revision_id=7,
        section_path=evidence.section_path,
        excerpt=evidence.text,
        source_url=evidence.source_url,
        start_char=0,
        end_char=len(evidence.text),
    )
    return Answer(
        question=case.question,
        language=case.language,
        status=AnswerStatus.ANSWERED,
        text=evidence.text,
        revision_id=7,
        claims=(
            Claim(
                claim_id="claim-1",
                text=evidence.text,
                citation_ids=("citation-1",),
                supported=True,
            ),
        ),
        citations=(citation,),
        retrieved_chunk_ids=(evidence.chunk_id,),
        provider="test",
        model="test",
    )


def abstention(case: EvalCase) -> Answer:
    return Answer(
        question=case.question,
        language=case.language,
        status=AnswerStatus.ABSTAINED,
        text="I do not know from the supplied snapshot.",
        revision_id=7,
        refusal_reason="test trap",
        provider="test",
        model="test",
    )


def case(case_id: str, *, answerable: bool, expected_ids: tuple[str, ...]) -> EvalCase:
    return EvalCase(
        case_id=case_id,
        question=case_id,
        language=Language.EN,
        category=EvalCategory.SIMPLE_FACT if answerable else EvalCategory.OUT_OF_SCOPE,
        revision_id=7,
        answerable=answerable,
        expected_answer="answer" if answerable else None,
        expected_chunk_ids=expected_ids,
    )


def tick_clock() -> Iterator[float]:
    value = 0.0
    while True:
        yield value
        value += 0.01


def test_evaluate_measures_retrieval_citations_abstention_and_latency() -> None:
    first = chunk("first", "First evidence.")
    second = chunk("second", "Second evidence.")
    distractor = chunk("distractor", "Distractor evidence.")
    answerable = case("answerable", answerable=True, expected_ids=("first", "second"))
    trap = case("trap", answerable=False, expected_ids=())
    retriever = FakeRetriever(
        chunks=(first, second, distractor),
        results={
            "answerable": (retrieved(distractor, 1), retrieved(first, 2), retrieved(second, 3)),
            "trap": (retrieved(distractor, 1),),
        },
    )
    service = FakeService({"answerable": answer(answerable, first), "trap": abstention(trap)})
    ticks = tick_clock()

    report = evaluate(
        (answerable, trap),
        retrievers={"dense": retriever},
        services={"dense": service},
        top_k=3,
        snapshot_sha256="a" * 64,
        index_hashes={"dense": {"index_sha256": "b" * 64}},
        generated_at=datetime_from_iso("2026-01-01T00:00:00+00:00"),
        clock=lambda: next(ticks),
    )

    metrics = report.modes["dense"].metrics
    assert metrics.retrieval_evaluated_cases == 1
    assert metrics.recall_at_k == 1.0
    assert metrics.mrr == 0.5
    assert metrics.ndcg_at_k == pytest.approx(0.6934264)
    assert metrics.citation_precision == metrics.citation_validity == 1.0
    assert metrics.abstention_accuracy == 1.0
    assert metrics.retrieval_latency.samples == metrics.answer_latency.samples == 2
    assert report.to_dict()["snapshot_sha256"] == "a" * 64


def test_evaluate_reports_retrieval_errors_without_synthetic_metrics() -> None:
    evidence = chunk("evidence", "Evidence.")
    answerable = case("answerable", answerable=True, expected_ids=("evidence",))
    report = evaluate(
        (answerable,),
        retrievers={
            "dense": FakeRetriever(
                chunks=(evidence,),
                results={},
                error=RuntimeError("local index unavailable"),
            )
        },
    )

    mode = report.modes["dense"]
    assert mode.metrics.recall_at_k is mode.metrics.mrr is mode.metrics.ndcg_at_k is None
    assert mode.cases[0].errors[0].phase == "retrieve"
    assert mode.cases[0].errors[0].message == "local index unavailable"


def test_jsonl_loader_filters_strict_cases_and_report_writes_atomically(tmp_path: Path) -> None:
    first = case("first", answerable=True, expected_ids=("first",))
    second = case("second", answerable=False, expected_ids=())
    cases_path = tmp_path / "questions.jsonl"
    cases_path.write_text(
        first.model_dump_json() + "\n" + second.model_dump_json() + "\n",
        encoding="utf-8",
    )

    assert load_evaluation_cases(cases_path, case_ids=("second",)) == (second,)
    with pytest.raises(EvaluationDataError, match="unknown evaluation case IDs"):
        load_evaluation_cases(cases_path, case_ids=("missing",))

    evidence = chunk("first", "Evidence.")
    report = evaluate((first,), retrievers={"dense": FakeRetriever((evidence,), {"first": ()})})
    report_path = tmp_path / "report.json"
    write_evaluation_report(report, report_path)
    assert '"revision_id": 7' in report_path.read_text(encoding="utf-8")


def datetime_from_iso(value: str) -> datetime:
    return datetime.fromisoformat(value)
