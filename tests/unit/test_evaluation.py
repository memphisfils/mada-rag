"""Offline tests for reproducible retrieval and answer evaluation."""

from __future__ import annotations

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


def case(
    case_id: str,
    *,
    answerable: bool,
    expected_ids: tuple[str, ...],
    expected_answer: str | None = None,
) -> EvalCase:
    return EvalCase(
        case_id=case_id,
        question=case_id,
        language=Language.EN,
        category=EvalCategory.SIMPLE_FACT if answerable else EvalCategory.OUT_OF_SCOPE,
        revision_id=7,
        answerable=answerable,
        expected_answer=expected_answer if answerable else None,
        expected_chunk_ids=expected_ids,
    )


def test_evaluate_measures_retrieval_citations_abstention_and_latency() -> None:
    first = chunk("first", "First evidence.")
    second = chunk("second", "Second evidence.")
    distractor = chunk("distractor", "Distractor evidence.")
    answerable = case(
        "answerable",
        answerable=True,
        expected_ids=("first", "second"),
        expected_answer=first.text,
    )
    trap = case("trap", answerable=False, expected_ids=())
    retriever = FakeRetriever(
        chunks=(first, second, distractor),
        results={
            "answerable": (retrieved(distractor, 1), retrieved(first, 2), retrieved(second, 3)),
            "trap": (retrieved(distractor, 1),),
        },
    )
    service = FakeService({"answerable": answer(answerable, first), "trap": abstention(trap)})
    ticks = iter((0.00, 0.01, 0.01, 0.04, 0.04, 0.06, 0.06, 0.11))

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
    assert report.schema_version == "2.0"
    assert metrics.retrieval_evaluated_cases == 1
    assert metrics.evidence_recall_at_k == 1.0
    assert metrics.mrr == 0.5
    assert metrics.ndcg_at_k == pytest.approx(0.6934264)
    assert metrics.hit_rate_at_k == metrics.complete_evidence_rate_at_k == 1.0
    assert metrics.citation_precision == metrics.citation_validity == 1.0
    assert metrics.answer_accuracy == 1.0
    assert metrics.answer_accuracy_correct == metrics.answer_accuracy_total == 1
    assert metrics.answerability_status_accuracy == 1.0
    assert metrics.answerability_status_correct == metrics.answerability_status_total == 2
    assert metrics.trap_false_positive_rate == 0.0
    assert metrics.trap_false_positives == 0
    assert metrics.trap_total == 1
    assert metrics.retrieval_latency.samples == 2
    assert metrics.retrieval_latency.cold_ms == pytest.approx(10.0)
    assert metrics.retrieval_latency.warm_samples == 1
    assert metrics.retrieval_latency.warm_mean_ms == pytest.approx(20.0)
    assert metrics.retrieval_latency.max_ms == pytest.approx(20.0)
    assert metrics.retrieval_latency.warm_max_ms == pytest.approx(20.0)
    assert metrics.answer_latency.samples == 2
    assert metrics.answer_latency.cold_ms == pytest.approx(30.0)
    assert metrics.answer_latency.warm_samples == 1
    assert metrics.answer_latency.warm_mean_ms == pytest.approx(50.0)
    assert metrics.answer_latency.max_ms == pytest.approx(50.0)
    assert metrics.answer_latency.warm_max_ms == pytest.approx(50.0)
    answer_record, trap_record = report.modes["dense"].cases
    assert answer_record.answer_text == first.text
    expected_answer = answer(answerable, first)
    assert answer_record.claims == (expected_answer.claims[0].model_dump(mode="json"),)
    assert answer_record.citations == (expected_answer.citations[0].model_dump(mode="json"),)
    assert trap_record.answer_text == "I do not know from the supplied snapshot."
    assert trap_record.claims == trap_record.citations == ()
    assert report.to_dict()["snapshot_sha256"] == "a" * 64


def test_evaluate_records_answer_text_mismatches_and_trap_false_positives() -> None:
    evidence = chunk("evidence", "Snapshot evidence.")
    answerable = case(
        "answerable",
        answerable=True,
        expected_ids=("evidence",),
        expected_answer="Different expected answer.",
    )
    trap = case("trap", answerable=False, expected_ids=())
    retriever = FakeRetriever(
        chunks=(evidence,),
        results={
            "answerable": (retrieved(evidence, 1),),
            "trap": (retrieved(evidence, 1),),
        },
    )
    service = FakeService(
        {
            "answerable": answer(answerable, evidence),
            "trap": answer(trap, evidence),
        }
    )
    report = evaluate(
        (answerable, trap),
        retrievers={"dense": retriever},
        services={"dense": service},
    )

    mode = report.modes["dense"]
    assert mode.metrics.answer_accuracy == 0.0
    assert mode.metrics.answer_accuracy_correct == 0
    assert mode.metrics.answer_accuracy_total == 1
    assert mode.metrics.answerability_status_accuracy == 0.5
    assert mode.metrics.trap_false_positive_rate == 1.0
    assert mode.metrics.trap_false_positives == mode.metrics.trap_total == 1
    assert mode.cases[0].answer_text == evidence.text
    assert mode.cases[1].answer_status == AnswerStatus.ANSWERED.value


def test_evaluate_reports_retrieval_errors_without_synthetic_metrics() -> None:
    evidence = chunk("evidence", "Evidence.")
    answerable = case(
        "answerable",
        answerable=True,
        expected_ids=("evidence",),
        expected_answer="Evidence.",
    )
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
    assert mode.metrics.evidence_recall_at_k is mode.metrics.mrr is mode.metrics.ndcg_at_k is None
    assert mode.metrics.hit_rate_at_k is mode.metrics.complete_evidence_rate_at_k is None
    assert mode.metrics.answer_accuracy is mode.metrics.answerability_status_accuracy is None
    assert mode.metrics.trap_false_positive_rate is None
    assert mode.cases[0].errors[0].phase == "retrieve"
    assert mode.cases[0].errors[0].message == "local index unavailable"


def test_jsonl_loader_filters_strict_cases_and_report_writes_atomically(tmp_path: Path) -> None:
    first = case(
        "first",
        answerable=True,
        expected_ids=("first",),
        expected_answer="Evidence.",
    )
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
