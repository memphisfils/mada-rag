"""Offline, reproducible retrieval and grounded-answer evaluation."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Protocol

from pydantic import ValidationError

from mada_rag.models import Answer, AnswerStatus, Chunk, EvalCase, Language, RetrievedChunk
from mada_rag.retrieval import Retriever
from mada_rag.storage import atomic_write_bytes


class EvaluationDataError(ValueError):
    """Raised when local evaluation inputs cannot be trusted."""


class AnsweringService(Protocol):
    """Minimal injectable answer contract used by the harness."""

    def ask(self, question: str, *, language: Language = Language.EN) -> Answer: ...


@dataclass(frozen=True, slots=True)
class EvaluationError:
    """One explicit per-case failure; failed cases never become metrics."""

    phase: str
    error_type: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"phase": self.phase, "error_type": self.error_type, "message": self.message}


@dataclass(frozen=True, slots=True)
class LatencyMetrics:
    """Observed latency distribution, in milliseconds."""

    samples: int
    mean_ms: float | None
    p50_ms: float | None
    p95_ms: float | None

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "samples": self.samples,
            "mean_ms": self.mean_ms,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
        }


@dataclass(frozen=True, slots=True)
class ModeMetrics:
    """Metrics with explicit denominators instead of synthetic zeroes."""

    retrieval_evaluated_cases: int
    recall_at_k: float | None
    mrr: float | None
    ndcg_at_k: float | None
    citation_precision: float | None
    citation_precision_count: int
    citation_precision_total: int
    citation_validity: float | None
    citation_valid_count: int
    citation_total: int
    abstention_accuracy: float | None
    abstention_correct: int
    abstention_total: int
    retrieval_latency: LatencyMetrics
    answer_latency: LatencyMetrics

    def to_dict(self) -> dict[str, object]:
        return {
            "retrieval_evaluated_cases": self.retrieval_evaluated_cases,
            "recall_at_k": self.recall_at_k,
            "mrr": self.mrr,
            "ndcg_at_k": self.ndcg_at_k,
            "citation_precision": self.citation_precision,
            "citation_precision_count": self.citation_precision_count,
            "citation_precision_total": self.citation_precision_total,
            "citation_validity": self.citation_validity,
            "citation_valid_count": self.citation_valid_count,
            "citation_total": self.citation_total,
            "abstention_accuracy": self.abstention_accuracy,
            "abstention_correct": self.abstention_correct,
            "abstention_total": self.abstention_total,
            "retrieval_latency": self.retrieval_latency.to_dict(),
            "answer_latency": self.answer_latency.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class CaseEvaluation:
    """Audit record for one case and one retrieval mode."""

    case_id: str
    answerable: bool
    expected_chunk_ids: tuple[str, ...]
    retrieved_chunk_ids: tuple[str, ...]
    retrieval_latency_ms: float | None
    answer_status: str | None
    answer_latency_ms: float | None
    citation_count: int
    errors: tuple[EvaluationError, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "answerable": self.answerable,
            "expected_chunk_ids": list(self.expected_chunk_ids),
            "retrieved_chunk_ids": list(self.retrieved_chunk_ids),
            "retrieval_latency_ms": self.retrieval_latency_ms,
            "answer_status": self.answer_status,
            "answer_latency_ms": self.answer_latency_ms,
            "citation_count": self.citation_count,
            "errors": [error.to_dict() for error in self.errors],
        }


@dataclass(frozen=True, slots=True)
class ModeEvaluation:
    """All records and aggregate measurements for one pipeline."""

    mode: str
    metrics: ModeMetrics
    cases: tuple[CaseEvaluation, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "metrics": self.metrics.to_dict(),
            "cases": [case.to_dict() for case in self.cases],
        }


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """JSON-serializable evaluation artifact tied to one local snapshot."""

    schema_version: str
    generated_at: datetime
    revision_id: int
    snapshot_sha256: str | None
    index_hashes: Mapping[str, Mapping[str, str]]
    parameters: Mapping[str, object]
    case_count: int
    modes: Mapping[str, ModeEvaluation]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at.isoformat(),
            "revision_id": self.revision_id,
            "snapshot_sha256": self.snapshot_sha256,
            "index_hashes": {
                mode: dict(hashes) for mode, hashes in sorted(self.index_hashes.items())
            },
            "parameters": dict(self.parameters),
            "case_count": self.case_count,
            "modes": {mode: result.to_dict() for mode, result in sorted(self.modes.items())},
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, sort_keys=True) + "\n"


def load_evaluation_cases(
    path: Path,
    *,
    case_ids: Iterable[str] | None = None,
) -> tuple[EvalCase, ...]:
    """Load strict JSONL cases without consulting any external source."""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise EvaluationDataError(f"cannot read evaluation cases from {path}") from exc

    cases: list[EvalCase] = []
    known_ids: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            case = EvalCase.model_validate_json(line)
        except (ValidationError, ValueError) as exc:
            raise EvaluationDataError(f"invalid evaluation case at {path}:{line_number}") from exc
        if case.case_id in known_ids:
            raise EvaluationDataError(f"duplicate evaluation case_id {case.case_id!r}")
        known_ids.add(case.case_id)
        cases.append(case)
    if not cases:
        raise EvaluationDataError(f"evaluation case file {path} is empty")

    requested = tuple(case_ids or ())
    if not requested:
        return tuple(cases)
    requested_set = set(requested)
    unknown = requested_set - known_ids
    if unknown:
        raise EvaluationDataError("unknown evaluation case IDs: " + ", ".join(sorted(unknown)))
    selected = tuple(case for case in cases if case.case_id in requested_set)
    if not selected:
        raise EvaluationDataError("no evaluation cases were selected")
    return selected


def write_evaluation_report(report: EvaluationReport, path: Path) -> None:
    """Persist a report atomically so an interrupted run cannot look complete."""

    atomic_write_bytes(path, report.to_json().encode("utf-8"))


def evaluate(
    cases: Sequence[EvalCase],
    *,
    retrievers: Mapping[str, Retriever],
    services: Mapping[str, AnsweringService] | None = None,
    top_k: int = 5,
    snapshot_sha256: str | None = None,
    index_hashes: Mapping[str, Mapping[str, str]] | None = None,
    parameters: Mapping[str, object] | None = None,
    generated_at: datetime | None = None,
    clock: Callable[[], float] = perf_counter,
) -> EvaluationReport:
    """Evaluate injected local pipelines against one immutable case revision.

    The harness never builds indexes, invokes models, or fetches sources itself;
    callers own those dependencies and can inject deterministic test doubles.
    """

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not cases:
        raise EvaluationDataError("at least one evaluation case is required")
    if not retrievers:
        raise EvaluationDataError("at least one retriever is required")
    case_revisions = {case.revision_id for case in cases}
    if len(case_revisions) != 1:
        raise EvaluationDataError("evaluation cases must share one snapshot revision")
    revision_id = next(iter(case_revisions))
    for mode, retriever in retrievers.items():
        if not mode.strip():
            raise EvaluationDataError("evaluation mode names cannot be empty")
        if retriever.revision_id != revision_id:
            raise EvaluationDataError(
                f"retriever {mode!r} revision {retriever.revision_id} differs from cases"
            )
    if services is not None and not set(services) <= set(retrievers):
        raise EvaluationDataError("answer services must correspond to configured retrievers")

    timestamp = generated_at or datetime.now(UTC)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise EvaluationDataError("generated_at must be timezone-aware")
    report_modes: dict[str, ModeEvaluation] = {}
    for mode, retriever in retrievers.items():
        report_modes[mode] = _evaluate_mode(
            mode=mode,
            cases=cases,
            retriever=retriever,
            service=None if services is None else services.get(mode),
            top_k=top_k,
            clock=clock,
        )
    return EvaluationReport(
        schema_version="1.0",
        generated_at=timestamp,
        revision_id=revision_id,
        snapshot_sha256=snapshot_sha256,
        index_hashes=index_hashes or {},
        parameters={"top_k": top_k, **(parameters or {})},
        case_count=len(cases),
        modes=report_modes,
    )


run_evaluation = evaluate


def _evaluate_mode(
    *,
    mode: str,
    cases: Sequence[EvalCase],
    retriever: Retriever,
    service: AnsweringService | None,
    top_k: int,
    clock: Callable[[], float],
) -> ModeEvaluation:
    corpus = {chunk.chunk_id: chunk for chunk in retriever.chunks}
    retrieval_latency_values: list[float] = []
    answer_latency_values: list[float] = []
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    citation_precision_count = 0
    citation_precision_total = 0
    citation_valid_count = 0
    citation_total = 0
    abstention_correct = 0
    abstention_total = 0
    records: list[CaseEvaluation] = []

    for case in cases:
        errors: list[EvaluationError] = []
        retrieved: tuple[RetrievedChunk, ...] = ()
        retrieval_latency_ms: float | None = None
        try:
            started = clock()
            retrieved = retriever.retrieve(case.question, top_k=top_k)
            retrieval_latency_ms = (clock() - started) * 1_000
            retrieval_latency_values.append(retrieval_latency_ms)
            _validate_retrieved_snapshot(retrieved, case.revision_id)
        except Exception as exc:
            errors.append(_error("retrieve", exc))

        retrieved_ids = tuple(candidate.chunk.chunk_id for candidate in retrieved[:top_k])
        if case.answerable and case.expected_chunk_ids and not errors:
            expected = set(case.expected_chunk_ids)
            matched = sum(chunk_id in expected for chunk_id in retrieved_ids)
            recalls.append(matched / len(expected))
            reciprocal_ranks.append(_reciprocal_rank(retrieved_ids, expected))
            ndcgs.append(_ndcg(retrieved_ids, expected, top_k=top_k))

        answer: Answer | None = None
        answer_latency_ms: float | None = None
        if service is not None:
            try:
                started = clock()
                answer = service.ask(case.question, language=case.language)
                answer_latency_ms = (clock() - started) * 1_000
                answer_latency_values.append(answer_latency_ms)
                if answer.revision_id != case.revision_id:
                    raise EvaluationDataError("answer revision differs from evaluation case")
                abstention_total += 1
                expected_status = (
                    AnswerStatus.ANSWERED if case.answerable else AnswerStatus.ABSTAINED
                )
                if answer.status is expected_status:
                    abstention_correct += 1
                valid, total, relevant = _citation_measurements(answer, corpus, case)
                citation_valid_count += valid
                citation_total += total
                if case.answerable and case.expected_chunk_ids:
                    citation_precision_count += relevant
                    citation_precision_total += total
            except Exception as exc:
                errors.append(_error("ask", exc))

        records.append(
            CaseEvaluation(
                case_id=case.case_id,
                answerable=case.answerable,
                expected_chunk_ids=case.expected_chunk_ids,
                retrieved_chunk_ids=retrieved_ids,
                retrieval_latency_ms=retrieval_latency_ms,
                answer_status=None if answer is None else answer.status.value,
                answer_latency_ms=answer_latency_ms,
                citation_count=0 if answer is None else len(answer.citations),
                errors=tuple(errors),
            )
        )

    metrics = ModeMetrics(
        retrieval_evaluated_cases=len(recalls),
        recall_at_k=_mean_or_none(recalls),
        mrr=_mean_or_none(reciprocal_ranks),
        ndcg_at_k=_mean_or_none(ndcgs),
        citation_precision=_ratio_or_none(citation_precision_count, citation_precision_total),
        citation_precision_count=citation_precision_count,
        citation_precision_total=citation_precision_total,
        citation_validity=_ratio_or_none(citation_valid_count, citation_total),
        citation_valid_count=citation_valid_count,
        citation_total=citation_total,
        abstention_accuracy=_ratio_or_none(abstention_correct, abstention_total),
        abstention_correct=abstention_correct,
        abstention_total=abstention_total,
        retrieval_latency=_latencies(retrieval_latency_values),
        answer_latency=_latencies(answer_latency_values),
    )
    return ModeEvaluation(mode=mode, metrics=metrics, cases=tuple(records))


def _validate_retrieved_snapshot(
    retrieved: Sequence[RetrievedChunk],
    revision_id: int,
) -> None:
    if any(candidate.chunk.revision_id != revision_id for candidate in retrieved):
        raise EvaluationDataError("retrieved chunks differ from the evaluation snapshot")


def _citation_measurements(
    answer: Answer,
    corpus: Mapping[str, Chunk],
    case: EvalCase,
) -> tuple[int, int, int]:
    valid = 0
    relevant = 0
    for citation in answer.citations:
        chunk = corpus.get(citation.chunk_id)
        is_valid = (
            citation.chunk_id in answer.retrieved_chunk_ids
            and citation.revision_id == answer.revision_id == case.revision_id
            and chunk is not None
            and citation.section_path == chunk.section_path
            and str(citation.source_url) == str(chunk.source_url)
            and citation.source_anchor == chunk.source_anchor
            and citation.table_id == chunk.table_id
            and citation.row_index == chunk.row_index
            and citation.start_char is not None
            and citation.end_char is not None
            and citation.end_char <= len(chunk.text)
            and chunk.text[citation.start_char : citation.end_char] == citation.excerpt
        )
        valid += int(is_valid)
        relevant += int(citation.chunk_id in case.expected_chunk_ids)
    return valid, len(answer.citations), relevant


def _reciprocal_rank(retrieved_ids: Sequence[str], expected: set[str]) -> float:
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in expected:
            return 1.0 / rank
    return 0.0


def _ndcg(retrieved_ids: Sequence[str], expected: set[str], *, top_k: int) -> float:
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, chunk_id in enumerate(retrieved_ids[:top_k], start=1)
        if chunk_id in expected
    )
    ideal_count = min(len(expected), top_k)
    if ideal_count == 0:
        return 0.0
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return dcg / ideal_dcg


def _latencies(values: Sequence[float]) -> LatencyMetrics:
    if not values:
        return LatencyMetrics(samples=0, mean_ms=None, p50_ms=None, p95_ms=None)
    ordered = sorted(values)
    return LatencyMetrics(
        samples=len(ordered),
        mean_ms=sum(ordered) / len(ordered),
        p50_ms=_percentile(ordered, 0.5),
        p95_ms=_percentile(ordered, 0.95),
    )


def _percentile(ordered: Sequence[float], fraction: float) -> float:
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _mean_or_none(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _ratio_or_none(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _error(phase: str, exc: Exception) -> EvaluationError:
    return EvaluationError(phase=phase, error_type=type(exc).__name__, message=str(exc))
