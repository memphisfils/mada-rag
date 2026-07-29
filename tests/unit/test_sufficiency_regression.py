"""Offline regressions for bilingual grounding and abstention policy."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import pytest

from mada_rag.evaluation import load_evaluation_cases
from mada_rag.generation import SufficiencyPolicy
from mada_rag.models import Chunk, RetrievalMethod, RetrievedChunk
from mada_rag.storage import load_chunks

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_CASES_PATH = REPOSITORY_ROOT / "data" / "eval" / "questions.jsonl"
EVALUATION_CHUNKS_FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "evaluation_chunks.json"
)


@lru_cache(maxsize=1)
def fixture_chunks() -> tuple[Chunk, ...]:
    return load_chunks(EVALUATION_CHUNKS_FIXTURE)


@lru_cache(maxsize=1)
def chunks_by_id() -> dict[str, Chunk]:
    return {chunk.chunk_id: chunk for chunk in fixture_chunks()}


@lru_cache(maxsize=1)
def evaluation_cases():
    return load_evaluation_cases(EVALUATION_CASES_PATH)


def retrieved(*chunk_ids: str) -> tuple[RetrievedChunk, ...]:
    return tuple(
        RetrievedChunk(
            chunk=chunks_by_id()[chunk_id],
            method=RetrievalMethod.DENSE,
            rank=rank,
            score=0.9,
            dense_rank=rank,
            dense_score=0.9,
        )
        for rank, chunk_id in enumerate(chunk_ids, start=1)
    )


def synthetic_candidate(text: str) -> RetrievedChunk:
    template = fixture_chunks()[0]
    chunk = template.model_copy(
        update={
            "chunk_id": "synthetic-grounding-candidate",
            "text": text,
            "token_count": len(text.split()),
            "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
        }
    )
    return RetrievedChunk(
        chunk=chunk,
        method=RetrievalMethod.DENSE,
        rank=1,
        score=0.9,
        dense_rank=1,
        dense_score=0.9,
    )


def test_all_answerable_cases_pass_with_their_expected_snapshot_evidence() -> None:
    policy = SufficiencyPolicy()
    answerable_cases = tuple(case for case in evaluation_cases() if case.answerable)

    assert len(answerable_cases) == 22
    assert all(case.expected_chunk_ids for case in answerable_cases)
    assert all(
        chunk_id in chunks_by_id()
        for case in answerable_cases
        for chunk_id in case.expected_chunk_ids
    )
    for case in answerable_cases:
        decision = policy.assess(
            retrieved(*case.expected_chunk_ids),
            question=case.question,
        )

        assert decision.sufficient, f"{case.case_id}: {decision.reason}"


@pytest.mark.parametrize(
    ("case_id", "topical_chunk_id"),
    [
        ("trap-national-dish-fr", "chunk-99242aac25101075735f086ff30995a1"),
        ("trap-official-flower-en", "chunk-74b1e95de6b47222f7e70a3a44b6b4ad"),
        ("trap-prime-minister-salary-fr", "chunk-1ae4be98401b54786d3903f2ffa57eae"),
    ],
)
def test_traps_stay_refused_with_thematically_related_chunks(
    case_id: str,
    topical_chunk_id: str,
) -> None:
    case = next(value for value in evaluation_cases() if value.case_id == case_id)

    decision = SufficiencyPolicy().assess(
        retrieved(topical_chunk_id),
        question=case.question,
    )

    assert not case.answerable
    assert not decision.sufficient
    assert "missing critical query concepts" in decision.reason


def test_one_generic_overlap_cannot_bypass_the_coverage_threshold() -> None:
    decision = SufficiencyPolicy().assess(
        (synthetic_candidate("context"),),
        question="What context belongs to?",
    )

    assert not decision.sufficient
    assert "does not cover enough query concepts" in decision.reason


@pytest.mark.parametrize(
    ("question", "evidence"),
    [
        ("What context belongs to 1960?", "context 1960"),
        ("What context belongs to group?", "context group"),
    ],
)
def test_number_anchor_or_two_noncritical_concepts_can_relax_coverage(
    question: str,
    evidence: str,
) -> None:
    decision = SufficiencyPolicy().assess(
        (synthetic_candidate(evidence),),
        question=question,
    )

    assert decision.sufficient
    assert decision.reason == "retrieved evidence has anchored non-critical concept coverage"


@pytest.mark.parametrize(
    "question",
    [
        "Which region has a population density of 198.0 people per km2?",
        "Quelle région a une densité de population de 198,0 habitants par km² ?",
    ],
)
def test_decimal_and_unit_variants_require_real_density_evidence(question: str) -> None:
    policy = SufficiencyPolicy()
    generic_madagascar = retrieved("chunk-2d28841c865831121b84bbeb7dff068b")
    analamanga_row = retrieved("chunk-58597e3558fa501ad857dd1e0bafd663")

    generic_decision = policy.assess(generic_madagascar, question=question)
    matching_decision = policy.assess(analamanga_row, question=question)

    assert not generic_decision.sufficient
    assert "missing critical query concepts" in generic_decision.reason
    assert matching_decision.sufficient
