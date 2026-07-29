"""Integrity checks for the evaluation holdout, without tuning on it."""

from __future__ import annotations

import re
from pathlib import Path

from mada_rag.evaluation import load_evaluation_cases
from mada_rag.storage import load_chunks

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_PATH = REPOSITORY_ROOT / "data" / "eval" / "questions.jsonl"
HOLDOUT_PATH = REPOSITORY_ROOT / "data" / "eval" / "holdout.jsonl"
CHUNKS_PATH = REPOSITORY_ROOT / "data" / "processed" / "chunks.json"


def _normalise_question(question: str) -> str:
    return re.sub(r"\s+", " ", question).strip().casefold()


def test_holdout_loads_and_is_disjoint_from_calibration_cases() -> None:
    calibration = load_evaluation_cases(CALIBRATION_PATH)
    holdout = load_evaluation_cases(HOLDOUT_PATH)

    calibration_ids = {case.case_id for case in calibration}
    holdout_ids = {case.case_id for case in holdout}
    calibration_questions = {_normalise_question(case.question) for case in calibration}
    holdout_questions = {_normalise_question(case.question) for case in holdout}

    assert len(holdout) == len(holdout_ids)
    assert len(holdout) == len(holdout_questions)
    assert not calibration_ids & holdout_ids
    assert not calibration_questions & holdout_questions


def test_holdout_evidence_is_exact_and_uses_the_calibration_snapshot() -> None:
    calibration = load_evaluation_cases(CALIBRATION_PATH)
    holdout = load_evaluation_cases(HOLDOUT_PATH)
    chunks_by_id = {chunk.chunk_id: chunk for chunk in load_chunks(CHUNKS_PATH)}

    assert (
        {case.revision_id for case in holdout}
        == {case.revision_id for case in calibration}
        == {1365949107}
    )
    for case in holdout:
        for chunk_id in case.expected_chunk_ids:
            assert chunk_id in chunks_by_id
            assert chunks_by_id[chunk_id].revision_id == case.revision_id
        for excerpt in case.evidence_excerpts:
            assert any(
                excerpt in chunks_by_id[chunk_id].text for chunk_id in case.expected_chunk_ids
            )
