"""Bilingual relevance-gate requirements."""

from pathlib import Path

import pytest

from mada_rag.generation import SufficiencyPolicy
from mada_rag.generation.relevance import (
    concept_coverage,
    normalized_tokens,
    ordering_direction,
    query_concepts,
)
from mada_rag.models import RetrievalMethod, RetrievedChunk
from mada_rag.storage import load_chunks

LIFE_EXPECTANCY_ID = "chunk-0e65f76953781b67a8443ddd710e1b79"
POWER_CHANGE_IDS = (
    "chunk-6c48d6ccc375bc5d68281fbe1bf05589",
    "chunk-a58c45b78b4d1eda76b143c0dbb0e545",
)
G3_CHUNKS_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "g3_chunks.json"


def evidence(*chunk_ids: str) -> tuple[RetrievedChunk, ...]:
    chunks = load_chunks(G3_CHUNKS_FIXTURE)
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    return tuple(
        RetrievedChunk(
            chunk=by_id[chunk_id],
            method=RetrievalMethod.DENSE,
            rank=rank,
            score=0.9,
            dense_rank=rank,
            dense_score=0.9,
        )
        for rank, chunk_id in enumerate(chunk_ids, start=1)
    )


@pytest.mark.parametrize(
    "question",
    [
        "Who is the current president according to the snapshot?",
        "Qui est l'actuel président selon ce snapshot ?",
    ],
)
def test_current_snapshot_framing_does_not_create_evidence_concepts(question: str) -> None:
    assert query_concepts(question) == frozenset({"president"})


def test_table_and_comparison_framing_is_normalized_separately() -> None:
    question = (
        "Selon le snapshot, quelle région a la plus forte densité "
        "de population dans le tableau de 2018 ?"
    )

    assert ordering_direction(question) == "maximum"
    concepts = query_concepts(question)
    assert {"region", "density", "population", "2018"} <= concepts
    assert not concepts & {"snapshot", "selon", "tableau", "plus"}


def test_required_french_and_english_framing_words_are_stopwords() -> None:
    tokens = set(
        normalized_tokens(
            "snapshot current actuel actuelle selon tableau table plus according",
        )
    )

    assert not tokens & {
        "snapshot",
        "current",
        "actuel",
        "actuelle",
        "selon",
        "tableau",
        "table",
        "plus",
        "according",
    }


@pytest.mark.parametrize(
    ("question", "chunk_ids"),
    [
        (
            "Quelle était l'espérance de vie adulte des hommes et des femmes en 2009 ?",
            (LIFE_EXPECTANCY_ID,),
        ),
        (
            "Comment le pouvoir a-t-il changé de mains entre 2023 et 2025 ?",
            POWER_CHANGE_IDS,
        ),
    ],
)
def test_french_answerable_questions_pass_coverage_with_expected_evidence(
    question: str,
    chunk_ids: tuple[str, ...],
) -> None:
    candidates = evidence(*chunk_ids)

    assert (
        concept_coverage(
            question,
            (candidate.chunk.text for candidate in candidates),
        )
        >= 0.8
    )
    assert SufficiencyPolicy().assess(candidates, question=question).sufficient
