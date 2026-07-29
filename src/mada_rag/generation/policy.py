"""Fail-closed evidence sufficiency policy."""

from __future__ import annotations

from dataclasses import dataclass

from mada_rag.generation.relevance import (
    concept_coverage,
    has_grounding_anchor,
    missing_critical_concepts,
)
from mada_rag.models import RetrievedChunk


@dataclass(frozen=True, slots=True)
class SufficiencyDecision:
    sufficient: bool
    reason: str


@dataclass(frozen=True, slots=True)
class SufficiencyPolicy:
    """Require enough candidates and a calibrated minimum top score."""

    minimum_score: float = 0.45
    minimum_candidates: int = 1
    minimum_concept_coverage: float = 0.8

    def __post_init__(self) -> None:
        if not -1.0 <= self.minimum_score <= 1.0:
            raise ValueError("minimum_score must be between -1 and 1")
        if self.minimum_candidates <= 0:
            raise ValueError("minimum_candidates must be positive")
        if not 0.0 <= self.minimum_concept_coverage <= 1.0:
            raise ValueError("minimum_concept_coverage must be between 0 and 1")

    def assess(
        self,
        candidates: tuple[RetrievedChunk, ...],
        *,
        question: str | None = None,
    ) -> SufficiencyDecision:
        if not candidates:
            return SufficiencyDecision(False, "no evidence was retrieved")
        if len(candidates) < self.minimum_candidates:
            return SufficiencyDecision(False, "too few evidence chunks were retrieved")
        dense_scores = [
            candidate.dense_score for candidate in candidates if candidate.dense_score is not None
        ]
        evidence_score = max(dense_scores) if dense_scores else candidates[0].score
        if evidence_score < self.minimum_score:
            return SufficiencyDecision(False, "the best evidence score is below the threshold")
        revision_id = candidates[0].chunk.revision_id
        if any(candidate.chunk.revision_id != revision_id for candidate in candidates):
            return SufficiencyDecision(False, "retrieved evidence mixes snapshot revisions")
        if question is not None:
            evidence = tuple(candidate.chunk.text for candidate in candidates)
            missing_attributes = missing_critical_concepts(question, evidence)
            if missing_attributes:
                return SufficiencyDecision(
                    False,
                    "retrieved evidence is missing critical query concepts: "
                    + ", ".join(sorted(missing_attributes)),
                )
            coverage = concept_coverage(question, evidence)
            if coverage < self.minimum_concept_coverage:
                if coverage >= 0.5 and has_grounding_anchor(question, evidence):
                    return SufficiencyDecision(
                        True,
                        "retrieved evidence has anchored non-critical concept coverage",
                    )
                return SufficiencyDecision(
                    False,
                    "retrieved evidence does not cover enough query concepts "
                    f"({coverage:.3f} < {self.minimum_concept_coverage:.3f})",
                )
        return SufficiencyDecision(True, "retrieved evidence passed the configured policy")
