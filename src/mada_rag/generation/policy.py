"""Fail-closed evidence sufficiency policy."""

from __future__ import annotations

from dataclasses import dataclass

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

    def __post_init__(self) -> None:
        if not -1.0 <= self.minimum_score <= 1.0:
            raise ValueError("minimum_score must be between -1 and 1")
        if self.minimum_candidates <= 0:
            raise ValueError("minimum_candidates must be positive")

    def assess(self, candidates: tuple[RetrievedChunk, ...]) -> SufficiencyDecision:
        if not candidates:
            return SufficiencyDecision(False, "no evidence was retrieved")
        if len(candidates) < self.minimum_candidates:
            return SufficiencyDecision(False, "too few evidence chunks were retrieved")
        if candidates[0].score < self.minimum_score:
            return SufficiencyDecision(False, "the best evidence score is below the threshold")
        revision_id = candidates[0].chunk.revision_id
        if any(candidate.chunk.revision_id != revision_id for candidate in candidates):
            return SufficiencyDecision(False, "retrieved evidence mixes snapshot revisions")
        return SufficiencyDecision(True, "retrieved evidence passed the configured policy")
