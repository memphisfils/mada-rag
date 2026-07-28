"""Secret-free exact-span generation."""

from __future__ import annotations

from typing import Protocol

from mada_rag.models import (
    Answer,
    AnswerStatus,
    Citation,
    Claim,
    Language,
    RetrievedChunk,
)


class AnswerGenerator(Protocol):
    """Provider interface shared by local extraction and future LLM adapters."""

    @property
    def provider_name(self) -> str: ...

    @property
    def model_name(self) -> str: ...

    def generate(
        self,
        question: str,
        language: Language,
        candidates: tuple[RetrievedChunk, ...],
    ) -> Answer: ...


def _exact_excerpt(text: str, maximum_chars: int) -> tuple[str, int, int]:
    start = len(text) - len(text.lstrip())
    available = text[start : start + maximum_chars]
    if not available:
        raise ValueError("cannot extract evidence from an empty chunk")
    if start + len(available) < len(text):
        boundaries = [available.rfind(marker) for marker in (". ", "? ", "! ", "\n")]
        boundary = max(boundaries)
        if boundary >= 31:
            available = available[: boundary + 1]
        else:
            word_boundary = available.rfind(" ")
            if word_boundary >= 31:
                available = available[:word_boundary]
    excerpt = available.rstrip()
    return excerpt, start, start + len(excerpt)


class ExtractiveGenerator:
    """Return exact evidence spans and no model-authored factual prose."""

    def __init__(self, *, max_claims: int = 3, max_excerpt_chars: int = 500) -> None:
        if max_claims <= 0:
            raise ValueError("max_claims must be positive")
        if max_excerpt_chars < 32:
            raise ValueError("max_excerpt_chars must be at least 32")
        self.max_claims = max_claims
        self.max_excerpt_chars = max_excerpt_chars

    @property
    def provider_name(self) -> str:
        return "extractive"

    @property
    def model_name(self) -> str:
        return "exact-span-v1"

    def generate(
        self,
        question: str,
        language: Language,
        candidates: tuple[RetrievedChunk, ...],
    ) -> Answer:
        if not candidates:
            raise ValueError("extractive generation requires retrieved evidence")

        claims: list[Claim] = []
        citations: list[Citation] = []
        excerpts_seen: set[str] = set()
        for candidate in candidates:
            excerpt, start, end = _exact_excerpt(
                candidate.chunk.text,
                self.max_excerpt_chars,
            )
            if excerpt in excerpts_seen:
                continue
            excerpts_seen.add(excerpt)
            number = len(claims) + 1
            citation_id = f"citation-{number}"
            claims.append(
                Claim(
                    claim_id=f"claim-{number}",
                    text=excerpt,
                    citation_ids=(citation_id,),
                    supported=True,
                )
            )
            citations.append(
                Citation(
                    citation_id=citation_id,
                    chunk_id=candidate.chunk.chunk_id,
                    revision_id=candidate.chunk.revision_id,
                    section_path=candidate.chunk.section_path,
                    excerpt=excerpt,
                    source_url=candidate.chunk.source_url,
                    source_anchor=candidate.chunk.source_anchor,
                    table_id=candidate.chunk.table_id,
                    row_index=candidate.chunk.row_index,
                    start_char=start,
                    end_char=end,
                )
            )
            if len(claims) == self.max_claims:
                break
        if not claims:
            raise ValueError("no distinct exact excerpts could be generated")

        return Answer(
            question=question,
            language=language,
            status=AnswerStatus.ANSWERED,
            text="\n\n".join(claim.text for claim in claims),
            revision_id=candidates[0].chunk.revision_id,
            claims=tuple(claims),
            citations=tuple(citations),
            retrieved_chunk_ids=tuple(candidate.chunk.chunk_id for candidate in candidates),
            provider=self.provider_name,
            model=self.model_name,
        )
