"""Secret-free exact-span generation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from mada_rag.generation.relevance import (
    normalized_tokens,
    ordering_direction,
    query_concepts,
)
from mada_rag.models import (
    Answer,
    AnswerStatus,
    Citation,
    Claim,
    Language,
    RetrievedChunk,
)

_SENTENCE_RE = re.compile(r"[^.!?\n]+(?:[.!?]+|$)")
_NUMBER_RE = re.compile(r"[-+]?\d[\d,\s]*(?:\.\d+)?")
_ENTITY_CONCEPTS = frozenset({"city", "country", "district", "province", "region", "state"})
_DERIVED_MEASURES = frozenset({"area", "density", "percent", "percentage", "ratio", "rate"})


@dataclass(frozen=True, slots=True)
class _EvidenceSpan:
    candidate: RetrievedChunk
    excerpt: str
    start: int
    end: int
    relevance: float


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


def _bounded_excerpt(
    text: str,
    start: int,
    end: int,
    maximum_chars: int,
) -> tuple[str, int, int]:
    while start < end and text[start].isspace():
        start += 1
    available = text[start : min(end, start + maximum_chars)]
    if not available:
        raise ValueError("cannot extract evidence from an empty chunk")
    if end - start > maximum_chars:
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


def _candidate_spans(
    question: str,
    candidate: RetrievedChunk,
    maximum_chars: int,
) -> tuple[_EvidenceSpan, ...]:
    concepts = query_concepts(question)
    spans: list[_EvidenceSpan] = []
    for match in _SENTENCE_RE.finditer(candidate.chunk.text):
        excerpt, start, end = _bounded_excerpt(
            candidate.chunk.text,
            match.start(),
            match.end(),
            maximum_chars,
        )
        if not excerpt:
            continue
        excerpt_concepts = set(normalized_tokens(excerpt))
        overlap = len(concepts & excerpt_concepts)
        coverage = overlap / len(concepts) if concepts else 1.0
        numeric_matches = sum(
            1 for concept in concepts if concept.replace(",", "") in excerpt.replace(",", "")
        )
        relevance = coverage * 10.0 + overlap + numeric_matches + candidate.score * 0.01
        spans.append(_EvidenceSpan(candidate, excerpt, start, end, relevance))
    return tuple(spans)


def _parse_number(value: str) -> float | None:
    match = _NUMBER_RE.search(value.replace("\u2212", "-").replace("\xa0", " "))
    if match is None:
        return None
    normalized = match.group(0).replace(",", "").replace(" ", "")
    try:
        return float(normalized)
    except ValueError:
        return None


def _row_assignments(candidate: RetrievedChunk) -> tuple[dict[str, str], str, int, int] | None:
    for line in candidate.chunk.text.splitlines():
        stripped = line.strip()
        if not stripped.casefold().startswith("row ") or ":" not in stripped:
            continue
        assignments: dict[str, str] = {}
        _prefix, values = stripped.split(":", 1)
        for assignment in values.split(" | "):
            if "=" not in assignment:
                continue
            header, value = assignment.split("=", 1)
            assignments[header.strip()] = value.strip()
        if not assignments:
            return None
        start = candidate.chunk.text.index(stripped)
        return assignments, stripped, start, start + len(stripped)
    return None


def _field_relevance(question: str, headers: set[str]) -> tuple[str | None, float]:
    concepts = query_concepts(question)
    header_concepts = {header: set(normalized_tokens(header)) for header in headers}
    document_frequency = {
        concept: sum(concept in values for values in header_concepts.values())
        for concept in concepts
    }
    best_header: str | None = None
    best_score = 0.0
    for header, values in header_concepts.items():
        overlap = concepts & values
        score = sum(1.0 / max(document_frequency[concept], 1) for concept in overlap)
        score += 2.0 * len(overlap & _DERIVED_MEASURES)
        if score > best_score:
            best_header = header
            best_score = score
    return best_header, best_score


def _table_comparison_spans(
    question: str,
    candidates: tuple[RetrievedChunk, ...],
) -> tuple[_EvidenceSpan, ...]:
    direction = ordering_direction(question)
    if direction is None:
        return ()
    groups: dict[str, list[tuple[RetrievedChunk, dict[str, str], str, int, int]]] = {}
    for candidate in candidates:
        if candidate.chunk.table_id is None or candidate.chunk.row_index is None:
            continue
        parsed = _row_assignments(candidate)
        if parsed is None:
            continue
        assignments, excerpt, start, end = parsed
        groups.setdefault(candidate.chunk.table_id, []).append(
            (candidate, assignments, excerpt, start, end)
        )

    concepts = query_concepts(question)
    best_spans: tuple[_EvidenceSpan, ...] = ()
    best_table_score = 0.0
    for rows in groups.values():
        headers = {
            header
            for _candidate, assignments, _excerpt, _start, _end in rows
            for header in assignments
        }
        field, field_score = _field_relevance(question, headers)
        if field is None or field_score <= 0:
            continue
        numeric_rows: list[tuple[float, RetrievedChunk, str, int, int]] = []
        for candidate, assignments, excerpt, start, end in rows:
            if concepts & _ENTITY_CONCEPTS and "total" in normalized_tokens(excerpt):
                continue
            value = _parse_number(assignments.get(field, ""))
            if value is not None:
                numeric_rows.append((value, candidate, excerpt, start, end))
        if len(numeric_rows) < 2:
            continue
        winning_value = (
            max(value for value, *_rest in numeric_rows)
            if direction == "maximum"
            else min(value for value, *_rest in numeric_rows)
        )
        winners = tuple(
            _EvidenceSpan(candidate, excerpt, start, end, field_score + candidate.score * 0.01)
            for value, candidate, excerpt, start, end in numeric_rows
            if value == winning_value
        )
        if winners and field_score > best_table_score:
            best_spans = winners
            best_table_score = field_score
    return best_spans


def _select_evidence(
    question: str,
    candidates: tuple[RetrievedChunk, ...],
    *,
    maximum_claims: int,
    maximum_chars: int,
) -> tuple[_EvidenceSpan, ...]:
    table_winners = _table_comparison_spans(question, candidates)
    if table_winners:
        return table_winners[:maximum_claims]
    spans = [
        span
        for candidate in candidates
        for span in _candidate_spans(question, candidate, maximum_chars)
    ]
    spans.sort(
        key=lambda span: (
            -span.relevance,
            span.candidate.rank,
            span.start,
        )
    )
    selected: list[_EvidenceSpan] = []
    seen: set[str] = set()
    for span in spans:
        if span.excerpt in seen:
            continue
        seen.add(span.excerpt)
        selected.append(span)
        if len(selected) == maximum_claims:
            break
    return tuple(selected)


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

        selected = _select_evidence(
            question,
            candidates,
            maximum_claims=self.max_claims,
            maximum_chars=self.max_excerpt_chars,
        )
        claims: list[Claim] = []
        citations: list[Citation] = []
        for evidence in selected:
            candidate = evidence.candidate
            excerpt = evidence.excerpt
            start = evidence.start
            end = evidence.end
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
