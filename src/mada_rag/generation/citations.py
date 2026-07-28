"""Structural and exact-span citation validation."""

from __future__ import annotations

from mada_rag.models import Answer, AnswerStatus, RetrievedChunk


class CitationValidationError(RuntimeError):
    """Raised when an affirmative answer is not closed over retrieved evidence."""


class CitationValidator:
    """Verify every citation against exact text in the retrieved chunk."""

    def validate(
        self,
        answer: Answer,
        retrieved: tuple[RetrievedChunk, ...],
    ) -> None:
        if answer.status is AnswerStatus.ABSTAINED:
            return
        chunks = {candidate.chunk.chunk_id: candidate.chunk for candidate in retrieved}
        if len(chunks) != len(retrieved):
            raise CitationValidationError("retrieved chunk IDs are not unique")
        retrieved_ids = tuple(candidate.chunk.chunk_id for candidate in retrieved)
        if answer.retrieved_chunk_ids != retrieved_ids:
            raise CitationValidationError(
                "answer retrieved_chunk_ids differ from the chunks actually retrieved"
            )

        citations = {citation.citation_id: citation for citation in answer.citations}
        for citation in answer.citations:
            chunk = chunks.get(citation.chunk_id)
            if chunk is None:
                raise CitationValidationError("citation references a chunk not retrieved")
            if (
                citation.revision_id != answer.revision_id
                or chunk.revision_id != answer.revision_id
            ):
                raise CitationValidationError("citation revision differs from answer revision")
            if citation.start_char is None or citation.end_char is None:
                raise CitationValidationError("extractive citations require exact offsets")
            if citation.end_char > len(chunk.text):
                raise CitationValidationError("citation offsets exceed chunk text")
            if chunk.text[citation.start_char : citation.end_char] != citation.excerpt:
                raise CitationValidationError("citation excerpt is not exact at its offsets")
            if citation.section_path != chunk.section_path:
                raise CitationValidationError("citation section differs from chunk section")
            if str(citation.source_url) != str(chunk.source_url):
                raise CitationValidationError("citation source URL differs from chunk source")
            if citation.source_anchor != chunk.source_anchor:
                raise CitationValidationError("citation anchor differs from chunk anchor")
            if citation.table_id != chunk.table_id or citation.row_index != chunk.row_index:
                raise CitationValidationError("citation table provenance differs from chunk")

        for claim in answer.claims:
            claim_citations = [
                citations[citation_id]
                for citation_id in claim.citation_ids
                if citation_id in citations
            ]
            if not claim_citations or not any(
                citation.excerpt == claim.text for citation in claim_citations
            ):
                raise CitationValidationError("extractive claim is not an exact cited excerpt")
