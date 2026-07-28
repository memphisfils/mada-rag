"""Shared application service for retrieval, sufficiency, and grounded answers."""

from __future__ import annotations

from time import perf_counter

from mada_rag.generation import (
    AnswerGenerator,
    CitationValidationError,
    CitationValidator,
    SufficiencyPolicy,
)
from mada_rag.models import Answer, AnswerStatus, Language, RetrievedChunk
from mada_rag.retrieval import ContextExpander, Retriever


class RagService:
    """One behavior shared by CLI now and API later."""

    def __init__(
        self,
        *,
        retriever: Retriever,
        generator: AnswerGenerator,
        sufficiency_policy: SufficiencyPolicy,
        citation_validator: CitationValidator | None = None,
        context_expander: ContextExpander | None = None,
        context_top_k: int = 5,
    ) -> None:
        if context_top_k <= 0:
            raise ValueError("context_top_k must be positive")
        self.retriever = retriever
        self.generator = generator
        self.sufficiency_policy = sufficiency_policy
        self.citation_validator = citation_validator or CitationValidator()
        self.context_expander = context_expander
        self.context_top_k = context_top_k

    def retrieve(self, question: str, *, top_k: int | None = None) -> tuple[RetrievedChunk, ...]:
        """Return the retriever's ranked output without generation-only expansion."""

        return self.retriever.retrieve(question, top_k=top_k)

    def ask(self, question: str, *, language: Language = Language.EN) -> Answer:
        started = perf_counter()
        candidates = self.retrieve(question, top_k=self.context_top_k)
        if self.context_expander is not None:
            candidates = self.context_expander.expand(candidates)
        decision = self.sufficiency_policy.assess(candidates, question=question)
        if not decision.sufficient:
            return self._abstain(
                question,
                language,
                candidates,
                decision.reason,
                started,
            )
        try:
            answer = self.generator.generate(question, language, candidates)
            self.citation_validator.validate(answer, candidates)
        except (CitationValidationError, ValueError):
            return self._abstain(
                question,
                language,
                candidates,
                "generated evidence failed exact citation validation",
                started,
            )
        return answer.model_copy(update={"latency_ms": (perf_counter() - started) * 1_000})

    def _abstain(
        self,
        question: str,
        language: Language,
        candidates: tuple[RetrievedChunk, ...],
        reason: str,
        started: float,
    ) -> Answer:
        message = (
            "Je ne sais pas à partir du snapshot fourni."
            if language is Language.FR
            else "I do not know from the supplied snapshot."
        )
        return Answer(
            question=question,
            language=language,
            status=AnswerStatus.ABSTAINED,
            text=message,
            revision_id=self.retriever.revision_id,
            retrieved_chunk_ids=tuple(candidate.chunk.chunk_id for candidate in candidates),
            refusal_reason=reason,
            provider=self.generator.provider_name,
            model=self.generator.model_name,
            latency_ms=(perf_counter() - started) * 1_000,
        )
