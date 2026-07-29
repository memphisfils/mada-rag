"""Fail-closed structured generation through OpenAI-compatible chat APIs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from mada_rag.models import Answer, AnswerStatus, Chunk, Citation, Claim, Language, RetrievedChunk

ProviderName = Literal["openai", "openai-compatible"]


class StructuredLLMError(ValueError):
    """Base error for an untrusted or unavailable structured LLM response."""


class StructuredLLMUnavailableError(StructuredLLMError):
    """Raised when the optional SDK or configured provider is unavailable."""


class CompletionClient(Protocol):
    """Small injectable seam around a provider-specific chat completion client."""

    def complete(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        timeout_seconds: float,
    ) -> str: ...


CompletionClientFactory = Callable[[str, str | None], CompletionClient]


class _ModelCitation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    citation_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    excerpt: str = Field(min_length=1)


class _ModelClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    claim_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    citation_ids: tuple[str, ...] = Field(min_length=1)


class _ModelResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    text: str = Field(min_length=1)
    claims: tuple[_ModelClaim, ...] = Field(min_length=1)
    citations: tuple[_ModelCitation, ...] = Field(min_length=1)


class _OpenAICompletionClient:
    """Adapter loaded only after the optional provider is actually invoked."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def complete(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        timeout_seconds: float,
    ) -> str:
        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=(
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ),
                response_format={"type": "json_object"},
                temperature=0,
                timeout=timeout_seconds,
            )
            choices = response.choices
            content = choices[0].message.content if choices else None
        except Exception as exc:
            raise StructuredLLMUnavailableError("structured LLM request failed") from exc
        if not isinstance(content, str) or not content.strip():
            raise StructuredLLMError("structured LLM returned no JSON content")
        return content


def _default_client_factory(api_key: str, base_url: str | None) -> CompletionClient:
    try:
        from openai import OpenAI  # type: ignore[import-not-found]
    except ImportError as exc:
        raise StructuredLLMUnavailableError(
            "OpenAI support is not installed; install the optional openai dependency"
        ) from exc
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
    except Exception as exc:
        raise StructuredLLMUnavailableError(
            "could not initialize the structured LLM client"
        ) from exc
    return _OpenAICompletionClient(client)


class StructuredLLMGenerator:
    """Generate localized claims while reconstructing citations from local chunks."""

    def __init__(
        self,
        *,
        provider: ProviderName,
        api_key: str,
        model_name: str,
        base_url: str | None = None,
        timeout_seconds: float = 30.0,
        client_factory: CompletionClientFactory | None = None,
    ) -> None:
        if provider not in {"openai", "openai-compatible"}:
            raise ValueError("provider must be openai or openai-compatible")
        if not api_key.strip():
            raise ValueError("a non-empty API key is required for structured generation")
        if not model_name.strip():
            raise ValueError("model_name cannot be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if provider == "openai-compatible" and not base_url:
            raise ValueError("openai-compatible generation requires base_url")
        self._provider = provider
        self._api_key = api_key
        self._model_name = model_name
        self._base_url = base_url
        self._timeout_seconds = timeout_seconds
        self._client_factory = client_factory or _default_client_factory
        self._client: CompletionClient | None = None

    @property
    def provider_name(self) -> str:
        return self._provider

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(
        self,
        question: str,
        language: Language,
        candidates: tuple[RetrievedChunk, ...],
    ) -> Answer:
        if not candidates:
            raise StructuredLLMError("structured generation requires retrieved evidence")
        chunks = {candidate.chunk.chunk_id: candidate.chunk for candidate in candidates}
        if len(chunks) != len(candidates):
            raise StructuredLLMError("retrieved evidence contains duplicate chunk IDs")
        revision_id = candidates[0].chunk.revision_id
        if any(chunk.revision_id != revision_id for chunk in chunks.values()):
            raise StructuredLLMError("retrieved evidence mixes snapshot revisions")

        try:
            raw_response = self._get_client().complete(
                model=self.model_name,
                system_prompt=_system_prompt(language),
                user_prompt=_user_prompt(question, language, candidates),
                timeout_seconds=self._timeout_seconds,
            )
        except StructuredLLMError:
            raise
        except Exception as exc:
            raise StructuredLLMUnavailableError("structured LLM request failed") from exc
        try:
            response = _ModelResponse.model_validate_json(raw_response)
            citations = _build_citations(response.citations, chunks, revision_id)
            answer = Answer(
                question=question,
                language=language,
                status=AnswerStatus.ANSWERED,
                text=response.text,
                revision_id=revision_id,
                claims=tuple(
                    Claim(
                        claim_id=claim.claim_id,
                        text=claim.text,
                        citation_ids=claim.citation_ids,
                        supported=True,
                    )
                    for claim in response.claims
                ),
                citations=citations,
                retrieved_chunk_ids=tuple(candidate.chunk.chunk_id for candidate in candidates),
                provider=self.provider_name,
                model=self.model_name,
            )
        except (ValidationError, ValueError, TypeError) as exc:
            raise StructuredLLMError("structured LLM response failed local validation") from exc
        return answer

    def _get_client(self) -> CompletionClient:
        if self._client is None:
            try:
                self._client = self._client_factory(self._api_key, self._base_url)
            except StructuredLLMError:
                raise
            except Exception as exc:
                raise StructuredLLMUnavailableError(
                    "could not initialize the structured LLM client"
                ) from exc
        return self._client


def _build_citations(
    model_citations: tuple[_ModelCitation, ...],
    chunks: dict[str, Chunk],
    revision_id: int,
) -> tuple[Citation, ...]:
    """Create provenance from chunks; the model can only select exact spans."""

    citation_ids = [citation.citation_id for citation in model_citations]
    if len(citation_ids) != len(set(citation_ids)):
        raise StructuredLLMError("structured LLM reused a citation ID")
    citations: list[Citation] = []
    for model_citation in model_citations:
        chunk = chunks.get(model_citation.chunk_id)
        if chunk is None:
            raise StructuredLLMError("structured LLM cited a chunk that was not retrieved")
        start = chunk.text.find(model_citation.excerpt)
        if start < 0:
            raise StructuredLLMError("structured LLM citation excerpt is not exact evidence")
        citations.append(
            Citation(
                citation_id=model_citation.citation_id,
                chunk_id=chunk.chunk_id,
                revision_id=revision_id,
                section_path=chunk.section_path,
                excerpt=model_citation.excerpt,
                source_url=chunk.source_url,
                source_anchor=chunk.source_anchor,
                table_id=chunk.table_id,
                row_index=chunk.row_index,
                start_char=start,
                end_char=start + len(model_citation.excerpt),
            )
        )
    return tuple(citations)


def _system_prompt(language: Language) -> str:
    answer_language = "French" if language is Language.FR else "English"
    return (
        "You are a source-bounded answer generator. Use only the evidence blocks supplied by "
        "the user. Do not use background knowledge, infer missing facts, or cite anything outside "
        "those blocks. Answer and claim text must be in "
        f"{answer_language}. Citation excerpts must remain exact character-for-character copies of "
        "the supplied evidence, even when the evidence is in another language. Return one JSON "
        "object only, with this exact schema: "
        '{"text":"...","claims":[{"claim_id":"...","text":"...","citation_ids":["..."]}],'
        '"citations":[{"citation_id":"...","chunk_id":"...","excerpt":"..."}]}. '
        "Every claim must cite at least one citation. If the evidence is insufficient, return JSON "
        "that does not match this schema rather than inventing an answer."
    )


def _user_prompt(
    question: str,
    language: Language,
    candidates: tuple[RetrievedChunk, ...],
) -> str:
    evidence = "\n\n".join(
        f"[chunk_id={candidate.chunk.chunk_id}]\n{candidate.chunk.text}" for candidate in candidates
    )
    return (
        f"Question language: {language.value}\nQuestion: {question}\n\nEvidence blocks:\n{evidence}"
    )
