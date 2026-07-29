"""Fail-closed structured LLM generation with an injected, offline client."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import pytest

from mada_rag.generation import CitationValidator, ExtractiveGenerator
from mada_rag.generation.structured_llm import (
    StructuredLLMError,
    StructuredLLMGenerator,
    StructuredLLMUnavailableError,
)
from mada_rag.models import Chunk, ChunkType, Language, RetrievalMethod, RetrievedChunk

EVIDENCE = "Antananarivo is the capital of Madagascar."


def candidate() -> RetrievedChunk:
    chunk = Chunk(
        chunk_id="capital-evidence",
        revision_id=1365949107,
        chunk_type=ChunkType.TEXT,
        section_id="lead",
        section_path=("Lead",),
        ordinal=0,
        text=EVIDENCE,
        token_count=len(EVIDENCE.split()),
        content_sha256=hashlib.sha256(EVIDENCE.encode()).hexdigest(),
        source_url="https://en.wikipedia.org/wiki/Madagascar",
        source_anchor="Lead",
    )
    return RetrievedChunk(
        chunk=chunk,
        method=RetrievalMethod.DENSE,
        rank=1,
        score=0.9,
        dense_rank=1,
        dense_score=0.9,
    )


@dataclass
class FakeCompletionClient:
    response: str | None = None
    error: Exception | None = None
    calls: list[dict[str, object]] = field(default_factory=list)

    def complete(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        timeout_seconds: float,
    ) -> str:
        self.calls.append(
            {
                "model": model,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "timeout_seconds": timeout_seconds,
            }
        )
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response


def response_payload(*, text: str, claim: str, citation_ids: list[str] | None = None) -> str:
    return json.dumps(
        {
            "text": text,
            "claims": [
                {
                    "claim_id": "claim-1",
                    "text": claim,
                    "citation_ids": citation_ids or ["citation-1"],
                }
            ],
            "citations": [
                {
                    "citation_id": "citation-1",
                    "chunk_id": "capital-evidence",
                    "excerpt": EVIDENCE,
                }
            ],
        }
    )


@pytest.mark.parametrize(
    ("provider", "base_url", "language", "text", "claim", "prompt_language"),
    [
        (
            "openai",
            None,
            Language.EN,
            "Antananarivo is Madagascar's capital.",
            "Antananarivo is Madagascar's capital.",
            "English",
        ),
        (
            "openai-compatible",
            "https://llm.example.test/v1",
            Language.FR,
            "Antananarivo est la capitale de Madagascar.",
            "Antananarivo est la capitale de Madagascar.",
            "French",
        ),
    ],
)
def test_structured_payload_rebuilds_exact_provenance_and_keeps_answer_language(
    provider: str,
    base_url: str | None,
    language: Language,
    text: str,
    claim: str,
    prompt_language: str,
) -> None:
    client = FakeCompletionClient(response=response_payload(text=text, claim=claim))
    factory_calls: list[tuple[str, str | None]] = []

    def factory(api_key: str, configured_base_url: str | None) -> FakeCompletionClient:
        factory_calls.append((api_key, configured_base_url))
        return client

    generator = StructuredLLMGenerator(
        provider=provider,  # type: ignore[arg-type]
        api_key="test-key",
        model_name="test-model",
        base_url=base_url,
        timeout_seconds=12.5,
        client_factory=factory,
    )

    assert factory_calls == []
    answer = generator.generate("What is the capital?", language, (candidate(),))

    assert factory_calls == [("test-key", base_url)]
    assert answer.text == text
    assert answer.claims[0].text == claim
    assert answer.language is language
    assert answer.provider == provider
    assert answer.model == "test-model"
    citation = answer.citations[0]
    assert citation.excerpt == EVIDENCE
    assert citation.start_char is not None
    assert citation.end_char is not None
    assert candidate().chunk.text[citation.start_char : citation.end_char] == EVIDENCE
    assert citation.chunk_id == "capital-evidence"
    assert citation.revision_id == 1365949107
    CitationValidator().validate(answer, (candidate(),))
    assert client.calls[0]["model"] == "test-model"
    assert client.calls[0]["timeout_seconds"] == 12.5
    assert prompt_language in str(client.calls[0]["system_prompt"])
    assert EVIDENCE in str(client.calls[0]["user_prompt"])


@pytest.mark.parametrize(
    "payload",
    [
        json.dumps(
            {
                "text": "Forged source.",
                "claims": [
                    {
                        "claim_id": "claim-1",
                        "text": "Forged source.",
                        "citation_ids": ["citation-1"],
                    }
                ],
                "citations": [
                    {
                        "citation_id": "citation-1",
                        "chunk_id": "not-retrieved",
                        "excerpt": EVIDENCE,
                    }
                ],
            }
        ),
        json.dumps(
            {
                "text": "Forged excerpt.",
                "claims": [
                    {
                        "claim_id": "claim-1",
                        "text": "Forged excerpt.",
                        "citation_ids": ["citation-1"],
                    }
                ],
                "citations": [
                    {
                        "citation_id": "citation-1",
                        "chunk_id": "capital-evidence",
                        "excerpt": "Antananarivo is the official capital.",
                    }
                ],
            }
        ),
        response_payload(
            text="Claim with a forged citation ID.",
            claim="Claim with a forged citation ID.",
            citation_ids=["not-in-citations"],
        ),
    ],
)
def test_structured_generation_rejects_forged_chunk_excerpt_and_claim_references(
    payload: str,
) -> None:
    generator = StructuredLLMGenerator(
        provider="openai",
        api_key="test-key",
        model_name="test-model",
        client_factory=lambda _key, _base_url: FakeCompletionClient(response=payload),
    )

    with pytest.raises(StructuredLLMError, match=r"local validation|not retrieved|not exact"):
        generator.generate("What is the capital?", Language.EN, (candidate(),))


@pytest.mark.parametrize(
    "response",
    [
        "not JSON",
        json.dumps({"text": "Missing claims and citations."}),
        json.dumps({"text": "Wrong claims.", "claims": [], "citations": []}),
    ],
)
def test_malformed_json_and_pydantic_failures_are_fail_closed(response: str) -> None:
    generator = StructuredLLMGenerator(
        provider="openai",
        api_key="test-key",
        model_name="test-model",
        client_factory=lambda _key, _base_url: FakeCompletionClient(response=response),
    )

    with pytest.raises(StructuredLLMError, match="local validation"):
        generator.generate("What is the capital?", Language.EN, (candidate(),))


def test_injected_api_error_is_wrapped_as_unavailable() -> None:
    generator = StructuredLLMGenerator(
        provider="openai",
        api_key="test-key",
        model_name="test-model",
        client_factory=lambda _key, _base_url: FakeCompletionClient(error=OSError("offline")),
    )

    with pytest.raises(StructuredLLMUnavailableError, match="request failed"):
        generator.generate("What is the capital?", Language.EN, (candidate(),))


def test_missing_key_never_constructs_a_client_and_extractive_fallback_is_available() -> None:
    factory_called = False

    def factory(_api_key: str, _base_url: str | None) -> FakeCompletionClient:
        nonlocal factory_called
        factory_called = True
        return FakeCompletionClient(response=response_payload(text="unused", claim="unused"))

    with pytest.raises(ValueError, match="non-empty API key"):
        StructuredLLMGenerator(
            provider="openai",
            api_key="",
            model_name="test-model",
            client_factory=factory,
        )

    assert not factory_called
    assert ExtractiveGenerator().provider_name == "extractive"
