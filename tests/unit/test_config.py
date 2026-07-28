"""Contract tests for safe and internally consistent application settings."""

import pytest
from pydantic import ValidationError

from mada_rag.config import GenerationProvider, RetrievalMode, Settings


def settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_settings_defaults_are_local_secret_free_and_source_bounded() -> None:
    config = settings()

    assert config.source_page_title == "Madagascar"
    assert str(config.source_canonical_url) == "https://en.wikipedia.org/wiki/Madagascar"
    assert str(config.mediawiki_api_url) == "https://en.wikipedia.org/w/api.php"
    assert config.generation_provider is GenerationProvider.EXTRACTIVE
    assert config.generation_model is None
    assert config.llm_api_key is None
    assert config.llm_base_url is None
    assert config.api_host == "127.0.0.1"
    assert config.retrieval_mode is RetrievalMode.HYBRID
    assert not config.reranker_enabled
    assert config.context_top_k <= config.fused_top_k
    assert config.minimum_retrieved_chunks <= config.context_top_k


def test_settings_are_strict_for_direct_values() -> None:
    with pytest.raises(ValidationError):
        settings(dense_top_k="20")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"context_top_k": 11, "fused_top_k": 10},
            "context_top_k cannot exceed fused_top_k",
        ),
        (
            {
                "retrieval_mode": RetrievalMode.HYBRID_RERANK,
                "reranker_enabled": False,
            },
            "hybrid-rerank mode requires reranker_enabled=true",
        ),
        (
            {"generation_provider": GenerationProvider.DISABLED, "llm_api_key": "secret"},
            "llm_api_key must be unset for local generation providers",
        ),
        (
            {"generation_provider": GenerationProvider.EXTRACTIVE, "llm_api_key": "secret"},
            "llm_api_key must be unset for local generation providers",
        ),
        (
            {
                "generation_provider": GenerationProvider.EXTRACTIVE,
                "llm_base_url": "http://127.0.0.1:1234/v1",
            },
            "llm_base_url must be unset for local generation providers",
        ),
        (
            {"minimum_retrieved_chunks": 6, "context_top_k": 5},
            "minimum_retrieved_chunks cannot exceed context_top_k",
        ),
        (
            {"generation_provider": GenerationProvider.OPENAI},
            "generation_model is required when generation is enabled",
        ),
        (
            {
                "generation_provider": GenerationProvider.OPENAI,
                "generation_model": "test-model",
            },
            "llm_api_key is required for the OpenAI provider",
        ),
        (
            {
                "generation_provider": GenerationProvider.OPENAI_COMPATIBLE,
                "llm_base_url": "http://127.0.0.1:1234/v1",
            },
            "generation_model is required when generation is enabled",
        ),
    ],
)
def test_settings_reject_incoherent_combinations(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        settings(**overrides)


def test_openai_configuration_accepts_model_and_secret() -> None:
    config = settings(
        generation_provider=GenerationProvider.OPENAI,
        generation_model="test-model",
        llm_api_key="secret",
    )

    assert config.generation_provider is GenerationProvider.OPENAI
    assert config.llm_api_key is not None
    assert config.llm_api_key.get_secret_value() == "secret"
    assert "secret" not in repr(config)


def test_openai_compatible_provider_requires_base_url() -> None:
    with pytest.raises(ValidationError, match="base_url"):
        settings(
            generation_provider=GenerationProvider.OPENAI_COMPATIBLE,
            generation_model="local-test-model",
        )
