"""Validated application configuration loaded from environment variables."""

from enum import StrEnum
from pathlib import Path

from pydantic import AnyHttpUrl, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class GenerationProvider(StrEnum):
    """Supported generation backends.

    ``disabled`` is deliberately the default so a fresh checkout neither needs
    a secret nor accidentally sends retrieved evidence to an external service.
    """

    EXTRACTIVE = "extractive"
    DISABLED = "disabled"
    OPENAI = "openai"
    OPENAI_COMPATIBLE = "openai-compatible"


class RetrievalMode(StrEnum):
    """Retrieval pipelines exposed by the CLI, API, and evaluation harness."""

    DENSE = "dense"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid-rerank"


class Settings(BaseSettings):
    """Runtime settings with safe, secret-free defaults."""

    model_config = SettingsConfigDict(
        env_prefix="MADA_RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        strict=True,
        validate_default=True,
    )

    source_page_title: str = "Madagascar"
    source_canonical_url: AnyHttpUrl = AnyHttpUrl("https://en.wikipedia.org/wiki/Madagascar")
    mediawiki_api_url: AnyHttpUrl = AnyHttpUrl("https://en.wikipedia.org/w/api.php")
    mediawiki_user_agent: str = "mada-rag/0.1.0 (https://github.com/memphisfils/mada-rag)"
    mediawiki_timeout_seconds: float = Field(default=30.0, gt=0, le=120)

    snapshot_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    artifact_dir: Path = Path("artifacts")
    evaluation_dir: Path = Path("data/eval")
    parsed_article_path: Path = Path("data/processed/article.json")
    chunks_path: Path = Path("data/processed/chunks.json")
    dense_index_dir: Path = Path("artifacts/indexes/dense")

    embedding_model: str = "intfloat/multilingual-e5-base"
    reranker_enabled: bool = False
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID

    dense_top_k: int = Field(default=20, ge=1, le=100)
    lexical_top_k: int = Field(default=20, ge=1, le=100)
    fused_top_k: int = Field(default=10, ge=1, le=100)
    context_top_k: int = Field(default=5, ge=1, le=50)
    rrf_k: int = Field(default=60, ge=1, le=1_000)

    chunk_target_tokens: int = Field(default=350, ge=32, le=512)
    chunk_max_tokens: int = Field(default=450, ge=32, le=512)
    chunk_overlap_tokens: int = Field(default=50, ge=0, le=128)
    table_chunk_max_tokens: int = Field(default=450, ge=32, le=512)

    dense_score_threshold: float = Field(default=0.45, ge=-1.0, le=1.0)
    minimum_retrieved_chunks: int = Field(default=1, ge=1, le=20)
    minimum_concept_coverage: float = Field(default=0.8, ge=0.0, le=1.0)
    max_expanded_table_chunks: int = Field(default=100, ge=1, le=500)
    extractive_max_claims: int = Field(default=3, ge=1, le=10)
    extractive_max_excerpt_chars: int = Field(default=500, ge=32, le=4_000)
    max_query_chars: int = Field(default=1_000, ge=32, le=10_000)

    generation_provider: GenerationProvider = GenerationProvider.EXTRACTIVE
    generation_model: str | None = None
    llm_api_key: SecretStr | None = None
    llm_base_url: AnyHttpUrl | None = None
    llm_timeout_seconds: float = Field(default=30.0, gt=0, le=300)

    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8000, ge=1, le=65_535)

    @model_validator(mode="after")
    def validate_retrieval_and_provider(self) -> "Settings":
        """Reject inconsistent retrieval or generation configuration."""

        if self.context_top_k > self.fused_top_k:
            raise ValueError("context_top_k cannot exceed fused_top_k")
        if self.minimum_retrieved_chunks > self.context_top_k:
            raise ValueError("minimum_retrieved_chunks cannot exceed context_top_k")
        if self.chunk_target_tokens > self.chunk_max_tokens:
            raise ValueError("chunk_target_tokens cannot exceed chunk_max_tokens")
        if self.chunk_overlap_tokens >= self.chunk_target_tokens:
            raise ValueError("chunk_overlap_tokens must be smaller than chunk_target_tokens")
        if self.retrieval_mode is RetrievalMode.HYBRID_RERANK and not self.reranker_enabled:
            raise ValueError("hybrid-rerank mode requires reranker_enabled=true")
        if self.generation_provider in {
            GenerationProvider.DISABLED,
            GenerationProvider.EXTRACTIVE,
        }:
            if self.llm_api_key is not None:
                raise ValueError("llm_api_key must be unset for local generation providers")
            if self.llm_base_url is not None:
                raise ValueError("llm_base_url must be unset for local generation providers")
            return self
        if not self.generation_model:
            raise ValueError("generation_model is required when generation is enabled")
        if self.generation_provider is GenerationProvider.OPENAI:
            if self.llm_api_key is None:
                raise ValueError("llm_api_key is required for the OpenAI provider")
            return self
        if self.llm_base_url is None:
            raise ValueError("llm_base_url is required for the OpenAI-compatible provider")
        if self.llm_api_key is None:
            raise ValueError("llm_api_key is required for the OpenAI-compatible provider")
        return self
