"""Lazy multilingual-E5 embeddings with mandatory query/passage prefixes."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

DEFAULT_E5_MODEL = "intfloat/multilingual-e5-base"


class EmbeddingError(RuntimeError):
    """Raised when an embedding backend returns invalid vectors or tokens."""


class EmbeddingBackend(Protocol):
    """Injectable contract shared by index construction and retrieval."""

    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def count_tokens(self, text: str) -> int: ...

    def embed_passages(self, texts: Sequence[str]) -> NDArray[np.float32]: ...

    def embed_query(self, text: str) -> NDArray[np.float32]: ...


class _SentenceModel(Protocol):
    def encode(self, sentences: Sequence[str], **kwargs: object) -> object: ...

    def get_sentence_embedding_dimension(self) -> int | None: ...


class _TokenizerBackend(Protocol):
    def __call__(self, text: str, **kwargs: object) -> Mapping[str, object]: ...


ModelFactory = Callable[[str], _SentenceModel]
TokenizerFactory = Callable[[str], _TokenizerBackend]


def _default_model_factory(model_name: str) -> _SentenceModel:
    from sentence_transformers import SentenceTransformer

    return cast(_SentenceModel, SentenceTransformer(model_name))


def _default_tokenizer_factory(model_name: str) -> _TokenizerBackend:
    from transformers import AutoTokenizer

    return cast(_TokenizerBackend, AutoTokenizer.from_pretrained(model_name))


def _normalize(vectors: NDArray[np.float32]) -> NDArray[np.float32]:
    if vectors.ndim != 2 or vectors.shape[0] == 0 or vectors.shape[1] == 0:
        raise EmbeddingError("embedding matrix must be non-empty and two-dimensional")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.all(np.isfinite(vectors)) or np.any(norms <= 0):
        raise EmbeddingError("embedding backend returned non-finite or zero vectors")
    return np.ascontiguousarray(vectors / norms, dtype=np.float32)


class E5Embedder:
    """Lazy E5 adapter; construction never downloads or loads a model."""

    def __init__(
        self,
        model_name: str = DEFAULT_E5_MODEL,
        *,
        model_factory: ModelFactory | None = None,
        tokenizer_factory: TokenizerFactory | None = None,
        max_sequence_tokens: int = 512,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name cannot be empty")
        if max_sequence_tokens <= 0:
            raise ValueError("max_sequence_tokens must be positive")
        self._model_name = model_name
        self._model_factory = model_factory or _default_model_factory
        self._tokenizer_factory = tokenizer_factory or _default_tokenizer_factory
        self._max_sequence_tokens = max_sequence_tokens
        self._model: _SentenceModel | None = None
        self._tokenizer: _TokenizerBackend | None = None
        self._dimension: int | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            dimension = self._get_model().get_sentence_embedding_dimension()
            if not isinstance(dimension, int) or dimension <= 0:
                raise EmbeddingError("embedding model did not expose a positive dimension")
            self._dimension = dimension
        return self._dimension

    def count_tokens(self, text: str) -> int:
        """Count the exact E5 passage input, including prefix and special tokens."""

        encoded = self._get_tokenizer()(
            f"passage: {text}",
            add_special_tokens=True,
            truncation=False,
        )
        token_ids = encoded.get("input_ids")
        if not isinstance(token_ids, Sequence) or isinstance(token_ids, (str, bytes)):
            raise EmbeddingError("tokenizer returned invalid input_ids")
        count = len(token_ids)
        if count > self._max_sequence_tokens:
            return count
        return count

    def embed_passages(self, texts: Sequence[str]) -> NDArray[np.float32]:
        if not texts:
            raise EmbeddingError("at least one passage is required")
        if any(not text.strip() for text in texts):
            raise EmbeddingError("passages cannot be empty")
        return self._encode(tuple(f"passage: {text}" for text in texts))

    def embed_query(self, text: str) -> NDArray[np.float32]:
        if not text.strip():
            raise EmbeddingError("query cannot be empty")
        return cast(NDArray[np.float32], self._encode((f"query: {text}",))[0])

    def _encode(self, prefixed_texts: Sequence[str]) -> NDArray[np.float32]:
        raw = self._get_model().encode(
            prefixed_texts,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        vectors = np.asarray(raw, dtype=np.float32)
        normalized = _normalize(vectors)
        if self._dimension is None:
            self._dimension = int(normalized.shape[1])
        elif normalized.shape[1] != self._dimension:
            raise EmbeddingError("embedding dimension changed between calls")
        return normalized

    def _get_model(self) -> _SentenceModel:
        if self._model is None:
            self._model = self._model_factory(self._model_name)
        return self._model

    def _get_tokenizer(self) -> _TokenizerBackend:
        if self._tokenizer is None:
            self._tokenizer = self._tokenizer_factory(self._model_name)
        return self._tokenizer
