"""Offline tests for lazy multilingual-E5 adaptation."""

from collections.abc import Mapping, Sequence

import numpy as np
import pytest

from mada_rag.indexing import E5Embedder, EmbeddingError


class FakeModel:
    def __init__(self, vectors: object, *, dimension: int | None = 3) -> None:
        self.vectors = vectors
        self.dimension = dimension
        self.calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    def encode(self, sentences: Sequence[str], **kwargs: object) -> object:
        self.calls.append((tuple(sentences), kwargs))
        return self.vectors

    def get_sentence_embedding_dimension(self) -> int | None:
        return self.dimension


class FakeTokenizer:
    def __init__(self, token_ids: object | None = None) -> None:
        self.token_ids = token_ids
        self.calls: list[tuple[str, dict[str, object]]] = []

    def __call__(self, text: str, **kwargs: object) -> Mapping[str, object]:
        self.calls.append((text, kwargs))
        token_ids = self.token_ids
        if token_ids is None:
            token_ids = [101, *range(len(text.split())), 102]
        return {"input_ids": token_ids}


def test_construction_is_lazy_and_factories_are_cached() -> None:
    model = FakeModel(np.array([[1.0, 0.0, 0.0]], dtype=np.float32))
    tokenizer = FakeTokenizer()
    model_names: list[str] = []
    tokenizer_names: list[str] = []

    embedder = E5Embedder(
        "fake/e5",
        model_factory=lambda name: (model_names.append(name), model)[1],
        tokenizer_factory=lambda name: (tokenizer_names.append(name), tokenizer)[1],
    )

    assert model_names == []
    assert tokenizer_names == []
    assert embedder.model_name == "fake/e5"
    assert embedder.dimension == 3
    assert embedder.dimension == 3
    embedder.count_tokens("alpha")
    embedder.count_tokens("beta")
    assert model_names == ["fake/e5"]
    assert tokenizer_names == ["fake/e5"]


def test_e5_prefixes_normalizes_vectors_and_uses_numpy_options() -> None:
    model = FakeModel(
        np.array(
            [
                [3.0, 4.0, 0.0],
                [0.0, 0.0, 2.0],
            ],
            dtype=np.float64,
        )
    )
    embedder = E5Embedder("fake/e5", model_factory=lambda _name: model)

    passages = embedder.embed_passages(("alpha", "beta"))

    assert model.calls[0][0] == ("passage: alpha", "passage: beta")
    assert model.calls[0][1] == {
        "convert_to_numpy": True,
        "normalize_embeddings": False,
        "show_progress_bar": False,
    }
    assert passages.dtype == np.float32
    assert passages.flags.c_contiguous
    np.testing.assert_allclose(np.linalg.norm(passages, axis=1), np.ones(2))

    model.vectors = np.array([[0.0, 5.0, 0.0]], dtype=np.float32)
    query = embedder.embed_query("question")
    assert model.calls[-1][0] == ("query: question",)
    np.testing.assert_allclose(query, np.array([0.0, 1.0, 0.0], dtype=np.float32))


def test_token_count_includes_passage_prefix_and_special_tokens() -> None:
    tokenizer = FakeTokenizer()
    embedder = E5Embedder("fake/e5", tokenizer_factory=lambda _name: tokenizer)

    count = embedder.count_tokens("two words")

    assert count == 5
    assert tokenizer.calls == [
        (
            "passage: two words",
            {"add_special_tokens": True, "truncation": False},
        )
    ]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"model_name": " "}, "model_name cannot be empty"),
        ({"max_sequence_tokens": 0}, "max_sequence_tokens must be positive"),
    ],
)
def test_invalid_configuration_is_rejected(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        E5Embedder(**kwargs)


def test_empty_inputs_are_rejected_without_loading_a_model() -> None:
    model_calls = 0

    def model_factory(_name: str) -> FakeModel:
        nonlocal model_calls
        model_calls += 1
        return FakeModel(np.ones((1, 3), dtype=np.float32))

    embedder = E5Embedder(model_factory=model_factory)
    with pytest.raises(EmbeddingError, match="at least one passage"):
        embedder.embed_passages(())
    with pytest.raises(EmbeddingError, match="passages cannot be empty"):
        embedder.embed_passages((" ",))
    with pytest.raises(EmbeddingError, match="query cannot be empty"):
        embedder.embed_query(" ")
    assert model_calls == 0


@pytest.mark.parametrize(
    "vectors",
    [
        np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        np.array([[np.nan, 1.0, 0.0]], dtype=np.float32),
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
        np.empty((0, 3), dtype=np.float32),
    ],
)
def test_invalid_embedding_matrices_are_rejected(vectors: np.ndarray) -> None:
    model = FakeModel(vectors)
    embedder = E5Embedder(model_factory=lambda _name: model)

    with pytest.raises(EmbeddingError):
        embedder.embed_passages(("alpha",))


def test_invalid_tokenizer_payload_is_rejected() -> None:
    tokenizer = FakeTokenizer(token_ids=123)
    embedder = E5Embedder(tokenizer_factory=lambda _name: tokenizer)

    with pytest.raises(EmbeddingError, match="invalid input_ids"):
        embedder.count_tokens("alpha")


def test_dimension_change_between_calls_is_rejected() -> None:
    model = FakeModel(np.ones((1, 3), dtype=np.float32))
    embedder = E5Embedder(model_factory=lambda _name: model)
    embedder.embed_query("first")
    model.vectors = np.ones((1, 4), dtype=np.float32)

    with pytest.raises(EmbeddingError, match="dimension changed"):
        embedder.embed_query("second")
