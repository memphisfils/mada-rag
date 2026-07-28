"""Local FAISS index and dense retrieval contract tests."""

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from mada_rag.indexing import DenseIndex, DenseIndexError, DenseIndexIntegrityError
from mada_rag.models import Chunk, ChunkType, RetrievalMethod
from mada_rag.retrieval import DenseRetriever

pytest.importorskip("faiss")

REVISION_ID = 123
SOURCE_URL = "https://en.wikipedia.org/wiki/Madagascar"


def make_chunk(index: int, *, revision_id: int = REVISION_ID) -> Chunk:
    text = ("alpha evidence", "beta evidence", "gamma evidence")[index]
    return Chunk(
        chunk_id=f"chunk-{index}",
        revision_id=revision_id,
        chunk_type=ChunkType.TEXT,
        section_id=f"section-{index}",
        section_path=(f"Section {index}",),
        ordinal=index,
        text=text,
        token_count=2,
        content_sha256=hashlib.sha256(text.encode()).hexdigest(),
        source_url=SOURCE_URL,
    )


class FakeEmbedder:
    model_name = "fake/e5"
    dimension = 3

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        vectors = {
            "alpha evidence": (10.0, 0.0, 0.0),
            "beta evidence": (0.0, 2.0, 0.0),
            "gamma evidence": (1.0, 1.0, 0.0),
        }
        return np.asarray([vectors[text] for text in texts], dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        assert text
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)


def make_index() -> DenseIndex:
    return DenseIndex.build(tuple(make_chunk(index) for index in range(3)), FakeEmbedder())


def test_build_normalizes_and_searches_in_similarity_order() -> None:
    index = make_index()

    matches = index.search(np.array([2.0, 0.0, 0.0], dtype=np.float32), top_k=3)

    assert index.dimension == 3
    assert index.revision_id == REVISION_ID
    assert [position for _score, position in matches] == [0, 2, 1]
    assert matches[0][0] == pytest.approx(1.0)
    assert matches[1][0] == pytest.approx(2**-0.5)


def test_save_and_load_preserve_manifest_chunks_and_results(tmp_path: Path) -> None:
    index = make_index()
    manifest = index.save(tmp_path)
    loaded = DenseIndex.load(
        tmp_path,
        expected_revision_id=REVISION_ID,
        expected_embedding_model="fake/e5",
        expected_dimension=3,
    )

    assert manifest.chunk_count == 3
    assert loaded.manifest == manifest
    assert loaded.chunks == index.chunks
    assert loaded.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), top_k=3) == index.search(
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        top_k=3,
    )


@pytest.mark.parametrize("filename", ["index.faiss", "chunks.json"])
def test_tampered_artifact_is_rejected(tmp_path: Path, filename: str) -> None:
    make_index().save(tmp_path)
    target = tmp_path / filename
    content = target.read_bytes()
    if filename == "chunks.json":
        tampered = content.replace(b"alpha evidence", b"Alpha evidence", 1)
        assert tampered != content
        target.write_bytes(tampered)
    else:
        target.write_bytes(content + b"tampered")

    with pytest.raises(DenseIndexIntegrityError, match="SHA-256"):
        DenseIndex.load(tmp_path)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"expected_revision_id": REVISION_ID + 1}, "revision differs"),
        ({"expected_embedding_model": "another/model"}, "embedding model differs"),
        ({"expected_dimension": 4}, "dimension differs"),
    ],
)
def test_load_rejects_wrong_expectations(
    tmp_path: Path, kwargs: dict[str, object], message: str
) -> None:
    make_index().save(tmp_path)

    with pytest.raises(DenseIndexIntegrityError, match=message):
        DenseIndex.load(tmp_path, **kwargs)


def test_build_rejects_mixed_revisions_and_duplicate_ids() -> None:
    with pytest.raises(DenseIndexError, match="one revision"):
        DenseIndex.build(
            (make_chunk(0), make_chunk(1, revision_id=REVISION_ID + 1)),
            FakeEmbedder(),
        )

    duplicate = make_chunk(0).model_copy(update={"text": "different evidence"})
    with pytest.raises(DenseIndexError, match="unique"):
        DenseIndex.build((make_chunk(0), duplicate), FakeEmbedder())


def test_search_rejects_wrong_query_dimension() -> None:
    with pytest.raises(DenseIndexError, match="dimension"):
        make_index().search(np.array([1.0, 0.0], dtype=np.float32), top_k=1)


def test_dense_retriever_exposes_contiguous_ranks_and_scores() -> None:
    retriever = DenseRetriever(make_index(), FakeEmbedder(), default_top_k=3)

    results = retriever.retrieve("alpha", top_k=3)

    assert [result.chunk.chunk_id for result in results] == [
        "chunk-0",
        "chunk-2",
        "chunk-1",
    ]
    assert [result.rank for result in results] == [1, 2, 3]
    assert all(result.method is RetrievalMethod.DENSE for result in results)
    assert all(result.dense_rank == result.rank for result in results)
    assert all(result.dense_score == result.score for result in results)


def test_dense_retriever_rejects_embedding_model_mismatch() -> None:
    embedder = FakeEmbedder()
    embedder.model_name = "another/model"

    with pytest.raises(DenseIndexIntegrityError, match="embedding model"):
        DenseRetriever(make_index(), embedder)
