"""Versioned FAISS IndexFlatIP construction, persistence, and validation."""

from __future__ import annotations

import hashlib
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, Self, cast

import numpy as np
from numpy.typing import NDArray

from mada_rag.indexing.e5 import EmbeddingBackend
from mada_rag.models import Chunk, DenseIndexManifest
from mada_rag.storage import (
    StorageError,
    atomic_write_bytes,
    load_chunks,
    load_model,
    serialize_chunks,
    serialize_model,
)

INDEX_FILENAME = "index.faiss"
CHUNKS_FILENAME = "chunks.json"
MANIFEST_FILENAME = "manifest.json"


class DenseIndexError(RuntimeError):
    """Base error for dense index construction and use."""


class DenseIndexConflictError(DenseIndexError):
    """Raised when saving would overwrite an existing artifact."""


class DenseIndexIntegrityError(DenseIndexError):
    """Raised when a saved artifact differs from its manifest."""


class _FaissIndex(Protocol):
    d: int
    ntotal: int
    metric_type: int

    def add(self, vectors: NDArray[np.float32]) -> None: ...

    def search(
        self,
        vectors: NDArray[np.float32],
        top_k: int,
    ) -> tuple[NDArray[np.float32], NDArray[np.int64]]: ...


def _faiss() -> Any:
    import faiss

    return faiss


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError as exc:
        raise DenseIndexIntegrityError(f"cannot read {path}") from exc


def _normalized_matrix(vectors: NDArray[np.float32]) -> NDArray[np.float32]:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise DenseIndexError("embedding matrix must be non-empty and two-dimensional")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.all(np.isfinite(matrix)) or np.any(norms <= 0):
        raise DenseIndexError("embedding matrix contains non-finite or zero vectors")
    return np.ascontiguousarray(matrix / norms, dtype=np.float32)


class DenseIndex:
    """In-memory FAISS index coupled to its ordered, immutable chunks."""

    def __init__(
        self,
        *,
        index: _FaissIndex,
        chunks: tuple[Chunk, ...],
        revision_id: int,
        embedding_model: str,
        dimension: int,
        manifest: DenseIndexManifest | None = None,
    ) -> None:
        if not chunks:
            raise DenseIndexError("dense index cannot be empty")
        if index.ntotal != len(chunks):
            raise DenseIndexError("FAISS vector count must match chunk count")
        if index.d != dimension:
            raise DenseIndexError("FAISS dimension differs from index metadata")
        if any(chunk.revision_id != revision_id for chunk in chunks):
            raise DenseIndexError("all chunks must share the index revision")
        self._index = index
        self.chunks = chunks
        self.revision_id = revision_id
        self.embedding_model = embedding_model
        self.dimension = dimension
        self.manifest = manifest

    @classmethod
    def build(cls, chunks: tuple[Chunk, ...], embedder: EmbeddingBackend) -> Self:
        if not chunks:
            raise DenseIndexError("cannot build an index from zero chunks")
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise DenseIndexError("chunk IDs must be unique")
        revision_id = chunks[0].revision_id
        if any(chunk.revision_id != revision_id for chunk in chunks):
            raise DenseIndexError("all chunks must share one revision")

        vectors = _normalized_matrix(embedder.embed_passages([chunk.text for chunk in chunks]))
        if vectors.shape[0] != len(chunks):
            raise DenseIndexError("embedding row count differs from chunk count")
        dimension = int(vectors.shape[1])
        faiss = _faiss()
        index = cast(_FaissIndex, faiss.IndexFlatIP(dimension))
        index.add(vectors)
        return cls(
            index=index,
            chunks=chunks,
            revision_id=revision_id,
            embedding_model=embedder.model_name,
            dimension=dimension,
        )

    def save(self, directory: Path, *, overwrite: bool = False) -> DenseIndexManifest:
        directory.mkdir(parents=True, exist_ok=True)
        index_path = directory / INDEX_FILENAME
        chunks_path = directory / CHUNKS_FILENAME
        manifest_path = directory / MANIFEST_FILENAME
        targets = (index_path, chunks_path, manifest_path)
        if not overwrite and any(path.exists() for path in targets):
            raise DenseIndexConflictError("dense index artifact already exists")

        chunks_bytes = serialize_chunks(self.chunks)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=directory,
            prefix=".index.",
            suffix=".faiss.tmp",
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        try:
            _faiss().write_index(self._index, str(temporary_path))
            index_bytes = temporary_path.read_bytes()
            os.replace(temporary_path, index_path)
        except OSError as exc:
            raise DenseIndexError("could not save FAISS index") from exc
        finally:
            temporary_path.unlink(missing_ok=True)

        atomic_write_bytes(chunks_path, chunks_bytes)
        manifest = DenseIndexManifest(
            revision_id=self.revision_id,
            embedding_model=self.embedding_model,
            dimension=self.dimension,
            chunk_count=len(self.chunks),
            chunk_ids=tuple(chunk.chunk_id for chunk in self.chunks),
            index_sha256=_sha256_bytes(index_bytes),
            chunks_sha256=_sha256_bytes(chunks_bytes),
            created_at=datetime.now(UTC),
        )
        atomic_write_bytes(manifest_path, serialize_model(manifest))
        self.manifest = manifest
        return manifest

    @classmethod
    def load(
        cls,
        directory: Path,
        *,
        expected_revision_id: int | None = None,
        expected_embedding_model: str | None = None,
        expected_dimension: int | None = None,
    ) -> Self:
        index_path = directory / INDEX_FILENAME
        chunks_path = directory / CHUNKS_FILENAME
        manifest_path = directory / MANIFEST_FILENAME
        if not all(path.is_file() for path in (index_path, chunks_path, manifest_path)):
            raise DenseIndexIntegrityError("dense index requires index, chunks, and manifest files")
        try:
            manifest = load_model(manifest_path, DenseIndexManifest)
            chunks = load_chunks(chunks_path)
        except StorageError as exc:
            raise DenseIndexIntegrityError("dense index metadata is invalid") from exc

        if _sha256_file(index_path) != manifest.index_sha256:
            raise DenseIndexIntegrityError("FAISS index SHA-256 differs from manifest")
        if _sha256_file(chunks_path) != manifest.chunks_sha256:
            raise DenseIndexIntegrityError("chunk corpus SHA-256 differs from manifest")
        if expected_revision_id is not None and manifest.revision_id != expected_revision_id:
            raise DenseIndexIntegrityError("dense index revision differs from expected revision")
        if (
            expected_embedding_model is not None
            and manifest.embedding_model != expected_embedding_model
        ):
            raise DenseIndexIntegrityError(
                "dense index embedding model differs from expected model"
            )
        if expected_dimension is not None and manifest.dimension != expected_dimension:
            raise DenseIndexIntegrityError("dense index dimension differs from expected dimension")
        if tuple(chunk.chunk_id for chunk in chunks) != manifest.chunk_ids:
            raise DenseIndexIntegrityError("chunk ordering or IDs differ from manifest")
        if any(chunk.revision_id != manifest.revision_id for chunk in chunks):
            raise DenseIndexIntegrityError("chunk revision differs from manifest")

        faiss = _faiss()
        index = cast(_FaissIndex, faiss.read_index(str(index_path)))
        if type(index).__name__ != manifest.index_type:
            raise DenseIndexIntegrityError("saved FAISS index type differs from manifest")
        if index.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise DenseIndexIntegrityError("saved FAISS metric is not inner product")
        if index.d != manifest.dimension:
            raise DenseIndexIntegrityError("saved FAISS dimension differs from manifest")
        if index.ntotal != manifest.chunk_count:
            raise DenseIndexIntegrityError("saved FAISS vector count differs from manifest")
        return cls(
            index=index,
            chunks=chunks,
            revision_id=manifest.revision_id,
            embedding_model=manifest.embedding_model,
            dimension=manifest.dimension,
            manifest=manifest,
        )

    def search(
        self,
        query_vector: NDArray[np.float32],
        *,
        top_k: int,
    ) -> tuple[tuple[float, int], ...]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        vector = np.asarray(query_vector, dtype=np.float32)
        if vector.ndim != 1 or vector.shape[0] != self.dimension:
            raise DenseIndexError("query vector dimension differs from dense index")
        matrix = _normalized_matrix(vector.reshape(1, -1))
        scores, positions = self._index.search(matrix, min(top_k, len(self.chunks)))
        return tuple(
            (float(score), int(position))
            for score, position in zip(scores[0], positions[0], strict=True)
            if int(position) >= 0
        )
