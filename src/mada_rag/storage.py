"""Atomic, versioned JSON serialization for processed local artifacts."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from pydantic import BaseModel, ValidationError

from mada_rag.models import Chunk, ChunkCorpus, ParsedArticle


class StorageError(RuntimeError):
    """Raised when a local processed artifact is unreadable or invalid."""


def atomic_write_bytes(path: Path, content: bytes) -> None:
    """Flush a temporary file and atomically replace one rebuildable artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except OSError as exc:
        raise StorageError(f"could not write artifact {path}") from exc
    finally:
        temporary_path.unlink(missing_ok=True)


def serialize_model(model: BaseModel) -> bytes:
    """Serialize deterministically enough for hashing and artifact validation."""

    return (model.model_dump_json(indent=2) + "\n").encode("utf-8")


def load_model[ModelT: BaseModel](path: Path, model_type: type[ModelT]) -> ModelT:
    """Read and strictly validate one Pydantic JSON artifact."""

    try:
        content = path.read_bytes()
        return model_type.model_validate_json(content)
    except (OSError, ValidationError, ValueError) as exc:
        raise StorageError(f"invalid artifact {path}") from exc


def save_parsed_article(article: ParsedArticle, path: Path) -> None:
    atomic_write_bytes(path, serialize_model(article))


def load_parsed_article(path: Path) -> ParsedArticle:
    return load_model(path, ParsedArticle)


def serialize_chunks(chunks: tuple[Chunk, ...]) -> bytes:
    if not chunks:
        raise StorageError("cannot serialize an empty chunk corpus")
    corpus = ChunkCorpus(revision_id=chunks[0].revision_id, chunks=chunks)
    return serialize_model(corpus)


def save_chunks(chunks: tuple[Chunk, ...], path: Path) -> None:
    atomic_write_bytes(path, serialize_chunks(chunks))


def load_chunks(path: Path) -> tuple[Chunk, ...]:
    return load_model(path, ChunkCorpus).chunks
