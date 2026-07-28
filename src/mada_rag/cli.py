"""Typer CLI for the revision-pinned dense vertical slice."""

from __future__ import annotations

import json
from typing import NoReturn

import typer

from mada_rag.chunking import ArticleChunker
from mada_rag.config import GenerationProvider, Settings
from mada_rag.generation import CitationValidator, ExtractiveGenerator, SufficiencyPolicy
from mada_rag.indexing import DenseIndex, E5Embedder
from mada_rag.ingestion import MediaWikiClient, SnapshotStore, ingest_snapshot
from mada_rag.models import Language, RetrievedChunk
from mada_rag.parsing import parse_article
from mada_rag.retrieval import DenseRetriever
from mada_rag.service import RagService
from mada_rag.storage import save_chunks, save_parsed_article

app = typer.Typer(
    name="mada-rag",
    help="Source-bounded RAG over one immutable Madagascar revision.",
    no_args_is_help=True,
)


def _fail(exc: Exception) -> NoReturn:
    typer.echo(json.dumps({"error": str(exc)}, ensure_ascii=False), err=True)
    raise typer.Exit(code=1)


def _load_dense(settings: Settings) -> tuple[DenseIndex, E5Embedder]:
    manifest, _html = SnapshotStore(settings.snapshot_dir).load_verified()
    embedder = E5Embedder(settings.embedding_model)
    index = DenseIndex.load(
        settings.dense_index_dir,
        expected_revision_id=manifest.revision_id,
        expected_embedding_model=embedder.model_name,
    )
    return index, embedder


def _build_service(settings: Settings) -> RagService:
    if settings.generation_provider is not GenerationProvider.EXTRACTIVE:
        raise ValueError("the G2 CLI supports the secret-free extractive provider only")
    index, embedder = _load_dense(settings)
    retriever = DenseRetriever(
        index,
        embedder,
        default_top_k=settings.dense_top_k,
    )
    return RagService(
        retriever=retriever,
        generator=ExtractiveGenerator(
            max_claims=settings.extractive_max_claims,
            max_excerpt_chars=settings.extractive_max_excerpt_chars,
        ),
        sufficiency_policy=SufficiencyPolicy(
            minimum_score=settings.dense_score_threshold,
            minimum_candidates=settings.minimum_retrieved_chunks,
        ),
        citation_validator=CitationValidator(),
        context_top_k=settings.context_top_k,
    )


def _retrieved_json(candidates: tuple[RetrievedChunk, ...]) -> str:
    payload = [candidate.model_dump(mode="json") for candidate in candidates]
    return json.dumps(payload, ensure_ascii=False, indent=2)


@app.command()
def ingest() -> None:
    """Network command: pin, save, parse, and chunk the Madagascar page."""

    settings = Settings()
    try:
        store = SnapshotStore(settings.snapshot_dir)
        with MediaWikiClient(
            api_url=str(settings.mediawiki_api_url),
            page_title=settings.source_page_title,
            user_agent=settings.mediawiki_user_agent,
            timeout_seconds=settings.mediawiki_timeout_seconds,
        ) as client:
            manifest = ingest_snapshot(client, store)
        _verified_manifest, html = store.load_verified()
        article = parse_article(html, manifest)
        embedder = E5Embedder(settings.embedding_model)
        chunks = ArticleChunker.from_settings(settings, tokenizer=embedder).chunk(article)
        save_parsed_article(article, settings.parsed_article_path)
        save_chunks(chunks, settings.chunks_path)
        typer.echo(manifest.model_dump_json(indent=2))
    except (RuntimeError, ValueError) as exc:
        _fail(exc)


@app.command("index")
def build_index(
    overwrite: bool = typer.Option(False, help="Replace rebuildable dense artifacts."),
) -> None:
    """Build a local E5/FAISS index from previously serialized chunks."""

    settings = Settings()
    try:
        snapshot, html = SnapshotStore(settings.snapshot_dir).load_verified()
        embedder = E5Embedder(settings.embedding_model)
        article = parse_article(html, snapshot)
        chunks = ArticleChunker.from_settings(settings, tokenizer=embedder).chunk(article)
        save_parsed_article(article, settings.parsed_article_path)
        save_chunks(chunks, settings.chunks_path)
        manifest = DenseIndex.build(chunks, embedder).save(
            settings.dense_index_dir,
            overwrite=overwrite,
        )
        typer.echo(manifest.model_dump_json(indent=2))
    except (RuntimeError, ValueError) as exc:
        _fail(exc)


@app.command()
def retrieve(
    question: str = typer.Argument(..., help="French or English natural-language query."),
    top_k: int | None = typer.Option(None, min=1, max=100),
) -> None:
    """Retrieve evidence from the validated local dense index."""

    settings = Settings()
    try:
        index, embedder = _load_dense(settings)
        candidates = DenseRetriever(
            index,
            embedder,
            default_top_k=settings.dense_top_k,
        ).retrieve(question, top_k=top_k)
        typer.echo(_retrieved_json(candidates))
    except (RuntimeError, ValueError) as exc:
        _fail(exc)


@app.command()
def ask(
    question: str = typer.Argument(..., help="French or English natural-language query."),
    language: Language = Language.EN,
) -> None:
    """Return exact cited spans or abstain when retrieval is insufficient."""

    settings = Settings()
    try:
        answer = _build_service(settings).ask(question, language=language)
        typer.echo(answer.model_dump_json(indent=2))
    except (RuntimeError, ValueError) as exc:
        _fail(exc)


if __name__ == "__main__":
    app()
