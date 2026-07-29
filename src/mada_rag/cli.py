"""Typer CLI for the revision-pinned dense vertical slice."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from mada_rag.chunking import ArticleChunker
from mada_rag.config import GenerationProvider, RetrievalMode, Settings
from mada_rag.evaluation import evaluate as run_evaluation
from mada_rag.evaluation import load_evaluation_cases, write_evaluation_report
from mada_rag.generation import (
    AnswerGenerator,
    CitationValidator,
    ExtractiveGenerator,
    StructuredLLMGenerator,
    SufficiencyPolicy,
)
from mada_rag.indexing import BM25Index, DenseIndex, E5Embedder
from mada_rag.ingestion import MediaWikiClient, SnapshotStore, ingest_snapshot
from mada_rag.models import Language, RetrievedChunk
from mada_rag.parsing import parse_article
from mada_rag.retrieval import (
    BM25Retriever,
    ContextExpander,
    CrossEncoderReranker,
    DenseRetriever,
    HybridRetriever,
    RerankedRetriever,
    Retriever,
)
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


@contextmanager
def _offline_model_loading() -> Iterator[None]:
    """Prevent evaluation from downloading an embedding or reranker model."""

    keys = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    previous = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ[key] = "1"
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _load_dense(settings: Settings) -> tuple[DenseIndex, E5Embedder]:
    manifest, _html = SnapshotStore(settings.snapshot_dir).load_verified()
    embedder = E5Embedder(settings.embedding_model)
    index = DenseIndex.load(
        settings.dense_index_dir,
        expected_revision_id=manifest.revision_id,
        expected_embedding_model=embedder.model_name,
    )
    return index, embedder


def _build_retriever(
    settings: Settings,
    mode: RetrievalMode | None = None,
) -> Retriever:
    selected_mode = mode or settings.retrieval_mode
    index, embedder = _load_dense(settings)
    dense = DenseRetriever(
        index,
        embedder,
        default_top_k=settings.dense_top_k,
    )
    if selected_mode is RetrievalMode.DENSE:
        return dense

    lexical = BM25Retriever(
        BM25Index(index.chunks),
        default_top_k=settings.lexical_top_k,
    )
    hybrid = HybridRetriever(
        dense,
        lexical,
        rrf_k=settings.rrf_k,
        dense_candidates=settings.dense_top_k,
        lexical_candidates=settings.lexical_top_k,
        default_top_k=settings.fused_top_k,
    )
    if selected_mode is RetrievalMode.HYBRID:
        return hybrid
    return RerankedRetriever(
        hybrid,
        CrossEncoderReranker(settings.reranker_model),
        candidate_top_k=settings.fused_top_k,
        default_top_k=settings.context_top_k,
    )


def _build_service(
    settings: Settings,
    mode: RetrievalMode | None = None,
) -> RagService:
    retriever = _build_retriever(settings, mode)
    generator: AnswerGenerator
    if settings.generation_provider is GenerationProvider.EXTRACTIVE:
        generator = ExtractiveGenerator(
            max_claims=settings.extractive_max_claims,
            max_excerpt_chars=settings.extractive_max_excerpt_chars,
        )
    elif settings.generation_provider is GenerationProvider.OPENAI:
        if settings.llm_api_key is None or settings.generation_model is None:
            raise ValueError("OpenAI structured generation is missing required configuration")
        generator = StructuredLLMGenerator(
            provider="openai",
            api_key=settings.llm_api_key.get_secret_value(),
            model_name=settings.generation_model,
            timeout_seconds=settings.llm_timeout_seconds,
        )
    elif settings.generation_provider is GenerationProvider.OPENAI_COMPATIBLE:
        if (
            settings.llm_api_key is None
            or settings.generation_model is None
            or settings.llm_base_url is None
        ):
            raise ValueError("OpenAI-compatible structured generation is missing configuration")
        generator = StructuredLLMGenerator(
            provider="openai-compatible",
            api_key=settings.llm_api_key.get_secret_value(),
            model_name=settings.generation_model,
            base_url=str(settings.llm_base_url),
            timeout_seconds=settings.llm_timeout_seconds,
        )
    else:
        raise ValueError("generation_provider=disabled cannot answer questions")
    return RagService(
        retriever=retriever,
        generator=generator,
        sufficiency_policy=SufficiencyPolicy(
            minimum_score=settings.dense_score_threshold,
            minimum_candidates=settings.minimum_retrieved_chunks,
            minimum_concept_coverage=settings.minimum_concept_coverage,
        ),
        citation_validator=CitationValidator(),
        context_expander=ContextExpander(
            retriever.chunks,
            max_expanded_chunks=settings.max_expanded_table_chunks,
        ),
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
    mode: RetrievalMode | None = None,
) -> None:
    """Retrieve evidence from the validated local dense index."""

    settings = Settings()
    try:
        candidates = _build_service(settings, mode).retrieve(question, top_k=top_k)
        typer.echo(_retrieved_json(candidates))
    except (RuntimeError, ValueError) as exc:
        _fail(exc)


@app.command()
def ask(
    question: str = typer.Argument(..., help="French or English natural-language query."),
    language: Language = Language.EN,
    mode: RetrievalMode | None = None,
) -> None:
    """Return exact cited spans or abstain when retrieval is insufficient."""

    settings = Settings()
    try:
        service = _build_service(settings) if mode is None else _build_service(settings, mode)
        answer = service.ask(question, language=language)
        typer.echo(answer.model_dump_json(indent=2))
    except (RuntimeError, ValueError) as exc:
        _fail(exc)


@app.command()
def evaluate(
    questions_file: Annotated[
        Path | None,
        typer.Option(
            "--file",
            "-f",
            help="Local JSONL EvalCase file; defaults to data/eval/questions.jsonl.",
        ),
    ] = None,
    case_ids: Annotated[
        list[str] | None,
        typer.Option(
            "--case",
            help="Evaluate only one case ID; may be specified multiple times.",
        ),
    ] = None,
    top_k: Annotated[
        int,
        typer.Option(min=1, max=100, help="Retrieval cutoff for ranking metrics."),
    ] = 5,
    modes: Annotated[
        list[RetrievalMode] | None,
        typer.Option(
            "--mode",
            help="Pipeline to evaluate; repeat for multiple modes.",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Optional local path for the JSON report.",
        ),
    ] = None,
) -> None:
    """Evaluate local dense/hybrid pipelines without source-network access."""

    settings = Settings()
    try:
        with _offline_model_loading():
            selected_modes = tuple(
                dict.fromkeys(modes or [RetrievalMode.DENSE, RetrievalMode.HYBRID])
            )
            if RetrievalMode.HYBRID_RERANK in selected_modes and not settings.reranker_enabled:
                raise ValueError("hybrid-rerank evaluation requires reranker_enabled=true")
            cases = load_evaluation_cases(
                questions_file or settings.evaluation_dir / "questions.jsonl",
                case_ids=case_ids,
            )
            snapshot, _html = SnapshotStore(settings.snapshot_dir).load_verified()
            if any(case.revision_id != snapshot.revision_id for case in cases):
                raise ValueError("evaluation cases differ from the verified snapshot revision")
            dense_index, _embedder = _load_dense(settings)
            if dense_index.manifest is None:
                raise RuntimeError("loaded dense index is missing its manifest")
            services = {mode.value: _build_service(settings, mode) for mode in selected_modes}
            index_hashes = {
                mode.value: {
                    "index_sha256": dense_index.manifest.index_sha256,
                    "chunks_sha256": dense_index.manifest.chunks_sha256,
                }
                for mode in selected_modes
            }
            report = run_evaluation(
                cases,
                retrievers={mode: service.retriever for mode, service in services.items()},
                services=services,
                top_k=top_k,
                snapshot_sha256=snapshot.html_sha256,
                index_hashes=index_hashes,
                parameters={
                    "modes": [mode.value for mode in selected_modes],
                    "dense_top_k": settings.dense_top_k,
                    "lexical_top_k": settings.lexical_top_k,
                    "fused_top_k": settings.fused_top_k,
                    "rrf_k": settings.rrf_k,
                    "context_top_k": settings.context_top_k,
                },
            )
            if output is not None:
                write_evaluation_report(report, output)
            typer.echo(report.to_json())
    except (RuntimeError, ValueError) as exc:
        _fail(exc)


@app.command()
def serve(
    host: str | None = typer.Option(None, help="Bind host; defaults to Settings."),
    port: int | None = typer.Option(None, min=1, max=65_535, help="Bind port."),
) -> None:
    """Serve the lazy FastAPI application."""

    import uvicorn

    from mada_rag.api import create_app

    settings = Settings()
    uvicorn.run(
        create_app(settings=settings),
        host=host or settings.api_host,
        port=port or settings.api_port,
    )


if __name__ == "__main__":
    app()
