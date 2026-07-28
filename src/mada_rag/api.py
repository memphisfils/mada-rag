"""Lazy injectable FastAPI surface over the shared RagService."""

from __future__ import annotations

from collections.abc import Callable
from threading import Lock
from typing import Annotated

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from mada_rag.config import RetrievalMode, Settings
from mada_rag.models import Answer, Language, RetrievedChunk
from mada_rag.retrieval import RerankerUnavailableError
from mada_rag.service import RagService

ServiceFactory = Callable[[RetrievalMode], RagService]
Question = Annotated[str, Field(min_length=1, max_length=10_000)]


class ApiRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RetrieveRequest(ApiRequest):
    question: Question
    top_k: Annotated[int, Field(ge=1, le=100)] = 10
    mode: RetrievalMode | None = None

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("question cannot be blank")
        return stripped


class AskRequest(ApiRequest):
    question: Question
    language: Language = Language.EN
    mode: RetrievalMode | None = None

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("question cannot be blank")
        return stripped


def create_app(
    *,
    settings: Settings | None = None,
    service_factory: ServiceFactory | None = None,
) -> FastAPI:
    """Create an app that does not load embedding or reranking models at startup."""

    runtime_settings = settings or Settings()

    def default_factory(mode: RetrievalMode) -> RagService:
        from mada_rag.cli import _build_service

        return _build_service(runtime_settings, mode)

    factory = service_factory or default_factory
    services: dict[RetrievalMode, RagService] = {}
    service_lock = Lock()

    def service_for(mode: RetrievalMode | None) -> RagService:
        selected = mode or runtime_settings.retrieval_mode
        with service_lock:
            service = services.get(selected)
            if service is None:
                service = factory(selected)
                services[selected] = service
            return service

    application = FastAPI(
        title="Mada RAG",
        version="0.1.0",
        description="Source-bounded answers from one revision of Wikipedia's Madagascar page.",
    )

    @application.get("/healthz")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "loaded_modes": sorted(mode.value for mode in services),
        }

    @application.post("/v1/retrieve", response_model=list[RetrievedChunk])
    def retrieve(request: RetrieveRequest) -> list[RetrievedChunk]:
        if len(request.question) > runtime_settings.max_query_chars:
            raise HTTPException(status_code=413, detail="question exceeds configured limit")
        try:
            return list(
                service_for(request.mode).retrieve(
                    request.question,
                    top_k=request.top_k,
                )
            )
        except RerankerUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail="local RAG artifacts unavailable") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @application.post("/v1/ask", response_model=Answer)
    def ask(request: AskRequest) -> Answer:
        if len(request.question) > runtime_settings.max_query_chars:
            raise HTTPException(status_code=413, detail="question exceeds configured limit")
        try:
            return service_for(request.mode).ask(
                request.question,
                language=request.language,
            )
        except RerankerUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail="local RAG artifacts unavailable") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return application


app = create_app()
