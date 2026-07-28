"""FastAPI tests with injected local fakes: no model or network access."""

from dataclasses import dataclass, field
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mada_rag.api import create_app
from mada_rag.config import RetrievalMode, Settings
from mada_rag.models import Answer, AnswerStatus, Language, RetrievalMethod, RetrievedChunk
from mada_rag.retrieval import RerankerUnavailableError
from mada_rag.storage import load_chunks

PRESIDENT_CHUNK_ID = "chunk-38a78b28a295ec0a7e9e667c5b3f43f1"
G3_CHUNKS_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "g3_chunks.json"


def retrieved_president() -> RetrievedChunk:
    chunk = next(
        item for item in load_chunks(G3_CHUNKS_FIXTURE) if item.chunk_id == PRESIDENT_CHUNK_ID
    )
    return RetrievedChunk(
        chunk=chunk,
        method=RetrievalMethod.DENSE,
        rank=1,
        score=0.9,
        dense_rank=1,
        dense_score=0.9,
    )


@dataclass
class FakeService:
    candidate: RetrievedChunk
    error: Exception | None = None
    retrieve_calls: list[tuple[str, int | None]] = field(default_factory=list)
    ask_calls: list[tuple[str, Language]] = field(default_factory=list)

    def retrieve(
        self,
        question: str,
        *,
        top_k: int | None = None,
    ) -> tuple[RetrievedChunk, ...]:
        self.retrieve_calls.append((question, top_k))
        if self.error is not None:
            raise self.error
        return (self.candidate,)

    def ask(self, question: str, *, language: Language = Language.EN) -> Answer:
        self.ask_calls.append((question, language))
        if self.error is not None:
            raise self.error
        return Answer(
            question=question,
            language=language,
            status=AnswerStatus.ABSTAINED,
            text="I do not know from the supplied snapshot.",
            revision_id=self.candidate.chunk.revision_id,
            retrieved_chunk_ids=(self.candidate.chunk.chunk_id,),
            refusal_reason="test abstention",
            provider="test",
            model="test",
            latency_ms=0.1,
        )


def make_settings(**updates: object) -> Settings:
    values: dict[str, object] = {
        "_env_file": None,
        "retrieval_mode": RetrievalMode.HYBRID,
        "max_query_chars": 32,
    }
    values.update(updates)
    return Settings(**values)


def test_health_is_lazy_and_services_are_cached_per_mode() -> None:
    candidate = retrieved_president()
    factory_calls: list[RetrievalMode] = []
    services: dict[RetrievalMode, FakeService] = {}

    def factory(mode: RetrievalMode) -> FakeService:
        factory_calls.append(mode)
        return services.setdefault(mode, FakeService(candidate))

    client = TestClient(create_app(settings=make_settings(), service_factory=factory))

    assert client.get("/healthz").json() == {"status": "ok", "loaded_modes": []}
    first = client.post(
        "/v1/retrieve",
        json={"question": "president", "top_k": 1, "mode": "dense"},
    )
    second = client.post(
        "/v1/retrieve",
        json={"question": "president", "top_k": 1, "mode": "dense"},
    )

    assert first.status_code == second.status_code == 200
    assert factory_calls == [RetrievalMode.DENSE]
    assert services[RetrievalMode.DENSE].retrieve_calls == [
        ("president", 1),
        ("president", 1),
    ]
    assert client.get("/healthz").json() == {
        "status": "ok",
        "loaded_modes": ["dense"],
    }


def test_retrieve_and_ask_responses_follow_domain_schemas() -> None:
    candidate = retrieved_president()
    service = FakeService(candidate)
    client = TestClient(
        create_app(
            settings=make_settings(),
            service_factory=lambda _mode: service,
        )
    )

    retrieval = client.post("/v1/retrieve", json={"question": "president", "top_k": 1})
    answer = client.post(
        "/v1/ask",
        json={"question": "president", "language": "fr"},
    )

    assert retrieval.status_code == 200
    payload = retrieval.json()[0]
    assert payload["chunk"]["chunk_id"] == PRESIDENT_CHUNK_ID
    assert payload["method"] == "dense"
    assert payload["dense_rank"] == 1
    assert answer.status_code == 200
    assert answer.json()["status"] == "abstained"
    assert answer.json()["language"] == "fr"
    assert service.ask_calls == [("president", Language.FR)]


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "   "},
        {"question": "president", "unknown": True},
        {"question": "president", "top_k": 0},
        {"question": "president", "top_k": 101},
        {"question": "president", "mode": "unsupported"},
    ],
)
def test_request_schema_rejects_blank_extra_invalid_limit_and_mode(
    payload: dict[str, object],
) -> None:
    client = TestClient(
        create_app(
            settings=make_settings(),
            service_factory=lambda _mode: FakeService(retrieved_president()),
        )
    )

    assert client.post("/v1/retrieve", json=payload).status_code == 422


def test_configured_query_limit_returns_413_without_loading_service() -> None:
    factory_calls: list[RetrievalMode] = []

    def factory(mode: RetrievalMode) -> FakeService:
        factory_calls.append(mode)
        return FakeService(retrieved_president())

    client = TestClient(
        create_app(
            settings=make_settings(max_query_chars=32),
            service_factory=factory,
        )
    )

    response = client.post("/v1/ask", json={"question": "x" * 33})

    assert response.status_code == 413
    assert factory_calls == []


@pytest.mark.parametrize(
    ("error", "expected_status", "expected_detail"),
    [
        (RerankerUnavailableError("reranker offline"), 503, "reranker offline"),
        (RuntimeError("artifact path"), 503, "local RAG artifacts unavailable"),
        (ValueError("bad top-k"), 400, "bad top-k"),
    ],
)
def test_runtime_failures_map_to_safe_http_errors(
    error: Exception,
    expected_status: int,
    expected_detail: str,
) -> None:
    service = FakeService(retrieved_president(), error=error)
    client = TestClient(
        create_app(
            settings=make_settings(),
            service_factory=lambda _mode: service,
        )
    )

    response = client.post(
        "/v1/retrieve",
        json={"question": "president", "top_k": 1},
    )

    assert response.status_code == expected_status
    assert response.json() == {"detail": expected_detail}
