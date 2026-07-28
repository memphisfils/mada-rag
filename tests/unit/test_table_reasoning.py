"""End-to-end local acceptance tests over the committed snapshot chunks."""

from dataclasses import dataclass
from pathlib import Path

from mada_rag.generation import ExtractiveGenerator, SufficiencyPolicy
from mada_rag.models import AnswerStatus, Chunk, Language, RetrievalMethod, RetrievedChunk
from mada_rag.retrieval import ContextExpander
from mada_rag.service import RagService
from mada_rag.storage import load_chunks

ADMIN_DENSITY_ROW_ID = "chunk-58597e3558fa501ad857dd1e0bafd663"
CURRENT_PRESIDENT_ROW_ID = "chunk-38a78b28a295ec0a7e9e667c5b3f43f1"
LIFE_EXPECTANCY_ID = "chunk-0e65f76953781b67a8443ddd710e1b79"
POWER_CHANGE_IDS = (
    "chunk-6c48d6ccc375bc5d68281fbe1bf05589",
    "chunk-a58c45b78b4d1eda76b143c0dbb0e545",
)
CUISINE_ID = "chunk-99242aac25101075735f086ff30995a1"
G3_CHUNKS_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "g3_chunks.json"


@dataclass(frozen=True)
class FakeRetriever:
    chunks: tuple[Chunk, ...]
    selected_ids: tuple[str, ...]

    @property
    def revision_id(self) -> int:
        return self.chunks[0].revision_id

    def retrieve(
        self,
        _query: str,
        *,
        top_k: int | None = None,
    ) -> tuple[RetrievedChunk, ...]:
        by_id = {chunk.chunk_id: chunk for chunk in self.chunks}
        candidates = tuple(
            RetrievedChunk(
                chunk=by_id[chunk_id],
                method=RetrievalMethod.DENSE,
                rank=rank,
                score=0.9 - rank * 0.01,
                dense_rank=rank,
                dense_score=0.9 - rank * 0.01,
            )
            for rank, chunk_id in enumerate(self.selected_ids, start=1)
        )
        return candidates[:top_k]


def corpus() -> tuple[Chunk, ...]:
    return load_chunks(G3_CHUNKS_FIXTURE)


def service_for(
    *selected_ids: str,
    expand_tables: bool = True,
    max_claims: int = 3,
) -> RagService:
    chunks = corpus()
    retriever = FakeRetriever(chunks, selected_ids)
    return RagService(
        retriever=retriever,
        generator=ExtractiveGenerator(max_claims=max_claims),
        sufficiency_policy=SufficiencyPolicy(
            minimum_score=0.45,
            minimum_concept_coverage=0.8,
        ),
        context_expander=ContextExpander(chunks) if expand_tables else None,
        context_top_k=5,
    )


def test_highest_2018_density_answers_from_analamanga_row() -> None:
    question = (
        "Quelle région a la plus forte densité de population "
        "dans le tableau du recensement de 2018 ?"
    )

    answer = service_for(ADMIN_DENSITY_ROW_ID).ask(question, language=Language.FR)

    assert answer.status is AnswerStatus.ANSWERED
    assert "Analamanga" in answer.text
    assert "198.0" in answer.text
    assert any(
        citation.chunk_id == ADMIN_DENSITY_ROW_ID and citation.row_index == 3
        for citation in answer.citations
    )


def test_current_president_selects_michael_row_not_an_unrelated_table_start() -> None:
    question = "Who is the current president according to the snapshot?"

    answer = service_for(CURRENT_PRESIDENT_ROW_ID, max_claims=1).ask(question)

    assert answer.status is AnswerStatus.ANSWERED
    assert "Michael Randrianirina" in answer.text
    assert answer.citations[0].chunk_id == CURRENT_PRESIDENT_ROW_ID
    assert answer.citations[0].row_index == 9


def test_official_national_dish_trap_abstains_despite_retrieved_cuisine_text() -> None:
    question = "Quel est le plat national officiel de Madagascar ?"

    answer = service_for(CUISINE_ID, expand_tables=False).ask(
        question,
        language=Language.FR,
    )

    assert answer.status is AnswerStatus.ABSTAINED
    assert not answer.claims
    assert not answer.citations


def test_french_life_expectancy_is_not_artificially_refused_by_coverage() -> None:
    question = "Quelle était l'espérance de vie adulte des hommes et des femmes en 2009 ?"

    answer = service_for(LIFE_EXPECTANCY_ID, expand_tables=False).ask(
        question,
        language=Language.FR,
    )

    assert answer.status is AnswerStatus.ANSWERED
    assert "63 years for men and 67 years for women" in answer.text
    assert answer.citations[0].chunk_id == LIFE_EXPECTANCY_ID


def test_french_multi_passage_power_change_is_not_artificially_refused() -> None:
    question = "Comment le pouvoir a-t-il changé de mains entre 2023 et 2025 ?"

    answer = service_for(*POWER_CHANGE_IDS, expand_tables=False).ask(
        question,
        language=Language.FR,
    )

    assert answer.status is AnswerStatus.ANSWERED
    assert "2023" in answer.text
    assert "2025" in answer.text
    assert "Michael Randrianirina" in answer.text
    assert set(POWER_CHANGE_IDS) <= set(answer.retrieved_chunk_ids)


def test_context_expander_adds_global_and_row_chunks_with_source_provenance() -> None:
    chunks = corpus()
    source_chunk = next(chunk for chunk in chunks if chunk.chunk_id == ADMIN_DENSITY_ROW_ID)
    source = RetrievedChunk(
        chunk=source_chunk,
        method=RetrievalMethod.DENSE,
        rank=1,
        score=0.88,
        dense_rank=1,
        dense_score=0.88,
    )

    expanded = ContextExpander(chunks, max_expanded_chunks=100).expand((source,))

    assert expanded[0] == source
    assert len(expanded) > 2
    assert len({item.chunk.chunk_id for item in expanded}) == len(expanded)
    assert {item.chunk.chunk_type.value for item in expanded} >= {"table-part", "table-row"}
    assert all(item.chunk.table_id == source_chunk.table_id for item in expanded)
    assert all(item.expanded_from_chunk_id == ADMIN_DENSITY_ROW_ID for item in expanded[1:])
    assert all(item.dense_rank == 1 and item.dense_score == 0.88 for item in expanded[1:])
    assert [item.chunk.ordinal for item in expanded[1:]] == sorted(
        item.chunk.ordinal for item in expanded[1:]
    )


def test_context_expander_enforces_cap_and_does_not_expand_plain_text() -> None:
    chunks = corpus()
    table_chunk = next(chunk for chunk in chunks if chunk.chunk_id == ADMIN_DENSITY_ROW_ID)
    text_chunk = next(chunk for chunk in chunks if chunk.chunk_id == LIFE_EXPECTANCY_ID)

    def retrieved(chunk: Chunk) -> RetrievedChunk:
        return RetrievedChunk(
            chunk=chunk,
            method=RetrievalMethod.DENSE,
            rank=1,
            score=0.9,
            dense_rank=1,
            dense_score=0.9,
        )

    expander = ContextExpander(chunks, max_expanded_chunks=2)

    assert len(expander.expand((retrieved(table_chunk),))) == 2
    assert expander.expand((retrieved(text_chunk),)) == (retrieved(text_chunk),)
