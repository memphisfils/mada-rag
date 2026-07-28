"""Offline Typer CLI tests, including reconstruction from raw snapshot only."""

import shutil
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

import mada_rag.cli as cli
from mada_rag.models import Answer, AnswerStatus, Language

runner = CliRunner()


class OfflineEmbedder:
    model_name = "intfloat/multilingual-e5-base"
    dimension = 3

    def __init__(self, model_name: str) -> None:
        assert model_name == self.model_name

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            [
                (
                    float((len(text) % 7) + 1),
                    float((sum(map(ord, text[:20])) % 11) + 1),
                    1.0,
                )
                for text in texts
            ],
            dtype=np.float32,
        )

    def embed_query(self, text: str) -> np.ndarray:
        assert text
        return np.array([1.0, 1.0, 1.0], dtype=np.float32)


class ForbiddenNetworkClient:
    def __init__(self, **_kwargs: object) -> None:
        raise AssertionError("MediaWiki must not be called by an offline command")


class StubService:
    def ask(self, question: str, *, language: Language) -> Answer:
        return Answer(
            question=question,
            language=language,
            status=AnswerStatus.ABSTAINED,
            text=(
                "Je ne sais pas à partir du snapshot fourni."
                if language is Language.FR
                else "I do not know from the supplied snapshot."
            ),
            revision_id=1365949107,
            refusal_reason="offline test",
            provider="extractive",
            model="exact-span-v1",
        )


def test_ask_command_is_offline_and_emits_structured_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "MediaWikiClient", ForbiddenNetworkClient)
    monkeypatch.setattr(cli, "_build_service", lambda _settings: StubService())

    result = runner.invoke(cli.app, ["ask", "Question absente ?", "--language", "fr"])

    assert result.exit_code == 0
    assert '"status": "abstained"' in result.stdout
    assert "Je ne sais pas" in result.stdout


def test_index_rebuilds_offline_from_only_raw_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("faiss")
    repository_root = Path(__file__).parents[2]
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    shutil.copy2(repository_root / "data" / "raw" / "manifest.json", raw_dir)
    shutil.copy2(repository_root / "data" / "raw" / "madagascar.html", raw_dir)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "MediaWikiClient", ForbiddenNetworkClient)
    monkeypatch.setattr(cli, "E5Embedder", OfflineEmbedder)

    assert not (tmp_path / "data" / "processed").exists()
    assert not (tmp_path / "artifacts").exists()

    result = runner.invoke(cli.app, ["index"])

    assert result.exit_code == 0, result.stderr
    assert '"revision_id": 1365949107' in result.stdout
    assert (tmp_path / "data" / "processed" / "article.json").is_file()
    assert (tmp_path / "data" / "processed" / "chunks.json").is_file()
    assert (tmp_path / "artifacts" / "indexes" / "dense" / "index.faiss").is_file()
    assert (tmp_path / "artifacts" / "indexes" / "dense" / "manifest.json").is_file()
