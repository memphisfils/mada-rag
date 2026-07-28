"""Generic bilingual concept normalization used for grounding, not knowledge."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from typing import Literal

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)?", re.IGNORECASE)
_STOPWORDS = frozenset(
    {
        "a",
        "according",
        "an",
        "and",
        "apres",
        "article",
        "au",
        "actuel",
        "actuelle",
        "actuelles",
        "actuels",
        "aux",
        "avec",
        "based",
        "census",
        "ce",
        "ces",
        "comment",
        "current",
        "currently",
        "d",
        "dans",
        "data",
        "de",
        "des",
        "do",
        "does",
        "donnee",
        "donnees",
        "du",
        "en",
        "entre",
        "est",
        "et",
        "etait",
        "for",
        "had",
        "has",
        "have",
        "how",
        "in",
        "indicate",
        "indicated",
        "indique",
        "indiquee",
        "indiquees",
        "is",
        "il",
        "l",
        "la",
        "le",
        "les",
        "mention",
        "mentioned",
        "moins",
        "of",
        "on",
        "ou",
        "page",
        "par",
        "plus",
        "pour",
        "qu",
        "que",
        "quel",
        "quelle",
        "quels",
        "quelles",
        "qui",
        "recensement",
        "selon",
        "show",
        "shown",
        "shows",
        "snapshot",
        "table",
        "tableau",
        "t",
        "the",
        "to",
        "un",
        "une",
        "what",
        "when",
        "which",
        "who",
        "whose",
    }
)
_CONCEPT_ALIASES = {
    "adulte": "adult",
    "capitale": "capital",
    "change": "transition",
    "changed": "transition",
    "changee": "transition",
    "changees": "transition",
    "changer": "transition",
    "changes": "transition",
    "changing": "transition",
    "densite": "density",
    "esperance": "expectancy",
    "faible": "minimum",
    "faibles": "minimum",
    "highest": "maximum",
    "largest": "maximum",
    "least": "minimum",
    "lowest": "minimum",
    "fleur": "flower",
    "fleurs": "flower",
    "femme": "female",
    "femmes": "female",
    "hand": "control",
    "handover": "control",
    "handovers": "control",
    "hands": "control",
    "homme": "male",
    "hommes": "male",
    "main": "control",
    "mains": "control",
    "man": "male",
    "men": "male",
    "nationale": "national",
    "nationales": "national",
    "nationaux": "national",
    "officiel": "official",
    "officielle": "official",
    "officiels": "official",
    "officielles": "official",
    "plat": "dish",
    "plats": "dish",
    "population": "population",
    "pouvoir": "control",
    "power": "control",
    "presidente": "president",
    "region": "region",
    "regions": "region",
    "salaire": "salary",
    "salaires": "salary",
    "smallest": "minimum",
    "superficie": "area",
    "taux": "rate",
    "vie": "life",
    "woman": "female",
    "women": "female",
    "eleve": "maximum",
    "elevee": "maximum",
    "eleves": "maximum",
    "elevees": "maximum",
}
_ORDERING_CONCEPTS = frozenset({"maximum", "minimum"})
_CRITICAL_ATTRIBUTE_CONCEPTS = frozenset(
    {
        "area",
        "capital",
        "density",
        "dish",
        "flower",
        "national",
        "official",
        "population",
        "president",
        "rate",
        "salary",
    }
)


def _ascii_fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(character for character in decomposed if not unicodedata.combining(character))


def _singularize(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return f"{token[:-3]}y"
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def normalized_tokens(text: str, *, drop_stopwords: bool = True) -> tuple[str, ...]:
    """Tokenize names, words, numbers, and units without discarding exact values."""

    folded = _ascii_fold(text)
    tokens: list[str] = []
    for raw_token in _TOKEN_RE.findall(folded):
        token = _CONCEPT_ALIASES.get(raw_token, _singularize(raw_token))
        if drop_stopwords and token in _STOPWORDS:
            continue
        tokens.append(token)
    return tuple(tokens)


def query_concepts(question: str) -> frozenset[str]:
    """Return content concepts; comparison operators are handled separately."""

    return frozenset(normalized_tokens(question)) - _ORDERING_CONCEPTS


def concept_coverage(question: str, evidence: Iterable[str]) -> float:
    concepts = query_concepts(question)
    if not concepts:
        return 1.0
    evidence_concepts: set[str] = set()
    for text in evidence:
        evidence_concepts.update(normalized_tokens(text))
    return len(concepts & evidence_concepts) / len(concepts)


def missing_critical_concepts(question: str, evidence: Iterable[str]) -> frozenset[str]:
    """Return requested factual attributes that are absent from the evidence."""

    requested = query_concepts(question) & _CRITICAL_ATTRIBUTE_CONCEPTS
    if not requested:
        return frozenset()
    evidence_concepts: set[str] = set()
    for text in evidence:
        evidence_concepts.update(normalized_tokens(text))
    return requested - evidence_concepts


def ordering_direction(question: str) -> Literal["maximum", "minimum"] | None:
    folded = _ascii_fold(question)
    maximum_phrases = (
        "highest",
        "largest",
        "most ",
        "plus eleve",
        "plus elevee",
        "plus forte",
        "plus grand",
        "plus grande",
    )
    minimum_phrases = (
        "lowest",
        "smallest",
        "least ",
        "plus bas",
        "plus basse",
        "plus faible",
        "moins eleve",
        "moins elevee",
    )
    if any(phrase in folded for phrase in maximum_phrases):
        return "maximum"
    if any(phrase in folded for phrase in minimum_phrases):
        return "minimum"
    return None
