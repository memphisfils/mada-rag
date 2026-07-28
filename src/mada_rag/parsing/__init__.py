"""Section and table parsing from the immutable HTML snapshot."""

from mada_rag.parsing.article import (
    ArticleParser,
    ArticleParsingError,
    element_text,
    normalize_text,
    parse_article,
)

__all__ = [
    "ArticleParser",
    "ArticleParsingError",
    "element_text",
    "normalize_text",
    "parse_article",
]
