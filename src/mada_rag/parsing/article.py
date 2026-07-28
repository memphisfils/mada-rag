"""BeautifulSoup parser that preserves section and table provenance."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Final

from bs4 import BeautifulSoup, Tag

from mada_rag.models import ParsedArticle, SectionRecord, SnapshotManifest, TableRecord

_BLOCK_TAGS: Final[tuple[str, ...]] = ("h2", "h3", "h4", "h5", "h6", "p", "ul", "ol", "table")
_REMOVED_SELECTORS: Final[tuple[str, ...]] = (
    "script",
    "style",
    "noscript",
    "template",
    "sup.reference",
    ".mw-editsection",
    ".reflist",
    "ol.references",
    ".navbox",
    ".vertical-navbox",
    ".metadata",
    ".ambox",
    ".toc",
    "[role='navigation']",
)
_EXCLUDED_SECTIONS: Final[frozenset[str]] = frozenset(
    {
        "bibliography",
        "citations",
        "external links",
        "further reading",
        "notes",
        "references",
        "see also",
        "sources",
    }
)
_SPACE_RE: Final[re.Pattern[str]] = re.compile(r"\s+")
_SPACE_BEFORE_PUNCTUATION_RE: Final[re.Pattern[str]] = re.compile(r"\s+([,.;:!?%)\]])")


class ArticleParsingError(RuntimeError):
    """Raised when the source HTML cannot produce a valid article."""


@dataclass
class _SectionBuilder:
    section_id: str
    title: str
    level: int
    path: tuple[str, ...]
    ordinal: int
    source_anchor: str | None
    parent_section_id: str | None
    paragraphs: list[str] = field(default_factory=list)

    def build(self, revision_id: int) -> SectionRecord:
        return SectionRecord(
            section_id=self.section_id,
            revision_id=revision_id,
            title=self.title,
            level=self.level,
            path=self.path,
            ordinal=self.ordinal,
            text="\n\n".join(self.paragraphs),
            paragraphs=tuple(self.paragraphs),
            source_anchor=self.source_anchor,
            parent_section_id=self.parent_section_id,
        )


@dataclass(frozen=True)
class _Span:
    value: str
    remaining_rows: int
    is_header: bool


@dataclass(frozen=True)
class _GridRow:
    values: tuple[str, ...]
    header_flags: tuple[bool, ...]
    in_thead: bool
    all_explicit_headers: bool
    has_row_scope: bool


def normalize_text(text: str) -> str:
    """Normalize visible text without removing numbers, dates, or units."""

    normalized = unicodedata.normalize("NFC", text)
    normalized = _SPACE_RE.sub(" ", normalized).strip()
    return _SPACE_BEFORE_PUNCTUATION_RE.sub(r"\1", normalized)


def element_text(element: Tag) -> str:
    """Return normalized visible text from an already-sanitized element."""

    return normalize_text(element.get_text(" ", strip=True))


def _attribute(tag: Tag, name: str) -> str | None:
    value = tag.get(name)
    return value.strip() if isinstance(value, str) and value.strip() else None


def _stable_id(prefix: str, *parts: object) -> str:
    material = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(material).hexdigest()[:24]}"


def _parse_span(value: object) -> int:
    if not isinstance(value, str):
        return 1
    try:
        parsed = int(value)
    except ValueError:
        return 1
    return max(parsed, 1)


def _table_rows(table: Tag) -> list[Tag]:
    return [
        row
        for row in table.find_all("tr")
        if isinstance(row, Tag) and row.find_parent("table") is table
    ]


def _build_grid(table: Tag) -> list[_GridRow]:
    active_spans: dict[int, _Span] = {}
    sparse_rows: list[tuple[dict[int, str], dict[int, bool], bool, bool, bool]] = []
    maximum_width = 0

    for row in _table_rows(table):
        values: dict[int, str] = {}
        header_flags: dict[int, bool] = {}
        for column, span in tuple(active_spans.items()):
            values[column] = span.value
            header_flags[column] = span.is_header
            if span.remaining_rows == 1:
                del active_spans[column]
            else:
                active_spans[column] = _Span(
                    value=span.value,
                    remaining_rows=span.remaining_rows - 1,
                    is_header=span.is_header,
                )

        cells = [
            cell for cell in row.find_all(("th", "td"), recursive=False) if isinstance(cell, Tag)
        ]
        if not cells and not values:
            continue

        column = 0
        for cell in cells:
            while column in values:
                column += 1
            colspan = _parse_span(cell.get("colspan"))
            rowspan = _parse_span(cell.get("rowspan"))
            value = element_text(cell)
            is_header = cell.name == "th"
            placed_columns: list[int] = []
            for _ in range(colspan):
                while column in values:
                    column += 1
                values[column] = value
                header_flags[column] = is_header
                placed_columns.append(column)
                column += 1
            if rowspan > 1:
                for placed_column in placed_columns:
                    active_spans[placed_column] = _Span(value, rowspan - 1, is_header)

        maximum_width = max(maximum_width, max(values, default=-1) + 1)
        scopes = {_attribute(cell, "scope") for cell in cells}
        sparse_rows.append(
            (
                values,
                header_flags,
                row.find_parent("thead") is not None,
                bool(cells) and all(cell.name == "th" for cell in cells),
                bool({"row", "rowgroup"} & scopes),
            )
        )

    rows: list[_GridRow] = []
    for values, header_flags, in_thead, all_headers, has_row_scope in sparse_rows:
        rows.append(
            _GridRow(
                values=tuple(values.get(column, "") for column in range(maximum_width)),
                header_flags=tuple(
                    header_flags.get(column, False) for column in range(maximum_width)
                ),
                in_thead=in_thead,
                all_explicit_headers=all_headers,
                has_row_scope=has_row_scope,
            )
        )
    return rows


def _header_row_indexes(grid: list[_GridRow]) -> tuple[int, ...]:
    explicit = tuple(index for index, row in enumerate(grid) if row.in_thead)
    if explicit:
        return explicit

    inferred: list[int] = []
    for index, row in enumerate(grid):
        if not row.all_explicit_headers or row.has_row_scope:
            break
        inferred.append(index)
    return tuple(inferred)


def _group_headers(grid: list[_GridRow], indexes: tuple[int, ...], width: int) -> tuple[str, ...]:
    if not indexes:
        return tuple(f"Column {column + 1}" for column in range(width))

    headers: list[str] = []
    for column in range(width):
        levels: list[str] = []
        for row_index in indexes:
            value = grid[row_index].values[column]
            if value and (not levels or levels[-1] != value):
                levels.append(value)
        headers.append(" / ".join(levels) or f"Column {column + 1}")
    return tuple(headers)


def _parse_table(
    table: Tag,
    *,
    revision_id: int,
    section: _SectionBuilder,
    ordinal: int,
) -> TableRecord | None:
    grid = _build_grid(table)
    if not grid:
        return None
    width = len(grid[0].values)
    if width == 0:
        return None

    header_indexes = _header_row_indexes(grid)
    header_index_set = set(header_indexes)
    headers = _group_headers(grid, header_indexes, width)
    rows = tuple(row.values for index, row in enumerate(grid) if index not in header_index_set)
    rows = tuple(row for row in rows if any(cell for cell in row))
    if not rows:
        return None

    caption_tag = table.find("caption")
    caption = element_text(caption_tag) if isinstance(caption_tag, Tag) else None
    source_anchor = _attribute(table, "id") or section.source_anchor
    table_id = _stable_id(
        "table",
        revision_id,
        section.section_id,
        ordinal,
        caption or "",
    )
    return TableRecord(
        table_id=table_id,
        revision_id=revision_id,
        section_id=section.section_id,
        section_path=section.path,
        ordinal=ordinal,
        caption=caption,
        headers=headers,
        rows=rows,
        source_anchor=source_anchor,
    )


def _iter_blocks(container: Tag) -> list[Tag]:
    blocks: list[Tag] = []
    for tag in container.find_all(_BLOCK_TAGS):
        if not isinstance(tag, Tag):
            continue
        if tag.name == "table" and tag.find_parent("table") is not None:
            continue
        if tag.name != "table" and tag.find_parent("table") is not None:
            continue
        if tag.name in {"ul", "ol"} and tag.find_parent(("ul", "ol")) is not None:
            continue
        if tag.name == "p" and tag.find_parent(("li", "ul", "ol")) is not None:
            continue
        blocks.append(tag)
    return blocks


def _list_text(list_tag: Tag) -> str:
    items = [
        element_text(item)
        for item in list_tag.find_all("li", recursive=False)
        if isinstance(item, Tag)
    ]
    return "\n".join(f"- {item}" for item in items if item)


class ArticleParser:
    """Parse only the HTML already captured in a verified snapshot."""

    def parse(self, html: str, manifest: SnapshotManifest) -> ParsedArticle:
        if not html.strip():
            raise ArticleParsingError("article HTML cannot be empty")
        soup = BeautifulSoup(html, "html.parser")
        for selector in _REMOVED_SELECTORS:
            for node in soup.select(selector):
                node.decompose()

        container = soup.select_one(".mw-parser-output")
        if not isinstance(container, Tag):
            container = soup.select_one("#mw-content-text")
        if not isinstance(container, Tag):
            raise ArticleParsingError("MediaWiki article content container was not found")

        section_ordinal = 0
        lead = _SectionBuilder(
            section_id="lead",
            title="Lead",
            level=1,
            path=("Lead",),
            ordinal=section_ordinal,
            source_anchor=None,
            parent_section_id=None,
        )
        sections: list[_SectionBuilder] = [lead]
        current: _SectionBuilder | None = lead
        title_stack: dict[int, str] = {}
        id_stack: dict[int, str] = {}
        excluded_level: int | None = None
        tables: list[TableRecord] = []
        table_ordinal = 0

        for block in _iter_blocks(container):
            if block.name in {"h2", "h3", "h4", "h5", "h6"}:
                level = int(block.name[1])
                title = element_text(block)
                if not title:
                    continue
                if excluded_level is not None and level > excluded_level:
                    current = None
                    continue
                excluded_level = None
                for stack_level in tuple(title_stack):
                    if stack_level >= level:
                        del title_stack[stack_level]
                        id_stack.pop(stack_level, None)
                if title.casefold() in _EXCLUDED_SECTIONS:
                    excluded_level = level
                    current = None
                    continue

                title_stack[level] = title
                path = tuple(title_stack[key] for key in sorted(title_stack))
                parent_id = id_stack[max(id_stack)] if id_stack else None
                section_ordinal += 1
                source_anchor = _attribute(block, "id")
                section_id = _stable_id(
                    "section",
                    manifest.revision_id,
                    "/".join(path),
                    section_ordinal,
                )
                current = _SectionBuilder(
                    section_id=section_id,
                    title=title,
                    level=level,
                    path=path,
                    ordinal=section_ordinal,
                    source_anchor=source_anchor,
                    parent_section_id=parent_id,
                )
                sections.append(current)
                id_stack[level] = section_id
                continue

            if current is None:
                continue
            if block.name == "table":
                table = _parse_table(
                    block,
                    revision_id=manifest.revision_id,
                    section=current,
                    ordinal=table_ordinal,
                )
                if table is not None:
                    tables.append(table)
                    table_ordinal += 1
                continue

            text = _list_text(block) if block.name in {"ul", "ol"} else element_text(block)
            if text:
                current.paragraphs.append(text)

        records = tuple(section.build(manifest.revision_id) for section in sections)
        return ParsedArticle(
            revision_id=manifest.revision_id,
            source_url=manifest.canonical_url,
            sections=records,
            tables=tuple(tables),
        )


def parse_article(html: str, manifest: SnapshotManifest) -> ParsedArticle:
    """Convenience wrapper around :class:`ArticleParser`."""

    return ArticleParser().parse(html, manifest)
