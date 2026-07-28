"""Parser tests against the explicitly synthetic MediaWiki HTML fixture."""

from datetime import UTC, datetime
from pathlib import Path

from mada_rag.models import SnapshotManifest, TableRecord
from mada_rag.parsing import parse_article

FIXTURE_PATH = Path("tests/fixtures/madagascar_synthetic.html")
REVISION_ID = 123


def make_manifest() -> SnapshotManifest:
    return SnapshotManifest(
        page_id=42,
        revision_id=REVISION_ID,
        parent_revision_id=122,
        revision_timestamp=datetime(2026, 7, 28, 8, 0, tzinfo=UTC),
        fetched_at=datetime(2026, 7, 28, 8, 1, tzinfo=UTC),
        canonical_url="https://en.wikipedia.org/wiki/Madagascar",
        api_url="https://en.wikipedia.org/w/api.php",
        raw_html_path=Path("madagascar.html"),
        html_sha256="a" * 64,
        parser_version="test",
    )


def parse_fixture():
    return parse_article(FIXTURE_PATH.read_text(encoding="utf-8"), make_manifest())


def table_by_caption(caption: str) -> TableRecord:
    article = parse_fixture()
    return next(table for table in article.tables if table.caption == caption)


def test_lead_and_nested_sections_preserve_order_and_paths() -> None:
    article = parse_fixture()
    titles = [section.title for section in article.sections]

    assert titles == [
        "Lead",
        "Fixture history",
        "Nested fixture period",
        "Synthetic regions",
        "Units and missing values",
    ]
    assert [section.ordinal for section in article.sections] == list(range(5))

    history = article.sections[1]
    nested = article.sections[2]
    units = article.sections[4]
    assert nested.path == ("Fixture history", "Nested fixture period")
    assert nested.parent_section_id == history.section_id
    assert units.path == ("Synthetic regions", "Units and missing values")


def test_cleanup_keeps_visible_link_text_and_excludes_reference_material() -> None:
    article = parse_fixture()
    text = "\n".join(section.text for section in article.sections)

    assert "synthetic internal link" in text
    assert "whose visible anchor text should be retained" in text
    assert "/wiki/Internal_fixture_target" not in text
    assert "[1]" not in text
    assert "[2]" not in text
    assert "Synthetic citation text that should be excluded" not in text
    assert "Another fabricated reference to exclude" not in text
    assert "Navigation content that must not become a knowledge chunk" not in text
    assert "window.fixtureScriptMustBeIgnored" not in text
    assert "References" not in [section.title for section in article.sections]


def test_infobox_is_parsed_as_a_rectangular_lead_table() -> None:
    table = table_by_caption("Synthetic Madagascar fixture")

    assert table.section_id == "lead"
    assert table.section_path == ("Lead",)
    assert table.headers == ("Parser-only infobox", "Parser-only infobox")
    assert table.rows == (
        ("Fixture capital", "Example Capital"),
        ("Fixture area", "999 fixture km 2"),
        ("As-of date", "2 January 2099"),
    )
    assert all(len(row) == len(table.headers) for row in table.rows)


def test_table_caption_grouped_headers_colspan_and_rowspan_are_preserved() -> None:
    table = table_by_caption("Fabricated regional observations for parser tests only")

    assert table.section_path == ("Synthetic regions",)
    assert table.headers == (
        "Fixture zone",
        "Synthetic region",
        "Area (fixture km 2)",
        "Fabricated population snapshot / 1 Jan 2098 (people)",
        "Fabricated population snapshot / Density (people/fixture km 2)",
    )
    assert table.rows[0] == ("Example North", "Region Alpha", "101", "1,010", "10.0")
    assert table.rows[1] == ("Example North", "Region Beta", "202", "4,040", "20.0")
    assert table.rows[2] == ("Example South", "Region Gamma", "303", "9,090", "30.0")
    assert table.rows[3] == (
        "Synthetic total",
        "Synthetic total",
        "606",
        "14,140",
        "not applicable",
    )
    assert all(len(row) == len(table.headers) for row in table.rows)


def test_units_dates_and_missing_values_remain_queryable() -> None:
    article = parse_fixture()
    units = next(
        section for section in article.sections if section.title == "Units and missing values"
    )

    assert "- Distance: 12.5 fixture km." in units.text
    assert "- Observation date: 1 January 2098." in units.text
    assert "- Unavailable value: N/A." in units.text
    assert any("fixture km 2" in " ".join(table.headers) for table in article.tables)
