import json
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

from ingestion.fetch_papers import (
    init_db,
    filter_by_keywords,
    fetch_papers_by_category,
    download_pdf,
    extract_sections,
    store_paper,
    run_ingestion,
)


# ── init_db ──────────────────────────────────────────────────────────────────

def test_init_db_creates_table(tmp_path):
    conn = init_db(str(tmp_path / "test.db"))
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='papers'"
    )
    assert cursor.fetchone() is not None
    conn.close()


def test_init_db_schema_has_correct_columns(tmp_path):
    conn = init_db(str(tmp_path / "test.db"))
    cursor = conn.execute("PRAGMA table_info(papers)")
    columns = {row[1] for row in cursor.fetchall()}
    assert columns == {
        "paper_id", "title", "authors", "pdf_url",
        "categories", "published_date", "sections", "fetched_at",
    }
    conn.close()


def test_init_db_is_idempotent(tmp_path):
    db_path = str(tmp_path / "test.db")
    conn = init_db(db_path)
    conn.close()
    conn2 = init_db(db_path)  # must not raise on second call
    conn2.close()


# ── filter_by_keywords ───────────────────────────────────────────────────────

def _make_paper(title: str, summary: str):
    paper = MagicMock()
    paper.title = title
    paper.summary = summary
    return paper


def test_filter_matches_keyword_in_title():
    papers = [_make_paper("AI Safety in LLMs", "Something else")]
    assert len(filter_by_keywords(papers, ["AI safety"])) == 1


def test_filter_matches_keyword_in_abstract():
    papers = [_make_paper("Language Models", "This paper studies alignment problems")]
    assert len(filter_by_keywords(papers, ["alignment"])) == 1


def test_filter_is_case_insensitive():
    papers = [_make_paper("ALIGNMENT IN TRANSFORMERS", "no keywords here")]
    assert len(filter_by_keywords(papers, ["alignment"])) == 1


def test_filter_excludes_non_matching_papers():
    papers = [_make_paper("Gradient Descent", "Optimization techniques")]
    assert len(filter_by_keywords(papers, ["alignment", "AI safety"])) == 0


def test_filter_partial_match_across_multiple_papers():
    papers = [
        _make_paper("AI Safety", "General topic"),
        _make_paper("Gradient Methods", "Optimization"),
    ]
    assert len(filter_by_keywords(papers, ["AI safety"])) == 1


def test_filter_returns_empty_list_for_empty_input():
    assert filter_by_keywords([], ["alignment"]) == []


# ── fetch_papers_by_category ─────────────────────────────────────────────────

def _make_arxiv_result(days_ago: int):
    result = MagicMock()
    result.published = datetime.now(timezone.utc) - timedelta(days=days_ago)
    result.title = "Test Paper"
    result.summary = "Test abstract"
    return result


@patch("ingestion.fetch_papers.arxiv.Client")
@patch("ingestion.fetch_papers.arxiv.Search")
def test_fetch_returns_recent_papers_and_stops_at_cutoff(mock_search_cls, mock_client_cls):
    recent = _make_arxiv_result(days_ago=30)
    old = _make_arxiv_result(days_ago=400)
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.results.return_value = iter([recent, old])

    results = fetch_papers_by_category("cs.AI", months_back=12)

    assert recent in results
    assert old not in results


@patch("ingestion.fetch_papers.arxiv.Client")
@patch("ingestion.fetch_papers.arxiv.Search")
def test_fetch_passes_correct_category_to_search(mock_search_cls, mock_client_cls):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.results.return_value = iter([])

    fetch_papers_by_category("cs.LG", months_back=12)

    call_kwargs = mock_search_cls.call_args.kwargs
    assert "cs.LG" in call_kwargs["query"]


@patch("ingestion.fetch_papers.arxiv.Client")
@patch("ingestion.fetch_papers.arxiv.Search")
def test_fetch_returns_empty_list_when_no_results(mock_search_cls, mock_client_cls):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.results.return_value = iter([])

    assert fetch_papers_by_category("cs.AI", months_back=12) == []


# ── download_pdf ─────────────────────────────────────────────────────────────

def _make_arxiv_paper(short_id: str = "2301.07041v1"):
    paper = MagicMock()
    paper.get_short_id.return_value = short_id
    return paper


def test_download_pdf_calls_download_and_returns_path(tmp_path):
    paper = _make_arxiv_paper("2301.07041v1")

    def fake_download(dirpath, filename):
        (Path(dirpath) / filename).touch()

    paper.download_pdf.side_effect = fake_download

    result = download_pdf(paper, tmp_path)

    assert result == tmp_path / "2301.07041v1.pdf"
    assert result.exists()


def test_download_pdf_skips_if_file_already_exists(tmp_path):
    paper = _make_arxiv_paper("2301.07041v1")
    existing = tmp_path / "2301.07041v1.pdf"
    existing.touch()

    result = download_pdf(paper, tmp_path)

    paper.download_pdf.assert_not_called()
    assert result == existing


# ── extract_sections ─────────────────────────────────────────────────────────

_SAMPLE_TEI_XML = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <abstract>
        <p>This paper studies AI safety techniques for large language models.</p>
      </abstract>
    </fileDesc>
  </teiHeader>
  <text>
    <body>
      <div>
        <head>1 Introduction</head>
        <p>Safety is a critical concern for modern AI systems.</p>
      </div>
      <div>
        <head>3 Results</head>
        <p>Our experiments demonstrate improved robustness on standard benchmarks.</p>
      </div>
      <div>
        <head>4 Conclusion</head>
        <p>We conclude that safety techniques significantly reduce harmful outputs.</p>
      </div>
    </body>
  </text>
</TEI>"""

_MINIMAL_TEI_XML = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader><fileDesc></fileDesc></teiHeader>
  <text><body></body></text>
</TEI>"""


def _mock_grobid_response(tei_xml: str):
    mock_response = MagicMock()
    mock_response.text = tei_xml
    mock_response.raise_for_status.return_value = None
    return mock_response


@patch("ingestion.fetch_papers.requests.post")
def test_extract_sections_returns_all_requested_keys(mock_post, tmp_path):
    mock_post.return_value = _mock_grobid_response(_SAMPLE_TEI_XML)
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.touch()

    sections = extract_sections(pdf_path, ["abstract", "introduction", "results", "conclusion"])

    assert set(sections.keys()) == {"abstract", "introduction", "results", "conclusion"}


@patch("ingestion.fetch_papers.requests.post")
def test_extract_abstract_from_tei_header(mock_post, tmp_path):
    mock_post.return_value = _mock_grobid_response(_SAMPLE_TEI_XML)
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.touch()

    sections = extract_sections(pdf_path, ["abstract"])

    assert "AI safety" in sections["abstract"]


@patch("ingestion.fetch_papers.requests.post")
def test_extract_body_section_by_heading_substring_match(mock_post, tmp_path):
    mock_post.return_value = _mock_grobid_response(_SAMPLE_TEI_XML)
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.touch()

    sections = extract_sections(pdf_path, ["introduction"])

    assert "critical" in sections["introduction"]


@patch("ingestion.fetch_papers.requests.post")
def test_extract_missing_section_returns_empty_string(mock_post, tmp_path):
    mock_post.return_value = _mock_grobid_response(_MINIMAL_TEI_XML)
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.touch()

    sections = extract_sections(pdf_path, ["abstract"])

    assert sections["abstract"] == ""


@patch("ingestion.fetch_papers.requests.post")
def test_extract_sections_posts_to_correct_grobid_endpoint(mock_post, tmp_path):
    mock_post.return_value = _mock_grobid_response(_SAMPLE_TEI_XML)
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.touch()

    extract_sections(pdf_path, ["abstract"])

    call_url = mock_post.call_args.args[0]
    assert "/api/processFulltextDocument" in call_url


# ── store_paper ──────────────────────────────────────────────────────────────

def _make_full_paper(short_id: str = "2301.07041v1"):
    paper = MagicMock()
    paper.get_short_id.return_value = short_id
    paper.title = "Test Paper on AI Safety"
    author_a, author_b = MagicMock(), MagicMock()
    author_a.name = "Alice Smith"
    author_b.name = "Bob Jones"
    paper.authors = [author_a, author_b]
    paper.pdf_url = "https://arxiv.org/pdf/2301.07041"
    paper.categories = ["cs.AI", "cs.LG"]
    paper.published = datetime(2024, 1, 15, tzinfo=timezone.utc)
    return paper


_SECTIONS = {
    "abstract": "We study safety",
    "introduction": "Intro text",
    "results": "Results here",
    "conclusion": "Conclusions here",
}


def test_store_paper_inserts_new_paper(tmp_path):
    conn = init_db(str(tmp_path / "test.db"))
    paper = _make_full_paper()

    inserted = store_paper(conn, paper, _SECTIONS)

    assert inserted is True
    row = conn.execute(
        "SELECT paper_id FROM papers WHERE paper_id = '2301.07041v1'"
    ).fetchone()
    assert row is not None


def test_store_paper_sections_stored_as_valid_json(tmp_path):
    conn = init_db(str(tmp_path / "test.db"))
    paper = _make_full_paper()

    store_paper(conn, paper, _SECTIONS)

    row = conn.execute(
        "SELECT sections FROM papers WHERE paper_id = '2301.07041v1'"
    ).fetchone()
    parsed = json.loads(row[0])
    assert parsed["abstract"] == "We study safety"
    assert parsed["introduction"] == "Intro text"


def test_store_paper_skips_duplicate_and_returns_false(tmp_path):
    conn = init_db(str(tmp_path / "test.db"))
    paper = _make_full_paper()

    store_paper(conn, paper, _SECTIONS)
    second = store_paper(conn, paper, _SECTIONS)

    assert second is False
    count = conn.execute(
        "SELECT COUNT(*) FROM papers WHERE paper_id = '2301.07041v1'"
    ).fetchone()[0]
    assert count == 1


def test_store_paper_authors_stored_as_comma_separated(tmp_path):
    conn = init_db(str(tmp_path / "test.db"))
    paper = _make_full_paper()

    store_paper(conn, paper, _SECTIONS)

    row = conn.execute(
        "SELECT authors FROM papers WHERE paper_id = '2301.07041v1'"
    ).fetchone()
    assert row[0] == "Alice Smith, Bob Jones"


# ── run_ingestion ────────────────────────────────────────────────────────────

import config as _config


@patch("ingestion.fetch_papers.time.sleep")
@patch("ingestion.fetch_papers.store_paper")
@patch("ingestion.fetch_papers.extract_sections")
@patch("ingestion.fetch_papers.download_pdf")
@patch("ingestion.fetch_papers.filter_by_keywords")
@patch("ingestion.fetch_papers.fetch_papers_by_category")
@patch("ingestion.fetch_papers.init_db")
def test_run_ingestion_fetches_one_per_category(
    mock_init_db, mock_fetch, mock_filter, mock_download,
    mock_extract, mock_store, mock_sleep,
):
    mock_conn = MagicMock()
    mock_init_db.return_value = mock_conn
    mock_paper = MagicMock()
    mock_fetch.return_value = [mock_paper]
    mock_filter.return_value = [mock_paper]
    mock_download.return_value = Path("data/pdfs/test.pdf")
    mock_extract.return_value = {
        "abstract": "x", "introduction": "x", "results": "x", "conclusion": "x"
    }
    mock_store.return_value = True

    run_ingestion()

    assert mock_fetch.call_count == len(_config.ARXIV_CATEGORIES)


@patch("ingestion.fetch_papers.time.sleep")
@patch("ingestion.fetch_papers.store_paper")
@patch("ingestion.fetch_papers.extract_sections")
@patch("ingestion.fetch_papers.download_pdf")
@patch("ingestion.fetch_papers.filter_by_keywords")
@patch("ingestion.fetch_papers.fetch_papers_by_category")
@patch("ingestion.fetch_papers.init_db")
def test_run_ingestion_skips_paper_on_pdf_error(
    mock_init_db, mock_fetch, mock_filter, mock_download,
    mock_extract, mock_store, mock_sleep,
):
    mock_conn = MagicMock()
    mock_init_db.return_value = mock_conn
    mock_paper = MagicMock()
    mock_fetch.return_value = [mock_paper]
    mock_filter.return_value = [mock_paper]
    mock_download.side_effect = Exception("Network error")

    run_ingestion()  # must not raise

    mock_store.assert_not_called()
