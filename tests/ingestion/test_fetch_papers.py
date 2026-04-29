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
