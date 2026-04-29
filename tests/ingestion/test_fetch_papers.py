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
