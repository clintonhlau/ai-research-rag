import json
import sqlite3
from unittest.mock import MagicMock

import pytest

from ingestion.embed_papers import migrate_db, load_unembedded_papers


def _make_db(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "test.db"))
    conn.execute("""
        CREATE TABLE papers (
            paper_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            authors TEXT NOT NULL,
            pdf_url TEXT NOT NULL,
            categories TEXT NOT NULL,
            published_date TEXT NOT NULL,
            sections TEXT NOT NULL,
            fetched_at TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def _insert_paper(conn, paper_id="2301.00001v1"):
    conn.execute(
        "INSERT INTO papers (paper_id, title, authors, pdf_url, categories, published_date, sections, fetched_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            paper_id,
            "Test Paper",
            "Alice, Bob",
            "https://arxiv.org/pdf/test",
            "cs.AI, cs.LG",
            "2024-01-15T00:00:00+00:00",
            json.dumps({"abstract": "We study safety.", "introduction": "Intro text.", "results": "", "conclusion": "Done."}),
            "2024-01-16T00:00:00+00:00",
        ),
    )
    conn.commit()


# migrate_db

def test_migrate_db_adds_embedded_at_column(tmp_path):
    conn = _make_db(tmp_path)
    migrate_db(conn)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(papers)").fetchall()}
    assert "embedded_at" in cols
    conn.close()


def test_migrate_db_is_idempotent(tmp_path):
    conn = _make_db(tmp_path)
    migrate_db(conn)
    migrate_db(conn)  # must not raise
    conn.close()


# load_unembedded_papers

def test_load_unembedded_returns_paper_with_null_embedded_at(tmp_path):
    conn = _make_db(tmp_path)
    migrate_db(conn)
    _insert_paper(conn)
    rows = load_unembedded_papers(conn)
    assert len(rows) == 1
    assert rows[0]["paper_id"] == "2301.00001v1"
    conn.close()


def test_load_unembedded_excludes_already_embedded(tmp_path):
    conn = _make_db(tmp_path)
    migrate_db(conn)
    _insert_paper(conn)
    conn.execute(
        "UPDATE papers SET embedded_at = '2024-01-17T00:00:00+00:00' WHERE paper_id = '2301.00001v1'"
    )
    conn.commit()
    assert load_unembedded_papers(conn) == []
    conn.close()


def test_load_unembedded_parses_sections_json(tmp_path):
    conn = _make_db(tmp_path)
    migrate_db(conn)
    _insert_paper(conn)
    rows = load_unembedded_papers(conn)
    assert isinstance(rows[0]["sections"], dict)
    assert "abstract" in rows[0]["sections"]
    conn.close()


def test_load_unembedded_returns_expected_keys(tmp_path):
    conn = _make_db(tmp_path)
    migrate_db(conn)
    _insert_paper(conn)
    row = load_unembedded_papers(conn)[0]
    assert set(row.keys()) == {"paper_id", "title", "authors", "categories", "published_date", "sections"}
    conn.close()


def test_load_unembedded_returns_empty_when_no_papers(tmp_path):
    conn = _make_db(tmp_path)
    migrate_db(conn)
    assert load_unembedded_papers(conn) == []
    conn.close()
