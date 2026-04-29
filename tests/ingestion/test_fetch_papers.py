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
