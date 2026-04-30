import config
import json
import sqlite3
from datetime import datetime, timezone


def migrate_db(conn: sqlite3.Connection) -> None:
    try:
        conn.execute("ALTER TABLE papers ADD COLUMN embedded_at TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # column already exists


def load_unembedded_papers(conn: sqlite3.Connection) -> list[dict]:
    cursor = conn.execute(
        "SELECT paper_id, title, authors, categories, published_date, sections "
        "FROM papers WHERE embedded_at IS NULL"
    )
    rows = []
    for paper_id, title, authors, categories, published_date, sections_json in cursor.fetchall():
        rows.append({
            "paper_id": paper_id,
            "title": title,
            "authors": authors,
            "categories": categories,
            "published_date": published_date,
            "sections": json.loads(sections_json),
        })
    return rows
