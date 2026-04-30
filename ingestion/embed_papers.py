import config
import json
import sqlite3

from llama_index.core import Document


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


def build_documents(row: dict) -> list[Document]:
    docs = []
    for section_name, text in row["sections"].items():
        if not text.strip():
            continue
        docs.append(
            Document(
                text=text,
                metadata={
                    "arxiv_id": row["paper_id"],
                    "title": row["title"],
                    "published_date": row["published_date"],
                    "section": section_name,
                    "categories": row["categories"],
                    "authors": row["authors"],
                },
            )
        )
    return docs
