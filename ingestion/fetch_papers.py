import config
import arxiv
import json
import re
import requests
import sqlite3
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from pathlib import Path


def init_db(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            paper_id       TEXT PRIMARY KEY,
            title          TEXT NOT NULL,
            authors        TEXT NOT NULL,
            pdf_url        TEXT NOT NULL,
            categories     TEXT NOT NULL,
            published_date TEXT NOT NULL,
            sections       TEXT NOT NULL,
            fetched_at     TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def filter_by_keywords(papers: list, keywords: list[str]) -> list:
    raise NotImplementedError


def fetch_papers_by_category(category: str, months_back: int) -> list:
    raise NotImplementedError


def download_pdf(paper, pdfs_dir: Path) -> Path:
    raise NotImplementedError


def extract_sections(pdf_path: Path, sections_to_extract: list[str]) -> dict[str, str]:
    raise NotImplementedError


def store_paper(
    conn: sqlite3.Connection, paper, sections: dict[str, str]
) -> bool:
    raise NotImplementedError


def run_ingestion() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    run_ingestion()
