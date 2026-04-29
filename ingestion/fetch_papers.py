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
    keywords_lower = [kw.lower() for kw in keywords]

    def matches(paper) -> bool:
        text = (paper.title + " " + paper.summary).lower()
        return any(kw in text for kw in keywords_lower)

    return [p for p in papers if matches(p)]


def fetch_papers_by_category(category: str, months_back: int) -> list:
    cutoff = datetime.now(timezone.utc) - timedelta(days=months_back * 30)
    client = arxiv.Client()
    search = arxiv.Search(
        query=f"cat:{category}",
        max_results=float("inf"),
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    results = []
    for paper in client.results(search):
        if paper.published < cutoff:
            break
        results.append(paper)
    return results


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
