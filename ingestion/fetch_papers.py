import config
import arxiv
import json
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
        max_results=None,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    results = []
    for paper in client.results(search):
        if paper.published < cutoff:
            break  # safe: results are sorted newest-first
        results.append(paper)
    return results


def download_pdf(paper, pdfs_dir: Path) -> Path:
    paper_id = paper.get_short_id()
    pdf_path = pdfs_dir / f"{paper_id}.pdf"
    if pdf_path.exists():
        return pdf_path
    paper.download_pdf(dirpath=str(pdfs_dir), filename=f"{paper_id}.pdf")
    return pdf_path


def _parse_tei_sections(tei_xml: str, sections_to_extract: list[str]) -> dict[str, str]:
    root = ET.fromstring(tei_xml)
    ns = {"tei": "http://www.tei-c.org/ns/1.0"}
    sections = {s: "" for s in sections_to_extract}

    if "abstract" in sections_to_extract:
        abstract_elem = root.find(".//tei:abstract", ns)
        if abstract_elem is not None:
            sections["abstract"] = " ".join(
                t.strip() for t in abstract_elem.itertext() if t.strip()
            )

    for div in root.findall(".//tei:body//tei:div", ns):
        head = div.find("tei:head", ns)
        if head is None:
            continue
        heading_lower = (head.text or "").lower()
        for section in sections_to_extract:
            if section == "abstract":
                continue
            if section.lower() in heading_lower and not sections[section]:
                sections[section] = " ".join(
                    " ".join(p.itertext()).strip()
                    for p in div.findall("tei:p", ns)
                    if "".join(p.itertext()).strip()
                )

    return sections


def extract_sections(pdf_path: Path, sections_to_extract: list[str]) -> dict[str, str]:
    with open(pdf_path, "rb") as f:
        response = requests.post(
            f"{config.GROBID_URL}/api/processFulltextDocument",
            files={"input": f},
            data={"consolidateHeader": "0"},
            timeout=config.GROBID_TIMEOUT,
        )
    response.raise_for_status()
    return _parse_tei_sections(response.text, sections_to_extract)


def store_paper(
    conn: sqlite3.Connection, paper, sections: dict[str, str]
) -> bool:
    raise NotImplementedError


def run_ingestion() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    run_ingestion()
