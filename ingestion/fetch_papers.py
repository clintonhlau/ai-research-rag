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


_PAGE_SIZE = 100


def _fetch_page_with_backoff(category: str, offset: int) -> list:
    """Fetch one page of results at `offset`, retrying with exponential backoff on 429."""
    search = arxiv.Search(
        query=f"cat:{category}",
        max_results=offset + _PAGE_SIZE,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    # delay_seconds=0: we own the inter-page sleep in fetch_papers_by_category
    client = arxiv.Client(page_size=_PAGE_SIZE, delay_seconds=0, num_retries=1)
    for attempt in range(config.ARXIV_NUM_RETRIES):
        try:
            return list(client.results(search, offset=offset))
        except arxiv.HTTPError as e:
            if e.status == 429 and attempt < config.ARXIV_NUM_RETRIES - 1:
                wait = config.ARXIV_RATE_LIMIT_SLEEP * (config.ARXIV_BACKOFF_BASE ** attempt)
                print(f"  429 at offset {offset}, retrying in {wait}s (attempt {attempt + 1})")
                time.sleep(wait)
            else:
                raise
    return []  # unreachable


def fetch_papers_by_category(category: str, months_back: int) -> list:
    cutoff = datetime.now(timezone.utc) - timedelta(days=months_back * 30)
    results = []
    offset = 0

    while True:
        page = _fetch_page_with_backoff(category, offset)
        if not page:
            break
        for paper in page:
            if paper.published < cutoff:
                return results  # sorted newest-first, so we're done
            results.append(paper)
        if len(page) < _PAGE_SIZE:
            break  # last page
        offset += _PAGE_SIZE
        time.sleep(config.ARXIV_RATE_LIMIT_SLEEP)

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
    paper_id = paper.get_short_id()
    if conn.execute("SELECT 1 FROM papers WHERE paper_id = ?", (paper_id,)).fetchone():
        return False
    conn.execute(
        """INSERT INTO papers
           (paper_id, title, authors, pdf_url, categories, published_date, sections, fetched_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            paper_id,
            paper.title,
            ", ".join(a.name for a in paper.authors),
            paper.pdf_url,
            ", ".join(paper.categories),
            paper.published.isoformat(),
            json.dumps(sections),
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    return True


def run_ingestion() -> None:
    pdfs_dir = Path("data/pdfs")
    pdfs_dir.mkdir(parents=True, exist_ok=True)

    conn = init_db(config.DB_PATH)
    total_new = 0

    for category in config.ARXIV_CATEGORIES:
        print(f"Fetching {category}...")
        papers = fetch_papers_by_category(category, config.ARXIV_MONTHS_BACK)
        filtered = filter_by_keywords(papers, config.ARXIV_SAFETY_KEYWORDS)
        print(f"  {len(papers)} fetched, {len(filtered)} matched keywords")

        for paper in filtered:
            try:
                pdf_path = download_pdf(paper, pdfs_dir)
                sections = extract_sections(pdf_path, config.SECTIONS_TO_EXTRACT)
                if store_paper(conn, paper, sections):
                    total_new += 1
            except Exception as e:
                print(f"  Skipping {paper.get_short_id()}: {e}")
            time.sleep(config.ARXIV_RATE_LIMIT_SLEEP)

    conn.close()
    print(f"Done. {total_new} new papers added.")


if __name__ == "__main__":
    run_ingestion()
