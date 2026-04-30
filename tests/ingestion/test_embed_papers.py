import json
import sqlite3

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


# build_documents

from ingestion.embed_papers import build_documents


_SAMPLE_ROW = {
    "paper_id": "2301.00001v1",
    "title": "AI Safety Paper",
    "authors": "Alice Smith, Bob Jones",
    "categories": "cs.AI, cs.LG",
    "published_date": "2024-01-15T00:00:00+00:00",
    "sections": {
        "abstract": "We study alignment.",
        "introduction": "Safety matters.",
        "results": "",
        "conclusion": "We conclude success.",
    },
}


def test_build_documents_skips_empty_sections():
    docs = build_documents(_SAMPLE_ROW)
    section_names = [d.metadata["section"] for d in docs]
    assert "results" not in section_names


def test_build_documents_returns_one_doc_per_nonempty_section():
    docs = build_documents(_SAMPLE_ROW)
    assert len(docs) == 3  # abstract, introduction, conclusion


def test_build_documents_text_matches_section_content():
    docs = build_documents(_SAMPLE_ROW)
    abstract_doc = next(d for d in docs if d.metadata["section"] == "abstract")
    assert abstract_doc.text == "We study alignment."


def test_build_documents_metadata_contains_all_required_keys():
    docs = build_documents(_SAMPLE_ROW)
    required = {"arxiv_id", "title", "published_date", "section", "categories", "authors"}
    for doc in docs:
        assert required.issubset(doc.metadata.keys()), f"Missing keys in {doc.metadata}"


def test_build_documents_arxiv_id_matches_paper_id():
    docs = build_documents(_SAMPLE_ROW)
    for doc in docs:
        assert doc.metadata["arxiv_id"] == "2301.00001v1"


def test_build_documents_returns_empty_list_when_all_sections_empty():
    row = {**_SAMPLE_ROW, "sections": {"abstract": "", "introduction": "", "results": "", "conclusion": ""}}
    assert build_documents(row) == []


def test_build_documents_whitespace_only_section_is_skipped():
    row = {**_SAMPLE_ROW, "sections": {"abstract": "   ", "introduction": "Real text."}}
    docs = build_documents(row)
    assert len(docs) == 1
    assert docs[0].metadata["section"] == "introduction"


from datetime import datetime
from unittest.mock import MagicMock, patch

from ingestion.embed_papers import mark_embedded, run_embedding


# mark_embedded

def test_mark_embedded_sets_embedded_at(tmp_path):
    conn = _make_db(tmp_path)
    migrate_db(conn)
    _insert_paper(conn)
    mark_embedded(conn, "2301.00001v1")
    row = conn.execute(
        "SELECT embedded_at FROM papers WHERE paper_id = '2301.00001v1'"
    ).fetchone()
    assert row[0] is not None
    conn.close()


def test_mark_embedded_stores_iso_timestamp(tmp_path):
    conn = _make_db(tmp_path)
    migrate_db(conn)
    _insert_paper(conn)
    mark_embedded(conn, "2301.00001v1")
    row = conn.execute(
        "SELECT embedded_at FROM papers WHERE paper_id = '2301.00001v1'"
    ).fetchone()
    datetime.fromisoformat(row[0])  # must not raise
    conn.close()


# run_embedding

@patch("ingestion.embed_papers.VectorStoreIndex")
@patch("ingestion.embed_papers.load_unembedded_papers")
@patch("ingestion.embed_papers.migrate_db")
@patch("ingestion.embed_papers.sqlite3.connect")
def test_run_embedding_skips_when_no_unembedded_papers(
    mock_connect, mock_migrate, mock_load, mock_index,
):
    mock_connect.return_value = MagicMock()
    mock_load.return_value = []
    run_embedding()
    mock_index.from_documents.assert_not_called()


@patch("ingestion.embed_papers.mark_embedded")
@patch("ingestion.embed_papers.build_documents")
@patch("ingestion.embed_papers.load_unembedded_papers")
@patch("ingestion.embed_papers.migrate_db")
@patch("ingestion.embed_papers.VectorStoreIndex")
@patch("ingestion.embed_papers.ChromaVectorStore")
@patch("ingestion.embed_papers.StorageContext")
@patch("ingestion.embed_papers.HuggingFaceEmbedding")
@patch("ingestion.embed_papers.SentenceSplitter")
@patch("ingestion.embed_papers.chromadb")
@patch("ingestion.embed_papers.sqlite3.connect")
def test_run_embedding_marks_all_papers_embedded(
    mock_connect, mock_chromadb, mock_splitter,
    mock_embed, mock_storage_ctx, mock_chroma_vs, mock_index,
    mock_migrate, mock_load, mock_build, mock_mark,
):
    mock_conn = MagicMock()
    mock_connect.return_value = mock_conn
    fake_row = {
        "paper_id": "2301.00001v1",
        "title": "AI Safety Paper",
        "authors": "Alice Smith, Bob Jones",
        "categories": "cs.AI, cs.LG",
        "published_date": "2024-01-15T00:00:00+00:00",
        "sections": {"abstract": "We study alignment.", "introduction": "Safety matters.", "results": "", "conclusion": "Done."},
    }
    mock_load.return_value = [fake_row]
    mock_build.return_value = [MagicMock()]

    run_embedding()

    mock_mark.assert_called_once_with(mock_conn, "2301.00001v1")


@patch("ingestion.embed_papers.mark_embedded")
@patch("ingestion.embed_papers.build_documents")
@patch("ingestion.embed_papers.load_unembedded_papers")
@patch("ingestion.embed_papers.migrate_db")
@patch("ingestion.embed_papers.VectorStoreIndex")
@patch("ingestion.embed_papers.ChromaVectorStore")
@patch("ingestion.embed_papers.StorageContext")
@patch("ingestion.embed_papers.HuggingFaceEmbedding")
@patch("ingestion.embed_papers.SentenceSplitter")
@patch("ingestion.embed_papers.chromadb")
@patch("ingestion.embed_papers.sqlite3.connect")
def test_run_embedding_calls_from_documents_with_all_docs(
    mock_connect, mock_chromadb, mock_splitter,
    mock_embed, mock_storage_ctx, mock_chroma_vs, mock_index,
    mock_migrate, mock_load, mock_build, mock_mark,
):
    mock_conn = MagicMock()
    mock_connect.return_value = mock_conn
    doc1, doc2 = MagicMock(), MagicMock()
    fake_row = {
        "paper_id": "2301.00001v1",
        "title": "AI Safety Paper",
        "authors": "Alice Smith, Bob Jones",
        "categories": "cs.AI, cs.LG",
        "published_date": "2024-01-15T00:00:00+00:00",
        "sections": {"abstract": "We study alignment."},
    }
    mock_load.return_value = [fake_row]
    mock_build.return_value = [doc1, doc2]

    run_embedding()

    call_docs = mock_index.from_documents.call_args.args[0]
    assert doc1 in call_docs and doc2 in call_docs


def test_run_embedding_is_decorated_with_observe():
    assert hasattr(run_embedding, "__wrapped__"), (
        "run_embedding must be decorated with @observe"
    )


def test_load_unembedded_papers_is_decorated_with_observe():
    assert hasattr(load_unembedded_papers, "__wrapped__"), (
        "load_unembedded_papers must be decorated with @observe"
    )
