import chromadb
import config
import json
import sqlite3

from datetime import datetime, timezone
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from tracing import observe


def migrate_db(conn: sqlite3.Connection) -> None:
    try:
        conn.execute("ALTER TABLE papers ADD COLUMN embedded_at TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # column already exists


@observe(name="load_unembedded_papers")
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


@observe(name="build_documents")
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


def mark_embedded(conn: sqlite3.Connection, paper_id: str) -> None:
    conn.execute(
        "UPDATE papers SET embedded_at = ? WHERE paper_id = ?",
        (datetime.now(timezone.utc).isoformat(), paper_id),
    )
    conn.commit()


@observe(name="embed_papers")
def run_embedding() -> None:
    conn = sqlite3.connect(config.DB_PATH)
    try:
        migrate_db(conn)
        rows = load_unembedded_papers(conn)
        if not rows:
            print("No new papers to embed.")
            return

        all_docs = []
        for row in rows:
            all_docs.extend(build_documents(row))

        chroma_client = chromadb.PersistentClient(path=config.CHROMA_PATH)
        chroma_collection = chroma_client.get_or_create_collection(config.CHROMA_COLLECTION)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        embed_model = HuggingFaceEmbedding(model_name=config.EMBEDDING_MODEL)
        splitter = SentenceSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)

        VectorStoreIndex.from_documents(
            all_docs,
            storage_context=storage_context,
            embed_model=embed_model,
            transformations=[splitter],
        )

        for row in rows:
            mark_embedded(conn, row["paper_id"])

        print(f"Embedded {len(rows)} papers ({len(all_docs)} section documents).")
    finally:
        conn.close()


if __name__ == "__main__":
    run_embedding()
