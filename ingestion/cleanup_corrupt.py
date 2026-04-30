import argparse
import os
import sqlite3
from pathlib import Path

import config


def is_corrupt(pdf_path: Path) -> bool:
    try:
        with open(pdf_path, "rb") as f:
            header = f.read(8)
            if not header.startswith(b"%PDF-"):
                return True
            f.seek(-1024, os.SEEK_END)
            tail = f.read()
        return b"%%EOF" not in tail
    except OSError:
        return True


def find_corrupt_pdfs(pdfs_dir: Path) -> list[Path]:
    return [p for p in pdfs_dir.glob("*.pdf") if is_corrupt(p)]


def run_cleanup(dry_run: bool) -> None:
    pdfs_dir = Path(config.DB_PATH).parent / "pdfs"
    if not pdfs_dir.exists():
        print("No pdfs directory found. Nothing to do.")
        return

    all_pdfs = list(pdfs_dir.glob("*.pdf"))
    print(f"Scanning {len(all_pdfs)} PDFs...")

    corrupt = find_corrupt_pdfs(pdfs_dir)
    if not corrupt:
        print("No corrupt PDFs found.")
        return

    paper_ids = [p.stem for p in corrupt]

    conn = sqlite3.connect(config.DB_PATH)
    placeholders = ",".join("?" * len(paper_ids))
    in_db = {
        row[0]
        for row in conn.execute(
            f"SELECT paper_id FROM papers WHERE paper_id IN ({placeholders})",
            paper_ids,
        )
    }
    conn.close()

    in_db_count = len(in_db)
    disk_only_count = len(corrupt) - in_db_count

    print(f"Found {len(corrupt)} corrupt PDFs (missing PDF header or %%EOF trailer).")
    print(f"  {in_db_count} have SQLite entries → {'will delete rows' if not dry_run else 'would delete rows'}")
    print(f"  {disk_only_count} are on disk only (no DB entry)")

    if dry_run:
        print("\n[dry-run] No changes made.")
        if corrupt:
            print("Files that would be removed:")
            for p in sorted(corrupt):
                marker = "[DB+disk]" if p.stem in in_db else "[disk only]"
                print(f"  {marker} {p.name}")
        return

    if in_db:
        conn = sqlite3.connect(config.DB_PATH)
        conn.execute(
            f"DELETE FROM papers WHERE paper_id IN ({placeholders})",
            list(in_db),
        )
        conn.commit()
        conn.close()
        print(f"\nDeleted {in_db_count} SQLite rows.")

    for p in corrupt:
        p.unlink()
    print(f"Deleted {len(corrupt)} PDF files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove corrupt/truncated ArXiv PDFs and their SQLite entries.")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing.")
    args = parser.parse_args()
    run_cleanup(dry_run=args.dry_run)
