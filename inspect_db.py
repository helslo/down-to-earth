"""
Quick inspector for an unknown SQLite .db file. Run this first so we can
see what tables/columns your dataset actually has, then adapt
load_data_from_db() in train_example.py to match.

Usage:
    python inspect_db.py path/to/your_dataset.db
"""
import sqlite3
import sys


def inspect(db_path: str, sample_rows: int = 3):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cur.fetchall()]

    if not tables:
        print("No tables found -- is this actually a SQLite file?")
        return

    for table in tables:
        print(f"\n=== TABLE: {table} ===")
        cur.execute(f"PRAGMA table_info('{table}');")
        columns = cur.fetchall()  # (cid, name, type, notnull, dflt_value, pk)
        for col in columns:
            print(f"  {col[1]:30s} {col[2]}")

        cur.execute(f"SELECT COUNT(*) FROM '{table}';")
        n_rows = cur.fetchone()[0]
        print(f"  -> {n_rows} rows")

        if n_rows > 0:
            col_names = [c[1] for c in columns]
            cur.execute(f"SELECT * FROM '{table}' LIMIT {sample_rows};")
            print(f"  sample rows ({', '.join(col_names)}):")
            for row in cur.fetchall():
                print(f"    {row}")

    conn.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_db.py path/to/your_dataset.db")
        sys.exit(1)
    inspect(sys.argv[1])
