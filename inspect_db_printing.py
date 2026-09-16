"""
Standalone SQLite database inspector. No dependencies beyond the Python
standard library (sqlite3, array, collections) -- runs on its own before
you've installed torch/numpy/sklearn, and works on any .db file, not just
the OSSL schema.

Usage:
    python inspect_db.py path/to/your_dataset.db
    python inspect_db.py path/to/your_dataset.db --full     # no truncation

What it reports, per table:
  - columns and declared types
  - row count
  - for numeric (REAL/INTEGER) columns: non-null coverage, min, max, mean
    (all computed with a single SQL aggregate query per table, so this
    stays fast even on tables with 100k+ rows)
  - for TEXT columns with low cardinality (e.g. categorical fields like
    a train/test split, or a dataset source code): value counts
  - for TEXT columns with high cardinality: just a distinct count (to
    avoid flooding the output with every sample_id)
  - for BLOB columns: row count and, for one sample row, the decoded
    length assuming float32 (common for spectral data) -- never dumps
    raw bytes

Also, since it's common for related tables to share an ID column (e.g.
multiple tables with a 'sample_id' column), it detects any column name
shared by 2+ tables and reports pairwise overlap counts -- this is the
first thing you want to know before writing a join.
"""
import sqlite3
import sys
import array

CATEGORICAL_MAX_DISTINCT = 20
MAX_VALUES_PRINTED = 20


def get_tables(conn):
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    return [row[0] for row in cur.fetchall()]


def get_columns(conn, table):
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info('{table}');")
    # (cid, name, type, notnull, dflt_value, pk)
    return [(row[1], row[2]) for row in cur.fetchall()]


def get_row_count(conn, table):
    cur = conn.cursor()
    cur.execute(f"SELECT COUNT(*) FROM '{table}';")
    return cur.fetchone()[0]


def numeric_stats(conn, table, numeric_cols):
    """One aggregate query for every numeric column in the table."""
    if not numeric_cols:
        return {}
    parts = []
    for col in numeric_cols:
        parts.append(f'COUNT("{col}") AS "{col}__n"')
        parts.append(f'MIN("{col}") AS "{col}__min"')
        parts.append(f'MAX("{col}") AS "{col}__max"')
        parts.append(f'AVG("{col}") AS "{col}__avg"')
    query = f"SELECT {', '.join(parts)} FROM '{table}';"
    cur = conn.cursor()
    cur.execute(query)
    row = cur.fetchone()
    desc = [d[0] for d in cur.description]
    values = dict(zip(desc, row))
    stats = {}
    for col in numeric_cols:
        stats[col] = {
            "n": values[f"{col}__n"],
            "min": values[f"{col}__min"],
            "max": values[f"{col}__max"],
            "avg": values[f"{col}__avg"],
        }
    return stats


def text_value_counts(conn, table, col):
    cur = conn.cursor()
    cur.execute(
        f'SELECT "{col}", COUNT(*) FROM \'{table}\' GROUP BY "{col}" '
        f'ORDER BY COUNT(*) DESC LIMIT {CATEGORICAL_MAX_DISTINCT + 1};'
    )
    return cur.fetchall()


def distinct_count(conn, table, col):
    cur = conn.cursor()
    cur.execute(f'SELECT COUNT(DISTINCT "{col}") FROM \'{table}\';')
    return cur.fetchone()[0]


def blob_summary(conn, table, col, dtype_char="f"):
    """Decodes one sample BLOB (assumes packed floats) just to report
    its length -- never prints raw bytes."""
    cur = conn.cursor()
    cur.execute(f'SELECT "{col}" FROM \'{table}\' WHERE "{col}" IS NOT NULL LIMIT 1;')
    row = cur.fetchone()
    if row is None or row[0] is None:
        return None
    blob = row[0]
    try:
        arr = array.array(dtype_char)
        arr.frombytes(blob)
        return {"sample_length": len(arr), "bytes": len(blob), "dtype_guess": "float32"}
    except Exception:
        return {"bytes": len(blob), "dtype_guess": "unknown (not float32-decodable)"}


def inspect(db_path: str, full: bool = False):
    conn = sqlite3.connect(db_path)
    tables = get_tables(conn)
    if not tables:
        print("No tables found -- is this actually a SQLite file?")
        return

    print(f"Database: {db_path}")
    print(f"Tables: {len(tables)}\n")

    # Track which columns appear in which tables, for the cross-table
    # ID-overlap report at the end.
    col_to_tables = {}

    per_table_columns = {}
    for table in tables:
        cols = get_columns(conn, table)
        per_table_columns[table] = cols
        for name, _ in cols:
            col_to_tables.setdefault(name, []).append(table)

    for table in tables:
        cols = per_table_columns[table]
        n_rows = get_row_count(conn, table)
        print(f"=== {table} ({n_rows} rows, {len(cols)} columns) ===")

        numeric_cols = [c for c, t in cols if t.upper() in ("REAL", "INTEGER") and c != "idx"]
        text_cols = [c for c, t in cols if t.upper() in ("TEXT",)]
        blob_cols = [c for c, t in cols if t.upper() == "BLOB"]

        if numeric_cols and n_rows > 0:
            stats = numeric_stats(conn, table, numeric_cols)
            # sort by coverage descending so the most-populated columns
            # (usually your best target candidates) show up first
            ordered = sorted(stats.items(), key=lambda kv: -(kv[1]["n"] or 0))
            shown = ordered if full else ordered[:30]
            for col, s in shown:
                pct = 100.0 * (s["n"] or 0) / n_rows
                min_s = f"{s['min']:.4g}" if s["min"] is not None else "None"
                max_s = f"{s['max']:.4g}" if s["max"] is not None else "None"
                avg_s = f"{s['avg']:.4g}" if s["avg"] is not None else "None"
                print(
                    f"  {col:35s} coverage={pct:5.1f}%  "
                    f"range=[{min_s}, {max_s}]  mean={avg_s}"
                )
            if not full and len(ordered) > 30:
                print(f"  ... and {len(ordered) - 30} more numeric columns (use --full to see all)")

        for col in text_cols:
            n_distinct = distinct_count(conn, table, col)
            if n_distinct <= CATEGORICAL_MAX_DISTINCT:
                counts = text_value_counts(conn, table, col)
                counts_str = ", ".join(f"{v!r}: {c}" for v, c in counts)
                print(f"  {col:35s} categorical ({n_distinct} values): {counts_str}")
            else:
                print(f"  {col:35s} text, {n_distinct} distinct values (not shown)")

        for col in blob_cols:
            info = blob_summary(conn, table, col)
            if info is None:
                print(f"  {col:35s} BLOB, no non-null sample found")
            elif "sample_length" in info:
                print(
                    f"  {col:35s} BLOB, {info['bytes']} bytes/sample "
                    f"-> {info['sample_length']} float32 values (one sample checked)"
                )
            else:
                print(f"  {col:35s} BLOB, {info['bytes']} bytes/sample, {info['dtype_guess']}")

        print()

    # Cross-table ID overlap: for any column name shared by 2+ tables,
    # report how many values overlap between each pair of tables.
    shared_cols = {c: ts for c, ts in col_to_tables.items() if len(ts) > 1}
    if shared_cols:
        print("=== Cross-table column overlap ===")
        for col, owning_tables in shared_cols.items():
            for i in range(len(owning_tables)):
                for j in range(i + 1, len(owning_tables)):
                    t1, t2 = owning_tables[i], owning_tables[j]
                    cur = conn.cursor()
                    cur.execute(
                        f'SELECT COUNT(DISTINCT a."{col}") FROM \'{t1}\' a '
                        f'INNER JOIN \'{t2}\' b ON a."{col}" = b."{col}";'
                    )
                    overlap = cur.fetchone()[0]
                    n1 = get_row_count(conn, t1)
                    n2 = get_row_count(conn, t2)
                    print(
                        f"  '{col}': {t1} ({n1} rows) <-> {t2} ({n2} rows) "
                        f"-> {overlap} matching values"
                    )
        print()

    conn.close()


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    full = "--full" in sys.argv
    if len(args) != 1:
        print("Usage: python inspect_db.py path/to/your_dataset.db [--full]")
        sys.exit(1)
    inspect(args[0], full=full)
