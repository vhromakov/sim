#!/usr/bin/env python3
"""
Query a CSV file with SQL (SQLite in-memory), treating ALL values as TEXT.

Example:
  python query_csv_textsql.py data.csv \
    --query 'SELECT * FROM data LIMIT 10'

If headers have spaces/symbols, quote them:
  --query 'SELECT "Total Cost" FROM data WHERE "Total Cost" != ""'
"""

import argparse
import csv
import sqlite3
import sys


def qident(name: str) -> str:
    """Quote an SQLite identifier safely."""
    return '"' + name.replace('"', '""') + '"'


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", help="Path to CSV file")
    ap.add_argument("--query", required=True, help='SQL query, e.g. SELECT * FROM data LIMIT 10')
    ap.add_argument("--table", default="data", help="Table name (default: data)")
    ap.add_argument("--delimiter", default=",", help="CSV delimiter (default: ,)")
    ap.add_argument("--encoding", default="utf-8", help="File encoding (default: utf-8)")
    args = ap.parse_args()

    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()

    with open(args.csv_path, "r", encoding=args.encoding, newline="") as f:
        reader = csv.reader(f, delimiter=args.delimiter)

        try:
            headers = next(reader)
        except StopIteration:
            print("CSV is empty.", file=sys.stderr)
            return 2

        headers = [h.strip() if h is not None else "" for h in headers]
        # Handle empty/duplicate column names
        seen = {}
        fixed = []
        for h in headers:
            h0 = h or "col"
            n = seen.get(h0, 0) + 1
            seen[h0] = n
            fixed.append(h0 if n == 1 else f"{h0}_{n}")

        table = args.table
        col_defs = ", ".join(f"{qident(c)} TEXT" for c in fixed)
        cur.execute(f"CREATE TABLE {qident(table)} ({col_defs});")

        placeholders = ", ".join(["?"] * len(fixed))
        ins_sql = f"INSERT INTO {qident(table)} VALUES ({placeholders});"

        batch = []
        BATCH_SIZE = 5000

        for row in reader:
            # pad/trim ragged rows
            row = (row + [""] * len(fixed))[: len(fixed)]
            # keep everything as string (including empty)
            batch.append([str(x) for x in row])

            if len(batch) >= BATCH_SIZE:
                cur.executemany(ins_sql, batch)
                batch.clear()

        if batch:
            cur.executemany(ins_sql, batch)

    conn.commit()

    args.query = """
SELECT
  CASE
    WHEN NOT EXISTS (
      SELECT 1 FROM data
      WHERE "CashUpdateType" IN ('1','3')
    ) THEN 1
    WHEN EXISTS (SELECT 1 FROM pragma_table_info('data') WHERE name IN ('OpenTradeDataCashAmount', 'OpenSettlementDateCashAmount'))
    THEN 1
    ELSE 0
  END AS ok;
"""

    # Run query
    try:
        cur.execute(args.query)
        rows = cur.fetchall()
        headers_out = [d[0] for d in cur.description] if cur.description else []
    except sqlite3.Error as e:
        print(f"SQL error: {e}", file=sys.stderr)
        return 1

    # Print results as TSV to stdout
    if headers_out:
        print("\t".join(headers_out))
    for r in rows:
        print("\t".join("" if v is None else str(v) for v in r))

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
