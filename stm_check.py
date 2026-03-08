import sqlite3

DB = "/app/data/memory.db"
conn = sqlite3.connect(DB)
cur = conn.cursor()

cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in cur.fetchall()]
print("Tables:", tables)
print()

for t in tables:
    cur.execute(f"SELECT COUNT(*) FROM {t}")
    count = cur.fetchone()[0]
    print(f"=== {t} ({count} rows) ===")
    cur.execute(f"SELECT * FROM {t} ORDER BY rowid DESC")
    cols = [d[0] for d in cur.description]
    print("Columns:", cols)
    print()
    for row in cur.fetchall():
        print("---")
        for col, val in zip(cols, row):
            if col in ("raw_input", "final_output", "content") and val:
                print(f"  {col}: {str(val)[:400]}")
            else:
                print(f"  {col}: {val}")
        print()

conn.close()
