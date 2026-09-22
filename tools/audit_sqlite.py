import sqlite3
conn = sqlite3.connect('data/processed/foodprice_nutrition.sqlite')
cur = conn.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
tables = [r[0] for r in cur.fetchall()]
for t in tables:
    cur.execute(f'SELECT COUNT(*) FROM "{t}"')
    count = cur.fetchone()[0]
    print(f"  {t}: {count} rows")
conn.close()
print("SQLite OK")
