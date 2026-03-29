"""Remove 0-fact entries from session cache so they get re-extracted."""
import sqlite3, json

db = r'C:\Capstone\LLMP-Capstone\v-benchmark\benchmark_results\.cache\session_extract_cache.db'
conn = sqlite3.connect(db)

rows = conn.execute('SELECT session_id, facts_json FROM session_cache').fetchall()
empty = [r[0] for r in rows if len(json.loads(r[1])) == 0]

print(f'Total cached sessions: {len(rows)}')
print(f'Empty (0-fact) sessions: {len(empty)}')
for sid in empty:
    print(f'  {sid}')

if empty:
    conn.executemany('DELETE FROM session_cache WHERE session_id = ?', [(s,) for s in empty])
    conn.commit()
    print(f'\nDeleted {len(empty)} empty entries. They will be re-extracted on next run.')
else:
    print('\nNo empty entries found.')

conn.close()
