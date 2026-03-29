import sqlite3, json

conn = sqlite3.connect(r'C:\Capstone\LLMP-Capstone\v-benchmark\benchmark_results\.cache\session_extract_cache.db')
sid = '47f00a5f'
row = conn.execute('SELECT facts_json FROM session_cache WHERE session_id=?', (sid,)).fetchone()
if row:
    facts = json.loads(row[0])
    print(f'Facts stored: {len(facts)}')
    for i, f in enumerate(facts, 1):
        print(f'{i}. [{f.get("source","?").upper()}] {f["text"]}')
else:
    print('Session not in cache at all')
conn.close()
