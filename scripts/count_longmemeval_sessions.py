import json
from pathlib import Path

p = Path("data/longmemeval_s_cleaned.json")
data = json.loads(p.read_text(encoding="utf-8"))
q = len(data)
mentions = 0
unique = set()

for item in data:
    sids = item.get("haystack_session_ids") or []
    mentions += len(sids)
    unique.update(map(str, sids))

print(f"questions={q}")
print(f"session_mentions={mentions}")
print(f"unique_sessions={len(unique)}")
print(f"avg_sessions_per_question={mentions / q:.2f}")
print(f"reuse_ratio={1 - (len(unique) / mentions):.4f}")
