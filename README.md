# Vektori

> Open-source, self-hostable memory engine for AI agents.

**Knowledge graphs give agents the answer. Vektori gives agents the answer, the reasoning, AND the story.**

Vektori uses a three-layer sentence graph — Facts (L0), Insights (L1), Sentences (L2) — to preserve full conversational context across sessions. Unlike Mem0 or Zep that compress conversations into entity-relationship triples, Vektori preserves the actual conversational flow so agents understand *what happened*, not just *what is*.

## Architecture

```
FACT LAYER (L0)      ← Primary vector search surface. Short, crisp statements.
      ↕ insight_facts
INSIGHT LAYER (L1)   ← Discovered via graph traversal. NOT vector search.
      ↕ insight_sources / fact_sources
SENTENCE LAYER (L2)  ← Raw conversation. Sequential NEXT edges within sessions.
```

Search hits facts → graph traversal discovers insights → trace to source sentences → expand session context. One database (Postgres + pgvector), no Neo4j, no Qdrant.

## Quickstart (zero config)

```bash
pip install vektori
```

```python
import asyncio
from vektori import Vektori

async def main():
    # SQLite by default — no Docker, no setup
    v = Vektori(
        embedding_model="openai:text-embedding-3-small",
        extraction_model="openai:gpt-4o-mini",
    )

    await v.add(
        messages=[
            {"role": "user", "content": "I only use WhatsApp, please don't email me."},
            {"role": "assistant", "content": "Got it, WhatsApp only."},
        ],
        session_id="call-001",
        user_id="user-123",
    )

    results = await v.search(
        query="How does this user prefer to communicate?",
        user_id="user-123",
        depth="l1",  # facts + insights
    )
    print(results)
    await v.close()

asyncio.run(main())
```

## Retrieval Depths

| Depth | Returns | ~Tokens | Use Case |
|-------|---------|---------|----------|
| `l0`  | Facts only | 50–200 | Quick lookup, agent planning |
| `l1`  | Facts + insights | 200–500 | **Default.** Answer + actionable context |
| `l2`  | Facts + insights + sentences | 1000–3000 | Full story, trajectory analysis |

```python
# L0: cheapest, just the facts
results = await v.search(query, user_id, depth="l0")

# L1: default — facts + cross-session patterns
results = await v.search(query, user_id, depth="l1")

# L2: full story with session context window
results = await v.search(query, user_id, depth="l2", context_window=3)
```

## Storage Backends

```python
# SQLite (default, zero config)
v = Vektori()

# PostgreSQL + pgvector (production)
v = Vektori(database_url="postgresql://localhost:5432/vektori")

# In-memory (testing / CI)
v = Vektori(storage_backend="memory")
```

## Multi-Model Support

```python
# OpenAI
v = Vektori(
    embedding_model="openai:text-embedding-3-small",
    extraction_model="openai:gpt-4o-mini",
)

# Anthropic
v = Vektori(
    embedding_model="anthropic:voyage-3",
    extraction_model="anthropic:claude-haiku-4-5-20251001",
)

# Fully local — no API keys needed
v = Vektori(
    embedding_model="ollama:nomic-embed-text",
    extraction_model="ollama:llama3",
)

# Sentence Transformers (local, no Ollama)
v = Vektori(embedding_model="sentence-transformers:all-MiniLM-L6-v2")
```

## Docker (Postgres)

```bash
git clone https://github.com/vektori-ai/vektori
cd vektori
docker compose up -d
DATABASE_URL=postgresql://vektori:vektori@localhost:5432/vektori python examples/quickstart_postgres.py
```

## Why Not Mem0 / Zep?

| | Mem0 / Zep | Vektori |
|---|---|---|
| Storage model | Entity-relation triples | Three-layer sentence graph |
| What you get | The answer | The answer + reasoning + story |
| Cross-session patterns | Manual graph queries | Auto-discovered via insight layer |
| Default backend | Requires external DB | SQLite, zero config |
| Local/offline | No | Yes (Ollama) |
| Open source | Partial | Full MIT |

## License

MIT — see [LICENSE](LICENSE).
