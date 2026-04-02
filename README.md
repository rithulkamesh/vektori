# Vektori

> **Memory that remembers the story, not just the facts.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/vektori)](https://pypi.org/project/vektori/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Discord](https://img.shields.io/badge/Discord-join-5865F2?logo=discord&logoColor=white)](YOUR_DISCORD_LINK)

<!-- DEMO PLACEHOLDER — replace with a GIF or video showing add → search → facts + insights output -->
<!-- ![Vektori Demo](assets/demo.gif) -->
> 📹 **Demo coming soon** — add → search → facts + insights, in under 5 seconds.

---

Other memory layers tell your agent *what is*.
Vektori tells it *what happened, why it matters, and what it means.*

Most memory systems compress conversations into entity-relationship triples. You lose the texture. You lose the reasoning. You lose the story. Vektori uses a **three-layer sentence graph** so agents don't just recall preferences — they understand trajectories.

```
FACT LAYER (L0)      ← vector search surface. Short, crisp statements.
        ↕
INSIGHT LAYER (L1)   ← cross-session patterns, auto-discovered via graph traversal.
        ↕
SENTENCE LAYER (L2)  ← raw conversation. Sequential NEXT edges. The full story.
```

Search hits Facts → graph discovers Insights → traces back to source Sentences.
One database. Postgres or SQLite. No Neo4j. No Qdrant. No infra drama.

---

## Install

```bash
pip install vektori
```

That's it. No Docker, no external services. SQLite by default.

---

## 30-Second Quickstart

```python
import asyncio
from vektori import Vektori

async def main():
    v = Vektori(
        embedding_model="openai:text-embedding-3-small",
        extraction_model="openai:gpt-4o-mini",
    )

    await v.add(
        messages=[
            {"role": "user", "content": "I only use WhatsApp, please don't email me."},
            {"role": "assistant", "content": "Got it, WhatsApp only."},
            {"role": "user", "content": "My outstanding amount is ₹45,000 and I can pay by Friday."},
        ],
        session_id="call-001",
        user_id="user-123",
    )

    results = await v.search(
        query="How does this user prefer to communicate?",
        user_id="user-123",
        depth="l1",  # facts + cross-session insights
    )

    for fact in results["facts"]:
        print(f"[{fact['score']:.2f}] {fact['text']}")
    for insight in results["insights"]:
        print(f"insight: {insight['text']}")

    await v.close()

asyncio.run(main())
```

**Output:**
```
[0.94] User prefers WhatsApp communication
[0.81] Outstanding balance of ₹45,000, payment expected Friday
insight: User consistently avoids email — route all comms to WhatsApp
```

---

## Retrieval Depths

Pick how deep you want to go. Pay only for what you need.

| Depth | Returns | ~Tokens | When to use |
|-------|---------|---------|-------------|
| `l0`  | Facts only | 50–200 | Fast lookup, agent planning, tool calls |
| `l1`  | Facts + Insights | 200–500 | **Default.** Full answer with context |
| `l2`  | Facts + Insights + raw Sentences | 1000–3000 | Trajectory analysis, full story replay |

```python
# Just the facts — cheapest, fastest
results = await v.search(query, user_id, depth="l0")

# Facts + cross-session patterns (recommended)
results = await v.search(query, user_id, depth="l1")

# Everything — with surrounding conversation context
results = await v.search(query, user_id, depth="l2", context_window=3)
```

---

## Build an Agent with Memory

Drop Vektori into any agent loop in three lines:

```python
import asyncio
from openai import AsyncOpenAI
from vektori import Vektori

client = AsyncOpenAI()

async def chat(user_id: str):
    v = Vektori(
        embedding_model="openai:text-embedding-3-small",
        extraction_model="openai:gpt-4o-mini",
    )
    session_id = f"session-{user_id}-001"
    history = []

    print("Chat with memory (type 'quit' to exit)\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            break

        # 1. Pull relevant memory
        mem = await v.search(query=user_input, user_id=user_id, depth="l1")
        facts = "\n".join(f"- {f['text']}" for f in mem.get("facts", []))
        insights = "\n".join(f"- {i['text']}" for i in mem.get("insights", []))

        # 2. Inject into system prompt
        system = "You are a helpful assistant with memory.\n"
        if facts:    system += f"\nKnown facts:\n{facts}"
        if insights: system += f"\nBehavioral insights:\n{insights}"

        # 3. Get response
        history.append({"role": "user", "content": user_input})
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system}, *history],
        )
        reply = resp.choices[0].message.content
        history.append({"role": "assistant", "content": reply})
        print(f"Assistant: {reply}\n")

        # 4. Store exchange
        await v.add(
            messages=[{"role": "user", "content": user_input},
                      {"role": "assistant", "content": reply}],
            session_id=session_id,
            user_id=user_id,
        )

    await v.close()

asyncio.run(chat("demo-user"))
```

More examples in [`/examples`](examples/):
- [`quickstart.py`](examples/quickstart.py) — fully local, zero API keys (Ollama)
- [`openai_agent.py`](examples/openai_agent.py) — OpenAI agent loop
- [`crewai_integration.py`](examples/crewai_integration.py) — CrewAI
- [`langgraph_integration.py`](examples/langgraph_integration.py) — LangGraph

---

## Storage Backends

```python
# SQLite (default) — zero config, starts instantly
v = Vektori()

# PostgreSQL + pgvector — production scale
v = Vektori(database_url="postgresql://localhost:5432/vektori")

# In-memory — tests / CI
v = Vektori(storage_backend="memory")
```

**Docker (Postgres):**
```bash
git clone https://github.com/vektori-ai/vektori
cd vektori
docker compose up -d
DATABASE_URL=postgresql://vektori:vektori@localhost:5432/vektori python examples/quickstart_postgres.py
```

---

## Model Support

Bring your own model stack. No vendor lock-in.

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

# Fully local — no API keys, no internet
v = Vektori(
    embedding_model="ollama:nomic-embed-text",
    extraction_model="ollama:llama3",
)

# Sentence Transformers (local, no Ollama required)
v = Vektori(embedding_model="sentence-transformers:all-MiniLM-L6-v2")

# BGE-M3 — multilingual, 1024-dim, best-in-class local embeddings
v = Vektori(embedding_model="bge:BAAI/bge-m3")

# LiteLLM — 100+ providers through one interface
v = Vektori(extraction_model="litellm:groq/llama3-8b-8192")
```

---

## Why Not Mem0 / Zep?

| | Mem0 / Zep | **Vektori** |
|---|---|---|
| Memory model | Entity-relation triples | Three-layer sentence graph |
| What you get | The answer | The answer + reasoning + story |
| Cross-session patterns | Manual graph queries | Auto-discovered (Insight layer) |
| Default backend | Requires external DB | **SQLite, zero config** |
| Fully local / offline | No | **Yes — Ollama, BGE-M3, SentenceTransformers** |
| License | Partial OSS | **Full MIT** |

Mem0 and Zep are great tools. But they compress conversations into triples — you get the *what*, not the *why* or *how it got there*. Vektori preserves conversational flow so agents can reason about change over time, not just current state.

---

## Contributing

Issues, PRs, and ideas welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) (coming soon).

```bash
git clone https://github.com/vektori-ai/vektori
cd vektori
pip install -e ".[dev]"
pytest tests/unit/
```

---

## License

MIT — see [LICENSE](LICENSE).

---

<p align="center">Built by <a href="https://github.com/vektori-ai">Laxman & Manit</a> · Star ⭐ if Vektori helps you ship smarter agents</p>
