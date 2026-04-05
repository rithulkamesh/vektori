"""
Vektori Quickstart — SQLite, zero config.

No Docker. No API keys (uses Ollama locally).

Prerequisites:
    pip install vektori
    ollama pull nomic-embed-text
    ollama pull llama3
"""

import asyncio
from vektori import Vektori


async def main():
    # SQLite default at ~/.vektori/vektori.db
    v = Vektori(
        embedding_model="ollama:nomic-embed-text",
        extraction_model="ollama:llama3",
    )

    print("Adding memories...")
    result = await v.add(
        messages=[
            {"role": "user", "content": "I only use WhatsApp, please don't email me."},
            {"role": "assistant", "content": "Understood, WhatsApp only."},
            {"role": "user", "content": "My outstanding amount is ₹45,000 and I can pay by Friday."},
        ],
        session_id="call-001",
        user_id="user-123",
    )
    print(f"Stored: {result}")

    await asyncio.sleep(5)  # wait for async fact extraction

    print("\nSearching (L1 — facts + episodes)...")
    results = await v.search(
        query="How does this user prefer to communicate?",
        user_id="user-123",
        depth="l1",
    )

    print("\nFacts:")
    for fact in results.get("facts", []):
        print(f"  [{fact.get('score', 0):.3f}] {fact['text']}")

    print("\nEpisodes:")
    for episode in results.get("episodes", []):
        print(f"  {episode['text']}")

    await v.close()


if __name__ == "__main__":
    asyncio.run(main())
