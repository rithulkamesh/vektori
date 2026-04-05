"""
Vektori Quickstart — PostgreSQL backend.

Prerequisites:
    docker compose up -d
    pip install 'vektori[postgres]'
    export OPENAI_API_KEY=sk-...
"""

import asyncio
import os
from vektori import Vektori

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://vektori:vektori@localhost:5432/vektori")


async def main():
    async with Vektori(
        database_url=DATABASE_URL,
        embedding_model="openai:text-embedding-3-small",
        extraction_model="openai:gpt-4o-mini",
    ) as v:
        print("Adding memories...")
        result = await v.add(
            messages=[
                {"role": "user", "content": "I only use WhatsApp, please don't email me."},
                {"role": "assistant", "content": "Got it, WhatsApp only."},
                {"role": "user", "content": "I can pay ₹45,000 by end of Friday."},
            ],
            session_id="call-001",
            user_id="user-123",
        )
        print(f"Stored: {result}")

        await asyncio.sleep(6)  # wait for async extraction

        # L2: full story — facts + episodes + source sentences + session context
        results = await v.search(
            query="What are the payment details for this user?",
            user_id="user-123",
            depth="l2",
            context_window=3,
        )

        print("\nFacts:")
        for f in results.get("facts", []):
            print(f"  [{f.get('score', 0):.3f}] {f['text']}")

        print("\nEpisodes:")
        for ep in results.get("episodes", []):
            print(f"  {ep['text']}")

        print("\nSource sentences:")
        for s in results.get("sentences", []):
            print(f"  [{s['session_id']}] {s['text']}")


if __name__ == "__main__":
    asyncio.run(main())
