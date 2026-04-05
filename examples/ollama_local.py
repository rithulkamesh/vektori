"""
Fully local Vektori — no API keys, no cloud, no Docker required.

Prerequisites:
    pip install vektori
    ollama pull nomic-embed-text
    ollama pull llama3
"""

import asyncio
from vektori import Vektori


async def main():
    v = Vektori(
        embedding_model="ollama:nomic-embed-text",
        extraction_model="ollama:llama3",
        async_extraction=False,  # block for demo so we see facts immediately
    )

    await v.add(
        messages=[
            {"role": "user", "content": "I live in Mumbai and work as a software engineer at Razorpay."},
            {"role": "assistant", "content": "Got it, noted."},
            {"role": "user", "content": "I prefer morning calls before 10am. Never call after 6pm."},
        ],
        session_id="session-001",
        user_id="local-user",
    )

    results = await v.search(
        query="When is the best time to reach this user?",
        user_id="local-user",
        depth="l1",
    )
    print("Facts:", [f["text"] for f in results.get("facts", [])])
    print("Episodes:", [ep["text"] for ep in results.get("episodes", [])])
    await v.close()


if __name__ == "__main__":
    asyncio.run(main())
