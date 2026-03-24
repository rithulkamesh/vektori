"""Schema migration runner."""

import asyncio
import os
from pathlib import Path

SCHEMA_PATH = Path(__file__).parent / "init.sql"


async def migrate(database_url: str) -> None:
    import asyncpg
    conn = await asyncpg.connect(database_url)
    try:
        schema = SCHEMA_PATH.read_text()
        await conn.execute(schema)
        print("Migration complete.")
    finally:
        await conn.close()


if __name__ == "__main__":
    url = os.getenv("DATABASE_URL", "postgresql://vektori:vektori@localhost:5432/vektori")
    asyncio.run(migrate(url))
