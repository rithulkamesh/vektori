"""Integration tests for the retrieval pipeline (memory backend)."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from vektori.storage.memory import MemoryBackend
from vektori.retrieval.search import SearchPipeline


@pytest.fixture
async def search_with_facts():
    """SearchPipeline with pre-populated facts in memory backend."""
    db = MemoryBackend()
    await db.initialize()

    embedder = MagicMock()
    # Query embedding slightly different from stored to test cosine similarity
    embedder.embed = AsyncMock(return_value=[1.0] + [0.0] * 1535)

    # Insert a fact
    fact_id = await db.insert_fact(
        text="User prefers WhatsApp over email",
        embedding=[1.0] + [0.0] * 1535,
        user_id="u1",
        confidence=0.95,
    )

    # Insert an insight linked to that fact
    insight_id = await db.insert_insight(
        text="Offering email after WhatsApp preference leads to disconnection",
        embedding=[0.9] + [0.0] * 1535,
        user_id="u1",
        confidence=0.80,
    )
    await db.insert_insight_fact(insight_id, fact_id)

    pipeline = SearchPipeline(db=db, embedder=embedder)
    return pipeline, db


async def test_l0_returns_facts_only(search_with_facts):
    pipeline, db = search_with_facts
    results = await pipeline.search("communication preference", "u1", depth="l0")
    assert "facts" in results
    assert "insights" not in results
    assert len(results["facts"]) >= 1


async def test_l1_returns_facts_and_insights(search_with_facts):
    pipeline, db = search_with_facts
    results = await pipeline.search("communication preference", "u1", depth="l1")
    assert "facts" in results
    assert "insights" in results
    assert len(results["insights"]) >= 1


async def test_l2_returns_all_layers(search_with_facts):
    pipeline, db = search_with_facts
    results = await pipeline.search("communication preference", "u1", depth="l2")
    assert "facts" in results
    assert "insights" in results
    assert "sentences" in results


async def test_empty_result_for_unknown_user(search_with_facts):
    pipeline, db = search_with_facts
    results = await pipeline.search("anything", "unknown-user", depth="l1")
    assert results["facts"] == []
