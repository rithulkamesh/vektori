"""Integration tests for ChromaBackend — uses EphemeralClient (no server needed).

Tests are skipped automatically when chromadb is not installed.
"""

from __future__ import annotations

import pytest


def _is_chroma_available() -> bool:
    try:
        import chromadb  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.fixture
async def chroma_backend():
    if not _is_chroma_available():
        pytest.skip("chromadb not installed")

    from vektori.storage.chroma_backend import ChromaBackend

    # EphemeralClient (in-memory, no disk) — path=None, host=None
    backend = ChromaBackend(prefix="test", embedding_dim=4)
    await backend.initialize()
    yield backend
    await backend.delete_user("test-user")
    await backend.delete_user("delete-test-user")
    await backend.close()


# ── Sentences ──────────────────────────────────────────────────────────────────


async def test_upsert_and_search_sentences(chroma_backend):
    emb = [0.1, 0.2, 0.3, 0.4]
    sentences = [
        {
            "id": "cs1",
            "text": "I love hiking in the mountains.",
            "session_id": "csess1",
            "turn_number": 0,
            "sentence_index": 0,
            "role": "user",
        }
    ]
    count = await chroma_backend.upsert_sentences(sentences, [emb], user_id="test-user")
    assert count == 1

    results = await chroma_backend.search_sentences(emb, user_id="test-user", limit=5)
    assert len(results) >= 1
    assert results[0]["text"] == "I love hiking in the mountains."
    assert "distance" in results[0]


async def test_sentence_dedup_increments_mentions(chroma_backend):
    emb = [0.2, 0.3, 0.4, 0.5]
    sent = [
        {
            "id": "cs-dedup",
            "text": "I enjoy running.",
            "session_id": "csess2",
            "turn_number": 1,
            "sentence_index": 0,
            "role": "user",
        }
    ]
    await chroma_backend.upsert_sentences(sent, [emb], user_id="test-user")
    await chroma_backend.upsert_sentences(sent, [emb], user_id="test-user")

    results = await chroma_backend.search_sentences(emb, user_id="test-user", limit=5)
    match = next((r for r in results if r["id"] == "cs-dedup"), None)
    assert match is not None
    assert match["mentions"] == 2


async def test_find_sentence_containing(chroma_backend):
    emb = [0.3, 0.4, 0.5, 0.6]
    sent = [
        {
            "id": "cs-find",
            "text": "The weather is sunny today.",
            "session_id": "csess3",
            "turn_number": 0,
            "sentence_index": 0,
            "role": "user",
        }
    ]
    await chroma_backend.upsert_sentences(sent, [emb], user_id="test-user")

    result = await chroma_backend.find_sentence_containing("csess3", "sunny")
    assert result is not None
    assert result["id"] == "cs-find"

    none_result = await chroma_backend.find_sentence_containing("csess3", "rainy")
    assert none_result is None


# ── Facts ──────────────────────────────────────────────────────────────────────


async def test_insert_and_search_facts(chroma_backend):
    emb = [0.1, 0.2, 0.3, 0.4]
    fact_id = await chroma_backend.insert_fact(
        text="User prefers Python over Java.",
        embedding=emb,
        user_id="test-user",
        subject="language",
    )
    assert fact_id

    results = await chroma_backend.search_facts(emb, user_id="test-user", limit=5)
    assert len(results) >= 1
    match = next((r for r in results if r["id"] == fact_id), None)
    assert match is not None
    assert match["text"] == "User prefers Python over Java."
    assert match["subject"] == "language"
    assert "distance" in match


async def test_deactivate_fact_with_supersession(chroma_backend):
    emb = [0.2, 0.3, 0.4, 0.5]
    old_id = await chroma_backend.insert_fact(
        text="User lives in NYC.", embedding=emb, user_id="test-user"
    )
    new_id = await chroma_backend.insert_fact(
        text="User lives in SF.", embedding=emb, user_id="test-user"
    )
    await chroma_backend.deactivate_fact(old_id, superseded_by=new_id)

    active = await chroma_backend.get_active_facts("test-user")
    ids = [f["id"] for f in active]
    assert old_id not in ids
    assert new_id in ids


async def test_increment_fact_mentions(chroma_backend):
    emb = [0.3, 0.4, 0.5, 0.6]
    fact_id = await chroma_backend.insert_fact(
        text="User likes tea.", embedding=emb, user_id="test-user"
    )
    await chroma_backend.increment_fact_mentions(fact_id)
    await chroma_backend.increment_fact_mentions(fact_id)

    active = await chroma_backend.get_active_facts("test-user")
    match = next((f for f in active if f["id"] == fact_id), None)
    assert match is not None
    assert match["mentions"] == 3


async def test_find_fact_by_text(chroma_backend):
    emb = [0.4, 0.5, 0.6, 0.7]
    await chroma_backend.insert_fact(text="User is vegetarian.", embedding=emb, user_id="test-user")
    found = await chroma_backend.find_fact_by_text("test-user", "vegetarian")
    assert found is not None
    assert "vegetarian" in found["text"]


# ── Episodes ──────────────────────────────────────────────────────────────────


async def test_insert_and_search_episodes(chroma_backend):
    emb = [0.1, 0.2, 0.3, 0.4]
    ep_id = await chroma_backend.insert_episode(
        text="User discussed travel plans.",
        embedding=emb,
        user_id="test-user",
        session_id="csess4",
    )
    assert ep_id

    # Idempotent insert
    ep_id2 = await chroma_backend.insert_episode(
        text="User discussed travel plans.",
        embedding=emb,
        user_id="test-user",
        session_id="csess4",
    )
    assert ep_id == ep_id2

    results = await chroma_backend.search_episodes(emb, user_id="test-user", limit=5)
    assert any(r["id"] == ep_id for r in results)


async def test_episode_fact_link(chroma_backend):
    emb = [0.2, 0.3, 0.4, 0.5]
    ep_id = await chroma_backend.insert_episode(
        text="User likes outdoor activities.",
        embedding=emb,
        user_id="test-user",
    )
    fact_id = await chroma_backend.insert_fact(
        text="User goes hiking weekly.", embedding=emb, user_id="test-user"
    )
    await chroma_backend.insert_episode_fact(ep_id, fact_id)

    episodes = await chroma_backend.get_episodes_for_facts([fact_id])
    assert any(e["id"] == ep_id for e in episodes)


# ── Join tables ────────────────────────────────────────────────────────────────


async def test_fact_source_sentences(chroma_backend):
    emb = [0.1, 0.2, 0.3, 0.4]
    sent = [
        {
            "id": "cs-src1",
            "text": "I go to the gym every morning.",
            "session_id": "csess5",
            "turn_number": 0,
            "sentence_index": 0,
            "role": "user",
        }
    ]
    await chroma_backend.upsert_sentences(sent, [emb], user_id="test-user")
    fact_id = await chroma_backend.insert_fact(
        text="User exercises daily.", embedding=emb, user_id="test-user"
    )
    await chroma_backend.insert_fact_source(fact_id, "cs-src1")

    sources = await chroma_backend.get_source_sentences([fact_id])
    assert "cs-src1" in sources


# ── Sessions ───────────────────────────────────────────────────────────────────


async def test_upsert_and_get_session(chroma_backend):
    await chroma_backend.upsert_session(
        session_id="csess-get",
        user_id="test-user",
        metadata={"source": "test"},
    )
    session = await chroma_backend.get_session("csess-get", "test-user")
    assert session is not None
    assert session["user_id"] == "test-user"
    assert session["sentences"] == []

    # Wrong user returns None
    assert await chroma_backend.get_session("csess-get", "other-user") is None


async def test_count_sessions(chroma_backend):
    before = await chroma_backend.count_sessions("test-user")
    await chroma_backend.upsert_session("csess-cnt1", "test-user")
    await chroma_backend.upsert_session("csess-cnt2", "test-user")
    after = await chroma_backend.count_sessions("test-user")
    assert after == before + 2


# ── GDPR ───────────────────────────────────────────────────────────────────────


async def test_delete_user(chroma_backend):
    emb = [0.1, 0.2, 0.3, 0.4]
    await chroma_backend.upsert_session("csess-del", "delete-test-user")
    await chroma_backend.insert_fact(text="Delete me.", embedding=emb, user_id="delete-test-user")
    deleted = await chroma_backend.delete_user("delete-test-user")
    assert deleted >= 2

    active = await chroma_backend.get_active_facts("delete-test-user")
    assert active == []
