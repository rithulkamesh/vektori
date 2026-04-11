"""Integration tests for LanceDBBackend — uses a temporary directory (no server needed).

Tests are skipped automatically when lancedb/pyarrow is not installed.
"""

from __future__ import annotations

import pytest


def _is_lancedb_available() -> bool:
    try:
        import lancedb  # noqa: F401
        import pyarrow  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.fixture
async def lancedb_backend(tmp_path):
    if not _is_lancedb_available():
        pytest.skip("lancedb or pyarrow not installed")

    from vektori.storage.lancedb_backend import LanceDBBackend

    backend = LanceDBBackend(uri=str(tmp_path / "lancedb"), prefix="test", embedding_dim=4)
    await backend.initialize()
    yield backend
    await backend.close()


# ── Sentences ──────────────────────────────────────────────────────────────────


async def test_upsert_and_search_sentences(lancedb_backend):
    emb = [0.1, 0.2, 0.3, 0.4]
    sentences = [
        {
            "id": "ls1",
            "text": "I love hiking in the mountains.",
            "session_id": "lsess1",
            "turn_number": 0,
            "sentence_index": 0,
            "role": "user",
        }
    ]
    count = await lancedb_backend.upsert_sentences(sentences, [emb], user_id="test-user")
    assert count == 1

    results = await lancedb_backend.search_sentences(emb, user_id="test-user", limit=5)
    assert len(results) >= 1
    assert results[0]["text"] == "I love hiking in the mountains."
    assert "distance" in results[0]


async def test_sentence_dedup_increments_mentions(lancedb_backend):
    emb = [0.2, 0.3, 0.4, 0.5]
    sent = [
        {
            "id": "ls-dedup",
            "text": "I enjoy running.",
            "session_id": "lsess2",
            "turn_number": 1,
            "sentence_index": 0,
            "role": "user",
        }
    ]
    await lancedb_backend.upsert_sentences(sent, [emb], user_id="test-user")
    await lancedb_backend.upsert_sentences(sent, [emb], user_id="test-user")

    results = await lancedb_backend.search_sentences(emb, user_id="test-user", limit=10)
    match = next((r for r in results if r["id"] == "ls-dedup"), None)
    assert match is not None
    assert match["mentions"] == 2


async def test_find_sentence_containing(lancedb_backend):
    emb = [0.3, 0.4, 0.5, 0.6]
    sent = [
        {
            "id": "ls-find",
            "text": "The weather is sunny today.",
            "session_id": "lsess3",
            "turn_number": 0,
            "sentence_index": 0,
            "role": "user",
        }
    ]
    await lancedb_backend.upsert_sentences(sent, [emb], user_id="test-user")

    result = await lancedb_backend.find_sentence_containing("lsess3", "sunny")
    assert result is not None
    assert result["id"] == "ls-find"

    none_result = await lancedb_backend.find_sentence_containing("lsess3", "rainy")
    assert none_result is None


async def test_expand_session_context(lancedb_backend):
    emb = [0.1, 0.2, 0.3, 0.4]
    sentences = [
        {
            "id": f"ls-ctx{i}",
            "text": f"Sentence {i}.",
            "session_id": "lsess-ctx",
            "turn_number": 0,
            "sentence_index": i,
            "role": "user",
        }
        for i in range(5)
    ]
    embs = [emb] * 5
    await lancedb_backend.upsert_sentences(sentences, embs, user_id="test-user")

    # Seed with middle sentence → should expand ±2
    expanded = await lancedb_backend.expand_session_context(["ls-ctx2"], window=2)
    ids = [r["id"] for r in expanded]
    assert "ls-ctx0" in ids
    assert "ls-ctx2" in ids
    assert "ls-ctx4" in ids


# ── Facts ──────────────────────────────────────────────────────────────────────


async def test_insert_and_search_facts(lancedb_backend):
    emb = [0.1, 0.2, 0.3, 0.4]
    fact_id = await lancedb_backend.insert_fact(
        text="User prefers Python over Java.",
        embedding=emb,
        user_id="test-user",
        subject="language",
    )
    assert fact_id

    results = await lancedb_backend.search_facts(emb, user_id="test-user", limit=5)
    assert len(results) >= 1
    match = next((r for r in results if r["id"] == fact_id), None)
    assert match is not None
    assert match["text"] == "User prefers Python over Java."
    assert match["subject"] == "language"
    assert "distance" in match


async def test_deactivate_fact_with_supersession(lancedb_backend):
    emb = [0.2, 0.3, 0.4, 0.5]
    old_id = await lancedb_backend.insert_fact(
        text="User lives in NYC.", embedding=emb, user_id="test-user"
    )
    new_id = await lancedb_backend.insert_fact(
        text="User lives in SF.", embedding=emb, user_id="test-user"
    )
    await lancedb_backend.deactivate_fact(old_id, superseded_by=new_id)

    active = await lancedb_backend.get_active_facts("test-user")
    ids = [f["id"] for f in active]
    assert old_id not in ids
    assert new_id in ids


async def test_increment_fact_mentions(lancedb_backend):
    emb = [0.3, 0.4, 0.5, 0.6]
    fact_id = await lancedb_backend.insert_fact(
        text="User likes tea.", embedding=emb, user_id="test-user"
    )
    await lancedb_backend.increment_fact_mentions(fact_id)
    await lancedb_backend.increment_fact_mentions(fact_id)

    active = await lancedb_backend.get_active_facts("test-user")
    match = next((f for f in active if f["id"] == fact_id), None)
    assert match is not None
    assert match["mentions"] == 3


async def test_find_fact_by_text(lancedb_backend):
    emb = [0.4, 0.5, 0.6, 0.7]
    await lancedb_backend.insert_fact(
        text="User is vegetarian.", embedding=emb, user_id="test-user"
    )
    found = await lancedb_backend.find_fact_by_text("test-user", "vegetarian")
    assert found is not None
    assert "vegetarian" in found["text"]


async def test_supersession_chain(lancedb_backend):
    emb = [0.1, 0.2, 0.3, 0.4]
    old_id = await lancedb_backend.insert_fact(
        text="User is 25.", embedding=emb, user_id="test-user"
    )
    new_id = await lancedb_backend.insert_fact(
        text="User is 26.", embedding=emb, user_id="test-user"
    )
    await lancedb_backend.deactivate_fact(old_id, superseded_by=new_id)

    chain = await lancedb_backend.get_supersession_chain(old_id)
    assert chain[0]["id"] == old_id
    assert chain[1]["id"] == new_id


# ── Episodes ──────────────────────────────────────────────────────────────────


async def test_insert_and_search_episodes(lancedb_backend):
    emb = [0.1, 0.2, 0.3, 0.4]
    ep_id = await lancedb_backend.insert_episode(
        text="User discussed travel plans.",
        embedding=emb,
        user_id="test-user",
        session_id="lsess4",
    )
    assert ep_id

    # Idempotent insert
    ep_id2 = await lancedb_backend.insert_episode(
        text="User discussed travel plans.",
        embedding=emb,
        user_id="test-user",
        session_id="lsess4",
    )
    assert ep_id == ep_id2

    results = await lancedb_backend.search_episodes(emb, user_id="test-user", limit=5)
    assert any(r["id"] == ep_id for r in results)


async def test_episode_fact_link(lancedb_backend):
    emb = [0.2, 0.3, 0.4, 0.5]
    ep_id = await lancedb_backend.insert_episode(
        text="User likes outdoor activities.",
        embedding=emb,
        user_id="test-user",
    )
    fact_id = await lancedb_backend.insert_fact(
        text="User goes hiking weekly.", embedding=emb, user_id="test-user"
    )
    await lancedb_backend.insert_episode_fact(ep_id, fact_id)

    episodes = await lancedb_backend.get_episodes_for_facts([fact_id])
    assert any(e["id"] == ep_id for e in episodes)


# ── Join tables ────────────────────────────────────────────────────────────────


async def test_fact_source_sentences(lancedb_backend):
    emb = [0.1, 0.2, 0.3, 0.4]
    sent = [
        {
            "id": "ls-src1",
            "text": "I go to the gym every morning.",
            "session_id": "lsess5",
            "turn_number": 0,
            "sentence_index": 0,
            "role": "user",
        }
    ]
    await lancedb_backend.upsert_sentences(sent, [emb], user_id="test-user")
    fact_id = await lancedb_backend.insert_fact(
        text="User exercises daily.", embedding=emb, user_id="test-user"
    )
    await lancedb_backend.insert_fact_source(fact_id, "ls-src1")

    sources = await lancedb_backend.get_source_sentences([fact_id])
    assert "ls-src1" in sources

    fetched = await lancedb_backend.get_sentences_by_ids(["ls-src1"])
    assert len(fetched) == 1
    assert fetched[0]["text"] == "I go to the gym every morning."


# ── Sessions ───────────────────────────────────────────────────────────────────


async def test_upsert_and_get_session(lancedb_backend):
    await lancedb_backend.upsert_session(
        session_id="lsess-get",
        user_id="test-user",
        metadata={"source": "test"},
    )
    session = await lancedb_backend.get_session("lsess-get", "test-user")
    assert session is not None
    assert session["user_id"] == "test-user"
    assert session["sentences"] == []

    assert await lancedb_backend.get_session("lsess-get", "other-user") is None


async def test_count_sessions(lancedb_backend):
    before = await lancedb_backend.count_sessions("test-user")
    await lancedb_backend.upsert_session("lsess-cnt1", "test-user")
    await lancedb_backend.upsert_session("lsess-cnt2", "test-user")
    after = await lancedb_backend.count_sessions("test-user")
    assert after == before + 2


# ── GDPR ───────────────────────────────────────────────────────────────────────


async def test_delete_user(lancedb_backend):
    emb = [0.1, 0.2, 0.3, 0.4]
    await lancedb_backend.upsert_session("lsess-del", "delete-test-user")
    await lancedb_backend.insert_fact(
        text="Delete me.", embedding=emb, user_id="delete-test-user"
    )
    deleted = await lancedb_backend.delete_user("delete-test-user")
    assert deleted >= 2

    active = await lancedb_backend.get_active_facts("delete-test-user")
    assert active == []
