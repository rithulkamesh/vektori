"""LanceDB storage backend — embedded columnar vector database."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from typing import Any

from vektori.storage.base import StorageBackend

logger = logging.getLogger(__name__)

_FACTS = "facts"
_SENTENCES = "sentences"
_EPISODES = "episodes"
_SESSIONS = "sessions"


def _tbl(prefix: str, suffix: str) -> str:
    return f"{prefix}_{suffix}"


def _esc(value: str) -> str:
    """Escape a string value for use in LanceDB SQL filter expressions."""
    return value.replace("'", "''")


class LanceDBBackend(StorageBackend):
    """
    LanceDB storage backend — embedded, serverless, Apache Arrow-based.

    Local (default):
        LanceDBBackend(uri="/path/to/lancedb")

    Cloud (S3 / GCS / Azure):
        LanceDBBackend(uri="s3://bucket/path")

    Install:
        pip install 'vektori[lancedb]'
    """

    def __init__(
        self,
        uri: str = ".lancedb",
        prefix: str = "vektori",
        embedding_dim: int = 1024,
    ) -> None:
        self.uri = uri
        self.prefix = prefix
        self.embedding_dim = embedding_dim
        self._db: Any = None
        self._facts_tbl: Any = None
        self._sentences_tbl: Any = None
        self._episodes_tbl: Any = None
        self._sessions_tbl: Any = None

    # ── Table name helpers ────────────────────────────────────────────────────

    @property
    def _facts_name(self) -> str:
        return _tbl(self.prefix, _FACTS)

    @property
    def _sentences_name(self) -> str:
        return _tbl(self.prefix, _SENTENCES)

    @property
    def _episodes_name(self) -> str:
        return _tbl(self.prefix, _EPISODES)

    @property
    def _sessions_name(self) -> str:
        return _tbl(self.prefix, _SESSIONS)

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        try:
            import lancedb
            import pyarrow as pa
        except ImportError as e:
            raise ImportError("lancedb and pyarrow required: pip install 'vektori[lancedb]'") from e

        self._db = await lancedb.connect_async(self.uri)

        dim = self.embedding_dim
        vec_type = pa.list_(pa.float32(), list_size=dim)

        facts_schema = pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("vector", vec_type),
                pa.field("text", pa.string()),
                pa.field("user_id", pa.string()),
                pa.field("agent_id", pa.string()),
                pa.field("session_id", pa.string()),
                pa.field("subject", pa.string()),
                pa.field("confidence", pa.float64()),
                pa.field("superseded_by", pa.string()),
                pa.field("metadata", pa.string()),
                pa.field("event_time", pa.string()),
                pa.field("mentions", pa.int64()),
                pa.field("is_active", pa.bool_()),
                pa.field("created_at", pa.string()),
                pa.field("source_sentence_ids", pa.string()),
            ]
        )
        sentences_schema = pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("vector", vec_type),
                pa.field("text", pa.string()),
                pa.field("user_id", pa.string()),
                pa.field("agent_id", pa.string()),
                pa.field("session_id", pa.string()),
                pa.field("turn_number", pa.int64()),
                pa.field("sentence_index", pa.int64()),
                pa.field("role", pa.string()),
                pa.field("content_hash", pa.string()),
                pa.field("mentions", pa.int64()),
                pa.field("is_active", pa.bool_()),
                pa.field("created_at", pa.string()),
                pa.field("next_sentence_id", pa.string()),
            ]
        )
        episodes_schema = pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("vector", vec_type),
                pa.field("text", pa.string()),
                pa.field("user_id", pa.string()),
                pa.field("agent_id", pa.string()),
                pa.field("session_id", pa.string()),
                pa.field("is_active", pa.bool_()),
                pa.field("created_at", pa.string()),
                pa.field("fact_ids", pa.string()),
            ]
        )
        sessions_schema = pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), list_size=1)),
                pa.field("user_id", pa.string()),
                pa.field("agent_id", pa.string()),
                pa.field("metadata", pa.string()),
                pa.field("started_at", pa.string()),
                pa.field("ended_at", pa.string()),
            ]
        )

        self._facts_tbl = await self._db.create_table(
            self._facts_name, schema=facts_schema, exist_ok=True
        )
        self._sentences_tbl = await self._db.create_table(
            self._sentences_name, schema=sentences_schema, exist_ok=True
        )
        self._episodes_tbl = await self._db.create_table(
            self._episodes_name, schema=episodes_schema, exist_ok=True
        )
        self._sessions_tbl = await self._db.create_table(
            self._sessions_name, schema=sessions_schema, exist_ok=True
        )
        logger.info("LanceDB backend initialized at %s (prefix=%s)", self.uri, self.prefix)

    async def close(self) -> None:
        self._db = None

    # ── Sentences ──────────────────────────────────────────────────────────────

    async def upsert_sentences(
        self,
        sentences: list[dict[str, Any]],
        embeddings: list[list[float]],
        user_id: str,
        agent_id: str | None = None,
    ) -> int:
        from vektori.ingestion.hasher import generate_content_hash

        if not sentences:
            return 0

        # Fetch existing to handle mentions increment
        ids = [s["id"] for s in sentences]
        id_list = ", ".join(f"'{_esc(i)}'" for i in ids)
        try:
            existing_rows = (
                await self._sentences_tbl.query()
                .where(f"id IN ({id_list})")
                .select(["id", "mentions"])
                .to_list()
            )
        except Exception:
            existing_rows = []
        existing_mentions = {r["id"]: r["mentions"] for r in existing_rows}

        records = []
        for sent, emb in zip(sentences, embeddings):
            sid = sent["id"]
            content_hash = generate_content_hash(
                sent["session_id"],
                f"{sent['turn_number']}_{sent['sentence_index']}",
                sent["text"],
            )
            mentions = existing_mentions.get(sid, 0) + 1 if sid in existing_mentions else 1
            records.append(
                {
                    "id": sid,
                    "vector": [float(x) for x in emb],
                    "text": sent["text"],
                    "user_id": user_id,
                    "agent_id": agent_id or "",
                    "session_id": sent["session_id"],
                    "turn_number": int(sent["turn_number"]),
                    "sentence_index": int(sent["sentence_index"]),
                    "role": sent.get("role", "user"),
                    "content_hash": content_hash,
                    "mentions": mentions,
                    "is_active": True,
                    "created_at": datetime.utcnow().isoformat(),
                    "next_sentence_id": "",
                }
            )

        await (
            self._sentences_tbl.merge_insert("id")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(records)
        )
        return len(records)

    async def search_sentences(
        self,
        embedding: list[float],
        user_id: str,
        agent_id: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        where = f"user_id = '{_esc(user_id)}' AND is_active = true"
        if agent_id is not None:
            where += f" AND agent_id = '{_esc(agent_id)}'"

        rows = (
            await self._sentences_tbl.vector_search(embedding).where(where).limit(limit).to_list()
        )
        return [_row_to_sentence(r) for r in rows]

    async def find_sentences_by_similarity(
        self,
        quotes: list[str],
        session_id: str,
        threshold: float = 0.75,
    ) -> list[str]:
        return []

    async def search_sentences_in_session(
        self,
        embedding: list[float],
        session_id: str,
        limit: int = 3,
        threshold: float = 0.75,
    ) -> list[str]:
        where = f"session_id = '{_esc(session_id)}' AND is_active = true"
        rows = (
            await self._sentences_tbl.vector_search(embedding).where(where).limit(limit).to_list()
        )
        return [r["id"] for r in rows if r.get("_distance", 1.0) <= (1.0 - threshold)]

    async def find_sentence_containing(
        self,
        session_id: str,
        quote: str,
    ) -> dict[str, Any] | None:
        rows = (
            await self._sentences_tbl.query()
            .where(f"session_id = '{_esc(session_id)}'")
            .select(
                [
                    "id",
                    "text",
                    "session_id",
                    "turn_number",
                    "sentence_index",
                    "role",
                    "mentions",
                    "is_active",
                    "created_at",
                ]
            )
            .to_list()
        )
        lower_quote = quote.lower()
        for r in rows:
            if lower_quote in (r.get("text") or "").lower():
                return _row_to_sentence(r)
        return None

    # ── Facts ──────────────────────────────────────────────────────────────────

    async def insert_fact(
        self,
        text: str,
        embedding: list[float],
        user_id: str,
        agent_id: str | None = None,
        session_id: str | None = None,
        subject: str | None = None,
        confidence: float = 1.0,
        superseded_by_target: str | None = None,
        metadata: dict[str, Any] | None = None,
        event_time: datetime | None = None,
    ) -> str:
        fact_id = str(uuid.uuid4())
        await self._facts_tbl.add(
            [
                {
                    "id": fact_id,
                    "vector": [float(x) for x in embedding],
                    "text": text,
                    "user_id": user_id,
                    "agent_id": agent_id or "",
                    "session_id": session_id or "",
                    "subject": subject or "",
                    "confidence": float(confidence),
                    "superseded_by": superseded_by_target or "",
                    "metadata": json.dumps(metadata or {}),
                    "event_time": event_time.isoformat() if event_time else "",
                    "mentions": 1,
                    "is_active": True,
                    "created_at": datetime.utcnow().isoformat(),
                    "source_sentence_ids": "[]",
                }
            ]
        )
        return fact_id

    async def search_facts(
        self,
        embedding: list[float],
        user_id: str,
        agent_id: str | None = None,
        session_id: str | None = None,
        subject: str | None = None,
        limit: int = 10,
        active_only: bool = True,
        before_date: datetime | None = None,
        after_date: datetime | None = None,
    ) -> list[dict[str, Any]]:
        where = f"user_id = '{_esc(user_id)}'"
        if active_only:
            where += " AND is_active = true"
        if agent_id is not None:
            where += f" AND agent_id = '{_esc(agent_id)}'"
        if session_id is not None:
            where += f" AND session_id = '{_esc(session_id)}'"
        if subject is not None:
            where += f" AND subject = '{_esc(subject)}'"
        if before_date is not None:
            where += f" AND event_time <= '{before_date.isoformat()}'"
        if after_date is not None:
            where += f" AND event_time >= '{after_date.isoformat()}'"

        rows = await self._facts_tbl.vector_search(embedding).where(where).limit(limit).to_list()
        return [_row_to_fact(r) for r in rows]

    async def get_active_facts(
        self,
        user_id: str,
        agent_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        where = f"user_id = '{_esc(user_id)}' AND is_active = true"
        if agent_id is not None:
            where += f" AND agent_id = '{_esc(agent_id)}'"

        rows = await self._facts_tbl.query().where(where).limit(limit + offset).to_list()
        return [_row_to_fact(r) for r in rows[offset:]]

    async def deactivate_fact(self, fact_id: str, superseded_by: str | None = None) -> None:
        updates: dict[str, Any] = {"is_active": False}
        if superseded_by is not None:
            updates["superseded_by"] = superseded_by
        await self._facts_tbl.update(
            where=f"id = '{_esc(fact_id)}'",
            values=updates,
        )

    async def increment_fact_mentions(self, fact_id: str) -> None:
        rows = (
            await self._facts_tbl.query()
            .where(f"id = '{_esc(fact_id)}'")
            .select(["mentions"])
            .to_list()
        )
        current = rows[0]["mentions"] if rows else 1
        await self._facts_tbl.update(
            where=f"id = '{_esc(fact_id)}'",
            values={"mentions": current + 1},
        )

    async def find_fact_by_text(
        self,
        user_id: str,
        text: str,
        agent_id: str | None = None,
    ) -> dict[str, Any] | None:
        where = f"user_id = '{_esc(user_id)}' AND is_active = true"
        if agent_id is not None:
            where += f" AND agent_id = '{_esc(agent_id)}'"

        rows = await self._facts_tbl.query().where(where).to_list()
        lower_text = text.lower()
        for r in rows:
            if lower_text in (r.get("text") or "").lower():
                return _row_to_fact(r)
        return None

    async def get_supersession_chain(self, fact_id: str) -> list[dict[str, Any]]:
        chain = []
        current_id: str | None = fact_id
        visited: set[str] = set()

        while current_id and current_id not in visited and len(chain) < 50:
            visited.add(current_id)
            rows = await self._facts_tbl.query().where(f"id = '{_esc(current_id)}'").to_list()
            if not rows:
                break
            fact = _row_to_fact(rows[0])
            chain.append(fact)
            nxt = rows[0].get("superseded_by") or ""
            current_id = nxt if nxt else None

        return chain

    # ── Edges ──────────────────────────────────────────────────────────────────

    async def insert_edges(self, edges: list[dict[str, Any]]) -> int:
        if not edges:
            return 0
        for edge in edges:
            if edge.get("edge_type") != "NEXT":
                continue
            await self._sentences_tbl.update(
                where=f"id = '{_esc(edge['source_id'])}'",
                values={"next_sentence_id": edge["target_id"]},
            )
        return len(edges)

    async def expand_session_context(
        self,
        sentence_ids: list[str],
        window: int = 3,
    ) -> list[dict[str, Any]]:
        if not sentence_ids:
            return []

        id_list = ", ".join(f"'{_esc(i)}'" for i in sentence_ids)
        seeds = (
            await self._sentences_tbl.query()
            .where(f"id IN ({id_list})")
            .select(["id", "session_id", "turn_number", "sentence_index"])
            .to_list()
        )

        seen_ids: set[str] = set()
        all_results: list[dict[str, Any]] = []

        for seed in seeds:
            sess = seed.get("session_id")
            turn = seed.get("turn_number")
            idx = seed.get("sentence_index")
            if not sess or turn is None or idx is None:
                continue

            lo, hi = int(idx) - window, int(idx) + window
            where = (
                f"session_id = '{_esc(sess)}'"
                f" AND turn_number = {int(turn)}"
                f" AND sentence_index >= {lo}"
                f" AND sentence_index <= {hi}"
                f" AND is_active = true"
            )
            rows = await self._sentences_tbl.query().where(where).to_list()
            for r in rows:
                rid = r.get("id", "")
                if rid not in seen_ids:
                    seen_ids.add(rid)
                    all_results.append(_row_to_sentence(r))

        all_results.sort(
            key=lambda r: (
                r.get("session_id", ""),
                r.get("turn_number", 0),
                r.get("sentence_index", 0),
            )
        )
        return all_results

    # ── Join tables ────────────────────────────────────────────────────────────

    async def insert_fact_source(self, fact_id: str, sentence_id: str) -> None:
        await self.insert_fact_sources([(fact_id, sentence_id)])

    async def insert_fact_sources(self, pairs: list[tuple[str, str]]) -> None:
        if not pairs:
            return
        by_fact: dict[str, list[str]] = {}
        for fact_id, sentence_id in pairs:
            by_fact.setdefault(fact_id, []).append(sentence_id)

        for fact_id, new_sids in by_fact.items():
            rows = (
                await self._facts_tbl.query()
                .where(f"id = '{_esc(fact_id)}'")
                .select(["source_sentence_ids"])
                .to_list()
            )
            if not rows:
                continue
            current = json.loads(rows[0].get("source_sentence_ids", "[]") or "[]")
            merged = list(dict.fromkeys(current + new_sids))
            await self._facts_tbl.update(
                where=f"id = '{_esc(fact_id)}'",
                values={"source_sentence_ids": json.dumps(merged)},
            )

    async def get_source_sentences(self, fact_ids: list[str]) -> list[str]:
        if not fact_ids:
            return []
        id_list = ", ".join(f"'{_esc(i)}'" for i in fact_ids)
        rows = (
            await self._facts_tbl.query()
            .where(f"id IN ({id_list})")
            .select(["source_sentence_ids"])
            .to_list()
        )
        seen: set[str] = set()
        result: list[str] = []
        for r in rows:
            for sid in json.loads(r.get("source_sentence_ids", "[]") or "[]"):
                if sid not in seen:
                    seen.add(sid)
                    result.append(sid)
        return result

    async def get_sentences_by_ids(self, sentence_ids: list[str]) -> list[dict[str, Any]]:
        if not sentence_ids:
            return []
        id_list = ", ".join(f"'{_esc(i)}'" for i in sentence_ids)
        rows = await self._sentences_tbl.query().where(f"id IN ({id_list})").to_list()
        results = [_row_to_sentence(r) for r in rows if r.get("is_active", True)]
        results.sort(
            key=lambda r: (
                r.get("session_id", ""),
                r.get("turn_number", 0),
                r.get("sentence_index", 0),
            )
        )
        return results

    # ── Episodes ──────────────────────────────────────────────────────────────

    async def insert_episode(
        self,
        text: str,
        embedding: list[float],
        user_id: str,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> str:
        episode_id = str(uuid.uuid5(uuid.NAMESPACE_OID, f"{user_id}::{text}"))
        existing = (
            await self._episodes_tbl.query()
            .where(f"id = '{_esc(episode_id)}'")
            .select(["id"])
            .to_list()
        )
        if existing:
            return episode_id  # idempotent

        await self._episodes_tbl.add(
            [
                {
                    "id": episode_id,
                    "vector": [float(x) for x in embedding],
                    "text": text,
                    "user_id": user_id,
                    "agent_id": agent_id or "",
                    "session_id": session_id or "",
                    "is_active": True,
                    "created_at": datetime.utcnow().isoformat(),
                    "fact_ids": "[]",
                }
            ]
        )
        return episode_id

    async def insert_episode_fact(self, episode_id: str, fact_id: str) -> None:
        rows = (
            await self._episodes_tbl.query()
            .where(f"id = '{_esc(episode_id)}'")
            .select(["fact_ids"])
            .to_list()
        )
        if not rows:
            return
        current = json.loads(rows[0].get("fact_ids", "[]") or "[]")
        if fact_id in current:
            return
        await self._episodes_tbl.update(
            where=f"id = '{_esc(episode_id)}'",
            values={"fact_ids": json.dumps(current + [fact_id])},
        )

    async def get_episodes_for_facts(self, fact_ids: list[str]) -> list[dict[str, Any]]:
        if not fact_ids:
            return []
        # LanceDB can't filter inside JSON arrays — fetch all active and filter in Python.
        rows = (
            await self._episodes_tbl.query()
            .where("is_active = true")
            .select(["id", "text", "session_id", "created_at", "fact_ids"])
            .to_list()
        )
        fact_set = set(fact_ids)
        seen: set[str] = set()
        out: list[dict[str, Any]] = []
        for r in rows:
            linked = json.loads(r.get("fact_ids", "[]") or "[]")
            eid = r.get("id", "")
            if fact_set.intersection(linked) and eid not in seen:
                seen.add(eid)
                out.append(
                    {
                        "id": eid,
                        "text": r.get("text"),
                        "session_id": r.get("session_id") or None,
                        "created_at": r.get("created_at"),
                    }
                )
        return out

    async def search_episodes(
        self,
        embedding: list[float],
        user_id: str,
        agent_id: str | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        where = f"user_id = '{_esc(user_id)}' AND is_active = true"
        if agent_id is not None:
            where += f" AND agent_id = '{_esc(agent_id)}'"

        rows = await self._episodes_tbl.vector_search(embedding).where(where).limit(limit).to_list()
        return [
            {
                "id": r.get("id"),
                "text": r.get("text"),
                "session_id": r.get("session_id") or None,
                "created_at": r.get("created_at"),
                "distance": r.get("_distance", 0.0),
            }
            for r in rows
        ]

    # ── Sessions ───────────────────────────────────────────────────────────────

    async def upsert_session(
        self,
        session_id: str,
        user_id: str,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        started_at: datetime | None = None,
    ) -> None:
        existing = (
            await self._sessions_tbl.query()
            .where(f"id = '{_esc(session_id)}'")
            .select(["started_at"])
            .to_list()
        )
        existing_started = existing[0]["started_at"] if existing else ""

        record = {
            "id": session_id,
            "vector": [0.0],
            "user_id": user_id,
            "agent_id": agent_id or "",
            "metadata": json.dumps(metadata or {}),
            "started_at": (
                started_at.isoformat()
                if started_at
                else existing_started or datetime.utcnow().isoformat()
            ),
            "ended_at": "",
        }
        await (
            self._sessions_tbl.merge_insert("id")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute([record])
        )

    async def get_session(
        self,
        session_id: str,
        user_id: str,
    ) -> dict[str, Any] | None:
        rows = await self._sessions_tbl.query().where(f"id = '{_esc(session_id)}'").to_list()
        if not rows:
            return None
        r = rows[0]
        if r.get("user_id") != user_id:
            return None

        sent_rows = (
            await self._sentences_tbl.query()
            .where(f"session_id = '{_esc(session_id)}' AND is_active = true")
            .to_list()
        )
        sentences = [_row_to_sentence(s) for s in sent_rows]
        sentences.sort(key=lambda s: (s.get("turn_number", 0), s.get("sentence_index", 0)))

        return {
            "id": session_id,
            "user_id": r.get("user_id"),
            "agent_id": r.get("agent_id") or None,
            "started_at": r.get("started_at"),
            "ended_at": r.get("ended_at") or None,
            "metadata": r.get("metadata"),
            "sentences": sentences,
        }

    async def count_sessions(
        self,
        user_id: str,
        agent_id: str | None = None,
    ) -> int:
        where = f"user_id = '{_esc(user_id)}'"
        if agent_id is not None:
            where += f" AND agent_id = '{_esc(agent_id)}'"
        rows = await self._sessions_tbl.query().where(where).select(["id"]).to_list()
        return len(rows)

    # ── GDPR ───────────────────────────────────────────────────────────────────

    async def delete_user(self, user_id: str) -> int:
        total = 0
        for tbl in (
            self._facts_tbl,
            self._sentences_tbl,
            self._episodes_tbl,
            self._sessions_tbl,
        ):
            rows = await tbl.query().where(f"user_id = '{_esc(user_id)}'").select(["id"]).to_list()
            total += len(rows)
            if rows:
                await tbl.delete(f"user_id = '{_esc(user_id)}'")
        logger.info("Deleted %d LanceDB rows for user %s", total, user_id)
        return total


# ── Result conversion helpers ─────────────────────────────────────────────────


def _row_to_sentence(r: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": r.get("id"),
        "text": r.get("text"),
        "session_id": r.get("session_id") or None,
        "turn_number": r.get("turn_number"),
        "sentence_index": r.get("sentence_index"),
        "role": r.get("role"),
        "mentions": r.get("mentions", 1),
        "is_active": bool(r.get("is_active", True)),
        "created_at": r.get("created_at"),
    }
    if "_distance" in r:
        row["distance"] = r["_distance"]
    return row


def _row_to_fact(r: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": r.get("id"),
        "text": r.get("text"),
        "confidence": float(r.get("confidence", 1.0)),
        "mentions": int(r.get("mentions", 1)),
        "session_id": r.get("session_id") or None,
        "subject": r.get("subject") or None,
        "is_active": bool(r.get("is_active", True)),
        "superseded_by": r.get("superseded_by") or None,
        "metadata": r.get("metadata"),
        "event_time": r.get("event_time") or None,
        "created_at": r.get("created_at"),
    }
    if "_distance" in r:
        row["distance"] = r["_distance"]
    return row
