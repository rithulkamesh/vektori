"""ChromaDB storage backend — embedded or HTTP-server vector database."""

from __future__ import annotations

import asyncio
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


def _col(prefix: str, suffix: str) -> str:
    return f"{prefix}_{suffix}"


def _bool_to_int(v: bool) -> int:
    return 1 if v else 0


def _int_to_bool(v: Any) -> bool:
    return bool(v)


def _build_where(*conditions: dict[str, Any]) -> dict[str, Any] | None:
    """Build a Chroma WHERE clause from one or more condition dicts."""
    active = [c for c in conditions if c]
    if not active:
        return None
    if len(active) == 1:
        return active[0]
    return {"$and": active}


class ChromaBackend(StorageBackend):
    """
    ChromaDB storage backend.

    Modes:
        Embedded (persistent, default):
            ChromaBackend(path="/path/to/chroma_db")
        Embedded (ephemeral / in-memory, useful for tests):
            ChromaBackend()   # path=None, host=None
        HTTP server:
            docker run -p 8000:8000 chromadb/chroma
            ChromaBackend(host="localhost", port=8000)

    Install:
        pip install 'vektori[chroma]'
    """

    def __init__(
        self,
        path: str | None = None,
        host: str | None = None,
        port: int = 8000,
        prefix: str = "vektori",
        embedding_dim: int = 1024,
    ) -> None:
        self.path = path
        self.host = host
        self.port = port
        self.prefix = prefix
        self.embedding_dim = embedding_dim
        self._client: Any = None
        self._facts_c: Any = None
        self._sentences_c: Any = None
        self._episodes_c: Any = None
        self._sessions_c: Any = None

    # ── Collection name helpers ────────────────────────────────────────────────

    @property
    def _facts_col(self) -> str:
        return _col(self.prefix, _FACTS)

    @property
    def _sentences_col(self) -> str:
        return _col(self.prefix, _SENTENCES)

    @property
    def _episodes_col(self) -> str:
        return _col(self.prefix, _EPISODES)

    @property
    def _sessions_col(self) -> str:
        return _col(self.prefix, _SESSIONS)

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _run(self, fn, *args, **kwargs) -> Any:
        """Execute a synchronous chromadb call off the event-loop thread."""
        return await asyncio.to_thread(fn, *args, **kwargs)

    def _safe_query(
        self,
        collection: Any,
        query_embeddings: list[list[float]],
        n_results: int,
        where: dict[str, Any] | None,
        include: list[str],
    ) -> dict[str, Any]:
        """Wrap collection.query, capping n_results to collection size."""
        count = collection.count()
        actual = min(n_results, count) if count > 0 else 0
        if actual == 0:
            return {"ids": [[]], "distances": [[]], "documents": [[]], "metadatas": [[]]}
        kwargs: dict[str, Any] = {
            "query_embeddings": query_embeddings,
            "n_results": actual,
            "include": include,
        }
        if where:
            kwargs["where"] = where
        return collection.query(**kwargs)

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        try:
            import chromadb
        except ImportError as e:
            raise ImportError("chromadb required: pip install 'vektori[chroma]'") from e

        cosine_meta = {"hnsw:space": "cosine"}

        if self.host:
            self._client = await self._run(chromadb.HttpClient, host=self.host, port=self.port)
        elif self.path:
            self._client = await self._run(chromadb.PersistentClient, path=self.path)
        else:
            self._client = await self._run(chromadb.EphemeralClient)

        self._facts_c = await self._run(
            self._client.get_or_create_collection,
            self._facts_col,
            metadata=cosine_meta,
        )
        self._sentences_c = await self._run(
            self._client.get_or_create_collection,
            self._sentences_col,
            metadata=cosine_meta,
        )
        self._episodes_c = await self._run(
            self._client.get_or_create_collection,
            self._episodes_col,
            metadata=cosine_meta,
        )
        self._sessions_c = await self._run(
            self._client.get_or_create_collection,
            self._sessions_col,
            metadata=cosine_meta,
        )
        logger.info("ChromaDB backend initialized (prefix=%s)", self.prefix)

    async def close(self) -> None:
        self._client = None

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

        ids = [s["id"] for s in sentences]

        # Fetch existing to handle mentions increment
        existing = await self._run(
            self._sentences_c.get,
            ids=ids,
            include=["metadatas"],
        )
        existing_mentions: dict[str, int] = {}
        for eid, emeta in zip(existing["ids"], existing["metadatas"] or []):
            existing_mentions[eid] = (emeta or {}).get("mentions", 1)

        docs, metas = [], []
        for sent in sentences:
            sid = sent["id"]
            content_hash = generate_content_hash(
                sent["session_id"],
                f"{sent['turn_number']}_{sent['sentence_index']}",
                sent["text"],
            )
            mentions = existing_mentions.get(sid, 0) + 1 if sid in existing_mentions else 1
            metas.append(
                {
                    "user_id": user_id,
                    "agent_id": agent_id or "",
                    "session_id": sent["session_id"],
                    "turn_number": int(sent["turn_number"]),
                    "sentence_index": int(sent["sentence_index"]),
                    "role": sent.get("role", "user"),
                    "content_hash": content_hash,
                    "mentions": mentions,
                    "is_active": 1,
                    "created_at": datetime.utcnow().isoformat(),
                    "next_sentence_id": "",
                }
            )
            docs.append(sent["text"])

        await self._run(
            self._sentences_c.upsert,
            ids=ids,
            embeddings=embeddings,
            documents=docs,
            metadatas=metas,
        )
        return len(ids)

    async def search_sentences(
        self,
        embedding: list[float],
        user_id: str,
        agent_id: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        where_parts = [{"user_id": {"$eq": user_id}}, {"is_active": {"$eq": 1}}]
        if agent_id is not None:
            where_parts.append({"agent_id": {"$eq": agent_id}})

        results = await self._run(
            self._safe_query,
            self._sentences_c,
            query_embeddings=[embedding],
            n_results=limit,
            where=_build_where(*where_parts),
            include=["documents", "metadatas", "distances"],
        )
        return [
            _row_to_sentence(sid, doc, meta, dist)
            for sid, doc, meta, dist in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

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
        where = {"$and": [{"session_id": {"$eq": session_id}}, {"is_active": {"$eq": 1}}]}
        results = await self._run(
            self._safe_query,
            self._sentences_c,
            query_embeddings=[embedding],
            n_results=limit,
            where=where,
            include=["distances"],
        )
        return [
            sid
            for sid, dist in zip(results["ids"][0], results["distances"][0])
            if dist <= (1.0 - threshold)
        ]

    async def find_sentence_containing(
        self,
        session_id: str,
        quote: str,
    ) -> dict[str, Any] | None:
        where = {"session_id": {"$eq": session_id}}
        results = await self._run(
            self._sentences_c.get,
            where=where,
            include=["documents", "metadatas"],
        )
        lower_quote = quote.lower()
        for sid, doc, meta in zip(
            results["ids"],
            results["documents"] or [],
            results["metadatas"] or [],
        ):
            if lower_quote in (doc or "").lower():
                return _row_to_sentence(sid, doc, meta)
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
        meta = {
            "user_id": user_id,
            "agent_id": agent_id or "",
            "session_id": session_id or "",
            "subject": subject or "",
            "confidence": float(confidence),
            "superseded_by": superseded_by_target or "",
            "metadata": json.dumps(metadata or {}),
            "event_time": event_time.isoformat() if event_time else "",
            "mentions": 1,
            "is_active": 1,
            "created_at": datetime.utcnow().isoformat(),
            "source_sentence_ids": "[]",
        }
        await self._run(
            self._facts_c.add,
            ids=[fact_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[meta],
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
        where_parts: list[dict[str, Any]] = [{"user_id": {"$eq": user_id}}]
        if active_only:
            where_parts.append({"is_active": {"$eq": 1}})
        if agent_id is not None:
            where_parts.append({"agent_id": {"$eq": agent_id}})
        if session_id is not None:
            where_parts.append({"session_id": {"$eq": session_id}})
        if subject is not None:
            where_parts.append({"subject": {"$eq": subject}})

        results = await self._run(
            self._safe_query,
            self._facts_c,
            query_embeddings=[embedding],
            n_results=limit,
            where=_build_where(*where_parts),
            include=["documents", "metadatas", "distances"],
        )

        facts = [
            _row_to_fact(fid, doc, meta, dist)
            for fid, doc, meta, dist in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

        # Post-filter by date (Chroma lacks datetime range filtering)
        if before_date is not None:
            facts = [
                f
                for f in facts
                if f.get("event_time") and f["event_time"] <= before_date.isoformat()
            ]
        if after_date is not None:
            facts = [
                f
                for f in facts
                if f.get("event_time") and f["event_time"] >= after_date.isoformat()
            ]

        return facts

    async def get_active_facts(
        self,
        user_id: str,
        agent_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        where_parts: list[dict[str, Any]] = [
            {"user_id": {"$eq": user_id}},
            {"is_active": {"$eq": 1}},
        ]
        if agent_id is not None:
            where_parts.append({"agent_id": {"$eq": agent_id}})

        results = await self._run(
            self._facts_c.get,
            where=_build_where(*where_parts),
            include=["documents", "metadatas"],
            limit=limit,
            offset=offset,
        )
        return [
            _row_to_fact(fid, doc, meta)
            for fid, doc, meta in zip(
                results["ids"],
                results["documents"] or [],
                results["metadatas"] or [],
            )
        ]

    async def deactivate_fact(self, fact_id: str, superseded_by: str | None = None) -> None:
        existing = await self._run(
            self._facts_c.get,
            ids=[fact_id],
            include=["documents", "metadatas", "embeddings"],
        )
        if not existing["ids"]:
            return
        meta = dict(existing["metadatas"][0] or {})
        meta["is_active"] = 0
        if superseded_by is not None:
            meta["superseded_by"] = superseded_by
        await self._run(
            self._facts_c.update,
            ids=[fact_id],
            metadatas=[meta],
        )

    async def increment_fact_mentions(self, fact_id: str) -> None:
        existing = await self._run(
            self._facts_c.get,
            ids=[fact_id],
            include=["metadatas"],
        )
        if not existing["ids"]:
            return
        meta = dict(existing["metadatas"][0] or {})
        meta["mentions"] = int(meta.get("mentions", 1)) + 1
        await self._run(
            self._facts_c.update,
            ids=[fact_id],
            metadatas=[meta],
        )

    async def find_fact_by_text(
        self,
        user_id: str,
        text: str,
        agent_id: str | None = None,
    ) -> dict[str, Any] | None:
        where_parts: list[dict[str, Any]] = [
            {"user_id": {"$eq": user_id}},
            {"is_active": {"$eq": 1}},
        ]
        if agent_id is not None:
            where_parts.append({"agent_id": {"$eq": agent_id}})

        results = await self._run(
            self._facts_c.get,
            where=_build_where(*where_parts),
            where_document={"$contains": text},
            include=["documents", "metadatas"],
        )
        if not results["ids"]:
            return None
        return _row_to_fact(results["ids"][0], results["documents"][0], results["metadatas"][0])

    async def get_supersession_chain(self, fact_id: str) -> list[dict[str, Any]]:
        chain = []
        current_id: str | None = fact_id
        visited: set[str] = set()

        while current_id and current_id not in visited and len(chain) < 50:
            visited.add(current_id)
            existing = await self._run(
                self._facts_c.get,
                ids=[current_id],
                include=["documents", "metadatas"],
            )
            if not existing["ids"]:
                break
            fact = _row_to_fact(
                existing["ids"][0],
                existing["documents"][0],
                existing["metadatas"][0],
            )
            chain.append(fact)
            nxt = fact.get("superseded_by")
            current_id = nxt if nxt else None

        return chain

    # ── Edges ──────────────────────────────────────────────────────────────────

    async def insert_edges(self, edges: list[dict[str, Any]]) -> int:
        if not edges:
            return 0
        for edge in edges:
            if edge.get("edge_type") != "NEXT":
                continue
            src = edge["source_id"]
            existing = await self._run(
                self._sentences_c.get,
                ids=[src],
                include=["metadatas"],
            )
            if not existing["ids"]:
                continue
            meta = dict(existing["metadatas"][0] or {})
            meta["next_sentence_id"] = edge["target_id"]
            await self._run(
                self._sentences_c.update,
                ids=[src],
                metadatas=[meta],
            )
        return len(edges)

    async def expand_session_context(
        self,
        sentence_ids: list[str],
        window: int = 3,
    ) -> list[dict[str, Any]]:
        if not sentence_ids:
            return []

        seeds = await self._run(
            self._sentences_c.get,
            ids=sentence_ids,
            include=["metadatas"],
        )

        seen_ids: set[str] = set()
        all_results: list[dict[str, Any]] = []

        for sid, meta in zip(seeds["ids"], seeds["metadatas"] or []):
            if not meta:
                continue
            sess = meta.get("session_id")
            turn = meta.get("turn_number")
            idx = meta.get("sentence_index")
            if sess is None or turn is None or idx is None:
                continue

            where = {
                "$and": [
                    {"session_id": {"$eq": sess}},
                    {"turn_number": {"$eq": int(turn)}},
                    {"sentence_index": {"$gte": int(idx) - window}},
                    {"sentence_index": {"$lte": int(idx) + window}},
                    {"is_active": {"$eq": 1}},
                ]
            }
            results = await self._run(
                self._sentences_c.get,
                where=where,
                include=["documents", "metadatas"],
            )
            for rsid, rdoc, rmeta in zip(
                results["ids"],
                results["documents"] or [],
                results["metadatas"] or [],
            ):
                if rsid not in seen_ids:
                    seen_ids.add(rsid)
                    all_results.append(_row_to_sentence(rsid, rdoc, rmeta))

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
            existing = await self._run(
                self._facts_c.get,
                ids=[fact_id],
                include=["metadatas"],
            )
            if not existing["ids"]:
                continue
            meta = dict(existing["metadatas"][0] or {})
            current = json.loads(meta.get("source_sentence_ids", "[]"))
            merged = list(dict.fromkeys(current + new_sids))
            meta["source_sentence_ids"] = json.dumps(merged)
            await self._run(
                self._facts_c.update,
                ids=[fact_id],
                metadatas=[meta],
            )

    async def get_source_sentences(self, fact_ids: list[str]) -> list[str]:
        if not fact_ids:
            return []
        existing = await self._run(
            self._facts_c.get,
            ids=fact_ids,
            include=["metadatas"],
        )
        seen: set[str] = set()
        result: list[str] = []
        for meta in existing["metadatas"] or []:
            for sid in json.loads((meta or {}).get("source_sentence_ids", "[]")):
                if sid not in seen:
                    seen.add(sid)
                    result.append(sid)
        return result

    async def get_sentences_by_ids(self, sentence_ids: list[str]) -> list[dict[str, Any]]:
        if not sentence_ids:
            return []
        existing = await self._run(
            self._sentences_c.get,
            ids=sentence_ids,
            include=["documents", "metadatas"],
        )
        results = [
            _row_to_sentence(sid, doc, meta)
            for sid, doc, meta in zip(
                existing["ids"],
                existing["documents"] or [],
                existing["metadatas"] or [],
            )
            if _int_to_bool((meta or {}).get("is_active", 1))
        ]
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
        existing = await self._run(
            self._episodes_c.get,
            ids=[episode_id],
            include=["metadatas"],
        )
        if existing["ids"]:
            return episode_id  # idempotent

        await self._run(
            self._episodes_c.add,
            ids=[episode_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[
                {
                    "user_id": user_id,
                    "agent_id": agent_id or "",
                    "session_id": session_id or "",
                    "is_active": 1,
                    "created_at": datetime.utcnow().isoformat(),
                    "fact_ids": "[]",
                }
            ],
        )
        return episode_id

    async def insert_episode_fact(self, episode_id: str, fact_id: str) -> None:
        existing = await self._run(
            self._episodes_c.get,
            ids=[episode_id],
            include=["metadatas"],
        )
        if not existing["ids"]:
            return
        meta = dict(existing["metadatas"][0] or {})
        current = json.loads(meta.get("fact_ids", "[]"))
        if fact_id in current:
            return
        meta["fact_ids"] = json.dumps(current + [fact_id])
        await self._run(
            self._episodes_c.update,
            ids=[episode_id],
            metadatas=[meta],
        )

    async def get_episodes_for_facts(self, fact_ids: list[str]) -> list[dict[str, Any]]:
        if not fact_ids:
            return []
        # Chroma can't filter inside JSON arrays — fetch all active and filter in Python.
        results = await self._run(
            self._episodes_c.get,
            where={"is_active": {"$eq": 1}},
            include=["documents", "metadatas"],
        )
        fact_set = set(fact_ids)
        seen: set[str] = set()
        out: list[dict[str, Any]] = []
        for eid, doc, meta in zip(
            results["ids"],
            results["documents"] or [],
            results["metadatas"] or [],
        ):
            linked = json.loads((meta or {}).get("fact_ids", "[]"))
            if fact_set.intersection(linked) and eid not in seen:
                seen.add(eid)
                out.append(
                    {
                        "id": eid,
                        "text": doc,
                        "session_id": (meta or {}).get("session_id") or None,
                        "created_at": (meta or {}).get("created_at"),
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
        where_parts: list[dict[str, Any]] = [
            {"user_id": {"$eq": user_id}},
            {"is_active": {"$eq": 1}},
        ]
        if agent_id is not None:
            where_parts.append({"agent_id": {"$eq": agent_id}})

        results = await self._run(
            self._safe_query,
            self._episodes_c,
            query_embeddings=[embedding],
            n_results=limit,
            where=_build_where(*where_parts),
            include=["documents", "metadatas", "distances"],
        )
        return [
            {
                "id": eid,
                "text": doc,
                "session_id": (meta or {}).get("session_id") or None,
                "created_at": (meta or {}).get("created_at"),
                "distance": dist,
            }
            for eid, doc, meta, dist in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
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
        dummy_vec = [0.0] * self.embedding_dim
        existing = await self._run(
            self._sessions_c.get,
            ids=[session_id],
            include=["metadatas"],
        )
        if existing["ids"]:
            existing_started = (existing["metadatas"][0] or {}).get("started_at", "")
        else:
            existing_started = ""

        meta = {
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
        await self._run(
            self._sessions_c.upsert,
            ids=[session_id],
            embeddings=[dummy_vec],
            documents=[session_id],
            metadatas=[meta],
        )

    async def get_session(
        self,
        session_id: str,
        user_id: str,
    ) -> dict[str, Any] | None:
        existing = await self._run(
            self._sessions_c.get,
            ids=[session_id],
            include=["metadatas"],
        )
        if not existing["ids"]:
            return None
        meta = existing["metadatas"][0] or {}
        if meta.get("user_id") != user_id:
            return None

        sent_results = await self._run(
            self._sentences_c.get,
            where={"$and": [{"session_id": {"$eq": session_id}}, {"is_active": {"$eq": 1}}]},
            include=["documents", "metadatas"],
        )
        sentences = [
            _row_to_sentence(sid, doc, smeta)
            for sid, doc, smeta in zip(
                sent_results["ids"],
                sent_results["documents"] or [],
                sent_results["metadatas"] or [],
            )
        ]
        sentences.sort(key=lambda r: (r.get("turn_number", 0), r.get("sentence_index", 0)))

        return {
            "id": session_id,
            "user_id": meta.get("user_id"),
            "agent_id": meta.get("agent_id") or None,
            "started_at": meta.get("started_at"),
            "ended_at": meta.get("ended_at") or None,
            "metadata": meta.get("metadata"),
            "sentences": sentences,
        }

    async def count_sessions(
        self,
        user_id: str,
        agent_id: str | None = None,
    ) -> int:
        where_parts: list[dict[str, Any]] = [{"user_id": {"$eq": user_id}}]
        if agent_id is not None:
            where_parts.append({"agent_id": {"$eq": agent_id}})

        results = await self._run(
            self._sessions_c.get,
            where=_build_where(*where_parts),
            include=[],
        )
        return len(results["ids"])

    # ── GDPR ───────────────────────────────────────────────────────────────────

    async def delete_user(self, user_id: str) -> int:
        total = 0
        for col in (self._facts_c, self._sentences_c, self._episodes_c, self._sessions_c):
            existing = await self._run(
                col.get,
                where={"user_id": {"$eq": user_id}},
                include=[],
            )
            total += len(existing["ids"])
            if existing["ids"]:
                await self._run(col.delete, ids=existing["ids"])
        logger.info("Deleted %d Chroma records for user %s", total, user_id)
        return total


# ── Result conversion helpers ─────────────────────────────────────────────────


def _row_to_sentence(
    sid: str,
    doc: str | None,
    meta: dict[str, Any] | None,
    distance: float | None = None,
) -> dict[str, Any]:
    m = meta or {}
    row: dict[str, Any] = {
        "id": sid,
        "text": doc,
        "session_id": m.get("session_id") or None,
        "turn_number": m.get("turn_number"),
        "sentence_index": m.get("sentence_index"),
        "role": m.get("role"),
        "mentions": m.get("mentions", 1),
        "is_active": _int_to_bool(m.get("is_active", 1)),
        "created_at": m.get("created_at"),
    }
    if distance is not None:
        row["distance"] = distance
    return row


def _row_to_fact(
    fid: str,
    doc: str | None,
    meta: dict[str, Any] | None,
    distance: float | None = None,
) -> dict[str, Any]:
    m = meta or {}
    row: dict[str, Any] = {
        "id": fid,
        "text": doc,
        "confidence": float(m.get("confidence", 1.0)),
        "mentions": int(m.get("mentions", 1)),
        "session_id": m.get("session_id") or None,
        "subject": m.get("subject") or None,
        "is_active": _int_to_bool(m.get("is_active", 1)),
        "superseded_by": m.get("superseded_by") or None,
        "metadata": m.get("metadata"),
        "event_time": m.get("event_time") or None,
        "created_at": m.get("created_at"),
    }
    if distance is not None:
        row["distance"] = distance
    return row
