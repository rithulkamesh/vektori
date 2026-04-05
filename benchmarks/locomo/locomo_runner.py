"""
LoCoMo Benchmark Runner for Vektori
=====================================

Evaluates Vektori's persistent agent memory on the LoCoMo-10 dataset.

Ingestion strategy
------------------
Unlike LongMemEval (per-question isolation), LoCoMo groups all QA items by
sample — every QA item in the same sample shares the **exact same** haystack
sessions.  We therefore ingest once per sample, answer all pending questions
for that sample, then delete the user.

* **Per-sample isolation** — ``user_id = "locomo_{sample_id}"``.
  After all questions for a sample are answered, ``delete_user()`` wipes rows.

* **No extract cache needed** — sessions are never shared across samples, so
  there is nothing to cache between samples.

Checkpointing
-------------
Progress is saved per-question (same BenchmarkCheckpoint as LongMemEval).
Re-running resumes from the first unfinished question.  A sample whose
questions are ALL already done is skipped entirely (no re-ingestion).

Pilot mode
----------
Set ``LoCoMoConfig.max_questions`` to a small number (e.g. 5) to run a quick
sanity-check before committing to the full dataset.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmarks.longmemeval.checkpoint import BenchmarkCheckpoint
from benchmarks.longmemeval.session_cache import SessionExtractCache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class LoCoMoConfig:
    """Configuration for a LoCoMo benchmark run."""

    # Dataset
    data_dir: str = "data"
    dataset_name: str = "locomo10_cooked"

    # Vektori
    storage_backend: str = "sqlite"
    database_url: str | None = None
    embedding_model: str = "cloudflare:@cf/baai/bge-m3"
    extraction_model: str = "gemini:gemini-2.5-flash-lite"
    max_extraction_output_tokens: int = 32768

    # Retrieval
    retrieval_depth: str = "l1"
    top_k: int = 10
    context_window: int = 3
    enable_retrieval_gate: bool = False

    # Output
    output_dir: str = "benchmark_results"
    run_name: str | None = None

    # Evaluation
    eval_model: str = "gemini:gemini-2.5-flash-lite"

    # Pilot mode — set to a small number to test before full run
    max_questions: int | None = None

    # Extraction cache
    use_cache: bool = True
    cache_namespace: str | None = None


# ── Runner ────────────────────────────────────────────────────────────────────

class LoCoMoBenchmark:
    """Main LoCoMo benchmark runner."""

    def __init__(self, config: LoCoMoConfig) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.vektori_client = None
        self.storage = None
        self.dataset: list[dict[str, Any]] = []
        self.dataset_path: Path | None = None
        self._session_cache: SessionExtractCache | None = None
        self._checkpoint: BenchmarkCheckpoint | None = None

    # ── Setup ────────────────────────────────────────────────────────────────

    async def setup(self) -> None:
        logger.info("Setting up LoCoMo benchmark…")
        await self._init_vektori()
        await self._load_dataset()
        await self._init_session_cache()
        self._init_checkpoint()
        pilot_note = (
            f" [PILOT MODE: first {self.config.max_questions} questions only]"
            if self.config.max_questions else ""
        )
        logger.info(
            "Setup complete — %d questions in dataset%s", len(self.dataset), pilot_note
        )

    async def _init_vektori(self) -> None:
        from vektori import Vektori
        from vektori.config import VektoriConfig

        logger.info(
            "Initialising Vektori (backend=%s, embedding=%s, extraction=%s)",
            self.config.storage_backend,
            self.config.embedding_model,
            self.config.extraction_model,
        )
        self.vektori_client = Vektori(
            config=VektoriConfig(
                database_url=self.config.database_url,
                storage_backend=self.config.storage_backend,
                embedding_model=self.config.embedding_model,
                extraction_model=self.config.extraction_model,
                default_top_k=self.config.top_k,
                context_window=self.config.context_window,
                enable_retrieval_gate=self.config.enable_retrieval_gate,
                async_extraction=False,
                max_extraction_output_tokens=self.config.max_extraction_output_tokens,
            )
        )
        await self.vektori_client._ensure_initialized()
        self.storage = self.vektori_client.db

    async def _load_dataset(self) -> None:
        filename = f"{self.config.dataset_name}.json"
        self.dataset_path = Path(self.config.data_dir) / filename

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"LoCoMo cooked dataset not found at: {self.dataset_path}\n"
                "Cook it first with:\n"
                "  python -m benchmarks.locomo.cook_locomo "
                "--output data/locomo10_cooked.json --download-if-missing"
            )

        logger.info("Loading dataset from %s", self.dataset_path)
        with open(self.dataset_path, encoding="utf-8") as f:
            all_items = json.load(f)

        # Apply pilot cap if set
        if self.config.max_questions is not None:
            all_items = all_items[: self.config.max_questions]
            logger.info(
                "Pilot mode: using %d/%d questions",
                len(all_items), self.config.max_questions,
            )

        self.dataset = all_items
        logger.info("Loaded %d QA items", len(self.dataset))

    async def _init_session_cache(self) -> None:
        if not self.config.use_cache:
            logger.info("Session extract cache disabled (--no-cache)")
            self._session_cache = None
            return

        # Shared with LongMemEval cache dir so sessions extracted by either
        # benchmark can be reused by the other (same session_id namespace).
        cache_path = Path(self.config.output_dir).parent / ".cache" / "session_extract_cache.db"
        self._session_cache = SessionExtractCache(cache_path)
        await self._session_cache.initialize()
        logger.info("Session cache namespace: %s", self._cache_namespace())

    def _cache_namespace(self) -> str:
        if self.config.cache_namespace:
            return self.config.cache_namespace
        raw = (
            f"locomo|{self.config.extraction_model}|"
            f"out={self.config.max_extraction_output_tokens}"
        )
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]

    def _cache_key(self, session_id: str) -> str:
        return f"{self._cache_namespace()}::{session_id}"

    def _init_checkpoint(self) -> None:
        run_name = self.config.run_name or self.config.dataset_name
        chk_path = self.output_dir / f"{run_name}_checkpoint.json"
        self._checkpoint = BenchmarkCheckpoint(chk_path)
        n_done = self._checkpoint.load()
        remaining = len(self.dataset) - n_done
        if n_done:
            logger.info("Resuming — %d done, %d remaining", n_done, remaining)

    # ── Main loop ────────────────────────────────────────────────────────────

    async def run(self) -> None:
        try:
            await self.setup()
            await self._run_samples()
            await self.evaluate()
            await self.save_results()
            logger.info("LoCoMo benchmark complete!")
            self._print_summary()
        finally:
            await self.cleanup()

    async def _run_samples(self) -> None:
        """Group dataset by sample_id; ingest once per sample, answer all QA items."""
        # Group questions by sample_id (preserves order within each sample)
        samples: dict[str, list[dict[str, Any]]] = {}
        sample_order: list[str] = []
        for item in self.dataset:
            sid = _sample_id_from_question_id(item["question_id"])
            if sid not in samples:
                samples[sid] = []
                sample_order.append(sid)
            samples[sid].append(item)

        total_qs = len(self.dataset)
        logger.info("Processing %d samples (%d total questions)", len(sample_order), total_qs)

        for sample_id in sample_order:
            qa_items = samples[sample_id]
            pending = [qa for qa in qa_items if not self._checkpoint.is_done(qa["question_id"])]

            if not pending:
                logger.debug("Sample %s: all questions done, skipping ingestion", sample_id)
                continue

            user_id = f"locomo_{sample_id}"
            logger.info(
                "Sample %s — ingesting sessions, then answering %d/%d questions",
                sample_id, len(pending), len(qa_items),
            )

            try:
                ingest_t0 = time.perf_counter()
                await self._ingest_sample(qa_items[0], user_id)
                ingestion_ms = (time.perf_counter() - ingest_t0) * 1000
                logger.info("Sample %s ingested in %.0f ms", sample_id, ingestion_ms)

                for qa_item in pending:
                    qid = qa_item["question_id"]
                    try:
                        q_t0 = time.perf_counter()
                        result = await self._answer_question(qa_item, user_id)
                        total_ms = (time.perf_counter() - q_t0) * 1000

                        result["ingestion_ms"] = round(ingestion_ms, 1)
                        result["total_question_ms"] = round(total_ms, 1)

                        self._checkpoint.mark_done(qid, result)
                        self._checkpoint.save()

                        done = self._checkpoint.n_completed
                        logger.info(
                            "  [%d/%d] %s → retrieval=%.0fms  qa=%.0fms  total=%.0fms",
                            done, total_qs, qid,
                            result["retrieval_ms"], result["qa_ms"], result["total_question_ms"],
                        )
                    except Exception as e:
                        logger.error("Question %s failed: %s", qid, e, exc_info=True)
                        self._checkpoint.mark_failed(qid, str(e))
                        self._checkpoint.save()

            except Exception as e:
                logger.error("Sample %s ingestion failed: %s", sample_id, e, exc_info=True)
                for qa_item in pending:
                    self._checkpoint.mark_failed(qa_item["question_id"], f"ingestion_failed: {e}")
                self._checkpoint.save()
            finally:
                try:
                    await self.vektori_client.delete_user(user_id)
                except Exception as e:
                    logger.warning("delete_user(%s) failed: %s", user_id, e)

    # ── Ingestion ────────────────────────────────────────────────────────────

    async def _ingest_sample(self, reference_item: dict[str, Any], user_id: str) -> None:
        """Ingest all haystack sessions for a sample.

        Cache hit  → replay pre-extracted facts (no LLM, only local re-embed).
        Cache miss → full LLM extraction, write to cache.

        This enables crash recovery: if the run is interrupted mid-sample,
        sessions already extracted are cached and won't cost another LLM call
        on the next run.
        """
        haystack_sessions = reference_item["haystack_sessions"]
        haystack_sids = reference_item["haystack_session_ids"]
        haystack_dates = reference_item.get("haystack_dates") or []
        n_sessions = len(haystack_sessions)

        for i, (session, hsid) in enumerate(zip(haystack_sessions, haystack_sids)):
            session_date = haystack_dates[i] if i < len(haystack_dates) else None
            session_time = _parse_date(session_date) if session_date else None

            sess_t0 = time.perf_counter()
            cached_facts = (
                await self._session_cache.get(self._cache_key(hsid))
                if self._session_cache
                else None
            )
            if cached_facts is not None:
                await self._replay_session(session, hsid, user_id, session_time, session_date, cached_facts)
                elapsed = (time.perf_counter() - sess_t0) * 1000
                logger.info(
                    "  Session %d/%d [CACHE HIT ] %s — %.0f ms (%d facts replayed)",
                    i + 1, n_sessions, hsid, elapsed, len(cached_facts),
                )
            else:
                new_facts = await self._full_ingest_session(
                    session, hsid, user_id, session_time, session_date
                )
                elapsed = (time.perf_counter() - sess_t0) * 1000
                logger.info(
                    "  Session %d/%d [CACHE MISS] %s — %.0f ms (%d facts extracted)",
                    i + 1, n_sessions, hsid, elapsed, len(new_facts),
                )
                if new_facts and self._session_cache:
                    await self._session_cache.put(self._cache_key(hsid), new_facts)

    async def _replay_session(
        self,
        session: list[dict[str, str]],
        haystack_sid: str,
        user_id: str,
        session_time: datetime | None,
        session_date: str | None,
        cached_facts: list[dict[str, Any]],
    ) -> None:
        """Cache hit: store sentences + replay pre-extracted facts + run episode extraction."""
        pipeline = self.vektori_client._pipeline
        extractor = self.vektori_client._extractor

        await pipeline.ingest(
            messages=session,
            session_id=haystack_sid,
            user_id=user_id,
            metadata={"timestamp": session_date} if session_date else None,
            session_time=session_time,
            skip_extraction=True,
        )

        inserted_facts: list[tuple[str, str]] = []
        await extractor.replay_facts(
            cached_facts=cached_facts,
            session_id=haystack_sid,
            user_id=user_id,
            session_time=session_time,
            _inserted_facts_out=inserted_facts,
        )

        if inserted_facts:
            conversation = "\n".join(
                f"{msg['role'].upper()}: {msg['content']}" for msg in session
            )
            try:
                await extractor._extract_episodes(
                    inserted_facts=inserted_facts,
                    conversation=conversation,
                    session_id=haystack_sid,
                    user_id=user_id,
                    agent_id=None,
                    session_time=session_time,
                )
            except Exception as e:
                logger.warning(
                    "Episode extraction failed for cached session %s: %s", haystack_sid, e
                )

    async def _full_ingest_session(
        self,
        session: list[dict[str, str]],
        haystack_sid: str,
        user_id: str,
        session_time: datetime | None,
        session_date: str | None,
    ) -> list[dict[str, Any]]:
        """Cache miss: full LLM extraction. Returns cacheable facts."""
        pipeline = self.vektori_client._pipeline
        extractor = self.vektori_client._extractor

        await pipeline.ingest(
            messages=session,
            session_id=haystack_sid,
            user_id=user_id,
            metadata={"timestamp": session_date} if session_date else None,
            session_time=session_time,
            skip_extraction=True,
        )

        captured_facts: list[dict[str, Any]] = []
        await extractor.extract(
            messages=session,
            session_id=haystack_sid,
            user_id=user_id,
            session_time=session_time,
            _capture_out=captured_facts,
        )

        return captured_facts

    # ── Retrieval + QA ────────────────────────────────────────────────────────

    async def _answer_question(
        self, item: dict[str, Any], user_id: str
    ) -> dict[str, Any]:
        qid = item["question_id"]
        question = item["question"]
        question_type = item["question_type"]
        question_date = item.get("question_date") or ""

        retrieval_t0 = time.perf_counter()
        search_results = await self.vektori_client.search(
            query=question,
            user_id=user_id,
            depth=self.config.retrieval_depth,
            reference_date=_parse_date(question_date) if question_date else None,
        )
        retrieval_ms = (time.perf_counter() - retrieval_t0) * 1000

        context = _format_retrieved_context(search_results)

        qa_t0 = time.perf_counter()
        answer = await self._generate_answer(question, context, question_date)
        qa_ms = (time.perf_counter() - qa_t0) * 1000

        return {
            "question_id": qid,
            "question": question,
            "question_type": question_type,
            "hypothesis": answer,
            "expected_answer": item["answer"],
            "retrieved_context": context,
            "retrieval_depth": self.config.retrieval_depth,
            "retrieval_ms": round(retrieval_ms, 1),
            "qa_ms": round(qa_ms, 1),
        }

    async def _generate_answer(
        self, question: str, context: str, question_date: str = ""
    ) -> str:
        from vektori.models.factory import create_llm

        if "No relevant context" in context:
            return "I don't have relevant information to answer this question."

        llm = create_llm(self.config.eval_model)
        date_line = f"TODAY'S DATE: {question_date}\n\n" if question_date else ""

        prompt = (
            "You are an AI assistant answering questions based on provided context "
            "from a long-term conversation history.\n\n"
            f"{date_line}"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION:\n{question}\n\n"
            "INSTRUCTIONS:\n"
            "- Answer the question based ONLY on the provided context\n"
            "- Be concise and direct — a short phrase or sentence is preferred\n"
            "- If the context does not contain enough information, say "
            "\"I don't have that information\"\n\n"
            "ANSWER:"
        )
        max_tokens = 500
        try:
            return (await llm.generate(prompt, max_tokens=max_tokens)).strip()
        except Exception as e:
            logger.warning("Answer generation failed: %s", e)
            return "Unable to generate answer due to API error."

    # ── Evaluation ────────────────────────────────────────────────────────────

    async def evaluate(self) -> None:
        logger.info("Computing evaluation metrics…")
        qa_results = list(self._checkpoint.get_completed().values())
        if not qa_results:
            logger.warning("No completed QA results to evaluate")
            return

        run_name = self.config.run_name or self.config.dataset_name
        jsonl_path = self.output_dir / f"{run_name}_qa_results.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for r in qa_results:
                f.write(
                    json.dumps({
                        "question_id": r["question_id"],
                        "hypothesis": r["hypothesis"],
                    }) + "\n"
                )
        logger.info("QA pairs saved to %s", jsonl_path)

        self._metrics = self._compute_metrics(qa_results)

    def _compute_metrics(self, qa_results: list[dict[str, Any]]) -> dict[str, Any]:
        metrics: dict[str, Any] = {
            "total_questions": len(qa_results),
            "answered": 0,
            "abstained": 0,
            "by_type": {},
        }
        for r in qa_results:
            hyp = (r.get("hypothesis") or "").lower()
            answered = bool(hyp) and "not available" not in hyp
            if answered:
                metrics["answered"] += 1
            else:
                metrics["abstained"] += 1

            qt = r.get("question_type", "unknown")
            metrics["by_type"].setdefault(qt, {"total": 0, "answered": 0})
            metrics["by_type"][qt]["total"] += 1
            if answered:
                metrics["by_type"][qt]["answered"] += 1

        def _collect(field: str) -> list[float]:
            return [float(r[field]) for r in qa_results if isinstance(r.get(field), (int, float))]

        def _avg(vals: list[float]) -> float | None:
            return round(sum(vals) / len(vals), 1) if vals else None

        retrieval_vals = _collect("retrieval_ms")
        qa_vals = _collect("qa_ms")
        total_vals = _collect("total_question_ms")

        metrics["latency_ms"] = {
            "retrieval_avg": _avg(retrieval_vals),
            "qa_avg": _avg(qa_vals),
            "total_question_avg": _avg(total_vals),
        }

        return metrics

    # ── Save results ──────────────────────────────────────────────────────────

    async def save_results(self) -> None:
        run_name = self.config.run_name or self.config.dataset_name
        qa_results = list(self._checkpoint.get_completed().values())
        metrics = getattr(self, "_metrics", None)

        full = {
            "config": self.config.__dict__,
            "metrics": metrics,
            "qa_results": qa_results,
            "cache_sessions": await self._session_cache.count() if self._session_cache else None,
        }
        full_path = self.output_dir / f"{run_name}_full_results.json"
        with open(full_path, "w", encoding="utf-8") as f:
            json.dump(full, f, indent=2, default=str)
        logger.info("Full results → %s", full_path)

        summary = {
            "config": self.config.__dict__,
            "metrics": metrics,
            "completed": self._checkpoint.n_completed,
            "failed": self._checkpoint.n_failed,
            "cache_sessions": await self._session_cache.count() if self._session_cache else None,
        }
        summary_path = self.output_dir / f"{run_name}_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info("Summary → %s", summary_path)

    # ── Cleanup ───────────────────────────────────────────────────────────────

    async def cleanup(self) -> None:
        if self._session_cache:
            await self._session_cache.close()
        if self.vektori_client:
            await self.vektori_client.close()
        logger.info("Cleanup complete")

    # ── Print summary ─────────────────────────────────────────────────────────

    def _print_summary(self) -> None:
        metrics = getattr(self, "_metrics", None)
        print("\n" + "=" * 60)
        print("LOCOMO BENCHMARK RESULTS")
        print("=" * 60)

        if metrics:
            print(f"\nTotal     : {metrics['total_questions']}")
            print(f"Answered  : {metrics['answered']}")
            print(f"Abstained : {metrics['abstained']}")
            lat = metrics.get("latency_ms") or {}
            if lat:
                print("\nLatency (ms):")
                print(f"  Retrieval      avg={lat.get('retrieval_avg')}")
                print(f"  QA generation  avg={lat.get('qa_avg')}")
                print(f"  Total/question avg={lat.get('total_question_avg')}")
            if metrics.get("by_type"):
                print("\nBy question type:")
                for qt, counts in metrics["by_type"].items():
                    pct = counts["answered"] / counts["total"] * 100 if counts["total"] else 0
                    print(f"  {qt:<35} {counts['answered']}/{counts['total']}  ({pct:.1f} %)")

        print(f"\nCompleted : {self._checkpoint.n_completed}")
        print(f"Failed    : {self._checkpoint.n_failed}")
        print(f"\nResults in: {self.output_dir}")
        print("=" * 60 + "\n")


# ── Module-level helpers ──────────────────────────────────────────────────────

_QID_RE = re.compile(r"^locomo_(.+)_q\d+$")


def _sample_id_from_question_id(qid: str) -> str:
    """Extract sample_id from 'locomo_{sample_id}_q{N}'."""
    m = _QID_RE.match(qid)
    if not m:
        raise ValueError(f"Cannot parse sample_id from question_id: {qid!r}")
    return m.group(1)


def _parse_date(date_str: str) -> datetime | None:
    """Parse LoCoMo session date strings.

    Supports:
    - LoCoMo native format: '9:55 am on 22 October, 2023'
    - ISO-ish strings like '2023-05-30' or '2023-05-30T14:30:00'
    - LongMemEval-style fallback strings
    """
    if not date_str:
        return None

    clean = date_str.strip()

    # LoCoMo native format
    for fmt in ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %b, %Y"):
        try:
            return datetime.strptime(clean.upper(), fmt)
        except ValueError:
            continue

    # Try ISO formats first
    for fmt in (
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(clean, fmt)
        except ValueError:
            continue

    # LongMemEval-style fallback: '2023/05/30 (Tue) 23:40'
    clean = date_str.split("(")[0].strip() + " " + date_str.split(")")[-1].strip()
    for fmt in ("%Y/%m/%d %H:%M", "%Y/%m/%d"):
        try:
            return datetime.strptime(clean.strip(), fmt)
        except ValueError:
            continue

    logger.debug("Could not parse date string: %r", date_str)
    return None


def _format_retrieved_context(search_results: Any) -> str:
    if not search_results:
        return "No relevant context retrieved."

    lines: list[str] = []

    facts = search_results.get("facts") or []
    if facts:
        facts = sorted(
            facts,
            key=lambda f: f.get("event_time") or f.get("created_at") or "",
        )
        lines.append("## Facts")
        for i, fact in enumerate(facts, 1):
            date_prefix = ""
            ts = fact.get("event_time") or fact.get("created_at") or ""
            if ts:
                date_prefix = f"[{str(ts)[:10]}] "
            lines.append(f"{i}. {date_prefix}{fact.get('text', str(fact))}")

    episodes = search_results.get("episodes") or []
    if episodes:
        lines.append("\n## Episodes")
        for i, ep in enumerate(episodes, 1):
            lines.append(f"{i}. {ep.get('text', str(ep))}")

    sentences = search_results.get("sentences") or []
    if sentences:
        session_sents: dict[str, list[dict[str, Any]]] = {}
        session_order: list[str] = []
        for sent in sentences:
            ssid = sent.get("session_id") or "unknown"
            if ssid not in session_sents:
                session_sents[ssid] = []
                session_order.append(ssid)
            session_sents[ssid].append(sent)

        lines.append("\n## Session Context")
        for n, ssid in enumerate(session_order, 1):
            sents = session_sents[ssid]
            date_hint = ""
            for s in sents:
                ts = s.get("created_at") or ""
                if ts:
                    date_hint = f" [{str(ts)[:10]}]"
                    break
            lines.append(f"\n### Session {n}{date_hint}")
            for sent in sents:
                role = sent.get("role", "")
                prefix = f"[{role.upper()}] " if role else ""
                lines.append(f"  {prefix}{sent.get('text', str(sent))}")

    return "\n".join(lines) if lines else "No relevant context retrieved."


# ── CLI entry point ───────────────────────────────────────────────────────────

async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run Vektori benchmark on LoCoMo-10")
    parser.add_argument("--depth", choices=["l0", "l1", "l2"], default="l1")
    parser.add_argument(
        "--embedding-model", default="cloudflare:@cf/baai/bge-m3",
        help="Embedding model string (provider:model_name)"
    )
    parser.add_argument(
        "--extraction-model", default="gemini:gemini-2.5-flash-lite",
        help="LLM for fact/episode extraction"
    )
    parser.add_argument(
        "--eval-model", default="gemini:gemini-2.5-flash-lite",
        help="LLM for QA answer generation"
    )
    parser.add_argument(
        "--max-extraction-output-tokens", type=int, default=32768,
        help="Max output tokens for extraction LLM calls"
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Disable session extract cache for this run"
    )
    parser.add_argument(
        "--cache-namespace", default=None,
        help="Optional cache namespace override to isolate cached extractions"
    )
    parser.add_argument("--output-dir", default="benchmark_results")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--run-name", help="Run name for checkpointing/output files")
    parser.add_argument(
        "--max-questions", type=int, default=None,
        help="Limit to first N questions (pilot mode; omit for full run)"
    )

    args = parser.parse_args()

    config = LoCoMoConfig(
        retrieval_depth=args.depth,
        embedding_model=args.embedding_model,
        extraction_model=args.extraction_model,
        eval_model=args.eval_model,
        max_extraction_output_tokens=args.max_extraction_output_tokens,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        top_k=args.top_k,
        run_name=args.run_name,
        max_questions=args.max_questions,
        use_cache=not args.no_cache,
        cache_namespace=args.cache_namespace,
    )

    logger.info("Starting LoCoMo benchmark — config: %s", config)
    await LoCoMoBenchmark(config).run()


if __name__ == "__main__":
    asyncio.run(main())
