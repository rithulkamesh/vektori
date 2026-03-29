"""Atomic per-question checkpoint for the LongMemEval benchmark runner.

Writes progress after every completed question so a crashed or interrupted
run can resume from exactly where it left off without re-running (expensive)
LLM extraction or QA generation for already-finished questions.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class BenchmarkCheckpoint:
    """Atomic JSON checkpoint.

    File layout::

        {
            "completed": {
                "<question_id>": { <full qa result dict> }
            },
            "failed": {
                "<question_id>": "<error message>"
            }
        }

    Writes are atomic: data is written to a ``.tmp`` sibling, then
    ``os.replace``d over the real file so a crash mid-write never corrupts
    the checkpoint.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._completed: dict[str, dict[str, Any]] = {}
        self._failed: dict[str, str] = {}

    # ── Load / Save ──────────────────────────────────────────────────────────

    def load(self) -> int:
        """Load an existing checkpoint file.  Returns number of completed questions."""
        if not self._path.exists():
            return 0
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._completed = data.get("completed", {})
            self._failed = data.get("failed", {})
            n = len(self._completed)
            logger.info(
                "Checkpoint loaded from %s — %d completed, %d failed",
                self._path, n, len(self._failed),
            )
            return n
        except Exception as e:
            logger.warning("Could not load checkpoint %s (%s) — starting fresh", self._path, e)
            return 0

    def save(self) -> None:
        """Atomically persist current state to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"completed": self._completed, "failed": self._failed}, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, self._path)

    # ── Status checks ────────────────────────────────────────────────────────

    def is_done(self, question_id: str) -> bool:
        return question_id in self._completed

    # ── Updates ──────────────────────────────────────────────────────────────

    def mark_done(self, question_id: str, result: dict[str, Any]) -> None:
        self._completed[question_id] = result
        self._failed.pop(question_id, None)

    def mark_failed(self, question_id: str, error: str) -> None:
        if question_id not in self._completed:
            self._failed[question_id] = error

    # ── Accessors ────────────────────────────────────────────────────────────

    def get_completed(self) -> dict[str, dict[str, Any]]:
        return self._completed

    @property
    def n_completed(self) -> int:
        return len(self._completed)

    @property
    def n_failed(self) -> int:
        return len(self._failed)
