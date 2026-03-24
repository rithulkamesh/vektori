"""Deterministic ID generation for sentences."""

from __future__ import annotations

import hashlib
import uuid


def generate_sentence_id(session_id: str, sentence_index: str | int, text: str) -> str:
    """
    Generate a deterministic UUID from sentence content and position.

    Same content in the same session position always produces the same ID.
    This enables deduplication via ON CONFLICT without application-level checks.
    On re-encounter, the storage layer increments the mentions counter (IDF weighting).
    """
    raw = f"{session_id}:{sentence_index}:{text}"
    hash_bytes = hashlib.sha256(raw.encode()).digest()
    return str(uuid.UUID(bytes=hash_bytes[:16]))


def generate_content_hash(session_id: str, sentence_index: str | int, text: str) -> str:
    """Generate SHA-256 hex content hash (used as the unique index key)."""
    raw = f"{session_id}:{sentence_index}:{text}"
    return hashlib.sha256(raw.encode()).hexdigest()
