"""Unit tests for Chroma and LanceDB factory routing — no live server required."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from vektori.config import VektoriConfig
from vektori.storage.chroma_backend import ChromaBackend
from vektori.storage.factory import create_storage
from vektori.storage.lancedb_backend import LanceDBBackend


# ── ChromaBackend constructor ──────────────────────────────────────────────────


def test_chroma_defaults():
    b = ChromaBackend()
    assert b.path is None
    assert b.host is None
    assert b.port == 8000
    assert b.prefix == "vektori"
    assert b.embedding_dim == 1024


def test_chroma_custom_params():
    b = ChromaBackend(path="/tmp/chroma", prefix="myapp", embedding_dim=512)
    assert b.path == "/tmp/chroma"
    assert b.prefix == "myapp"
    assert b.embedding_dim == 512
    assert b._facts_col == "myapp_facts"
    assert b._sentences_col == "myapp_sentences"
    assert b._episodes_col == "myapp_episodes"
    assert b._sessions_col == "myapp_sessions"


def test_chroma_http_params():
    b = ChromaBackend(host="chroma-host", port=9000)
    assert b.host == "chroma-host"
    assert b.port == 9000
    assert b.path is None


# ── LanceDBBackend constructor ────────────────────────────────────────────────


def test_lancedb_defaults():
    b = LanceDBBackend()
    assert b.uri == ".lancedb"
    assert b.prefix == "vektori"
    assert b.embedding_dim == 1024


def test_lancedb_custom_params():
    b = LanceDBBackend(uri="/tmp/lancedb", prefix="myapp", embedding_dim=768)
    assert b.uri == "/tmp/lancedb"
    assert b._facts_name == "myapp_facts"
    assert b._sentences_name == "myapp_sentences"
    assert b._episodes_name == "myapp_episodes"
    assert b._sessions_name == "myapp_sessions"


# ── Factory routing ────────────────────────────────────────────────────────────


async def test_factory_chroma_by_key():
    cfg = VektoriConfig(storage_backend="chroma", embedding_dimension=512)
    with patch.object(ChromaBackend, "initialize", new_callable=AsyncMock):
        backend = await create_storage(cfg)
    assert isinstance(backend, ChromaBackend)
    assert backend.embedding_dim == 512
    assert backend.path is None  # ephemeral when no database_url


async def test_factory_chroma_with_path():
    cfg = VektoriConfig(
        storage_backend="chroma",
        database_url="/tmp/mydb",
        embedding_dimension=256,
    )
    with patch.object(ChromaBackend, "initialize", new_callable=AsyncMock):
        backend = await create_storage(cfg)
    assert isinstance(backend, ChromaBackend)
    assert backend.path == "/tmp/mydb"
    assert backend.host is None


async def test_factory_chroma_http_url():
    cfg = VektoriConfig(
        storage_backend="chroma",
        database_url="http://chroma-host:8000",
        embedding_dimension=256,
    )
    with patch.object(ChromaBackend, "initialize", new_callable=AsyncMock):
        backend = await create_storage(cfg)
    assert isinstance(backend, ChromaBackend)
    assert backend.host == "chroma-host"
    assert backend.port == 8000
    assert backend.path is None


async def test_factory_lancedb_by_key():
    cfg = VektoriConfig(storage_backend="lancedb", embedding_dimension=768)
    with patch.object(LanceDBBackend, "initialize", new_callable=AsyncMock):
        backend = await create_storage(cfg)
    assert isinstance(backend, LanceDBBackend)
    assert backend.embedding_dim == 768


async def test_factory_lancedb_with_uri():
    cfg = VektoriConfig(
        storage_backend="lancedb",
        database_url="s3://my-bucket/lancedb",
        embedding_dimension=512,
    )
    with patch.object(LanceDBBackend, "initialize", new_callable=AsyncMock):
        backend = await create_storage(cfg)
    assert isinstance(backend, LanceDBBackend)
    assert backend.uri == "s3://my-bucket/lancedb"
