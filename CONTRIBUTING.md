# Contributing to Vektori

Thank you for your interest!

## Setup

```bash
git clone https://github.com/vektori-ai/vektori
cd vektori
pip install -e ".[dev]"
python -m spacy download en_core_web_sm
pre-commit install
```

## Running Tests

```bash
# Unit tests (no external deps, fast)
pytest tests/unit/ -v

# Integration tests (memory backend, no Docker)
pytest tests/integration/ -v

# All
pytest -v
```

## Code Style

```bash
ruff check .       # lint
ruff format .      # format
mypy vektori/      # type check
```

## Commit Convention

[Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add SQLite storage backend
fix: handle empty conversation in splitter
docs: add CrewAI integration example
refactor: extract model factory from client.py
test: add conflict resolution integration tests
```

## Architecture

The three-layer graph (Facts → Episodes → Sentences) is the core invariant.
See `VEKTORI_TECHNICAL_SPEC.md` for full architecture docs.

- **L0 Facts**: Primary search surface, vector search lands here.
- **L1 Episodes**: Discovered via `episode_facts` graph traversal, not vector search.
- **L2 Sentences**: Raw conversation, sequential `NEXT` edges within sessions.

Do not break this layering without discussion.

## Adding a New Model Provider

1. Create `vektori/models/yourprovider.py` implementing `EmbeddingProvider` and/or `LLMProvider`
2. Register in `vektori/models/factory.py` (`EMBEDDING_REGISTRY` / `LLM_REGISTRY`)
3. Add to `pyproject.toml` optional-dependencies
4. Add an example in `examples/`
5. Add unit tests in `tests/unit/test_factory.py`

## Adding a New Storage Backend

1. Create `vektori/storage/yourbackend.py` implementing all methods in `StorageBackend`
2. Register in `vektori/storage/factory.py`
3. Add integration tests
