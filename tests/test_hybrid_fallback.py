"""
Tests for hybrid search fallback behavior.

When a collection has no sparse vector slot (dense-only) or the sparse model
name doesn't match what's stored, search_hybrid_rrf must NOT silently return [].
It should fall back to dense-only results and surface a warning.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from raggy_mcp.qdrant import QdrantConnector, _retrieval_warnings
from tests.test_provider_resolver import FakeProvider


class FakeSparseProvider:
    def get_vector_name(self):
        return "sparse"

    async def embed_query(self, text: str) -> dict:
        return {"indices": [0, 1], "values": [0.5, 0.5]}


def make_connector():
    provider = FakeProvider("dense")
    connector = QdrantConnector(":memory:", None, "docs", provider)
    # Prevent the client from actually connecting anywhere
    connector._client = None
    return connector, provider


def _fake_entry():
    from raggy_mcp.settings import METADATA_PATH

    return SimpleNamespace(
        id="1",
        payload={"document": "hello", METADATA_PATH: {}},
        score=0.9,
    )


@pytest.mark.asyncio
async def test_hybrid_rrf_exception_falls_back_to_dense():
    """RRF call raises (e.g. missing sparse slot) → returns dense results, not []."""
    connector, _ = make_connector()

    fake_client = MagicMock()
    fake_client.collection_exists = AsyncMock(return_value=True)
    fake_client.query_points = AsyncMock(
        side_effect=Exception("sparse vector not found")
    )
    connector._client = fake_client

    # _hybrid_search_client_side also calls query_points with dense only; mock it to succeed
    dense_result = SimpleNamespace(points=[_fake_entry()])
    # First call raises (RRF attempt), second call succeeds (dense fallback)
    fake_client.query_points = AsyncMock(
        side_effect=[Exception("sparse vector not found"), dense_result]
    )

    results = await connector.search_hybrid_rrf(
        "test query",
        "my_collection",
        FakeSparseProvider(),
        limit=5,
    )

    assert len(results) == 1, "should return dense-only results, not empty"
    assert results[0][0].content == "hello"  # payload["document"] key


@pytest.mark.asyncio
async def test_hybrid_rrf_exception_populates_warnings_contextvar():
    """When RRF fails, the fallback warning must be pushed to _retrieval_warnings."""
    connector, _ = make_connector()

    fake_client = MagicMock()
    dense_result = SimpleNamespace(points=[_fake_entry()])
    fake_client.query_points = AsyncMock(
        side_effect=[Exception("no sparse vector slot"), dense_result]
    )
    connector._client = fake_client

    sink: list[str] = []
    _retrieval_warnings.set(sink)

    await connector.search_hybrid_rrf(
        "test query", "my_collection", FakeSparseProvider(), limit=5
    )

    assert sink, "warning should be appended to the ContextVar sink"
    assert "dense-only" in sink[0].lower() or "hybrid rrf failed" in sink[0].lower()


@pytest.mark.asyncio
async def test_hybrid_rrf_empty_rrf_falls_back_to_dense():
    """RRF succeeds but returns 0 results → fall back to dense, not []."""
    connector, _ = make_connector()

    fake_client = MagicMock()
    empty_result = SimpleNamespace(points=[])
    dense_result = SimpleNamespace(points=[_fake_entry()])
    # First call: RRF returns empty. Second call: dense returns results.
    fake_client.query_points = AsyncMock(side_effect=[empty_result, dense_result])
    connector._client = fake_client

    results = await connector.search_hybrid_rrf(
        "test query", "my_collection", FakeSparseProvider(), limit=5
    )

    assert len(results) == 1, "should fall back to dense when RRF returns empty"


@pytest.mark.asyncio
async def test_hybrid_rrf_empty_rrf_populates_warnings():
    """Empty RRF result also pushes to the warnings ContextVar."""
    connector, _ = make_connector()

    fake_client = MagicMock()
    empty_result = SimpleNamespace(points=[])
    dense_result = SimpleNamespace(points=[_fake_entry()])
    fake_client.query_points = AsyncMock(side_effect=[empty_result, dense_result])
    connector._client = fake_client

    sink: list[str] = []
    _retrieval_warnings.set(sink)

    await connector.search_hybrid_rrf(
        "test query", "my_collection", FakeSparseProvider(), limit=5
    )

    assert sink, "empty-RRF warning must be pushed to ContextVar"
    assert (
        "sparse" in sink[0].lower()
        or "0 results" in sink[0].lower()
        or "dense" in sink[0].lower()
    )


@pytest.mark.asyncio
async def test_hybrid_rrf_no_warning_when_contextvar_not_set():
    """If _retrieval_warnings is None (not set), fallback still works without error."""
    connector, _ = make_connector()

    fake_client = MagicMock()
    dense_result = SimpleNamespace(points=[_fake_entry()])
    fake_client.query_points = AsyncMock(
        side_effect=[Exception("sparse slot missing"), dense_result]
    )
    connector._client = fake_client

    _retrieval_warnings.set(None)  # explicitly unset

    # Should not raise even though ContextVar is None
    results = await connector.search_hybrid_rrf(
        "test query", "my_collection", FakeSparseProvider(), limit=5
    )
    assert len(results) == 1


@pytest.mark.asyncio
async def test_hybrid_rrf_both_fallbacks_fail_returns_empty():
    """If both RRF and dense fallback fail, return [] without raising."""
    connector, _ = make_connector()

    fake_client = MagicMock()
    fake_client.query_points = AsyncMock(side_effect=Exception("everything is broken"))
    connector._client = fake_client

    results = await connector.search_hybrid_rrf(
        "test query", "my_collection", FakeSparseProvider(), limit=5
    )

    assert results == [], "should return [] when all paths fail"


# ── get_sparse_vector_name tests ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_sparse_vector_name_returns_name_for_hybrid_collection():
    """Returns the first sparse vector name when the collection has one."""
    connector, _ = make_connector()

    fake_sparse_config = {"sparse-bm25": MagicMock()}
    fake_params = SimpleNamespace(sparse_vectors=fake_sparse_config)
    fake_config = SimpleNamespace(params=fake_params)
    fake_info = SimpleNamespace(config=fake_config)

    fake_client = MagicMock()
    fake_client.get_collection = AsyncMock(return_value=fake_info)
    connector._client = fake_client

    name = await connector.get_sparse_vector_name("my_collection")
    assert name == "sparse-bm25"


@pytest.mark.asyncio
async def test_get_sparse_vector_name_returns_none_for_dense_only_collection():
    """Returns None when the collection has no sparse vectors."""
    connector, _ = make_connector()

    fake_params = SimpleNamespace(sparse_vectors=None)
    fake_config = SimpleNamespace(params=fake_params)
    fake_info = SimpleNamespace(config=fake_config)

    fake_client = MagicMock()
    fake_client.get_collection = AsyncMock(return_value=fake_info)
    connector._client = fake_client

    name = await connector.get_sparse_vector_name("my_collection")
    assert name is None


@pytest.mark.asyncio
async def test_get_sparse_vector_name_returns_none_on_error():
    """Returns None (not raises) if the Qdrant client call fails."""
    connector, _ = make_connector()

    fake_client = MagicMock()
    fake_client.get_collection = AsyncMock(side_effect=Exception("network error"))
    connector._client = fake_client

    name = await connector.get_sparse_vector_name("my_collection")
    assert name is None
