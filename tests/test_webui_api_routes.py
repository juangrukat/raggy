"""
Lightweight tests for the REST API route structure.

These tests verify that routes are registered correctly and that
request/response models behave as expected, without requiring a real Qdrant
connection. Qdrant and embedding calls are mocked at the connector level.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest
from fastapi.testclient import TestClient

from raggy_mcp.webui.api import create_app
from raggy_mcp.qdrant import CollectionInfo


def _build_test_app():
    """Create a REST app with a mocked QdrantConnector."""
    mock_connector = MagicMock()
    mock_connector.get_collection_names = AsyncMock(return_value=["docs", "hybrid_test"])
    mock_connector.get_sparse_vector_name = AsyncMock(return_value=None)
    mock_connector.create_hybrid_collection = AsyncMock(return_value=True)
    mock_connector.ensure_macos_metadata_indexes = AsyncMock()

    mock_mgr = MagicMock()
    mock_model_info_4b = MagicMock()
    mock_model_info_4b.vector_size = 2560
    mock_model_info_8b = MagicMock()
    mock_model_info_8b.vector_size = 4096
    mock_mgr.get_model_info = MagicMock(return_value=mock_model_info_4b)

    mock_provider_4b = MagicMock()
    mock_provider_4b.get_vector_name = MagicMock(return_value="qwen3-qwen3-embedding-4b")
    mock_provider_4b.get_model_name = MagicMock(return_value="Qwen/Qwen3-Embedding-4B")
    mock_provider_4b.get_vector_size = MagicMock(return_value=2560)
    mock_mgr.get_default_provider = MagicMock(return_value=mock_provider_4b)

    # create_provider_for_model returns the right mock based on the model name
    # so that set_active_model tests can verify the provider was actually swapped.
    mock_provider_8b = MagicMock()
    mock_provider_8b.get_vector_name = MagicMock(return_value="qwen3-qwen3-embedding-8b")
    mock_provider_8b.get_model_name = MagicMock(return_value="Qwen/Qwen3-Embedding-8B")
    mock_provider_8b.get_vector_size = MagicMock(return_value=4096)
    def _create_provider_for_model(model_name: str):
        if "8B" in model_name:
            return mock_provider_8b
        return mock_provider_4b
    mock_mgr.create_provider_for_model = MagicMock(side_effect=_create_provider_for_model)

    class _FakeQueueStats:
        max_concurrency = 1
        max_queue_size = 8
        running = 0
        waiting = 0

    # Patch the constructor so create_app uses our mock objects
    with patch("raggy_mcp.webui.api.QdrantConnector", return_value=mock_connector), \
         patch("raggy_mcp.webui.api.EnhancedEmbeddingModelManager", return_value=mock_mgr), \
         patch("raggy_mcp.webui.api.WriteQueue") as mock_wq_cls:
        mock_wq = MagicMock()
        mock_wq.stats = AsyncMock(return_value=_FakeQueueStats())
        mock_wq_cls.return_value = mock_wq
        app = create_app()

    return app, mock_connector


def test_post_collections_hybrid_route_exists():
    """POST /collections/hybrid is registered as a distinct route."""
    app, _ = _build_test_app()
    routes = {r.path for r in app.routes if hasattr(r, "path")}
    assert "/collections/hybrid" in routes


def test_post_collections_hybrid_calls_create_hybrid_collection():
    """POST /collections/hybrid calls connector.create_hybrid_collection with correct args."""
    app, mock_connector = _build_test_app()
    client = TestClient(app, raise_server_exceptions=True)

    resp = client.post(
        "/collections/hybrid",
        json={
            "collection_name": "my_hybrid",
            "embedding_model": "Qwen/Qwen3-Embedding-4B",
            "distance": "cosine",
        },
    )

    assert resp.status_code == 201, resp.text
    data = resp.json()
    assert data["collection_name"] == "my_hybrid"
    assert data["sparse_vector_name"] == "sparse-bm25"
    assert data["dense_vector_name"] == "qwen3-qwen3-embedding-4b"
    mock_connector.create_hybrid_collection.assert_called_once()


def test_collections_hybrid_route_not_swallowed_by_parameterized_route():
    """
    POST /collections/hybrid must not be caught by DELETE /collections/{name}.

    The two routes share the path pattern but differ on HTTP method —
    FastAPI routes by (method, path). Verify that a POST to /collections/hybrid
    reaches our handler (201) and not the DELETE handler (which would 405).
    """
    app, _ = _build_test_app()
    client = TestClient(app, raise_server_exceptions=True)

    # If the route were swallowed by the DELETE handler, this would return 405.
    resp = client.post(
        "/collections/hybrid",
        json={"collection_name": "test_col", "embedding_model": "Qwen/Qwen3-Embedding-4B"},
    )
    assert resp.status_code == 201, (
        f"Expected 201 from POST /collections/hybrid, got {resp.status_code}: {resp.text}"
    )


def test_set_active_model_updates_health_provider():
    """POST /embedding_models/active updates the REST app's active provider."""
    app, mock_connector = _build_test_app()
    client = TestClient(app, raise_server_exceptions=True)

    # Initially the health endpoint should report the default 4B model
    assert client.get("/health").json()["embedding_model"] == "Qwen/Qwen3-Embedding-4B"

    # Switch to 8B
    resp = client.post(
        "/embedding_models/active",
        json={"model_name": "Qwen/Qwen3-Embedding-8B"},
    )

    assert resp.status_code == 200, resp.text
    assert resp.json()["active_model"] == "Qwen/Qwen3-Embedding-8B"
    # Health endpoint must reflect the switch — not the startup default
    assert client.get("/health").json()["embedding_model"] == "Qwen/Qwen3-Embedding-8B"
    mock_connector.set_embedding_provider.assert_called_once()


# ---------------------------------------------------------------------------
# search_documents mode tests
# ---------------------------------------------------------------------------

_FAKE_DOCS = [
    {
        "document_id": "doc-adler-001",
        "filename": "socratic_circles.pdf",
        "path": "/books/socratic_circles.pdf",
        "score": 0.92,
        "chunks": [
            {
                "content": "Adler wrote about raising their minds up from a state of understanding.",
                "score": 0.92,
                "chunk_index": 42,
                "metadata": {"page": 18},
            }
        ],
    }
]


def _build_search_test_app():
    """Extend _build_test_app with get_detailed_collection_info mocked for search endpoints."""
    app, mock_connector = _build_test_app()
    mock_connector.get_detailed_collection_info = AsyncMock(
        return_value=CollectionInfo(name="socratic_circles_qwen3_4b", vector_size=2560)
    )
    return app, mock_connector


@pytest.mark.parametrize(
    "mode,expect_sparse,expect_reranker",
    [
        ("dense", False, False),
        ("hybrid", True, False),
        ("rerank", True, True),
    ],
)
def test_search_documents_modes_dense_hybrid_rerank(mode, expect_sparse, expect_reranker):
    """search_documents routes dense/hybrid/rerank modes with the correct providers."""
    app, _ = _build_search_test_app()
    client = TestClient(app, raise_server_exceptions=True)

    with (
        patch(
            "raggy_mcp.search.document_search.search_documents_grouped",
            new_callable=AsyncMock,
            return_value=_FAKE_DOCS,
        ) as mock_search,
        patch(
            "raggy_mcp.webui.api.SparseEmbeddingProvider",
        ) as mock_sparse_cls,
        patch(
            "raggy_mcp.webui.api.build_default_reranker",
            return_value=MagicMock(),
        ) as mock_reranker_builder,
    ):
        resp = client.post(
            "/search_documents",
            json={"query": "adler", "collection_name": "socratic_circles_qwen3_4b", "mode": mode},
        )

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["documents"] == _FAKE_DOCS

    mock_search.assert_awaited_once()
    _, kwargs = mock_search.call_args

    if expect_sparse:
        mock_sparse_cls.assert_called_once()
        assert kwargs["sparse_provider"] is not None
    else:
        mock_sparse_cls.assert_not_called()
        assert kwargs["sparse_provider"] is None

    if expect_reranker:
        mock_reranker_builder.assert_called_once()
        assert kwargs["reranker"] is not None
    else:
        mock_reranker_builder.assert_not_called()
        assert kwargs["reranker"] is None

    assert kwargs["late_interaction_provider"] is None


def test_search_documents_late_interaction_mode():
    """search_documents with mode=late_interaction passes a LateInteractionEmbeddingProvider."""
    app, _ = _build_search_test_app()
    client = TestClient(app, raise_server_exceptions=True)

    mock_li_provider = MagicMock()

    with (
        patch(
            "raggy_mcp.search.document_search.search_documents_grouped",
            new_callable=AsyncMock,
            return_value=_FAKE_DOCS,
        ) as mock_search,
        patch(
            "raggy_mcp.embeddings.late_interaction.LateInteractionEmbeddingProvider",
            return_value=mock_li_provider,
        ),
    ):
        resp = client.post(
            "/search_documents",
            json={
                "query": "adler",
                "collection_name": "socratic_circles_qwen3_4b",
                "mode": "late_interaction",
            },
        )

    assert resp.status_code == 200, resp.text
    mock_search.assert_awaited_once()
    _, kwargs = mock_search.call_args
    assert kwargs["sparse_provider"] is None
    assert kwargs["reranker"] is None
    assert kwargs["late_interaction_provider"] is mock_li_provider


def test_search_documents_vector_size_mismatch_rejected():
    """search_documents returns 400 when the active provider dimensions don't match the collection."""
    app, mock_connector = _build_test_app()
    # Collection expects 4096D (8B), but active provider is 2560D (4B)
    mock_connector.get_detailed_collection_info = AsyncMock(
        return_value=CollectionInfo(name="wrong_collection", vector_size=4096)
    )
    client = TestClient(app, raise_server_exceptions=True)

    resp = client.post(
        "/search_documents",
        json={"query": "adler", "collection_name": "wrong_collection", "mode": "dense"},
    )

    assert resp.status_code == 400
    assert "4096" in resp.text and "2560" in resp.text


def test_search_documents_missing_collection_returns_404():
    """search_documents returns 404 when the collection does not exist."""
    app, mock_connector = _build_test_app()
    mock_connector.get_detailed_collection_info = AsyncMock(return_value=None)
    client = TestClient(app, raise_server_exceptions=True)

    resp = client.post(
        "/search_documents",
        json={"query": "adler", "collection_name": "no_such_collection", "mode": "dense"},
    )

    assert resp.status_code == 404
