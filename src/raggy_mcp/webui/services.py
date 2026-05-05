"""Service layer — singleton connectors shared across Gradio tabs."""

import asyncio
import os
from pathlib import Path
from typing import Any

from raggy_mcp.embedding_manager import EnhancedEmbeddingModelManager
from raggy_mcp.embeddings.late_interaction import (
    DEFAULT_LATE_INTERACTION_MODEL,
    LateInteractionEmbeddingProvider,
)
from raggy_mcp.embeddings.sparse import SparseEmbeddingProvider
from raggy_mcp.mcp_runtime.write_queue import WriteQueue
from raggy_mcp.qdrant import QdrantConnector
from raggy_mcp.search.reranker import build_default_reranker
from raggy_mcp.settings import EmbeddingProviderSettings, QdrantSettings

# ── Singletons ──
_qdrant: QdrantSettings | None = None
_qdrant_lock: asyncio.Lock = asyncio.Lock()
_embedding: EmbeddingProviderSettings | None = None
_connector: QdrantConnector | None = None
_embedding_manager: EnhancedEmbeddingModelManager | None = None
_embedding_provider: Any = None
_write_queue: WriteQueue | None = None
_late_interaction_providers: dict[str, Any] = {}
_config_data: dict | None = None


def _detect_config_path() -> str | None:
    for p in [
        os.getenv("QDRANT_CONFIG"),
        os.path.join(os.getcwd(), "raggy.yaml"),
        os.path.expanduser("~/.config/raggy-mcp/config.yaml"),
    ]:
        if p and os.path.exists(p):
            return p
    return None


async def init_services() -> None:
    """Initialize all services. Called once at startup."""
    global _qdrant, _embedding, _connector, _embedding_manager, _embedding_provider
    global _write_queue, _config_data, _qdrant_lock

    _qdrant = QdrantSettings()
    _embedding = EmbeddingProviderSettings()

    _embedding_manager = EnhancedEmbeddingModelManager(_embedding)
    _embedding_provider = _embedding_manager.get_default_provider()

    _connector = QdrantConnector(
        qdrant_url=_qdrant.location,
        qdrant_api_key=_qdrant.api_key,
        collection_name=_qdrant.collection_name,
        embedding_provider=_embedding_provider,
        qdrant_local_path=_qdrant.local_path,
    )

    _write_queue = WriteQueue(
        max_concurrency=_qdrant.write_max_concurrency,
        max_queue_size=_qdrant.write_queue_size,
    )

    config_path = _detect_config_path()
    if config_path:
        import yaml

        _config_data = yaml.safe_load(Path(config_path).read_text()) or {}

    global _qdrant_lock
    _qdrant_lock = asyncio.Lock()  # serialize all Qdrant HTTP/2 access

    if callable(getattr(_embedding_provider, "warm_up", None)):
        try:
            await _embedding_provider.warm_up()
        except Exception:
            pass


# ── Accessors ──


def get_qdrant_settings() -> QdrantSettings:
    assert _qdrant is not None
    return _qdrant


def get_embedding_settings() -> EmbeddingProviderSettings:
    assert _embedding is not None
    return _embedding


def get_connector() -> QdrantConnector:
    assert _connector is not None
    return _connector


def get_embedding_manager() -> EnhancedEmbeddingModelManager:
    assert _embedding_manager is not None
    return _embedding_manager


def get_embedding_provider():
    assert _embedding_provider is not None
    return _embedding_provider


def get_write_queue() -> WriteQueue:
    assert _write_queue is not None
    return _write_queue


def get_config_data() -> dict | None:
    return _config_data


async def guarded_qdrant_call(coro):
    """Run a Qdrant coroutine under the lock to serialize HTTP/2 access.

    Use this for ALL await conn.something() calls in Gradio event handlers.
    The lock prevents httpx HTTP/2 multiplexing errors when Gradio
    runs concurrent requests against the shared AsyncQdrantClient.
    """
    async with _qdrant_lock:
        return await coro


async def get_health() -> dict:
    """Collect health/status for dashboard."""
    q = get_qdrant_settings()
    ep = get_embedding_provider()
    wq = get_write_queue()
    stats = await wq.stats()

    mode = "embedded"
    url = None
    if q.location:
        mode = "server"
        url = q.location
    elif q.local_path:
        mode = "embedded"
        url = q.local_path

    collections = []
    try:
        collections = await guarded_qdrant_call(get_connector().get_collection_names())
    except Exception:
        pass

    return {
        "status": "ok",
        "qdrant_mode": mode,
        "qdrant_url": url,
        "embedding_model": ep.get_model_name()
        if hasattr(ep, "get_model_name")
        else str(ep),
        "vector_size": ep.get_vector_size(),
        "tool_profile": q.mcp_tool_profile,
        "read_only": q.read_only,
        "default_collection": q.collection_name,
        "default_search_mode": getattr(q, "default_search_mode", None) or "dense",
        "sparse_model": q.sparse_model,
        "reranker_model": q.default_reranker_model,
        "write_queue": {
            "size": stats.current_size if hasattr(stats, "current_size") else 0,
            "max_concurrency": q.write_max_concurrency,
            "max_queue_size": q.write_queue_size,
        },
        "collection_count": len(collections),
        "collections": collections,
    }


async def search_documents(
    query: str,
    collection_name: str,
    limit: int = 10,
    mode: str | None = None,
    embedding_model: str | None = None,
    prefetch_limit: int | None = None,
    rerank_top_k: int | None = None,
) -> dict:
    """Search with optional system-default fallback for mode and model."""
    from raggy_mcp.qdrant import _retrieval_warnings
    from raggy_mcp.search.document_search import search_documents_grouped
    from raggy_mcp.search.retrieval_mode import RetrievalMode

    conn = get_connector()
    qs = get_qdrant_settings()

    # Resolve mode
    if mode is None:
        mode = getattr(qs, "default_search_mode", None) or "dense"

    rmode = RetrievalMode.parse(mode)
    sparse_provider = None
    if rmode in (RetrievalMode.HYBRID, RetrievalMode.RERANK):
        sparse_provider = SparseEmbeddingProvider(qs.sparse_model)

    reranker = None
    if rmode == RetrievalMode.RERANK:
        reranker = build_default_reranker(qs.default_reranker_model)

    late_interaction_provider = None
    if rmode == RetrievalMode.LATE_INTERACTION:
        li = _late_interaction_providers.get(DEFAULT_LATE_INTERACTION_MODEL)
        if li is None:
            li = LateInteractionEmbeddingProvider(DEFAULT_LATE_INTERACTION_MODEL)
            _late_interaction_providers[DEFAULT_LATE_INTERACTION_MODEL] = li
        late_interaction_provider = li

    # Resolve embedding provider: explicit > default
    provider = get_embedding_provider()
    if embedding_model:
        mgr = get_embedding_manager()
        provider = mgr.create_provider_for_model(embedding_model)

    warnings_sink: list[str] = []
    _retrieval_warnings.set(warnings_sink)

    # Serialized: the actual Qdrant call is the only part that hits httpx
    async with _qdrant_lock:
        docs = await search_documents_grouped(
            conn,
            query=query,
            collection_name=collection_name,
            limit=limit,
            query_filter=None,
            sparse_provider=sparse_provider,
            late_interaction_provider=late_interaction_provider,
            reranker=reranker,
            embedding_provider=provider,
            prefetch_limit=prefetch_limit,
            rerank_top_k=rerank_top_k,
        )

    results = []
    for d in docs:
        results.append(
            {
                "document_id": d["document_id"],
                "path": d.get("path", ""),
                "filename": d.get("filename", ""),
                "score": d["score"],
                "chunks": d["chunks"],
            }
        )

    return {
        "query": query,
        "mode": mode,
        "grouped_by_document": True,
        "results": results,
        "warnings": warnings_sink or [],
    }
