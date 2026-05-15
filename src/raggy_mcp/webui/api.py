"""FastAPI REST surface for the web UI and lightweight route tests."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from raggy_mcp.config import load_qdrant_config
from raggy_mcp.embedding_manager import EnhancedEmbeddingModelManager
from raggy_mcp.embeddings.late_interaction import DEFAULT_LATE_INTERACTION_MODEL
from raggy_mcp.embeddings.sparse import SparseEmbeddingProvider
from raggy_mcp.mcp_runtime.write_queue import WriteQueue
from raggy_mcp.qdrant import QdrantConnector
from raggy_mcp.search.reranker import build_default_reranker
from raggy_mcp.search.retrieval_mode import RetrievalMode
from raggy_mcp.settings import EmbeddingProviderSettings, QdrantSettings


class HybridCollectionRequest(BaseModel):
    collection_name: str
    embedding_model: str | None = None
    distance: str = "cosine"


class ActiveModelRequest(BaseModel):
    model_name: str


class SearchDocumentsRequest(BaseModel):
    query: str
    collection_name: str
    mode: str = "dense"
    limit: int = 10
    chunks_per_document: int = 3
    embedding_model: str | None = None
    prefetch_limit: int | None = None
    rerank_top_k: int | None = None
    additional_queries: list[str] | None = None
    filter: dict[str, Any] | None = Field(default=None)


def create_app() -> FastAPI:
    """Create the REST API app with its own connector/provider state."""
    load_qdrant_config()
    qdrant_settings = QdrantSettings()
    embedding_settings = EmbeddingProviderSettings()
    manager = EnhancedEmbeddingModelManager(embedding_settings)
    active_provider = manager.get_default_provider()
    connector = QdrantConnector(
        qdrant_url=qdrant_settings.location,
        qdrant_api_key=qdrant_settings.api_key,
        collection_name=qdrant_settings.collection_name,
        embedding_provider=active_provider,
        qdrant_local_path=qdrant_settings.local_path,
    )
    write_queue = WriteQueue(
        max_concurrency=qdrant_settings.write_max_concurrency,
        max_queue_size=qdrant_settings.write_queue_size,
    )
    late_interaction_providers: dict[str, Any] = {}

    app = FastAPI(title="raggy-mcp web API")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        stats = await write_queue.stats()
        return {
            "status": "ok",
            "embedding_model": active_provider.get_model_name(),
            "vector_size": active_provider.get_vector_size(),
            "default_collection": qdrant_settings.collection_name,
            "write_queue": {
                "max_concurrency": stats.max_concurrency,
                "max_queue_size": stats.max_queue_size,
                "running": stats.running,
                "waiting": stats.waiting,
            },
        }

    @app.post("/embedding_models/active")
    async def set_active_model(request: ActiveModelRequest) -> dict[str, Any]:
        nonlocal active_provider
        try:
            active_provider = manager.create_provider_for_model(request.model_name)
            connector.set_embedding_provider(active_provider)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "active_model": active_provider.get_model_name(),
            "vector_size": active_provider.get_vector_size(),
        }

    @app.post("/collections/hybrid", status_code=201)
    async def create_hybrid_collection(
        request: HybridCollectionRequest,
    ) -> dict[str, Any]:
        provider = active_provider
        if request.embedding_model:
            provider = manager.create_provider_for_model(request.embedding_model)
        dense_vector_name = provider.get_vector_name()
        sparse_provider = SparseEmbeddingProvider(qdrant_settings.sparse_model)
        sparse_vector_name = sparse_provider.get_vector_name()
        ok = await connector.create_hybrid_collection(
            collection_name=request.collection_name,
            dense_size=provider.get_vector_size(),
            dense_vector_name=dense_vector_name,
            sparse_vector_name=sparse_vector_name,
            distance=request.distance,
        )
        if not ok:
            raise HTTPException(status_code=500, detail="Failed to create collection")
        await connector.ensure_macos_metadata_indexes(request.collection_name)
        return {
            "collection_name": request.collection_name,
            "dense_vector_name": dense_vector_name,
            "sparse_vector_name": sparse_vector_name,
            "embedding_model": provider.get_model_name(),
            "vector_size": provider.get_vector_size(),
        }

    @app.post("/search_documents")
    async def search_documents(request: SearchDocumentsRequest) -> dict[str, Any]:
        from raggy_mcp.search.document_search import search_documents_grouped
        from raggy_mcp.search.filter_grammar import compile_filter

        info = await connector.get_detailed_collection_info(request.collection_name)
        if info is None:
            raise HTTPException(status_code=404, detail="Collection not found")

        provider = active_provider
        if request.embedding_model:
            provider = manager.create_provider_for_model(request.embedding_model)
        if info.vector_size and info.vector_size != provider.get_vector_size():
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Collection vector size {info.vector_size} does not match "
                    f"embedding provider size {provider.get_vector_size()}."
                ),
            )

        rmode = RetrievalMode.parse(request.mode)
        sparse_provider = (
            SparseEmbeddingProvider(qdrant_settings.sparse_model)
            if rmode in (RetrievalMode.HYBRID, RetrievalMode.RERANK)
            else None
        )
        reranker = (
            build_default_reranker(qdrant_settings.default_reranker_model)
            if rmode == RetrievalMode.RERANK
            else None
        )
        late_interaction_provider = None
        if rmode == RetrievalMode.LATE_INTERACTION:
            late_interaction_provider = late_interaction_providers.get(
                DEFAULT_LATE_INTERACTION_MODEL
            )
            if late_interaction_provider is None:
                from raggy_mcp.embeddings.late_interaction import (
                    LateInteractionEmbeddingProvider,
                )

                late_interaction_provider = LateInteractionEmbeddingProvider(
                    DEFAULT_LATE_INTERACTION_MODEL
                )
                late_interaction_providers[DEFAULT_LATE_INTERACTION_MODEL] = (
                    late_interaction_provider
                )

        docs = await search_documents_grouped(
            connector,
            query=request.query,
            collection_name=request.collection_name,
            limit=request.limit,
            chunks_per_document=request.chunks_per_document,
            query_filter=compile_filter(request.filter) if request.filter else None,
            sparse_provider=sparse_provider,
            late_interaction_provider=late_interaction_provider,
            reranker=reranker,
            embedding_provider=provider,
            prefetch_limit=request.prefetch_limit,
            rerank_top_k=request.rerank_top_k,
            additional_queries=request.additional_queries,
        )
        return {"query": request.query, "mode": request.mode, "documents": docs}

    return app
