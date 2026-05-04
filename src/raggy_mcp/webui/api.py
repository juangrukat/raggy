"""
FastAPI REST surface for raggy-mcp-enhanced-MacOS.
Wraps the same QdrantConnector + ingest pipeline used by the MCP server,
exposing every operation as a JSON HTTP endpoint with auto-generated OpenAPI docs.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from raggy_mcp.embedding_manager import EnhancedEmbeddingModelManager
from raggy_mcp.embeddings.sparse import SparseEmbeddingProvider
from raggy_mcp.mcp_runtime.write_queue import WriteQueue, WriteQueueFullError
from raggy_mcp.qdrant import BatchEntry, QdrantConnector
from raggy_mcp.search.reranker import build_default_reranker
from raggy_mcp.settings import (
    EmbeddingProviderSettings,
    QdrantSettings,
)


# ---- Request/response models ----

class CollectionCreateRequest(BaseModel):
    collection_name: str
    embedding_model: str
    vector_size: int = 0  # 0 = infer from model
    distance: str = "cosine"


class HybridCollectionCreateRequest(BaseModel):
    collection_name: str
    embedding_model: str
    vector_size: int = 0  # 0 = infer from model
    distance: str = "cosine"
    sparse_model: str = "Qdrant/bm25"  # BM25 by default


class StoreRequest(BaseModel):
    content: str
    collection_name: str
    metadata: Optional[dict] = None
    entry_id: Optional[str] = None


class BatchStoreRequest(BaseModel):
    collection_name: str
    entries: list[dict]


class SearchRequest(BaseModel):
    query: str
    collection_name: str
    limit: int = 10
    min_score: float | None = None
    filter: Optional[dict] = None


class SearchDocumentsRequest(BaseModel):
    query: str
    collection_name: str
    limit: int = 10
    chunks_per_document: int = 4
    filter: Optional[dict] = None
    mode: str = "dense"
    additional_queries: Optional[list[str]] = None
    prefetch_limit: Optional[int] = None
    rerank_top_k: Optional[int] = None
    reranker_model: Optional[str] = None
    reranker_instruction: Optional[str] = None
    late_interaction_model: Optional[str] = None


class IngestFileRequest(BaseModel):
    file_path: str
    collection_name: str
    extra_metadata: Optional[dict] = None


class IngestFolderRequest(BaseModel):
    folder_path: str
    collection_name: str
    recursive: bool = True
    skip_hidden: bool = True
    extra_metadata: Optional[dict] = None


class SetEmbeddingModelRequest(BaseModel):
    model_name: str


# ---- App factory ----

def create_app(
    qdrant_settings: Optional[QdrantSettings] = None,
    embedding_settings: Optional[EmbeddingProviderSettings] = None,
    cors_origins: Optional[list[str]] = None,
) -> FastAPI:
    """Build the FastAPI app, sharing config defaults with the MCP server."""
    qdrant_settings = qdrant_settings or QdrantSettings()
    embedding_settings = embedding_settings or EmbeddingProviderSettings()

    embedding_manager = EnhancedEmbeddingModelManager(embedding_settings)
    embedding_provider = embedding_manager.get_default_provider()
    active_provider = {"provider": embedding_provider}
    _late_interaction_providers: dict[str, object] = {}

    connector = QdrantConnector(
        qdrant_url=qdrant_settings.location,
        qdrant_api_key=qdrant_settings.api_key,
        collection_name=qdrant_settings.collection_name,
        embedding_provider=embedding_provider,
        qdrant_local_path=qdrant_settings.local_path,
    )
    write_queue = WriteQueue(
        max_concurrency=qdrant_settings.write_max_concurrency,
        max_queue_size=qdrant_settings.write_queue_size,
    )

    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        # Warm up the primary embedding provider (pre-starts Qwen3 sidecar).
        provider = active_provider["provider"]
        if callable(getattr(provider, "warm_up", None)):
            try:
                await provider.warm_up()
            except Exception:
                pass  # non-fatal: warmup failure should not prevent startup

        # Pre-download auxiliary model providers in the background so the first
        # query using sparse, rerank, or late_interaction modes doesn't pay the
        # HuggingFace download penalty (can be 1-3 minutes for large models).
        async def _warm_sparse():
            try:
                SparseEmbeddingProvider(qdrant_settings.sparse_model)
            except Exception:
                pass

        async def _warm_reranker():
            try:
                r = build_default_reranker(
                    qdrant_settings.default_reranker_model,
                    instruction=qdrant_settings.reranker_instruction,
                )
                # Force-load the ONNX model if it's a FastEmbedReranker.
                if hasattr(r, "_load"):
                    await asyncio.get_event_loop().run_in_executor(None, r._load)
            except Exception:
                pass

        async def _warm_late_interaction():
            try:
                from raggy_mcp.embeddings.late_interaction import (
                    DEFAULT_LATE_INTERACTION_MODEL,
                    LateInteractionEmbeddingProvider,
                )
                LateInteractionEmbeddingProvider(DEFAULT_LATE_INTERACTION_MODEL)
            except Exception:
                pass

        # Fire and forget — each downloads/caches independently.
        for _fn in (_warm_sparse, _warm_reranker, _warm_late_interaction):
            asyncio.ensure_future(_fn())

        yield

    app = FastAPI(
        title="raggy-mcp-enhanced-MacOS",
        description="REST API for Qdrant + macOS file ingestion. Mirrors the MCP tool surface.",
        version="0.8.0",
        lifespan=_lifespan,
    )

    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # --- Health ---

    @app.get("/health")
    async def health() -> dict:
        queue_stats = await write_queue.stats()
        provider = active_provider["provider"]
        return {
            "status": "ok",
            "embedding_model": provider.get_model_name(),
            "vector_size": provider.get_vector_size(),
            "write_queue": queue_stats.__dict__,
        }

    async def ensure_collection_matches_active_provider(collection_name: str) -> None:
        """Reject searches before loading a mismatched embedding model."""
        info = await connector.get_detailed_collection_info(collection_name)
        if not info:
            raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")

        provider = active_provider["provider"]
        provider_size = provider.get_vector_size()
        if info.vector_size and info.vector_size != provider_size:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Active embedding model '{provider.get_model_name()}' produces "
                    f"{provider_size}D vectors, but collection '{collection_name}' "
                    f"expects {info.vector_size}D vectors. Start the server with the "
                    "matching EMBEDDING_MODEL or call /embedding_models/active first."
                ),
            )

    # --- Collections ---

    @app.get("/collections")
    async def list_collections() -> list[str]:
        return await connector.get_collection_names()

    @app.get("/collections/{name}")
    async def collection_info(name: str) -> dict:
        info = await connector.get_detailed_collection_info(name)
        if not info:
            raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")
        return info.model_dump()

    @app.post("/collections", status_code=201)
    async def create_collection(req: CollectionCreateRequest) -> dict:
        info = embedding_manager.get_model_info(req.embedding_model)
        if not info:
            raise HTTPException(status_code=400, detail=f"Unknown embedding model: {req.embedding_model}")

        size = req.vector_size or info.vector_size
        if req.vector_size and req.vector_size > info.vector_size:
            raise HTTPException(
                status_code=400,
                detail=f"vector_size {req.vector_size} exceeds model max {info.vector_size}",
            )

        # Use a provider matching the requested model so vector name is consistent
        provider = embedding_manager.create_provider_for_model(req.embedding_model)
        success = await connector.create_collection_with_config(
            req.collection_name,
            size,
            req.distance,
            embedding_provider=provider,
        )
        if not success:
            raise HTTPException(status_code=500, detail="Failed to create collection")
        return {
            "collection_name": req.collection_name,
            "vector_size": size,
            "distance": req.distance,
            "embedding_model": req.embedding_model,
        }

    @app.post("/collections/hybrid", status_code=201)
    async def create_hybrid_collection(req: HybridCollectionCreateRequest) -> dict:
        info = embedding_manager.get_model_info(req.embedding_model)
        if not info:
            raise HTTPException(status_code=400, detail=f"Unknown embedding model: {req.embedding_model}")

        size = req.vector_size or info.vector_size
        provider = embedding_manager.create_provider_for_model(req.embedding_model)
        dense_vector_name = provider.get_vector_name()
        sparse_vector_name = f"sparse-bm25"

        success = await connector.create_hybrid_collection(
            req.collection_name,
            dense_size=size,
            dense_vector_name=dense_vector_name,
            sparse_vector_name=sparse_vector_name,
            distance=req.distance,
        )
        if not success:
            raise HTTPException(status_code=500, detail="Failed to create hybrid collection")
        return {
            "collection_name": req.collection_name,
            "vector_size": size,
            "distance": req.distance,
            "embedding_model": req.embedding_model,
            "sparse_model": req.sparse_model,
            "dense_vector_name": dense_vector_name,
            "sparse_vector_name": sparse_vector_name,
        }

    @app.delete("/collections/{name}")
    async def delete_collection(name: str) -> dict:
        ok = await connector.delete_collection(name)
        if not ok:
            raise HTTPException(status_code=500, detail=f"Failed to delete collection '{name}'")
        return {"deleted": name}

    @app.post("/collections/{name}/bootstrap_indexes")
    async def bootstrap_indexes(name: str) -> dict:
        await connector.ensure_macos_metadata_indexes(name)
        return {"collection": name, "indexes_ensured": True}

    # --- Storage ---

    @app.post("/store")
    async def store(req: StoreRequest) -> dict:
        entry = BatchEntry(content=req.content, metadata=req.metadata, id=req.entry_id)
        try:
            n = await write_queue.run(
                "store",
                lambda: connector.batch_store([entry], req.collection_name),
            )
        except WriteQueueFullError as e:
            raise HTTPException(status_code=429, detail=str(e)) from e
        return {"stored": n}

    @app.post("/store_batch")
    async def store_batch(req: BatchStoreRequest) -> dict:
        entries = [
            BatchEntry(
                content=e["content"],
                metadata=e.get("metadata"),
                id=e.get("id"),
            )
            for e in req.entries
            if "content" in e
        ]
        try:
            n = await write_queue.run(
                "store_batch",
                lambda: connector.batch_store(entries, req.collection_name),
            )
        except WriteQueueFullError as e:
            raise HTTPException(status_code=429, detail=str(e)) from e
        return {"stored": n}

    @app.get("/scroll/{name}")
    async def scroll(name: str, limit: int = 50, offset: str | None = None) -> dict:
        entries, next_offset = await connector.scroll_collection(
            collection_name=name,
            limit=limit,
            offset=offset,
        )
        return {
            "entries": [{"content": e.content, "metadata": e.metadata} for e in entries],
            "next_offset": next_offset,
        }

    # --- Search ---

    @app.post("/search")
    async def search(req: SearchRequest) -> dict:
        from raggy_mcp.search.filter_grammar import compile_filter

        await ensure_collection_matches_active_provider(req.collection_name)
        qfilter = compile_filter(req.filter) if req.filter else None
        results = await connector.hybrid_search(
            query=req.query,
            collection_name=req.collection_name,
            limit=req.limit,
            query_filter=qfilter,
            min_score=req.min_score,
        )
        return {
            "results": [
                {
                    "content": entry.content,
                    "metadata": entry.metadata,
                    "score": score,
                }
                for entry, score in results
            ]
        }

    @app.post("/search_documents")
    async def search_documents(req: SearchDocumentsRequest) -> dict:
        from raggy_mcp.search.document_search import search_documents_grouped
        from raggy_mcp.search.filter_grammar import compile_filter
        from raggy_mcp.search.retrieval_mode import RetrievalMode
        from raggy_mcp.qdrant import _retrieval_warnings

        qfilter = compile_filter(req.filter) if req.filter else None
        rmode = RetrievalMode.parse(req.mode)
        if rmode != RetrievalMode.LATE_INTERACTION:
            await ensure_collection_matches_active_provider(req.collection_name)
        use_sparse = rmode in (RetrievalMode.HYBRID, RetrievalMode.RERANK)
        sparse_provider = SparseEmbeddingProvider(qdrant_settings.sparse_model) if use_sparse else None

        reranker = None
        if rmode == RetrievalMode.RERANK:
            reranker_name = req.reranker_model or qdrant_settings.default_reranker_model
            reranker = build_default_reranker(
                reranker_name,
                instruction=req.reranker_instruction or qdrant_settings.reranker_instruction,
            )

        late_interaction_provider = None
        if rmode == RetrievalMode.LATE_INTERACTION:
            from raggy_mcp.embeddings.late_interaction import (
                DEFAULT_LATE_INTERACTION_MODEL,
                LateInteractionEmbeddingProvider,
            )
            li_model = req.late_interaction_model or DEFAULT_LATE_INTERACTION_MODEL
            if li_model not in _late_interaction_providers:
                _late_interaction_providers[li_model] = LateInteractionEmbeddingProvider(li_model)
            late_interaction_provider = _late_interaction_providers[li_model]

        warnings_sink: list[str] = []
        _retrieval_warnings.set(warnings_sink)

        groups = await search_documents_grouped(
            connector,
            query=req.query,
            collection_name=req.collection_name,
            limit=req.limit,
            chunks_per_document=req.chunks_per_document,
            query_filter=qfilter,
            sparse_provider=sparse_provider,
            late_interaction_provider=late_interaction_provider,
            reranker=reranker,
            embedding_provider=active_provider["provider"],
            prefetch_limit=req.prefetch_limit,
            rerank_top_k=req.rerank_top_k,
            additional_queries=req.additional_queries or None,
        )
        return {
            "documents": groups,
            "warnings": warnings_sink or None,
        }

    # --- Ingestion ---

    @app.post("/ingest/file")
    async def ingest_file(req: IngestFileRequest) -> dict:
        from raggy_mcp.ingest.extractor import (
            SUPPORTED_EXTENSIONS, build_chunks, extract_text,
        )
        from raggy_mcp.ingest.macos_metadata import get_macos_metadata
        from raggy_mcp.ingest.document_id import compute_document_id

        path = Path(req.file_path).resolve()
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {req.file_path}")
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {path.suffix}")

        file_meta = get_macos_metadata(str(path))
        doc = extract_text(str(path))
        file_meta.update({
            "has_text": bool(doc.text),
            "extractor_used": doc.extractor_used,
            "char_count": doc.char_count,
            "ingested_at": datetime.now(timezone.utc).isoformat(),
            "document_id": compute_document_id(str(path)),
            "parent_path": str(path.parent),
        })
        if doc.page_count is not None:
            file_meta["page_count"] = doc.page_count
        if req.extra_metadata:
            file_meta.update(req.extra_metadata)

        if not doc.text:
            raise HTTPException(
                status_code=422,
                detail=f"No text extracted: {doc.error or 'empty'}",
            )

        await connector.ensure_macos_metadata_indexes(req.collection_name)
        chunks = build_chunks(doc, file_meta)
        entries = [BatchEntry(content=c.text, metadata=c.metadata) for c in chunks]

        # Auto-detect hybrid vs dense-only collection and route accordingly.
        sparse_vector_name = await connector.get_sparse_vector_name(req.collection_name)
        ingest_mode = "hybrid" if sparse_vector_name else "dense"

        try:
            if ingest_mode == "hybrid":
                from raggy_mcp.embeddings.sparse import SparseEmbeddingProvider
                sparse_provider = SparseEmbeddingProvider(qdrant_settings.sparse_model)
                stored = await write_queue.run(
                    "ingest_file",
                    lambda: connector.batch_store_hybrid(
                        entries, req.collection_name, sparse_provider
                    ),
                )
            else:
                stored = await write_queue.run(
                    "ingest_file",
                    lambda: connector.batch_store(entries, req.collection_name),
                )
        except WriteQueueFullError as e:
            raise HTTPException(status_code=429, detail=str(e)) from e
        return {
            "filename": path.name,
            "chunks_stored": stored,
            "extractor_used": doc.extractor_used,
            "char_count": doc.char_count,
            "page_count": doc.page_count,
            "ingest_mode": ingest_mode,
        }

    @app.post("/ingest/folder")
    async def ingest_folder(req: IngestFolderRequest) -> dict:
        from raggy_mcp.ingest.extractor import (
            SUPPORTED_EXTENSIONS, build_chunks, extract_text,
        )
        from raggy_mcp.ingest.macos_metadata import get_macos_metadata
        from raggy_mcp.ingest.document_id import compute_document_id

        folder = Path(req.folder_path).resolve()
        if not folder.is_dir():
            raise HTTPException(status_code=400, detail=f"Not a directory: {req.folder_path}")

        pattern = "**/*" if req.recursive else "*"
        files = [
            p for p in folder.glob(pattern)
            if p.is_file()
            and p.suffix.lower() in SUPPORTED_EXTENSIONS
            and (not req.skip_hidden or not any(part.startswith(".") for part in p.parts))
        ]

        if not files:
            return {"files_processed": 0, "chunks_stored": 0, "errors": []}

        await connector.ensure_macos_metadata_indexes(req.collection_name)
        ingested_at = datetime.now(timezone.utc).isoformat()

        # Detect hybrid vs dense-only once for the whole folder ingest.
        sparse_vector_name = await connector.get_sparse_vector_name(req.collection_name)
        folder_ingest_mode = "hybrid" if sparse_vector_name else "dense"
        if folder_ingest_mode == "hybrid":
            from raggy_mcp.embeddings.sparse import SparseEmbeddingProvider
            sparse_provider_folder = SparseEmbeddingProvider(qdrant_settings.sparse_model)
        else:
            sparse_provider_folder = None

        total_chunks = 0
        files_done = 0
        errors: list[dict] = []

        for path in sorted(files):
            try:
                file_meta = get_macos_metadata(str(path))
                doc = extract_text(str(path))
                file_meta.update({
                    "has_text": bool(doc.text),
                    "extractor_used": doc.extractor_used,
                    "char_count": doc.char_count,
                    "ingested_at": ingested_at,
                    "document_id": compute_document_id(str(path)),
                    "parent_path": str(path.parent),
                })
                if doc.page_count is not None:
                    file_meta["page_count"] = doc.page_count
                if req.extra_metadata:
                    file_meta.update(req.extra_metadata)

                if not doc.text:
                    errors.append({"file": str(path), "error": doc.error or "empty"})
                    continue

                chunks = build_chunks(doc, file_meta)
                entries = [BatchEntry(content=c.text, metadata=c.metadata) for c in chunks]
                if folder_ingest_mode == "hybrid":
                    _sp = sparse_provider_folder
                    stored = await write_queue.run(
                        "ingest_folder",
                        lambda: connector.batch_store_hybrid(entries, req.collection_name, _sp),
                    )
                else:
                    stored = await write_queue.run(
                        "ingest_folder",
                        lambda: connector.batch_store(entries, req.collection_name),
                    )
                total_chunks += stored
                files_done += 1
            except WriteQueueFullError as e:
                raise HTTPException(status_code=429, detail=str(e)) from e
            except Exception as e:
                errors.append({"file": str(path), "error": str(e)})

        return {
            "files_processed": files_done,
            "files_total": len(files),
            "chunks_stored": total_chunks,
            "errors": errors[:50],
        }

    # --- Embedding model management ---

    @app.get("/embedding_models")
    async def list_embedding_models() -> list[dict]:
        return [m.to_dict() for m in embedding_manager.list_available_models()]

    @app.post("/embedding_models/active")
    async def set_active_model(req: SetEmbeddingModelRequest) -> dict:
        info = embedding_manager.get_model_info(req.model_name)
        if not info:
            raise HTTPException(status_code=400, detail=f"Unknown model: {req.model_name}")
        provider = embedding_manager.create_provider_for_model(req.model_name)
        active_provider["provider"] = provider
        connector.set_embedding_provider(provider)
        return {
            "active_model": req.model_name,
            "vector_size": info.vector_size,
        }

    return app
