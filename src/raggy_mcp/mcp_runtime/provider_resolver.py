"""
Per-request embedding provider resolution.

When a single MCP server is shared by multiple agents over Streamable HTTP,
no client should be able to silently swap the active embedding provider out
from under another client. This module provides a thread-safe resolver that
returns providers without mutating shared state.

Resolution order, highest priority first:
  1. explicit ``embedding_model`` argument supplied by the caller;
  2. collection-level configuration recorded by
     ``set_collection_embedding_model``;
  3. the server's default provider (immutable for the process lifetime).

Providers themselves are stateless from a request's perspective — fastembed
loads the ONNX runtime lazily but reads are safe to share across coroutines.
We cache providers by ``(provider_type, model_name)`` so we don't re-load
ONNX models for every request.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from raggy_mcp.embedding_manager import EnhancedEmbeddingModelManager
from raggy_mcp.embeddings.base import EmbeddingProvider
from raggy_mcp.settings import DEFAULT_LOCAL_STORAGE_PATH

logger = logging.getLogger(__name__)


class ProviderResolver:
    """Resolve and cache dense embedding providers without mutating shared state."""

    def __init__(
        self,
        manager: EnhancedEmbeddingModelManager,
        default: EmbeddingProvider,
        storage_root: str | os.PathLike[str] | None = None,
    ):
        self._manager = manager
        self._default = default
        self._cache: dict[str, EmbeddingProvider] = {}
        # Seed the cache with the default provider so the lookup path is uniform
        self._cache[default.get_model_name()] = default
        self._lock = asyncio.Lock()
        # Per-collection model assignments (set by set_collection_embedding_model)
        self._collection_models: dict[str, str] = {}
        self._storage_root = Path(storage_root or DEFAULT_LOCAL_STORAGE_PATH)
        self._collection_models_path = self._storage_root / "collection_models.json"
        self._collection_models_file_lock = asyncio.Lock()
        self._load_collection_models()

    @property
    def default_provider(self) -> EmbeddingProvider:
        """The server's startup-time default. Immutable for the process lifetime."""
        return self._default

    def assign_collection_model(self, collection_name: str, model_name: str) -> None:
        """Record that a collection should use this embedding model by default."""
        self._collection_models[collection_name] = model_name

    async def assign_collection_model_persisted(
        self, collection_name: str, model_name: str
    ) -> None:
        """Record and persist a collection→model assignment."""
        self._collection_models[collection_name] = model_name
        await self._save_collection_models()

    def collection_model(self, collection_name: str) -> str | None:
        return self._collection_models.get(collection_name)

    def _load_collection_models(self) -> None:
        if not self._collection_models_path.exists():
            return
        try:
            raw = json.loads(self._collection_models_path.read_text(encoding="utf-8"))
            loaded: dict[str, str] = {}
            for collection, config in raw.items():
                if isinstance(config, dict):
                    model = config.get("embedding_model")
                else:
                    model = config
                if isinstance(collection, str) and isinstance(model, str) and model:
                    loaded[collection] = model
            self._collection_models.update(loaded)
        except Exception as e:
            logger.warning(
                f"Could not load collection model assignments from {self._collection_models_path}: {e}"
            )

    async def _save_collection_models(self) -> None:
        async with self._collection_models_file_lock:
            self._storage_root.mkdir(parents=True, exist_ok=True)
            now = datetime.now(timezone.utc).isoformat()
            payload = {
                collection: {"embedding_model": model_name, "updated_at": now}
                for collection, model_name in sorted(self._collection_models.items())
            }
            tmp_path = self._collection_models_path.with_suffix(".json.tmp")
            tmp_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            tmp_path.replace(self._collection_models_path)

    async def resolve(
        self,
        *,
        embedding_model: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> EmbeddingProvider:
        """Pick the right provider for this request without mutating shared state."""
        # 1) explicit override
        if embedding_model:
            return await self._get_or_load(embedding_model)
        # 2) collection assignment
        if collection_name and collection_name in self._collection_models:
            return await self._get_or_load(self._collection_models[collection_name])
        # 3) server default
        return self._default

    async def _get_or_load(self, model_name: str) -> EmbeddingProvider:
        cached = self._cache.get(model_name)
        if cached is not None:
            return cached
        async with self._lock:
            cached = self._cache.get(model_name)
            if cached is not None:
                return cached
            info = self._manager.get_model_info(model_name)
            if info is None:
                raise ValueError(f"Unknown embedding model '{model_name}'.")
            logger.info(f"Loading embedding provider for '{model_name}' (cache miss)")
            provider = self._manager.create_provider_for_model(model_name)
            self._cache[model_name] = provider
            return provider
