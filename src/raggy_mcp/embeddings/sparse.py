"""
Sparse text embedding provider, used for hybrid retrieval.
Defaults to Qdrant/bm25 (lightweight, no neural model download needed).
"""

import asyncio
from typing import Any

from fastembed import SparseTextEmbedding


_DEFAULT_SPARSE_MODEL = "Qdrant/bm25"


class SparseEmbeddingProvider:
    """Wraps fastembed's SparseTextEmbedding with the same async surface as the dense provider."""

    def __init__(self, model_name: str = _DEFAULT_SPARSE_MODEL):
        self.model_name = model_name
        self.embedding_model = SparseTextEmbedding(model_name)

    async def embed_documents(self, documents: list[str]) -> list[dict[str, Any]]:
        """Embed documents into sparse vectors (dict of indices/values per doc)."""
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, lambda: list(self.embedding_model.embed(documents))
        )
        return [self._to_dict(e) for e in embeddings]

    async def embed_query(self, query: str) -> dict[str, Any]:
        """Embed a single query into a sparse vector."""
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, lambda: list(self.embedding_model.query_embed([query]))
        )
        return self._to_dict(embeddings[0])

    @staticmethod
    def _to_dict(sparse_embedding) -> dict[str, Any]:
        """Convert a fastembed SparseEmbedding to {indices, values} suitable for Qdrant SparseVector."""
        return {
            "indices": sparse_embedding.indices.tolist(),
            "values": sparse_embedding.values.tolist(),
        }

    def get_vector_name(self) -> str:
        """Sparse vector name in Qdrant — kept stable so collections can be reopened."""
        suffix = self.model_name.split("/")[-1].lower().replace(".", "-")
        return f"sparse-{suffix}"

    def get_model_name(self) -> str:
        return self.model_name
