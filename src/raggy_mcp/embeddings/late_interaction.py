"""Late-interaction text embedding provider for Qdrant multivectors."""

import asyncio
from typing import Any

from fastembed import LateInteractionTextEmbedding


DEFAULT_LATE_INTERACTION_MODEL = "colbert-ir/colbertv2.0"
DEFAULT_LATE_INTERACTION_VECTOR_NAME = "colbert"


class LateInteractionEmbeddingProvider:
    """Async wrapper around FastEmbed's ColBERT-style late-interaction models."""

    def __init__(
        self,
        model_name: str = DEFAULT_LATE_INTERACTION_MODEL,
        vector_name: str = DEFAULT_LATE_INTERACTION_VECTOR_NAME,
    ):
        self.model_name = model_name
        self.vector_name = vector_name
        self.embedding_model = LateInteractionTextEmbedding(model_name)

    async def embed_documents(self, documents: list[str]) -> list[list[list[float]]]:
        """Embed passages into per-token multivectors."""
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, lambda: list(self.embedding_model.embed(documents))
        )
        return [embedding.tolist() for embedding in embeddings]

    async def embed_query(self, query: str) -> list[list[float]]:
        """Embed a query into a per-token multivector."""
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, lambda: list(self.embedding_model.query_embed([query]))
        )
        return embeddings[0].tolist()

    def get_vector_name(self) -> str:
        """Qdrant vector field for the multivector token matrix."""
        return self.vector_name

    def get_vector_size(self) -> int:
        """Return the per-token vector dimension."""
        description: Any = self.embedding_model._get_model_description(self.model_name)
        if getattr(description, "dim", None) is not None:
            return description.dim
        raise ValueError(f"Cannot determine late-interaction vector size for {self.model_name}")

    def get_model_name(self) -> str:
        return self.model_name
