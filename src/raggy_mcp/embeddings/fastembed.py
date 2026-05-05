from fastembed import TextEmbedding
from fastembed.common.model_description import DenseModelDescription

from raggy_mcp.common.embed_utils import embed_in_executor, embed_query_in_executor
from raggy_mcp.embeddings.base import EmbeddingProvider

# Fallback dimensions for models that may not appear in fastembed's registry
KNOWN_MODEL_DIMS: dict[str, int] = {
    "Qwen/Qwen3-Embedding-8B": 4096,
    "Qwen/Qwen3-Embedding-4B": 2560,
    "Qwen/Qwen3-Embedding-0.6B": 1024,
}


class FastEmbedProvider(EmbeddingProvider):
    """
    FastEmbed implementation of the embedding provider.
    :param model_name: The name of the FastEmbed model to use.
    """

    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.embedding_model = TextEmbedding(model_name, device=device)

    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        return await embed_in_executor(self.embedding_model.passage_embed, documents)

    async def embed_query(self, query: str) -> list[float]:
        return await embed_query_in_executor(self.embedding_model.query_embed, query)

    def get_vector_name(self) -> str:
        model_name = self.embedding_model.model_name.split("/")[-1].lower()
        return f"fast-{model_name}"

    def get_vector_size(self) -> int:
        try:
            model_description: DenseModelDescription = (
                self.embedding_model._get_model_description(self.model_name)
            )
            if model_description.dim is not None:
                return model_description.dim
        except Exception:
            pass
        if self.model_name in KNOWN_MODEL_DIMS:
            return KNOWN_MODEL_DIMS[self.model_name]
        raise ValueError(f"Cannot determine vector size for model: {self.model_name}")

    def get_model_name(self) -> str:
        return self.model_name
