from raggy_mcp.embeddings.base import EmbeddingProvider
from raggy_mcp.mcp_runtime.provider_resolver import ProviderResolver


class FakeProvider(EmbeddingProvider):
    def __init__(self, model_name: str = "default-model"):
        self.model_name = model_name

    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in documents]

    async def embed_query(self, query: str) -> list[float]:
        return [1.0, 0.0]

    def get_vector_name(self) -> str:
        return self.model_name.replace("/", "_")

    def get_vector_size(self) -> int:
        return 2

    def get_model_name(self) -> str:
        return self.model_name


class FakeManager:
    def get_model_info(self, model_name: str):
        return object()

    def create_provider_for_model(self, model_name: str):
        return FakeProvider(model_name)


async def test_collection_model_assignment_persists(tmp_path):
    resolver = ProviderResolver(FakeManager(), FakeProvider(), storage_root=tmp_path)

    await resolver.assign_collection_model_persisted("docs", "custom-model")

    assert resolver.collection_model("docs") == "custom-model"
    assert (tmp_path / "collection_models.json").exists()

    reloaded = ProviderResolver(FakeManager(), FakeProvider(), storage_root=tmp_path)
    assert reloaded.collection_model("docs") == "custom-model"
    provider = await reloaded.resolve(collection_name="docs")
    assert provider.get_model_name() == "custom-model"
