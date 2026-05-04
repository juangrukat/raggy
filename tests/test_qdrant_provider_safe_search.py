from types import SimpleNamespace

from raggy_mcp.qdrant import QdrantConnector
from tests.test_provider_resolver import FakeProvider


class FakeClient:
    def __init__(self):
        self.query = None
        self.using = None

    async def collection_exists(self, collection_name: str) -> bool:
        return True

    async def query_points(self, **kwargs):
        self.query = kwargs["query"]
        self.using = kwargs["using"]
        return SimpleNamespace(points=[])


async def test_search_uses_explicit_embedding_provider():
    default = FakeProvider("default")
    override = FakeProvider("override")
    connector = QdrantConnector(":memory:", None, "docs", default)
    fake_client = FakeClient()
    connector._client = fake_client

    await connector.search("hello", embedding_provider=override)

    assert fake_client.using == override.get_vector_name()
    assert fake_client.query == await override.embed_query("hello")
