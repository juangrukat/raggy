from types import SimpleNamespace

import pytest
from qdrant_client import models

from raggy_mcp.qdrant import BatchEntry, QdrantConnector
from raggy_mcp.search.document_search import search_documents_grouped


class FakeDenseProvider:
    async def embed_documents(self, documents):
        return [[0.1, 0.2] for _ in documents]

    async def embed_query(self, query):
        return [0.1, 0.2]

    def get_vector_name(self):
        return "dense"

    def get_vector_size(self):
        return 2

    def get_model_name(self):
        return "fake-dense"


class FakeLateInteractionProvider:
    async def embed_documents(self, documents):
        return [
            [[float(i + 1), 0.0], [0.0, float(i + 1)]] for i, _ in enumerate(documents)
        ]

    async def embed_query(self, query):
        return [[1.0, 0.0], [0.0, 1.0]]

    def get_vector_name(self):
        return "colbert"

    def get_vector_size(self):
        return 2

    def get_model_name(self):
        return "fake-colbert"


class FakeQdrantClient:
    def __init__(self):
        self.exists = False
        self.created = None
        self.upserted = None
        self.query = None

    async def collection_exists(self, collection_name):
        return self.exists

    async def create_collection(self, **kwargs):
        self.exists = True
        self.created = kwargs

    async def create_payload_index(self, **kwargs):
        return None

    async def upsert(self, **kwargs):
        self.upserted = kwargs

    async def query_points(self, **kwargs):
        self.query = kwargs
        point = SimpleNamespace(
            payload={
                "document": "alpha beta",
                "metadata": {"document_id": "doc-1", "filename": "a.txt"},
            },
            score=0.91,
        )
        return SimpleNamespace(points=[point])


@pytest.mark.asyncio
async def test_late_interaction_collection_uses_qdrant_multivectors():
    connector = QdrantConnector(":memory:", None, "docs", FakeDenseProvider())
    fake_client = FakeQdrantClient()
    connector._client = fake_client

    ok = await connector.create_late_interaction_collection(
        collection_name="docs_colbert",
        vector_size=2,
        vector_name="colbert",
    )

    assert ok is True
    config = fake_client.created["vectors_config"]["colbert"]
    assert config.size == 2
    assert config.multivector_config.comparator == models.MultiVectorComparator.MAX_SIM
    assert config.hnsw_config.m == 0


@pytest.mark.asyncio
async def test_late_interaction_store_and_search_use_2d_vectors():
    connector = QdrantConnector(":memory:", None, "docs", FakeDenseProvider())
    fake_client = FakeQdrantClient()
    connector._client = fake_client
    provider = FakeLateInteractionProvider()

    stored = await connector.batch_store_late_interaction(
        [BatchEntry(content="alpha beta", metadata={"document_id": "doc-1"})],
        "docs_colbert",
        provider,
    )
    results = await connector.search_late_interaction(
        "alpha",
        "docs_colbert",
        provider,
        limit=5,
    )

    assert stored == 1
    assert fake_client.upserted["points"][0].vector["colbert"] == [
        [1.0, 0.0],
        [0.0, 1.0],
    ]
    assert fake_client.query["using"] == "colbert"
    assert fake_client.query["query"] == [[1.0, 0.0], [0.0, 1.0]]
    assert results[0][0].content == "alpha beta"
    assert results[0][1] == 0.91


@pytest.mark.asyncio
async def test_grouped_search_routes_late_interaction_provider():
    connector = QdrantConnector(":memory:", None, "docs", FakeDenseProvider())
    fake_client = FakeQdrantClient()
    fake_client.exists = True
    connector._client = fake_client

    docs = await search_documents_grouped(
        connector,
        query="alpha",
        collection_name="docs_colbert",
        late_interaction_provider=FakeLateInteractionProvider(),
    )

    assert docs[0]["document_id"] == "doc-1"
    assert fake_client.query["using"] == "colbert"
