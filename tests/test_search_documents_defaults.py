from raggy_mcp.qdrant import Entry
from raggy_mcp.search.document_search import search_documents_grouped


class FakeConnector:
    async def hybrid_search(self, **kwargs):
        return [
            (
                Entry(
                    content=f"chunk {index}",
                    metadata={
                        "document_id": "doc-1",
                        "filename": "book.pdf",
                        "path": "/tmp/book.pdf",
                        "chunk_index": index,
                    },
                ),
                1.0 - index / 10,
            )
            for index in range(6)
        ]


async def test_search_documents_default_surfaces_multiple_chunks():
    docs = await search_documents_grouped(
        FakeConnector(),
        query="student improvements",
        collection_name="docs",
        limit=1,
    )

    assert len(docs) == 1
    assert [chunk["content"] for chunk in docs[0]["chunks"]] == [
        "chunk 0",
        "chunk 1",
        "chunk 2",
    ]
