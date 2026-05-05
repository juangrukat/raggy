import asyncio
import os
import statistics
import time

# ruff: noqa: E702

os.environ.pop("QDRANT_LOCAL_PATH", None)
from raggy_mcp.embeddings.factory import create_embedding_provider
from raggy_mcp.embeddings.late_interaction import (
    DEFAULT_LATE_INTERACTION_MODEL,
    LateInteractionEmbeddingProvider,
)
from raggy_mcp.qdrant import QdrantConnector
from raggy_mcp.search.document_search import search_documents_grouped
from raggy_mcp.settings import EmbeddingProviderSettings, QdrantSettings

QUERY = (
    "Socratic Circles, does the author say that students gain more knowledge or understanding "
    "from using Socratic circles? If so, what kinds of knowledge, skills, or learning "
    "improvements does he describe? Please also explain what Adler contributes to this idea, "
    "especially his view of education and Socratic discussion."
)

qs = QdrantSettings()
es = EmbeddingProviderSettings()
ep = create_embedding_provider(es)
li = LateInteractionEmbeddingProvider(DEFAULT_LATE_INTERACTION_MODEL)


async def main():
    conn = QdrantConnector(qs.location, qs.api_key, None, ep, qs.local_path)
    times = []
    for i in range(11):
        t0 = time.perf_counter()
        result = await search_documents_grouped(
            conn,
            query=QUERY,
            collection_name="default_li",
            limit=5,
            chunks_per_document=2,
            late_interaction_provider=li,
            embedding_provider=ep,
            prefetch_limit=40,
        )
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        label = "COLD" if i == 0 else f"warm {i}"
        print(
            f"  late_interaction {label:8s} {elapsed:.3f}s ({len(result)} docs)",
            flush=True,
        )
    cold = times[0]
    warm = times[1:]
    print(f"\n  cold:  {cold:.3f}s")
    print(
        f"  warm:  {statistics.mean(warm):.3f}s  "
        f"(min={min(warm):.3f} max={max(warm):.3f} std={statistics.stdev(warm):.3f})"
    )


asyncio.run(main())
