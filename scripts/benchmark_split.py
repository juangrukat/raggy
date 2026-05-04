"""
Split-timing benchmark: isolate encoder vs index latency.
Runs same query 10x, then 10x with cached embedding to measure pure Qdrant time.
"""
import asyncio, statistics, time, os
os.environ.pop('QDRANT_LOCAL_PATH', None)
from raggy_mcp.settings import QdrantSettings, EmbeddingProviderSettings
from raggy_mcp.qdrant import QdrantConnector
from raggy_mcp.embeddings.factory import create_embedding_provider

QUERY = (
    "Socratic Circles, does the author say that students gain more knowledge or understanding "
    "from using Socratic circles? If so, what kinds of knowledge, skills, or learning "
    "improvements does he describe?"
)

qs = QdrantSettings(); es = EmbeddingProviderSettings()
ep = create_embedding_provider(es)

async def search_with_embedding(conn, query, collection):
    t0 = time.perf_counter()
    emb_t0 = time.perf_counter()
    vec = await ep.embed_query(query)
    emb_ms = (time.perf_counter() - emb_t0) * 1000

    idx_t0 = time.perf_counter()
    from qdrant_client import models
    results = await conn._client.query_points(
        collection_name=collection,
        query=vec,
        using=ep.get_vector_name(),
        limit=5,
        with_payload=["document"],
        with_vectors=False,
    )
    idx_ms = (time.perf_counter() - idx_t0) * 1000

    total_ms = (time.perf_counter() - t0) * 1000
    return total_ms, emb_ms, idx_ms, len(results.points)

async def main():
    conn = QdrantConnector(qs.location, qs.api_key, None, ep, qs.local_path)

    print("=== Pipeline: embed + search (10 runs) ===")
    pipeline_times = []
    for i in range(11):
        total, emb, idx, n = await search_with_embedding(conn, QUERY, "King")
        label = "COLD" if i == 0 else f"warm {i}"
        pipeline_times.append((total, emb, idx))
        if i > 0:
            print(f"  {label:8s}  total={total:6.1f}ms  embed={emb:6.1f}ms  qdrant={idx:5.1f}ms  ({n} hits)", flush=True)

    warm = pipeline_times[1:]
    avg_total = statistics.mean(t[0] for t in warm)
    avg_embed = statistics.mean(t[1] for t in warm)
    avg_idx   = statistics.mean(t[2] for t in warm)

    print(f"\n  Pipeline warm avg: {avg_total:.0f}ms total = {avg_embed:.0f}ms embed + {avg_idx:.0f}ms qdrant")
    print(f"  Encoder share: {avg_embed/avg_total*100:.0f}%")

    # Now cache the embedding and measure pure Qdrant
    vec = await ep.embed_query(QUERY)
    print(f"\n=== Index only: pre-cached embedding (10 runs) ===")
    idx_times = []
    for i in range(10):
        t0 = time.perf_counter()
        results = await conn._client.query_points(
            collection_name="King", query=vec,
            using=ep.get_vector_name(), limit=5,
            with_payload=["document"], with_vectors=False)
        ms = (time.perf_counter() - t0) * 1000
        idx_times.append(ms)
        print(f"  run {i+1:2d}  qdrant={ms:5.1f}ms  ({len(results.points)} hits)", flush=True)

    print(f"\n  Index-only avg: {statistics.mean(idx_times):.0f}ms  "
          f"(min={min(idx_times):.0f} max={max(idx_times):.0f})")

    # Summary
    print(f"\n=== Split ===")
    print(f"  Encoder (Qwen3-4B Metal f16):  {avg_embed:.0f}ms")
    print(f"  Qdrant index (query+group):    {avg_idx:.0f}ms")
    print(f"  Pipeline overhead (grouping):  {avg_total - avg_embed - avg_idx:.0f}ms")
    print(f"  Total pipeline:                {avg_total:.0f}ms")

asyncio.run(main())
