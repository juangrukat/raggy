"""
Benchmark all 4 retrieval modes against the Qdrant server.
10 warm runs per mode. Cold start measured separately.
Reports per-mode averages with min/max/stddev.
"""

import asyncio
import json
import statistics
import time

from raggy_mcp.embeddings.factory import create_embedding_provider
from raggy_mcp.embeddings.late_interaction import (
    DEFAULT_LATE_INTERACTION_MODEL,
    LateInteractionEmbeddingProvider,
)
from raggy_mcp.embeddings.sparse import SparseEmbeddingProvider
from raggy_mcp.qdrant import QdrantConnector
from raggy_mcp.search.document_search import search_documents_grouped
from raggy_mcp.search.reranker import build_default_reranker
from raggy_mcp.settings import EmbeddingProviderSettings, QdrantSettings

QUERY = (
    "Socratic Circles, does the author say that students gain more knowledge "
    "or understanding from using Socratic circles? If so, what kinds of "
    "knowledge, skills, or learning improvements does he describe? Please "
    "also explain what Adler contributes to this idea, especially his view "
    "of education and Socratic discussion."
)

MODES = {
    "dense": {"collection": "King", "sparse": False, "rerank": False, "late": False},
    "hybrid": {
        "collection": "socratic_circles_hybrid_v2",
        "sparse": True,
        "rerank": False,
        "late": False,
    },
    "rerank": {
        "collection": "socratic_circles_hybrid_v2",
        "sparse": True,
        "rerank": True,
        "late": False,
    },
    "late_interaction": {
        "collection": "socratic_circles_li",
        "sparse": False,
        "rerank": False,
        "late": True,
    },
}

WARM_RUNS = 10  # after discarding cold start

qs = QdrantSettings()
es = EmbeddingProviderSettings()
ep = create_embedding_provider(es)


def fmt(n, unit="ms"):
    return round(n, 1) if unit == "ms" else f"{n:.3f}s"


async def run_one(connector, mode_cfg, sparse_provider, li_provider, reranker):
    """Run a single search and return elapsed wall-clock seconds."""
    t0 = time.perf_counter()
    await search_documents_grouped(
        connector,
        query=QUERY,
        collection_name=mode_cfg["collection"],
        limit=5,
        chunks_per_document=2,
        sparse_provider=sparse_provider,
        late_interaction_provider=li_provider,
        reranker=reranker,
        embedding_provider=ep,
        prefetch_limit=40,
    )
    return time.perf_counter() - t0


async def benchmark_mode(name, mode_cfg):
    connector = QdrantConnector(qs.location, qs.api_key, None, ep, qs.local_path)

    sparse = SparseEmbeddingProvider("Qdrant/bm25") if mode_cfg["sparse"] else None
    li = (
        LateInteractionEmbeddingProvider(DEFAULT_LATE_INTERACTION_MODEL)
        if mode_cfg["late"]
        else None
    )
    reranker = None
    if mode_cfg["rerank"]:
        reranker = build_default_reranker(qs.default_reranker_model)

    total_runs = WARM_RUNS + 1  # +1 for cold start
    times = []

    for i in range(total_runs):
        elapsed = await run_one(connector, mode_cfg, sparse, li, reranker)
        times.append(elapsed)
        label = "COLD" if i == 0 else f"warm {i}"
        print(f"  {name:20s} {label:8s}  {elapsed:.3f}s", flush=True)

    cold = times[0]
    warm = times[1:]

    return {
        "mode": name,
        "collection": mode_cfg["collection"],
        "cold_start_s": round(cold, 3),
        "warm_avg_s": round(statistics.mean(warm), 3),
        "warm_min_s": round(min(warm), 3),
        "warm_max_s": round(max(warm), 3),
        "warm_std_s": round(statistics.stdev(warm), 3) if len(warm) > 1 else 0,
        "warm_runs": len(warm),
    }


async def main():
    print(f"Qdrant: {qs.location}")
    print(f"Model:  {ep.get_model_name()} ({ep.get_vector_size()}D, {es.device})")
    print(f"Query:  {QUERY[:80]}...")
    print(f"Runs:   {WARM_RUNS} warm (+1 cold start discarded)")
    print()
    print("=" * 60)

    results = []
    for name, cfg in MODES.items():
        print(f"\n--- {name} ({cfg['collection']}) ---")
        result = await benchmark_mode(name, cfg)
        results.append(result)

    print("\n" + "=" * 60)
    print(
        f"{'Mode':20s} {'Cold':>8s} {'Warm Avg':>8s} {'Min':>8s} {'Max':>8s}  Collection"
    )
    print("-" * 70)
    for r in results:
        print(
            f"{r['mode']:20s} "
            f"{r['cold_start_s']:>7.3f}s "
            f"{r['warm_avg_s']:>7.3f}s "
            f"{r['warm_min_s']:>7.3f}s "
            f"{r['warm_max_s']:>7.3f}s  "
            f"{r['collection']}"
        )

    # JSON output for machine consumption
    print("\n--- JSON ---")
    print(json.dumps(results, indent=2))


asyncio.run(main())
