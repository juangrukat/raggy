"""Diagnose memory during MCP server init + search (in-process, STDIO mode)."""
import os, time, psutil, asyncio

def snap(label):
    proc = psutil.Process()
    py_rss = proc.memory_info().rss / 1024**2
    vm = psutil.virtual_memory()
    q_rss = 0
    for p in psutil.process_iter(['name','memory_info']):
        try:
            if 'qdrant' in p.info['name'].lower():
                q_rss = p.info['memory_info'].rss / 1024**2
        except: pass
    print(f"[{label:30s}] py={py_rss:.0f}MB qdrant={q_rss:.0f}MB sys={vm.used/1024**3:.1f}/{vm.total/1024**3:.1f}GB")
    return py_rss, q_rss

async def main():
    snap("baseline")

    # Simulate server startup (same path as server.py)
    from raggy_mcp._warnings import filter_upstream_warnings
    filter_upstream_warnings()

    from raggy_mcp.config import load_qdrant_config
    load_qdrant_config()
    snap("after_config_load")

    from raggy_mcp.settings import EmbeddingProviderSettings, QdrantSettings, ToolSettings
    qs = QdrantSettings()
    es = EmbeddingProviderSettings()
    ts = ToolSettings()
    snap("after_settings")

    from raggy_mcp.embeddings.factory import create_embedding_provider
    pp = create_embedding_provider(es)
    snap("after_embedding_provider")

    from raggy_mcp.qdrant import QdrantConnector
    conn = QdrantConnector(
        qdrant_url=qs.location,
        qdrant_api_key=qs.api_key,
        collection_name=qs.collection_name,
        embedding_provider=pp,
        qdrant_local_path=qs.local_path,
    )
    snap("after_connector_init")

    # Warm up the embedding provider
    if callable(getattr(pp, "warm_up", None)):
        try:
            await pp.warm_up()
        except: pass
    snap("after_warmup")

    # Run the search
    from raggy_mcp.search.document_search import search_documents_grouped
    for i in range(3):
        results = await search_documents_grouped(
            conn,
            query="what are some things adler said",
            collection_name="socratic_circles_hybrid_v2",
            limit=5, chunks_per_document=4,
            embedding_provider=pp,
        )
        snap(f"after_search_{i+1}")
        print(f"  Search {i+1}: {len(results)} results")

    snap("final")

if __name__ == "__main__":
    asyncio.run(main())
