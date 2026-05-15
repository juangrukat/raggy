"""Line profiling of raggy-mcp hot paths."""
import os
os.environ["QDRANT_LOCAL_PATH"] = os.environ.get("QDRANT_LOCAL_PATH", "storage")

from raggy_mcp._warnings import filter_upstream_warnings
filter_upstream_warnings()

@profile
def profile_settings():
    from raggy_mcp.settings import (
        EmbeddingProviderSettings, QdrantSettings, ToolSettings,
    )
    for _ in range(5):
        QdrantSettings()
        EmbeddingProviderSettings()
        ToolSettings()

@profile
def profile_mcp_server_init():
    import logging
    logging.disable(logging.CRITICAL)

    from raggy_mcp.qdrant import QdrantConnector
    from raggy_mcp.embeddings.factory import create_embedding_provider
    from raggy_mcp.settings import (
        EmbeddingProviderSettings, QdrantSettings,
    )

    qdrant_settings = QdrantSettings()
    embed_settings = EmbeddingProviderSettings()

    pp = create_embedding_provider(embed_settings)
    conn = QdrantConnector(
        qdrant_url=qdrant_settings.location,
        qdrant_api_key=qdrant_settings.api_key,
        collection_name=qdrant_settings.collection_name,
        embedding_provider=pp,
        qdrant_local_path=qdrant_settings.local_path,
    )
    return conn

if __name__ == "__main__":
    profile_settings()
    profile_mcp_server_init()
    print("Done.")
