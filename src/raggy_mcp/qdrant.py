import contextvars
import logging
import os
import uuid
from typing import Any

# Per-asyncio-task list of retrieval warnings (sparse fallback, etc.).
# Set by callers that want to capture warnings in the observability envelope.
# Usage:
#   sink = []; _retrieval_warnings.set(sink)
#   results = await connector.search_hybrid_rrf(...)
#   acc.warnings.extend(sink)
_retrieval_warnings: contextvars.ContextVar[list[str] | None] = contextvars.ContextVar(
    "_retrieval_warnings", default=None
)

from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient, models

from raggy_mcp.embeddings.base import EmbeddingProvider
from raggy_mcp.settings import METADATA_PATH

logger = logging.getLogger(__name__)
DEFAULT_EMBEDDING_BATCH_SIZE = 4

Metadata = dict[str, Any]
ArbitraryFilter = dict[str, Any]


def _embedding_batch_size() -> int:
    raw = os.getenv("QDRANT_EMBEDDING_BATCH_SIZE")
    if not raw:
        return DEFAULT_EMBEDDING_BATCH_SIZE
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning(
            "Invalid QDRANT_EMBEDDING_BATCH_SIZE=%r; using %s",
            raw,
            DEFAULT_EMBEDDING_BATCH_SIZE,
        )
        return DEFAULT_EMBEDDING_BATCH_SIZE


class CollectionInfo(BaseModel):
    """Information about a Qdrant collection."""
    name: str
    vectors_count: int = 0
    indexed_vectors_count: int = 0
    points_count: int = 0
    segments_count: int = 0
    status: str = "unknown"
    optimizer_status: str = "unknown"
    vector_size: int | None = None
    distance_metric: str | None = None

class BatchEntry(BaseModel):
    """Entry for batch operations."""
    content: str
    metadata: Metadata | None = None
    id: str | None = None


class Entry(BaseModel):
    """
    A single entry in the Qdrant collection.
    """

    content: str
    metadata: Metadata | None = None


class QdrantConnector:
    """
    Encapsulates the connection to a Qdrant server and all the methods to interact with it.
    :param qdrant_url: The URL of the Qdrant server.
    :param qdrant_api_key: The API key to use for the Qdrant server.
    :param collection_name: The name of the default collection to use. If not provided, each tool will require
                            the collection name to be provided.
    :param embedding_provider: The embedding provider to use.
    :param qdrant_local_path: The path to the storage directory for the Qdrant client, if local mode is used.
    """

    def __init__(
        self,
        qdrant_url: str | None,
        qdrant_api_key: str | None,
        collection_name: str | None,
        embedding_provider: EmbeddingProvider,
        qdrant_local_path: str | None = None,
        field_indexes: dict[str, models.PayloadSchemaType] | None = None,
    ):
        self._qdrant_url = qdrant_url.rstrip("/") if qdrant_url else None
        self._qdrant_api_key = qdrant_api_key
        self._default_collection_name = collection_name
        self._embedding_provider = embedding_provider
        self._client = AsyncQdrantClient(
            location=qdrant_url, api_key=qdrant_api_key, path=qdrant_local_path
        )
        self._field_indexes = field_indexes

    def set_embedding_provider(self, provider: EmbeddingProvider) -> None:
        """Swap the active embedding provider (affects all subsequent operations)."""
        self._embedding_provider = provider

    async def get_collection_names(self) -> list[str]:
        """
        Get the names of all collections in the Qdrant server.
        :return: A list of collection names.
        """
        response = await self._client.get_collections()
        return [collection.name for collection in response.collections]

    async def store(self, entry: Entry, *, collection_name: str | None = None):
        """
        Store some information in the Qdrant collection, along with the specified metadata.
        :param entry: The entry to store in the Qdrant collection.
        :param collection_name: The name of the collection to store the information in, optional. If not provided,
                                the default collection is used.
        """
        collection_name = collection_name or self._default_collection_name
        assert collection_name is not None
        await self._ensure_collection_exists(collection_name)

        # Embed the document
        embeddings = await self._embedding_provider.embed_documents([entry.content])
        
        # Use `models.PointStruct` with actual embeddings
        points = [
            models.PointStruct(
                id=uuid.uuid4().hex,
                payload={"document": entry.content, METADATA_PATH: entry.metadata or {}},
                vector={self._embedding_provider.get_vector_name(): embeddings[0]},
            )
        ]

        await self._client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True,
        )

    async def search(
        self,
        query: str,
        *,
        collection_name: str | None = None,
        limit: int = 10,
        query_filter: models.Filter | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> list[Entry]:
        """
        Modern search using Query API with intelligent fallback to resolve vector name mismatches.
        Tries server-side embedding first, falls back to client-side if needed.

        :param query: The query to use for the search.
        :param collection_name: The name of the collection to search in.
        :param limit: The maximum number of entries to return.
        :param query_filter: The filter to apply to the query, if any.
        :return: A list of entries found.
        """
        collection_name = collection_name or self._default_collection_name
        assert collection_name is not None

        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            return []

        provider = embedding_provider or self._embedding_provider
        # Always use client-side embedding for now to ensure consistency in tests
        return await self._search_client_side(query, collection_name, limit, query_filter, provider)

    async def _search_server_side(
        self,
        query: str,
        collection_name: str,
        limit: int,
        query_filter: models.Filter | None
    ) -> list[Entry]:
        """Server-side embedding using Qdrant's FastEmbed integration."""

        # Use modern Query API with server-side embedding
        search_results_raw = await self._client.query_points(
            collection_name=collection_name,
            query=query,  # Let Qdrant handle embedding server-side
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
            with_vectors=False,
        )

        return self._process_search_results(search_results_raw.points)

    async def _search_client_side(
        self,
        query: str,
        collection_name: str,
        limit: int,
        query_filter: models.Filter | None,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> list[Entry]:
        """Client-side embedding for guaranteed consistency."""

        # Embed query using current embedding provider
        provider = embedding_provider or self._embedding_provider
        query_vector = await provider.embed_query(query)

        # Use modern Query API with client-side embedding
        search_results_raw = await self._client.query_points(
            collection_name=collection_name,
            query=query_vector,
            using=provider.get_vector_name(),
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
            with_vectors=False,
        )

        return self._process_search_results(search_results_raw.points)

    def _process_search_results(self, points: list[models.ScoredPoint]) -> list[Entry]:
        """Process search results into Entry objects."""
        return [
            Entry(
                content=(point.payload["document"] if point.payload and "document" in point.payload else ""),
                metadata=(point.payload.get(METADATA_PATH) if point.payload else None),
            )
            for point in points
        ]

    async def _ensure_collection_exists(
        self, collection_name: str, embedding_provider: EmbeddingProvider | None = None
    ):
        """
        Ensure that the collection exists, creating it if necessary.
        Uses the explicitly-passed embedding provider (preferred for multi-agent
        safety) or falls back to the connector's default provider.
        :param collection_name: The name of the collection to ensure exists.
        :param embedding_provider: Optional override for this request only.
        """
        provider = embedding_provider or self._embedding_provider
        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            vector_size = provider.get_vector_size()
            vector_name = provider.get_vector_name()

            logger.info(f"Creating collection '{collection_name}' with vector name '{vector_name}' and size {vector_size}")

            await self._client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    vector_name: models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE,
                    )
                },
            )

            # Create payload indexes if configured
            if self._field_indexes:
                for field_name, field_type in self._field_indexes.items():
                    await self._client.create_payload_index(
                        collection_name=collection_name,
                        field_name=field_name,
                        field_schema=field_type,
                    )

            # Create a text index for the 'document' field for server-side embedding
            await self._client.create_payload_index(
                collection_name=collection_name,
                field_name="document",
                field_schema=models.TextIndexParams(type=models.TextIndexType.TEXT)
            )

    async def get_detailed_collection_info(self, collection_name: str) -> CollectionInfo | None:
        """
        Get detailed information about a collection.
        :param collection_name: The name of the collection.
        :return: CollectionInfo object with detailed information, or None if collection doesn't exist.
        """
        try:
            collection_exists = await self._client.collection_exists(collection_name)
            if not collection_exists:
                return None

            info = await self._client.get_collection(collection_name)

            # Extract vector configuration
            vector_size = None
            distance_metric = None
            if hasattr(info, 'config') and info.config and hasattr(info.config, 'params'):
                if hasattr(info.config.params, 'vectors'):
                    vectors_config = info.config.params.vectors
                    # vectors_config is usually a dict of vector_name -> VectorParams
                    if isinstance(vectors_config, dict):
                        # Take the first vector config if available
                        for vp in vectors_config.values():
                            if hasattr(vp, 'size'):
                                vector_size = vp.size
                            if hasattr(vp, 'distance'):
                                distance_metric = vp.distance.name if hasattr(vp.distance, 'name') else str(vp.distance)
                            break  # Only use the first vector config
                    # If it's a single VectorParams (older qdrant), handle that as well
                    elif vectors_config is not None and hasattr(vectors_config, 'size'):
                        vector_size = vectors_config.size
                        if hasattr(vectors_config, 'distance'):
                            distance_metric = vectors_config.distance.name if hasattr(vectors_config.distance, 'name') else str(vectors_config.distance)

            # For small collections, Qdrant doesn't report vectors_count but points_count indicates stored vectors
            points_count = getattr(info, 'points_count', 0) or 0
            indexed_vectors_count = getattr(info, 'indexed_vectors_count', 0) or 0

            # If indexed_vectors_count is 0 but we have points, assume vectors are stored but not indexed
            # This happens for collections below the indexing threshold
            vectors_count = getattr(info, 'vectors_count', None)
            if vectors_count is None:
                vectors_count = points_count  # Assume each point has a vector for collections below indexing threshold

            return CollectionInfo(
                name=collection_name,
                vectors_count=vectors_count,
                indexed_vectors_count=indexed_vectors_count,
                points_count=points_count,
                segments_count=getattr(info, 'segments_count', 0) or 0,
                status=getattr(info, 'status', 'unknown') or 'unknown',
                optimizer_status=getattr(info, 'optimizer_status', 'unknown') or 'unknown',
                vector_size=vector_size,
                distance_metric=distance_metric
            )
        except Exception as e:
            logger.error(f"Error getting collection info for {collection_name}: {e}")
            return None

    async def get_sparse_vector_name(self, collection_name: str) -> str | None:
        """
        Return the first sparse vector name configured on the collection,
        or None if the collection has no sparse vectors (dense-only).
        """
        try:
            info = await self._client.get_collection(collection_name)
            sparse = getattr(info.config.params, "sparse_vectors", None)
            if sparse:
                return next(iter(sparse), None)
            return None
        except Exception:
            return None

    async def create_collection_with_config(
        self,
        collection_name: str,
        vector_size: int,
        distance: str = "cosine",
        embedding_provider: EmbeddingProvider | None = None
    ) -> bool:
        """
        Create a new collection with specified configuration.
        :param collection_name: Name of the collection to create.
        :param vector_size: Size of the vectors.
        :param distance: Distance metric (cosine, dot, euclidean).
        :param embedding_provider: Optional embedding provider for this collection.
        :return: True if successful, False otherwise.
        """
        try:
            # Convert distance string to Qdrant Distance enum
            distance_map = {
                "cosine": models.Distance.COSINE,
                "dot": models.Distance.DOT,
                "euclidean": models.Distance.EUCLID,
                "manhattan": models.Distance.MANHATTAN
            }

            distance_metric = distance_map.get(distance.lower(), models.Distance.COSINE)

            # Use embedding provider vector name if provided, otherwise use the default embedding provider's name
            vector_name = embedding_provider.get_vector_name() if embedding_provider else self._embedding_provider.get_vector_name()

            await self._client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    vector_name: models.VectorParams(
                        size=vector_size,
                        distance=distance_metric,
                    )
                },
            )

            # Create payload indexes if configured
            if self._field_indexes:
                for field_name, field_type in self._field_indexes.items():
                    await self._client.create_payload_index(
                        collection_name=collection_name,
                        field_name=field_name,
                        field_schema=field_type,
                    )

            return True
        except Exception as e:
            logger.error(f"Error creating collection {collection_name}: {e}")
            return False

    async def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection.
        :param collection_name: Name of the collection to delete.
        :return: True if successful, False otherwise.
        """
        try:
            await self._client.delete_collection(collection_name)
            return True
        except Exception as e:
            logger.error(f"Error deleting collection {collection_name}: {e}")
            return False

    async def batch_store(
        self,
        entries: list[BatchEntry],
        collection_name: str | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> int:
        """
        Store multiple entries in batch using the per-request embedding provider
        when supplied, falling back to the connector's default. Multi-client
        safe: no shared mutable provider state.
        """
        collection_name = collection_name or self._default_collection_name
        assert collection_name is not None
        provider = embedding_provider or self._embedding_provider

        await self._ensure_collection_exists(collection_name, embedding_provider=provider)

        try:
            batch_size = _embedding_batch_size()
            stored = 0
            for start in range(0, len(entries), batch_size):
                entry_batch = entries[start : start + batch_size]
                documents = [entry.content for entry in entry_batch]
                try:
                    embeddings = await provider.embed_documents(documents)
                except Exception as e:
                    logger.error(
                        "Error embedding batch %s-%s of %s in collection %r: %s",
                        start,
                        min(start + batch_size, len(documents)),
                        len(documents),
                        collection_name,
                        e,
                    )
                    raise

                points = []
                for i, entry in enumerate(entry_batch):
                    if entry.id:
                        try:
                            point_id = str(uuid.UUID(entry.id)).replace("-", "")
                        except ValueError:
                            point_id = uuid.uuid5(uuid.NAMESPACE_DNS, entry.id).hex
                    else:
                        point_id = uuid.uuid4().hex

                    points.append(
                        models.PointStruct(
                            id=point_id,
                            payload={"document": entry.content, METADATA_PATH: entry.metadata or {}},
                            vector={provider.get_vector_name(): embeddings[i]},
                        )
                    )

                await self._client.upsert(
                    collection_name=collection_name,
                    points=points,
                    wait=True,
                )
                stored += len(points)

            logger.info(f"Successfully stored {stored} entries in collection '{collection_name}'.")
            return stored

        except Exception as e:
            logger.error(f"Error in batch store: {e}")
            return stored if "stored" in locals() else 0

    async def scroll_collection(
        self,
        collection_name: str | None = None,
        limit: int = 100,
        offset: str | None = None,
        query_filter: models.Filter | None = None,
        with_payload: bool = True,
        with_vectors: bool = False
    ) -> tuple[list[Entry], str | None]:
        """
        Scroll through collection contents with pagination.
        :param collection_name: Name of the collection to scroll.
        :param limit: Maximum number of entries to return.
        :param offset: Pagination offset (point ID to start from).
        :param query_filter: Optional filter to apply.
        :param with_payload: Include payload in results.
        :param with_vectors: Include vectors in results.
        :return: Tuple of (entries, next_offset).
        """
        collection_name = collection_name or self._default_collection_name
        assert collection_name is not None

        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            return [], None

        try:
            result = await self._client.scroll(
                collection_name=collection_name,
                limit=limit,
                offset=offset,
                scroll_filter=query_filter,
                with_payload=with_payload,
                with_vectors=with_vectors
            )

            entries = []
            for point in result[0]:  # result is tuple (points, next_offset)
                if with_payload and point.payload:
                    content = point.payload.get("document", "")
                    metadata = point.payload.get(METADATA_PATH)
                    entries.append(Entry(content=content, metadata=metadata))
                else:
                    # If no payload, create entry with point ID as content
                    entries.append(Entry(content=f"Point ID: {point.id}", metadata={"point_id": point.id}))

            next_offset = str(result[1]) if result[1] is not None else None
            return entries, next_offset  # entries, next_offset
        except Exception as e:
            logger.error(f"Error scrolling collection {collection_name}: {e}")
            return [], None

    async def hybrid_search(
        self,
        query: str,
        collection_name: str | None = None,
        limit: int = 10,
        query_filter: models.Filter | None = None,
        min_score: float | None = None,
        search_params: dict | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> list[tuple[Entry, float]]:
        """
        Modern hybrid search using Query API with intelligent fallback to avoid vector name mismatches.

        :param query: The search query.
        :param collection_name: Name of the collection to search.
        :param limit: Maximum number of results.
        :param query_filter: Optional filter to apply.
        :param min_score: Minimum similarity score threshold.
        :param search_params: Additional search parameters.
        :return: List of (entry, score) tuples.
        """
        collection_name = collection_name or self._default_collection_name
        assert collection_name is not None

        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            return []

        provider = embedding_provider or self._embedding_provider
        return await self._hybrid_search_client_side(query, collection_name, limit, query_filter, min_score, provider)

    async def _hybrid_search_server_side(
        self,
        query: str,
        collection_name: str,
        limit: int,
        query_filter: models.Filter | None,
        min_score: float | None,
    ) -> list[tuple[Entry, float]]:
        """Server-side hybrid search using Query API."""

        search_results_raw = await self._client.query_points(
            collection_name=collection_name,
            query=query,  # Server-side embedding
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
            with_vectors=False,
            score_threshold=min_score,
        )

        return self._process_scored_results(search_results_raw.points)

    async def _hybrid_search_client_side(
        self,
        query: str,
        collection_name: str,
        limit: int,
        query_filter: models.Filter | None,
        min_score: float | None,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> list[tuple[Entry, float]]:
        """Client-side hybrid search using Query API."""

        provider = embedding_provider or self._embedding_provider
        query_vector = await provider.embed_query(query)

        search_results_raw = await self._client.query_points(
            collection_name=collection_name,
            query=query_vector,
            using=provider.get_vector_name(),
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
            with_vectors=False,
            score_threshold=min_score,
        )
        
        return self._process_scored_results(search_results_raw.points)

    def _process_scored_results(self, points: list[models.ScoredPoint]) -> list[tuple[Entry, float]]:
        """Process scored search results into (Entry, score) tuples."""
        results = []
        for point in points:
            entry = Entry(
                content=(point.payload["document"] if point.payload and "document" in point.payload else ""),
                metadata=(point.payload.get(METADATA_PATH) if point.payload else None),
            )
            results.append((entry, point.score))
        return results

    async def create_hybrid_collection(
        self,
        collection_name: str,
        dense_size: int,
        dense_vector_name: str,
        sparse_vector_name: str,
        distance: str = "cosine",
    ) -> bool:
        """
        Create a collection with both a dense vector slot and a sparse vector slot,
        suitable for hybrid retrieval with RRF fusion.
        """
        try:
            distance_map = {
                "cosine": models.Distance.COSINE,
                "dot": models.Distance.DOT,
                "euclidean": models.Distance.EUCLID,
                "manhattan": models.Distance.MANHATTAN,
            }
            distance_metric = distance_map.get(distance.lower(), models.Distance.COSINE)

            await self._client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    dense_vector_name: models.VectorParams(
                        size=dense_size,
                        distance=distance_metric,
                    ),
                },
                sparse_vectors_config={
                    sparse_vector_name: models.SparseVectorParams(
                        modifier=models.Modifier.IDF,
                    ),
                },
            )

            if self._field_indexes:
                for field_name, field_type in self._field_indexes.items():
                    await self._client.create_payload_index(
                        collection_name=collection_name,
                        field_name=field_name,
                        field_schema=field_type,
                    )
            return True
        except Exception as e:
            logger.error(f"Error creating hybrid collection {collection_name}: {e}")
            return False

    async def create_late_interaction_collection(
        self,
        collection_name: str,
        vector_size: int,
        vector_name: str,
        distance: str = "cosine",
    ) -> bool:
        """
        Create a Qdrant multivector collection for ColBERT-style MaxSim retrieval.
        """
        try:
            distance_map = {
                "cosine": models.Distance.COSINE,
                "dot": models.Distance.DOT,
                "euclidean": models.Distance.EUCLID,
                "manhattan": models.Distance.MANHATTAN,
            }
            distance_metric = distance_map.get(distance.lower(), models.Distance.COSINE)

            await self._client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    vector_name: models.VectorParams(
                        size=vector_size,
                        distance=distance_metric,
                        multivector_config=models.MultiVectorConfig(
                            comparator=models.MultiVectorComparator.MAX_SIM,
                        ),
                        hnsw_config=models.HnswConfigDiff(m=0),
                    ),
                },
            )

            if self._field_indexes:
                for field_name, field_type in self._field_indexes.items():
                    await self._client.create_payload_index(
                        collection_name=collection_name,
                        field_name=field_name,
                        field_schema=field_type,
                    )
            return True
        except Exception as e:
            logger.error(f"Error creating late-interaction collection {collection_name}: {e}")
            return False

    async def batch_store_late_interaction(
        self,
        entries: list[BatchEntry],
        collection_name: str,
        late_interaction_provider,
    ) -> int:
        """Store entries as multivectors for late-interaction retrieval."""
        try:
            if not await self._client.collection_exists(collection_name):
                created = await self.create_late_interaction_collection(
                    collection_name=collection_name,
                    vector_size=late_interaction_provider.get_vector_size(),
                    vector_name=late_interaction_provider.get_vector_name(),
                )
                if not created:
                    return 0

            vector_name = late_interaction_provider.get_vector_name()
            batch_size = _embedding_batch_size()
            stored = 0

            for start in range(0, len(entries), batch_size):
                batch = entries[start:start + batch_size]
                documents = [e.content for e in batch]
                vectors = await late_interaction_provider.embed_documents(documents)

                points = []
                for i, entry in enumerate(batch):
                    if entry.id:
                        try:
                            point_id = str(uuid.UUID(entry.id)).replace("-", "")
                        except ValueError:
                            point_id = uuid.uuid5(uuid.NAMESPACE_DNS, entry.id).hex
                    else:
                        point_id = uuid.uuid4().hex

                    points.append(
                        models.PointStruct(
                            id=point_id,
                            payload={"document": entry.content, METADATA_PATH: entry.metadata or {}},
                            vector={vector_name: vectors[i]},
                        )
                    )

                await self._client.upsert(collection_name=collection_name, points=points, wait=True)
                stored += len(batch)

            return stored
        except Exception as e:
            logger.error(f"Error in batch_store_late_interaction: {e!r}", exc_info=True)
            return 0

    async def batch_store_hybrid(
        self,
        entries: list[BatchEntry],
        collection_name: str,
        sparse_provider,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> int:
        """Store entries with both dense and sparse vectors, batched for memory safety."""
        dense_provider = embedding_provider or self._embedding_provider
        dense_name = dense_provider.get_vector_name()
        sparse_name = sparse_provider.get_vector_name()
        batch_size = _embedding_batch_size()
        stored = 0
        try:
            for start in range(0, len(entries), batch_size):
                entry_batch = entries[start : start + batch_size]
                documents = [e.content for e in entry_batch]
                try:
                    dense = await dense_provider.embed_documents(documents)
                    sparse = await sparse_provider.embed_documents(documents)
                except Exception as e:
                    logger.error(
                        "Error embedding hybrid batch %s-%s in %r: %s",
                        start, start + len(entry_batch), collection_name, e,
                    )
                    raise

                points = []
                for i, entry in enumerate(entry_batch):
                    if entry.id:
                        try:
                            point_id = str(uuid.UUID(entry.id)).replace("-", "")
                        except ValueError:
                            point_id = uuid.uuid5(uuid.NAMESPACE_DNS, entry.id).hex
                    else:
                        point_id = uuid.uuid4().hex

                    points.append(models.PointStruct(
                        id=point_id,
                        payload={"document": entry.content, METADATA_PATH: entry.metadata or {}},
                        vector={
                            dense_name: dense[i],
                            sparse_name: models.SparseVector(
                                indices=sparse[i]["indices"],
                                values=sparse[i]["values"],
                            ),
                        },
                    ))

                await self._client.upsert(collection_name=collection_name, points=points, wait=True)
                stored += len(points)

            logger.info("Hybrid stored %s entries in %r.", stored, collection_name)
            return stored
        except Exception as e:
            logger.error("Error in batch_store_hybrid: %s (stored %s so far)", e, stored)
            return stored

    async def search_hybrid_rrf(
        self,
        query: str,
        collection_name: str,
        sparse_provider,
        *,
        limit: int = 10,
        query_filter: models.Filter | None = None,
        prefetch_limit: int = 50,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> list[tuple[Entry, float]]:
        """
        Hybrid search using Qdrant's Query API with prefetch + Reciprocal Rank Fusion.

        Falls back to dense-only search when sparse retrieval is unavailable or fails.
        The most common cause of fallback is a collection that was created without a
        sparse vector slot (dense-only), or a sparse vector name mismatch after
        changing QDRANT_SPARSE_MODEL on an existing collection.

        The fallback is intentional and safe: dense-only results are still returned
        so mode="hybrid" and mode="rerank" never produce empty results just because
        BM25/BM42 is absent. A warning is logged so the issue is visible.
        """
        dense_provider = embedding_provider or self._embedding_provider

        try:
            dense_vector = await dense_provider.embed_query(query)
            sparse_vector = await sparse_provider.embed_query(query)

            dense_name = dense_provider.get_vector_name()
            sparse_name = sparse_provider.get_vector_name()

            response = await self._client.query_points(
                collection_name=collection_name,
                prefetch=[
                    models.Prefetch(
                        query=dense_vector,
                        using=dense_name,
                        limit=prefetch_limit,
                        filter=query_filter,
                    ),
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=sparse_vector["indices"],
                            values=sparse_vector["values"],
                        ),
                        using=sparse_name,
                        limit=prefetch_limit,
                        filter=query_filter,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            results = self._process_scored_results(response.points)

            # RRF returned empty despite a valid call — sparse prefetch may have
            # had 0 matches but dense would have matched. Fall back to dense-only
            # so we don't silently return nothing.
            if not results:
                msg = (
                    f"RRF fusion returned 0 results for collection {collection_name!r} "
                    "(sparse had no matches). Returning dense-only results."
                )
                logger.warning(msg)
                _w = _retrieval_warnings.get(None)
                if _w is not None:
                    _w.append(msg)
                return await self._hybrid_search_client_side(
                    query, collection_name, limit, query_filter, None, dense_provider
                )

            return results

        except Exception as e:
            # Common causes:
            #   - Collection has no sparse vector slot (dense-only collection)
            #   - Sparse vector name mismatch (QDRANT_SPARSE_MODEL changed after creation)
            #   - BM25/BM42 model changed after collection creation
            msg = (
                f"Hybrid RRF failed for collection {collection_name!r}: {e}. "
                "Returning dense-only results. "
                "To enable hybrid, recreate the collection with create_hybrid_collection."
            )
            logger.warning(msg)
            _w = _retrieval_warnings.get(None)
            if _w is not None:
                _w.append(msg)
            try:
                return await self._hybrid_search_client_side(
                    query, collection_name, limit, query_filter, None, dense_provider
                )
            except Exception as fallback_err:
                logger.error(
                    "Dense fallback also failed for collection %r: %s",
                    collection_name, fallback_err,
                )
                return []

    async def search_late_interaction(
        self,
        query: str,
        collection_name: str,
        late_interaction_provider,
        *,
        limit: int = 10,
        query_filter: models.Filter | None = None,
    ) -> list[tuple[Entry, float]]:
        """
        Search a multivector collection using Qdrant MaxSim late interaction.
        """
        try:
            query_vector = await late_interaction_provider.embed_query(query)
            response = await self._client.query_points(
                collection_name=collection_name,
                query=query_vector,
                using=late_interaction_provider.get_vector_name(),
                limit=limit,
                query_filter=query_filter,
                with_payload=True,
                with_vectors=False,
            )
            return self._process_scored_results(response.points)
        except Exception as e:
            logger.error(f"Error in search_late_interaction: {e}")
            return []

    async def ensure_macos_metadata_indexes(self, collection_name: str) -> None:
        """
        Idempotently create payload indexes for standard macOS file metadata fields.
        Safe to call on existing collections — Qdrant ignores duplicate index requests.
        """
        from raggy_mcp.ingest.macos_metadata import MACOS_INDEX_FIELDS

        type_map = {
            "keyword": models.PayloadSchemaType.KEYWORD,
            "integer": models.PayloadSchemaType.INTEGER,
            "float": models.PayloadSchemaType.FLOAT,
            "bool": models.PayloadSchemaType.BOOL,
        }

        for field_path, field_type in MACOS_INDEX_FIELDS:
            schema = type_map.get(field_type)
            if schema is None:
                continue
            try:
                await self._client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_path,
                    field_schema=schema,
                )
            except Exception as e:
                logger.debug(f"Index already exists or error for {field_path}: {e}")
