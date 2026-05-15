"""
Enhanced configuration settings with removed limits and better defaults.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings

from raggy_mcp.embeddings.types import EmbeddingProviderType

# Import enhanced descriptions
from raggy_mcp.enhanced_tool_descriptions import (
    DEFAULT_TOOL_BATCH_STORE_DESCRIPTION,
    DEFAULT_TOOL_BOOTSTRAP_INDEXES_DESCRIPTION,
    DEFAULT_TOOL_CREATE_COLLECTION_DESCRIPTION,
    DEFAULT_TOOL_CREATE_HYBRID_COLLECTION_DESCRIPTION,
    DEFAULT_TOOL_DELETE_COLLECTION_DESCRIPTION,
    DEFAULT_TOOL_FIND_DESCRIPTION,
    DEFAULT_TOOL_GET_COLLECTION_INFO_DESCRIPTION,
    DEFAULT_TOOL_HYBRID_SEARCH_DESCRIPTION,
    DEFAULT_TOOL_INGEST_FILE_DESCRIPTION,
    DEFAULT_TOOL_INGEST_FOLDER_DESCRIPTION,
    DEFAULT_TOOL_LIST_COLLECTIONS_DESCRIPTION,
    DEFAULT_TOOL_LIST_EMBEDDING_MODELS_DESCRIPTION,
    DEFAULT_TOOL_SCROLL_DESCRIPTION,
    DEFAULT_TOOL_SEARCH_DOCUMENTS_DESCRIPTION,
    DEFAULT_TOOL_SET_COLLECTION_EMBEDDING_MODEL_DESCRIPTION,
    DEFAULT_TOOL_SET_COLLECTION_EMBEDDING_MODEL_IMPL_DESCRIPTION,
    DEFAULT_TOOL_STORE_DESCRIPTION,
)

METADATA_PATH = "metadata"
DEFAULT_COLLECTION_NAME = "documents"
DEFAULT_LOCAL_STORAGE_PATH = str(Path(__file__).resolve().parents[2] / "storage")


class ToolSettings(BaseSettings):
    """
    Configuration for all the tools with enhanced descriptions.
    """

    tool_store_description: str = Field(
        default=DEFAULT_TOOL_STORE_DESCRIPTION,
        validation_alias="TOOL_STORE_DESCRIPTION",
    )
    tool_find_description: str = Field(
        default=DEFAULT_TOOL_FIND_DESCRIPTION,
        validation_alias="TOOL_FIND_DESCRIPTION",
    )
    tool_batch_store_description: str = Field(
        default=DEFAULT_TOOL_BATCH_STORE_DESCRIPTION,
        validation_alias="TOOL_BATCH_STORE_DESCRIPTION",
    )
    tool_scroll_description: str = Field(
        default=DEFAULT_TOOL_SCROLL_DESCRIPTION,
        validation_alias="TOOL_SCROLL_DESCRIPTION",
    )
    tool_list_collections_description: str = Field(
        default=DEFAULT_TOOL_LIST_COLLECTIONS_DESCRIPTION,
        validation_alias="TOOL_LIST_COLLECTIONS_DESCRIPTION",
    )
    tool_create_collection_description: str = Field(
        default=DEFAULT_TOOL_CREATE_COLLECTION_DESCRIPTION,
        validation_alias="TOOL_CREATE_COLLECTION_DESCRIPTION",
    )
    tool_get_collection_info_description: str = Field(
        default=DEFAULT_TOOL_GET_COLLECTION_INFO_DESCRIPTION,
        validation_alias="TOOL_GET_COLLECTION_INFO_DESCRIPTION",
    )
    tool_delete_collection_description: str = Field(
        default=DEFAULT_TOOL_DELETE_COLLECTION_DESCRIPTION,
        validation_alias="TOOL_DELETE_COLLECTION_DESCRIPTION",
    )
    tool_hybrid_search_description: str = Field(
        default=DEFAULT_TOOL_HYBRID_SEARCH_DESCRIPTION,
        validation_alias="TOOL_HYBRID_SEARCH_DESCRIPTION",
    )
    tool_set_collection_embedding_model_description: str = Field(
        default=DEFAULT_TOOL_SET_COLLECTION_EMBEDDING_MODEL_DESCRIPTION,
        validation_alias="TOOL_SET_COLLECTION_EMBEDDING_MODEL_DESCRIPTION",
    )
    tool_list_embedding_models_description: str = Field(
        default=DEFAULT_TOOL_LIST_EMBEDDING_MODELS_DESCRIPTION,
        validation_alias="TOOL_LIST_EMBEDDING_MODELS_DESCRIPTION",
    )
    tool_set_collection_embedding_model_impl_description: str = Field(
        default=DEFAULT_TOOL_SET_COLLECTION_EMBEDDING_MODEL_IMPL_DESCRIPTION,
        validation_alias="TOOL_SET_COLLECTION_EMBEDDING_MODEL_IMPL_DESCRIPTION",
    )
    tool_ingest_file_description: str = Field(
        default=DEFAULT_TOOL_INGEST_FILE_DESCRIPTION,
        validation_alias="TOOL_INGEST_FILE_DESCRIPTION",
    )
    tool_ingest_folder_description: str = Field(
        default=DEFAULT_TOOL_INGEST_FOLDER_DESCRIPTION,
        validation_alias="TOOL_INGEST_FOLDER_DESCRIPTION",
    )
    tool_search_documents_description: str = Field(
        default=DEFAULT_TOOL_SEARCH_DOCUMENTS_DESCRIPTION,
        validation_alias="TOOL_SEARCH_DOCUMENTS_DESCRIPTION",
    )
    tool_bootstrap_indexes_description: str = Field(
        default=DEFAULT_TOOL_BOOTSTRAP_INDEXES_DESCRIPTION,
        validation_alias="TOOL_BOOTSTRAP_INDEXES_DESCRIPTION",
    )
    tool_create_hybrid_collection_description: str = Field(
        default=DEFAULT_TOOL_CREATE_HYBRID_COLLECTION_DESCRIPTION,
        validation_alias="TOOL_CREATE_HYBRID_COLLECTION_DESCRIPTION",
    )


class EmbeddingProviderSettings(BaseSettings):
    """
    Configuration for the embedding provider.
    """

    provider_type: EmbeddingProviderType = Field(
        default=EmbeddingProviderType.FASTEMBED,
        validation_alias="EMBEDDING_PROVIDER",
    )
    model_name: str = Field(
        default="Qwen/Qwen3-Embedding-4B",
        validation_alias="EMBEDDING_MODEL",
    )
    device: str = Field(
        default="auto",
        validation_alias="EMBEDDING_DEVICE",
        description="Embedding device, e.g. auto, cpu, cuda, mps, or metal when supported by the local runtime.",
    )
    qwen3_sidecar_path: str | None = Field(
        default=None,
        validation_alias="QWEN3_SIDECAR_PATH",
        description="Optional path to the Rust Qwen3 sidecar binary.",
    )
    qwen3_max_length: int = Field(default=1024, validation_alias="QWEN3_MAX_LENGTH")
    qwen3_dtype: str = Field(default="auto", validation_alias="QWEN3_DTYPE")
    qwen3_output_dimension: int | None = Field(
        default=None,
        validation_alias="QWEN3_OUTPUT_DIMENSION",
        description="Optional Matryoshka output dimension for Qwen3 embeddings. Must be <= the model's native size.",
    )
    qwen3_response_limit_bytes: int = Field(
        default=64 * 1024 * 1024,
        validation_alias="QWEN3_RESPONSE_LIMIT_BYTES",
        description="Async stdout read limit for large Qwen3 sidecar JSON embedding responses.",
    )
    qwen3_idle_timeout_seconds: int = Field(
        default=300,
        validation_alias="QWEN3_IDLE_TIMEOUT_SECONDS",
        description="Seconds to keep the Qwen3 sidecar alive after the last request. 0 disables automatic idle shutdown.",
    )
    qwen3_metrics_path: str | None = Field(
        default=None,
        validation_alias="QWEN3_METRICS_PATH",
        description="Optional JSONL path for Qwen3 embedding timing metrics.",
    )


class FilterableField(BaseModel):
    name: str = Field(description="The name of the field payload field to filter on")
    description: str = Field(
        description="A description for the field used in the tool description"
    )
    field_type: Literal["keyword", "integer", "float", "boolean"] = Field(
        description="The type of the field"
    )
    condition: Literal["==", "!=", ">", ">=", "<", "<=", "any", "except"] | None = (
        Field(
            default=None,
            description=(
                "The condition to use for the filter. If not provided, the field will be indexed, but no "
                "filter argument will be exposed to MCP tool."
            ),
        )
    )
    required: bool = Field(
        default=False,
        description="Whether the field is required for the filter.",
    )


class QdrantSettings(BaseSettings):
    """
    Configuration for the Qdrant connector with sensible defaults and no artificial limits.
    """

    location: str | None = Field(default=None, validation_alias="QDRANT_URL")
    api_key: str | None = Field(default=None, validation_alias="QDRANT_API_KEY")
    collection_name: str | None = Field(
        default=DEFAULT_COLLECTION_NAME, validation_alias="COLLECTION_NAME"
    )
    local_path: str | None = Field(
        default=DEFAULT_LOCAL_STORAGE_PATH, validation_alias="QDRANT_LOCAL_PATH"
    )

    @model_validator(mode="before")
    @classmethod
    def clean_empty_strings(cls, values):
        """Convert empty strings to None for proper validation."""
        if isinstance(values, dict):
            for key, value in values.items():
                if value == "":
                    values[key] = None
        return values

    # Increased default search limit for better results
    search_limit: int = Field(default=50, validation_alias="QDRANT_SEARCH_LIMIT")
    read_only: bool = Field(default=False, validation_alias="QDRANT_READ_ONLY")

    filterable_fields: list[FilterableField] | None = Field(default=None)

    allow_arbitrary_filter: bool = Field(
        default=False, validation_alias="QDRANT_ALLOW_ARBITRARY_FILTER"
    )

    # Enhanced settings for multi-collection support
    enable_collection_management: bool = Field(
        default=True, validation_alias="QDRANT_ENABLE_COLLECTION_MANAGEMENT"
    )
    enable_dynamic_embedding_models: bool = Field(
        default=True, validation_alias="QDRANT_ENABLE_DYNAMIC_EMBEDDING_MODELS"
    )
    default_vector_size: int = Field(
        default=384, validation_alias="QDRANT_DEFAULT_VECTOR_SIZE"
    )
    default_distance_metric: str = Field(
        default="cosine", validation_alias="QDRANT_DEFAULT_DISTANCE_METRIC"
    )

    # Removed artificial batch size limit - now unlimited
    max_batch_size: int = Field(
        default=10000,
        validation_alias="QDRANT_MAX_BATCH_SIZE",
        description="Maximum batch size for operations. Default is 10000 (effectively unlimited for most use cases)",
    )
    write_max_concurrency: int = Field(
        default=1,
        validation_alias="QDRANT_WRITE_MAX_CONCURRENCY",
        description="Maximum concurrent embedding/upsert write jobs per server process.",
    )
    write_queue_size: int = Field(
        default=8,
        validation_alias="QDRANT_WRITE_QUEUE_SIZE",
        description="Maximum queued write jobs waiting for an embedding/upsert slot.",
    )

    enable_resources: bool = Field(
        default=True, validation_alias="QDRANT_ENABLE_RESOURCES"
    )

    # MCP tool exposure profile (minimal | canonical | full).
    # Default: canonical. See mcp_runtime/profiles.py for the per-tool mapping.
    mcp_tool_profile: str = Field(
        default="canonical",
        validation_alias="QDRANT_MCP_TOOL_PROFILE",
    )

    # Sparse model for hybrid retrieval.
    # "Qdrant/bm25" is the pure-algorithmic default (no download).
    # "Qdrant/bm42-all-minilm-l6-v2-attentions" is a neural BM25 improvement.
    sparse_model: str = Field(
        default="Qdrant/bm25",
        validation_alias="QDRANT_SPARSE_MODEL",
    )

    # Default reranker for mode="rerank". Overridable per-request via reranker_model arg.
    # FastEmbed cross-encoders: "Xenova/ms-marco-MiniLM-L-6-v2", "BAAI/bge-reranker-base"
    # Qwen3 generative rerankers (requires torch+transformers): "Qwen/Qwen3-Reranker-4B"
    default_reranker_model: str = Field(
        default="Xenova/ms-marco-MiniLM-L-6-v2",
        validation_alias="QDRANT_RERANKER_MODEL",
    )

    # Reranker instruction for Qwen3-Reranker models. Improving specificity
    # toward your retrieval task raises nDCG by ~1-5%.
    # Example: "Retrieve passages that directly answer the question using
    #   explicit claims, definitions, or cited evidence from the document."
    reranker_instruction: str | None = Field(
        default=None,
        validation_alias="QDRANT_RERANKER_INSTRUCTION",
        description="Task instruction for Qwen3 generative rerankers.",
    )

    # Candidate pool size before reranking. 0 = auto (80). Raise to 100 for deeper
    # recall, or lower to 30 for fast MiniLM reranking.
    rerank_prefetch_limit: int = Field(
        default=0,
        validation_alias="QDRANT_RERANK_PREFETCH_LIMIT",
        description="Raw candidate pool size fed to the reranker. 0 = auto.",
    )

    # Max chunks fed to the reranker from the candidate pool. 0 = balanced auto (50).
    rerank_top_k: int = Field(
        default=0,
        validation_alias="QDRANT_RERANK_TOP_K",
        description="Max candidates scored by the reranker. 0 = all prefetched.",
    )

    def filterable_fields_dict(self) -> dict[str, FilterableField]:
        if self.filterable_fields is None:
            return {}
        return {field.name: field for field in self.filterable_fields}

    def filterable_fields_dict_with_conditions(self) -> dict[str, FilterableField]:
        if self.filterable_fields is None:
            return {}
        return {
            field.name: field
            for field in self.filterable_fields
            if field.condition is not None
        }

    @model_validator(mode="after")
    def check_local_path_conflict(self) -> "QdrantSettings":
        if self.local_path and (self.location is not None or self.api_key is not None):
            local_path_was_explicit = "local_path" in self.model_fields_set
            if (
                not local_path_was_explicit
                and self.local_path == DEFAULT_LOCAL_STORAGE_PATH
            ):
                self.local_path = None
            else:
                raise ValueError(
                    "If 'local_path' is set, 'location' and 'api_key' must be None."
                )
        return self
