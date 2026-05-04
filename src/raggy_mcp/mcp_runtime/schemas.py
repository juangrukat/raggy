"""JSON Schemas for structured MCP tool responses."""

from __future__ import annotations

from typing import Any


def _envelope(data_schema: dict[str, Any]) -> dict[str, Any]:
    observability = {
        "type": "object",
        "properties": {
            "duration_ms": {"type": "integer"},
            "warnings": {"type": "array", "items": {"type": "string"}},
            "stats": {"type": "object", "additionalProperties": True},
        },
        "required": ["duration_ms", "warnings", "stats"],
        "additionalProperties": True,
    }
    contract = {
        "type": "object",
        "properties": {
            "contract_version": {"type": "string"},
            "toolset_version": {"type": "string"},
            "profile": {"type": "string"},
        },
        "required": ["contract_version", "toolset_version", "profile"],
        "additionalProperties": False,
    }
    error = {
        "type": "object",
        "properties": {
            "code": {"type": "string"},
            "message": {"type": "string"},
            "retryable": {"type": "boolean"},
        },
        "required": ["code", "message", "retryable"],
        "additionalProperties": True,
    }
    return {
        "type": "object",
        "properties": {
            "contract": contract,
            "data": data_schema,
            "error": error,
            "observability": observability,
        },
        "required": ["contract", "observability"],
        "oneOf": [{"required": ["data"]}, {"required": ["error"]}],
        "additionalProperties": False,
    }


COLLECTION_RESULT_SCHEMA: dict[str, Any] = _envelope({
    "type": "object",
    "properties": {
        "collection_name": {"type": "string"},
        "vector_size": {"type": "integer"},
        "distance": {"type": "string"},
        "embedding_model": {"type": "string"},
        "sparse_model": {"type": "string"},
        "hybrid": {"type": "boolean"},
        "late_interaction_model": {"type": "string"},
        "late_interaction": {"type": "boolean"},
        "vector_name": {"type": "string"},
    },
    "required": ["collection_name", "vector_size", "distance"],
    "additionalProperties": True,
})

INGEST_FILE_SCHEMA: dict[str, Any] = _envelope({
    "type": "object",
    "properties": {
        "file_path": {"type": "string"},
        "filename": {"type": "string"},
        "document_id": {"type": "string"},
        "collection": {"type": "string"},
        "chunks_stored": {"type": "integer"},
        "extractor_used": {"type": "string"},
        "char_count": {"type": "integer"},
        "page_count": {"type": ["integer", "null"]},
    },
    "required": ["file_path", "filename", "document_id", "collection", "chunks_stored", "extractor_used", "char_count"],
    "additionalProperties": True,
})

INGEST_FOLDER_SCHEMA: dict[str, Any] = _envelope({
    "type": "object",
    "additionalProperties": True,
})

SEARCH_DOCUMENTS_SCHEMA: dict[str, Any] = _envelope({
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "mode": {"type": "string"},
        "grouped_by_document": {"type": "boolean"},
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "document_id": {"type": "string"},
                    "path": {"type": "string"},
                    "filename": {"type": "string"},
                    "title": {"type": ["string", "null"]},
                    "snippet": {"type": "string"},
                    "score": {"type": "number"},
                    "chunk_count": {"type": "integer"},
                    "chunks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string"},
                                "score": {"type": "number"},
                                "chunk_index": {"type": ["integer", "null"]},
                                "metadata": {"type": "object", "additionalProperties": True},
                            },
                            "required": ["content", "score", "chunk_index", "metadata"],
                            "additionalProperties": True,
                        },
                    },
                    "metadata": {"type": "object", "additionalProperties": True},
                },
                "required": ["document_id", "snippet", "score", "chunk_count", "chunks", "metadata"],
                "additionalProperties": True,
            },
        },
    },
    "required": ["query", "mode", "grouped_by_document", "results"],
    "additionalProperties": True,
})

INDEX_SCHEMA: dict[str, Any] = _envelope({
    "type": "object",
    "properties": {
        "collection": {"type": "string"},
        "indexes_ensured": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["collection", "indexes_ensured"],
    "additionalProperties": True,
})

DISCOVERY_SCHEMA: dict[str, Any] = _envelope({"type": "object", "additionalProperties": True})
DELETE_COLLECTION_SCHEMA: dict[str, Any] = _envelope({"type": "object", "additionalProperties": True})
SET_COLLECTION_MODEL_SCHEMA: dict[str, Any] = _envelope({
    "type": "object",
    "properties": {
        "collection": {"type": "string"},
        "embedding_model": {"type": "string"},
        "vector_size": {"type": "integer"},
    },
    "required": ["collection", "embedding_model", "vector_size"],
    "additionalProperties": True,
})

TOOL_OUTPUT_SCHEMAS: dict[str, dict[str, Any]] = {
    "create_collection": COLLECTION_RESULT_SCHEMA,
    "create_hybrid_collection": COLLECTION_RESULT_SCHEMA,
    "create_late_interaction_collection": COLLECTION_RESULT_SCHEMA,
    "delete_collection": DELETE_COLLECTION_SCHEMA,
    "ingest_file": INGEST_FILE_SCHEMA,
    "ingest_folder": INGEST_FOLDER_SCHEMA,
    "search_documents": SEARCH_DOCUMENTS_SCHEMA,
    "bootstrap_collection_indexes": INDEX_SCHEMA,
    "set_collection_embedding_model": SET_COLLECTION_MODEL_SCHEMA,
    "get_indexed_fields": DISCOVERY_SCHEMA,
    "get_supported_extractors": DISCOVERY_SCHEMA,
    "list_search_modes": DISCOVERY_SCHEMA,
    "get_server_capabilities": DISCOVERY_SCHEMA,
    "get_collection_schema": DISCOVERY_SCHEMA,
}
