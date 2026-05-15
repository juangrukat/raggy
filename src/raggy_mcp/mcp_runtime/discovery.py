"""
Static discovery payloads — what fields are filterable, what extractors exist,
what search modes are supported, what the server is capable of overall.

These payloads are emitted by the discovery tools added in commit 4. They are
deliberately data-driven (no per-request work) so an agent can call them cheaply
to plan a task instead of guessing.
"""

from __future__ import annotations

from typing import Any

from raggy_mcp.ingest.extractor import SUPPORTED_EXTENSIONS
from raggy_mcp.ingest.macos_metadata import MACOS_INDEX_FIELDS

# Field-level documentation for indexed payload fields.
# Shape: { field_path: {type, operators, indexed, role, description} }
_INDEXED_FIELD_DOCS: dict[str, dict[str, Any]] = {
    "metadata.document_id": {
        "type": "keyword",
        "indexed": True,
        "role": "identity",
        "operators": ["==", "!=", "any", "except"],
        "description": "Stable sha1-based identifier per source file. Same file → same id across ingests.",
    },
    "metadata.path": {
        "type": "keyword",
        "indexed": True,
        "role": "identity",
        "operators": ["==", "!=", "any", "except"],
        "description": "Absolute filesystem path of the source file at ingest time.",
    },
    "metadata.parent_path": {
        "type": "keyword",
        "indexed": True,
        "role": "identity",
        "operators": ["==", "!=", "any", "except"],
        "description": "Parent directory path. Useful for filtering by folder.",
    },
    "metadata.filename": {
        "type": "keyword",
        "indexed": True,
        "role": "identity",
        "operators": ["==", "!=", "any", "except"],
        "description": "Filename including extension.",
    },
    "metadata.extension": {
        "type": "keyword",
        "indexed": True,
        "role": "identity",
        "operators": ["==", "!=", "any", "except"],
        "description": "Lowercase extension without leading dot.",
    },
    "metadata.content_type": {
        "type": "keyword",
        "indexed": True,
        "role": "identity",
        "operators": ["==", "!=", "any", "except"],
        "description": "macOS Spotlight UTI (e.g. com.adobe.pdf).",
    },
    "metadata.tags": {
        "type": "keyword[]",
        "indexed": True,
        "role": "filter",
        "operators": ["any", "except", "=="],
        "description": (
            "Finder tags. Note that Qdrant's `except` semantics on arrays "
            "match if any element is outside the excluded set; for strict "
            "exclusion use must_not + any."
        ),
    },
    "metadata.authors": {
        "type": "keyword[]",
        "indexed": True,
        "role": "filter",
        "operators": ["any", "except", "=="],
        "description": "Document authors from Spotlight kMDItemAuthors.",
    },
    "metadata.keywords": {
        "type": "keyword[]",
        "indexed": True,
        "role": "filter",
        "operators": ["any", "except", "=="],
        "description": "Document keywords from Spotlight kMDItemKeywords.",
    },
    "metadata.size_bytes": {
        "type": "integer",
        "indexed": True,
        "role": "filter",
        "operators": [">", ">=", "<", "<=", "=="],
        "description": "File size in bytes.",
    },
    "metadata.is_hidden": {
        "type": "bool",
        "indexed": True,
        "role": "filter",
        "operators": ["=="],
        "description": "True for dotfiles or hidden files.",
    },
    "metadata.has_text": {
        "type": "bool",
        "indexed": True,
        "role": "filter",
        "operators": ["=="],
        "description": "False if extraction returned no text.",
    },
    "metadata.created_at": {
        "type": "iso_datetime_string",
        "indexed": True,
        "role": "filter",
        "operators": [">", ">=", "<", "<=", "=="],
        "description": "ISO-8601 created timestamp from filesystem.",
    },
    "metadata.modified_at": {
        "type": "iso_datetime_string",
        "indexed": True,
        "role": "filter",
        "operators": [">", ">=", "<", "<=", "=="],
        "description": "ISO-8601 modified timestamp from filesystem.",
    },
    "metadata.last_opened_at": {
        "type": "iso_datetime_string",
        "indexed": True,
        "role": "filter",
        "operators": [">", ">=", "<", "<=", "=="],
        "description": "ISO-8601 last-opened timestamp from Spotlight.",
    },
    "metadata.page_count": {
        "type": "integer",
        "indexed": True,
        "role": "filter",
        "operators": [">", ">=", "<", "<=", "=="],
        "description": "Page count for paged formats (PDF, DOCX).",
    },
    "metadata.extractor_used": {
        "type": "keyword",
        "indexed": True,
        "role": "stat",
        "operators": ["==", "!=", "any"],
        "description": "Which extractor produced the text (pdfminer, pypdf, python-docx, plain-utf-8).",
    },
    "metadata.char_count": {
        "type": "integer",
        "indexed": True,
        "role": "stat",
        "operators": [">", ">=", "<", "<=", "=="],
        "description": "Length of extracted text in characters.",
    },
    "metadata.ingested_at": {
        "type": "iso_datetime_string",
        "indexed": True,
        "role": "stat",
        "operators": [">", ">=", "<", "<=", "=="],
        "description": "ISO-8601 ingestion timestamp.",
    },
}


def indexed_fields_payload() -> dict[str, Any]:
    """Build the get_indexed_fields response data."""
    fields = []
    indexed_paths = {p for p, _ in MACOS_INDEX_FIELDS}
    for path, info in _INDEXED_FIELD_DOCS.items():
        fields.append(
            {
                "field": path,
                "type": info["type"],
                "operators": info["operators"],
                "indexed": path in indexed_paths,
                "role": info["role"],
                "description": info["description"],
            }
        )
    return {
        "filter_grammar": {
            "structure": {"must": [], "should": [], "must_not": []},
            "operators": ["==", "!=", ">", ">=", "<", "<=", "any", "except"],
            "notes": [
                "Field names are auto-prefixed with 'metadata.' if absent.",
                "ISO date strings work with range operators.",
                "For strict array exclusion, prefer must_not + any over except.",
            ],
        },
        "fields": fields,
    }


def supported_extractors_payload() -> dict[str, Any]:
    """Build the get_supported_extractors response data."""
    return {
        "supported_extensions": sorted(SUPPORTED_EXTENSIONS),
        "extractors": [
            {
                "extension": ".txt and plain-text-ish formats",
                "primary": "plain UTF-8 read",
                "fallback": "charset autodetect (utf-8-sig, latin-1, cp1252)",
                "limitations": "formatting is preserved as text; no language-specific parsing",
            },
            {
                "extension": ".md/.markdown",
                "primary": "plain UTF-8 read",
                "fallback": "charset autodetect; YAML/TOML frontmatter is stripped",
                "limitations": "markdown formatting is preserved as-is in chunk text",
            },
            {
                "extension": ".json/.jsonl",
                "primary": "stdlib JSON parser rendered as path/value text",
                "fallback": "plain text read only if decoding fails",
                "limitations": "large nested arrays are flattened into searchable text, not row-level points",
            },
            {
                "extension": ".csv/.tsv",
                "primary": "stdlib CSV parser rendered as row/column text",
                "fallback": "plain text read only if decoding fails",
                "limitations": "all rows from a file share document-level metadata",
            },
            {
                "extension": ".pdf",
                "primary": "preflight scan, then PyMuPDF",
                "fallback": "pdfminer.six, then pypdf when earlier extractors return no text",
                "limitations": "OCR is not performed; probably scanned PDFs fail early with an OCR-needed message",
            },
            {
                "extension": ".docx",
                "primary": "python-docx",
                "fallback": None,
                "limitations": "paragraphs and tables are extracted; embedded images are not",
            },
        ],
        "chunking": {
            "strategy": "paragraph-aware with hard split for oversized paragraphs",
            "target_chunk_chars": 1000,
            "overlap_chars": 150,
        },
    }


def search_modes_payload() -> dict[str, Any]:
    """Build the list_search_modes response data."""
    return {
        "modes": [
            {
                "name": "dense",
                "description": "Single-stage dense vector search using the active embedding model.",
                "when_to_use": "Default mode. Strong semantic recall; works on any collection.",
                "requires_hybrid_collection": False,
            },
            {
                "name": "hybrid",
                "description": "Dense + sparse (BM25) retrieval fused server-side with Reciprocal Rank Fusion.",
                "when_to_use": "Filename, identifier, exact-phrase, and code-symbol queries blended with semantic search.",
                "requires_hybrid_collection": True,
            },
            {
                "name": "rerank",
                "description": "Hybrid first-stage, then cross-encoder rerank over top candidates.",
                "when_to_use": "When precision at top-K matters more than latency.",
                "requires_hybrid_collection": True,
                "default_reranker": "Xenova/ms-marco-MiniLM-L-6-v2",
            },
            {
                "name": "late_interaction",
                "description": "ColBERT-style multivector retrieval using Qdrant MaxSim scoring.",
                "when_to_use": "High-precision semantic matching where exact token-level interactions matter more than latency/storage.",
                "requires_hybrid_collection": False,
                "requires_late_interaction_collection": True,
                "default_model": "colbert-ir/colbertv2.0",
            },
        ],
    }


def server_capabilities_payload(
    *,
    profile: str,
    transports: list[str],
    dynamic_embedding_models: bool,
    resources_enabled: bool,
    auth_enabled: bool,
    available_models: list[str],
) -> dict[str, Any]:
    return {
        "transports_supported": transports,
        "active_profile": profile,
        "profiles_supported": ["minimal", "canonical", "full"],
        "search_modes_supported": [m["name"] for m in search_modes_payload()["modes"]],
        "extraction_extensions": sorted(SUPPORTED_EXTENSIONS),
        "dynamic_embedding_models": dynamic_embedding_models,
        "hybrid_retrieval": True,
        "late_interaction_retrieval": True,
        "resources_enabled": resources_enabled,
        "auth_enabled_for_http": auth_enabled,
        "report_apply_gating": ["delete_collection", "ingest_folder"],
        "available_embedding_models_count": len(available_models),
    }
