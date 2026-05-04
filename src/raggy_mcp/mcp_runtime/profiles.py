"""
Tool exposure profiles.

The server registers many tools internally, but only the ones whose declared
profile is at-or-below the active profile are actually visible to MCP clients.
This keeps the curated surface small and predictable for agents that aren't
expected to operate at admin level.

Profile ladder (broader includes narrower):

* ``minimal``    — daily-driver tools only: search, ingest, basic capability
* ``canonical``  — minimal + collection lifecycle and embedding-model setup
* ``full``       — everything, including raw / admin / mutation-heavy tools

The active profile defaults to ``canonical`` and can be overridden via the
``QDRANT_MCP_TOOL_PROFILE`` env var.
"""

from __future__ import annotations

from enum import IntEnum


class ToolProfile(IntEnum):
    """Profile ordering — higher value = broader exposure."""

    MINIMAL = 1
    CANONICAL = 2
    FULL = 3

    @classmethod
    def parse(cls, value: str | None) -> "ToolProfile":
        if not value:
            return cls.CANONICAL
        v = value.strip().lower()
        if v in ("minimal", "min", "1"):
            return cls.MINIMAL
        if v in ("canonical", "default", "2"):
            return cls.CANONICAL
        if v in ("full", "all", "admin", "3"):
            return cls.FULL
        raise ValueError(
            f"Unknown tool profile '{value}'. Valid: minimal, canonical, full."
        )

    def __str__(self) -> str:  # pragma: no cover
        return self.name.lower()


# Tool name → minimum profile at which the tool is exposed.
# A tool with profile=MINIMAL appears under all three profiles.
# A tool with profile=FULL appears only under 'full'.
TOOL_PROFILES: dict[str, ToolProfile] = {
    # --- minimal: safe daily tools ---
    "search_documents": ToolProfile.MINIMAL,
    "ingest_file": ToolProfile.MINIMAL,
    "ingest_folder": ToolProfile.MINIMAL,
    "list_embedding_models": ToolProfile.MINIMAL,
    "get_collection_info": ToolProfile.MINIMAL,
    "list_collections": ToolProfile.MINIMAL,
    # discovery tools (added in commit 4) — also minimal
    "get_indexed_fields": ToolProfile.MINIMAL,
    "get_supported_extractors": ToolProfile.MINIMAL,
    "get_collection_schema": ToolProfile.MINIMAL,
    "list_search_modes": ToolProfile.MINIMAL,
    "get_server_capabilities": ToolProfile.MINIMAL,

    # --- canonical: collection lifecycle + setup ---
    "create_collection": ToolProfile.CANONICAL,
    "create_hybrid_collection": ToolProfile.CANONICAL,
    "create_late_interaction_collection": ToolProfile.CANONICAL,
    "bootstrap_collection_indexes": ToolProfile.CANONICAL,
    "set_collection_embedding_model": ToolProfile.CANONICAL,

    # --- full: admin / raw / mutation-heavy ---
    "delete_collection": ToolProfile.FULL,
    "qdrant_find": ToolProfile.FULL,        # raw chunk-level search
    "qdrant_store": ToolProfile.FULL,       # raw chunk-level store
    "qdrant_store_batch": ToolProfile.FULL, # raw batch store
    "scroll_collection": ToolProfile.FULL,  # raw browse
    "hybrid_search": ToolProfile.FULL,      # raw chunk-level dense (legacy)
}


def is_tool_visible(tool_name: str, active: ToolProfile) -> bool:
    """Whether a tool is exposed under the given active profile."""
    required = TOOL_PROFILES.get(tool_name, ToolProfile.FULL)
    return active >= required
