"""
Structure-aware chunker registry.

Each format gets its own chunker that preserves structural metadata
(heading paths, page numbers, symbol names) in the chunk metadata.
The generic paragraph-boundary chunker serves as the fallback.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from raggy_mcp.ingest.types import Chunk, ExtractedDocument


class Chunker(ABC):
    """Abstract base for all chunkers."""

    @abstractmethod
    def chunk(self, doc: ExtractedDocument, metadata: dict[str, Any]) -> list[Chunk]:
        """Split extracted text into chunks, enriching metadata per chunk."""
        ...


def resolve_chunker(ext: str | None) -> Chunker:
    """Dispatch to the right chunker based on file extension."""
    if ext in {".md", ".markdown", ".rst"}:
        from raggy_mcp.ingest.chunkers.markdown import MarkdownChunker

        return MarkdownChunker()
    if ext in {".py", ".js", ".jsx", ".ts", ".tsx", ".rs", ".java", ".c", ".cpp", ".h", ".hpp", ".go", ".rb", ".php", ".swift", ".kt", ".scala"}:
        from raggy_mcp.ingest.chunkers.code import CodeChunker

        return CodeChunker()
    # Generic fallback covers: .txt, .pdf, .docx, .json, .csv, .tsv, .yaml, .toml, .xml, .html, .log, etc.
    from raggy_mcp.ingest.chunkers.generic import GenericChunker

    return GenericChunker()