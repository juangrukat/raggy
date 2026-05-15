"""
Generic paragraph-boundary overlap chunker.

This is the fallback chunker that works for any text format. It preserves
the original algorithm from extractor.py: split on double-newlines, then
merge paragraphs up to chunk_size with overlap.

Used for: .txt, .pdf, .docx, .json, .csv, .yaml, .toml, .xml, .html, .log, etc.
"""

from __future__ import annotations

import os
import re
from typing import Any

from raggy_mcp.ingest.chunkers import Chunker
from raggy_mcp.ingest.types import Chunk, ExtractedDocument

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


class GenericChunker(Chunker):
    """Paragraph-boundary respecting overlap chunker (original algorithm)."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        self._chunk_size = chunk_size or _env_int(
            "QDRANT_INGEST_CHUNK_SIZE", DEFAULT_CHUNK_SIZE
        )
        raw_overlap = chunk_overlap or _env_int(
            "QDRANT_INGEST_CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP
        )
        self._chunk_overlap = min(raw_overlap, max(0, self._chunk_size - 1))
        self._stride = max(1, self._chunk_size - self._chunk_overlap)

    def chunk(self, doc: ExtractedDocument, metadata: dict[str, Any]) -> list[Chunk]:
        normalized = re.sub(r"[^\S\n]+", " ", doc.text)
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", normalized) if p.strip()]
        raw_chunks: list[str] = []
        current = ""

        for para in paragraphs:
            if len(para) > self._chunk_size:
                if current:
                    raw_chunks.append(current)
                    current = ""
                for i in range(0, len(para), self._stride):
                    raw_chunks.append(para[i : i + self._chunk_size])
            elif len(current) + len(para) + 2 > self._chunk_size:
                raw_chunks.append(current)
                current = para
            else:
                current = (current + "\n\n" + para).strip() if current else para

        if current:
            raw_chunks.append(current)

        total = len(raw_chunks)
        chunks = []
        for i, text in enumerate(raw_chunks):
            chunk_meta = {**metadata, "chunk_index": i, "total_chunks": total}
            chunks.append(
                Chunk(text=text, chunk_index=i, total_chunks=total, metadata=chunk_meta)
            )
        return chunks
