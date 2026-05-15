"""
Shared type definitions for the ingestion pipeline.

Separated from extractor.py to avoid circular imports with chunkers/.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExtractedDocument:
    text: str
    extractor_used: str
    char_count: int
    page_count: int | None = None
    error: str | None = None


@dataclass
class Chunk:
    text: str
    chunk_index: int
    total_chunks: int
    metadata: dict = field(default_factory=dict)