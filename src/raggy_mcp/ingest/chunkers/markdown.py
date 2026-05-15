"""
Structure-aware Markdown chunker.

Splits Markdown documents by heading boundaries (##, ###, etc.), preserving
the heading hierarchy as ``heading_path`` and ``section`` in chunk metadata.
This lets users search/filter by document section.

Example heading_path: ``["Installation", "Configuration", "API Keys"]``

Used for: .md, .markdown, .rst
"""

from __future__ import annotations

import re
from typing import Any

from raggy_mcp.ingest.chunkers import Chunker
from raggy_mcp.ingest.chunkers.generic import GenericChunker
from raggy_mcp.ingest.types import Chunk, ExtractedDocument

# Match ATX headings: ## Heading text
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


class MarkdownChunker(Chunker):
    """Splits Markdown by heading boundaries, preserving heading_path metadata."""

    def __init__(self, max_chunk_size: int = 1024):
        self._max_chunk_size = max_chunk_size
        self._fallback = GenericChunker()

    def chunk(self, doc: ExtractedDocument, metadata: dict[str, Any]) -> list[Chunk]:
        text = doc.text
        if not text.strip():
            return []

        # Locate all headings with their level and line position
        heading_positions: list[dict[str, Any]] = []
        for m in _HEADING_RE.finditer(text):
            level = len(m.group(1))  # 1-6
            heading_text = m.group(2).strip()
            heading_positions.append(
                {
                    "start": m.start(),
                    "end": m.end(),
                    "level": level,
                    "text": heading_text,
                }
            )

        if not heading_positions:
            # No headings — fall back to generic paragraph chunking
            return self._fallback.chunk(doc, metadata)

        # Split the document into sections at heading boundaries
        sections: list[dict[str, Any]] = []
        active_heading_path: list[str] = []

        for i, h in enumerate(heading_positions):
            # Compute section text: from this heading to the next heading (or EOF)
            section_start = h["end"] + 1  # content starts after heading line
            if i + 1 < len(heading_positions):
                section_end = heading_positions[i + 1]["start"]
            else:
                section_end = len(text)

            section_text = text[section_start:section_end].strip()

            # Update heading path: pop deeper levels, push current
            while active_heading_path and h["level"] <= _last_heading_level(
                active_heading_path, heading_positions, i
            ):
                active_heading_path.pop()
            active_heading_path.append(h["text"])

            sections.append(
                {
                    "text": section_text,
                    "heading_text": h["text"],
                    "heading_level": h["level"],
                    "heading_path": list(active_heading_path),
                }
            )

        # If there's content before the first heading, treat it as a preamble
        if heading_positions and heading_positions[0]["start"] > 0:
            preamble = text[: heading_positions[0]["start"]].strip()
            if preamble:
                sections.insert(
                    0,
                    {
                        "text": preamble,
                        "heading_text": "",
                        "heading_level": 0,
                        "heading_path": [],
                    },
                )

        # Build chunks from sections
        chunks: list[Chunk] = []
        for sec in sections:
            if not sec["text"]:
                continue

            section_doc = ExtractedDocument(
                text=sec["text"],
                extractor_used=doc.extractor_used,
                char_count=len(sec["text"]),
            )
            section_meta = {
                **metadata,
                "section": sec["heading_text"],
                "heading_path": sec["heading_path"],
                "heading_level": sec["heading_level"],
            }

            # If section is small enough, keep as one chunk
            if len(sec["text"]) <= self._max_chunk_size:
                chunk_idx = len(chunks)
                chunks.append(
                    Chunk(
                        text=sec["text"],
                        chunk_index=chunk_idx,
                        total_chunks=0,
                        metadata=dict(
                            section_meta,
                            chunk_index=chunk_idx,
                        ),
                    )
                )
            else:
                # Large section — sub-chunk using generic chunker
                sub_chunks = self._fallback.chunk(section_doc, section_meta)
                for sc in sub_chunks:
                    sc.metadata["section"] = sec["heading_text"]
                    sc.metadata["heading_path"] = sec["heading_path"]
                chunks.extend(sub_chunks)

        # Fix total_chunks
        total = len(chunks)
        for c in chunks:
            c.total_chunks = total
            c.metadata["total_chunks"] = total
        return chunks


def _last_heading_level(
    active_path: list[str],
    heading_positions: list[dict[str, Any]],
    current_index: int,
) -> int:
    """Determine the level of the last heading in the active path."""
    if current_index > 0 and heading_positions:
        # Walk backwards to find the last heading before current
        for idx in range(current_index - 1, -1, -1):
            h = heading_positions[idx]
            if h["text"] == active_path[-1]:
                return h["level"]
    return 1