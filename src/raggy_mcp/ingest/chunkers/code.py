"""
Structure-aware code chunker.

Splits source code by top-level definitions (function, class, method)
using regex-based pattern matching. Preserves the symbol name and type
in chunk metadata.

No AST dependency — uses simple line-based heuristics that work across
languages. Falls back to GenericChunker for files that don't match.

Used for: .py, .js, .ts, .rs, .java, .c, .cpp, .go, .rb, .php, etc.
"""

from __future__ import annotations

import re
from typing import Any

from raggy_mcp.ingest.chunkers import Chunker
from raggy_mcp.ingest.chunkers.generic import GenericChunker
from raggy_mcp.ingest.types import Chunk, ExtractedDocument


# Patterns that indicate the start of a new top-level definition.
# Order matters — more-specific patterns checked first.
_DEF_PATTERNS: list[tuple[str, str, str]] = [
    # Python
    (r"^class\s+(\w+)", "class", "py"),
    (r"^async\s+def\s+(\w+)", "function", "py"),
    (r"^def\s+(\w+)", "function", "py"),
    # Rust
    (r"^fn\s+(\w+)", "function", "rs"),
    (r"^struct\s+(\w+)", "struct", "rs"),
    (r"^enum\s+(\w+)", "enum", "rs"),
    (r"^impl\s+(\w+)", "impl", "rs"),
    (r"^trait\s+(\w+)", "trait", "rs"),
    # TypeScript / JavaScript
    (r"^(export\s+)?(class|interface|type)\s+(\w+)", "declaration", "ts"),
    (r"^(export\s+)?function\s+(\w+)", "function", "ts"),
    # Java / C / C++
    (r"^\s*(public|private|protected|static)?\s*(class|interface|enum)\s+(\w+)", "class", "java"),
    # Go
    (r"^func\s+(\w+)", "function", "go"),
    (r"^type\s+(\w+)\s+", "type", "go"),
]

# Compiled regexes: (pattern, symbol_type)
_COMPILED: list[tuple[re.Pattern, str]] = [
    (re.compile(pat, re.MULTILINE), sym_type)
    for pat, sym_type, _lang in _DEF_PATTERNS
]


def _find_top_level_defs(lines: list[str]) -> list[dict[str, Any]]:
    """Scan lines and locate top-level definition boundaries."""
    defs: list[dict[str, Any]] = []
    for lineno, line in enumerate(lines):
        for pat, sym_type in _COMPILED:
            m = pat.match(line)
            if m:
                # Extract the symbol name — it's the last non-None group
                groups = m.groups()
                symbol_name = next((g for g in reversed(groups) if g is not None), "unknown")
                defs.append({
                    "line": lineno,
                    "symbol": symbol_name,
                    "type": sym_type,
                })
                break  # first match wins per line
    return defs


class CodeChunker(Chunker):
    """Splits source code by top-level definition boundaries."""

    def __init__(self, max_chunk_lines: int = 200):
        self._max_chunk_lines = max_chunk_lines
        self._fallback = GenericChunker()

    def chunk(self, doc: ExtractedDocument, metadata: dict[str, Any]) -> list[Chunk]:
        lines = doc.text.split("\n")
        if not lines or len(lines) < 3:
            # Too small for structure-aware splitting — use generic
            return self._fallback.chunk(doc, metadata)

        defs = _find_top_level_defs(lines)

        if not defs:
            # No recognizable definitions — delegate to generic
            return self._fallback.chunk(doc, metadata)

        # Split at definition boundaries
        boundaries = [d["line"] for d in defs]
        boundaries.append(len(lines))  # sentinel

        chunks: list[Chunk] = []
        for i, start_line in enumerate(boundaries[:-1]):
            end_line = boundaries[i + 1]
            section_lines = lines[start_line:end_line]

            # If a single definition is huge, sub-split by generic chunker
            if len(section_lines) > self._max_chunk_lines:
                sub_text = "\n".join(section_lines)
                sub_doc = ExtractedDocument(
                    text=sub_text,
                    extractor_used=doc.extractor_used,
                    char_count=len(sub_text),
                )
                sub_meta = {**metadata, "code_symbol": defs[i]["symbol"] if i < len(defs) else ""}
                sub_chunks = self._fallback.chunk(sub_doc, sub_meta)
                for sc in sub_chunks:
                    sc.metadata["code_symbol"] = defs[i]["symbol"] if i < len(defs) else ""
                    sc.metadata["code_type"] = defs[i]["type"] if i < len(defs) else ""
                chunks.extend(sub_chunks)
            else:
                text = "\n".join(section_lines)
                chunk_meta = {
                    **metadata,
                    "chunk_index": len(chunks),
                    "total_chunks": 0,  # set below
                    "code_symbol": defs[i]["symbol"] if i < len(defs) else "",
                    "code_type": defs[i]["type"] if i < len(defs) else "",
                }
                chunks.append(
                    Chunk(text=text, chunk_index=len(chunks), total_chunks=0, metadata=chunk_meta)
                )

        # Fix total_chunks after all chunks are built
        total = len(chunks)
        for c in chunks:
            c.total_chunks = total
            c.metadata["total_chunks"] = total
        return chunks