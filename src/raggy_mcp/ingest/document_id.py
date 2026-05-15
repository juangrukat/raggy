"""Stable identifiers for document content, sources, and versions."""

import hashlib
from pathlib import Path


def _sha256_short(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]


def normalize_document_text(text: str) -> str:
    """Normalize extracted text before hashing so trivial whitespace drift is ignored."""
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def compute_document_id(text: str, *, path: str | None = None) -> str:
    """
    Stable document_id derived from normalized extracted content.

    Same content in two paths gets the same document_id. `path` is accepted only
    as a compatibility fallback for empty/unextractable content.
    """
    normalized = normalize_document_text(text)
    if normalized:
        return _sha256_short(normalized)
    if path is None:
        return _sha256_short("")
    return _sha256_short(str(Path(path).resolve()))


def compute_source_id(path: str) -> str:
    """Stable source_id derived from the canonical absolute path."""
    return _sha256_short(str(Path(path).resolve()))


def compute_version_id(text: str, metadata_timestamp: str | None = None) -> str:
    """Version id that changes when content or source metadata timestamp changes."""
    normalized = normalize_document_text(text)
    return _sha256_short(f"{normalized}\n{metadata_timestamp or ''}")
