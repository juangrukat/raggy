"""Stable document identifier derived from absolute path."""

import hashlib
from pathlib import Path


def compute_document_id(path: str) -> str:
    """
    Stable, deterministic document_id derived from the canonical absolute path.
    Same file → same document_id across ingest runs, so chunk grouping works reliably.
    """
    canonical = str(Path(path).resolve())
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:16]
