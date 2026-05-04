"""
macOS file metadata extraction via mdls (Spotlight) and xattr.
Produces a normalized dict suitable for Qdrant payload storage.
"""

import asyncio
import os
import plistlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MAX_METADATA_PROCS = int(os.environ.get("QDRANT_METADATA_MAX_PROCS", "8"))
_metadata_semaphore: asyncio.Semaphore | None = None


def _base_metadata(p: Path) -> dict[str, Any]:
    return {
        "path": str(p),
        "filename": p.name,
        "extension": p.suffix.lstrip(".").lower() if p.suffix else "",
        "size_bytes": p.stat().st_size if p.exists() else 0,
        "is_hidden": p.name.startswith("."),
        "has_text": False,
    }


def get_macos_metadata(path: str) -> dict[str, Any]:
    """
    Extract macOS file metadata using mdls (Spotlight) with xattr fallback.
    Returns a normalized dict with stable field names for Qdrant indexing.
    """
    p = Path(path).resolve()
    meta: dict[str, Any] = _base_metadata(p)

    spotlight = _run_mdls(str(p))
    if spotlight:
        meta.update(_normalize_spotlight(spotlight))

    tags = _get_finder_tags(str(p))
    if tags:
        meta["tags"] = tags
    elif "tags" not in meta:
        meta["tags"] = []

    return meta


async def get_macos_metadata_async(path: str) -> dict[str, Any]:
    """
    Async metadata extraction for MCP handlers. Uses asyncio subprocesses and
    bounded concurrency to avoid blocking the event loop during folder ingest.
    """
    p = Path(path).resolve()
    meta = _base_metadata(p)

    spotlight = await _run_mdls_async(str(p))
    if spotlight:
        meta.update(_normalize_spotlight(spotlight))

    tags = await _get_finder_tags_async(str(p))
    if tags:
        meta["tags"] = tags
    elif "tags" not in meta:
        meta["tags"] = []

    return meta


def _run_mdls(path: str) -> dict[str, Any] | None:
    try:
        result = subprocess.run(
            ["mdls", "-plist", "-", path],
            capture_output=True,
            timeout=10,
        )
        if result.returncode != 0 or not result.stdout:
            return None
        return plistlib.loads(result.stdout)
    except Exception:
        return None


def _get_metadata_semaphore() -> asyncio.Semaphore:
    global _metadata_semaphore
    if _metadata_semaphore is None:
        _metadata_semaphore = asyncio.Semaphore(MAX_METADATA_PROCS)
    return _metadata_semaphore


async def _run_command_async(*args: str, timeout: float = 10) -> tuple[bytes, int]:
    async with _get_metadata_semaphore():
        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            return stdout, proc.returncode or 0
        except Exception:
            return b"", 1


async def _run_mdls_async(path: str) -> dict[str, Any] | None:
    stdout, returncode = await _run_command_async("mdls", "-plist", "-", path, timeout=10)
    if returncode != 0 or not stdout:
        return None
    try:
        return plistlib.loads(stdout)
    except Exception:
        return None


def _normalize_spotlight(raw: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}

    def _str(key: str) -> str | None:
        v = raw.get(key)
        return str(v) if v and v != "(null)" else None

    def _list(key: str) -> list[str]:
        v = raw.get(key)
        if isinstance(v, list):
            return [str(i) for i in v if i]
        return []

    def _date(key: str) -> str | None:
        v = raw.get(key)
        if isinstance(v, datetime):
            return v.astimezone(timezone.utc).isoformat()
        if isinstance(v, str) and v != "(null)":
            return v
        return None

    if v := _str("kMDItemContentType"):
        out["content_type"] = v
    if v := _str("kMDItemTitle"):
        out["title"] = v
    if v := _list("kMDItemAuthors"):
        out["authors"] = v
    if v := _list("kMDItemKeywords"):
        out["keywords"] = v
    if v := _list("kMDItemUserTags"):
        out["tags"] = v
    if v := _str("kMDItemComment"):
        out["comment"] = v
    if v := _list("kMDItemWhereFroms"):
        out["source_urls"] = v
    if v := _date("kMDItemFSCreationDate"):
        out["created_at"] = v
    if v := _date("kMDItemFSContentChangeDate"):
        out["modified_at"] = v
    if v := _date("kMDItemLastUsedDate"):
        out["last_opened_at"] = v

    pages = raw.get("kMDItemNumberOfPages")
    if pages is not None:
        out["page_count"] = int(pages)

    duration = raw.get("kMDItemDurationSeconds")
    if duration is not None:
        out["duration_seconds"] = float(duration)

    return out


def _get_finder_tags(path: str) -> list[str]:
    """Read Finder color/text tags from extended attributes."""
    try:
        result = subprocess.run(
            ["xattr", "-p", "com.apple.metadata:_kMDItemUserTags", path],
            capture_output=True,
            timeout=5,
        )
        if result.returncode != 0 or not result.stdout:
            return []
        # xattr -p outputs hex; use xattr -px to get raw hex then decode
        result2 = subprocess.run(
            ["xattr", "-px", "com.apple.metadata:_kMDItemUserTags", path],
            capture_output=True,
            timeout=5,
        )
        if result2.returncode != 0:
            return []
        hex_str = result2.stdout.decode().replace(" ", "").replace("\n", "")
        raw_bytes = bytes.fromhex(hex_str)
        tags_list = plistlib.loads(raw_bytes)
        # Each tag may be "TagName\n6" (name + newline + color number) — strip the color suffix
        return [t.split("\n")[0] for t in tags_list if isinstance(t, str)]
    except Exception:
        return []


async def _get_finder_tags_async(path: str) -> list[str]:
    stdout, returncode = await _run_command_async(
        "xattr", "-px", "com.apple.metadata:_kMDItemUserTags", path, timeout=5
    )
    if returncode != 0 or not stdout:
        return []
    try:
        hex_str = stdout.decode().replace(" ", "").replace("\n", "")
        raw_bytes = bytes.fromhex(hex_str)
        tags_list = plistlib.loads(raw_bytes)
        return [t.split("\n")[0] for t in tags_list if isinstance(t, str)]
    except Exception:
        return []


# Qdrant payload index definitions for macOS metadata fields
# Maps "<metadata_field_path>" -> qdrant PayloadSchemaType string
MACOS_INDEX_FIELDS: list[tuple[str, str]] = [
    # Identity
    ("metadata.document_id", "keyword"),
    ("metadata.path", "keyword"),
    ("metadata.parent_path", "keyword"),
    ("metadata.filename", "keyword"),
    ("metadata.extension", "keyword"),
    ("metadata.content_type", "keyword"),
    # Filter fields
    ("metadata.tags", "keyword"),
    ("metadata.authors", "keyword"),
    ("metadata.keywords", "keyword"),
    ("metadata.size_bytes", "integer"),
    ("metadata.is_hidden", "bool"),
    ("metadata.has_text", "bool"),
    ("metadata.created_at", "keyword"),
    ("metadata.modified_at", "keyword"),
    ("metadata.last_opened_at", "keyword"),
    ("metadata.page_count", "integer"),
    # Runtime stats
    ("metadata.extractor_used", "keyword"),
    ("metadata.char_count", "integer"),
    ("metadata.ingested_at", "keyword"),
]
