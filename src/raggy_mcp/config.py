"""
Configuration file loader for raggy-mcp.

Resolution order (first found wins):
  1. $QDRANT_CONFIG environment variable
  2. ./raggy.yaml (relative to cwd)
  3. ~/.config/raggy-mcp/config.yaml

Values from the config file are injected as environment variables *only if
the variable is not already set*. This preserves the priority chain:

  CLI args / request params
  → environment variables
  → config file
  → built-in defaults

Call ``load_qdrant_config()`` before instantiating any Pydantic Settings
objects. It is safe to call multiple times (subsequent calls are no-ops).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_loaded = False

# ── Config file discovery ──────────────────────────────────────────────────

def _find_config_file() -> Path | None:
    """Return the first existing config file path, or None."""
    # 1. Explicit env var
    env_path = os.getenv("QDRANT_CONFIG")
    if env_path:
        p = Path(env_path).expanduser().resolve()
        if p.exists():
            logger.info("Using config from QDRANT_CONFIG: %s", p)
            return p
        logger.warning("QDRANT_CONFIG=%s does not exist, falling back", env_path)

    # 2. Repo root / cwd
    cwd_path = Path("raggy.yaml").resolve()
    if cwd_path.exists():
        logger.info("Using config: %s", cwd_path)
        return cwd_path

    # 3. User config directory
    user_path = Path("~/.config/raggy-mcp/config.yaml").expanduser()
    if user_path.exists():
        logger.info("Using config: %s", user_path)
        return user_path

    return None


# ── YAML → env mapping ─────────────────────────────────────────────────────

# Maps YAML paths (dotted) to environment variable names.
# Only top-level sections relevant to current env vars are mapped;
# benchmark / collection-naming sections are informational.
_YAML_TO_ENV: dict[str, str] = {
    # runtime
    "runtime.qdrant_mode":              "QDRANT_MODE",
    "runtime.qdrant_url":               "QDRANT_URL",
    "runtime.qdrant_local_path":        "QDRANT_LOCAL_PATH",
    "runtime.mcp_tool_profile":         "QDRANT_MCP_TOOL_PROFILE",
    "runtime.read_only":                "QDRANT_READ_ONLY",
    # models
    "models.dense_embedding":           "EMBEDDING_MODEL",
    "models.embedding_provider":        "EMBEDDING_PROVIDER",
    "models.embedding_device":          "EMBEDDING_DEVICE",
    "models.sparse_embedding":          "QDRANT_SPARSE_MODEL",
    "models.reranker":                  "QDRANT_RERANKER_MODEL",
    "models.reranker_instruction":      "QDRANT_RERANKER_INSTRUCTION",
    "models.reranker_providers":        "QDRANT_RERANKER_PROVIDERS",
    "models.qwen3.max_length":          "QWEN3_MAX_LENGTH",
    "models.qwen3.dtype":               "QWEN3_DTYPE",
    "models.qwen3.response_limit_bytes":"QWEN3_RESPONSE_LIMIT_BYTES",
    # ingest
    "ingest.chunk_size":                "QDRANT_INGEST_CHUNK_SIZE",
    "ingest.chunk_overlap":             "QDRANT_INGEST_CHUNK_OVERLAP",
    "ingest.embedding_batch_size":      "QDRANT_EMBEDDING_BATCH_SIZE",
    "ingest.write_max_concurrency":     "QDRANT_WRITE_MAX_CONCURRENCY",
    "ingest.write_queue_size":          "QDRANT_WRITE_QUEUE_SIZE",
    # search
    "search.default_mode":              "QDRANT_DEFAULT_SEARCH_MODE",
    "search.rerank_prefetch_limit":     "QDRANT_RERANK_PREFETCH_LIMIT",
    "search.rerank_top_k":              "QDRANT_RERANK_TOP_K",
    # collections
    "collections.default_collection":   "COLLECTION_NAME",
}


def _get_nested(data: dict, path: str) -> Any:
    """Get a nested dict value by dotted path. Returns None if missing."""
    keys = path.split(".")
    for key in keys:
        if not isinstance(data, dict):
            return None
        data = data.get(key)
        if data is None:
            return None
    return data


def _inject_config(config: dict) -> int:
    """Inject YAML values into environment as overrides.

    Only sets env vars that are NOT already defined, preserving the
    priority chain: existing env > config file > built-in defaults.

    Returns the number of env vars set.
    """
    count = 0
    for yaml_path, env_var in _YAML_TO_ENV.items():
        if env_var in os.environ:
            continue  # already set — higher priority
        value = _get_nested(config, yaml_path)
        if value is None:
            continue
        # Convert lists to comma-separated strings
        if isinstance(value, list):
            value = ",".join(str(v) for v in value)
        elif isinstance(value, bool):
            value = str(value).lower()
        else:
            value = str(value)
        os.environ[env_var] = value
        count += 1
        logger.debug("config: %s → %s=%s", yaml_path, env_var, value)
    return count


# ── Public API ─────────────────────────────────────────────────────────────

def load_qdrant_config() -> bool:
    """Load configuration from the first-found YAML file.

    Safe to call multiple times. Returns True if a config was loaded.
    Call this before instantiating any Pydantic Settings objects.
    """
    global _loaded
    if _loaded:
        return True
    _loaded = True

    config_path = _find_config_file()
    if config_path is None:
        logger.info("No raggy.yaml found — using env vars and defaults only")
        return False

    try:
        raw = config_path.read_text(encoding="utf-8")
        config = yaml.safe_load(raw)
        if not isinstance(config, dict):
            logger.warning("Config file %s is not a valid mapping", config_path)
            return False
        count = _inject_config(config)
        logger.info("Loaded %d settings from %s", count, config_path)
        return True
    except Exception as exc:
        logger.warning("Failed to load config from %s: %s", config_path, exc)
        return False
