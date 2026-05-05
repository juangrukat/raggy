#!/usr/bin/env python3
"""Install/update the Hermes qdrant MCP entry for local server-mode launch."""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import yaml


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _qdrant_entry(project_root: Path) -> dict:
    return {
        "command": str(project_root / ".venv" / "bin" / "raggy-mcp"),
        "cwd": str(project_root),
        "args": [],
        "env": {
            "QDRANT_MODE": "server",
            "QDRANT_URL": "http://127.0.0.1:6333",
            "EMBEDDING_PROVIDER": "fastembed",
            "EMBEDDING_MODEL": "Qwen/Qwen3-Embedding-4B",
            "QWEN3_DEVICE": "auto",
            "QWEN3_MAX_LENGTH": "1024",
            "QWEN3_DTYPE": "auto",
            "QWEN3_SIDECAR_PATH": str(
                project_root
                / "rust"
                / "qwen3_embedder"
                / "target"
                / "release"
                / "qwen3-embedder"
            ),
            "QWEN3_METRICS_PATH": str(
                project_root / ".local" / "logs" / "qwen3-embeddings.jsonl"
            ),
            "QWEN3_RESPONSE_LIMIT_BYTES": "67108864",
            "QDRANT_EMBEDDING_BATCH_SIZE": "4",
            "QDRANT_INGEST_CHUNK_SIZE": "700",
            "QDRANT_INGEST_CHUNK_OVERLAP": "70",
            "QDRANT_WRITE_MAX_CONCURRENCY": "1",
            "QDRANT_WRITE_QUEUE_SIZE": "8",
            "PYTHONUNBUFFERED": "1",
        },
        "enabled": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(Path.home() / ".hermes" / "config.yaml"),
        help="Hermes config path.",
    )
    parser.add_argument("--name", default="qdrant", help="MCP server name to write.")
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create a timestamped backup before writing an existing config.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser()
    project_root = _project_root()
    entry = _qdrant_entry(project_root)

    if not (project_root / ".venv" / "bin" / "raggy-mcp").exists():
        raise SystemExit(
            "Missing .venv/bin/raggy-mcp. Run ./scripts/local-install.sh first."
        )

    if config_path.exists():
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not args.no_backup:
            stamp = datetime.now().strftime("%Y%m%d%H%M%S")
            shutil.copy2(config_path, config_path.with_suffix(f".yaml.{stamp}.bak"))
    else:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config = {}

    mcp_servers = config.setdefault("mcp_servers", {})
    mcp_servers[args.name] = entry

    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    print(f"Updated Hermes MCP server '{args.name}' in {config_path}")
    print(f"Command: {entry['command']}")
    print("Qdrant URL: http://127.0.0.1:6333")


if __name__ == "__main__":
    main()
