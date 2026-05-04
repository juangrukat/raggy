#!/usr/bin/env python3
"""Copy collections from Qdrant Python local mode into a Qdrant server."""

from __future__ import annotations

import argparse
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.migrate import migrate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate Qdrant local-mode collections into a running Qdrant server."
    )
    parser.add_argument(
        "--source-path",
        default=str(Path(__file__).resolve().parents[1] / ".local" / "qdrant-storage"),
        help="Existing Qdrant Python local-mode path.",
    )
    parser.add_argument(
        "--target-url",
        default="http://127.0.0.1:6333",
        help="Target Qdrant server URL.",
    )
    parser.add_argument(
        "--collection",
        action="append",
        dest="collections",
        help="Collection to migrate. Repeat for multiple collections. Defaults to all collections.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate target collections when they already exist.",
    )
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()

    source_path = Path(args.source_path).expanduser().resolve()
    if not source_path.exists():
        raise SystemExit(f"Source path does not exist: {source_path}")

    source = QdrantClient(path=str(source_path))
    target = QdrantClient(url=args.target_url)
    migrate(
        source,
        target,
        collection_names=args.collections,
        recreate_on_collision=args.recreate,
        batch_size=args.batch_size,
    )
    migrated = ", ".join(args.collections) if args.collections else "all collections"
    print(f"Migrated {migrated} from {source_path} to {args.target_url}.")


if __name__ == "__main__":
    main()
