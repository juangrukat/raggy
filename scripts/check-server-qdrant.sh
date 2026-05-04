#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:6333}"

if ! curl -fsS "${QDRANT_URL}/readyz" >/dev/null; then
  echo "fail  Qdrant server is not reachable at ${QDRANT_URL}" >&2
  echo "      Start it with: ./scripts/local-run-qdrant.sh" >&2
  exit 1
fi

echo "ok    Qdrant server is reachable at ${QDRANT_URL}"

uv run python - <<'PY'
import os

from qdrant_client import QdrantClient

url = os.environ["QDRANT_URL"]
client = QdrantClient(url=url)


def vector_summary(vectors) -> str:
    if not vectors:
        return "none"
    if isinstance(vectors, dict):
        parts = []
        for name, vector in vectors.items():
            size = getattr(vector, "size", "?")
            distance = getattr(vector, "distance", "")
            distance_value = getattr(distance, "value", distance)
            parts.append(f"{name}:{size}:{distance_value}")
        return ", ".join(parts)
    size = getattr(vectors, "size", "?")
    distance = getattr(vectors, "distance", "")
    distance_value = getattr(distance, "value", distance)
    return f"default:{size}:{distance_value}"


def sparse_summary(sparse_vectors) -> str:
    if isinstance(sparse_vectors, dict) and sparse_vectors:
        return ", ".join(sparse_vectors.keys())
    return "none"


try:
    collections = client.get_collections().collections
    if not collections:
        print("collections: none")
    for item in collections:
        info = client.get_collection(item.name)
        params = info.config.params
        sparse_vectors = params.sparse_vectors or {}
        kind = "hybrid" if isinstance(sparse_vectors, dict) and sparse_vectors else "dense-only"
        points = info.points_count or 0
        print(f"collection={item.name}")
        print(f"  type={kind}")
        print(f"  points={points}")
        print(f"  dense_vectors={vector_summary(params.vectors)}")
        print(f"  sparse_vectors={sparse_summary(sparse_vectors)}")
finally:
    client.close()
PY
