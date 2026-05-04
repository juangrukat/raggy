#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export QDRANT_MODE=server
export QDRANT_URL=http://127.0.0.1:6333

exec "${PROJECT_ROOT}/scripts/local-run-mcp.sh" "$@"
