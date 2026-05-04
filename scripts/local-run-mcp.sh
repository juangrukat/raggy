#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ -z "${QDRANT_MODE:-}" && -z "${QDRANT_URL:-}" ]]; then
  cat >&2 <<'EOF'
Warning: Defaulting to embedded mode. For the standard Docker/server setup, run ./scripts/run-server-mcp.sh
EOF
fi

source "${PROJECT_ROOT}/scripts/local-env.sh"

exec uv run --locked raggy-mcp "$@"
