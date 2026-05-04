#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

source "${PROJECT_ROOT}/scripts/local-env.sh"

uv sync --frozen --group dev
cargo build --release --manifest-path "${PROJECT_ROOT}/rust/qwen3_embedder/Cargo.toml"

cat <<EOF
Local install is ready.

Runtime state:
  ${LOCAL_ROOT}

Qdrant mode:
  ${QDRANT_MODE}
EOF

if [[ "${QDRANT_MODE}" == "embedded" ]]; then
  echo "Embedded Qdrant storage:"
  echo "  ${QDRANT_LOCAL_PATH}"
else
  echo "Qdrant URL:"
  echo "  ${QDRANT_URL}"
fi

cat <<EOF

Hermes setup:
  ./scripts/local-configure-hermes.py

Health check:
  ./scripts/local-doctor.sh
EOF
