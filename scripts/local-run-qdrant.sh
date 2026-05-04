#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_ROOT="${PROJECT_ROOT}/.local"
STORAGE_PATH="${QDRANT_DOCKER_STORAGE_PATH:-${LOCAL_ROOT}/qdrant-server-storage}"
CONTAINER_NAME="${QDRANT_CONTAINER_NAME:-qdrant_mcp_server}"
IMAGE="${QDRANT_IMAGE:-qdrant/qdrant:v1.17.1}"
QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:6333}"

# ---------------------------------------------------------------------------
# If Qdrant is already listening on the target port, do nothing.
# This handles both the LaunchAgent-managed native binary and a
# previously started Docker container.
# ---------------------------------------------------------------------------
if curl -fsS "${QDRANT_URL}/readyz" >/dev/null 2>&1; then
  echo "Qdrant server is already running at ${QDRANT_URL}."
  exit 0
fi

# ---------------------------------------------------------------------------
# Default path: start the native Qdrant binary (recommended).
# The LaunchAgent at ~/Library/LaunchAgents/com.qdrant.server.plist
# handles auto-start on login; this script is a manual fallback.
# ---------------------------------------------------------------------------
NATIVE_QDRANT="${PROJECT_ROOT}/.local/bin/qdrant"
if [[ -x "${NATIVE_QDRANT}" ]]; then
  mkdir -p "${STORAGE_PATH}"
  echo "Starting native Qdrant server at ${QDRANT_URL}..."
  exec "${NATIVE_QDRANT}" \
    --config-path "${LOCAL_ROOT}/qdrant-config.yaml" \
    --uri "${QDRANT_URL}" \
    --disable-telemetry
fi

# ---------------------------------------------------------------------------
# Fallback: Docker. Only attempted if the native binary isn't installed.
# ---------------------------------------------------------------------------
if ! docker info >/dev/null 2>&1; then
  cat >&2 <<'EOF'
Docker is installed, but the Docker daemon is not reachable.
Install the native Qdrant binary (recommended) or start Docker Desktop:

  Native: download from https://github.com/qdrant/qdrant/releases
  Docker: open Docker Desktop, then rerun this script.
EOF
  exit 1
fi

mkdir -p "${STORAGE_PATH}"

if docker inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  if [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}")" == "true" ]]; then
    echo "Qdrant is already running in Docker (${CONTAINER_NAME})."
    exit 0
  fi
  docker rm "${CONTAINER_NAME}" >/dev/null
fi

docker run -d \
  --name "${CONTAINER_NAME}" \
  -p 6333:6333 \
  -p 6334:6334 \
  -v "${STORAGE_PATH}:/qdrant/storage" \
  "${IMAGE}" >/dev/null

echo "Qdrant server started at ${QDRANT_URL} (Docker)."
echo "Server storage: ${STORAGE_PATH}"
