#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:6333}"
VENV_MCP="${PROJECT_ROOT}/.venv/bin/raggy-mcp"
SIDECAR="${PROJECT_ROOT}/rust/qwen3_embedder/target/release/qwen3-embedder"

ok() { printf 'ok    %s\n' "$1"; }
warn() { printf 'warn  %s\n' "$1"; }
fail() { printf 'fail  %s\n' "$1"; }

if [[ -x "${VENV_MCP}" ]]; then
  ok "Hermes-safe MCP binary exists: ${VENV_MCP}"
else
  fail "Missing ${VENV_MCP}; run ./scripts/local-install.sh"
fi

if [[ -x "${SIDECAR}" ]]; then
  ok "Qwen3 sidecar exists: ${SIDECAR}"
else
  warn "Missing Qwen3 sidecar; run ./scripts/local-install.sh before Qwen3 embedding"
fi

if command -v docker >/dev/null 2>&1; then
  ok "Docker CLI is installed"
  if docker info >/dev/null 2>&1; then
    ok "Docker daemon is reachable"
    if docker ps --filter name=qdrant_mcp_server --format '{{.Names}}' | grep -qx qdrant_mcp_server; then
      ok "qdrant_mcp_server container is running"
    else
      warn "qdrant_mcp_server is not running; run ./scripts/local-run-qdrant.sh"
    fi
  else
    fail "Docker daemon is not reachable; start Docker Desktop or your Docker service"
  fi
else
  fail "Docker CLI is not installed"
fi

if curl -fsS "${QDRANT_URL}/readyz" >/dev/null 2>&1; then
  ok "Qdrant server is ready at ${QDRANT_URL}"
  curl -fsS "${QDRANT_URL}/collections" || true
  printf '\n'
else
  fail "Qdrant server is not reachable at ${QDRANT_URL}"
fi

if command -v hermes >/dev/null 2>&1; then
  ok "Hermes CLI is installed"
  if hermes mcp test qdrant >/dev/null 2>&1; then
    ok "Hermes qdrant MCP connects"
  else
    warn "Hermes qdrant MCP test failed; run ./scripts/local-configure-hermes.py and retry"
  fi
else
  warn "Hermes CLI is not installed or not on PATH"
fi
