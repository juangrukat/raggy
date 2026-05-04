#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_ROOT="${PROJECT_ROOT}/.local"

mkdir -p "${LOCAL_ROOT}/logs"

export QDRANT_MODE="${QDRANT_MODE:-embedded}"
export QDRANT_AUTO_DOCKER="${QDRANT_AUTO_DOCKER:-false}"
export QDRANT_MCP_TOOL_PROFILE="${QDRANT_MCP_TOOL_PROFILE:-canonical}"
export EMBEDDING_PROVIDER="${EMBEDDING_PROVIDER:-fastembed}"
export EMBEDDING_MODEL="${EMBEDDING_MODEL:-Qwen/Qwen3-Embedding-4B}"
export EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-auto}"
export QWEN3_DEVICE="${QWEN3_DEVICE:-${EMBEDDING_DEVICE}}"
export QWEN3_MAX_LENGTH="${QWEN3_MAX_LENGTH:-1024}"
export QWEN3_DTYPE="${QWEN3_DTYPE:-auto}"
export QWEN3_SIDECAR_PATH="${QWEN3_SIDECAR_PATH:-${PROJECT_ROOT}/rust/qwen3_embedder/target/release/qwen3-embedder}"
export QWEN3_METRICS_PATH="${QWEN3_METRICS_PATH:-${LOCAL_ROOT}/logs/qwen3-embeddings.jsonl}"
export QWEN3_RESPONSE_LIMIT_BYTES="${QWEN3_RESPONSE_LIMIT_BYTES:-67108864}"
export QDRANT_EMBEDDING_BATCH_SIZE="${QDRANT_EMBEDDING_BATCH_SIZE:-4}"
export QDRANT_INGEST_CHUNK_SIZE="${QDRANT_INGEST_CHUNK_SIZE:-700}"
export QDRANT_INGEST_CHUNK_OVERLAP="${QDRANT_INGEST_CHUNK_OVERLAP:-70}"
export QDRANT_WRITE_MAX_CONCURRENCY="${QDRANT_WRITE_MAX_CONCURRENCY:-1}"
export QDRANT_WRITE_QUEUE_SIZE="${QDRANT_WRITE_QUEUE_SIZE:-8}"
# Sparse model for hybrid retrieval. "Qdrant/bm25" (default, no download) or
# "Qdrant/bm42-all-minilm-l6-v2-attentions" (neural BM25 improvement, ~22 MB download).
export QDRANT_SPARSE_MODEL="${QDRANT_SPARSE_MODEL:-Qdrant/bm25}"
# Default reranker for mode="rerank". FastEmbed options need no extra deps;
# Qwen3 options require: uv pip install 'raggy-mcp[reranking]'
# Examples: "Xenova/ms-marco-MiniLM-L-6-v2", "BAAI/bge-reranker-base",
#           "Qwen/Qwen3-Reranker-4B", "Qwen/Qwen3-Reranker-0.6B"
export QDRANT_RERANKER_MODEL="${QDRANT_RERANKER_MODEL:-Xenova/ms-marco-MiniLM-L-6-v2}"
# Custom task instruction for Qwen3 rerankers (optional). Improves ranking quality
# ~1-5% when tailored to your retrieval task. Leave unset for the generic default.
# export QDRANT_RERANKER_INSTRUCTION="Retrieve passages that directly answer the question using explicit claims, definitions, or evidence from the document."
# Candidate pool size before grouping/reranking. 0 = auto (100 with reranker, 80 without).
export QDRANT_RERANK_PREFETCH_LIMIT="${QDRANT_RERANK_PREFETCH_LIMIT:-0}"
# Max candidates scored by the reranker. 0 = all prefetched.
export QDRANT_RERANK_TOP_K="${QDRANT_RERANK_TOP_K:-0}"
export MCP_HOST="${MCP_HOST:-127.0.0.1}"
export MCP_PORT="${MCP_PORT:-8000}"
export FASTMCP_PORT="${FASTMCP_PORT:-${MCP_PORT}}"

if [[ -n "${QDRANT_URL:-}" || "${QDRANT_MODE}" == "server" ]]; then
  export QDRANT_MODE="server"
  export QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:6333}"
  unset QDRANT_LOCAL_PATH
elif [[ "${QDRANT_MODE}" == "docker" ]]; then
  export QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:6333}"
  export QDRANT_DOCKER_STORAGE_PATH="${QDRANT_DOCKER_STORAGE_PATH:-${LOCAL_ROOT}/qdrant-server-storage}"
  unset QDRANT_LOCAL_PATH
else
  export QDRANT_MODE="embedded"
  export QDRANT_LOCAL_PATH="${QDRANT_LOCAL_PATH:-${LOCAL_ROOT}/qdrant-storage}"
  mkdir -p "${QDRANT_LOCAL_PATH}"
fi
