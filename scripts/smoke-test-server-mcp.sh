#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export QDRANT_MODE=server
export QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:6333}"
export MCP_START_TIMEOUT_SECONDS="${MCP_START_TIMEOUT_SECONDS:-30}"

if ! curl -fsS "${QDRANT_URL}/readyz" >/dev/null; then
  echo "fail  Qdrant server is not reachable at ${QDRANT_URL}" >&2
  echo "      Start it with: ./scripts/local-run-qdrant.sh" >&2
  exit 1
fi

echo "ok    Qdrant server is reachable at ${QDRANT_URL}"

uv run python - <<'PY'
import asyncio
import os

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

project_root = os.getcwd()
collection_name = os.environ.get("QDRANT_COLLECTION_NAME", "").strip()
timeout = float(os.environ["MCP_START_TIMEOUT_SECONDS"])


async def main() -> None:
    params = StdioServerParameters(
        command=os.path.join(project_root, "scripts", "run-server-mcp.sh"),
        args=[],
        env=dict(os.environ),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await asyncio.wait_for(session.initialize(), timeout=timeout)
            result = await asyncio.wait_for(session.call_tool("list_collections", {}), timeout=timeout)
            text = "\n".join(item.text for item in result.content if hasattr(item, "text"))
            print("ok    MCP starts in server mode and list_collections works")
            print("collections:")
            print(text or "[]")
            if collection_name and collection_name not in text:
                raise SystemExit(f"missing requested collection: {collection_name}")
            if collection_name:
                print(f"ok    requested collection is available: {collection_name}")


asyncio.run(main())
PY
