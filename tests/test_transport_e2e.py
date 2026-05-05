# ruff: noqa: E402

import asyncio
import os
import socket
import sys
import threading
import time

import pytest

from raggy_mcp._warnings import filter_upstream_warnings
from raggy_mcp.mcp_server import QdrantMCPServer
from raggy_mcp.settings import EmbeddingProviderSettings, QdrantSettings, ToolSettings

filter_upstream_warnings()

from fastmcp import Client
from fastmcp.client.transports import StdioTransport


def _test_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "QDRANT_URL": ":memory:",
            "QDRANT_LOCAL_PATH": "",
            "QDRANT_MODE": "embedded",
            "QDRANT_AUTO_DOCKER": "false",
            "QDRANT_MCP_TOOL_PROFILE": "canonical",
        }
    )
    return env


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    try:
        return sock.getsockname()[1]
    finally:
        sock.close()


def _wait_for_port(port: int, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.05)
    raise TimeoutError(f"Timed out waiting for port {port}")


@pytest.mark.asyncio
async def test_stdio_transport_lists_tools():
    transport = StdioTransport(
        command=sys.executable,
        args=["-m", "raggy_mcp.main", "--transport", "stdio"],
        env=_test_env(),
        cwd=str(os.getcwd()),
    )

    async with Client(transport, init_timeout=15, timeout=15) as client:
        tools = await client.list_tools()

    names = {tool.name for tool in tools}
    assert "search_documents" in names
    assert "ingest_file" in names
    assert "create_late_interaction_collection" in names


@pytest.mark.asyncio
async def test_streamable_http_supports_concurrent_clients():
    import uvicorn

    port = _free_port()
    server = QdrantMCPServer(
        ToolSettings(),
        QdrantSettings(QDRANT_URL=":memory:", QDRANT_LOCAL_PATH=None),
        EmbeddingProviderSettings(),
        name="transport-http-test",
    )
    app = server.http_app(transport="streamable-http")
    config = uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="warning", ws="none"
    )
    uvicorn_server = uvicorn.Server(config)
    thread = threading.Thread(target=uvicorn_server.run, daemon=True)
    thread.start()
    _wait_for_port(port)

    async def list_tool_names() -> set[str]:
        async with Client(
            f"http://127.0.0.1:{port}/mcp/", init_timeout=15, timeout=15
        ) as client:
            tools = await client.list_tools()
            return {tool.name for tool in tools}

    try:
        results = await asyncio.gather(*(list_tool_names() for _ in range(3)))
    finally:
        uvicorn_server.should_exit = True
        thread.join(timeout=5)

    assert all("search_documents" in names for names in results)
    assert all("create_late_interaction_collection" in names for names in results)
