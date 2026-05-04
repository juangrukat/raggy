"""
Entry point for the `raggy-mcp` CLI.

Supports two MCP transports:

* ``stdio`` (default) — single-client, per-process. Best for Claude Desktop,
  LM Studio, or any client that spawns the server.
* ``streamable-http`` — long-lived local server that multiple agents can share
  on the same machine. Binds to ``127.0.0.1`` by default, validates the
  ``Origin`` header to mitigate DNS rebinding, and supports an optional bearer
  token via ``MCP_HTTP_AUTH_TOKEN``.

Configuration precedence: CLI flag > environment variable > default.
Env vars: ``MCP_TRANSPORT``, ``MCP_HOST``, ``MCP_PORT``, ``MCP_HTTP_AUTH_TOKEN``,
``MCP_HTTP_ALLOWED_ORIGINS``.
"""

import argparse
import os
import signal
import sys

from raggy_mcp.docker_utils import start_qdrant_container, stop_qdrant_container


def _env_or(default: str, *names: str) -> str:
    for n in names:
        v = os.getenv(n)
        if v:
            return v
    return default


def main() -> None:
    parser = argparse.ArgumentParser(description="raggy-mcp")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http", "http"],
        default=_env_or("stdio", "MCP_TRANSPORT"),
        help="MCP transport (default: stdio). 'http' is an alias for 'streamable-http'.",
    )
    parser.add_argument(
        "--host",
        default=_env_or("127.0.0.1", "MCP_HOST"),
        help="HTTP bind address (default: 127.0.0.1; only loopback is recommended).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(_env_or("8000", "MCP_PORT", "FASTMCP_PORT")),
        help="HTTP port (default: 8000).",
    )
    parser.add_argument(
        "--auth-token",
        default=os.getenv("MCP_HTTP_AUTH_TOKEN"),
        help="Bearer token required on every HTTP request when set.",
    )
    args = parser.parse_args()

    # Normalize transport alias
    transport = "streamable-http" if args.transport == "http" else args.transport

    # Auto-start local Qdrant container (no-op if external Qdrant is configured).
    start_qdrant_container()

    def signal_handler(sig, frame):
        print(f"\nReceived signal {sig}, shutting down gracefully...", file=sys.stderr)
        stop_qdrant_container()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Import after env is in place so settings pick up the right values.
    from raggy_mcp.server import mcp

    try:
        if transport == "stdio":
            mcp.run(transport="stdio")
        elif transport == "sse":
            mcp.run(transport="sse", host=args.host, port=args.port)
        else:
            # streamable-http: attach Origin validation + optional Bearer auth middleware
            from raggy_mcp.mcp_runtime.http_security import build_middleware_stack

            middleware = build_middleware_stack(auth_token=args.auth_token)
            print(
                f"[raggy-mcp] Streamable HTTP on http://{args.host}:{args.port}/mcp/ "
                f"(auth={'on' if args.auth_token or os.getenv('MCP_HTTP_AUTH_TOKEN') else 'off'})",
                file=sys.stderr,
            )

            # Build the app with middleware then run via uvicorn
            import uvicorn
            app = mcp.http_app(middleware=middleware, transport="streamable-http")
            uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected, shutting down...", file=sys.stderr)
        stop_qdrant_container()
        sys.exit(0)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        stop_qdrant_container()
        sys.exit(1)


if __name__ == "__main__":
    main()
