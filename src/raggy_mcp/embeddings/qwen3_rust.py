import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any

from raggy_mcp.embeddings.base import EmbeddingProvider


KNOWN_QWEN3_DIMS: dict[str, int] = {
    "Qwen/Qwen3-Embedding-0.6B": 1024,
    "Qwen/Qwen3-Embedding-4B": 2560,
    "Qwen/Qwen3-Embedding-8B": 4096,
}


class Qwen3RustProvider(EmbeddingProvider):
    """Qwen3 embeddings through the Rust fastembed qwen3/Candle sidecar."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "auto",
        max_length: int = 1024,
        dtype: str = "auto",
        binary_path: str | None = None,
        metrics_path: str | None = None,
        response_limit_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.dtype = dtype
        self.binary_path = Path(binary_path) if binary_path else self._default_binary_path()
        self.metrics_path = Path(metrics_path) if metrics_path else self._default_metrics_path()
        self.response_limit_bytes = response_limit_bytes
        self._process: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._ready: dict[str, Any] | None = None

    async def warm_up(self) -> None:
        """Pre-start the sidecar subprocess so the first real request has no cold start."""
        await self._ensure_process()

    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        passages = [self._document_input(document) for document in documents]
        response = await self._request(
            {"type": "embed_documents", "texts": passages},
            operation="embed_documents",
            text_count=len(documents),
            char_count=sum(len(document) for document in documents),
        )
        return response["embeddings"]

    async def embed_query(self, query: str) -> list[float]:
        response = await self._request(
            {"type": "embed_query", "text": query},
            operation="embed_query",
            text_count=1,
            char_count=len(query),
        )
        return response["embeddings"][0]

    def get_vector_name(self) -> str:
        model_name = self.model_name.split("/")[-1].lower()
        return f"qwen3-{model_name}"

    def get_vector_size(self) -> int:
        if self._ready and "vector_size" in self._ready:
            return int(self._ready["vector_size"])
        if self.model_name in KNOWN_QWEN3_DIMS:
            return KNOWN_QWEN3_DIMS[self.model_name]
        raise ValueError(f"Cannot determine vector size for model: {self.model_name}")

    def get_model_name(self) -> str:
        return self.model_name

    @staticmethod
    def _document_input(document: str) -> str:
        return document if document.startswith("passage: ") else f"passage: {document}"

    async def _request(
        self,
        payload: dict[str, Any],
        *,
        operation: str,
        text_count: int,
        char_count: int,
    ) -> dict[str, Any]:
        total_started_at = time.perf_counter()
        cold_start_ms = 0
        request_ms = 0
        success = False
        error: str | None = None
        response: dict[str, Any] | None = None
        async with self._lock:
            try:
                process_started_at = time.perf_counter()
                process, started = await self._ensure_process()
                if started:
                    cold_start_ms = self._elapsed_ms(process_started_at)

                assert process.stdin is not None
                assert process.stdout is not None
                request_started_at = time.perf_counter()
                process.stdin.write(json.dumps(payload).encode("utf-8") + b"\n")
                await process.stdin.drain()
                line = await process.stdout.readline()
                request_ms = self._elapsed_ms(request_started_at)
                if not line:
                    raise RuntimeError("Qwen3 sidecar exited without a response")
                response = json.loads(line)
                if response.get("type") == "error":
                    raise RuntimeError(response.get("message", "Qwen3 sidecar error"))
                success = True
                return response
            except Exception as exc:
                error = str(exc)
                raise
            finally:
                self._record_metrics(
                    operation=operation,
                    text_count=text_count,
                    char_count=char_count,
                    cold_start_ms=cold_start_ms,
                    request_ms=request_ms,
                    total_ms=self._elapsed_ms(total_started_at),
                    success=success,
                    error=error,
                    embedding_count=len(response.get("embeddings", [])) if response else 0,
                )

    async def _ensure_process(self) -> tuple[asyncio.subprocess.Process, bool]:
        if self._process and self._process.returncode is None:
            return self._process, False
        if not self.binary_path.exists():
            raise RuntimeError(
                f"Qwen3 sidecar binary not found at {self.binary_path}. "
                "Run `cargo build --release --manifest-path rust/qwen3_embedder/Cargo.toml`."
            )

        env = os.environ.copy()
        env.setdefault("QWEN3_EMBEDDING_MODEL", self.model_name)
        env.setdefault("QWEN3_DEVICE", self.device)
        env.setdefault("QWEN3_MAX_LENGTH", str(self.max_length))
        if self.dtype != "auto":
            env.setdefault("QWEN3_DTYPE", self.dtype)

        self._process = await asyncio.create_subprocess_exec(
            str(self.binary_path),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            limit=self.response_limit_bytes,
        )
        assert self._process.stdout is not None
        line = await self._process.stdout.readline()
        if not line:
            stderr = b""
            if self._process.stderr is not None:
                stderr = await self._process.stderr.read()
            raise RuntimeError(f"Qwen3 sidecar failed to start: {stderr.decode('utf-8', 'replace')}")
        ready = json.loads(line)
        if ready.get("type") == "error":
            raise RuntimeError(ready.get("message", "Qwen3 sidecar failed to start"))
        self._ready = ready
        return self._process, True

    def _record_metrics(
        self,
        *,
        operation: str,
        text_count: int,
        char_count: int,
        cold_start_ms: int,
        request_ms: int,
        total_ms: int,
        success: bool,
        error: str | None,
        embedding_count: int,
    ) -> None:
        if not self.metrics_path:
            return

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "operation": operation,
            "model": self.model_name,
            "requested_device": self.device,
            "backend": self._ready.get("backend") if self._ready else None,
            "requested_dtype": self.dtype,
            "dtype": self._ready.get("dtype") if self._ready else None,
            "max_length": self.max_length,
            "vector_size": self.get_vector_size(),
            "text_count": text_count,
            "char_count": char_count,
            "embedding_count": embedding_count,
            "cold_start_ms": cold_start_ms,
            "request_ms": request_ms,
            "total_ms": total_ms,
            "success": success,
        }
        if error:
            record["error"] = error

        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with self.metrics_path.open("a", encoding="utf-8") as metrics_file:
            metrics_file.write(json.dumps(record, sort_keys=True) + "\n")

    @staticmethod
    def _elapsed_ms(started_at: float) -> int:
        return int((time.perf_counter() - started_at) * 1000)

    @staticmethod
    def _default_binary_path() -> Path:
        repo_root = Path(__file__).resolve().parents[3]
        release = repo_root / "rust" / "qwen3_embedder" / "target" / "release" / "qwen3-embedder"
        debug = repo_root / "rust" / "qwen3_embedder" / "target" / "debug" / "qwen3-embedder"
        return release if release.exists() else debug

    @staticmethod
    def _default_metrics_path() -> Path:
        repo_root = Path(__file__).resolve().parents[3]
        return repo_root / ".local" / "logs" / "qwen3-embeddings.jsonl"
