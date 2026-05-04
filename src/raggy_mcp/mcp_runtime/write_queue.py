from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


class WriteQueueFullError(RuntimeError):
    """Raised when the local write queue has no room for another job."""


@dataclass(frozen=True)
class WriteQueueStats:
    max_concurrency: int
    max_queue_size: int
    running: int
    waiting: int


class WriteQueue:
    """Bounded in-process queue for embedding/upsert work.

    Qdrant server can safely accept concurrent clients, but local embedding and
    book ingestion can overwhelm a laptop. This queue provides app-level
    backpressure before expensive vector writes begin.
    """

    def __init__(self, *, max_concurrency: int = 1, max_queue_size: int = 8):
        self.max_concurrency = max(1, max_concurrency)
        self.max_queue_size = max(0, max_queue_size)
        self._semaphore = asyncio.Semaphore(self.max_concurrency)
        self._lock = asyncio.Lock()
        self._running = 0
        self._waiting = 0

    async def run(self, name: str, operation: Callable[[], Awaitable[T]]) -> T:
        async with self._lock:
            pending = self._running + self._waiting
            if pending >= self.max_concurrency + self.max_queue_size:
                raise WriteQueueFullError(
                    f"Write queue is full while scheduling {name}. "
                    "Retry later or raise QDRANT_WRITE_QUEUE_SIZE."
                )
            self._waiting += 1

        await self._semaphore.acquire()
        async with self._lock:
            self._waiting -= 1
            self._running += 1

        try:
            return await operation()
        finally:
            async with self._lock:
                self._running -= 1
            self._semaphore.release()

    async def stats(self) -> WriteQueueStats:
        async with self._lock:
            return WriteQueueStats(
                max_concurrency=self.max_concurrency,
                max_queue_size=self.max_queue_size,
                running=self._running,
                waiting=self._waiting,
            )
