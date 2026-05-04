import asyncio

import pytest

from raggy_mcp.mcp_runtime.write_queue import WriteQueue, WriteQueueFullError


@pytest.mark.asyncio
async def test_write_queue_serializes_work():
    queue = WriteQueue(max_concurrency=1, max_queue_size=2)
    order: list[str] = []

    async def work(name: str) -> str:
        order.append(f"start:{name}")
        await asyncio.sleep(0.01)
        order.append(f"end:{name}")
        return name

    results = await asyncio.gather(
        queue.run("one", lambda: work("one")),
        queue.run("two", lambda: work("two")),
    )

    assert results == ["one", "two"]
    assert order == ["start:one", "end:one", "start:two", "end:two"]


@pytest.mark.asyncio
async def test_write_queue_rejects_when_full():
    queue = WriteQueue(max_concurrency=1, max_queue_size=0)
    release = asyncio.Event()

    async def blocking_work() -> str:
        await release.wait()
        return "done"

    first = asyncio.create_task(queue.run("first", blocking_work))
    await asyncio.sleep(0)

    with pytest.raises(WriteQueueFullError):
        await queue.run("second", blocking_work)

    release.set()
    assert await first == "done"
