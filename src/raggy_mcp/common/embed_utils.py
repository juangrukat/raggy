"""Shared async embedding helpers — deduplicates run_in_executor + tolist() pattern."""

import asyncio
from typing import Any, Callable


async def embed_in_executor(fn: Callable, input_: list[str]) -> list[list[float]]:
    """Run a synchronous embedding function in the default executor and convert
    numpy arrays to Python lists.

    Parameters
    ----------
    fn : Callable
        A synchronous embedding function that accepts a list[str] and returns
        an iterable of array-like objects (numpy ndarray, torch tensor, etc.)
        or raw lists.
    input_ : list[str]
        The batch of texts to embed.

    Returns
    -------
    list[list[float]]
        Embeddings as nested Python lists.
    """
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, fn, input_)
    return [_to_list(v) for v in result]


async def embed_query_in_executor(fn: Callable, query: str) -> list[float]:
    """Embed a single query string and return the first vector."""
    result = await embed_in_executor(fn, [query])
    return result[0]


def _to_list(v: Any) -> list[float]:
    """Convert a numpy array / torch tensor to a Python list."""
    if hasattr(v, "tolist"):
        return v.tolist()
    return list(v)
