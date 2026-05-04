"""Tests for reranker implementations."""

import pytest

from raggy_mcp.search.reranker import (
    NoOpReranker,
    FastEmbedReranker,
    QwenReranker,
    RerankCandidate,
    build_default_reranker,
)


def _candidates(texts: list[str], scores: list[float] | None = None) -> list[RerankCandidate]:
    if scores is None:
        scores = [float(i) for i in range(len(texts))]
    return [
        RerankCandidate(content=t, metadata=None, first_stage_score=s)
        for t, s in zip(texts, scores)
    ]


@pytest.mark.asyncio
async def test_noop_reranker_preserves_order():
    r = NoOpReranker()
    candidates = _candidates(["a", "b", "c"], scores=[0.9, 0.5, 0.1])
    results = await r.rerank("query", candidates)
    assert [rc.content for rc, _ in results] == ["a", "b", "c"]
    assert [s for _, s in results] == [0.9, 0.5, 0.1]


@pytest.mark.asyncio
async def test_noop_reranker_top_k():
    r = NoOpReranker()
    candidates = _candidates(["a", "b", "c"], scores=[0.9, 0.5, 0.1])
    results = await r.rerank("query", candidates, top_k=2)
    assert len(results) == 2
    assert results[0][0].content == "a"


@pytest.mark.asyncio
async def test_noop_reranker_empty():
    r = NoOpReranker()
    results = await r.rerank("query", [])
    assert results == []


def test_build_default_reranker_noop():
    r = build_default_reranker(None)
    assert isinstance(r, NoOpReranker)
    r2 = build_default_reranker("noop")
    assert isinstance(r2, NoOpReranker)


def test_build_default_reranker_fastembed():
    r = build_default_reranker("Xenova/ms-marco-MiniLM-L-6-v2")
    assert isinstance(r, FastEmbedReranker)
    assert r.model_name == "Xenova/ms-marco-MiniLM-L-6-v2"


def test_build_default_reranker_qwen():
    r = build_default_reranker("Qwen/Qwen3-Reranker-4B")
    assert isinstance(r, QwenReranker)
    assert r.model_name == "Qwen/Qwen3-Reranker-4B"

    r2 = build_default_reranker("Qwen/Qwen3-Reranker-0.6B")
    assert isinstance(r2, QwenReranker)


def test_qwen_reranker_format_pair():
    """Verify the prompt template is assembled correctly before any model load."""
    r = QwenReranker(instruction="Retrieve relevant passages")
    formatted = r._format_pair("What is gravity?", "Gravity is a force.")
    assert "<Instruct>: Retrieve relevant passages" in formatted
    assert "<Query>: What is gravity?" in formatted
    assert "<Document>: Gravity is a force." in formatted


@pytest.mark.asyncio
@pytest.mark.optional
async def test_qwen_reranker_raises_without_torch():
    """QwenReranker raises RuntimeError (not NotImplementedError) if torch is missing."""
    import importlib
    import sys

    torch_present = importlib.util.find_spec("torch") is not None
    transformers_present = importlib.util.find_spec("transformers") is not None

    if torch_present and transformers_present:
        pytest.skip("torch+transformers installed — load test not applicable")

    r = QwenReranker(model_name="Qwen/Qwen3-Reranker-0.6B")
    with pytest.raises(RuntimeError, match="requires torch"):
        await r.rerank("test", _candidates(["doc"]))
