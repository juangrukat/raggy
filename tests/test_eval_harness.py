"""Tests for the eval harness — scoring math and config dispatch, not full Qdrant."""

from __future__ import annotations

from tools.eval_harness import (
    EvalConfig,
    EvalSample,
    _precision_at_k,
    _recall_at_k,
    _mrr,
    _normalize_chunk_size,
    default_config_grid,
    report_markdown,
)


def test_recall_at_k_standard() -> None:
    retrieved = ["a", "b", "c", "d", "e"]
    relevant = {"a", "b", "f"}
    assert _recall_at_k(retrieved, relevant, 1) == 1.0 / 3
    assert _recall_at_k(retrieved, relevant, 2) == 2.0 / 3
    assert _recall_at_k(retrieved, relevant, 3) == 2.0 / 3
    assert _recall_at_k(retrieved, relevant, 10) == 2.0 / 3


def test_recall_at_k_empty_relevant() -> None:
    assert _recall_at_k(["a", "b"], set(), 5) == 0.0


def test_recall_at_k_no_match() -> None:
    assert _recall_at_k(["a", "b"], {"c"}, 2) == 0.0


def test_recall_at_k_fewer_than_k() -> None:
    retrieved = ["a"]
    relevant = {"a"}
    assert _recall_at_k(retrieved, relevant, 5) == 1.0


def test_precision_at_k_standard() -> None:
    retrieved = ["a", "b", "c", "d", "e"]
    relevant = {"a", "b", "f"}
    # top 5: {a,b,c,d,e} ∩ {a,b,f} = {a,b} → 2/5
    assert _precision_at_k(retrieved, relevant, 5) == 2.0 / 5
    # top 2: {a,b} ∩ {a,b,f} = {a,b} → 2/2
    assert _precision_at_k(retrieved, relevant, 2) == 1.0
    # top 1: {a} ∩ {a,b,f} = {a} → 1/1
    assert _precision_at_k(retrieved, relevant, 1) == 1.0


def test_precision_at_k_zero() -> None:
    assert _precision_at_k(["a", "b"], {"c"}, 0) == 0.0


def test_mrr_standard() -> None:
    assert _mrr(["x", "a", "y"], {"a"}) == 1.0 / 2
    assert _mrr(["a", "b"], {"a"}) == 1.0
    assert _mrr(["x", "y", "z"], {"a"}) == 0.0


def test_mrr_multiple_relevant() -> None:
    assert _mrr(["x", "a", "b"], {"a", "b"}) == 1.0 / 2  # first at rank 2


def test_normalize_chunk_size_clamps() -> None:
    cs, ov = _normalize_chunk_size(20000, 100)
    assert cs == 8192
    cs, ov = _normalize_chunk_size(10, 5)
    assert cs == 64

    cs, ov = _normalize_chunk_size(1024, 2000)
    assert ov < cs  # overlap clamped


def test_default_config_grid_returns_configs() -> None:
    configs = default_config_grid()
    assert len(configs) > 0
    for c in configs:
        assert isinstance(c, EvalConfig)
        assert c.label  # non-empty label
    # Should have unique labels
    labels = [c.label for c in configs]
    assert len(labels) == len(set(labels))


def test_eval_config_label_auto() -> None:
    c = EvalConfig(mode="dense", embedding_model="org/model", chunk_size=512)
    # rsplit("/", 1)[-1] strips org prefix
    assert "model" in c.label
    assert "dense" in c.label
    assert "512" in c.label


def test_eval_config_label_with_reranker() -> None:
    c = EvalConfig(
        mode="rerank",
        embedding_model="BAAI/bge-base-en-v1.5",
        chunk_size=1024,
        reranker_model="Xenova/ms-marco-MiniLM-L-6-v2",
    )
    assert "rerank" in c.label
    assert "ms-marco" in c.label


def test_eval_sample_from_dict() -> None:
    d = {"query": "test query", "relevant_doc_ids": ["doc1", "doc2"]}
    s = EvalSample.from_dict(d)
    assert s.query == "test query"
    assert s.relevant_doc_ids == ["doc1", "doc2"]


def test_report_markdown_format() -> None:
    from tools.eval_harness import EvalResult

    c1 = EvalConfig(label="dense/all-MiniLM/1024")
    c2 = EvalConfig(label="hybrid/bge/512", chunk_size=512)
    results = [
        EvalResult(config=c1, recall_at_5=0.65, mrr=0.72, avg_latency_ms=45.0, num_queries=10),
        EvalResult(config=c2, recall_at_5=0.70, mrr=0.78, avg_latency_ms=80.0, num_queries=10),
    ]
    md = report_markdown(results)
    assert "Recall@5" in md
    assert "0.650" in md
    assert "0.700" in md
    assert "45ms" in md
    assert "80ms" in md
    assert "dense/all-MiniLM/1024" in md
    assert "hybrid/bge/512" in md