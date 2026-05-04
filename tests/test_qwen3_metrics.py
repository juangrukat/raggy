import json

from raggy_mcp.embeddings.qwen3_rust import Qwen3RustProvider


def test_qwen3_metrics_are_written_as_jsonl(tmp_path):
    metrics_path = tmp_path / "qwen3-embeddings.jsonl"
    provider = Qwen3RustProvider(
        "Qwen/Qwen3-Embedding-8B",
        device="auto",
        max_length=1024,
        dtype="auto",
        metrics_path=str(metrics_path),
    )
    provider._ready = {
        "backend": "metal",
        "dtype": "f16",
        "vector_size": 4096,
    }

    provider._record_metrics(
        operation="embed_query",
        text_count=1,
        char_count=22,
        cold_start_ms=1234,
        request_ms=56,
        total_ms=1290,
        success=True,
        error=None,
        embedding_count=1,
    )

    record = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert record["operation"] == "embed_query"
    assert record["model"] == "Qwen/Qwen3-Embedding-8B"
    assert record["backend"] == "metal"
    assert record["dtype"] == "f16"
    assert record["vector_size"] == 4096
    assert record["text_count"] == 1
    assert record["char_count"] == 22
    assert record["embedding_count"] == 1
    assert record["cold_start_ms"] == 1234
    assert record["request_ms"] == 56
    assert record["total_ms"] == 1290
    assert record["success"] is True
    assert "timestamp" in record


def test_qwen3_document_inputs_are_prefixed_as_passages():
    assert Qwen3RustProvider._document_input("hello") == "passage: hello"
    assert Qwen3RustProvider._document_input("passage: hello") == "passage: hello"
