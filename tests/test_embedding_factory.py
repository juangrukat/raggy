from raggy_mcp.embeddings.factory import create_embedding_provider
from raggy_mcp.embeddings.fastembed import FastEmbedProvider
from raggy_mcp.embeddings.qwen3_rust import Qwen3RustProvider
from raggy_mcp.settings import EmbeddingProviderSettings


def test_qwen3_models_use_rust_sidecar_provider():
    provider = create_embedding_provider(
        EmbeddingProviderSettings(
            EMBEDDING_MODEL="Qwen/Qwen3-Embedding-8B",
            QWEN3_METRICS_PATH="/tmp/qwen3-metrics.jsonl",
            QWEN3_RESPONSE_LIMIT_BYTES=123456,
            QWEN3_IDLE_TIMEOUT_SECONDS=999,
            QWEN3_OUTPUT_DIMENSION=2048,
        )
    )

    assert isinstance(provider, Qwen3RustProvider)
    assert provider.get_vector_size() == 2048
    assert provider.output_dimension == 2048
    assert str(provider.metrics_path) == "/tmp/qwen3-metrics.jsonl"
    assert provider.response_limit_bytes == 123456
    assert provider.idle_timeout_seconds == 999


def test_qwen3_4b_uses_rust_sidecar_with_2560_dimensions():
    provider = create_embedding_provider(
        EmbeddingProviderSettings(EMBEDDING_MODEL="Qwen/Qwen3-Embedding-4B")
    )

    assert isinstance(provider, Qwen3RustProvider)
    assert provider.get_vector_size() == 2560


def test_qwen3_output_dimension_truncates_and_normalizes():
    provider = Qwen3RustProvider("Qwen/Qwen3-Embedding-4B", output_dimension=2)

    shaped = provider._shape_embeddings([[3.0, 4.0, 12.0]])

    assert len(shaped[0]) == 2
    assert shaped[0] == [0.6, 0.8]


def test_non_qwen_models_use_python_fastembed_provider():
    provider = create_embedding_provider(
        EmbeddingProviderSettings(EMBEDDING_MODEL="BAAI/bge-small-en-v1.5")
    )

    assert isinstance(provider, FastEmbedProvider)
