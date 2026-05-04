import pytest

FlagEmbedding = pytest.importorskip(
    "FlagEmbedding",
    reason="FlagEmbedding is optional; skipping BGE compatibility smoke test",
)


@pytest.mark.optional
def test_bge_base_flagembedding_vector_shape():
    model = FlagEmbedding.FlagModel(
        "BAAI/bge-base-en-v1.5",
        query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
        use_fp16=False,
    )

    vectors = model.encode(["hello world"])

    assert vectors.shape == (1, 768)
