"""Retrieval mode enum, used by search tools to select first-stage retrieval and rerank stack."""

from enum import Enum


class RetrievalMode(str, Enum):
    """Retrieval pipeline selection.

    - DENSE: single-stage dense vector search with the active embedding model.
    - HYBRID: dense + sparse (BM25) retrieval fused server-side with RRF.
    - RERANK: HYBRID first-stage, then cross-encoder reranking on the top candidates.
    - LATE_INTERACTION: ColBERT-style multivector MaxSim retrieval.
    """

    DENSE = "dense"
    HYBRID = "hybrid"
    RERANK = "rerank"
    LATE_INTERACTION = "late_interaction"

    @classmethod
    def parse(cls, value: str | None) -> "RetrievalMode":
        if value is None:
            return cls.DENSE
        try:
            return cls(value)
        except ValueError as e:
            raise ValueError(
                f"Unknown retrieval mode '{value}'. "
                f"Valid: {[m.value for m in cls]}"
            ) from e
