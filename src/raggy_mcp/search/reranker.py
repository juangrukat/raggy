"""
Reranker abstraction.

A reranker takes a query and a candidate list (from first-stage retrieval) and
returns the candidates re-ordered by a more expensive but more accurate scorer —
typically a cross-encoder.

Why an abstraction now: the search pipeline already has dense + hybrid first-stage
retrieval, and reranking is the highest-leverage next quality improvement. Having
the interface in place lets us plug in different rerankers (FastEmbed cross-encoders,
Qwen3-Reranker, ColBERT late interaction) without touching call sites.

The default `NoOpReranker` is the safe fallback when reranking is disabled. The
`FastEmbedReranker` lazy-loads a cross-encoder via fastembed when available.

`QwenReranker` implements Qwen3-Reranker-{0.6B,4B,8B} via the transformers
library (AutoModelForCausalLM). These models score relevance by predicting "yes"
or "no" at the final token position. Install optional deps first:

    pip install torch transformers accelerate
    # or: uv pip install 'raggy-mcp[reranking]'
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RerankCandidate:
    """One candidate passed into the reranker; preserves original payload for re-emission."""

    content: str
    metadata: dict | None
    first_stage_score: float
    payload: Any | None = None  # opaque slot for callers to keep their own object


class Reranker(ABC):
    """Reranker interface — implementations should be safe to call repeatedly."""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        *,
        top_k: int | None = None,
    ) -> list[tuple[RerankCandidate, float]]:
        """
        Rerank candidates and return them ordered by descending relevance score.
        The returned scores are reranker scores, not first-stage scores.
        """

    @property
    @abstractmethod
    def model_name(self) -> str: ...


class NoOpReranker(Reranker):
    """Returns candidates in their first-stage order with first-stage scores. Safe default."""

    @property
    def model_name(self) -> str:
        return "noop"

    async def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        *,
        top_k: int | None = None,
    ) -> list[tuple[RerankCandidate, float]]:
        ordered = sorted(candidates, key=lambda c: c.first_stage_score, reverse=True)
        if top_k:
            ordered = ordered[:top_k]
        return [(c, c.first_stage_score) for c in ordered]


class FastEmbedReranker(Reranker):
    """
    Cross-encoder reranker via fastembed's TextCrossEncoder, when available.

    Recommended models (in order of size/speed tradeoff):
      - Xenova/ms-marco-MiniLM-L-6-v2  (light, fast)
      - jinaai/jina-reranker-v1-tiny-en
      - BAAI/bge-reranker-base
      - BAAI/bge-reranker-large

    The Qwen3-Reranker family is not yet exposed via fastembed at the time of writing;
    use `QwenReranker` (transformers-based) for those.

    On Apple Silicon, the ONNX session uses the CoreMLExecutionProvider when
    available, giving ~2-4x speedup over CPU-only inference.
    """

    def __init__(self, model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2"):
        self._model_name = model_name
        self._encoder = None  # lazy

    @staticmethod
    def _resolve_providers() -> list[str]:
        """Pick the best available ONNX execution providers for this platform.

        NOTE: CoreMLExecutionProvider is available on Apple Silicon but incurs
        a 30-40s first-time model compilation penalty and does not reliably
        outperform CPU for BERT-style cross-encoders.
        Set QDRANT_RERANKER_PROVIDERS=CoreMLExecutionProvider,CPUExecutionProvider
        to opt in.
        """
        import os

        env_providers = os.getenv("QDRANT_RERANKER_PROVIDERS")
        if env_providers:
            return [p.strip() for p in env_providers.split(",")]

        return ["CPUExecutionProvider"]

    @property
    def model_name(self) -> str:
        return self._model_name

    def _load(self):
        if self._encoder is not None:
            return
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
        except ImportError as e:
            raise RuntimeError(
                "FastEmbedReranker requires fastembed>=0.4 with TextCrossEncoder. "
                f"Import error: {e}"
            )
        providers = self._resolve_providers()
        import logging

        _log = logging.getLogger(__name__)
        _log.info("FastEmbedReranker using providers: %s", providers)
        self._encoder = TextCrossEncoder(self._model_name, providers=providers)

    async def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        *,
        top_k: int | None = None,
    ) -> list[tuple[RerankCandidate, float]]:
        if not candidates:
            return []
        self._load()
        loop = asyncio.get_event_loop()
        texts = [c.content for c in candidates]
        scores = await loop.run_in_executor(
            None,
            lambda: list(self._encoder.rerank(query, texts)),  # type: ignore[union-attr,attr-defined]
        )
        scored = list(zip(candidates, scores))
        scored.sort(key=lambda kv: kv[1], reverse=True)
        if top_k:
            scored = scored[:top_k]
        return scored


class QwenReranker(Reranker):
    """
    Qwen3-Reranker-{0.6B,4B,8B} via transformers (AutoModelForCausalLM).

    These are generative rerankers: each (query, doc) pair is formatted with a
    prompt template and the relevance score is P("yes") from the final token logits.

    Requires: pip install torch transformers accelerate
    Recommended local use: model_name="Qwen/Qwen3-Reranker-4B", device auto-selects MPS.

    Custom instruction improves quality by 1-5% when task is known:
        QwenReranker(instruction="Given a book search query, retrieve relevant passages")
    """

    _SYSTEM_PROMPT = (
        "<|im_start|>system\nJudge whether the Document meets the requirements "
        "based on the Query and the Instruct provided. Note that the answer can "
        'only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    )
    _RESPONSE_TEMPLATE = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    _DEFAULT_INSTRUCTION = (
        "Given a web search query, retrieve relevant passages that answer the query"
    )

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Reranker-4B",
        device: str | None = None,
        max_length: int = 8192,
        instruction: str | None = None,
        batch_size: int = 4,
    ):
        self._model_name = model_name
        self._device = device  # None → auto-detect at load time
        self._max_length = max_length
        self._instruction = instruction or self._DEFAULT_INSTRUCTION
        self._batch_size = batch_size
        self._model = None
        self._tokenizer = None
        self._token_true_id: int | None = None
        self._token_false_id: int | None = None
        self._prefix_tokens: list[int] = []
        self._suffix_tokens: list[int] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise RuntimeError(
                "QwenReranker requires torch and transformers. "
                "Install with: pip install torch transformers accelerate\n"
                f"Import error: {e}"
            ) from e

        device = self._device
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        logger.info("Loading %s on %s ...", self._model_name, device)
        dtype = torch.float32 if device == "cpu" else torch.float16

        tokenizer = AutoTokenizer.from_pretrained(self._model_name, padding_side="left")
        model = (
            AutoModelForCausalLM.from_pretrained(
                self._model_name,
                torch_dtype=dtype,
            )
            .to(device)
            .eval()
        )

        self._tokenizer = tokenizer
        self._model = model
        self._device = device
        self._token_true_id = tokenizer.convert_tokens_to_ids("yes")
        self._token_false_id = tokenizer.convert_tokens_to_ids("no")
        self._prefix_tokens = tokenizer.encode(
            self._SYSTEM_PROMPT, add_special_tokens=False
        )
        self._suffix_tokens = tokenizer.encode(
            self._RESPONSE_TEMPLATE, add_special_tokens=False
        )
        logger.info(
            "%s loaded (%s, dtype=%s, true_id=%s, false_id=%s)",
            self._model_name,
            device,
            dtype,
            self._token_true_id,
            self._token_false_id,
        )

    def _format_pair(self, query: str, doc: str) -> str:
        return f"<Instruct>: {self._instruction}\n<Query>: {query}\n<Document>: {doc}"

    def _score_batch_sync(self, query: str, texts: list[str]) -> list[float]:
        import torch

        assert self._tokenizer is not None and self._model is not None

        pairs = [self._format_pair(query, t) for t in texts]
        usable_len = (
            self._max_length - len(self._prefix_tokens) - len(self._suffix_tokens)
        )
        inputs = self._tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=usable_len,
        )
        for i, ids in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = self._prefix_tokens + ids + self._suffix_tokens
        inputs = self._tokenizer.pad(
            inputs,
            padding=True,
            return_tensors="pt",
            max_length=self._max_length,
        )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self._model(**inputs).logits[:, -1, :]

        true_vec = logits[:, self._token_true_id]
        false_vec = logits[:, self._token_false_id]
        stacked = torch.stack([false_vec, true_vec], dim=1)
        log_probs = torch.nn.functional.log_softmax(stacked, dim=1)
        return log_probs[:, 1].exp().tolist()

    async def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        *,
        top_k: int | None = None,
    ) -> list[tuple[RerankCandidate, float]]:
        if not candidates:
            return []
        self._load()

        loop = asyncio.get_event_loop()
        texts = [c.content for c in candidates]
        all_scores: list[float] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            scores = await loop.run_in_executor(
                None,
                lambda b=batch: self._score_batch_sync(query, b),  # type: ignore[misc]
            )
            all_scores.extend(scores)

        scored = list(zip(candidates, all_scores))
        scored.sort(key=lambda kv: kv[1], reverse=True)
        if top_k:
            scored = scored[:top_k]
        return scored


def build_default_reranker(
    model_name: str | None = None,
    instruction: str | None = None,
) -> Reranker:
    """
    Factory that returns the appropriate Reranker for the given model name.

    - None / "noop"           → NoOpReranker (first-stage order, no scoring)
    - "Qwen/Qwen3-Reranker-*" → QwenReranker (CausalLM, requires torch+transformers)
    - anything else           → FastEmbedReranker (ONNX cross-encoder, no extra deps)

    instruction is forwarded to QwenReranker only; FastEmbed rerankers ignore it.
    Custom instructions improve Qwen reranker quality by ~1-5% on focused tasks.
    """
    if not model_name or model_name.lower() == "noop":
        return NoOpReranker()
    if model_name.startswith("Qwen/"):
        return QwenReranker(model_name=model_name, instruction=instruction)
    return FastEmbedReranker(model_name=model_name)
