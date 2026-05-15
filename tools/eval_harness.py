"""
Repeatable eval harness for RAG quality measurement.

Sweeps over chunking parameters, embedding models, retrieval modes, and
reranking strategies — then reports recall@k, precision@k, MRR, and latency
so decisions are empirically grounded instead of guessed.

Design
------
- Connects directly to existing APIs (QdrantConnector, search_documents_grouped,
  create_provider_for_model, build_reranker, extract_text, build_chunks).
- Does NOT go through the MCP server layer — that is plumbing, not retrieval.
- Operates on a temporary Qdrant collection per config trial.
- Produces a Markdown report and a JSON export for further analysis.

Usage
-----
    python -m tools.eval_harness \\
        --corpus docs/ \\
        --samples eval_samples.json \\
        --qdrant-url http://localhost:6333 \\
        --report eval_report.md
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class EvalSample:
    """A single ground-truth query with its expected document IDs."""

    query: str
    relevant_doc_ids: list[str]

    @classmethod
    def from_dict(cls, d: dict) -> EvalSample:
        return cls(query=d["query"], relevant_doc_ids=d["relevant_doc_ids"])


@dataclass
class EvalConfig:
    """Knobs that affect retrieval quality."""

    chunk_size: int = 1024
    chunk_overlap: int = 128
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    mode: str = "dense"  # dense | hybrid | rerank | late_interaction
    reranker_model: str | None = None
    prefetch_limit: int | None = None
    rerank_top_k: int | None = None
    label: str = ""

    def __post_init__(self):
        if not self.label:
            parts = [
                self.mode,
                self.embedding_model.rsplit("/", 1)[-1],
                str(self.chunk_size),
            ]
            if self.reranker_model:
                parts.insert(1, self.reranker_model.rsplit("/", 1)[-1])
            self.label = "/".join(parts)


@dataclass
class EvalResult:
    """Scores for one (config × sample) combination, aggregated per config."""

    config: EvalConfig
    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    precision_at_5: float = 0.0
    mrr: float = 0.0
    avg_latency_ms: float = 0.0
    num_queries: int = 0


# ---------------------------------------------------------------------------
# In-memory corpus builder (no Qdrant dependency for unit tests)
# ---------------------------------------------------------------------------

_MEMORY_COLLECTIONS: dict[str, list[dict]] = {}


class InMemoryCorpus:
    """Minimal in-memory store so the harness works without a running Qdrant."""

    def __init__(self, collection_name: str = "eval"):
        self.collection_name = collection_name
        self._docs: dict[str, list[dict]] = {}  # document_id → chunks

    def add_chunks(self, document_id: str, chunks: list[dict]) -> None:
        self._docs.setdefault(document_id, []).extend(chunks)

    def search(self, query: str, limit: int = 10) -> list[dict]:
        """Dummy search: returns all docs sorted by a toy score (for testing)."""
        results = []
        for doc_id, chunks in self._docs.items():
            results.append(
                {
                    "document_id": doc_id,
                    "chunks": chunks[:4],
                    "score": 1.0,
                    "path": chunks[0].get("metadata", {}).get("path", ""),
                    "filename": chunks[0].get("metadata", {}).get("filename", ""),
                }
            )
        return results[:limit]


# ---------------------------------------------------------------------------
# Core eval logic
# ---------------------------------------------------------------------------


def _precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Precision@k = |retrieved[:k] ∩ relevant| / k"""
    if k == 0 or not retrieved:
        return 0.0
    top_k = retrieved[:k]
    return len(set(top_k) & relevant) / k


def _recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Recall@k = |retrieved[:k] ∩ relevant| / |relevant|"""
    if not relevant or not retrieved:
        return 0.0
    top_k = retrieved[:k]
    return len(set(top_k) & relevant) / len(relevant)


def _mrr(retrieved: list[str], relevant: set[str]) -> float:
    """Mean reciprocal rank of the first relevant result."""
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def _normalize_chunk_size(size: int, overlap: int) -> tuple[int, int]:
    """Ensure overlap < size and both are sane."""
    size = max(64, min(size, 8192))
    overlap = max(0, min(overlap, size - 1))
    return size, overlap


def _find_corpus_files(corpus_dir: str) -> list[Path]:
    """Walk *corpus_dir* and return all files with supported extensions."""
    SUPPORTED = {
        ".txt",
        ".md",
        ".json",
        ".jsonl",
        ".csv",
        ".tsv",
        ".pdf",
        ".docx",
        ".py",
        ".js",
        ".ts",
        ".rs",
        ".java",
        ".c",
        ".cpp",
        ".h",
        ".html",
        ".htm",
        ".yaml",
        ".yml",
        ".toml",
        ".xml",
        ".rst",
        ".rtf",
        ".log",
        ".sh",
        ".bash",
        ".sql",
    }
    base = Path(corpus_dir)
    if not base.is_dir():
        raise NotADirectoryError(f"Corpus path is not a directory: {corpus_dir}")
    return sorted(p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED)


async def _ingest_corpus(
    corpus_dir: str,
    config: EvalConfig,
    embedding_manager: Any,
    qdrant_connector: Any,
    collection_name: str,
) -> None:
    """
    Ingest a corpus directory using the chunking parameters from *config*.
    Mocks are used when real Qdrant is unavailable.
    """
    from raggy_mcp.ingest.document_id import compute_document_id
    from raggy_mcp.ingest.extractor import SUPPORTED_EXTENSIONS, build_chunks, extract_text
    from raggy_mcp.ingest.macos_metadata import get_macos_metadata_async

    # Apply config chunking parameters
    import os

    os.environ["QDRANT_INGEST_CHUNK_SIZE"] = str(config.chunk_size)
    os.environ["QDRANT_INGEST_CHUNK_OVERLAP"] = str(config.chunk_overlap)

    files = _find_corpus_files(corpus_dir)
    if not files:
        logger.warning("No supported files found in %s", corpus_dir)
        return

    for path in files:
        try:
            meta = await get_macos_metadata_async(str(path))
            doc = extract_text(str(path))
            if not doc.text:
                continue
            meta["document_id"] = compute_document_id(doc.text, path=str(path))
            meta["has_text"] = bool(doc.text)
            meta["char_count"] = doc.char_count

            chunks = build_chunks(doc, meta)
            if not chunks:
                continue

            if qdrant_connector is not None:
                from raggy_mcp.qdrant import BatchEntry

                batch_entries = [
                    BatchEntry(content=c.text, metadata=c.metadata) for c in chunks
                ]
                await qdrant_connector.batch_store(batch_entries, collection_name)
            else:
                # In-memory fallback for testing
                doc_id = meta["document_id"]
                _MEMORY_COLLECTIONS.setdefault(collection_name, [])
                for c in chunks:
                    _MEMORY_COLLECTIONS[collection_name].append(
                        {
                            "document_id": doc_id,
                            "content": c.text,
                            "metadata": c.metadata,
                        }
                    )
        except Exception as e:
            logger.warning("Skipping %s: %s", path, e)


async def _run_single_config(
    samples: list[EvalSample],
    corpus_dir: str,
    config: EvalConfig,
    embedding_manager: Any,
    qdrant_connector: Any,
    collection_name: str,
) -> EvalResult:
    """
    Ingest corpus under *config* settings, run all queries, return aggregated scores.
    """
    from raggy_mcp.search.reranker import build_default_reranker
    from raggy_mcp.search.retrieval_mode import RetrievalMode

    # 1. Ingest (reloads env vars for chunk_size/overlap)
    await _ingest_corpus(corpus_dir, config, embedding_manager, qdrant_connector, collection_name)

    # 2. Prepare providers
    embedding_provider = embedding_manager.create_provider_for_model(config.embedding_model)

    sparse_provider = None
    late_interaction_provider = None
    reranker = None

    rmode = RetrievalMode.parse(config.mode)

    if rmode in (RetrievalMode.HYBRID, RetrievalMode.RERANK):
        from raggy_mcp.embeddings.sparse import SparseEmbeddingProvider

        sparse_provider = SparseEmbeddingProvider("Qdrant/bm25")

    if rmode == RetrievalMode.LATE_INTERACTION:
        from raggy_mcp.embeddings.late_interaction import LateInteractionEmbeddingProvider

        late_interaction_provider = LateInteractionEmbeddingProvider("colbert-ir/colbertv2.0")

    if rmode == RetrievalMode.RERANK and config.reranker_model:
        reranker = build_default_reranker(config.reranker_model)

    # 3. Run queries
    from raggy_mcp.search.document_search import search_documents_grouped

    total_latency = 0.0
    all_recall_1: list[float] = []
    all_recall_5: list[float] = []
    all_recall_10: list[float] = []
    all_precision_5: list[float] = []
    all_mrr: list[float] = []

    for sample in samples:
        t0 = time.perf_counter()
        try:
            docs = await search_documents_grouped(
                qdrant_connector,
                query=sample.query,
                collection_name=collection_name,
                limit=10,
                chunks_per_document=1,
                embedding_provider=embedding_provider,
                sparse_provider=sparse_provider,
                late_interaction_provider=late_interaction_provider,
                reranker=reranker,
                prefetch_limit=config.prefetch_limit,
                rerank_top_k=config.rerank_top_k,
            )
        except Exception as e:
            logger.warning("Query %r failed: %s", sample.query[:60], e)
            docs = []
        elapsed = time.perf_counter() - t0
        total_latency += elapsed

        retrieved_ids = [d["document_id"] for d in docs]
        relevant_set = set(sample.relevant_doc_ids)

        all_recall_1.append(_recall_at_k(retrieved_ids, relevant_set, 1))
        all_recall_5.append(_recall_at_k(retrieved_ids, relevant_set, 5))
        all_recall_10.append(_recall_at_k(retrieved_ids, relevant_set, 10))
        all_precision_5.append(_precision_at_k(retrieved_ids, relevant_set, 5))
        all_mrr.append(_mrr(retrieved_ids, relevant_set))

    n = len(samples)
    return EvalResult(
        config=config,
        recall_at_1=sum(all_recall_1) / n if n else 0.0,
        recall_at_5=sum(all_recall_5) / n if n else 0.0,
        recall_at_10=sum(all_recall_10) / n if n else 0.0,
        precision_at_5=sum(all_precision_5) / n if n else 0.0,
        mrr=sum(all_mrr) / n if n else 0.0,
        avg_latency_ms=(total_latency / n * 1000) if n else 0.0,
        num_queries=n,
    )


# ---------------------------------------------------------------------------
# Config grid
# ---------------------------------------------------------------------------


def default_config_grid() -> list[EvalConfig]:
    """Return a reasonable sweep over chunking, models, and modes."""
    configs: list[EvalConfig] = []

    chunk_sizes = [512, 1024]
    overlaps = [64, 128]
    models = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "BAAI/bge-base-en-v1.5",
    ]
    modes = ["dense", "hybrid", "rerank"]

    for cs in chunk_sizes:
        for ov in overlaps:
            for model in models:
                for mode in modes:
                    reranker = (
                        "Xenova/ms-marco-MiniLM-L-6-v2" if mode == "rerank" else None
                    )
                    configs.append(
                        EvalConfig(
                            chunk_size=cs,
                            chunk_overlap=ov,
                            embedding_model=model,
                            mode=mode,
                            reranker_model=reranker,
                        )
                    )

    # Deduplicate by label
    seen: set[str] = set()
    deduped: list[EvalConfig] = []
    for c in configs:
        if c.label not in seen:
            seen.add(c.label)
            deduped.append(c)
    return deduped


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


async def run_eval(
    corpus_dir: str,
    samples: list[EvalSample],
    configs: list[EvalConfig] | None = None,
    qdrant_url: str | None = None,
    qdrant_api_key: str | None = None,
    collection_prefix: str = "eval_",
) -> list[EvalResult]:
    """
    Run an eval sweep.

    Parameters
    ----------
    corpus_dir : str
        Path to a directory of documents to ingest.
    samples : list[EvalSample]
        Ground-truth Q&A pairs.
    configs : list[EvalConfig] | None
        Configs to sweep. Defaults to ``default_config_grid()``.
    qdrant_url : str | None
        Qdrant URL. If None, uses in-memory fallback (for testing).
    collection_prefix : str
        Prefix for temporary eval collections.

    Returns
    -------
    list[EvalResult]
        One result per config, ordered by label.
    """
    if configs is None:
        configs = default_config_grid()

    # Build Qdrant connector (or None for in-memory fallback)
    qdrant_connector = None
    if qdrant_url:
        from raggy_mcp.qdrant import QdrantConnector

        qdrant_connector = QdrantConnector(
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
            collection_name=None,
            embedding_provider=None,  # set per-request
        )

    # Build embedding manager
    from raggy_mcp.embedding_manager import EnhancedEmbeddingModelManager
    from raggy_mcp.settings import EmbeddingProviderSettings

    embedding_manager = EnhancedEmbeddingModelManager(EmbeddingProviderSettings())

    results: list[EvalResult] = []
    for idx, config in enumerate(configs):
        coll_name = f"{collection_prefix}{idx}"
        logger.info(
            "Eval config %d/%d: %s  (collection=%s)",
            idx + 1,
            len(configs),
            config.label,
            coll_name,
        )

        # Ensure collection exists
        if qdrant_connector is not None:
            try:
                await qdrant_connector._client.recreate_collection(
                    collection_name=coll_name,
                    vectors_config={
                        "default": {
                            "size": 384,
                            "distance": "Cosine",
                        }
                    },
                )
            except Exception:
                logger.warning("Could not create eval collection, may already exist")
                try:
                    await qdrant_connector._client.delete_collection(coll_name)
                    await qdrant_connector._client.recreate_collection(
                        collection_name=coll_name,
                        vectors_config={
                            "default": {
                                "size": 384,
                                "distance": "Cosine",
                            }
                        },
                    )
                except Exception:
                    pass

        result = await _run_single_config(
            samples=samples,
            corpus_dir=corpus_dir,
            config=config,
            embedding_manager=embedding_manager,
            qdrant_connector=qdrant_connector,
            collection_name=coll_name,
        )
        results.append(result)

        # Cleanup collection
        if qdrant_connector is not None:
            try:
                await qdrant_connector._client.delete_collection(coll_name)
            except Exception:
                pass

    results.sort(key=lambda r: r.config.label)
    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def report_markdown(results: list[EvalResult]) -> str:
    """Render results as a Markdown table."""
    lines = [
        "## Eval Results\n",
        f"| Config | Recall@1 | Recall@5 | Recall@10 | Precision@5 | MRR   | Latency | Queries |",
        f"|--------|----------|----------|-----------|-------------|-------|---------|---------|",
    ]
    for r in results:
        lines.append(
            f"| {r.config.label} "
            f"| {r.recall_at_1:.3f} "
            f"| {r.recall_at_5:.3f} "
            f"| {r.recall_at_10:.3f} "
            f"| {r.precision_at_5:.3f} "
            f"| {r.mrr:.3f} "
            f"| {r.avg_latency_ms:.0f}ms "
            f"| {r.num_queries} |"
        )
    lines.append("")
    return "\n".join(lines)


def report_json(results: list[EvalResult]) -> str:
    """Render results as JSON."""
    return json.dumps(
        [
            {
                "config": {
                    "label": r.config.label,
                    "chunk_size": r.config.chunk_size,
                    "chunk_overlap": r.config.chunk_overlap,
                    "embedding_model": r.config.embedding_model,
                    "mode": r.config.mode,
                    "reranker_model": r.config.reranker_model,
                },
                "recall_at_1": r.recall_at_1,
                "recall_at_5": r.recall_at_5,
                "recall_at_10": r.recall_at_10,
                "precision_at_5": r.precision_at_5,
                "mrr": r.mrr,
                "avg_latency_ms": r.avg_latency_ms,
                "num_queries": r.num_queries,
            }
            for r in results
        ],
        indent=2,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG eval harness")
    parser.add_argument(
        "--corpus",
        required=True,
        help="Directory with ground-truth documents",
    )
    parser.add_argument(
        "--samples",
        required=True,
        help="JSON file with EvalSample list: [{query, relevant_doc_ids}]",
    )
    parser.add_argument(
        "--qdrant-url",
        default="http://localhost:6333",
        help="Qdrant HTTP URL (default: http://localhost:6333)",
    )
    parser.add_argument(
        "--configs",
        default=None,
        help="Optional JSON file with EvalConfig list to sweep (omit for defaults)",
    )
    parser.add_argument(
        "--report",
        default="eval_report.md",
        help="Output Markdown report path (default: eval_report.md)",
    )
    parser.add_argument(
        "--json",
        default=None,
        help="Output JSON results path (optional)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load samples
    with open(args.samples) as f:
        raw = json.load(f)
    samples = [EvalSample.from_dict(item) for item in raw]

    # Load configs
    configs = None
    if args.configs:
        with open(args.configs) as f:
            raw_configs = json.load(f)
        configs = [EvalConfig(**item) for item in raw_configs]

    results = asyncio.run(
        run_eval(
            corpus_dir=args.corpus,
            samples=samples,
            configs=configs,
            qdrant_url=args.qdrant_url,
        )
    )

    md = report_markdown(results)
    with open(args.report, "w") as f:
        f.write(md)
    print(f"Report written to {args.report}")
    print(md)

    if args.json:
        js = report_json(results)
        with open(args.json, "w") as f:
            f.write(js)
        print(f"JSON written to {args.json}")


if __name__ == "__main__":
    main()
