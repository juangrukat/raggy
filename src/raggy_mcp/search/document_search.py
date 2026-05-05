"""
Document-grouped search.

Instead of returning raw chunk-level hits (where one long PDF can dominate
the top results with many adjacent chunks), this fetches a wider chunk pool,
groups by `metadata.document_id`, scores each document by its best chunk,
and returns one summary entry per document with its top representative chunk(s).

Two-stage pipeline
------------------
Stage 1 — retrieval (fast):
  Dense + sparse (BM25/BM42) candidates are fetched in parallel via Qdrant
  prefetch + RRF fusion, yielding `prefetch_limit` unique chunks.

  Multi-query mode: if `additional_queries` is provided, each query retrieves
  its own candidate pool. The pools are merged and deduplicated by content
  hash. The highest first-stage score per chunk is preserved. Reranking always
  uses the PRIMARY query, so rare conceptual terms in the original question
  influence the final ranking even if they weren't in every subquery.

Stage 2 — reranking (optional, expensive):
  A cross-encoder or generative reranker (e.g. Qwen3-Reranker-4B) re-scores
  the merged candidate pool against the PRIMARY query. A diversity pass then
  reduces redundancy across pages/sections before the final grouping step.

Why multi-query matters
-----------------------
A single rewritten query often compresses out discriminative terms. For
"Does knowledge increase using Socratic circles?" an agent might send one
short retrieval query that drops "Three Columns of Learning", "Enlargement
of the Understanding", "maieutic", or "possession of knowledge". Those
passages exist in the collection but are never surfaced as candidates.

Multi-query fixes this at the retrieval stage:
  primary_query  = "Socratic circles increase student knowledge understanding"
  additional     = [
    "Adler Three Columns Enlargement of the Understanding maieutic teaching",
    "critical reading critical thinking vocabulary student ownership empowerment",
    "Socratic circles expand students learning idea theme subject",
  ]

Each subquery fetches independently. The pools merge. The reranker scores all
unique candidates against the PRIMARY query. The best evidence wins.
"""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any

from qdrant_client import models

from raggy_mcp.qdrant import QdrantConnector
from raggy_mcp.search.reranker import RerankCandidate, Reranker

# Default candidate pool sizes.
_DEFAULT_PREFETCH_DENSE = 80  # dense-only or hybrid without reranking
_DEFAULT_PREFETCH_RERANK = 100  # with reranking: wider pool feeds the reranker
_MAX_PREFETCH = 300  # hard ceiling to avoid memory/latency blowout

# Diversity: max chunks from the same inferred "section" before we defer extras.
# A section is the page number if available, else chunk_index // 5.
_DIVERSITY_MAX_PER_SECTION = 2


def _content_hash(content: str) -> str:
    """Short hash of chunk content used for deduplication across queries."""
    return hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()[:16]


def _section_key(meta: dict) -> str:
    """Derive a coarse section identifier for diversity de-duplication."""
    page = meta.get("page") or meta.get("page_number")
    if page is not None:
        return f"p{page}"
    idx = meta.get("chunk_index")
    if idx is not None:
        return f"s{int(idx) // 5}"
    # fallback: no section info → treat every chunk as its own section
    return _content_hash(meta.get("path", "") + str(meta.get("chunk_index", "")))


def _diversity_pass(
    scored: list[tuple[Any, float]],
    max_per_section: int = _DIVERSITY_MAX_PER_SECTION,
) -> list[tuple[Any, float]]:
    """
    Prefer coverage across distinct pages/sections when many high-scoring
    chunks come from the same region.

    Algorithm: greedy two-pass.
      Pass 1 — pick up to `max_per_section` chunks per section in score order.
      Pass 2 — fill remaining slots from deferred (same-section) chunks.
    """
    section_counts: dict[str, int] = {}
    selected: list[tuple[Any, float]] = []
    deferred: list[tuple[Any, float]] = []

    for entry, score in scored:
        meta = (entry.metadata or {}) if hasattr(entry, "metadata") else {}
        key = _section_key(meta)
        if section_counts.get(key, 0) < max_per_section:
            selected.append((entry, score))
            section_counts[key] = section_counts.get(key, 0) + 1
        else:
            deferred.append((entry, score))

    selected.extend(deferred)
    return selected


async def _fetch_candidates(
    connector: QdrantConnector,
    query: str,
    collection_name: str,
    prefetch_limit: int,
    query_filter: models.Filter | None,
    sparse_provider: Any,
    late_interaction_provider: Any,
    embedding_provider: Any,
    min_score: float | None,
) -> list[tuple[Any, float]]:
    """
    Single-query retrieval — shared by primary and each additional query.

    Fallback chain:
      late_interaction → late-interaction MaxSim (no fallback; mode is explicit)
      sparse_provider  → hybrid RRF (falls back to dense internally)
                       → if still empty: dense-only as outer safety net
      else             → dense-only
    """
    if late_interaction_provider is not None:
        return await connector.search_late_interaction(
            query=query,
            collection_name=collection_name,
            late_interaction_provider=late_interaction_provider,
            limit=prefetch_limit,
            query_filter=query_filter,
        )
    elif sparse_provider is not None:
        results = await connector.search_hybrid_rrf(
            query=query,
            collection_name=collection_name,
            sparse_provider=sparse_provider,
            limit=prefetch_limit,
            query_filter=query_filter,
            embedding_provider=embedding_provider,
        )
        # Outer safety net: search_hybrid_rrf already falls back to dense internally,
        # but if it returned empty for any reason, try pure dense before giving up.
        if not results:
            import logging as _logging

            _logging.getLogger(__name__).warning(
                "Hybrid search returned 0 candidates for %r (query=%r). "
                "Attempting pure dense fallback.",
                collection_name,
                query[:80],
            )
            results = await connector.hybrid_search(
                query=query,
                collection_name=collection_name,
                limit=prefetch_limit,
                query_filter=query_filter,
                min_score=min_score,
                embedding_provider=embedding_provider,
            )
        return results
    else:
        return await connector.hybrid_search(
            query=query,
            collection_name=collection_name,
            limit=prefetch_limit,
            query_filter=query_filter,
            min_score=min_score,
            embedding_provider=embedding_provider,
        )


def _merge_candidates(
    result_sets: list[list[tuple[Any, float]]],
) -> list[tuple[Any, float]]:
    """
    Merge multiple retrieval result sets, deduplicating by content hash.
    When the same chunk appears from multiple queries, keep the highest
    first-stage score. Order is preserved (best score first).
    """
    best: dict[str, tuple[Any, float]] = {}
    for results in result_sets:
        for entry, score in results:
            key = _content_hash(entry.content)
            if key not in best or score > best[key][1]:
                best[key] = (entry, score)
    # Sort by best score descending
    return sorted(best.values(), key=lambda kv: kv[1], reverse=True)


async def search_documents_grouped(
    connector: QdrantConnector,
    query: str,
    collection_name: str,
    *,
    limit: int = 10,
    chunks_per_document: int = 4,
    query_filter: models.Filter | None = None,
    min_score: float | None = None,
    sparse_provider: Any = None,
    late_interaction_provider: Any = None,
    reranker: Reranker | None = None,
    embedding_provider: Any = None,
    prefetch_limit: int | None = None,
    rerank_top_k: int | None = None,
    diversity: bool = True,
    additional_queries: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Search and return distinct documents, ordered by best-chunk score.

    Parameters
    ----------
    query : str
        The primary search query. Used for both retrieval and reranking.
        Reranking ALWAYS scores candidates against this query, regardless of
        what additional_queries were used for retrieval.
    additional_queries : list[str] | None
        Extra retrieval queries run in parallel with the primary query.
        Their candidate pools are merged and deduplicated before reranking.
        Use these to preserve rare conceptual terms that the primary query
        alone might not recall (e.g. "Adler Three Columns Enlargement").
        Example:
          query = "Socratic circles increase student knowledge"
          additional_queries = [
            "Adler Three Columns Enlargement of the Understanding maieutic",
            "critical reading critical thinking vocabulary ownership empowerment",
          ]
    limit : int
        Maximum distinct documents to return.
    chunks_per_document : int
        Best chunks to include per document.
    prefetch_limit : int | None
        How many raw chunks to fetch per query before grouping/reranking.
        None → auto (100 when reranker is active, 80 otherwise).
    rerank_top_k : int | None
        How many chunks to pass to the reranker after merging. None → all.
    diversity : bool
        Apply a section-diversity pass after reranking. Default True.

    Returns
    -------
    List of documents:
      {
        "document_id": str,
        "filename": str | None,
        "path": str | None,
        "score": float,
        "chunks": [{"content", "score", "chunk_index", "metadata"}]
      }
    """
    use_reranker = reranker is not None

    if prefetch_limit is None:
        if use_reranker:
            prefetch_limit = _DEFAULT_PREFETCH_RERANK
        else:
            prefetch_limit = max(limit * 8, _DEFAULT_PREFETCH_DENSE)

    prefetch_limit = min(prefetch_limit, _MAX_PREFETCH)

    # Fetch candidates for the primary query
    primary_results = await _fetch_candidates(
        connector,
        query,
        collection_name,
        prefetch_limit,
        query_filter,
        sparse_provider,
        late_interaction_provider,
        embedding_provider,
        min_score,
    )

    # Fetch additional queries concurrently if provided
    if additional_queries:
        extra_tasks = [
            _fetch_candidates(
                connector,
                q,
                collection_name,
                prefetch_limit,
                query_filter,
                sparse_provider,
                late_interaction_provider,
                embedding_provider,
                min_score,
            )
            for q in additional_queries
        ]
        extra_results = await asyncio.gather(*extra_tasks)
        raw = _merge_candidates([primary_results, *extra_results])
    else:
        raw = primary_results

    # Rerank + diversity
    raw = await _maybe_rerank(
        raw, use_reranker, reranker, query, rerank_top_k, diversity
    )

    # Group by document
    return _group_by_document(raw, limit, chunks_per_document)


async def _maybe_rerank(
    raw: list[tuple[Any, float]],
    use_reranker: bool,
    reranker: Reranker | None,
    query: str,
    rerank_top_k: int | None = None,
    diversity: bool = True,
) -> list[tuple[Any, float]]:
    """Optional reranking pass — always against the PRIMARY query.

    Parameters
    ----------
    raw : list[tuple[Any, float]]
        Prefetch candidates as (entry, score) pairs.
    use_reranker : bool
        Whether a reranker is available.
    reranker : Reranker | None
        The reranker instance.
    query : str
        Primary query — reranker always scores against this.
    rerank_top_k : int | None
        Max candidates to pass to the reranker. None → all.
    diversity : bool
        Apply section-diversity pass after reranking.
    """
    if not use_reranker or not raw:
        return raw

    pool = raw if rerank_top_k is None else raw[:rerank_top_k]
    candidates = [
        RerankCandidate(
            content=entry.content,
            metadata=entry.metadata,
            first_stage_score=score,
            payload=entry,
        )
        for entry, score in pool
    ]
    assert reranker is not None
    reranked = await reranker.rerank(query, candidates)  # type: ignore[union-attr]
    raw = [(c.payload, score) for c, score in reranked]

    if diversity:
        raw = _diversity_pass(raw)
    return raw


def _group_by_document(
    raw: list[tuple[Any, float]],
    limit: int,
    chunks_per_document: int,
) -> list[dict[str, Any]]:
    """Group raw chunk results by document_id and return top documents.

    Parameters
    ----------
    raw : list[tuple[Any, float]]
        Scored (entry, score) pairs after retrieval and optional reranking.
    limit : int
        Maximum distinct documents to return.
    chunks_per_document : int
        Best chunks to include per document.
    """
    groups: dict[str, dict[str, Any]] = {}
    for entry, score in raw:
        meta = entry.metadata or {}
        doc_id = meta.get("document_id") or meta.get("path") or f"_chunk_{id(entry)}"
        bucket = groups.setdefault(
            doc_id,
            {
                "document_id": doc_id,
                "filename": meta.get("filename"),
                "path": meta.get("path"),
                "content_type": meta.get("content_type"),
                "tags": meta.get("tags"),
                "modified_at": meta.get("modified_at"),
                "score": score,
                "chunks": [],
            },
        )
        bucket["chunks"].append(
            {
                "content": entry.content,
                "score": score,
                "chunk_index": meta.get("chunk_index"),
                "metadata": meta,
            }
        )
        if score > bucket["score"]:
            bucket["score"] = score

    for bucket in groups.values():
        bucket["chunks"].sort(key=lambda c: c["score"], reverse=True)
        bucket["chunks"] = bucket["chunks"][:chunks_per_document]

    ordered = sorted(groups.values(), key=lambda g: g["score"], reverse=True)
    return ordered[:limit]
