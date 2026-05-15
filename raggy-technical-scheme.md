# Raggy — Technical Architecture Scheme

**Git Sync Status:** ✅ Local (`58ed622`) and origin (`58ed622`) are synchronized.
**Untracked (local only):** Profiling artifacts (`*.lprof`, `*.dat`, `*.pstats`) + `tools/` directory.

---

## 1. Retrieval Quality: The Actual RAG Knobs

The MCP server infrastructure (transports, profiles, plan registry, write queues, security middleware) is plumbing. It enables multi-agent access but **none of it guarantees better retrieval**. The following are the levers that actually determine RAG quality, ordered by impact:

### A. Chunking Strategy (highest leverage)

| Parameter | Current | What matters |
|-----------|---------|-------------|
| **chunk_size** | 1000 chars (QDRANT_INGEST_CHUNK_SIZE) | Narrative text needs richer context than factual snippets. Test 1000 and 1200 for book PDFs. |
| **chunk_overlap** | 150 chars (QDRANT_INGEST_CHUNK_OVERLAP) | Prevents boundary splits from destroying pronoun/event continuity. Test 150 and 180 for book PDFs. |
| **Structure awareness** | None — pure paragraph-boundary split | **Biggest gap.** Markdown heading hierarchy, code symbols, PDF page/section boundaries, DOCX heading styles are all discarded. Paragraph splits cross logical sections. |
| **Format-specific parsing** | Generic for all types | Markdown should split by ## headings (preserving heading path in metadata). Code should split by function/class boundaries. PDF should retain page numbers per chunk. DOCX should preserve heading level. |

### B. Document Parsing Quality

| Format | Extractor | Issues |
|--------|-----------|--------|
| PDF | PyMuPDF → pdfminer → pypdf fallback | Fast default for clean digital PDFs, with pdfminer kept for layout fallback. No table extraction. No figure caption extraction. No PDF outline/story-title synthesis yet. |
| DOCX | python-docx | Loses heading style information (Heading 1/2/3) — paragraphs are extracted flat. Tables rendered to plain text with [TABLE] markers. |
| Code (.py, .js, .tsx, .rs, etc.) | Plain text read | No AST or symbol awareness. A 500-line file becomes one text blob. |
| JSON/CSV | Rendered to line-delimited text | Row structure discarded. Chunks merge across rows. |

### C. Embedding Model Choice

Currently supports:
- **FastEmbed (ONNX):** sentence-transformers running locally via ONNX runtime. ~50+ available models, from 384D (all-MiniLM-L6-v2) to 4096D (Qwen3-Embedding-8B).
- **Qwen3 Rust sidecar:** Qwen3-Embedding-8B/4B/0.6B running in a separate Rust process via ONNX. Avoids Python GIL but adds IPC overhead.
- **Sparse:** splade/bm25 for hybrid retrieval.
- **Late-interaction:** ColBERTv2 (ColBERT MaxSim scoring) as an alternative retrieval mode.

**Unanswered questions:**
- Which embedding model gives best recall on *your* corpus? No eval harness exists to compare.
- Does higher dimensionality always help? Qwen3-8B (4096D) vs all-MiniLM (384D) — tradeoffs in latency, storage, and retrieval quality are unmeasured.
- When does hybrid (dense + sparse) outperform dense alone? Corpus-dependent, unmeasured.

### D. Retrieval Modes

| Mode | Mechanism | When to use |
|------|-----------|-------------|
| `dense` | Single vector search | General purpose, works for most corpora |
| `hybrid` | Dense + sparse → RRF merge | Best when keyword matches matter (code, technical docs, proper nouns) |
| `rerank` | Hybrid prefetch → cross-encoder/generative rerank | Higher quality at cost of latency. MiniLM is default; Qwen3-Reranker is available only through Python `transformers`, not the Rust sidecar. |
| `late_interaction` | ColBERT MaxSim | Potentially better for long documents or multi-faceted queries. Unmeasured in this codebase. |

**Key gap:** No guidance on when to select each mode. No benchmarks.

### E. Reranker Architecture

The current Rust sidecar is **embedding-only**. It should not be stretched into a Qwen reranker unless the sidecar grows a separate sequence-classification/score API. The current Python `QwenReranker` is functionally correct for official Qwen3-Reranker models, but it uses the original causal-LM yes/no-logit path and should be treated as an offline/deep-eval backend, not the interactive default.

Most viable path from the current codebase:

| Mode | Backend | Model | Candidate pool | Status |
|------|---------|-------|----------------|--------|
| `fast` | FastEmbed cross-encoder | `Xenova/ms-marco-MiniLM-L-6-v2` | 50 | Implemented; keep as default. |
| `balanced` | ONNX Runtime or vLLM score service | Qwen3-Reranker-0.6B sequence-classification conversion | 30 | Best next implementation target. |
| `quality` | vLLM score service | Qwen3-Reranker-4B sequence-classification conversion | 20 | Best quality/latency architecture if GPU serving is available. |
| `deep` | Python `transformers` causal-LM yes/no scoring | `Qwen/Qwen3-Reranker-4B` | 20 | Existing implementation path; evaluation/offline only. |

Recommended serving architecture:

```
raggy-mcp search_documents
  → hybrid candidate retrieval
  → exact/near-duplicate + story/page diversification
  → optional cheap MiniLM first-pass rerank
  → external Qwen reranker service for top 20-30 only
      ├── vLLM score endpoint for GPU/server deployments
      └── ONNX Runtime provider for CPU/local deployments
```

Do **not** run Qwen3-Reranker-4B over 100 candidates inline inside an MCP request handler. For interactive use, either:

1. Retrieve 80-100 hybrid candidates, MiniLM-rerank to top 30, then Qwen-rerank top 20-30.
2. Retrieve 80 hybrid candidates, diversify by story/page, then Qwen-rerank top 20-30 directly.

Implementation target:

```python
class QwenRerankerProvider:
    backend: Literal["transformers", "onnxruntime", "vllm_http"]
    model_name: str
    pool_size: int
```

Default policy should remain MiniLM until `onnxruntime` or `vllm_http` exists. Once implemented, prefer sequence-classification Qwen conversions over the original causal-LM yes/no-logit path for interactive use.

### F. Query Processing (unimplemented)

**Not present in this codebase:**
- Query rewriting/expansion before embedding
- Hypothetical document embedding (HyDE)
- Multi-query fusion
- Query-to-filter mapping (extract structured filters from natural language)

### G. Document Identity & Deduplication

Document identity must separate **what the document is** from **where it came from** and **which observed version was indexed**:

| Field | Definition | Purpose |
|-------|------------|---------|
| `document_id` | `sha256(normalized_extracted_content)` | Detect duplicate content, renamed files, and moved files. |
| `source_id` | `sha256(canonical_absolute_path)` | Track the path/source location independently from content identity. |
| `version_id` | `sha256(normalized_extracted_content + metadata_timestamp)` | Detect a concrete indexed version when content or source metadata changes. |

Hashing the absolute path as `document_id` is unsafe for RAG because moving or renaming a file creates a new document, duplicate content in different folders cannot be detected, and stale old versions are hard to distinguish from current content. Path identity belongs in `source_id`, not `document_id`.

Chunk-level dedup is still incomplete because Qdrant point IDs are generated at write time. A complete ingestion cleanup/checkpoint layer should upsert deterministic chunk IDs and retire stale points by `source_id` and/or superseded `version_id`.

### H. Citation & Response Formatting

Results return chunks with scores, paths, and metadata. **No citation formatting** (document-level grouping exists but no markdown citation rendering, no source attribution formatting for LLM consumption).

---

## 2. The Missing Eval Harness (Highest Priority Gap)

**This is the single most important missing piece.** Without it, every knob above is tuned by intuition rather than measurement.

### What an eval harness needs:

```
Corpus (ground-truth Q&A pairs)
    │
    ▼
For each (query, expected_docs):
    ├── RAG pipeline: embed → search → rerank → format
    ├── Compare retrieved docs vs expected_docs
    └── Score: recall@k, precision@k, MRR, NDCG
    │
    ▼
Report: per-mode, per-model, per-chunking-strategy scores
```

### Concrete design:

```python
# raggy/tools/eval_harness.py

@dataclass
class EvalSample:
    query: str
    relevant_doc_ids: list[str]  # ground truth

@dataclass
class EvalResult:
    recall_at_k: float
    precision_at_k: float
    mrr: float
    latency_ms: float
    config: EvalConfig  # chunk_size, overlap, embedding_model, mode, reranker

def run_eval(
    corpus_dir: str,           # Path with documents
    samples: list[EvalSample],  # Ground truth
    configs: list[EvalConfig],  # Sweep over knobs
) -> list[EvalResult]:
    """For each config: ingest corpus → search each query → score vs ground truth."""
    ...

def report(results: list[EvalResult]) -> str:
    """Produce a markdown table comparing configs."""
    ...
```

### Knobs the eval harness should sweep:

| Knob | Values to test |
|------|---------------|
| chunk_size | 256, 512, 1024, 2048 |
| chunk_overlap | 0, 64, 128, 256 |
| embedding_model | all-MiniLM-L6-v2, BAAI/bge-base-en-v1.5, Qwen3-Embedding-0.6B, Qwen3-Embedding-8B |
| retrieval_mode | dense, hybrid, rerank, late_interaction |
| reranker | none, ms-marco-MiniLM-L-6-v2, bge-reranker-base, Qwen3-Reranker-4B |
| prefetch_limit | 50, 80, 120 |
| rerank_top_k | 20, 40, 60 |

### Expected output:

```
## Eval Results (corpus: 42 docs, 50 queries)

| Config                  | Recall@5 | Recall@10 | Precision@5 | MRR   | Latency |
|-------------------------|----------|-----------|-------------|-------|---------|
| dense/all-MiniLM/1024    | 0.62     | 0.74      | 0.28        | 0.71  | 45ms    |
| hybrid/all-MiniLM/1024   | 0.68     | 0.79      | 0.31        | 0.76  | 82ms    |
| rerank/bge-reranker/512  | 0.74     | 0.83      | 0.35        | 0.81  | 340ms   |
| dense/Qwen3-8B/1024      | 0.71     | 0.81      | 0.33        | 0.78  | 120ms   |
```

Without this, every chunk_size change, every model swap, every mode toggle is guesswork.

---

## 3. Current Local Test Stack

The repository has a `raggy.yaml` that overrides built-in defaults for normal runs:

| Layer | Configured value | Notes |
|-------|------------------|-------|
| Qdrant mode | `server` | Uses `http://127.0.0.1:6333`, not embedded local storage. |
| Default collection | `documents_hybrid` | From `collections.default_collection`. |
| Dense embedder | `Qwen/Qwen3-Embedding-4B` | Routed through the Rust Qwen3 sidecar because Qwen3 model names use `Qwen3RustProvider`. |
| Dense dimension | 2560 | Must match the collection vector size. |
| Sparse embedder | `Qdrant/bm25` | FastEmbed sparse provider; no neural model download. |
| Default reranker | `Xenova/ms-marco-MiniLM-L-6-v2` | FastEmbed ONNX cross-encoder, CPU provider by default. |
| Qwen reranker | Python `transformers` only | The Rust sidecar is embedding-only; Qwen3-Reranker does not run through the sidecar path. |
| Qwen sidecar idle timeout | `300` seconds | Keeps the embedding sidecar warm for five minutes after the last request. Set `QWEN3_IDLE_TIMEOUT_SECONDS=0` to disable automatic idle shutdown. |
| Late interaction | `colbert-ir/colbertv2.0` | Only for late-interaction collections and ingestion. |
| PDF extractor | `PyMuPDF`, then `pdfminer.six`, then `pypdf` | Scanned pages still need OCR; this pipeline extracts embedded text only. |
| Chunking | `chunk_size=1000`, `chunk_overlap=150` | From `raggy.yaml`; also test 1200/180 for narrative corpora. |
| Search defaults | `rerank_prefetch_limit=80`, `rerank_top_k=50` | Balanced MiniLM default; test 30 fast, 50 balanced, 100 deep. |

For `/Users/kat/Downloads/Fairy-Tale.pdf`, the intended high-quality test path is:

1. Create/use a hybrid collection with dense vector size `2560` and sparse slot `sparse-bm25`.
2. Ingest the PDF with `mode="hybrid"` so dense and sparse vectors are stored together.
3. Search with `mode="rerank"` for answer-quality checks: hybrid candidate retrieval, then MiniLM reranking, then document-aware grouping.

Operational caveat: if `rust/qwen3_embedder/target/release/qwen3_embedder` is missing, the configured Qwen3 dense embedder cannot run. Either build the Rust sidecar first or use a pure FastEmbed dense model such as `BAAI/bge-base-en-v1.5` with a matching collection dimension.

### Repository Health Check — 2026-05-15

Current evaluation status:

| Check | Result |
|-------|--------|
| Python test suite | `96 passed, 2 skipped` via `PYTHONPATH=src uv run python -m pytest` |
| PDF extraction for `/Users/kat/Downloads/Fairy-Tale.pdf` | OK before extractor-order change: `pdfminer`, 611 pages, ~1,198,400 extracted characters, no extractor error |
| Qdrant server | Reachable at `http://127.0.0.1:6333` |
| Existing collections | `default_li`, `socratic_circles_hybrid_v2`, `King` |
| Suitable existing hybrid schema | `socratic_circles_hybrid_v2` has dense `qwen3-qwen3-embedding-4b` at 2560D and sparse `sparse-bm25` |
| Qwen3 Rust sidecar | Built successfully at `rust/qwen3_embedder/target/release/qwen3-embedder` |
| Web API route layer | Restored with `src/raggy_mcp/webui/api.py`; route tests pass |

Do not ingest the Fairy-Tale PDF into an unrelated existing collection. Create a dedicated hybrid collection, for example `fairy_tale_hybrid`, with the same 2560D Qwen3 dense vector and `sparse-bm25` sparse vector, then ingest the PDF in `hybrid` mode.

---

## 4. Current Architecture (for context)

```
┌──────────────────────────────────────────────────────────────────┐
│                        MCP CLIENTS                              │
│    (Claude Desktop, Windsurf, Cursor, LM Studio, Custom Agent)  │
└──────────┬──────────────┬──────────────┬────────────────────────┘
           │ stdio         │ SSE          │ streamable-http (with auth)
           ▼               ▼              ▼
┌──────────────────────────────────────────────────────────────────┐
│                     raggy-mcp (FastMCP)                         │
│                                                                │
│  Server        Tool         Embedding     Provider    Plan     │
│  (FastMCP)     Registry     Manager       Resolver    Registry │
│                WriteQueue   ToolProfile   Envelope    Discovery │
│                http_security                        schemas    │
└──────────────────────────┬───────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────┐
│                    SEARCH & RETRIEVAL                            │
│  dense ──── hybrid (RRF) ──── rerank ──── late_interaction      │
└──────────────────────────┬───────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────┐
│                        INGESTION                                 │
│  extract_text (per-format) → build_chunks (paragraph split)     │
│  → embed → WriteQueue → Qdrant                                  │
└──────────────────────────┬───────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────┐
│                     QDRANT (Vector DB)                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Module Map (What Exists Today)

```
src/raggy_mcp/
├── main.py              CLI entry (stdio/SSE/HTTP transport)
├── mcp_server.py        QdrantMCPServer — registers all MCP tools
├── qdrant.py            QdrantConnector — async client wrapper
├── settings.py          Pydantic config (ToolSettings, QdrantSettings)
├── embedding_manager.py Model registry + factory
│
├── ingest/
│   ├── extractor.py     Text extraction + paragraph-boundary chunking
│   ├── macos_metadata.py Spotlight metadata (additive-only)
│   └── document_id.py   document_id/content hash + source_id/path hash + version_id
│
├── embeddings/          Provider implementations
│   ├── base.py          ABC
│   ├── fastembed.py     ONNX sentence-transformers
│   ├── sparse.py        splade/bm25
│   ├── late_interaction.py ColBERTv2
│   └── qwen3_rust.py    Qwen3 via Rust sidecar
│
├── search/
│   ├── document_search.py   Document-grouped search
│   ├── retrieval_mode.py    Mode enum
│   ├── reranker.py          Cross-encoder reranking
│   └── filter_grammar.py    High-level filter compiler
│
├── mcp_runtime/         Infrastructure (plumbing)
│   ├── envelope.py      Success/failure result wrapper
│   ├── profiles.py      Tool visibility profiles
│   ├── plan_registry.py Report/apply gating
│   ├── provider_resolver.py Per-collection model assignment
│   ├── write_queue.py   Concurrency limiter
│   ├── discovery.py     Capability payloads
│   └── http_security.py Auth middleware
│
├── common/              Utilities
│   ├── filters.py       Qdrant payload index builder
│   ├── func_tools.py    Partial function builder
│   └── wrap_filters.py  Filter injection decorator
│
├── webui/               FastAPI dashboard
│   └── services.py      WebUI → Qdrant bridge
│
├── config.py            YAML config loader
├── docker_utils.py      Qdrant container lifecycle
├── enhanced_tool_descriptions.py
└── list_fastembed_models.py

rust/qwen3_embedder/     Qwen3 ONNX in Rust (sidecar binary)
scripts/                 Shell helpers
tests/                   20+ pytest files
tools/                   Diagnostics, profiling
```

---

## 6. Platform Metadata Strategy

Metadata is **layered**, platform-specific data is **additive only**:

```
Every OS: ──── path, filename, extension, size_bytes, is_hidden, has_text
macOS only: ── Spotlight (mdls) → content_type, title, authors, tags, dates (silent fail if absent)
Linux/Windows: ── (reserved for future)
```

Key rule: `_base_metadata()` always succeeds. macOS Spotlight is enhancement, never a requirement. Ingestion works identically on any OS.

---

## 7. Pipeline Detail

### Ingestion Pipeline

```
File Path
    │
    ├── get_macos_metadata_async()        → additive metadata dict
    │
    ├── extract_text(path)               → ExtractedDocument
    │     dispatcher by extension:
    │       .txt/.md     → readfile (charset fallback)
    │       .pdf         → PyMuPDF → pdfminer → pypdf fallback
    │       .docx        → python-docx (heading styles discarded)
    │       .json/.jsonl → formatted text
    │       .csv/.tsv    → delimited text
    │       .py/.js/...  → plain text read (no AST)
    │
    ├── build_chunks(doc, metadata)       → list[Chunk]
    │     algorithm: paragraph-split → overlap-merge
    │     config: chunk_size=1000, overlap=150
    │     issue: no format-specific structure awareness
    │
    ├── compute_document_id(doc.text)     → SHA-256 normalized content hash
    ├── compute_source_id(path)           → SHA-256 canonical absolute path hash
    ├── compute_version_id(text, mtime)   → SHA-256 content + metadata timestamp hash
    │
    ├── ProviderResolver.resolve()        → pick embedding model
    │     priority: explicit arg > collection mapping > default provider
    │
    ├── WriteQueue.run()                  → bounded concurrency
    │
    └── QdrantConnector.batch_store*()    → upsert vectors + payload
          dense | hybrid (dense+sparse) | late_interaction
```

### Search Pipeline

```
Query + mode
    │
    ├── RetrievalMode.parse(mode)         → dense | hybrid | rerank | late_interaction
    │
    ├── ProviderResolver.resolve()        → embedding model (per-request, multi-agent safe)
    │
    ├── Compile filter                    → Qdrant Filter or None
    │
    ├── Sparse provider (if hybrid/rerank)→ lazily initialized, cached by model name
    │
    ├── Reranker (if mode=rerank)         → FastEmbed | Qwen3 via Python transformers
    │
    ├── Prefetch + additional_queries     → parallel candidate retrieval
    │
    ├── Retrieve chunk pool               → top 40+ chunks by mode/prefetch settings
    ├── Rerank chunks (if mode=rerank)    → score chunks before document grouping
    ├── Diversify/group after reranking   → allow multiple chunks per document
    │                                        default max_chunks_per_doc: 2–3
    │                                        score = max(chunk_scores)
    │
    └── Result envelope                   → success + data + stats + warnings
```

---

## 8. MCP Tool Profiles

| Tool | Profile | Mutates | Purpose |
|------|---------|---------|---------|
| `qdrant_find` | minimal | ✗ | Basic dense search |
| `qdrant_store` | minimal | ✓ | Single-entry store |
| `search_documents` | canonical | ✗ | Grouped doc search (4 modes, the main retrieval entry point) |
| `ingest_file` | canonical | ✓ | Single file → extract → chunk → embed → store |
| `ingest_folder` | canonical | ✓ | Recursive folder ingestion (report/apply gated) |
| `create_collection` | full | ✓ | Collection with specified embedding model |
| `create_hybrid_collection` | full | ✓ | Dense + sparse vector slots |
| `create_late_interaction_collection` | full | ✓ | ColBERT multivector |
| `delete_collection` | full | ✓ | Destructive (report/apply gated) |
| `set_collection_embedding_model` | full | ✓ | Per-collection model assignment |
| Various discovery tools | full | ✗ | Schema, extractors, modes, capabilities |

---

## 9. Retrieval Compatibility Invariant

Collection schema, indexed vectors, embedding model, sparse model, late-interaction model, reranker, and query mode must not drift independently. Every collection should have a persisted `RetrievalProfile`:

```python
@dataclass
class RetrievalProfile:
    dense_model: str
    dense_dimension: int
    sparse_model: str | None
    late_interaction_model: str | None
    reranker_model: str | None
    supported_search_modes: set[str]
    created_at: str
    schema_version: int
```

Search should validate the requested mode against this profile before embedding or querying:

| Requested mode | Required profile fields |
|----------------|-------------------------|
| `dense` | `dense_model`, matching `dense_dimension` |
| `hybrid` | dense fields + `sparse_model`; reject if sparse vectors were not indexed |
| `rerank` | dense or hybrid candidate mode + compatible `reranker_model` |
| `late_interaction` | `late_interaction_model`; reject if multivectors were not indexed |

Invalid combinations should fail early with a clear error. For example, `mode=hybrid` must not silently run against a dense-only collection, and a late-interaction collection should not be searched with a dense-only provider just because a default provider exists.

---

## 10. Result Grouping Policy

Document grouping is useful for avoiding duplicate-looking results, but grouping too early hides evidence when a single long PDF or notebook contains several relevant sections. The default retrieval policy should be:

1. Retrieve a sufficiently wide chunk pool, usually at least top 40 chunks.
2. Rerank chunks before document grouping when a reranker is active.
3. Apply section/document diversification after reranking.
4. Return more than one chunk per document by default, with `max_chunks_per_doc = 2` or `3`.
5. Keep one-chunk-per-document as an explicit high-diversity option, not the default.

This keeps the UX benefit of document grouping while preserving enough evidence for synthesis-style RAG answers.

---

## 11. Key Insights & Recommendations

### What's working well
1. **Format-specific text extraction** — PDF, DOCX, JSON, CSV all have dedicated parsers, not a one-size-fits-all approach
2. **Multiple retrieval modes** — dense, hybrid, rerank, late_interaction gives flexibility
3. **Per-request ProviderResolver** — multi-agent safe model selection without global mutable state
4. **Additive-only metadata** — platform portability is baked into the design
5. **WriteQueue concurrency control** — prevents stampeding in multi-agent scenarios

### What's missing (highest priority first)

| Priority | Gap | Impact |
|----------|-----|--------|
| **P0** | No eval harness | Every tuning decision is guesswork. Cannot compare chunk sizes, models, or modes empirically. |
| **P1** | Structure-aware chunking | Markdown heading paths, code symbols, PDF page/section boundaries, DOCX heading styles all discarded. Chunks cross logical sections. |
| **P1** | Format-specific metadata in chunks | No heading_path, section, page number — severely limits filter/search precision for structured docs. |
| **P2** | No query rewriting/expansion | Single-query embedding means no HyDE, no multi-query fusion, no query-to-filter mapping. |
| **P2** | Incomplete document lifecycle cleanup | Content/source/version IDs exist, but repeated ingestion can still leave stale points without deterministic chunk IDs and cleanup by `source_id`/`version_id`. |
| **P2** | No persisted RetrievalProfile invariant | Collection type, embedding model, sparse vectors, late-interaction vectors, reranker, and search mode can drift unless validated together. |
| **P2** | PDF scanning detection exists but silent failure | Scanned PDFs produce empty text with no actionable warning. |
| **P3** | No citation formatting | Results have paths and scores but no ready-to-use citation format for LLM consumption. |
| **P3** | DOCX heading styles discarded | python-docx extracts headings as plain paragraphs; hierarchical structure lost. |

---

The XMind mind map is in `raggy-technical-scheme.xmind` (109 nodes, 14 topics).
