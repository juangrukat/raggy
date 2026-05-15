# raggy-mcp

Local retrieval infrastructure for MCP clients, desktop agents, and REST users.
It stores documents in Qdrant, embeds them with the configured model, and exposes
tools for ingestion, collection setup, semantic search, hybrid search, reranking,
metadata discovery, and safe destructive operations.

The repository is a Python package named `raggy-mcp`. It provides two
entry points:

- `raggy-mcp`: MCP server over `stdio`, `sse`, or `streamable-http`.
- `raggy-mcp-webui`: NiceGUI dashboard + optional REST API for browser-based management.

The default local profile is intended for real day-to-day use: document-level
search, folder/file ingestion, collection creation, hybrid collection creation,
late-interaction collection creation, embedding model assignment, and discovery
tools are enabled. Raw chunk-level and destructive tools are available only when
the server is started with the `full` tool profile.

## Memory and Model Selection

Raggy supports multiple embedding models, ranging from lightweight (700 MB)
to large (14+ GB on disk, 8-16 GB when loaded into unified memory). The
default model is `Qwen/Qwen3-Embedding-4B` which requires approximately
**8-16 GB of unified memory** when loaded via the Candle/Metal sidecar.

**Warning:** On machines with 16 GB or less of RAM, loading the default
4B-parameter model alongside other applications (browser, IDE, Qdrant) can
exhaust available memory and cause the system to swap or become unresponsive.

### Model tiers by memory footprint

| Model | Parameters | VRAM/RAM (F16) | Disk cache | Use case |
|---|---|---|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` | 22M | ~400 MB | ~90 MB | Testing / low-resource |
| `Qwen/Qwen3-Embedding-0.6B` | 0.6B | ~1.2 GB | ~1.1 GB | Lightweight production |
| `Qwen/Qwen3-Embedding-4B` | 4B | ~8 GB | ~7.5 GB | Balanced (default) |
| `Qwen/Qwen3-Embedding-8B` | 8B | ~16 GB | ~14 GB | High-quality, memory-heavy |

Set the model in `raggy.yaml` under `models.dense_embedding` or via the
`EMBEDDING_MODEL` environment variable.

### Memory optimizations applied

Raggy uses a Rust sidecar binary (`qwen3-embedder`) for Qwen3 embedding models
via Candle (a ML framework). The sidecar loads models in **F16 precision** on
Apple Silicon Metal — half the memory of FP32. Three additional mitigations
are built in by default:

1. **Pre-warm dummy embedding** — The sidecar is loaded and a short dummy
   embedding is run during server startup (not on the first user query). This
   forces Metal to pre-allocate activation buffers and inference workspace in
   a predictable warmup period rather than spiking during real requests.

2. **Idle keepalive (120s)** — After each embedding request, a 120-second
   countdown is reset. As long as queries arrive within the window, the
   sidecar process stays alive and Metal's pre-allocated buffers are reused —
   subsequent queries add nearly zero memory.

3. **F16 default on Metal** — The Rust sidecar defaults to float16 on Apple
   Silicon, cutting the model weight memory and inference activation workspace
   roughly in half compared to FP32.

These optimizations prevent the ~8 GB inference-time memory spike that would
otherwise occur on the first query, and keep the warm sidecar's memory
footprint stable across a burst of searches.

## What This Server Does

At a high level, the server turns files or text into searchable Qdrant points,
then returns ranked evidence grouped by document.

```mermaid
flowchart TD
    A["MCP client or REST caller"] --> B["Tool / API request"]
    B --> C{"Operation"}

    C -->|"create collection"| D["Create Qdrant vector schema"]
    C -->|"ingest file/folder"| E["Extract text and file metadata"]
    E --> F["Chunk extracted text"]

    subgraph Embedding [" "]
        direction LR
        G1{"Model type?"}
        G1 -->|"FastEmbed (lightweight)"| G2["ONNX Runtime<br>in-process"]
        G1 -->|"Qwen3 (default)"| G3["Rust sidecar<br>qwen3-embedder"]
        G3 --> G4["Candle + Metal<br>F16 precision"]
    end

    F --> G1
    G2 --> H["Upsert dense, sparse, or late-interaction vectors"]
    G4 --> H
    H --> I[("Qdrant collection")]

    C -->|"search_documents"| J{"Search mode"}
    J -->|"dense"| L["Dense vector search"]
    J -->|"hybrid"| M["Dense + sparse prefetch with fusion"]
    J -->|"rerank"| N["Hybrid prefetch, then reranker scoring"]
    J -->|"late_interaction"| O["ColBERT-style multivector search"]
    L --> P["Group chunks by document_id"]
    M --> P
    N --> P
    O --> P
    P --> Q["Return documents with best matching chunks"]

    style G3 fill:#4a4,color:#fff
    style G4 fill:#4a4,color:#fff

The ingest path preserves payload metadata such as path, filename, parent path,
document id, chunk index, total chunks, character count, page count when
available, extraction method, ingest time, and macOS metadata when available.

## Enabled MCP Functions

Tool visibility is controlled by `QDRANT_MCP_TOOL_PROFILE`. The default is
`canonical`.

### Minimal Profile

Set `QDRANT_MCP_TOOL_PROFILE=minimal` for the smallest useful read/search
surface.

| Tool | Purpose |
| --- | --- |
| `search_documents` | Main document-level search tool. Supports `dense`, `hybrid`, `rerank`, and `late_interaction` modes. Returns distinct documents with their best chunks. |
| `ingest_file` | Extracts, chunks, embeds, and stores one supported file. Supports dense, hybrid, and late-interaction ingest modes. |
| `ingest_folder` | Recursively ingests supported files from a folder. Supports `run_mode=report` dry runs and `run_mode=apply`. |
| `list_embedding_models` | Lists embedding models known to the server and supported distance metrics. |
| `list_collections` | Lists Qdrant collections. |
| `get_collection_info` | Returns collection counts, vector size, distance metric, status, and optimizer status. |
| `get_indexed_fields` | Shows indexed payload fields and the supported filter grammar. |
| `get_supported_extractors` | Lists supported file extensions and extraction methods. |
| `get_collection_schema` | Returns schema and status for a specific collection. |
| `list_search_modes` | Describes `dense`, `hybrid`, `rerank`, and `late_interaction`. |
| `get_server_capabilities` | Returns server profile, transports, enabled feature flags, extractors, search modes, and available models. |

### Canonical Profile

`canonical` includes all minimal tools plus setup and collection lifecycle tools.
This is the default profile.

| Tool | Purpose |
| --- | --- |
| `create_collection` | Creates a dense-vector collection for a chosen embedding model. |
| `create_hybrid_collection` | Creates a collection with dense and sparse vector slots for hybrid retrieval. |
| `create_late_interaction_collection` | Creates a multivector collection for ColBERT-style MaxSim retrieval. |
| `bootstrap_collection_indexes` | Creates macOS/document metadata payload indexes on an existing collection. |
| `set_collection_embedding_model` | Assigns an embedding model to a collection without changing global server state. |

### Full Profile

Set `QDRANT_MCP_TOOL_PROFILE=full` for raw/admin tools. Use this profile with
care because it exposes destructive and low-level operations.

| Tool | Purpose |
| --- | --- |
| `delete_collection` | Deletes a collection with a two-step `report` then `apply` plan gate. |
| `qdrant_find` | Legacy/raw chunk-level semantic search. |
| `qdrant_store` | Stores one raw text entry. |
| `qdrant_store_batch` | Stores multiple raw text entries. |
| `scroll_collection` | Browses raw collection entries with pagination. |
| `hybrid_search` | Legacy chunk-level scored search. |

When `QDRANT_READ_ONLY=true`, write and mutation tools are not registered.

## REST API Functions

The FastAPI app mirrors the main capabilities for JSON HTTP callers:

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Health, active embedding model, vector size, and write queue stats. |
| `GET /collections` | List collections. |
| `GET /collections/{name}` | Collection details. |
| `POST /collections` | Create a dense collection. |
| `POST /collections/hybrid` | Create a hybrid dense+sparse collection. |
| `DELETE /collections/{name}` | Delete a collection. |
| `POST /collections/{name}/bootstrap_indexes` | Ensure metadata payload indexes. |
| `POST /store` | Store one raw text entry. |
| `POST /store_batch` | Store raw text entries in bulk. |
| `GET /scroll/{name}` | Browse collection entries. |
| `POST /search` | Chunk-level search. |
| `POST /search_documents` | Document-level grouped search. |
| `POST /ingest/file` | Ingest one file. |
| `POST /ingest/folder` | Ingest a folder. |
| `GET /embedding_models` | List available embedding models. |
| `POST /embedding_models/active` | Change the active REST embedding provider. |

### Quick Start with Config File

The easiest way to get running is with the configuration file:

```bash
# 1. Copy and edit the config
cp raggy.yaml ~/.config/raggy-mcp/config.yaml

# 2. Start Qdrant (native binary or Docker)
./scripts/local-run-qdrant.sh

# 3. Start the MCP server — reads config automatically
./scripts/run-server-mcp.sh --transport streamable-http
```

## Configuration

All settings are consolidated in a single YAML file.

### Config file location

The first file found is used (priority order):
1. `$QDRANT_CONFIG` environment variable
2. `./raggy.yaml` (relative to working directory)
3. `~/.config/raggy-mcp/config.yaml`

### Priority chain

```
CLI args / request params
  → environment variables
    → config file (raggy.yaml)
      → built-in defaults
```

Environment variables always override the config file. Set `QDRANT_MODE=embedded`
to override `raggy.yaml` without editing it.

### Config reference

See `raggy.yaml` in the repo root for a fully annotated example.
Key sections:

| Section | Controls |
|---------|----------|
| `runtime` | Qdrant mode/URL, MCP transport, tool profile |
| `models` | Embedding model, sparse model, reranker, Qwen3 sidecar |
| `ingest` | Chunk size, batch size, write concurrency |
| `search` | Default mode, rerank limits, diversity |
| `collections` | Default collection name, naming conventions |

## Installation

Requirements:

- Python 3.10 or newer.
- `uv`.
- Native Qdrant binary (recommended for multi-agent server mode) or Docker.
  The project ships a macOS arm64 binary at `.local/bin/qdrant` (v1.17.1).
  A LaunchAgent at `~/Library/LaunchAgents/com.qdrant.server.plist` auto-starts it on login.
- Rust toolchain, only if building the Qwen3 embedding sidecar locally.
- macOS for Apple Silicon Metal/MPS acceleration and macOS metadata capture.

Install Python dependencies and build the Qwen3 sidecar:

```bash
./scripts/local-install.sh
```

Equivalent manual install:

```bash
uv sync --frozen --group dev
cargo build --release --manifest-path rust/qwen3_embedder/Cargo.toml
```

Optional reranker dependencies for Qwen3 reranker models:

```bash
uv pip install 'raggy-mcp[reranking]'
```

Run the test suite:

```bash
uv run --locked pytest
```

## Running Locally

### Recommended Setup

For multi-agent use, run Qdrant as a persistent server and connect MCP clients
to it.

Recommended infrastructure:

- **Qdrant**: native server at `http://127.0.0.1:6333`
- **MCP**: server mode with `QDRANT_MODE=server`
- **Embedded mode**: development/testing only

Recommended retrieval policy:

- Use **hybrid collections** for general-purpose RAG.
- Use `mode="hybrid"` as the normal starting point.
- Use `mode="rerank"` for higher-quality evidence selection.
- Use `mode="late_interaction"` for high-recall conceptual search, but only
  with a late-interaction collection.
- Do not switch embedding models on an existing collection unless you
  re-ingest.

### Shared Qdrant Server Mode

Server mode lets the MCP server, REST API, and multiple agents share the same
Qdrant instance.

```bash
./scripts/local-run-qdrant.sh
./scripts/run-server-mcp.sh --transport streamable-http
```

The MCP endpoint is:

```text
http://127.0.0.1:8000/mcp/
```

Useful checks:

```bash
./scripts/check-server-qdrant.sh
./scripts/smoke-test-server-mcp.sh
./scripts/local-doctor.sh
```

### WebUI Dashboard

The `raggy-mcp-webui` command starts a NiceGUI dashboard for managing your
local retrieval infrastructure through a browser. The dashboard provides:

- **Dashboard** — Server status, Qdrant mode, active embedding model, write queue
- **Search Playground** — Query collections with dense/hybrid/rerank/late-interaction modes
- **Collections** — Create, inspect, bootstrap indexes, and delete collections
- **Ingestion** — Ingest files and folders with supported format display
- **Models** — View and understand embedding, sparse, reranker, and late-interaction models
- **Configuration** — View and edit settings with source provenance (env/config/default)
- **Admin** — Tool profiles, maintenance commands, and logs

Quick start:

```bash
# Install with webui dependencies
uv sync --frozen --group dev

# Start the dashboard (port 8080 by default)
raggy-mcp-webui
```

Or with the REST API running alongside on a separate port:

```bash
# Dashboard on 8080, REST API on 8765
raggy-mcp-webui --rest-port 8765
```

Full options:

```bash
raggy-mcp-webui --help
# Usage:
#   --host HOST         Bind address (default: 127.0.0.1)
#   --port PORT         UI port (default: 8080)
#   --rest-port PORT    Also start REST API on this port
#   --rest-cors [...]   CORS origins for REST API
#   --reload            Auto-reload on code changes (dev)
```

Open http://127.0.0.1:8080 in your browser.

The webui reads the same `raggy.yaml` config and environment variables as the
MCP server. Any settings edited in the Configuration page are saved to
`raggy.yaml` and take effect on restart.

**Architecture note:** The webui initializes its own QdrantConnector and
embedding providers. When running in `QDRANT_MODE=server`, the webui and
MCP server can share the same Qdrant instance without conflict. In embedded
mode, only one process can hold the lock at a time.

### Stdio MCP Mode

For clients that spawn the server directly:

```bash
uv run --locked raggy-mcp
```

or with local defaults loaded:

```bash
./scripts/local-run-mcp.sh
```

### Embedded Qdrant Mode (Dev / Single-User Only)

Embedded mode stores Qdrant data in a local directory and locks the storage to
a single process. Use this for local testing, not multi-agent setups.

```bash
QDRANT_MODE=embedded ./scripts/local-run-mcp.sh
```

Embedded mode cannot be shared between the MCP server and the REST API
simultaneously — the storage is locked by whoever opens it first.

## Configuration

Important environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `QDRANT_MCP_TOOL_PROFILE` | `canonical` | Tool surface: `minimal`, `canonical`, or `full`. |
| `QDRANT_URL` | unset | Qdrant server URL. When set, server mode is used. |
| `QDRANT_LOCAL_PATH` | `storage` or `.local/qdrant-storage` via scripts | Embedded Qdrant storage path. |
| `COLLECTION_NAME` | `documents` | Default collection for tools that allow omission. |
| `QDRANT_READ_ONLY` | `false` | Disables write/mutation tools when true. |
| `EMBEDDING_PROVIDER` | `fastembed` | Embedding provider type. |
| `EMBEDDING_MODEL` | `Qwen/Qwen3-Embedding-4B` | Default dense embedding model. |
| `EMBEDDING_DEVICE` | `auto` | Device hint: `auto`, `cpu`, `cuda`, `mps`, or supported local equivalent. |
| `QWEN3_SIDECAR_PATH` | built sidecar path via scripts | Rust Qwen3 embedder binary path. |
| `QWEN3_IDLE_TIMEOUT_SECONDS` | `300` | Seconds to keep the Qwen3 sidecar warm after the last request. `0` disables automatic idle shutdown. |
| `QDRANT_SPARSE_MODEL` | `Qdrant/bm25` | Sparse model for hybrid/rerank retrieval. |
| `QDRANT_RERANKER_MODEL` | `Xenova/ms-marco-MiniLM-L-6-v2` | Default reranker for `mode=rerank`. |
| `QDRANT_RERANK_PREFETCH_LIMIT` | `0` | Candidate pool before reranking; `0` means auto. Try 30 fast, 50 balanced, 100 deep. |
| `QDRANT_RERANK_TOP_K` | `0` | Max candidates scored by reranker; `0` means balanced auto. |
| `QDRANT_INGEST_CHUNK_SIZE` | `1000` | Ingest chunk size in characters. |
| `QDRANT_INGEST_CHUNK_OVERLAP` | `150` | Character overlap between chunks. |
| `QDRANT_WRITE_MAX_CONCURRENCY` | `1` | Concurrent embedding/upsert jobs per server process. |
| `QDRANT_WRITE_QUEUE_SIZE` | `8` | Queued write jobs before requests are rejected. |
| `MCP_TRANSPORT` | `stdio` | `stdio`, `sse`, or `streamable-http`. |
| `MCP_HOST` | `127.0.0.1` | HTTP bind host. |
| `MCP_PORT` | `8000` | HTTP bind port. |
| `MCP_HTTP_AUTH_TOKEN` | unset | Optional bearer token for streamable HTTP. |
| `MCP_HTTP_ALLOWED_ORIGINS` | local defaults | Allowed origins for HTTP Origin validation. |

## How Data Flows

### Collection Setup

1. A client creates a dense, hybrid, or late-interaction collection.
2. The server resolves the requested embedding or late-interaction model.
3. Qdrant receives the correct vector schema: dense vector, dense+sparse
   vectors, or multivectors.
4. Metadata payload indexes can be created up front with
   `bootstrap_collection_indexes`; ingestion also ensures the metadata indexes.

### File Ingestion

1. `ingest_file` or `ingest_folder` receives an absolute path and target
   collection.
2. The extractor reads supported formats:
   `.txt`, `.md`, `.json`, `.jsonl`, `.csv`, `.tsv`, `.pdf`, `.docx`, and many
   code/config text formats.
3. The server collects file metadata and macOS metadata where available.
4. Text is split into paragraph-aware chunks using the configured chunk size and
   overlap.
5. The selected embedding provider embeds the chunks.
6. Dense mode stores dense vectors; hybrid mode stores dense plus sparse BM25 or
   BM42 vectors; late-interaction mode stores multivectors.
7. Qdrant payloads keep the original chunk text under `document` and metadata
   under `metadata`.

### Search

1. `search_documents` receives a query, collection, filter, and retrieval mode.
2. The server resolves the embedding model by request override, collection
   assignment, then process default.
3. Dense mode performs vector search.
4. Hybrid mode uses dense and sparse retrieval and fuses candidates.
5. Rerank mode performs hybrid prefetch, then scores candidates with the
   configured reranker.
6. Late-interaction mode uses a ColBERT-style provider against a multivector
   collection.
7. Results are deduplicated and grouped by `document_id`, then returned as
   documents with top matching chunks, scores, metadata, and warnings.

### Safe Mutations

`ingest_folder` supports `run_mode=report` to preview a folder ingest before
applying it. `delete_collection` is only visible in the `full` profile and
requires a `report` call that returns a `plan_id`, followed by `apply` with that
same plan id.

## Technical Details

- MCP runtime: FastMCP.
- WebUI runtime: NiceGUI.
- REST runtime: FastAPI and Uvicorn.
- Vector store: Qdrant client, either embedded local storage or server URL.
- Dense embeddings: FastEmbed-compatible providers, with Qwen3 sidecar support
  via a Rust subprocess (`qwen3-embedder`) that loads the model through Candle
  + Metal with F16 precision.
- Qwen3 sidecar memory: the sidecar is pre-warmed with a dummy embedding on
  startup and kept alive for 120 seconds of inactivity between requests. This
  prevents ~8 GB Metal activation-buffer spikes on real queries and keeps the
  warm sidecar's memory footprint stable across search bursts.
- Sparse retrieval: `Qdrant/bm25` by default, with BM42 option.
- Reranking: FastEmbed cross-encoders by default; Qwen3 rerankers require the
  `reranking` extra.
- Late interaction: FastEmbed late-interaction provider, defaulting to
  `colbert-ir/colbertv2.0`.
- Write safety: a bounded async write queue serializes or limits embedding and
  upsert work.
- Multi-client safety: collection embedding assignments are persisted per
  collection and resolved per request rather than mutating global MCP state.
- HTTP safety: streamable HTTP binds to loopback by default, validates Origin,
  and supports optional bearer auth.

## Useful Maintenance Commands

```bash
# Reset local server processes
./scripts/reset-server-qdrant.sh
```

Stops local MCP/REST/embedder processes without deleting server-mode Qdrant
storage. Use `--stop-docker`, `--remove-docker`, `--wipe`, or
`--wipe-embedded` only when intentionally changing or deleting local Qdrant
state.

```bash
./scripts/local-configure-hermes.py
```

Writes a local client configuration for Hermes.

## Example MCP Flow

1. Start Qdrant:

   ```bash
   ./scripts/local-run-qdrant.sh
   ```

2. Start MCP:

   ```bash
   ./scripts/run-server-mcp.sh --transport streamable-http
   ```

3. Create a collection with `create_hybrid_collection`.

4. Ingest files with `ingest_folder` using `mode="hybrid"`.

5. Query with `search_documents` using `mode="rerank"` when quality matters or
   `mode="hybrid"` when latency matters.
