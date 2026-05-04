# Enhanced Tool Descriptions for Qdrant MCP Server

## Improved Tool Descriptions with Clear Usage Examples

DEFAULT_TOOL_STORE_DESCRIPTION = (
    "Store a single piece of information in a Qdrant collection. Qdrant will automatically embed the content server-side. "
    "Usage: qdrant_store(content='Your text here', collection_name='my_collection', "
    "metadata='{\"key\": \"value\"}', entry_id='optional-custom-id'). "
    "Returns success/failure message."
)

DEFAULT_TOOL_FIND_DESCRIPTION = (
    "Search for information in a Qdrant collection using semantic similarity. Qdrant will automatically embed the query server-side. "
    "Usage: qdrant_find(query='search terms', collection_name='my_collection'). "
    "Returns relevant entries with their content and metadata."
)

DEFAULT_TOOL_BATCH_STORE_DESCRIPTION = (
    "Store multiple entries efficiently in one operation. Qdrant will automatically embed the content server-side. "
    "Usage: qdrant_store_batch(entries=[{\"content\": \"text1\", \"metadata\": {\"key\": \"val1\"}}, "
    "{\"content\": \"text2\", \"metadata\": {\"key\": \"val2\"}}], collection_name='my_collection'). "
    "Each entry is a dict with 'content' (required), 'metadata' (optional dict), and 'id' (optional string). "
    "No limit on batch size for storage - processes all entries provided."
)

DEFAULT_TOOL_LIST_COLLECTIONS_DESCRIPTION = (
    "List all Qdrant collections in the database. "
    "Usage: list_collections(). Returns list of collection names."
)

DEFAULT_TOOL_CREATE_COLLECTION_DESCRIPTION = (
    "Create a new Qdrant collection with specified vector size and embedding model. "
    "Usage: create_collection(collection_name='my_collection', vector_size=768, "
    "distance='cosine', embedding_model='sentence-transformers/all-mpnet-base-v2'). "
    "Common vector sizes: 384 (MiniLM), 768 (MPNet/BGE-base), 1024 (BGE-large). "
    "Model name should NOT include provider suffix like '(fastembed)'."
)

DEFAULT_TOOL_GET_COLLECTION_INFO_DESCRIPTION = (
    "Get detailed information about a collection including size, vector config, and embedding model. "
    "Usage: get_collection_info(collection_name='my_collection'). "
    "Returns points count, vector size, distance metric, and assigned embedding model."
)

DEFAULT_TOOL_DELETE_COLLECTION_DESCRIPTION = (
    "Permanently delete a collection and all its data. Two-step report/apply: "
    "first call with mode='report' to preview and get a plan_id, then call again "
    "with mode='apply' and plan_id=<id> to execute. "
    "Usage: delete_collection(collection_name='my_collection', mode='report') → "
    "delete_collection(collection_name='my_collection', mode='apply', plan_id='plan_xyz'). "
    "Plans expire after 10 minutes. Cannot be undone once applied."
)

DEFAULT_TOOL_HYBRID_SEARCH_DESCRIPTION = (
    "Advanced search with similarity scores and filtering options. Qdrant will automatically embed the query server-side. "
    "Usage: hybrid_search(query='search terms', collection_name='my_collection', "
    "limit=10, min_score=0.7, include_scores=True). "
    "Returns entries with similarity scores for ranking results."
)

DEFAULT_TOOL_SET_COLLECTION_EMBEDDING_MODEL_DESCRIPTION = (
    "Assign a specific embedding model to a collection. "
    "Usage: set_collection_embedding_model(collection_name='my_collection', "
    "model_name='sentence-transformers/all-mpnet-base-v2'). "
    "Model name should be exact as shown in list_embedding_models, WITHOUT provider suffix. "
    "Collection vector size must match model output size."
)

DEFAULT_TOOL_LIST_EMBEDDING_MODELS_DESCRIPTION = (
    "List all available embedding models with their specifications. "
    "Usage: list_embedding_models(). "
    "Returns model names, vector sizes, and descriptions. "
    "When using a model name in other tools, use ONLY the model name without '(fastembed)' suffix."
)

DEFAULT_TOOL_SCROLL_DESCRIPTION = (
    "Browse collection contents with pagination. "
    "Usage: scroll_collection(collection_name='my_collection', limit=20, offset='point_id'). "
    "Returns entries and next offset for pagination."
)

DEFAULT_TOOL_SET_COLLECTION_EMBEDDING_MODEL_IMPL_DESCRIPTION = (
    "Assign an embedding model to a specific collection. "
    "Usage: set_collection_embedding_model(collection_name='my_docs', "
    "model_name='Qwen/Qwen3-Embedding-8B'). "
    "Per-request safe: this does NOT mutate any process-global state, so multiple agents "
    "sharing one server can target different collections with different models without "
    "interfering with each other. The model still has to match the collection's existing "
    "vector dimensions, so for an empty/new collection prefer create_collection with the "
    "embedding_model argument instead. Use list_embedding_models to discover options."
)

DEFAULT_TOOL_INGEST_FILE_DESCRIPTION = (
    "Ingest a local file into a Qdrant collection. Extracts text and macOS Spotlight metadata automatically. "
    "Supported formats include .txt, .md, .json, .jsonl, .csv, .tsv, code/config text files, .pdf, and .docx. "
    "Usage: ingest_file(file_path='/path/to/file.pdf', collection_name='my_docs'); collection_name defaults to 'documents'. "
    "Optional: extra_metadata (JSON string of additional fields). "
    "macOS metadata (Finder tags, comments, dates, content type) is extracted and stored as filterable payload. "
    "Returns count of chunks stored and extraction stats."
)

DEFAULT_TOOL_INGEST_FOLDER_DESCRIPTION = (
    "Recursively ingest all supported files in a folder into a Qdrant collection. "
    "Supported formats include plain text, Markdown, JSON/JSONL, CSV/TSV, code/config text files, PDF, and DOCX. Hidden files and system directories are skipped by default. "
    "Usage: ingest_folder(folder_path='/path/to/folder', collection_name='my_docs'); collection_name defaults to 'documents'. "
    "Optional: recursive (default True), skip_hidden (default True), extra_metadata (JSON string), "
    "embedding_model (per-request override), mode ('dense' | 'hybrid'). "
    "Two-step gating supported: run_mode='report' returns a dry-run plan with file inventory and "
    "a plan_id; run_mode='apply' (default) ingests, optionally validating the supplied plan_id. "
    "Each file gets macOS Spotlight metadata extracted and stored as filterable payload. "
    "Returns the structured envelope with files_processed, chunks_stored, and per-file errors."
)

DEFAULT_TOOL_SEARCH_DOCUMENTS_DESCRIPTION = (
    "Semantic search with file-level grouping and reranking. "
    "Usage: search_documents(query='...', collection_name='my_docs', limit=10, "
    "chunks_per_document=4, filter={'must': [{'field':'extension','op':'==','value':'pdf'}]}, "
    "mode='dense'|'hybrid'|'rerank'|'late_interaction'). "
    "Groups chunks by document_id so a single long file does not dominate results, "
    "returning one summary entry per file with its best representative chunks. "
    "Filter grammar uses must/should/must_not clauses with ops: ==, !=, >, >=, <, <=, any, except. "
    "mode='hybrid' requires a hybrid collection (see create_hybrid_collection); "
    "mode='rerank' adds a cross-encoder rerank pass over hybrid candidates; "
    "mode='late_interaction' requires create_late_interaction_collection and ColBERT-style ingestion."
)

DEFAULT_TOOL_BOOTSTRAP_INDEXES_DESCRIPTION = (
    "Create all macOS metadata payload indexes on a collection up front. "
    "Usage: bootstrap_collection_indexes(collection_name='my_docs'). "
    "Recommended before bulk ingest of large folders — Qdrant filter-aware HNSW behavior "
    "improves when indexes exist before vectors are added."
)

DEFAULT_TOOL_CREATE_HYBRID_COLLECTION_DESCRIPTION = (
    "Create a collection set up for HYBRID retrieval (dense + sparse vectors with RRF fusion). "
    "Usage: create_hybrid_collection(collection_name='my_docs_hybrid', "
    "embedding_model='Qwen/Qwen3-Embedding-8B', sparse_model='Qdrant/bm25'). "
    "Use search_documents(mode='hybrid') or mode='rerank' to query. "
    "Best for filename, identifier, exact-phrase, and semantic queries blended together."
)
