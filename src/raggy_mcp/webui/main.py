"""Entry point: raggy-mcp-webui — Gradio dashboard."""

# ruff: noqa: E701, E702

import argparse
import sys

import gradio as gr

from raggy_mcp.config import load_qdrant_config

load_qdrant_config()

from raggy_mcp.webui.services import (  # noqa: E402
    get_connector,
    get_embedding_manager,
    get_embedding_provider,
    get_embedding_settings,
    get_health,
    get_qdrant_settings,
    get_write_queue,
    guarded_qdrant_call,
    init_services,
    search_documents,
)

SYSTEM_DEFAULT = "__system_default__"
SUPPORTED_EXTS = ".txt,.md,.json,.jsonl,.csv,.tsv,.pdf,.docx,.py,.js,.ts,.html,.css,.yaml,.yml,.toml,.xml,.cfg,.ini"


# ── Tab: Dashboard ──
def build_dashboard():
    gr.Markdown("## Dashboard")

    async def refresh():
        h = await get_health()
        return (
            h.get("qdrant_mode", "?").title(),
            h.get("qdrant_url", ""),
            h.get("tool_profile", "?").title(),
            str(h.get("read_only", False)),
            h.get("embedding_model", "?"),
            f"{h.get('vector_size', '?')}D",
            h.get("default_collection", "?"),
            h.get("default_search_mode", "?"),
            str(h.get("collection_count", 0)),
            str(h.get("write_queue", {}).get("size", 0)),
        )

    btn = gr.Button("Refresh", variant="secondary", size="sm")
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Infrastructure")
            qm = gr.Textbox(label="Qdrant Mode", interactive=False)
            qu = gr.Textbox(label="Qdrant URL", interactive=False)
            tp = gr.Textbox(label="Tool Profile", interactive=False)
            ro = gr.Textbox(label="Read-only", interactive=False)
        with gr.Column(scale=1):
            gr.Markdown("### Models & Collections")
            em = gr.Textbox(label="Active Embedding Model", interactive=False)
            es = gr.Textbox(label="Vector Size", interactive=False)
            dc = gr.Textbox(label="Default Collection", interactive=False)
            dm = gr.Textbox(label="Default Search Mode", interactive=False)
        with gr.Column(scale=1):
            gr.Markdown("### Activity")
            cc = gr.Textbox(label="Collections", interactive=False)
            qs = gr.Textbox(label="Write Queue Size", interactive=False)
    btn.click(refresh, None, [qm, qu, tp, ro, em, es, dc, dm, cc, qs])


# ── Tab: Search ──
def build_search():
    gr.Markdown("## Search")

    async def load_collections():
        try:
            cols = await guarded_qdrant_call(get_connector().get_collection_names())
            default = get_qdrant_settings().collection_name or (cols[0] if cols else "")
            return gr.Dropdown(
                choices=[(f"Use system default: {default}", SYSTEM_DEFAULT)]
                + [(c, c) for c in cols],
                value=SYSTEM_DEFAULT,
            )
        except Exception:
            return gr.Dropdown(choices=[], value=None)

    async def do_search(collection, query, mode, top_k, emb_override):
        if not query.strip():
            return "Enter a query.", "", "", gr.Dataframe(value=[])
        try:
            qs = get_qdrant_settings()
            coll = qs.collection_name if collection == SYSTEM_DEFAULT else collection
            smode = (
                getattr(qs, "default_search_mode", None) or "dense"
                if mode == SYSTEM_DEFAULT
                else mode
            )
            em = (
                None
                if emb_override == SYSTEM_DEFAULT or not emb_override
                else emb_override
            )
            result = await search_documents(
                query=query,
                collection_name=coll,
                limit=top_k,
                mode=smode,
                embedding_model=em,
            )
            docs = result.get("results", [])
            if not docs:
                return (
                    f"No results. Mode: {result['mode']}",
                    "",
                    "",
                    gr.Dataframe(value=[]),
                )
            summary = f"{len(docs)} document(s) — mode: {result['mode']}"
            best = docs[0]
            best_info = (
                f"**{best['filename']}** (score: {best['score']:.3f})\n{best['path']}"
            )
            rows = []
            for doc in docs:
                for ch in doc.get("chunks", []):
                    rows.append(
                        [
                            doc["filename"],
                            f"{ch['score']:.3f}",
                            str(ch.get("chunk_index", "")),
                            (ch.get("content", "") or "")[:300],
                        ]
                    )
            return (
                summary,
                best_info,
                "",
                gr.Dataframe(
                    value=rows, headers=["File", "Score", "Chunk #", "Content"]
                ),
            )
        except Exception as e:
            return f"Error: {e}", "", "", gr.Dataframe(value=[])

    with gr.Row():
        coll_dd = gr.Dropdown(
            label="Collection", choices=[], value=None, interactive=True
        )
        gr.Button("Load", size="sm").click(load_collections, None, coll_dd)
    with gr.Row():
        mode_dd = gr.Dropdown(
            label="Search mode",
            choices=[
                ("Use system default", SYSTEM_DEFAULT),
                ("Dense", "dense"),
                ("Hybrid", "hybrid"),
                ("Rerank", "rerank"),
                ("Late interaction", "late_interaction"),
            ],
            value=SYSTEM_DEFAULT,
        )
        top_k = gr.Slider(
            label="Top K documents", minimum=1, maximum=50, value=10, step=1
        )
    query_input = gr.Textbox(
        label="Query", placeholder="Enter your search query...", lines=2
    )
    with gr.Accordion("Advanced", open=False):
        embed_dd = gr.Dropdown(
            label="Embedding model override",
            choices=[("Use default/collection model", SYSTEM_DEFAULT)],
            value=SYSTEM_DEFAULT,
        )
        gr.Button("Load models", size="sm").click(
            lambda: (
                gr.Dropdown(
                    choices=[("Use default/collection model", SYSTEM_DEFAULT)]
                    + [
                        (m.model_name, m.model_name)
                        for m in get_embedding_manager().list_available_models()[:30]
                    ],
                    value=SYSTEM_DEFAULT,
                )
            ),
            None,
            embed_dd,
        )
    search_btn = gr.Button("Search", variant="primary")
    gr.Markdown("---")
    summary = gr.Markdown("")
    best_info = gr.Markdown("")
    warnings = gr.Markdown("")
    results_table = gr.Dataframe(
        label="Chunks",
        headers=["File", "Score", "Chunk #", "Content"],
        interactive=False,
    )
    search_btn.click(
        do_search,
        [coll_dd, query_input, mode_dd, top_k, embed_dd],
        [summary, best_info, warnings, results_table],
    )


# ── Tab: Ingest ──
def build_ingest():
    gr.Markdown("## Ingestion")

    async def load_collections():
        try:
            cols = await guarded_qdrant_call(get_connector().get_collection_names())
            default = get_qdrant_settings().collection_name or (cols[0] if cols else "")
            return gr.Dropdown(
                choices=[(f"Use system default: {default}", SYSTEM_DEFAULT)]
                + [(c, c) for c in cols],
                value=SYSTEM_DEFAULT,
            )
        except Exception:
            return gr.Dropdown(choices=[], value=None)

    with gr.Tab("File"):
        coll_dd = gr.Dropdown(
            label="Target collection", choices=[], value=None, interactive=True
        )
        gr.Button("Load collections", size="sm").click(load_collections, None, coll_dd)
        file_in = gr.File(
            label="Upload file",
            file_types=[".txt", ".md", ".pdf", ".docx", ".json", ".csv", ".py"],
        )
        file_path = gr.Textbox(
            label="Or enter file path", placeholder="/path/to/file.pdf"
        )

        async def ingest_file(coll, uploaded, path):
            from pathlib import Path as P

            from raggy_mcp.ingest.document_id import compute_document_id
            from raggy_mcp.ingest.extractor import build_chunks, extract_text
            from raggy_mcp.ingest.macos_metadata import get_macos_metadata
            from raggy_mcp.qdrant import BatchEntry

            target = path.strip() if path else (uploaded.name if uploaded else None)
            if not target:
                return "No file provided"
            p = P(target)
            if not p.exists():
                return f"File not found: {target}"
            cname = None if coll == SYSTEM_DEFAULT else coll
            conn, wq, qs = get_connector(), get_write_queue(), get_qdrant_settings()
            try:
                meta = get_macos_metadata(str(p))
                doc = extract_text(str(p))
                if not doc.text:
                    return f"No text extracted: {doc.error or 'empty'}"
                meta.update(
                    {
                        "has_text": True,
                        "extractor_used": doc.extractor_used,
                        "char_count": doc.char_count,
                        "document_id": compute_document_id(str(p)),
                        "parent_path": str(p.parent),
                    }
                )
                await guarded_qdrant_call(conn.ensure_macos_metadata_indexes(cname))
                chunks = build_chunks(doc, meta)
                entries = [
                    BatchEntry(content=c.text, metadata=c.metadata) for c in chunks
                ]
                sparse_name = await guarded_qdrant_call(
                    conn.get_sparse_vector_name(cname)
                )
                if sparse_name:
                    from raggy_mcp.embeddings.sparse import SparseEmbeddingProvider

                    sp = SparseEmbeddingProvider(qs.sparse_model)
                    stored = await guarded_qdrant_call(
                        wq.run(
                            "ingest",
                            lambda: conn.batch_store_hybrid(entries, cname, sp),
                        )
                    )
                else:
                    stored = await guarded_qdrant_call(
                        wq.run("ingest", lambda: conn.batch_store(entries, cname))
                    )
                return f"OK Ingested {p.name}: {stored} chunks into {cname}"
            except Exception as e:
                return f"Error: {e}"

        gr.Button("Ingest File", variant="primary").click(
            ingest_file,
            [coll_dd, file_in, file_path],
            gr.Textbox(label="Result", interactive=False),
        )

    with gr.Tab("Folder"):
        fcoll_dd = gr.Dropdown(
            label="Target collection", choices=[], value=None, interactive=True
        )
        gr.Button("Load collections", size="sm").click(load_collections, None, fcoll_dd)
        fpath = gr.Textbox(label="Folder path", placeholder="/path/to/documents/")
        frec = gr.Checkbox(label="Recursive", value=True)

        async def ingest_folder(coll, path, rec):
            from pathlib import Path as P

            from raggy_mcp.ingest.document_id import compute_document_id
            from raggy_mcp.ingest.extractor import (
                SUPPORTED_EXTENSIONS,
                build_chunks,
                extract_text,
            )
            from raggy_mcp.ingest.macos_metadata import get_macos_metadata
            from raggy_mcp.qdrant import BatchEntry

            p = P(path.strip())
            if not p.is_dir():
                return "Not a valid directory"
            cname = None if coll == SYSTEM_DEFAULT else coll
            conn, wq, qs = get_connector(), get_write_queue(), get_qdrant_settings()
            pattern = "**/*" if rec else "*"
            files = [
                f
                for f in p.glob(pattern)
                if f.is_file()
                and f.suffix.lower() in SUPPORTED_EXTENSIONS
                and not any(part.startswith(".") for part in f.parts)
            ]
            if not files:
                return "No supported files found"
            await guarded_qdrant_call(conn.ensure_macos_metadata_indexes(cname))
            sn = await guarded_qdrant_call(conn.get_sparse_vector_name(cname))
            from raggy_mcp.embeddings.sparse import SparseEmbeddingProvider

            sp = SparseEmbeddingProvider(qs.sparse_model) if sn else None
            total, done, errors = 0, 0, []
            for fp in sorted(files):
                try:
                    meta = get_macos_metadata(str(fp))
                    doc = extract_text(str(fp))
                    if not doc.text:
                        errors.append(f"{fp.name}: empty")
                        continue
                    meta.update(
                        {
                            "has_text": True,
                            "extractor_used": doc.extractor_used,
                            "char_count": doc.char_count,
                            "document_id": compute_document_id(str(fp)),
                            "parent_path": str(fp.parent),
                        }
                    )
                    chunks = build_chunks(doc, meta)
                    entries = [
                        BatchEntry(content=c.text, metadata=c.metadata) for c in chunks
                    ]
                    stored = await guarded_qdrant_call(
                        wq.run(
                            "ingest_folder",
                            lambda: conn.batch_store_hybrid(entries, cname, sp)
                            if sp
                            else conn.batch_store(entries, cname),
                        )
                    )
                    total += stored
                    done += 1
                except Exception as e:
                    errors.append(f"{fp.name}: {e}")
            msg = f"OK Done: {done} files, {total} chunks into {cname}"
            if errors:
                msg += f"\n{len(errors)} errors"
            return msg

        gr.Button("Ingest Folder", variant="primary").click(
            ingest_folder,
            [fcoll_dd, fpath, frec],
            gr.Textbox(label="Result", interactive=False),
        )

    with gr.Tab("Supported Formats"):
        gr.Markdown(f"**Supported extensions:** {SUPPORTED_EXTS}")


# ── Tab: Collections ──
def build_collections():
    gr.Markdown("## Collections")

    async def list_collections():
        try:
            conn = get_connector()
            cols = await guarded_qdrant_call(conn.get_collection_names())
            rows = []
            for c in cols:
                try:
                    info = await guarded_qdrant_call(
                        conn.get_detailed_collection_info(c)
                    )
                    rows.append(
                        [
                            c,
                            str(getattr(info, "points_count", 0)),
                            f"{getattr(info, 'vector_size', '?')}D",
                            getattr(info, "distance", "?"),
                        ]
                    )
                except Exception:
                    rows.append([c, "?", "?", "?"])
            return gr.Dataframe(
                value=rows, headers=["Collection", "Points", "Vector Size", "Distance"]
            )
        except Exception:
            return gr.Dataframe(
                value=[], headers=["Collection", "Points", "Vector Size", "Distance"]
            )

    async def create_collection(name, ctype, model_name, distance):
        if not name.strip():
            return "Collection name required"
        try:
            conn = get_connector()
            mgr = get_embedding_manager()
            info = mgr.get_model_info(model_name)
            if not info:
                return f"Unknown model: {model_name}"
            provider = mgr.create_provider_for_model(model_name)
            if ctype == "hybrid":
                ok = await guarded_qdrant_call(
                    conn.create_hybrid_collection(
                        name,
                        dense_size=info.vector_size,
                        dense_vector_name=provider.get_vector_name(),
                        sparse_vector_name="sparse-bm25",
                        distance=distance,
                    )
                )
            else:
                ok = await guarded_qdrant_call(
                    conn.create_collection_with_config(
                        name, info.vector_size, distance, embedding_provider=provider
                    )
                )
            return f"OK Created: {name}" if ok else "Failed"
        except Exception as e:
            return f"Error: {e}"

    async def bootstrap(name):
        try:
            await guarded_qdrant_call(
                get_connector().ensure_macos_metadata_indexes(name)
            )
            return f"OK Indexes bootstrapped for {name}"
        except Exception as e:
            return f"Error: {e}"

    async def delete_collection(name, confirm):
        if confirm != name:
            return "Names do not match"
        try:
            ok = await guarded_qdrant_call(get_connector().delete_collection(name))
            return f"OK Deleted: {name}" if ok else "Failed"
        except Exception as e:
            return f"Error: {e}"

    with gr.Tab("List"):
        gr.Button("Refresh", variant="secondary").click(
            list_collections,
            None,
            gr.Dataframe(
                label="Collections",
                headers=["Collection", "Points", "Vector Size", "Distance"],
                interactive=False,
            ),
        )
    with gr.Tab("Create"):
        name_in = gr.Textbox(label="Collection name", placeholder="my_collection")
        ctype_dd = gr.Dropdown(
            label="Type",
            choices=[("Dense", "dense"), ("Hybrid", "hybrid")],
            value="dense",
        )
        model_dd = gr.Dropdown(label="Embedding model", choices=[], value=None)
        gr.Button("Load models", size="sm").click(
            lambda: gr.Dropdown(
                choices=[
                    m.model_name
                    for m in get_embedding_manager().list_available_models()
                ],
                value=None,
            ),
            None,
            model_dd,
        )
        dist_dd = gr.Dropdown(
            label="Distance", choices=["cosine", "dot", "euclidean"], value="cosine"
        )
        gr.Button("Create", variant="primary").click(
            create_collection,
            [name_in, ctype_dd, model_dd, dist_dd],
            gr.Textbox(label="Result", interactive=False),
        )
    with gr.Tab("Bootstrap Indexes"):

        async def load_colls():
            return gr.Dropdown(
                choices=await guarded_qdrant_call(
                    get_connector().get_collection_names()
                )
            )

        boot_dd = gr.Dropdown(
            label="Collection", choices=[], value=None, interactive=True
        )
        gr.Button("Load", size="sm").click(load_colls, None, boot_dd)
        gr.Button("Bootstrap Indexes", variant="secondary").click(
            bootstrap, boot_dd, gr.Textbox(label="Result", interactive=False)
        )
    with gr.Tab("Delete"):

        async def load_del():
            return gr.Dropdown(
                choices=await guarded_qdrant_call(
                    get_connector().get_collection_names()
                )
            )

        del_dd = gr.Dropdown(
            label="Collection to delete", choices=[], value=None, interactive=True
        )
        gr.Button("Load", size="sm").click(load_del, None, del_dd)
        del_conf = gr.Textbox(label="Type collection name to confirm")
        gr.Button("Delete", variant="stop").click(
            delete_collection,
            [del_dd, del_conf],
            gr.Textbox(label="Result", interactive=False),
        )


# ── Tab: Models ──
def build_models():
    gr.Markdown("## Models")
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Active Embedding Provider")
            mb = gr.Textbox(label="Model", value="Loading...", interactive=False)
            sb = gr.Textbox(label="Vector Size", value="", interactive=False)
            pb = gr.Textbox(label="Provider", value="", interactive=False)
            db = gr.Textbox(label="Device", value="", interactive=False)
        with gr.Column():
            gr.Markdown("### Available Models")
            mt = gr.Dataframe(
                value=[],
                headers=["Model", "Dim", "Provider", "Description"],
                interactive=False,
            )

    async def load_models():
        ep = get_embedding_provider()
        es = get_embedding_settings()
        models = get_embedding_manager().list_available_models()
        rows = [
            [
                m.model_name,
                f"{m.vector_size}D",
                m.provider_type,
                getattr(m, "description", ""),
            ]
            for m in models[:30]
        ]
        name = ep.get_model_name() if hasattr(ep, "get_model_name") else str(ep)
        return (
            name,
            f"{ep.get_vector_size()}D",
            es.provider_type.value,
            es.device,
            gr.Dataframe(value=rows),
        )

    gr.Button("Load models", variant="secondary", size="sm").click(
        load_models, None, [mb, sb, pb, db, mt]
    )
    with gr.Accordion("Sparse Retrieval", open=False):
        smd = gr.Markdown("Loading...")
        gr.Markdown(
            "Sparse vectors (BM25) enable hybrid search combining dense semantic meaning with exact keyword matching."
        )
    with gr.Accordion("Reranking", open=False):
        rmd = gr.Markdown("Loading...")
        gr.Markdown(
            "Reranking adds a cross-encoder scoring pass on top of hybrid search for better evidence quality."
        )
    with gr.Accordion("Late Interaction", open=False):
        gr.Markdown("**Default:** colbert-ir/colbertv2.0")
        gr.Markdown(
            "Late interaction (ColBERT) stores per-token vectors and uses MaxSim scoring for higher recall."
        )

    async def load_extras():
        qs = get_qdrant_settings()
        return (
            f"**Model:** {qs.sparse_model}",
            f"**Default reranker:** {qs.default_reranker_model}",
        )

    gr.Button("Load details", variant="secondary", size="sm").click(
        load_extras, None, [smd, rmd]
    )


# ── Tab: Settings ──
def build_settings():
    gr.Markdown("## Settings")
    gr.Markdown(
        "Settings shows resolved defaults. Action tabs decide what happens for a specific request."
    )
    st = gr.Dataframe(
        value=[],
        headers=["Setting", "Current value", "Source", "Used when", "Override where"],
        interactive=False,
    )

    async def load_settings():
        qs = get_qdrant_settings()
        es = get_embedding_settings()
        rows = [
            [
                "Default search mode",
                getattr(qs, "default_search_mode", "dense"),
                "Config",
                "Search tab uses 'Use system default'",
                "Search tab",
            ],
            [
                "Default collection",
                qs.collection_name or "documents",
                "Config",
                "Collection = 'Use system default'",
                "Search / Ingest",
            ],
            [
                "Embedding model",
                es.model_name,
                "Config",
                "No override specified",
                "Search > Advanced",
            ],
            ["Embedding provider", es.provider_type.value, "Config", "—", "—"],
            ["Device", es.device, "Config", "—", "—"],
            ["Sparse model", qs.sparse_model, "Config", "Hybrid/rerank modes", "—"],
            ["Reranker", qs.default_reranker_model, "Config", "Rerank mode", "—"],
            ["Chunk size", "700", "Default", "No override on Ingest tab", "Ingest tab"],
            [
                "Chunk overlap",
                "100",
                "Default",
                "No override on Ingest tab",
                "Ingest tab",
            ],
            [
                "Write concurrency",
                str(qs.write_max_concurrency),
                "Config",
                "—",
                "Admin",
            ],
            ["Write queue size", str(qs.write_queue_size), "Config", "—", "Admin"],
            ["Tool profile", qs.mcp_tool_profile, "Config", "—", "Admin"],
            ["Read-only", str(qs.read_only).lower(), "Config", "—", "Admin"],
            [
                "Qdrant mode",
                "server" if qs.location else "embedded",
                "Config",
                "—",
                "—",
            ],
        ]
        return gr.Dataframe(value=rows)

    gr.Button("Load Settings", variant="secondary", size="sm").click(
        load_settings, None, st
    )


# ── Tab: Admin ──
def build_admin():
    gr.Markdown("## Admin")
    pb = gr.Textbox(label="Tool Profile", value="Loading...", interactive=False)
    rb = gr.Textbox(label="Read-only Mode", value="", interactive=False)

    async def load_admin():
        qs = get_qdrant_settings()
        return qs.mcp_tool_profile, str(qs.read_only).lower()

    gr.Button("Load", variant="secondary", size="sm").click(load_admin, None, [pb, rb])
    gr.Markdown("### Maintenance Commands")
    gr.Markdown(
        "```bash\n./scripts/reset-server-qdrant.sh\n./scripts/local-configure-hermes.py\n```"
    )


# ── App ──
def create_app():
    with gr.Blocks(title="raggy-mcp") as app:
        gr.Markdown("# raggy-mcp")
        with gr.Tab("Dashboard"):
            build_dashboard()
        with gr.Tab("Search"):
            build_search()
        with gr.Tab("Ingest"):
            build_ingest()
        with gr.Tab("Collections"):
            build_collections()
        with gr.Tab("Models"):
            build_models()
        with gr.Tab("Settings"):
            build_settings()
        with gr.Tab("Admin"):
            build_admin()

        async def _startup():
            await init_services()

        app.load(_startup, None, None)
    return app


def main():
    parser = argparse.ArgumentParser(description="raggy-mcp WebUI Dashboard (Gradio)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--share", action="store_true", help="Create a public share link"
    )
    args = parser.parse_args()

    app = create_app()
    print(f"[raggy-webui] Dashboard: http://{args.host}:{args.port}", file=sys.stderr)
    app.queue(default_concurrency_limit=1)
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        css="footer { display: none !important; }",
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
