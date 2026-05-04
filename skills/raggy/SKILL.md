# raggy

## When to Use

Use this skill when the user wants to search, inspect, add to, refresh, create, or safely delete data in a `raggy-mcp` vector store.

Use it for requests like:

- “Search my docs for…”
- “Find the passage about…”
- “Give me a brief from the indexed docs.”
- “Which collection should I use?”
- “Search across the hybrid and late-interaction collections.”
- “Add this file/folder to the vector store.”
- “Re-index these docs.”
- “Why didn’t search find this?”
- “Create a new collection.”
- “Delete this collection.”
- “Try again, that didn’t answer it.”

Do **not** use this skill for unrelated web search, ordinary file editing, or questions that do not involve `raggy-mcp`, Qdrant collections, local RAG, document search, ingestion, retrieval, or retrieval troubleshooting.

## Procedure

### 1. Start with an interactive action menu when appropriate

If the user is in an interactive terminal, asks what the agent can do, or has not yet provided a concrete question, show this menu:

```text
What would you like to do?

1. Search documents — balanced default search
2. Deep search — multi-query + broader context
3. Precise source search — exact names, terms, quotes, filenames, IDs
4. Synthesis search — broad evaluative answer with reranking
5. Cross-mode search — compare hybrid, rerank, and late-interaction
6. Add content — ingest a file or folder
7. Update or re-index content — refresh changed files/folders
8. Inspect collections — list collections, schema, counts, indexed fields
9. Create a collection — dense, hybrid, or late-interaction
10. Delete a collection — explicit confirmation required
11. Troubleshoot retrieval — diagnose missing or bad results

Press Enter for default document search.
>
```

If the user types a natural-language question directly, treat it as option 1 and run the default search. Do not force the user through the menu first.

### 2. Infer the collection before asking

Before searching, ingesting, updating, inspecting, or troubleshooting, infer the target collection when possible.

Use these clues:

1. Collection explicitly named by the user.
2. Current session collection.
3. Default collection from config or environment, such as `COLLECTION_NAME`.
4. The only available collection from `list_collections`.
5. Collection name matching the user’s project, folder, file, or domain.
6. Collection schema compatible with the requested mode.
7. Recent terminal/session context.

If there is exactly one collection, use it and say:

```text
I found one collection, `<collection>`, so I’ll use that.
```

If there are multiple plausible collections and no clear best match, ask:

```text
Which collection should I use?

1. documents
2. notes
3. codebase
4. research

Press Enter to use the configured default, if available.
>
```

If one collection is strongly implied, use it and state the inference:

```text
I’ll use `<collection>` because it matches the project you mentioned.
```

For destructive actions, never rely on inference alone. Require explicit confirmation of the exact collection name.

### 3. Understand collection families

`raggy-mcp` can expose different collection types. The agent should understand that collections are not interchangeable.

Common patterns:

- **Hybrid collection**: dense + sparse vectors. Best general-purpose RAG collection.
- **Dense collection**: semantic vectors only. Useful for broad conceptual similarity.
- **Late-interaction collection**: ColBERT-style multivectors. Best for high-recall, token-level concept/name matching when the collection supports it.

If multiple collections appear to contain the same corpus in different schemas, such as:

```text
socratic_circles_hybrid_v2
default_li
```

then treat them as complementary retrieval views over similar material, not as duplicates. Use the hybrid collection for `hybrid` and `rerank`; use the late-interaction collection only for `late_interaction`.

### 4. Choose mode by question type, not speed

Do not choose modes mainly by speed. Choose the mode by retrieval fit.

Use:

- `hybrid` for the default search and most normal questions.
- `dense` for broad conceptual questions where exact wording may not appear.
- `hybrid` for exact terminology, filenames, commands, config variables, API names, IDs, or domain jargon.
- `rerank` for broad synthesis, evaluative answers, central-claim extraction, noisy results, user dissatisfaction, or final evidence selection.
- `late_interaction` for exact conceptual source-finding when names, frameworks, labels, phrases, or specialized terms matter and the selected collection supports late-interaction vectors.
- `cross-mode` when the user wants the best possible brief or when initial results disagree.

Refined mode guidance from observed results:

| User intent | Best first mode | Why |
|---|---|---|
| “What is the overall claim?” | `rerank` | Better at ranking central passages for broad synthesis. |
| “Find the source passage about Adler / Paideia / Three Columns / maieutic” | `late_interaction` | Better at token-level matching of exact conceptual vocabulary. |
| “Find config/API/tool names” | `hybrid` or `late_interaction` | Exact terms matter. |
| “Give me a brief with confidence” | `cross-mode` | Compare whether modes surface the same evidence. |
| “This answer missed the source text” | `late_interaction` with targeted keywords | Often better for exact framework exposition. |
| “This answer found facts but not the main point” | `rerank` | Often better for central claims and synthesis. |

### 5. Run a strong default search first

For normal questions, run a document-level grouped search first.

Recommended default:

```json
{
  "tool": "search_documents",
  "arguments": {
    "query": "<user question>",
    "collection": "<inferred hybrid collection>",
    "mode": "hybrid",
    "limit": 8,
    "chunks_per_document": 8
  }
}
```

If the actual tool schema uses different argument names, preserve the intent:

- document-level search
- inferred collection
- hybrid mode
- about 8 documents
- about 8 chunks per document

After answering, ask:

```text
Did this answer it, or should I keep digging?

Options:
- more chunks from these documents
- more documents
- rerank for central claims
- late-interaction for exact source passages
- multi-query expansion
- exact/metadata search
- troubleshoot collection or ingestion
```

### 6. Use cross-mode search for briefs and high-stakes answers

When the user asks for a brief, a careful answer, or a comparison, run a cross-mode strategy when compatible collections exist.

Recommended cross-mode strategy:

1. Run `rerank` on the hybrid collection for broad synthesis and central claims.
2. Run `late_interaction` on the late-interaction collection for exact concept/name/source passages.
3. Optionally run `hybrid` as the baseline candidate finder.
4. Compare top chunks and identify:
   - evidence both modes found
   - evidence only rerank found
   - evidence only late-interaction found
   - mode disagreement
   - likely reason for disagreement

Example:

```json
{
  "rerank_search": {
    "collection": "<hybrid collection>",
    "mode": "rerank",
    "limit": 15,
    "chunks_per_document": 12
  },
  "late_interaction_search": {
    "collection": "<late_interaction collection>",
    "mode": "late_interaction",
    "limit": 15,
    "chunks_per_document": 12
  }
}
```

Use this especially when the query contains proper names, frameworks, canonical phrases, or technical vocabulary **and** also asks for a synthesized conclusion.

### 7. Use deep search for complex or unsatisfied queries

Use deep search when:

- The user asks for completeness.
- The question is technical, ambiguous, or multi-part.
- The first search fails.
- The user says the answer is missing, wrong, shallow, or incomplete.
- Recall matters.

Recommended deep search:

```json
{
  "mode": "hybrid",
  "limit": 15,
  "chunks_per_document": 12
}
```

For very large-context review:

```json
{
  "mode": "rerank",
  "limit": 20,
  "chunks_per_document": 15
}
```

For exact framework/source retrieval on late-interaction collections:

```json
{
  "mode": "late_interaction",
  "limit": 20,
  "chunks_per_document": 15
}
```

Do not dump all retrieved text to the user. Use the extra context to synthesize a better answer.

### 8. Escalate proactively when the user is unhappy

Important rule:

> If the user is unhappy or did not find the answer, increase chunks before giving up.

Use this ladder:

#### Step 1: More chunks from likely documents

Best when results look relevant but incomplete.

```json
{
  "mode": "<same mode as prior search>",
  "limit": 8,
  "chunks_per_document": 15
}
```

Say:

```text
The first documents look relevant, but we may need more surrounding context. I’ll pull more chunks from the likely matches.
```

#### Step 2: More documents

Best when the first result set is too narrow.

```json
{
  "mode": "<best-fit mode>",
  "limit": 18,
  "chunks_per_document": 10
}
```

Say:

```text
I’ll widen the document set so we can catch answers that live in a different file or section.
```

#### Step 3: Switch mode based on what failed

If the answer missed exact source text, names, frameworks, or key terms:

```json
{
  "mode": "late_interaction",
  "limit": 18,
  "chunks_per_document": 12
}
```

If the answer found related facts but missed the central claim:

```json
{
  "mode": "rerank",
  "limit": 18,
  "chunks_per_document": 12
}
```

If exact terms, filenames, config, commands, or IDs matter:

```json
{
  "mode": "hybrid",
  "limit": 18,
  "chunks_per_document": 12
}
```

#### Step 4: Multi-query expansion

Generate 4–8 intentional query variants:

1. Literal wording.
2. Conceptual phrasing.
3. Goal-oriented phrasing.
4. Implementation/API/config phrasing.
5. Metadata/file/path phrasing.
6. Failure-mode phrasing.
7. Broader parent-topic phrasing.
8. HyDE-style likely-answer phrasing, when useful.

#### Step 5: Cross-mode comparison

When available, compare rerank against late-interaction and explain which mode produced better evidence for the user’s actual intent.

#### Step 6: Troubleshoot collection setup

Run collection and server discovery tools if repeated retrieval attempts fail.

### 9. Generate multi-query searches intentionally

Classify the user’s intent first:

- How-to: steps or workflow.
- Debugging: why something failed or is missing.
- Reference: exact API, tool, config, command, or behavior.
- Conceptual: explanation or meaning.
- Comparative: differences among modes, settings, files, or approaches.
- Navigational: find a file, command, endpoint, section, or document.
- Completeness: find everything related to a topic.
- Ingestion/update: add, refresh, re-index, or delete data.
- Source-finding: find the underlying canonical passage or quotation.
- Synthesis: combine evidence into a brief, summary, recommendation, or argument.

Then generate distinct query variants:

```text
Literal:
<original user wording>

Concept:
<main concept> <related concepts>

Goal:
how to <user goal>

Implementation:
<tool names> <function names> <config variables> <endpoints>

Metadata/path:
<filename> <folder> <document type> <source> <date>

Named-source:
<proper names> <framework names> <canonical vocabulary>

Failure mode:
missing stale wrong no results unsupported mismatch fallback warning

Broader scope:
<parent topic> <adjacent terminology>

HyDE-style:
A document explaining that <likely answer shape>
```

Do not create near-duplicates. Each query should test a different hypothesis about how the answer may be written.

After all searches:

1. Merge results.
2. Deduplicate by `document_id`.
3. Deduplicate near-identical chunks.
4. Prefer documents retrieved by multiple query intents.
5. Keep unique evidence that answers part of the question.
6. Explain which query strategy worked best.
7. State remaining uncertainty.

### 10. Use targeted keyword enrichment for late-interaction

Late-interaction is especially useful when the question contains or implies named concepts.

When using `late_interaction`, enrich the query with precise vocabulary from the user’s intent.

Examples:

- User asks about Adler’s framework:
  - add terms like `Adler`, `Paideia`, `Three Columns`, `Acquisition of Knowledge`, `Development of Skill`, `Enlargement of Understanding`, `maieutic`.
- User asks about a code/config concept:
  - add tool names, config variables, endpoint names, class/function names.
- User asks for a source passage:
  - add likely quote terms, section headings, author names, and framework labels.

Avoid keyword stuffing. Add only terms that are plausible and helpful.

### 11. Use metadata filtering when scope is known

Use metadata filters when the user provides:

- filename
- path
- parent folder
- document type
- source
- date or ingest time
- project
- page count or document metadata, if indexed

If unsure whether a field exists, call `get_indexed_fields`.

If a metadata-filtered search fails, retry without the filter and explain that the filter may have been too narrow or not indexed.

### 12. Add content to the vector store

Use this flow when the user wants to add files, folders, notes, or text.

1. Infer or ask for target collection.
2. If the collection does not exist, offer to create one.
3. If file support is uncertain, call `get_supported_extractors`.
4. For one file, use `ingest_file`.
5. For a folder, first use `ingest_folder` with `run_mode="report"`.
6. Show the report: target collection, file count, supported files, skipped files, and warnings.
7. Ask before applying folder-wide ingest.
8. Apply with `run_mode="apply"` only after user confirmation.

Supported common formats include `.txt`, `.md`, `.json`, `.jsonl`, `.csv`, `.tsv`, `.pdf`, `.docx`, and many code/config text formats.

For interactive terminal wording:

```text
I can add that to the vector store.

Target collection: <collection>
Source: <file-or-folder>

For folders, I’ll run a report first so you can review what would be indexed before anything changes.
```

### 13. Update or re-index content

Use this flow when the user says content changed, search is stale, or they want to refresh indexed files.

1. Infer collection and source path.
2. Inspect collection info if needed.
3. Re-run `ingest_file` or `ingest_folder`.
4. For folders, prefer `run_mode="report"` before `run_mode="apply"`.
5. Warn that changing embedding models or chunking strategy usually requires re-ingestion.

If the collection’s embedding model changed or appears mismatched, recommend re-ingesting into a new collection instead of mutating the existing one casually.

### 14. Create collections

Use this flow when the user wants a new vector collection.

Recommend:

- Hybrid collection for general-purpose RAG.
- Dense collection for semantic-only use.
- Late-interaction collection when the user wants strong concept/name/source retrieval and will ingest accordingly.

For serious search quality, consider creating paired collections for the same corpus:

- one hybrid collection for `hybrid` and `rerank`
- one late-interaction collection for `late_interaction`

Flow:

1. Ask or infer collection name.
2. Ask collection type if unclear.
3. Choose embedding model or use configured default.
4. Create collection.
5. Bootstrap metadata indexes when appropriate.
6. Offer to ingest content.

### 15. Delete collections safely

Deletion is destructive. Never infer silently.

Flow:

1. Ask the user to confirm the exact collection name.
2. Run `delete_collection` with `run_mode="report"`.
3. Show the returned plan.
4. Ask for explicit confirmation.
5. Only then run `delete_collection` with `run_mode="apply"` and the returned `plan_id`.

Never delete based on a vague “yes” if multiple collections exist.

### 16. Troubleshoot retrieval

Use this when the user says search is bad, incomplete, stale, irrelevant, or missing expected files.

Check:

1. Correct collection selected?
2. Collection has points?
3. Collection schema supports the requested search mode?
4. Hybrid and late-interaction collections accidentally mixed up?
5. Content actually ingested?
6. Filters too restrictive?
7. Metadata indexes available?
8. File type supported?
9. Extraction produced enough text?
10. Embedding model changed after ingestion?
11. Server read-only?
12. Needed tools hidden by current tool profile?
13. More chunks, more documents, rerank, late-interaction, or multi-query needed?

Recommended tools:

1. `list_collections`
2. `get_collection_info`
3. `get_collection_schema`
4. `get_indexed_fields`
5. `list_search_modes`
6. `get_server_capabilities`
7. `list_embedding_models`
8. `get_supported_extractors`

### 17. Present results clearly

Always include:

1. Direct answer.
2. Evidence summary.
3. Source document metadata, if available.
4. Collection searched.
5. Mode used.
6. Retrieval scope.
7. If cross-mode was used, what each mode contributed.
8. Confidence.
9. Concrete next action if the user wants to continue.

Example:

```text
Answer:
The best evidence points to two related conclusions: Copeland’s own claim is about transforming how students read, think, discuss, write, and act, while Adler’s framework gives the theoretical language of knowledge, skill, and enlarged understanding.

Evidence:
- Rerank surfaced the central Copeland passages about student transformation and the limits of content-knowledge teaching.
- Late-interaction surfaced the exact Adlerian framework passages, including Three Columns and maieutic teaching.

Searched:
- rerank: socratic_circles_hybrid_v2, 15 docs, 12 chunks/doc
- late_interaction: default_li, 15 docs, 12 chunks/doc

Confidence: high.

Next:
I can pull more chunks around the Adler passages or run a precise quote search.
```

## Pitfalls

- Do not ask for a collection when there is only one obvious collection.
- Do not guess for destructive operations.
- Do not use a late-interaction collection for rerank or a hybrid collection for late-interaction.
- Do not stop after a weak or failed search.
- Do not keep the chunk budget too small for large-context models.
- Do not choose search mode primarily by speed.
- Do not assume rerank is always better. It may rank central claims well but miss exact source-framework exposition.
- Do not assume late-interaction is always better. It may find exact named concepts well but may not select the central synthesis passage.
- Do not use metadata filters before verifying fields when uncertain.
- Do not assume missing results mean missing knowledge; retrieval may need more chunks, multi-query expansion, mode switching, cross-mode comparison, or re-ingestion.
- Do not mutate data when the server is read-only.
- Do not switch embedding models on an existing collection casually.
- Do not delete collections without report/apply safety and exact-name confirmation.
- Do not dump huge retrieved context into the chat; synthesize it.

## Verification

Before ending a session or giving a final answer, verify:

- The skill was triggered only for `raggy-mcp`, RAG, Qdrant, vector search, ingestion, or retrieval-management work.
- The collection was inferred or selected appropriately.
- If many collections were plausible, the user was asked to choose.
- Mode matched intent:
  - `hybrid` for default and exact general search.
  - `rerank` for central claims and synthesis.
  - `late_interaction` for exact names, frameworks, and source passages.
  - cross-mode for briefs or important answers.
- A strong default search used about 8 documents and 8 chunks per document.
- If the user was dissatisfied, chunks were increased before giving up.
- Multi-query search used distinct query intents, not near-duplicates.
- Late-interaction queries were enriched with targeted vocabulary when appropriate.
- Results were merged and deduplicated by document.
- Cross-mode results identified what each mode contributed.
- Add/update operations used the right ingest tool.
- Folder ingestion used report before apply when possible.
- Destructive operations required exact-name confirmation and report/apply safety.
- The final answer included evidence, collection, mode, retrieval scope, confidence, and a next action.
