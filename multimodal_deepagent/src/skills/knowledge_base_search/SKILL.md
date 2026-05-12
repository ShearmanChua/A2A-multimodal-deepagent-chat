---
name: knowledge-base-search
description: Search the Weaviate knowledge base to answer factual questions from ingested documents. Use this skill for any question that may be answered by documents already in the knowledge base before attempting other approaches.
---

# Knowledge Base Search

Answer the user's question by searching the Weaviate knowledge base.

## Steps

1. Call `list_weaviate_collections` to discover available collections.
2. Choose the most relevant collection(s) for the query. When unsure, search multiple.
3. Call `query_weaviate` with the user's question as the `query` argument.
   - Default `alpha=0.5` balances keyword and semantic search equally.
   - Raise `alpha` toward 1.0 for conceptual questions; lower it toward 0.0 for exact term lookups.
   - Increase `limit` up to 20 if initial results lack sufficient context.
4. Synthesise the retrieved chunks into a clear, accurate answer.
   - Cite the `source_file` and `header_path` from each result's properties.
   - Quote or paraphrase directly from `content` when precise wording matters.
   - If images are present in a chunk's `images` list, retrieve and analyse them:
     - Call `get_object_store_image_base64` with the `objstore://` path to get the image for vision analysis.
     - Or call `get_object_store_presigned_url` to obtain a URL you can reference in your response.
5. If no relevant results are found, say so clearly and suggest the user ingest the relevant documents first.

## Tool Reference

| Tool | Arguments | Returns |
|------|-----------|---------|
| `list_weaviate_collections` | — | `[{name, description, properties}]` |
| `get_weaviate_collection_schema` | `collection_name` | `{name, description, properties, vectorizer}` |
| `query_weaviate` | `collection_name, query, limit=10, alpha=0.5` | `[{properties, score, uuid}]` |
| `get_object_store_image_base64` | `path` (objstore:// path) | Image content block |
| `get_object_store_presigned_url` | `path` (objstore:// path), `expiry` (optional seconds) | presigned URL string |

## Result Properties

Each result's `properties` dict includes:

- `content` — the document chunk text
- `source_file` — original filename
- `header_path` — document section breadcrumb (e.g. `"Chapter 1 > Overview"`)
- `chunk_index` — position of this chunk within the source file
- `file_type` — original file format (pdf, docx, txt, …)
- `images` — list of SeaweedFS image URLs embedded in this chunk
