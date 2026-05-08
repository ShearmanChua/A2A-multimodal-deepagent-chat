"""Weaviate client for creating collections, uploading chunks, and hybrid search."""
import logging
import os
import re
import uuid as uuid_lib
from datetime import datetime, timezone

import weaviate
import weaviate.classes as wvc
from weaviate.classes.query import MetadataQuery

logger = logging.getLogger("docling-ingestion.weaviate")

_client: weaviate.WeaviateClient | None = None


def sanitize_collection_name(name: str) -> str:
    """Convert an arbitrary string to a valid Weaviate class name (PascalCase, alphanumeric).

    Examples: 'us-army-doctrines' → 'UsArmyDoctrines'
              'my collection'     → 'MyCollection'
              '123docs'           → 'C123docs'
    """
    words = re.split(r"[^a-zA-Z0-9]+", name.strip())
    pascal = "".join(w.capitalize() for w in words if w)
    if not pascal:
        return "Documents"
    # Weaviate requires the first character to be a letter.
    if pascal[0].isdigit():
        pascal = "C" + pascal
    return pascal


def _get_client() -> weaviate.WeaviateClient:
    global _client
    if _client is not None and _client.is_connected():
        return _client

    host = os.environ.get("WEAVIATE_HOST", "weaviate")
    http_port = int(os.environ.get("WEAVIATE_HTTP_PORT", "8081"))
    grpc_port = int(os.environ.get("WEAVIATE_GRPC_PORT", "50051"))

    logger.info("Connecting to Weaviate at %s (http=%d, grpc=%d)", host, http_port, grpc_port)
    _client = weaviate.connect_to_custom(
        http_host=host,
        http_port=http_port,
        http_secure=False,
        grpc_host=host,
        grpc_port=grpc_port,
        grpc_secure=False,
    )
    return _client


def ensure_collection(collection_name: str) -> None:
    collection_name = sanitize_collection_name(collection_name)
    client = _get_client()
    if client.collections.exists(collection_name):
        return

    logger.info("Creating Weaviate collection '%s'", collection_name)
    client.collections.create(
        name=collection_name,
        description="Ingested document chunks for RAG",
        properties=[
            wvc.config.Property(name="content",     data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="source_file", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="header_path", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="chunk_index", data_type=wvc.config.DataType.INT),
            wvc.config.Property(name="file_type",   data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="ingested_at", data_type=wvc.config.DataType.DATE),
            wvc.config.Property(name="images",      data_type=wvc.config.DataType.TEXT_ARRAY),
        ],
        vectorizer_config=wvc.config.Configure.Vectorizer.none(),
    )


def list_collections() -> list[dict]:
    """Return a list of existing collection names and property counts."""
    client = _get_client()
    result = []
    for name, cfg in client.collections.list_all().items():
        result.append({
            "name": name,
            "property_count": len(cfg.properties),
            "properties": [p.name for p in cfg.properties],
        })
    return sorted(result, key=lambda c: c["name"])


def search_collection(
    collection_name: str,
    query_vector: list[float],
    query_text: str,
    limit: int = 10,
    alpha: float = 0.5,
) -> list[dict]:
    """Hybrid BM25 + vector search. Returns results ordered by relevance score."""
    collection_name = sanitize_collection_name(collection_name)
    client = _get_client()
    if not client.collections.exists(collection_name):
        return []

    collection = client.collections.get(collection_name)
    response = collection.query.hybrid(
        query=query_text,
        vector=query_vector,
        limit=limit,
        alpha=alpha,
        return_metadata=MetadataQuery(score=True, explain_score=True),
    )

    results = []
    for obj in response.objects:
        props = {}
        for k, v in obj.properties.items():
            props[k] = v.isoformat() if isinstance(v, datetime) else v
        results.append({
            "uuid": str(obj.uuid),
            "score": obj.metadata.score if obj.metadata else None,
            "properties": props,
        })
    return results


def delete_all_collections() -> list[str]:
    """Drop every collection in Weaviate. Returns the names of deleted collections."""
    client = _get_client()
    names = list(client.collections.list_all().keys())
    for name in names:
        client.collections.delete(name)
        logger.info("Deleted collection '%s'", name)
    return names


def upload_chunks(
    chunks: list[dict],
    vectors: list[list[float]],
    collection_name: str,
) -> None:
    collection_name = sanitize_collection_name(collection_name)
    ensure_collection(collection_name)
    client = _get_client()
    collection = client.collections.get(collection_name)
    now = datetime.now(timezone.utc)

    objects = [
        wvc.data.DataObject(
            properties={
                "content":     c["content"],
                "source_file": c["source_file"],
                "header_path": c["header_path"],
                "chunk_index": c["chunk_index"],
                "file_type":   c["file_type"],
                "ingested_at": now,
                "images":      c.get("images") or [],
            },
            vector=v,
            uuid=uuid_lib.uuid4(),
        )
        for c, v in zip(chunks, vectors)
    ]

    resp = collection.data.insert_many(objects)
    if resp.errors:
        for uid, err in resp.errors.items():
            logger.error("Insert error uuid=%s: %s", uid, err.message)

    inserted = len(objects) - len(resp.errors)
    logger.info("Uploaded %d/%d chunks to '%s'", inserted, len(objects), collection_name)
