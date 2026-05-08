"""
Weaviate vector-database tool for the MCP server.

Provides three capabilities:
1. **list_collections** – list all collections (classes) in the Weaviate instance.
2. **get_collection_schema** – retrieve the full schema of a specific collection.
3. **hybrid_query** – perform a hybrid (keyword + vector) search against a
   collection.  The query string is embedded locally using the model specified
   by the ``EMBEDDING_MODEL`` environment variable (default
   ``all-mpnet-base-v2``), so no external embedding service is required at
   query time.

Configuration is read from environment variables:
    WEAVIATE_HOST      – hostname (default ``weaviate``)
    WEAVIATE_HTTP_PORT – HTTP API port (default ``8081``)
    WEAVIATE_GRPC_PORT – gRPC port (default ``50051``)
    EMBEDDING_MODEL    – sentence-transformers model name (default
                         ``all-mpnet-base-v2``)
"""

import logging
import os
from typing import Any

import weaviate
from weaviate.classes.query import MetadataQuery
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)

logger = logging.getLogger("mcp-weaviate-store")

# ---------------------------------------------------------------------------
# Weaviate client (lazy singleton)
# ---------------------------------------------------------------------------
_client: weaviate.WeaviateClient | None = None


def _get_client() -> weaviate.WeaviateClient:
    """Return (and lazily create) the Weaviate client singleton."""
    global _client
    if _client is not None and _client.is_connected():
        return _client

    host = os.environ.get("WEAVIATE_HOST", "weaviate")
    http_port = int(os.environ.get("WEAVIATE_HTTP_PORT", "8081"))
    grpc_port = int(os.environ.get("WEAVIATE_GRPC_PORT", "50051"))

    logger.info(
        "Connecting to Weaviate at %s (http=%d, grpc=%d) …",
        host,
        http_port,
        grpc_port,
    )

    _client = weaviate.connect_to_custom(
        http_host=host,
        http_port=http_port,
        http_secure=False,
        grpc_host=host,
        grpc_port=grpc_port,
        grpc_secure=False,
    )
    return _client


# ---------------------------------------------------------------------------
# Embedding model (lazy singleton)
# ---------------------------------------------------------------------------
_embedder: SentenceTransformer | None = None


def _get_embedder() -> SentenceTransformer:
    """Return (and lazily load) the sentence-transformers embedding model."""
    global _embedder
    if _embedder is not None:
        return _embedder

    model_name = os.environ.get("EMBEDDING_MODEL", "all-mpnet-base-v2")
    logger.info("Loading embedding model '%s' …", model_name)
    _embedder = SentenceTransformer(model_name)
    logger.info("Embedding model '%s' loaded.", model_name)
    return _embedder


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_collections() -> list[dict[str, Any]]:
    """
    List all collections (classes) in the Weaviate instance.

    Returns
    -------
    list[dict[str, Any]]
        A list of dicts, each containing:
        - ``name`` – collection name
        - ``description`` – collection description (may be empty)
        - ``property_count`` – number of properties defined
        - ``properties`` – list of property names
    """
    client = _get_client()

    collections_response = client.collections.list_all()

    results: list[dict[str, Any]] = []
    for name, collection_config in collections_response.items():
        props = collection_config.properties
        results.append({
            "name": name,
            "description": collection_config.description or "",
            "property_count": len(props),
            "properties": [p.name for p in props],
        })

    logger.info("Listed %d collection(s) in Weaviate.", len(results))
    return results


def get_collection_schema(collection_name: str) -> dict[str, Any]:
    """
    Retrieve the full schema of a specific Weaviate collection.

    Parameters
    ----------
    collection_name : str
        The name of the collection (class) to inspect.

    Returns
    -------
    dict[str, Any]
        A dict containing:
        - ``name`` – collection name
        - ``description`` – collection description
        - ``properties`` – list of property dicts with ``name``,
          ``data_type``, and ``description``
        - ``vectorizer`` – the configured vectorizer module name
    """
    client = _get_client()

    collection = client.collections.get(collection_name)
    config = collection.config.get()

    properties: list[dict[str, Any]] = []
    for prop in config.properties:
        properties.append({
            "name": prop.name,
            "data_type": str(prop.data_type),
            "description": prop.description or "",
        })

    vectorizer_name = ""
    if config.vectorizer_config:
        vectorizer_name = str(config.vectorizer_config.vectorizer)

    schema: dict[str, Any] = {
        "name": config.name,
        "description": config.description or "",
        "properties": properties,
        "vectorizer": vectorizer_name,
    }

    logger.info(
        "Retrieved schema for collection '%s' (%d properties).",
        collection_name,
        len(properties),
    )
    return schema


def hybrid_query(
    collection_name: str,
    query: str,
    limit: int = 10,
    alpha: float = 0.5,
    properties: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Perform a hybrid (keyword + vector) search on a Weaviate collection.

    The *query* string is embedded locally using the sentence-transformers
    model specified by ``EMBEDDING_MODEL`` (default ``all-mpnet-base-v2``).
    The resulting vector is combined with BM25 keyword search via Weaviate's
    hybrid query API.

    Parameters
    ----------
    collection_name : str
        The target collection to search.
    query : str
        Natural-language query string.
    limit : int
        Maximum number of results to return (default 10).
    alpha : float
        Weighting between keyword (0.0) and vector (1.0) search.
        ``0.5`` gives equal weight to both (default).
    properties : list[str] | None
        Optional list of property names to return.  If ``None``, all
        properties are returned.

    Returns
    -------
    list[dict[str, Any]]
        A list of result dicts, each containing:
        - ``properties`` – the object's stored properties
        - ``score`` – hybrid relevance score
        - ``explain_score`` – score explanation string
        - ``uuid`` – the object's UUID
    """
    client = _get_client()
    embedder = _get_embedder()

    # Embed the query locally
    query_vector = embedder.encode(query).tolist()

    collection = client.collections.get(collection_name)

    # Build the hybrid query
    if properties:
        response = collection.query.hybrid(
            query=query,
            vector=query_vector,
            limit=limit,
            alpha=alpha,
            return_metadata=MetadataQuery(score=True, explain_score=True),
            return_properties=properties,
        )
    else:
        response = collection.query.hybrid(
            query=query,
            vector=query_vector,
            limit=limit,
            alpha=alpha,
            return_metadata=MetadataQuery(score=True, explain_score=True),
        )

    results: list[dict[str, Any]] = []
    for obj in response.objects:
        result: dict[str, Any] = {
            "properties": {k: _serialise_value(v) for k, v in obj.properties.items()},
            "uuid": str(obj.uuid),
        }
        if obj.metadata:
            result["score"] = obj.metadata.score
            result["explain_score"] = obj.metadata.explain_score
        results.append(result)

    logger.info(
        "Hybrid query on '%s' returned %d result(s) (query='%s').",
        collection_name,
        len(results),
        query[:80],
    )
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _serialise_value(value: Any) -> Any:
    """Make a property value JSON-serialisable."""
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value
