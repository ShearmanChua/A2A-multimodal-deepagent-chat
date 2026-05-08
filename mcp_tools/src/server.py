from fastmcp import FastMCP
from tools.weaviate_store import (
    list_collections as weaviate_list_collections,
    get_collection_schema as weaviate_get_collection_schema,
    hybrid_query as weaviate_hybrid_query,
)

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
)

logger = logging.getLogger("mcp-rag")

# Create MCP server
mcp = FastMCP("rag-server")

# ---------------------------------------------------------------------------
# Weaviate vector-database tools
# ---------------------------------------------------------------------------

@mcp.tool(
    name="list_weaviate_collections",
    description=(
        "List all collections (classes) stored in the Weaviate "
        "vector database. Returns each collection's name, "
        "description, property count, and property names. "
        "Use this to discover what data is available before querying."
    )
)
def list_weaviate_collections() -> list[dict]:
    """
    List every collection in the Weaviate instance.

    Returns:
        A list of dicts with name, description, property_count,
        and properties for each collection.
    """
    results = weaviate_list_collections()
    logger.info("Weaviate: listed %s collection(s).", len(results))
    return results


@mcp.tool(
    name="get_weaviate_collection_schema",
    description=(
        "Get the full schema (properties, data types, "
        "vectorizer config) of a specific Weaviate collection. "
        "Use this to understand the structure of a collection before querying it."
    )
)
def get_weaviate_collection_schema(collection_name: str) -> dict:
    """
    Retrieve the schema of a Weaviate collection.

    Args:
        collection_name: The name of the collection to inspect.

    Returns:
        A dict with name, description, properties (with data types), and vectorizer info.
    """
    schema = weaviate_get_collection_schema(collection_name)
    logger.info("Weaviate: retrieved schema for '%s'.", collection_name)
    return schema


@mcp.tool(
    name="query_weaviate",
    description=(
        "Search a Weaviate collection using hybrid search "
        "(combines BM25 keyword matching with vector similarity). "
        "The query string is automatically embedded into a vector "
        "using the configured embedding model. Adjust 'alpha' to "
        "control the balance: 0.0 = pure keyword, 1.0 = pure vector, "
        "0.5 = equal mix (default). Use list_weaviate_collections "
        "first to discover available collections, and "
        "get_weaviate_collection_schema to see queryable properties."
    )
)
def query_weaviate(
    collection_name: str,
    query: str,
    limit: int = 10,
    alpha: float = 0.5,
    properties: list[str] | None = None,
) -> list[dict]:
    """
    Perform a hybrid search on a Weaviate collection.

    Args:
        collection_name: The collection to search.
        query: Natural-language query string (will be embedded automatically).
        limit: Maximum number of results (default 10).
        alpha: Balance between keyword (0.0) and vector (1.0) search (default 0.5).
        properties: Optional list of property names to return. If omitted,
        all properties are returned.

    Returns:
        A list of result dicts with properties, score, explain_score, and uuid.
    """
    results = weaviate_hybrid_query(
        collection_name=collection_name,
        query=query,
        limit=limit,
        alpha=alpha,
        properties=properties,
    )
    logger.info(
        "Weaviate: hybrid query on '%s' returned %s result(s).",
        collection_name,
        len(results)
    )
    return results


if __name__ == "__main__":
    # Start MCP server
    mcp.run(transport="http", host="0.0.0.0", port=8000)
