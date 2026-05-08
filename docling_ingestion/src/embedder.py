import logging
import os

from sentence_transformers import SentenceTransformer

logger = logging.getLogger("docling-ingestion.embedder")

_embedder: SentenceTransformer | None = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is not None:
        return _embedder
    model = os.environ.get("EMBEDDING_MODEL", "all-mpnet-base-v2")
    logger.info("Loading embedding model '%s'...", model)
    _embedder = SentenceTransformer(model)
    logger.info("Embedding model '%s' loaded.", model)
    return _embedder
