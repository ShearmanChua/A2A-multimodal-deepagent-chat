"""FastAPI service for document ingestion: convert → chunk → embed → upload to Weaviate."""
import asyncio
import logging
import tempfile
import time
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .chunker import chunk_markdown
from .converter import convert_to_markdown
from .embedder import get_embedder
from .seaweedfs_client import delete_all_objects, get_presigned_url_from_path, seaweedfs_configured, upload_document_image
from .weaviate_client import delete_all_collections, list_collections, sanitize_collection_name, search_collection, upload_chunks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("docling-ingestion")

app = FastAPI(title="Docling Ingestion Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ACCEPTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".md"}

# In-memory job store: job_id → job dict
_jobs: dict[str, dict] = {}


def _make_job(job_id: str, collection: str, filenames: list[str]) -> dict:
    return {
        "id": job_id,
        "collection": collection,
        "status": "running",
        "created_at": time.time(),
        "completed_at": None,
        "files": [
            {"name": n, "status": "pending", "chunks": 0, "images": 0, "error": None}
            for n in filenames
        ],
    }


async def _process_job(
    job_id: str, files_data: list[tuple[str, bytes]], collection: str
) -> None:
    job = _jobs[job_id]
    upload_fn = upload_document_image if seaweedfs_configured() else None
    loop = asyncio.get_event_loop()
    any_error = False

    for name, content in files_data:
        suffix = Path(name).suffix.lower()
        entry = next((f for f in job["files"] if f["name"] == name), None)
        if entry is None:
            continue

        if suffix not in ACCEPTED_EXTENSIONS:
            entry["status"] = "error"
            entry["error"] = (
                f"Unsupported type '{suffix}'. "
                f"Accepted: {', '.join(sorted(ACCEPTED_EXTENSIONS))}"
            )
            any_error = True
            continue

        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(content)
                tmp_path = Path(tmp.name)

            entry["status"] = "converting"
            markdown, image_urls = await loop.run_in_executor(
                None, convert_to_markdown, tmp_path, suffix, upload_fn
            )
            entry["images"] = len(image_urls)

            entry["status"] = "chunking"
            chunks = chunk_markdown(markdown, source_file=name)

            entry["status"] = "embedding"
            embedder = get_embedder()
            texts = [c["content"] for c in chunks]
            vectors = await loop.run_in_executor(
                None,
                lambda t=texts: embedder.encode(t, show_progress_bar=False).tolist(),
            )

            entry["status"] = "uploading"
            await loop.run_in_executor(None, upload_chunks, chunks, vectors, collection)

            entry["status"] = "done"
            entry["chunks"] = len(chunks)

        except Exception as exc:
            logger.exception("Failed to process '%s'", name)
            entry["status"] = "error"
            entry["error"] = str(exc)
            any_error = True
        finally:
            if tmp_path and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    job["status"] = "error" if any_error else "done"
    job["completed_at"] = time.time()
    logger.info("Job %s finished with status '%s'", job_id, job["status"])


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
@app.get("/ingest/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Collections
# ---------------------------------------------------------------------------

@app.get("/ingest/collections")
def get_collections():
    return {"collections": list_collections()}


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

@app.get("/ingest/search")
async def search_documents(
    collection: str = Query(...),
    query: str = Query(...),
    limit: int = Query(default=10, ge=1, le=50),
    alpha: float = Query(default=0.5, ge=0.0, le=1.0),
):
    loop = asyncio.get_event_loop()
    embedder = get_embedder()
    vector = await loop.run_in_executor(
        None, lambda: embedder.encode(query, show_progress_bar=False).tolist()
    )
    results = await loop.run_in_executor(
        None, search_collection, collection, vector, query, limit, alpha
    )
    # Convert stored seaweedfs:// paths to fresh presigned URLs for the frontend
    if seaweedfs_configured():
        for chunk in results:
            props = chunk.get("properties", {})
            raw_images = props.get("images") or []
            resolved = []
            for ref in raw_images:
                if isinstance(ref, str) and ref.startswith("seaweedfs://"):
                    try:
                        resolved.append(get_presigned_url_from_path(ref))
                    except Exception:
                        resolved.append(ref)
                else:
                    resolved.append(ref)
            props["images"] = resolved
    return {"collection": collection, "query": query, "count": len(results), "results": results}


# ---------------------------------------------------------------------------
# Ingestion — background jobs
# ---------------------------------------------------------------------------

@app.post("/ingest")
async def ingest_documents(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    collection: str = Form(default="Documents"),
):
    """Start a background ingestion job. Returns a job_id immediately."""
    collection = sanitize_collection_name(collection)

    # Read file contents now — UploadFile cannot be read inside a background task.
    files_data = [
        (f.filename or f"file_{i}", await f.read())
        for i, f in enumerate(files)
    ]

    job_id = str(uuid.uuid4())
    filenames = [name for name, _ in files_data]
    _jobs[job_id] = _make_job(job_id, collection, filenames)
    background_tasks.add_task(_process_job, job_id, files_data, collection)

    logger.info("Started job %s for collection '%s' (%d file(s))", job_id, collection, len(files_data))
    return {"job_id": job_id}


@app.get("/ingest/jobs")
def list_jobs():
    """List all jobs, most recent first."""
    jobs = sorted(_jobs.values(), key=lambda j: j["created_at"], reverse=True)
    return {"jobs": jobs}


@app.get("/ingest/jobs/{job_id}")
def get_job(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.delete("/ingest/jobs/{job_id}")
def delete_job(job_id: str):
    _jobs.pop(job_id, None)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

@app.post("/ingest/reset")
async def reset_all():
    """Delete every Weaviate collection and every SeaweedFS object in the bucket."""
    loop = asyncio.get_event_loop()

    try:
        deleted_collections = await loop.run_in_executor(None, delete_all_collections)
    except Exception as exc:
        logger.exception("Error deleting Weaviate collections")
        deleted_collections = []
        weaviate_error = str(exc)
    else:
        weaviate_error = None

    try:
        deleted_objects = await loop.run_in_executor(None, delete_all_objects)
    except Exception as exc:
        logger.exception("Error deleting SeaweedFS objects")
        deleted_objects = 0
        seaweedfs_error = str(exc)
    else:
        seaweedfs_error = None

    # Also clear the in-memory job store.
    _jobs.clear()

    logger.info(
        "Reset complete: %d collection(s) dropped, %d object(s) deleted",
        len(deleted_collections), deleted_objects,
    )
    return {
        "collections_deleted": deleted_collections,
        "objects_deleted": deleted_objects,
        "weaviate_error": weaviate_error,
        "seaweedfs_error": seaweedfs_error,
    }
