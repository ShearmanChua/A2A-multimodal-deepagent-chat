"""DeepAgents BackendProtocol implementation backed by an S3-compatible object store.

Virtual path layout (after CompositeBackend prefix stripping):
    /uploads/<context_id>/<filename>
        → CompositeBackend strips /uploads/ prefix
        → backend receives /<context_id>/<filename>
        → S3 key: <context_id>/<filename>

Extends BackendProtocol directly so all async wrappers (aread, awrite, als,
agrep, aglob, aedit) are inherited for free — each dispatches via
asyncio.to_thread to the sync S3 calls below.
"""

from __future__ import annotations

import fnmatch
import logging
import os
import re

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

from deepagents.backends.protocol import (
    BackendProtocol,
    EditResult,
    FileData,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GlobResult,
    GrepMatch,
    GrepResult,
    LsResult,
    ReadResult,
    WriteResult,
)

logger = logging.getLogger(__name__)


class ObjectStoreBackend(BackendProtocol):
    """Maps DeepAgents virtual filesystem calls to an S3-compatible object store.

    Only the six sync methods need to be implemented; async variants
    (aread, awrite, als, agrep, aglob, aedit) are inherited from
    BackendProtocol and run each sync call via asyncio.to_thread.
    """

    def __init__(self, bucket: str | None = None) -> None:
        self.bucket = bucket or os.environ.get("OBJECT_STORE_UPLOADS_BUCKET", "uploads")
        self._client = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_client(self):
        if self._client is not None:
            return self._client
        endpoint = os.environ.get("OBJECT_STORE_ENDPOINT", "seaweedfs:8333")
        access_key = os.environ.get("OBJECT_STORE_ACCESS_KEY", "")
        secret_key = os.environ.get("OBJECT_STORE_SECRET_KEY", "")
        secure = os.environ.get("OBJECT_STORE_SECURE", "false").lower() in ("1", "true", "yes")
        scheme = "https" if secure else "http"
        self._client = boto3.client(
            "s3",
            endpoint_url=f"{scheme}://{endpoint}",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=BotoConfig(signature_version="s3v4"),
            region_name="us-east-1",
        )
        return self._client

    def _ensure_bucket(self) -> None:
        client = self._get_client()
        try:
            client.head_bucket(Bucket=self.bucket)
        except ClientError:
            client.create_bucket(Bucket=self.bucket)

    def _vpath_to_key(self, virtual_path: str) -> str:
        """Map a virtual path to an S3 key.

        Handles both stripped paths (/<ctx>/<file>) and full paths
        (/uploads/<ctx>/<file>) robustly regardless of CompositeBackend behaviour.
        """
        path = virtual_path.lstrip("/")
        if path.startswith("uploads/"):
            path = path[len("uploads/"):]
        return path

    def _key_to_vpath(self, key: str) -> str:
        return "/uploads/" + key

    def _get_text(self, key: str) -> str | None:
        try:
            resp = self._get_client().get_object(Bucket=self.bucket, Key=key)
            return resp["Body"].read().decode("utf-8")
        except Exception:
            return None

    def _key_exists(self, key: str) -> bool:
        try:
            self._get_client().head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False

    def _list_keys(self, prefix: str) -> list[str]:
        keys: list[str] = []
        paginator = self._get_client().get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return keys

    def _put_text(self, key: str, content: str) -> None:
        self._ensure_bucket()
        self._get_client().put_object(
            Bucket=self.bucket,
            Key=key,
            Body=content.encode("utf-8"),
            ContentType="text/plain; charset=utf-8",
        )

    # ── BackendProtocol sync methods ──────────────────────────────────────────

    def ls(self, path: str) -> LsResult:
        prefix = self._vpath_to_key(path).rstrip("/")
        if prefix:
            prefix += "/"
        try:
            keys = self._list_keys(prefix)
            entries: list[FileInfo] = []
            seen_dirs: set[str] = set()
            for key in keys:
                rel = key[len(prefix):]
                if not rel:
                    continue
                parts = rel.split("/")
                if len(parts) == 1:
                    entries.append({"path": self._key_to_vpath(key), "is_dir": False})
                else:
                    dir_key = prefix + parts[0]
                    if dir_key not in seen_dirs:
                        seen_dirs.add(dir_key)
                        entries.append({"path": self._key_to_vpath(dir_key) + "/", "is_dir": True})
            entries.sort(key=lambda e: e.get("path", ""))
            return LsResult(entries=entries)
        except Exception as exc:
            return LsResult(error=str(exc), entries=[])

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        key = self._vpath_to_key(file_path)
        text = self._get_text(key)
        if text is None:
            return ReadResult(error=f"File not found: {file_path}")
        lines = text.splitlines(keepends=True)
        if offset >= len(lines):
            return ReadResult(error=f"Line offset {offset} exceeds file length ({len(lines)} lines)")
        sliced = lines[offset: offset + limit]
        file_data: FileData = {"content": "".join(sliced), "encoding": "utf-8"}
        return ReadResult(file_data=file_data)

    def write(self, file_path: str, content: str) -> WriteResult:
        key = self._vpath_to_key(file_path)
        if self._key_exists(key):
            return WriteResult(
                error=f"Cannot write to {file_path} because it already exists. "
                      "Read and then make an edit, or write to a new path."
            )
        try:
            self._put_text(key, content)
            logger.info("ObjectStoreBackend.write s3://%s/%s (%d chars)", self.bucket, key, len(content))
            return WriteResult(path=file_path)
        except Exception as exc:
            logger.error("ObjectStoreBackend.write failed for %s: %s", file_path, exc)
            return WriteResult(error=str(exc))

    def edit(
        self, file_path: str, old_string: str, new_string: str, replace_all: bool = False
    ) -> EditResult:
        key = self._vpath_to_key(file_path)
        text = self._get_text(key)
        if text is None:
            return EditResult(error=f"Error: File '{file_path}' not found")
        old_string = old_string.replace("\r\n", "\n").replace("\r", "\n")
        new_string = new_string.replace("\r\n", "\n").replace("\r", "\n")
        count = text.count(old_string)
        if count == 0:
            return EditResult(error="old_string not found in file")
        if not replace_all and count > 1:
            return EditResult(
                error=f"old_string is not unique ({count} occurrences); pass replace_all=True"
            )
        updated = text.replace(old_string, new_string) if replace_all else text.replace(old_string, new_string, 1)
        try:
            self._put_text(key, updated)
            return EditResult(path=file_path, occurrences=count if replace_all else 1)
        except Exception as exc:
            return EditResult(error=str(exc))

    def grep(self, pattern: str, path: str | None = None, glob: str | None = None) -> GrepResult:
        prefix = self._vpath_to_key(path or "/")
        try:
            keys = self._list_keys(prefix)
        except Exception as exc:
            return GrepResult(error=str(exc), matches=[])
        if glob:
            keys = [k for k in keys if fnmatch.fnmatch(k.split("/")[-1], glob)]
        try:
            regex = re.compile(re.escape(pattern))
        except re.error as exc:
            return GrepResult(error=f"Invalid pattern: {exc}", matches=[])
        matches: list[GrepMatch] = []
        for key in keys:
            text = self._get_text(key)
            if not text:
                continue
            for lineno, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    matches.append({"path": self._key_to_vpath(key), "line": lineno, "text": line})
        return GrepResult(matches=matches)

    def glob(self, pattern: str, path: str = "/") -> GlobResult:
        prefix = self._vpath_to_key(path)
        try:
            keys = self._list_keys(prefix)
        except Exception as exc:
            return GlobResult(error=str(exc), matches=[])
        matches: list[FileInfo] = []
        for key in keys:
            vpath = self._key_to_vpath(key)
            filename = key.split("/")[-1]
            pat = pattern.lstrip("*/")
            if fnmatch.fnmatch(vpath, pattern) or fnmatch.fnmatch(filename, pat):
                matches.append({"path": vpath, "is_dir": False})
        matches.sort(key=lambda e: e.get("path", ""))
        return GlobResult(matches=matches)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        responses: list[FileUploadResponse] = []
        for path, content in files:
            key = self._vpath_to_key(path)
            try:
                self._put_text(key, content.decode("utf-8", errors="replace"))
                responses.append(FileUploadResponse(path=path))
            except Exception as exc:
                responses.append(FileUploadResponse(path=path, error="invalid_path"))
                logger.error("upload_files failed for %s: %s", path, exc)
        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        responses: list[FileDownloadResponse] = []
        for path in paths:
            key = self._vpath_to_key(path)
            try:
                resp = self._get_client().get_object(Bucket=self.bucket, Key=key)
                content = resp["Body"].read()
                responses.append(FileDownloadResponse(path=path, content=content))
            except ClientError:
                responses.append(FileDownloadResponse(path=path, content=None, error="file_not_found"))
            except Exception as exc:
                logger.error("download_files failed for %s: %s", path, exc)
                responses.append(FileDownloadResponse(path=path, content=None, error="invalid_path"))
        return responses
