import { useState, useRef, useCallback, useEffect } from "react";
import {
  Upload,
  FileText,
  X,
  CheckCircle,
  AlertCircle,
  Loader2,
  Database,
  RefreshCw,
  Image,
  Clock,
  Trash2,
} from "lucide-react";

const ACCEPTED_EXTS = [".pdf", ".docx", ".doc", ".txt", ".md"];
const ACCEPTED_MIME = [
  "application/pdf",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  "application/msword",
  "text/plain",
  "text/markdown",
];

const STEPS = ["converting", "chunking", "embedding", "uploading"];
const STEP_LABELS = { converting: "Convert", chunking: "Chunk", embedding: "Embed", uploading: "Upload" };
const NEW_COLLECTION_SENTINEL = "__new__";

export default function DocumentIngestion({ jobs = [], onJobStarted }) {
  const [pendingFiles, setPendingFiles]   = useState([]);
  const [submitting, setSubmitting]       = useState(false);
  const [isDragging, setIsDragging]       = useState(false);
  const [submitError, setSubmitError]     = useState(null);

  // Collection state
  const [collections, setCollections]             = useState([]);
  const [collectionsLoading, setCollectionsLoading] = useState(false);
  const [selectedCollection, setSelectedCollection] = useState("Documents");
  const [newCollectionName, setNewCollectionName]   = useState("");

  const fileInputRef = useRef(null);

  const effectiveCollection =
    selectedCollection === NEW_COLLECTION_SENTINEL
      ? newCollectionName.trim()
      : selectedCollection;

  const fetchCollections = useCallback(async () => {
    setCollectionsLoading(true);
    try {
      const resp = await fetch("/ingest/collections");
      if (resp.ok) {
        const data = await resp.json();
        setCollections(data.collections || []);
      }
    } catch { /* service may not be up yet */ } finally {
      setCollectionsLoading(false);
    }
  }, []);

  useEffect(() => { fetchCollections(); }, [fetchCollections]);

  // Refresh collection list when a job finishes so new collections appear.
  const prevJobsRef = useRef(jobs);
  useEffect(() => {
    const prev = prevJobsRef.current;
    const justFinished = jobs.some(
      (j) => j.status !== "running" &&
             prev.find((p) => p.id === j.id)?.status === "running"
    );
    if (justFinished) fetchCollections();
    prevJobsRef.current = jobs;
  }, [jobs, fetchCollections]);

  const addFiles = useCallback((fileList) => {
    const incoming = Array.from(fileList).filter((f) => {
      const ext = "." + f.name.split(".").pop().toLowerCase();
      return ACCEPTED_EXTS.includes(ext);
    });
    setPendingFiles((prev) => {
      const existing = new Set(prev.map((f) => f.name));
      return [...prev, ...incoming.filter((f) => !existing.has(f.name))];
    });
  }, []);

  const removeFile = (name) =>
    setPendingFiles((prev) => prev.filter((f) => f.name !== name));

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
    addFiles(e.dataTransfer.files);
  }, [addFiles]);

  const handleIngest = async () => {
    if (!pendingFiles.length || submitting || !effectiveCollection) return;
    setSubmitting(true);
    setSubmitError(null);

    const form = new FormData();
    pendingFiles.forEach((f) => form.append("files", f));
    form.append("collection", effectiveCollection);

    try {
      const resp = await fetch("/ingest", { method: "POST", body: form });
      if (!resp.ok) throw new Error(`Server error ${resp.status}`);
      await onJobStarted?.();
      setPendingFiles([]);
    } catch (err) {
      setSubmitError(err.message);
    } finally {
      setSubmitting(false);
    }
  };

  const handleDeleteJob = async (jobId) => {
    await fetch(`/ingest/jobs/${jobId}`, { method: "DELETE" });
    await onJobStarted?.(); // triggers a jobs refresh in App
  };

  const canIngest = pendingFiles.length > 0 && !submitting && !!effectiveCollection;

  return (
    <div className="max-w-3xl mx-auto px-6 py-8 space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-gray-100">Document Ingestion</h2>
        <p className="text-sm text-gray-400 mt-1">
          Convert, chunk, embed, and store documents in the Weaviate knowledge base.
          Jobs run in the background — you can switch tabs while they process.
        </p>
      </div>

      {/* ── Collection selector ── */}
      <div className="space-y-2">
        <label className="text-sm font-medium text-gray-300">Target collection</label>
        <div className="flex items-center gap-2">
          <select
            value={selectedCollection}
            onChange={(e) => setSelectedCollection(e.target.value)}
            disabled={submitting}
            className="flex-1 bg-gray-800 border border-gray-600 rounded-lg px-3 py-2 text-sm
              text-gray-200 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
          >
            {collections.map((c) => (
              <option key={c.name} value={c.name}>{c.name}</option>
            ))}
            <option value={NEW_COLLECTION_SENTINEL}>+ Create new collection…</option>
            {collections.length === 0 && <option value="Documents">Documents</option>}
          </select>
          <button
            onClick={fetchCollections}
            disabled={collectionsLoading || submitting}
            title="Refresh collections"
            className="p-2 text-gray-400 hover:text-gray-200 disabled:opacity-40 transition-colors"
          >
            <RefreshCw className={`w-4 h-4 ${collectionsLoading ? "animate-spin" : ""}`} />
          </button>
        </div>

        {selectedCollection === NEW_COLLECTION_SENTINEL && (
          <input
            type="text"
            value={newCollectionName}
            onChange={(e) => setNewCollectionName(e.target.value)}
            placeholder="Collection name (e.g. TechnicalDocs)"
            disabled={submitting}
            className="w-full bg-gray-800 border border-gray-600 rounded-lg px-3 py-2 text-sm
              text-gray-200 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        )}
      </div>

      {/* ── Drop zone ── */}
      <div
        onDrop={handleDrop}
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
        onDragLeave={() => setIsDragging(false)}
        onClick={() => fileInputRef.current?.click()}
        className={`border-2 border-dashed rounded-xl p-10 flex flex-col items-center gap-3
          cursor-pointer transition-colors select-none
          ${isDragging
            ? "border-blue-500 bg-blue-500/10"
            : "border-gray-600 hover:border-gray-500 bg-gray-800/40"}`}
      >
        <Upload className={`w-10 h-10 ${isDragging ? "text-blue-400" : "text-gray-500"}`} />
        <div className="text-center">
          <p className="text-gray-200 font-medium">Drop files here or click to browse</p>
          <p className="text-xs text-gray-500 mt-1">Accepted: {ACCEPTED_EXTS.join(", ")}</p>
        </div>
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept={[...ACCEPTED_EXTS, ...ACCEPTED_MIME].join(",")}
          className="hidden"
          onChange={(e) => addFiles(e.target.files)}
        />
      </div>

      {/* ── Pending file list ── */}
      {pendingFiles.length > 0 && (
        <div className="space-y-2">
          {pendingFiles.map((file) => {
            const sizeStr = file.size > 1024 * 1024
              ? `${(file.size / (1024 * 1024)).toFixed(1)} MB`
              : `${(file.size / 1024).toFixed(0)} KB`;
            return (
              <div key={file.name} className="flex items-center gap-3 bg-gray-800 rounded-lg px-4 py-3">
                <FileText className="w-5 h-5 text-gray-400 shrink-0" />
                <span className="flex-1 text-sm text-gray-200 truncate">{file.name}</span>
                <span className="text-xs text-gray-500 shrink-0">{sizeStr}</span>
                {!submitting && (
                  <button
                    onClick={() => removeFile(file.name)}
                    className="text-gray-600 hover:text-gray-400 transition-colors shrink-0"
                  >
                    <X className="w-4 h-4" />
                  </button>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* ── Submit ── */}
      {pendingFiles.length > 0 && (
        <div className="flex items-center gap-4 flex-wrap">
          <button
            onClick={handleIngest}
            disabled={!canIngest}
            className="flex items-center gap-2 px-5 py-2.5 bg-blue-600 hover:bg-blue-500
              disabled:opacity-50 disabled:cursor-not-allowed rounded-lg text-sm font-medium transition-colors"
          >
            {submitting
              ? <Loader2 className="w-4 h-4 animate-spin" />
              : <Database className="w-4 h-4" />}
            {submitting ? "Queueing…" : `Ingest into "${effectiveCollection || "…"}"`}
          </button>
          {!submitting && (
            <button
              onClick={() => setPendingFiles([])}
              className="px-4 py-2.5 text-sm text-gray-400 hover:text-gray-200 transition-colors"
            >
              Clear all
            </button>
          )}
          {submitError && (
            <p className="text-sm text-red-400">{submitError}</p>
          )}
        </div>
      )}

      {/* ── Job history ── */}
      {jobs.length > 0 && (
        <div className="space-y-3 pt-2">
          <h3 className="text-sm font-medium text-gray-400 uppercase tracking-wide">
            Ingestion Jobs
          </h3>
          {jobs.map((job) => (
            <JobCard key={job.id} job={job} onDelete={() => handleDeleteJob(job.id)} />
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Job card ─────────────────────────────────────────────────────────────────

function JobCard({ job, onDelete }) {
  const [expanded, setExpanded] = useState(job.status === "running");

  // Auto-expand when a job transitions to running.
  useEffect(() => {
    if (job.status === "running") setExpanded(true);
  }, [job.status]);

  const doneFiles  = job.files.filter((f) => f.status === "done").length;
  const errorFiles = job.files.filter((f) => f.status === "error").length;
  const totalFiles = job.files.length;
  const isRunning  = job.status === "running";

  const elapsed = job.completed_at
    ? `${Math.round(job.completed_at - job.created_at)}s`
    : null;

  return (
    <div className={`rounded-xl border ${
      isRunning              ? "border-blue-700/50 bg-blue-900/10" :
      job.status === "error" ? "border-red-800/50 bg-red-900/10"  :
                               "border-gray-700/50 bg-gray-800"
    }`}>
      {/* Card header */}
      <button
        onClick={() => setExpanded((v) => !v)}
        className="w-full flex items-center gap-3 px-4 py-3 text-left"
      >
        {isRunning
          ? <Loader2 className="w-4 h-4 text-blue-400 animate-spin shrink-0" />
          : job.status === "error"
            ? <AlertCircle className="w-4 h-4 text-red-400 shrink-0" />
            : <CheckCircle className="w-4 h-4 text-green-400 shrink-0" />
        }

        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-gray-200 truncate">
            {job.collection}
          </p>
          <p className="text-xs text-gray-500">
            {totalFiles} file{totalFiles !== 1 ? "s" : ""}
            {isRunning
              ? ` · ${doneFiles}/${totalFiles} done`
              : errorFiles > 0
                ? ` · ${errorFiles} error${errorFiles !== 1 ? "s" : ""}`
                : ` · ${job.files.reduce((s, f) => s + (f.chunks || 0), 0)} chunks`
            }
            {elapsed && ` · ${elapsed}`}
          </p>
        </div>

        <div className="flex items-center gap-1 shrink-0">
          {!isRunning && (
            <button
              onClick={(e) => { e.stopPropagation(); onDelete(); }}
              className="p-1 text-gray-600 hover:text-gray-400 transition-colors"
              title="Remove"
            >
              <Trash2 className="w-3.5 h-3.5" />
            </button>
          )}
          <Clock className="w-3 h-3 text-gray-600" />
          <span className="text-xs text-gray-600">
            {new Date(job.created_at * 1000).toLocaleTimeString()}
          </span>
        </div>
      </button>

      {/* Expanded file list */}
      {expanded && (
        <div className="px-4 pb-3 space-y-2 border-t border-gray-700/50 pt-3">
          {job.files.map((file) => (
            <FileProgressRow key={file.name} file={file} />
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Per-file progress row ────────────────────────────────────────────────────

function FileProgressRow({ file }) {
  const activeIdx = STEPS.indexOf(file.status);
  const isDone    = file.status === "done";
  const isError   = file.status === "error";
  const isPending = file.status === "pending";

  return (
    <div className="flex items-start gap-3 bg-gray-800/60 rounded-lg px-3 py-2.5">
      <FileText className="w-4 h-4 text-gray-400 shrink-0 mt-0.5" />

      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-1.5">
          <span className="text-xs text-gray-300 truncate">{file.name}</span>
          {isDone && (
            <span className="text-xs text-green-400 shrink-0">{file.chunks} chunks</span>
          )}
          {isDone && file.images > 0 && (
            <span className="text-xs text-blue-400 flex items-center gap-1 shrink-0">
              <Image className="w-3 h-3" /> {file.images}
            </span>
          )}
        </div>

        {isError && (
          <p className="text-xs text-red-400 line-clamp-2">{file.error}</p>
        )}

        {!isDone && !isError && (
          <>
            <div className="flex gap-1 mb-1">
              {STEPS.map((s, idx) => (
                <div
                  key={s}
                  className={`h-1 flex-1 rounded-full transition-all duration-300 ${
                    isPending        ? "bg-gray-700" :
                    idx < activeIdx  ? "bg-blue-500" :
                    idx === activeIdx ? "bg-blue-400 animate-pulse" :
                    "bg-gray-700"
                  }`}
                />
              ))}
            </div>
            <div className="flex gap-1">
              {STEPS.map((s, idx) => (
                <p key={s} className={`text-xs flex-1 text-center ${
                  idx === activeIdx ? "text-blue-400" : "text-gray-600"
                }`}>
                  {STEP_LABELS[s]}
                </p>
              ))}
            </div>
          </>
        )}
      </div>

      <div className="shrink-0 mt-0.5">
        {isDone    && <CheckCircle className="w-4 h-4 text-green-400" />}
        {isError   && <AlertCircle className="w-4 h-4 text-red-400"   />}
        {!isDone && !isError && !isPending && (
          <Loader2 className="w-4 h-4 text-blue-400 animate-spin" />
        )}
      </div>
    </div>
  );
}
