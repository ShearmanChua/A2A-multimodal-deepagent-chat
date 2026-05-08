import { useState } from "react";
import {
  Trash2,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Loader2,
  Database,
  HardDrive,
} from "lucide-react";

export default function Settings() {
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [resetting, setResetting]     = useState(false);
  const [result, setResult]           = useState(null); // null | {ok, summary, error}

  const handleReset = async () => {
    setResetting(true);
    setConfirmOpen(false);
    setResult(null);

    try {
      const resp = await fetch("/ingest/reset", { method: "POST" });
      const data = await resp.json();

      if (!resp.ok) {
        setResult({ ok: false, error: data.detail || `Server error ${resp.status}` });
        return;
      }

      setResult({
        ok: true,
        collections: data.collections_deleted ?? [],
        objects: data.objects_deleted ?? 0,
        weaviateError: data.weaviate_error ?? null,
        seaweedfsError: data.seaweedfs_error ?? null,
      });
    } catch (err) {
      setResult({ ok: false, error: err.message });
    } finally {
      setResetting(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto px-6 py-8 space-y-8">
      <div>
        <h2 className="text-xl font-semibold text-gray-100">Settings</h2>
        <p className="text-sm text-gray-400 mt-1">Manage application data and storage.</p>
      </div>

      {/* ── Reset card ── */}
      <div className="rounded-xl border border-red-800/50 bg-red-900/10 p-6 space-y-4">
        <div className="flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
          <div>
            <h3 className="text-base font-semibold text-red-300">Reset All Data</h3>
            <p className="text-sm text-gray-400 mt-1">
              Permanently deletes all data from the knowledge base. This cannot be undone.
            </p>
          </div>
        </div>

        <ul className="space-y-2 pl-8">
          <li className="flex items-center gap-2 text-sm text-gray-400">
            <Database className="w-4 h-4 text-gray-500 shrink-0" />
            All Weaviate collections and their document chunks
          </li>
          <li className="flex items-center gap-2 text-sm text-gray-400">
            <HardDrive className="w-4 h-4 text-gray-500 shrink-0" />
            All images stored in SeaweedFS
          </li>
        </ul>

        <button
          onClick={() => setConfirmOpen(true)}
          disabled={resetting}
          className="flex items-center gap-2 px-4 py-2 bg-red-600 hover:bg-red-500
            disabled:opacity-50 disabled:cursor-not-allowed rounded-lg text-sm font-medium
            text-white transition-colors"
        >
          {resetting
            ? <Loader2 className="w-4 h-4 animate-spin" />
            : <Trash2 className="w-4 h-4" />}
          {resetting ? "Resetting…" : "Reset Everything"}
        </button>

        {/* Result banner */}
        {result && <ResultBanner result={result} onDismiss={() => setResult(null)} />}
      </div>

      {/* ── Confirmation modal ── */}
      {confirmOpen && (
        <ConfirmModal
          onConfirm={handleReset}
          onCancel={() => setConfirmOpen(false)}
        />
      )}
    </div>
  );
}

// ─── Confirmation modal ───────────────────────────────────────────────────────

function ConfirmModal({ onConfirm, onCancel }) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="w-full max-w-md mx-4 bg-gray-800 rounded-2xl border border-gray-700 shadow-2xl p-6 space-y-5">
        <div className="flex items-start gap-3">
          <div className="p-2 bg-red-900/40 rounded-lg shrink-0">
            <AlertTriangle className="w-5 h-5 text-red-400" />
          </div>
          <div>
            <h3 className="text-base font-semibold text-gray-100">Delete all data?</h3>
            <p className="text-sm text-gray-400 mt-1">
              This will permanently erase every document chunk in Weaviate and
              every file in SeaweedFS. There is no undo.
            </p>
          </div>
        </div>

        <div className="flex gap-3 justify-end">
          <button
            onClick={onCancel}
            className="px-4 py-2 text-sm text-gray-300 hover:text-white bg-gray-700
              hover:bg-gray-600 rounded-lg transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-white
              bg-red-600 hover:bg-red-500 rounded-lg transition-colors"
          >
            <Trash2 className="w-4 h-4" />
            Yes, delete everything
          </button>
        </div>
      </div>
    </div>
  );
}

// ─── Result banner ────────────────────────────────────────────────────────────

function ResultBanner({ result, onDismiss }) {
  if (!result.ok) {
    return (
      <div className="flex items-start gap-3 p-4 bg-red-900/30 border border-red-700/50 rounded-lg">
        <XCircle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
        <div className="flex-1 text-sm text-red-300">{result.error}</div>
        <button onClick={onDismiss} className="text-gray-500 hover:text-gray-300 shrink-0">
          <XCircle className="w-4 h-4" />
        </button>
      </div>
    );
  }

  return (
    <div className="p-4 bg-green-900/20 border border-green-700/50 rounded-lg space-y-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-sm font-medium text-green-400">
          <CheckCircle className="w-4 h-4" />
          Reset complete
        </div>
        <button onClick={onDismiss} className="text-gray-500 hover:text-gray-300">
          <XCircle className="w-4 h-4" />
        </button>
      </div>
      <ul className="space-y-1 text-sm text-gray-400 pl-6">
        <li>
          <Database className="inline w-3.5 h-3.5 mr-1.5 text-gray-500" />
          {result.collections.length > 0
            ? `${result.collections.length} collection(s) deleted: ${result.collections.join(", ")}`
            : "No Weaviate collections found"}
          {result.weaviateError && (
            <span className="ml-2 text-yellow-400">(warning: {result.weaviateError})</span>
          )}
        </li>
        <li>
          <HardDrive className="inline w-3.5 h-3.5 mr-1.5 text-gray-500" />
          {result.objects} SeaweedFS object(s) deleted
          {result.seaweedfsError && (
            <span className="ml-2 text-yellow-400">(warning: {result.seaweedfsError})</span>
          )}
        </li>
      </ul>
    </div>
  );
}
