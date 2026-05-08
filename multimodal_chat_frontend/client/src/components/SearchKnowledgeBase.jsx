import { useState, useRef, useCallback, useEffect } from "react";
import {
  Search,
  RefreshCw,
  FileText,
  ChevronDown,
  ChevronUp,
  Image,
  ExternalLink,
} from "lucide-react";

// ─── helpers ────────────────────────────────────────────────────────────────

function scoreColor(score) {
  if (score == null) return "text-gray-500";
  if (score >= 0.6)  return "text-green-400";
  if (score >= 0.3)  return "text-yellow-400";
  return "text-red-400";
}

// ─── main component ──────────────────────────────────────────────────────────

export default function SearchKnowledgeBase() {
  const [collections, setCollections]           = useState([]);
  const [collectionsLoading, setCollLoading]    = useState(false);
  const [selectedCollection, setSelected]       = useState("");
  const [query, setQuery]                       = useState("");
  const [alpha, setAlpha]                       = useState(0.5);
  const [limit, setLimit]                       = useState(10);
  const [showAdvanced, setShowAdvanced]         = useState(false);
  const [results, setResults]                   = useState(null); // null = not yet searched
  const [loading, setLoading]                   = useState(false);
  const [error, setError]                       = useState(null);
  const [lastQuery, setLastQuery]               = useState("");

  const inputRef = useRef(null);

  // ── fetch collections ────────────────────────────────────────────────────
  const fetchCollections = useCallback(async () => {
    setCollLoading(true);
    try {
      const resp = await fetch("/ingest/collections");
      if (resp.ok) {
        const data = await resp.json();
        const cols = data.collections || [];
        setCollections(cols);
        if (cols.length > 0) {
          setSelected((prev) => prev || cols[0].name);
        }
      }
    } catch { /* service may not be running yet */ } finally {
      setCollLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchCollections();
    inputRef.current?.focus();
  }, [fetchCollections]);

  // ── search ───────────────────────────────────────────────────────────────
  const handleSearch = useCallback(async () => {
    const q = query.trim();
    if (!q || !selectedCollection || loading) return;

    setLoading(true);
    setError(null);
    setResults(null);
    setLastQuery(q);

    try {
      const params = new URLSearchParams({
        collection: selectedCollection,
        query: q,
        limit: String(limit),
        alpha: String(alpha),
      });
      const resp = await fetch(`/ingest/search?${params}`);
      if (!resp.ok) throw new Error(`Search failed (${resp.status})`);
      const data = await resp.json();
      setResults(data.results || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [query, selectedCollection, limit, alpha, loading]);

  const handleKeyDown = (e) => { if (e.key === "Enter") handleSearch(); };

  // ── render ───────────────────────────────────────────────────────────────
  return (
    <div className="max-w-4xl mx-auto px-6 py-8 space-y-6">
      {/* title */}
      <div>
        <h2 className="text-xl font-semibold text-gray-100">Knowledge Base Search</h2>
        <p className="text-sm text-gray-400 mt-1">
          Hybrid keyword + semantic search across ingested document chunks.
        </p>
      </div>

      {/* ── controls ── */}
      <div className="space-y-3">

        {/* collection picker + search bar */}
        <div className="flex gap-3">
          {/* collection dropdown */}
          <div className="flex items-center gap-1 shrink-0">
            <select
              value={selectedCollection}
              onChange={(e) => setSelected(e.target.value)}
              disabled={loading}
              className="bg-gray-800 border border-gray-600 rounded-lg px-3 py-2.5 text-sm
                text-gray-200 focus:outline-none focus:ring-2 focus:ring-blue-500
                disabled:opacity-50 max-w-[11rem]"
            >
              {collections.length === 0 && (
                <option value="">No collections</option>
              )}
              {collections.map((c) => (
                <option key={c.name} value={c.name}>{c.name}</option>
              ))}
            </select>
            <button
              onClick={fetchCollections}
              disabled={collectionsLoading || loading}
              title="Refresh collections"
              className="p-2 text-gray-500 hover:text-gray-300 disabled:opacity-40 transition-colors"
            >
              <RefreshCw className={`w-4 h-4 ${collectionsLoading ? "animate-spin" : ""}`} />
            </button>
          </div>

          {/* query input */}
          <div className="flex flex-1 gap-2">
            <input
              ref={inputRef}
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Search your knowledge base…"
              disabled={loading}
              className="flex-1 bg-gray-800 border border-gray-600 rounded-lg px-4 py-2.5 text-sm
                text-gray-200 placeholder-gray-500 focus:outline-none focus:ring-2
                focus:ring-blue-500 disabled:opacity-50"
            />
            <button
              onClick={handleSearch}
              disabled={!query.trim() || !selectedCollection || loading}
              className="flex items-center gap-2 px-5 py-2.5 bg-blue-600 hover:bg-blue-500
                disabled:opacity-50 disabled:cursor-not-allowed rounded-lg text-sm
                font-medium transition-colors shrink-0"
            >
              {loading
                ? <RefreshCw className="w-4 h-4 animate-spin" />
                : <Search className="w-4 h-4" />}
              Search
            </button>
          </div>
        </div>

        {/* advanced options toggle */}
        <div>
          <button
            onClick={() => setShowAdvanced((v) => !v)}
            className="flex items-center gap-1 text-xs text-gray-500 hover:text-gray-300 transition-colors"
          >
            {showAdvanced ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
            Advanced options
          </button>

          {showAdvanced && (
            <div className="mt-3 flex flex-wrap items-end gap-8 p-4 bg-gray-800/50 rounded-lg">
              {/* alpha slider */}
              <div className="space-y-1">
                <div className="flex items-center justify-between gap-4">
                  <span className="text-xs text-gray-400">Search mode</span>
                  <span className="text-xs font-mono text-blue-400">{alpha.toFixed(2)}</span>
                </div>
                <input
                  type="range" min="0" max="1" step="0.05"
                  value={alpha}
                  onChange={(e) => setAlpha(parseFloat(e.target.value))}
                  className="w-52 accent-blue-500"
                />
                <div className="flex justify-between text-xs text-gray-600 w-52">
                  <span>Keyword</span>
                  <span>Semantic</span>
                </div>
              </div>

              {/* result limit */}
              <div className="space-y-1">
                <label className="text-xs text-gray-400 block">Max results</label>
                <select
                  value={limit}
                  onChange={(e) => setLimit(parseInt(e.target.value))}
                  className="bg-gray-800 border border-gray-700 rounded px-2 py-1.5
                    text-sm text-gray-200 focus:outline-none focus:ring-1 focus:ring-blue-500"
                >
                  {[5, 10, 20, 50].map((n) => (
                    <option key={n} value={n}>{n}</option>
                  ))}
                </select>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* ── error ── */}
      {error && (
        <div className="p-4 bg-red-900/20 border border-red-800 rounded-lg text-sm text-red-400">
          {error}
        </div>
      )}

      {/* ── empty state ── */}
      {results !== null && results.length === 0 && (
        <div className="text-center py-16 text-gray-500">
          <Search className="w-10 h-10 mx-auto mb-3 opacity-20" />
          <p className="text-sm">
            No results for{" "}
            <span className="text-gray-400 font-medium">"{lastQuery}"</span>
            {" "}in <span className="text-gray-400">{selectedCollection}</span>
          </p>
        </div>
      )}

      {/* ── results ── */}
      {results !== null && results.length > 0 && (
        <div className="space-y-3">
          <p className="text-xs text-gray-500">
            {results.length} result{results.length !== 1 ? "s" : ""} ·{" "}
            <span className="text-gray-400">{selectedCollection}</span> ·{" "}
            <span className="text-gray-400">"{lastQuery}"</span>
          </p>
          {results.map((r) => (
            <ResultCard key={r.uuid} result={r} />
          ))}
        </div>
      )}
    </div>
  );
}

// ─── result card ─────────────────────────────────────────────────────────────

function ResultCard({ result }) {
  const [expanded, setExpanded] = useState(false);
  const { properties = {}, score } = result;

  const content    = properties.content     || "";
  const source     = properties.source_file || "";
  const header     = properties.header_path || "";
  const fileType   = properties.file_type   || "";
  const images     = Array.isArray(properties.images) ? properties.images : [];
  const chunkIndex = properties.chunk_index ?? null;

  const PREVIEW_LEN = 420;
  const needsExpand = content.length > PREVIEW_LEN;
  const displayed   = expanded || !needsExpand ? content : content.slice(0, PREVIEW_LEN) + "…";

  return (
    <div className="bg-gray-800 rounded-xl p-4 space-y-3 border border-gray-700/50">

      {/* top row: source + type + score */}
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-2 flex-wrap min-w-0">
          <FileText className="w-4 h-4 text-gray-400 shrink-0" />
          <span className="text-sm font-medium text-gray-200 truncate">{source || "Unknown"}</span>
          {fileType && (
            <span className="text-xs px-1.5 py-0.5 bg-gray-700 rounded text-gray-400 shrink-0 uppercase">
              {fileType}
            </span>
          )}
          {chunkIndex != null && (
            <span className="text-xs text-gray-600 shrink-0">#{chunkIndex}</span>
          )}
        </div>
        {score != null && (
          <span className={`text-xs font-mono shrink-0 tabular-nums ${scoreColor(score)}`}>
            {score.toFixed(4)}
          </span>
        )}
      </div>

      {/* header path breadcrumb */}
      {header && (
        <p className="text-xs text-gray-500 truncate">{header}</p>
      )}

      {/* content */}
      <p className="text-sm text-gray-300 leading-relaxed whitespace-pre-wrap break-words">
        {displayed}
      </p>
      {needsExpand && (
        <button
          onClick={() => setExpanded((v) => !v)}
          className="text-xs text-blue-400 hover:text-blue-300 transition-colors"
        >
          {expanded ? "Show less" : "Show more"}
        </button>
      )}

      {/* images */}
      {images.length > 0 && (
        <div className="flex flex-wrap gap-2 pt-1 border-t border-gray-700/50">
          {images.map((url, idx) => (
            <ImageThumb key={idx} url={url} idx={idx} />
          ))}
        </div>
      )}
    </div>
  );
}

// ─── image thumbnail with graceful fallback ───────────────────────────────────

function ImageThumb({ url, idx }) {
  const [errored, setErrored] = useState(false);

  if (errored) {
    return (
      <a
        href={url}
        target="_blank"
        rel="noreferrer"
        className="flex items-center gap-1 text-xs text-blue-400 hover:text-blue-300 transition-colors"
      >
        <Image className="w-3 h-3" />
        Image {idx + 1}
        <ExternalLink className="w-3 h-3" />
      </a>
    );
  }

  return (
    <a href={url} target="_blank" rel="noreferrer" title={`Image ${idx + 1}`}>
      <img
        src={url}
        alt={`Image ${idx + 1}`}
        onError={() => setErrored(true)}
        className="h-24 w-auto rounded-lg border border-gray-700 object-cover
          hover:border-blue-500 transition-colors cursor-zoom-in"
      />
    </a>
  );
}
