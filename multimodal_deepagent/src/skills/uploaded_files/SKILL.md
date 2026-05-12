---
name: uploaded-files
description: >
  Use this skill whenever the user has uploaded a document (PDF, Word, text, CSV, JSON,
  or similar) directly in the chat. Uploaded files land pre-processed at /uploads/ in
  your virtual filesystem. This skill is a router: it tells you which tool to call for
  each file type and how much to read so you answer precisely without loading gigabytes
  of context. Triggers: any user message that references an uploaded file, asks you to
  "read", "summarise", "analyse", or "search" a document, or when a prior message
  indicates files were written to /uploads/.
---

# Reading Uploaded Files

## Why this skill exists

When a user uploads a document in the chat, the server converts it and writes the
result to `/uploads/` in your virtual filesystem before your turn begins. The content
is **not** already in your context — you must read it with file tools.

The wrong first move is to call `read_file` on a large file blind. A 300-page PDF
becomes a wall of markdown that drowns the answer. This skill tells you the right
first move for each type, and how much to read.

## General protocol

1. **Discover what is there.**
   ```
   ls("/uploads/")
   ```
   This is always your first call. It tells you the filenames and sizes so you can
   calibrate how much to read.

2. **Look at the extension.** That is your dispatch key (see table below).

3. **Read just enough.** If the user asked "what is the executive summary?", a targeted
   `grep` followed by `read_file` on the matching section beats loading the whole file.

4. **For images embedded in a document**, the converted markdown contains
   `objstore://` paths. Call `get_object_store_image_base64` with that path to
   retrieve the image for vision analysis.

---

## File layout after upload

| Original upload | Stored at | Notes |
|---|---|---|
| `report.pdf` | `/uploads/report.md` | Converted to markdown by Docling |
| `memo.docx` / `.doc` | `/uploads/memo.md` | Converted to markdown by Docling |
| `notes.txt` / `.md` | `/uploads/notes.txt` | Stored as-is (already plain text) |
| `data.csv` / `.tsv` | `/uploads/data.csv` | Stored as-is |
| `config.json` / `.jsonl` | `/uploads/config.json` | Stored as-is |

Docling-converted files use the **original stem** with a `.md` extension.
Plain-text files keep their original name.

---

## Dispatch table

| Extension at `/uploads/` | First move | Notes |
|---|---|---|
| `.md` (from PDF / DOCX) | Size check → `grep` or `read_file` | See **Converted documents** section |
| `.txt` | Size check → `read_file` or `grep` | See **Plain text** section |
| `.csv` / `.tsv` | `read_file` with `max_lines` | See **CSV / TSV** section |
| `.json` / `.jsonl` | `read_file` then orient | See **JSON** section |
| Unknown | `read_file` with small `max_lines` to inspect | Ask user if unclear |

---

## Converted documents (`.md` — from PDF or Word)

Docling produces structured markdown: headings become `#`/`##`/`###`, tables become
GFM tables, lists and bold/italic are preserved. Images are referenced inline as
`![alt](objstore://bucket/path/to/image.png)`.

**Step 1 — Check the size.**
`ls("/uploads/")` already gave you sizes. Use this rule:

| Converted file size | Strategy |
|---|---|
| < 30 KB | `read_file` the whole file |
| 30 KB – 150 KB | `grep` for the user's keywords first; `read_file` the matching sections |
| > 150 KB | `grep` to locate relevant headings; read only those sections |

**Step 2 — Targeted read (medium/large files).**

Search for keywords from the user's question before loading the file:
```
grep("/uploads/report.md", "executive summary")
grep("/uploads/report.md", "revenue")
```

Then `read_file` with an offset/limit that brackets the matching region, not the whole
document.

**Step 3 — Handle embedded images.**

When you see a line like:
```
![Figure 3](objstore://media/docling/report/abc123.png)
```
Call `get_object_store_image_base64("objstore://media/docling/report/abc123.png")` to
retrieve the image and analyse it visually. Do this for every image that is relevant to
the user's question — do not skip figures, charts, or diagrams.

---

## Plain text (`.txt`, `.md` originals, logs)

```
read_file("/uploads/notes.txt")
```

If the file is large (> 20 KB from `ls`), grep first:
```
grep("/uploads/app.log", "ERROR")
```

For log files the user almost always cares about recent entries — read the end of
the file if you need context around a grep hit.

---

## CSV / TSV

Do **not** read a CSV blind — a file with 100,000 rows will flood your context.

Read the first few rows to understand the schema:
```
read_file("/uploads/data.csv", max_lines=10)
```

For row counts, trust the size from `ls` as a rough proxy (bytes ÷ average row size).
Scan for the column the user cares about with `grep` before reading more rows.

---

## JSON / JSONL

Read the whole file if it is small (< 10 KB). For larger files, read the first
20–30 lines to understand the structure, then grep for the specific key or value the
user asked about:

```
read_file("/uploads/config.json", max_lines=30)
grep("/uploads/events.jsonl", "\"type\": \"error\"")
```

JSONL (one object per line) can be enormous. Always grep before reading.

---

## Multi-file uploads

When several files are present in `/uploads/`, iterate through all of them unless the
user named a specific one. Announce which file you are reading before each section of
your answer so the user can follow along.

---

## Tool reference

| Tool | Key arguments | Use for |
|---|---|---|
| `ls` | `path` | Discover what is in `/uploads/` and check file sizes |
| `read_file` | `path`, `max_lines` (optional), `offset` (optional) | Read a file or a slice of it |
| `grep` | `path`, `pattern` | Find relevant lines/sections before reading |
| `glob` | `pattern` | Find files matching a pattern, e.g. `glob("/uploads/*.md")` |
| `write_file` | `path`, `content` | Save a summary or derived artefact back to `/uploads/` |
| `get_object_store_image_base64` | `objstore://` path | Retrieve an image embedded in a converted document |
| `get_object_store_presigned_url` | `objstore://` path | Get a shareable URL for an embedded image |

---

## What NOT to do

- Do **not** call `read_file` on a large file without checking its size from `ls` first.
- Do **not** skip `objstore://` image references — figures and charts often contain the
  key information the user is asking about.
- Do **not** assume a `.md` file is small just because it came from a single-page PDF —
  Docling preserves tables verbatim, which can be very wide.
- Do **not** call `ls` more than once per turn unless a write has happened in between.
