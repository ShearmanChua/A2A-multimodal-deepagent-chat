"""Markdown chunker that splits on natural boundaries and extracts image metadata."""
import re
from pathlib import Path

MAX_CHUNK_CHARS = 3500
MIN_CONTENT_LINES = 2  # chunks with fewer non-empty lines are dropped

_IMAGE_URL_RE = re.compile(r"!\[[^\]]*\]\(((?:https?|objstore)://[^)]+)\)")

# Split strategies applied in priority order.  Each is tried in sequence;
# the first one that yields > 1 part is used.  If none splits the text the
# segment is hard-cut at word boundaries.
_STRATEGIES = [
    re.compile(r"(?=\n\*\*[^*\n]+\*\*)"),  # bold paragraph openers
    re.compile(r"\n{2,}"),                   # paragraph breaks
    re.compile(r"(?<=[.!?])\s+"),            # sentence breaks
]


def _extract_image_urls(content: str) -> list[str]:
    return _IMAGE_URL_RE.findall(content)


def chunk_markdown(markdown: str, source_file: str = "") -> list[dict]:
    """
    Split markdown into chunks at natural boundaries.

    Priority: headers (H1-H4) → bold paragraph openers → paragraph breaks → sentences.
    Each chunk is ≤ MAX_CHUNK_CHARS and carries full header-path and image URLs as metadata.
    """
    file_type = Path(source_file).suffix.lower().lstrip(".") if source_file else ""

    header_re = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)
    matches = list(header_re.finditer(markdown))

    raw_sections: list[dict] = []

    preamble = markdown[: matches[0].start()].strip() if matches else markdown.strip()
    if preamble:
        raw_sections.append({"level": 0, "title": "", "body": preamble})

    for idx, m in enumerate(matches):
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(markdown)
        raw_sections.append({
            "level": len(m.group(1)),
            "title": m.group(2).strip(),
            "body": markdown[m.start() : end].strip(),
        })

    if not raw_sections:
        raw_sections = [{"level": 0, "title": "", "body": markdown.strip()}]

    chunks: list[dict] = []
    header_stack: list[dict] = []
    chunk_index = 0

    for section in raw_sections:
        level = section["level"]
        if level > 0:
            while header_stack and header_stack[-1]["level"] >= level:
                header_stack.pop()
            header_stack.append({"level": level, "title": section["title"]})

        body = section["body"].strip()
        if not body:
            continue

        header_path = " > ".join(s["title"] for s in header_stack if s["title"])
        base_meta = {
            "source_file": source_file,
            "file_type": file_type,
            "header_path": header_path,
        }

        sub = _split_body(body, base_meta, chunk_index)
        chunks.extend(sub)
        chunk_index += len(sub)

    for chunk in chunks:
        chunk["images"] = _extract_image_urls(chunk["content"])

    # Drop chunks whose content is only a single line (e.g. lone headers or
    # one-line labels that carry no useful retrieval signal).
    chunks = [
        c for c in chunks
        if sum(1 for l in c["content"].splitlines() if l.strip()) >= MIN_CONTENT_LINES
    ]

    # Re-index after filtering so chunk_index values are contiguous.
    for i, chunk in enumerate(chunks):
        chunk["chunk_index"] = i

    return chunks


def _split_body(text: str, base_meta: dict, start_index: int) -> list[dict]:
    """
    Iteratively split text into pieces ≤ MAX_CHUNK_CHARS, then merge
    adjacent small pieces into final chunks.  Uses a stack to avoid
    recursion-depth issues with very long unsplittable sections.
    """
    if len(text) <= MAX_CHUNK_CHARS:
        return [{**base_meta, "chunk_index": start_index, "content": text}]

    # Phase 1 — reduce to atomic pieces via a LIFO stack (depth-first,
    # in-order).  Each item is a text segment to process.
    stack: list[str] = [text.strip()]
    atomic: list[str] = []

    while stack:
        segment = stack.pop()
        if not segment:
            continue

        if len(segment) <= MAX_CHUNK_CHARS:
            atomic.append(segment)
            continue

        split_done = False
        for pattern in _STRATEGIES:
            parts = [p for p in pattern.split(segment) if p.strip()]
            if len(parts) > 1:
                # Push in reverse so the first part is processed next.
                for p in reversed(parts):
                    stack.append(p.strip())
                split_done = True
                break

        if not split_done:
            # No split strategy worked — hard-cut at word boundaries.
            seg = segment
            hard_pieces: list[str] = []
            while seg:
                if len(seg) <= MAX_CHUNK_CHARS:
                    hard_pieces.append(seg)
                    break
                cut = seg.rfind(" ", 0, MAX_CHUNK_CHARS)
                cut_at = cut if cut > 0 else MAX_CHUNK_CHARS
                hard_pieces.append(seg[:cut_at].strip())
                seg = seg[cut_at:].strip()
            # Pieces are already in order; reverse-push onto the stack so
            # they come out in forward order.
            for p in reversed(hard_pieces):
                stack.append(p)

    # Phase 2 — greedily merge adjacent pieces into chunks ≤ MAX_CHUNK_CHARS.
    chunks: list[dict] = []
    buf = ""

    for piece in atomic:
        joined = (buf + "\n\n" + piece).strip() if buf else piece
        if len(joined) <= MAX_CHUNK_CHARS:
            buf = joined
        else:
            if buf:
                chunks.append({
                    **base_meta,
                    "chunk_index": start_index + len(chunks),
                    "content": buf,
                })
            buf = piece

    if buf:
        chunks.append({
            **base_meta,
            "chunk_index": start_index + len(chunks),
            "content": buf,
        })

    return chunks or [{**base_meta, "chunk_index": start_index, "content": text[:MAX_CHUNK_CHARS]}]
