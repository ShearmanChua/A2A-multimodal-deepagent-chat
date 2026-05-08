"""Document-to-markdown conversion using Docling, with optional image upload to SeaweedFS."""
import io
import logging
import os
import re
from pathlib import Path
from typing import Callable

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    OcrAutoOptions,
    PdfPipelineOptions,
    RapidOcrOptions,
    ThreadedPdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

logger = logging.getLogger("docling-ingestion.converter")

_GLYPH_TOKENS = ["7KLV", "$SS", "SXE", "DYD", "UHO", "IRU"]


def _looks_like_glyph(text: str) -> bool:
    sample = text[:5000]
    caps_words = re.findall(r"\b[A-Z0-9$]{5,}\b", sample)
    if len(caps_words) > 20:
        return True
    letters = re.findall(r"[A-Z]", sample)
    if letters and sum(c in "AEIOU" for c in letters) / len(letters) < 0.22:
        return True
    return any(tok in sample for tok in _GLYPH_TOKENS)


def _pdf_pipeline_opts(force_ocr: bool = False) -> PdfPipelineOptions:
    use_cuda = os.environ.get("USE_CUDA", "false").lower() == "true"
    models_path = os.environ.get("DOCLING_MODELS_PATH", "~/.cache/docling/models")

    if use_cuda:
        # GPU path: threaded pipeline with larger batch sizes per the docling GPU docs.
        opts = ThreadedPdfPipelineOptions(
            do_ocr=force_ocr,
            do_table_structure=True,
            artifacts_path=models_path,
            generate_picture_images=True,
            ocr_batch_size=64,
            layout_batch_size=64,
            table_batch_size=4,
        )
        opts.table_structure_options.do_cell_matching = True
        opts.accelerator_options = AcceleratorOptions(device=AcceleratorDevice.CUDA)
        if force_ocr:
            opts.ocr_options = RapidOcrOptions(
                backend="torch", force_full_page_ocr=True
            )
        else:
            opts.ocr_options = RapidOcrOptions(backend="torch")
    else:
        opts = PdfPipelineOptions(
            do_ocr=force_ocr,
            do_table_structure=True,
            artifacts_path=models_path,
            generate_picture_images=True,
        )
        opts.table_structure_options.do_cell_matching = True
        opts.accelerator_options = AcceleratorOptions(
            num_threads=4,
            device=AcceleratorDevice.CPU,
        )
        opts.ocr_options = OcrAutoOptions(force_full_page_ocr=force_ocr)

    return opts


def _pdf_converter(force_ocr: bool = False) -> DocumentConverter:
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=_pdf_pipeline_opts(force_ocr))
        }
    )


def _pic_to_bytes(picture) -> bytes | None:
    """Extract raw PNG bytes from a docling PictureItem, handling version differences."""
    img_obj = getattr(picture, "image", None)
    if img_obj is None:
        return None

    # docling < 2.x: img_obj is a PIL Image directly
    if hasattr(img_obj, "save"):
        buf = io.BytesIO()
        img_obj.save(buf, format="PNG")
        return buf.getvalue()

    # docling >= 2.x: img_obj is ImageContent with .pil_image
    pil = getattr(img_obj, "pil_image", None)
    if pil is not None and hasattr(pil, "save"):
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return buf.getvalue()

    return None


def _upload_pictures(result, source_stem: str, upload_fn: Callable) -> list[str | None]:
    """
    Upload all pictures in the docling result to SeaweedFS.
    Returns a list of URLs (or None for any that failed) in document order.
    """
    urls: list[str | None] = []
    pictures = getattr(result.document, "pictures", [])
    for idx, pic in enumerate(pictures):
        try:
            img_bytes = _pic_to_bytes(pic)
            if img_bytes:
                url = upload_fn(img_bytes, "png", source_stem)
                urls.append(url)
            else:
                urls.append(None)
        except Exception as exc:
            logger.warning("Could not upload picture %d from '%s': %s", idx, source_stem, exc)
            urls.append(None)
    return urls


def _inject_image_urls(markdown: str, urls: list[str | None]) -> str:
    """
    Replace <!-- image --> placeholders in markdown with SeaweedFS URLs in order.
    Placeholders with no matching URL are left as-is.
    """
    valid_urls = [u for u in urls if u]
    url_iter = iter(valid_urls)

    def replacer(match: re.Match) -> str:
        url = next(url_iter, None)
        return f"![Document image]({url})" if url else match.group(0)

    return re.sub(r"<!-- image -->", replacer, markdown)


def convert_to_markdown(
    file_path: Path,
    suffix: str,
    upload_fn: Callable | None = None,
) -> tuple[str, list[str]]:
    """
    Convert a document file to markdown.

    If *upload_fn* is provided, images extracted by docling are uploaded via
    that callable and their URLs are injected back into the markdown as
    ![Document image](<url>) references.

    Returns (markdown_text, list_of_image_urls).
    """
    if suffix in {".txt", ".md"}:
        return file_path.read_text(encoding="utf-8", errors="replace"), []

    source_stem = file_path.stem

    if suffix == ".pdf":
        logger.info("Converting PDF '%s'", file_path.name)
        result = _pdf_converter(force_ocr=False).convert(str(file_path))
        markdown = result.document.export_to_markdown()

        if _looks_like_glyph(markdown):
            logger.info("Glyph text detected — retrying with forced OCR")
            result = _pdf_converter(force_ocr=True).convert(str(file_path))
            markdown = result.document.export_to_markdown()
    else:
        logger.info("Converting document '%s'", file_path.name)
        # Enable picture images for DOCX too
        result = DocumentConverter().convert(str(file_path))
        markdown = result.document.export_to_markdown()

    image_urls: list[str] = []
    if upload_fn is not None:
        raw_urls = _upload_pictures(result, source_stem, upload_fn)
        image_urls = [u for u in raw_urls if u]
        if raw_urls:
            markdown = _inject_image_urls(markdown, raw_urls)

    return markdown, image_urls
