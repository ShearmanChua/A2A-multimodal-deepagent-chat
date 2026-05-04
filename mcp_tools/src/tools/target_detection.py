"""
Target detection tool for MCP.

Detects targets (vehicles, vessels, aircraft, infrastructure, personnel, etc.)
in images using a YOLOv8 object-detection model.  Returns the annotated
image with bounding boxes and structured detection metadata.

Accepts media as either:
- A URL (http:// or https://) pointing to an image or video
- A base64-encoded string (with or without data URL prefix)

When no GPU-accelerated model is available the tool falls back to a stub that
generates plausible random detections so the MCP pipeline can still be exercised
end-to-end.
"""

import logging
import os
import time
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image as PILImage, ImageDraw, ImageFont
from fastmcp.utilities.types import Image  # MCP Image type

from tools.media_store import resolve_media_source

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)

logger = logging.getLogger("mcp-target-detection")

# Directory where received and annotated images are saved for verification
_SAVE_DIR = Path("saved_images/target_detection")
_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# YOLOv8 model loading (lazy singleton)
# ---------------------------------------------------------------------------
_yolo_model = None
_USE_YOLO = os.environ.get("USE_YOLO", "true").lower() in ("1", "true", "yes")

# YOLO COCO class-name → military/strategic target-type mapping.
# Only classes that are relevant for target detection are included.
_COCO_TO_TARGET: dict[str, str] = {
    "person": "personnel",
    "bicycle": "light_vehicle",
    "car": "light_vehicle",
    "motorcycle": "light_vehicle",
    "airplane": "fixed_wing_aircraft",
    "bus": "heavy_vehicle",
    "train": "rail_asset",
    "truck": "heavy_vehicle",
    "boat": "surface_vessel",
    "traffic light": "infrastructure",
    "fire hydrant": "infrastructure",
    "stop sign": "infrastructure",
    "bench": "infrastructure",
    "bird": "aerial_object",
    "helicopter": "rotary_wing_aircraft",
}

# Colours for bounding boxes (cycled)
_BOX_COLOURS = [
    "#FF0000", "#00FF00", "#0000FF", "#FFFF00",
    "#00FFFF", "#FF00FF", "#FF8800", "#8800FF",
]


def _get_yolo_model():
    """Lazily load the YOLOv8 model (downloads on first use)."""
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model

    try:
        from ultralytics import YOLO

        model_path = os.environ.get("YOLO_MODEL", "yolov8n.pt")
        logger.info("Loading YOLOv8 model from %s …", model_path)
        _yolo_model = YOLO(model_path)
        logger.info("YOLOv8 model loaded successfully.")
        return _yolo_model
    except Exception as exc:
        logger.warning("Failed to load YOLOv8 model: %s – falling back to stub.", exc)
        return None


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _get_font(size: int = 16) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Try to load a TrueType font; fall back to the default bitmap font."""
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size,
        )
    except (IOError, OSError):
        return ImageFont.load_default()


def _draw_detections(
    image: PILImage.Image,
    detections: list[dict[str, Any]],
) -> PILImage.Image:
    """
    Draw bounding boxes and labels on *image* for each detection dict.

    Each detection must contain at minimum:
        - ``bbox``: [x1, y1, x2, y2]
        - ``label``: str
        - ``confidence``: float
    """
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    font = _get_font(16)

    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]
        label = det.get("label", "target")
        conf = det.get("confidence", 0.0)
        colour = _BOX_COLOURS[idx % len(_BOX_COLOURS)]

        text = f"{label} {conf:.2f}"

        # Bounding box
        draw.rectangle([x1, y1, x2, y2], outline=colour, width=3)

        # Label background + text
        text_bbox = draw.textbbox((x1, y1), text, font=font)
        draw.rectangle(
            [text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2],
            fill=colour,
        )
        draw.text((x1, y1), text, fill="white", font=font)

    return annotated


# ---------------------------------------------------------------------------
# YOLO-based detection
# ---------------------------------------------------------------------------

def _detect_with_yolo(
    image: PILImage.Image,
    confidence_threshold: float = 0.25,
    max_detections: int = 20,
) -> list[dict[str, Any]]:
    """Run YOLOv8 inference and return structured detections."""
    model = _get_yolo_model()
    if model is None:
        return _detect_stub(image, num_boxes=min(max_detections, 5))

    results = model.predict(
        source=image,
        conf=confidence_threshold,
        verbose=False,
    )

    detections: list[dict[str, Any]] = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names.get(cls_id, f"class_{cls_id}")
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            target_type = _COCO_TO_TARGET.get(cls_name, "unknown")

            detections.append({
                "label": cls_name,
                "target_type": target_type,
                "confidence": round(conf, 4),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
            })

    # Sort by confidence descending and cap
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections[:max_detections]


# ---------------------------------------------------------------------------
# Stub detection (fallback when YOLO is unavailable)
# ---------------------------------------------------------------------------

import random

_STUB_TARGETS = [
    ("tank", "armoured_vehicle"),
    ("APC", "armoured_vehicle"),
    ("truck", "heavy_vehicle"),
    ("helicopter", "rotary_wing_aircraft"),
    ("fighter_jet", "fixed_wing_aircraft"),
    ("patrol_boat", "surface_vessel"),
    ("radar_installation", "infrastructure"),
    ("personnel", "personnel"),
    ("SAM_launcher", "air_defence"),
    ("cargo_ship", "surface_vessel"),
    ("drone", "UAV"),
    ("artillery", "artillery"),
]


def _detect_stub(
    image: PILImage.Image,
    num_boxes: int = 5,
) -> list[dict[str, Any]]:
    """Generate plausible random target detections (stub)."""
    w, h = image.size
    detections: list[dict[str, Any]] = []

    for _ in range(num_boxes):
        min_side_w = max(int(w * 0.08), 20)
        min_side_h = max(int(h * 0.08), 20)

        x1 = random.randint(0, max(w - min_side_w - 1, 0))
        y1 = random.randint(0, max(h - min_side_h - 1, 0))
        x2 = random.randint(x1 + min_side_w, min(x1 + int(w * 0.4), w))
        y2 = random.randint(y1 + min_side_h, min(y1 + int(h * 0.4), h))

        label, target_type = random.choice(_STUB_TARGETS)
        confidence = round(random.uniform(0.45, 0.97), 4)

        detections.append({
            "label": label,
            "target_type": target_type,
            "confidence": confidence,
            "bbox": [x1, y1, x2, y2],
        })

    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections


# ---------------------------------------------------------------------------
# Single-frame detection helper
# ---------------------------------------------------------------------------

def _detect_single_frame(
    pil_image: PILImage.Image,
    confidence_threshold: float,
    max_detections: int,
    frame_index: int | None = None,
) -> tuple[Image, list[dict[str, Any]]]:
    """Run detection on a single PIL image and return (MCP Image, detections)."""
    suffix = f"_frame{frame_index}" if frame_index is not None else ""

    logger.info(
        "Running target detection%s (%dx%d) …",
        suffix, pil_image.width, pil_image.height,
    )

    # Save the received (original) image for verification
    timestamp = int(time.time() * 1000)
    original_path = _SAVE_DIR / f"{timestamp}{suffix}_original.jpg"
    pil_image.save(original_path, format="JPEG", quality=90)
    logger.info("Saved received image to %s", original_path)

    # Run detection
    if _USE_YOLO:
        detections = _detect_with_yolo(pil_image, confidence_threshold, max_detections)
    else:
        detections = _detect_stub(pil_image, num_boxes=min(max_detections, 5))

    logger.info("Detected %d target(s)%s.", len(detections), suffix)

    # Draw bounding boxes on the image
    annotated_image = _draw_detections(pil_image, detections)

    # Save the annotated image for verification
    annotated_path = _SAVE_DIR / f"{timestamp}{suffix}_annotated.jpg"
    annotated_image.save(annotated_path, format="JPEG", quality=90)
    logger.info("Saved annotated image to %s", annotated_path)

    # Encode the annotated image as JPEG bytes for MCP transport
    buffer = BytesIO()
    annotated_image.save(buffer, format="JPEG", quality=85)
    mcp_image = Image(data=buffer.getvalue(), format="jpeg")

    # Tag detections with frame index if applicable
    if frame_index is not None:
        for det in detections:
            det["frame_index"] = frame_index

    return mcp_image, detections


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_targets_in_image(
    image_source: str,
    confidence_threshold: float = 0.25,
    max_detections: int = 20,
) -> tuple[Image | list[Image], list[dict[str, Any]]]:
    """
    Detect targets in an image or video.

    Accepts media as either a URL or base64-encoded string.
    If the source is a video, frames are extracted and detection runs on each
    frame independently.  Results from all frames are aggregated.

    Parameters
    ----------
    image_source : str
        Either:
        - A URL (http:// or https://) pointing to an image or video
        - A base64-encoded string (with or without data URL prefix)
    confidence_threshold : float, optional
        Minimum confidence score to include a detection (default 0.25).
    max_detections : int, optional
        Maximum number of detections to return **per frame** (default 20).

    Returns
    -------
    tuple[Image | list[Image], list[dict]]
        - For images: a single annotated MCP Image.
        - For videos: a list of annotated MCP Images (one per frame).
        - A list of detection dicts each containing ``label``,
          ``target_type``, ``confidence``, ``bbox``, and optionally
          ``frame_index`` (for video frames).

    Raises
    ------
    ValueError
        If the source is neither a valid URL nor valid base64 data.
    """
    media_type, frames = resolve_media_source(image_source)

    if media_type == "image":
        # Single image
        mcp_image, detections = _detect_single_frame(
            frames[0], confidence_threshold, max_detections,
        )
        return mcp_image, detections

    # Video — process each frame
    all_detections: list[dict[str, Any]] = []
    mcp_images: list[Image] = []

    for i, frame in enumerate(frames):
        mcp_img, dets = _detect_single_frame(
            frame, confidence_threshold, max_detections, frame_index=i,
        )
        mcp_images.append(mcp_img)
        all_detections.extend(dets)

    logger.info(
        "Video detection complete: %d frame(s), %d total detection(s).",
        len(frames), len(all_detections),
    )

    return mcp_images, all_detections
