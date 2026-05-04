"""
VLM-based target detection tool for MCP.

Uses a Vision Language Model (VLM) served via a vLLM-compatible endpoint
(OpenAI chat-completions API) to perform high-quality target detection in
images.  The VLM analyses the full image and returns structured JSON
describing every detected target with bounding-box coordinates, labels,
confidence estimates, and target types.

Accepts media as either:
- A URL (http:// or https://) pointing to an image or video
- A base64-encoded string (with or without data URL prefix)

This tool produces significantly richer and more accurate detections than
the YOLO-based ``target_detection`` tool because the VLM can reason about
context, identify camouflaged or partially occluded targets, and provide
natural-language descriptions.

Environment variables
---------------------
VLM_ENDPOINT : str
    Base URL of the vLLM / OpenAI-compatible API (e.g. ``http://vllm:8000/v1``).
    Falls back to ``TOOL_LLM_URL``, then ``MODEL_ENDPOINT``.
VLM_MODEL : str
    Model name to request (e.g. ``Qwen/Qwen2.5-VL-72B-Instruct``).
    Falls back to ``TOOL_LLM_NAME``, then ``MODEL_NAME``.
VLM_API_KEY : str
    API key for the endpoint.  Falls back to ``API_KEY``, then ``MODEL_API_KEY``.
"""

import base64
import json
import logging
import os
import re
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from PIL import Image as PILImage, ImageDraw, ImageFont
from fastmcp.utilities.types import Image  # MCP Image type

from tools.media_store import resolve_media_source

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)

logger = logging.getLogger("mcp-vlm-target-detection")

# Directory where annotated images are saved for verification
_SAVE_DIR = Path("saved_images/vlm_target_detection")
_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# VLM endpoint configuration
# ---------------------------------------------------------------------------

def _get_vlm_config() -> tuple[str, str, str]:
    """Return (endpoint, model, api_key) from environment variables."""
    endpoint = (
        os.environ.get("VLM_ENDPOINT")
        or os.environ.get("TOOL_LLM_URL")
        or os.environ.get("MODEL_ENDPOINT", "")
    )
    model = (
        os.environ.get("VLM_MODEL")
        or os.environ.get("TOOL_LLM_NAME")
        or os.environ.get("MODEL_NAME", "")
    )
    api_key = (
        os.environ.get("VLM_API_KEY")
        or os.environ.get("API_KEY")
        or os.environ.get("MODEL_API_KEY", "")
    )
    return endpoint, model, api_key


# ---------------------------------------------------------------------------
# Detection prompt
# ---------------------------------------------------------------------------

_DETECTION_SYSTEM_PROMPT = """\
You are a military imagery analyst AI. You will be given an image and must \
detect ALL targets/objects of interest visible in the image.

For EACH detected target, provide:
1. **label**: A concise name (e.g. "main battle tank", "patrol boat", "fighter jet", "infantry squad")
2. **target_type**: One of: armoured_vehicle, soft_vehicle, heavy_vehicle, artillery, air_defence, \
personnel, infrastructure, surface_vessel, submarine, fixed_wing_aircraft, rotary_wing_aircraft, \
UAV, rail_asset, unknown
3. **domain**: One of: land, maritime, aerial, unknown
4. **confidence**: Your confidence from 0.0 to 1.0
5. **bbox**: Bounding box as [x1, y1, x2, y2] in pixel coordinates (top-left origin). \
Estimate the coordinates as accurately as possible based on the image dimensions provided.
6. **description**: A brief description of the target and any notable features

You MUST respond with ONLY a valid JSON object in this exact format:
{
  "image_width": <int>,
  "image_height": <int>,
  "detections": [
    {
      "label": "<string>",
      "target_type": "<string>",
      "domain": "<string>",
      "confidence": <float>,
      "bbox": [<x1>, <y1>, <x2>, <y2>],
      "description": "<string>"
    }
  ]
}

If no targets are detected, return an empty detections list.
Do NOT include any text outside the JSON object.
"""

# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

_BOX_COLOURS = [
    "#FF0000", "#00FF00", "#0000FF", "#FFFF00",
    "#00FFFF", "#FF00FF", "#FF8800", "#8800FF",
    "#FF4444", "#44FF44", "#4444FF", "#FFAA00",
]


def _get_font(size: int = 14) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
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
    """Draw bounding boxes and labels on the image."""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    font = _get_font(14)

    for idx, det in enumerate(detections):
        bbox = det.get("bbox", [])
        if len(bbox) != 4:
            continue

        x1, y1, x2, y2 = bbox
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
# VLM API call
# ---------------------------------------------------------------------------

def _call_vlm(
    image_base64: str,
    image_width: int,
    image_height: int,
    endpoint: str,
    model: str,
    api_key: str,
) -> dict[str, Any]:
    """
    Call the VLM endpoint with the image and detection prompt.

    Returns the parsed JSON response from the VLM.
    """
    url = endpoint.rstrip("/")
    if not url.endswith("/chat/completions"):
        url = url.rstrip("/") + "/chat/completions"

    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    user_content = [
        {
            "type": "text",
            "text": (
                f"Analyze this image ({image_width}x{image_height} pixels) and detect all "
                f"targets/objects of interest. Return the results as JSON with bounding boxes "
                f"in pixel coordinates for the {image_width}x{image_height} image."
            ),
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_base64}",
            },
        },
    ]

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _DETECTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": 4096,
        "temperature": 0.1,
    }

    logger.info("Calling VLM endpoint %s with model %s …", url, model)

    response = requests.post(url, headers=headers, json=payload, timeout=120)
    response.raise_for_status()

    result = response.json()
    content = result["choices"][0]["message"]["content"]

    # Parse JSON from the response (handle markdown code blocks)
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
    if json_match:
        content = json_match.group(1).strip()

    # Try to find JSON object in the response
    json_obj_match = re.search(r"\{[\s\S]*\}", content)
    if json_obj_match:
        content = json_obj_match.group(0)

    parsed = json.loads(content)
    return parsed


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _vlm_detect_single_frame(
    pil_image: PILImage.Image,
    endpoint: str,
    model: str,
    api_key: str,
    frame_index: int | None = None,
) -> tuple[Image, list[dict[str, Any]]]:
    """Run VLM detection on a single PIL image."""
    suffix = f"_frame{frame_index}" if frame_index is not None else ""
    w, h = pil_image.size

    logger.info(
        "Running VLM target detection%s (%dx%d) …",
        suffix, w, h,
    )

    timestamp = int(time.time() * 1000)
    original_path = _SAVE_DIR / f"{timestamp}{suffix}_original.jpg"
    pil_image.save(original_path, format="JPEG", quality=90)

    # Encode image to base64 for the VLM
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG", quality=85)
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    try:
        vlm_result = _call_vlm(image_base64, w, h, endpoint, model, api_key)
    except Exception as exc:
        logger.error("VLM call failed%s: %s", suffix, exc)
        raise RuntimeError(f"VLM target detection failed: {exc}") from exc

    detections = vlm_result.get("detections", [])

    validated_detections: list[dict[str, Any]] = []
    for det in detections:
        bbox = det.get("bbox", [])
        if len(bbox) != 4:
            continue

        x1 = max(0, min(int(bbox[0]), w - 1))
        y1 = max(0, min(int(bbox[1]), h - 1))
        x2 = max(x1 + 1, min(int(bbox[2]), w))
        y2 = max(y1 + 1, min(int(bbox[3]), h))

        validated_det = {
            "label": det.get("label", "unknown"),
            "target_type": det.get("target_type", "unknown"),
            "domain": det.get("domain", "unknown"),
            "confidence": round(float(det.get("confidence", 0.5)), 4),
            "bbox": [x1, y1, x2, y2],
            "description": det.get("description", ""),
        }
        if frame_index is not None:
            validated_det["frame_index"] = frame_index
        validated_detections.append(validated_det)

    validated_detections.sort(key=lambda d: d["confidence"], reverse=True)

    logger.info("VLM detected %d target(s)%s.", len(validated_detections), suffix)

    annotated_image = _draw_detections(pil_image, validated_detections)

    annotated_path = _SAVE_DIR / f"{timestamp}{suffix}_vlm_annotated.jpg"
    annotated_image.save(annotated_path, format="JPEG", quality=90)

    buffer = BytesIO()
    annotated_image.save(buffer, format="JPEG", quality=85)
    mcp_image = Image(data=buffer.getvalue(), format="jpeg")

    return mcp_image, validated_detections


def vlm_detect_targets_in_image(
    image_source: str,
    detection_prompt: str = "",
) -> tuple[Image | list[Image], list[dict[str, Any]]]:
    """
    Detect targets in an image or video using a VLM.

    Accepts media as either a URL or base64-encoded string.
    If the source is a video, frames are extracted and detection
    runs on each frame independently.

    Parameters
    ----------
    image_source : str
        Either:
        - A URL (http:// or https://) pointing to an image or video
        - A base64-encoded string (with or without data URL prefix)
    detection_prompt : str, optional
        Additional instructions to guide the VLM's detection focus.

    Returns
    -------
    tuple[Image | list[Image], list[dict]]
        - For images: a single annotated MCP Image.
        - For videos: a list of annotated MCP Images (one per frame).
        - A list of detection dicts, each optionally containing
          ``frame_index`` for video frames.

    Raises
    ------
    ValueError
        If the source is invalid.
    RuntimeError
        If the VLM endpoint is not configured or the call fails.
    """
    endpoint, model, api_key = _get_vlm_config()
    if not endpoint or not model:
        raise RuntimeError(
            "VLM endpoint not configured. Set VLM_ENDPOINT and VLM_MODEL "
            "(or TOOL_LLM_URL and TOOL_LLM_NAME) environment variables."
        )

    media_type, frames = resolve_media_source(image_source)

    if media_type == "image":
        mcp_image, detections = _vlm_detect_single_frame(
            frames[0], endpoint, model, api_key,
        )
        return mcp_image, detections

    # Video — process each frame
    all_detections: list[dict[str, Any]] = []
    mcp_images: list[Image] = []

    for i, frame in enumerate(frames):
        mcp_img, dets = _vlm_detect_single_frame(
            frame, endpoint, model, api_key, frame_index=i,
        )
        mcp_images.append(mcp_img)
        all_detections.extend(dets)

    logger.info(
        "VLM video detection complete: %d frame(s), %d total detection(s).",
        len(frames), len(all_detections),
    )

    return mcp_images, all_detections
