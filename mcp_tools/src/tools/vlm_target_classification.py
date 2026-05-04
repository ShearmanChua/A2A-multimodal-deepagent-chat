"""
VLM-based target classification tool for MCP.

Uses a Vision Language Model (VLM) served via a vLLM-compatible endpoint
(OpenAI chat-completions API) to perform high-quality target classification
on a detected region within an image.

Accepts media as either:
- A URL (http:// or https://) pointing to an image or video
- A base64-encoded string (with or without data URL prefix)

Given an image and a bounding box (from any detection tool), the VLM crops
and analyses the target region to produce a detailed classification report
including type hierarchy, threat assessment, observable features, and
recommended actions.

This tool produces significantly richer classifications than the
ResNet-based ``target_classification`` tool because the VLM can reason
about visual context, markings, camouflage patterns, and operational
posture.

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

logger = logging.getLogger("mcp-vlm-target-classification")

# Directory where cropped/annotated images are saved for verification
_SAVE_DIR = Path("saved_images/vlm_target_classification")
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
# Classification prompt
# ---------------------------------------------------------------------------

_CLASSIFICATION_SYSTEM_PROMPT = """\
You are a military imagery analyst AI specialising in target classification. \
You will be given a cropped image of a detected target along with optional \
context (the full scene and detection label).

Analyse the target and provide a DETAILED classification report.

Your response MUST be a valid JSON object with this exact structure:
{
  "domain": "<land|maritime|aerial|unknown>",
  "category": "<string>",
  "sub_type": "<string>",
  "label": "<string - concise human-readable name>",
  "nato_designation": "<string or null - NATO reporting name if applicable>",
  "threat_level": "<critical|high|medium|low|minimal|unknown>",
  "confidence": <float 0.0-1.0>,
  "observable_features": [
    "<string - list of key visual features observed>"
  ],
  "camouflage_assessment": "<none|partial|full|unknown>",
  "operational_status": "<active|stationary|damaged|destroyed|unknown>",
  "estimated_heading": "<string or null - e.g. 'north-east' if determinable>",
  "description": "<string - detailed paragraph describing the target>",
  "recommended_actions": [
    "<string - suggested follow-up actions for the analyst>"
  ]
}

Classification categories by domain:

**Land targets:**
- armoured_vehicle: main_battle_tank, infantry_fighting_vehicle, APC, self_propelled_gun, armoured_recon
- soft_vehicle: military_truck, utility_vehicle, MRAP, command_vehicle, fuel_tanker
- artillery: towed_howitzer, self_propelled_howitzer, MLRS, mortar_system
- air_defence: SAM_launcher, MANPADS_team, AAA_gun, radar_system, C2_vehicle
- personnel: infantry_squad, special_forces, support_crew, observation_post
- infrastructure: bridge, bunker, command_post, supply_depot, comms_tower, checkpoint

**Maritime targets:**
- surface_combatant: destroyer, frigate, corvette, patrol_vessel, missile_boat
- submarine: attack_submarine, SSBN, midget_submarine
- auxiliary_vessel: supply_ship, tanker, landing_craft, MCM_vessel
- civilian_vessel: cargo_ship, container_ship, fishing_vessel, ferry, yacht

**Aerial targets:**
- fixed_wing: fighter_aircraft, bomber, transport, surveillance, tanker_aircraft
- rotary_wing: attack_helicopter, transport_helicopter, utility_helicopter
- UAV: tactical_UAV, MALE_UAV, HALE_UAV, loitering_munition, micro_UAV

Threat levels:
- critical: Strategic weapons, air defence systems, submarines
- high: Armoured vehicles, artillery, surface combatants, fighter aircraft
- medium: Helicopters, UAVs, infrastructure, radar
- low: Soft vehicles, personnel, auxiliary vessels
- minimal: Civilian vessels, non-military objects

Do NOT include any text outside the JSON object.
"""

# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

_THREAT_COLOURS: dict[str, str] = {
    "critical": "#FF0000",
    "high": "#FF6600",
    "medium": "#FFCC00",
    "low": "#00CC00",
    "minimal": "#00AAFF",
    "unknown": "#888888",
}


def _get_font(size: int = 14) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size,
        )
    except (IOError, OSError):
        return ImageFont.load_default()


def _draw_classification_overlay(
    image: PILImage.Image,
    bbox: list[int],
    classification: dict[str, Any],
) -> PILImage.Image:
    """Draw the classification result on the image with a highlighted bbox."""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    font = _get_font(13)

    x1, y1, x2, y2 = bbox
    threat = classification.get("threat_level", "unknown")
    colour = _THREAT_COLOURS.get(threat, "#888888")

    # Draw bounding box with threat-level colour
    draw.rectangle([x1, y1, x2, y2], outline=colour, width=4)

    # Build label lines
    lines = [
        f"{classification.get('label', 'unknown')} ({classification.get('confidence', 0):.2f})",
        f"Type: {classification.get('category', '?')} / {classification.get('sub_type', '?')}",
        f"Domain: {classification.get('domain', '?')}  |  Threat: {threat.upper()}",
        f"Status: {classification.get('operational_status', '?')}  |  Camo: {classification.get('camouflage_assessment', '?')}",
    ]

    nato = classification.get("nato_designation")
    if nato:
        lines.append(f"NATO: {nato}")

    # Draw label background
    line_height = 17
    total_height = line_height * len(lines) + 8
    max_width = max(draw.textlength(line, font=font) for line in lines) + 8

    label_y = max(y1 - total_height, 0)
    draw.rectangle(
        [x1, label_y, x1 + max_width, label_y + total_height],
        fill=colour,
    )

    # Draw label text
    for i, line in enumerate(lines):
        draw.text(
            (x1 + 4, label_y + 4 + i * line_height),
            line,
            fill="white",
            font=font,
        )

    return annotated


# ---------------------------------------------------------------------------
# VLM API call
# ---------------------------------------------------------------------------

def _call_vlm_classify(
    crop_base64: str,
    context_base64: str | None,
    detection_label: str,
    endpoint: str,
    model: str,
    api_key: str,
) -> dict[str, Any]:
    """
    Call the VLM endpoint with the cropped target image and classification prompt.
    """
    url = endpoint.rstrip("/")
    if not url.endswith("/chat/completions"):
        url = url.rstrip("/") + "/chat/completions"

    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Build user content with crop (and optionally the full scene for context)
    user_content: list[dict[str, Any]] = []

    context_text = "Classify the target shown in this cropped image."
    if detection_label:
        context_text += f" The detection stage labelled it as: '{detection_label}'."
    context_text += " Provide a detailed classification report as JSON."

    user_content.append({"type": "text", "text": context_text})

    # Add the cropped target image
    user_content.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{crop_base64}",
        },
    })

    # Optionally add the full scene for context
    if context_base64:
        user_content.append({
            "type": "text",
            "text": "Here is the full scene image for additional context:",
        })
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{context_base64}",
            },
        })

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _CLASSIFICATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": 2048,
        "temperature": 0.1,
    }

    logger.info("Calling VLM endpoint %s for classification …", url)

    response = requests.post(url, headers=headers, json=payload, timeout=120)
    response.raise_for_status()

    result = response.json()
    content = result["choices"][0]["message"]["content"]

    # Parse JSON from the response (handle markdown code blocks)
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
    if json_match:
        content = json_match.group(1).strip()

    json_obj_match = re.search(r"\{[\s\S]*\}", content)
    if json_obj_match:
        content = json_obj_match.group(0)

    parsed = json.loads(content)
    return parsed


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _vlm_classify_single_frame(
    pil_image: PILImage.Image,
    bbox: list[int],
    target_label: str,
    include_context: bool,
    endpoint: str,
    model: str,
    api_key: str,
    frame_index: int | None = None,
) -> tuple[Image, dict[str, Any]]:
    """Run VLM classification on a single PIL image."""
    suffix = f"_frame{frame_index}" if frame_index is not None else ""
    w, h = pil_image.size

    x1 = max(0, min(bbox[0], w - 1))
    y1 = max(0, min(bbox[1], h - 1))
    x2 = max(x1 + 1, min(bbox[2], w))
    y2 = max(y1 + 1, min(bbox[3], h))

    logger.info(
        "VLM classifying target%s, bbox=[%d,%d,%d,%d] (%dx%d) …",
        suffix, x1, y1, x2, y2, w, h,
    )

    crop = pil_image.crop((x1, y1, x2, y2))

    timestamp = int(time.time() * 1000)
    crop_path = _SAVE_DIR / f"{timestamp}{suffix}_crop.jpg"
    crop.save(crop_path, format="JPEG", quality=90)

    crop_buffer = BytesIO()
    crop.save(crop_buffer, format="JPEG", quality=85)
    crop_base64 = base64.b64encode(crop_buffer.getvalue()).decode("utf-8")

    context_base64 = None
    if include_context:
        ctx_buffer = BytesIO()
        pil_image.save(ctx_buffer, format="JPEG", quality=70)
        context_base64 = base64.b64encode(ctx_buffer.getvalue()).decode("utf-8")

    try:
        classification = _call_vlm_classify(
            crop_base64, context_base64, target_label, endpoint, model, api_key,
        )
    except Exception as exc:
        logger.error("VLM classification failed%s: %s", suffix, exc)
        raise RuntimeError(f"VLM target classification failed: {exc}") from exc

    classification["bbox"] = [x1, y1, x2, y2]
    if target_label:
        classification["detection_label"] = target_label
    if frame_index is not None:
        classification["frame_index"] = frame_index

    logger.info(
        "VLM classification%s: %s (%s) – threat: %s, confidence: %.4f",
        suffix,
        classification.get("label"),
        classification.get("sub_type"),
        classification.get("threat_level"),
        classification.get("confidence", 0),
    )

    annotated_image = _draw_classification_overlay(
        pil_image, [x1, y1, x2, y2], classification,
    )

    annotated_path = _SAVE_DIR / f"{timestamp}{suffix}_vlm_classified.jpg"
    annotated_image.save(annotated_path, format="JPEG", quality=90)

    buffer = BytesIO()
    annotated_image.save(buffer, format="JPEG", quality=85)
    mcp_image = Image(data=buffer.getvalue(), format="jpeg")

    return mcp_image, classification


def vlm_classify_target_in_image(
    image_source: str,
    bbox: list[int],
    target_label: str = "",
    include_context: bool = True,
) -> tuple[Image | list[Image], dict[str, Any] | list[dict[str, Any]]]:
    """
    Classify a target region within an image or video using a VLM.

    Accepts media as either a URL or base64-encoded string.
    If the source is a video, the same ``bbox`` is applied to every
    extracted frame and a classification is returned for each.

    Parameters
    ----------
    image_source : str
        Either:
        - A URL (http:// or https://) pointing to an image or video
        - A base64-encoded string (with or without data URL prefix)
    bbox : list[int]
        Bounding box coordinates [x1, y1, x2, y2].
    target_label : str, optional
        An optional label hint from the detection stage.
    include_context : bool, optional
        Whether to send the full scene alongside the crop (default True).

    Returns
    -------
    tuple[Image | list[Image], dict | list[dict]]
        - For images: a single annotated MCP Image and classification dict.
        - For videos: lists of annotated MCP Images and classification dicts.

    Raises
    ------
    ValueError
        If the source is invalid or the bbox is invalid.
    RuntimeError
        If the VLM endpoint is not configured or the call fails.
    """
    if len(bbox) != 4:
        raise ValueError(
            f"bbox must be a list of 4 integers [x1, y1, x2, y2], got {bbox}"
        )

    endpoint, model, api_key = _get_vlm_config()
    if not endpoint or not model:
        raise RuntimeError(
            "VLM endpoint not configured. Set VLM_ENDPOINT and VLM_MODEL "
            "(or TOOL_LLM_URL and TOOL_LLM_NAME) environment variables."
        )

    media_type, frames = resolve_media_source(image_source)

    if media_type == "image":
        mcp_image, classification = _vlm_classify_single_frame(
            frames[0], bbox, target_label, include_context,
            endpoint, model, api_key,
        )
        return mcp_image, classification

    # Video — classify the bbox region in each frame
    mcp_images: list[Image] = []
    classifications: list[dict[str, Any]] = []

    for i, frame in enumerate(frames):
        mcp_img, cls = _vlm_classify_single_frame(
            frame, bbox, target_label, include_context,
            endpoint, model, api_key, frame_index=i,
        )
        mcp_images.append(mcp_img)
        classifications.append(cls)

    logger.info(
        "VLM video classification complete: %d frame(s).", len(frames),
    )

    return mcp_images, classifications


def vlm_classify_all_targets(
    image_source: str,
    detections: list[dict[str, Any]],
    include_context: bool = True,
) -> tuple[Image | list[Image], list[dict[str, Any]]]:
    """
    Classify all detected targets in an image or video using a VLM.

    Accepts media as either a URL or base64-encoded string.
    For videos, iterates over each frame and classifies all detections in
    each frame.

    Parameters
    ----------
    image_source : str
        Either:
        - A URL (http:// or https://) pointing to an image or video
        - A base64-encoded string (with or without data URL prefix)
    detections : list[dict]
        A list of detection dicts, each containing at minimum a ``bbox``
        key with [x1, y1, x2, y2] coordinates.
    include_context : bool, optional
        Whether to send the full scene alongside each crop (default True).

    Returns
    -------
    tuple[Image | list[Image], list[dict]]
        - For images: a single annotated MCP Image.
        - For videos: a list of annotated MCP Images (one per frame).
        - A list of detailed classification dicts.

    Raises
    ------
    ValueError
        If the source is invalid.
    RuntimeError
        If the VLM endpoint is not configured or a call fails.
    """
    endpoint, model, api_key = _get_vlm_config()
    if not endpoint or not model:
        raise RuntimeError(
            "VLM endpoint not configured. Set VLM_ENDPOINT and VLM_MODEL "
            "(or TOOL_LLM_URL and TOOL_LLM_NAME) environment variables."
        )

    media_type, frames = resolve_media_source(image_source)

    all_classifications: list[dict[str, Any]] = []
    mcp_images: list[Image] = []

    for frame_idx, pil_image in enumerate(frames):
        w, h = pil_image.size

        # Encode full scene once per frame for context
        context_base64 = None
        if include_context:
            ctx_buffer = BytesIO()
            pil_image.save(ctx_buffer, format="JPEG", quality=70)
            context_base64 = base64.b64encode(ctx_buffer.getvalue()).decode("utf-8")

        annotated_image = pil_image.copy()

        for det in detections:
            bbox = det.get("bbox", [0, 0, w, h])
            label = det.get("label", "")

            x1 = max(0, min(bbox[0], w - 1))
            y1 = max(0, min(bbox[1], h - 1))
            x2 = max(x1 + 1, min(bbox[2], w))
            y2 = max(y1 + 1, min(bbox[3], h))

            crop = pil_image.crop((x1, y1, x2, y2))
            crop_buffer = BytesIO()
            crop.save(crop_buffer, format="JPEG", quality=85)
            crop_base64 = base64.b64encode(crop_buffer.getvalue()).decode("utf-8")

            try:
                classification = _call_vlm_classify(
                    crop_base64, context_base64, label, endpoint, model, api_key,
                )
            except Exception as exc:
                logger.warning("VLM classification failed for bbox %s: %s", bbox, exc)
                classification = {
                    "domain": "unknown",
                    "category": "unclassified",
                    "sub_type": "unclassified",
                    "label": "classification_failed",
                    "threat_level": "unknown",
                    "confidence": 0.0,
                    "error": str(exc),
                }

            classification["bbox"] = [x1, y1, x2, y2]
            if label:
                classification["detection_label"] = label
            if media_type == "video":
                classification["frame_index"] = frame_idx

            all_classifications.append(classification)

            annotated_image = _draw_classification_overlay(
                annotated_image, [x1, y1, x2, y2], classification,
            )

        timestamp = int(time.time() * 1000)
        suffix = f"_frame{frame_idx}" if media_type == "video" else ""
        annotated_path = _SAVE_DIR / f"{timestamp}{suffix}_vlm_all_classified.jpg"
        annotated_image.save(annotated_path, format="JPEG", quality=90)

        buffer = BytesIO()
        annotated_image.save(buffer, format="JPEG", quality=85)
        mcp_images.append(Image(data=buffer.getvalue(), format="jpeg"))

    logger.info(
        "VLM classified %d target(s) across %d frame(s).",
        len(all_classifications), len(frames),
    )

    if media_type == "image":
        return mcp_images[0], all_classifications
    return mcp_images, all_classifications
