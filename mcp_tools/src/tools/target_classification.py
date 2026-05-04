"""
Target classification tool for MCP.

Classifies a detected target region within an image into detailed
categories.  The tool accepts an image source (URL or base64) and
bounding-box coordinates (from the target_detection tool) and returns
a structured classification report including target type, sub-type,
threat level, and domain.

Accepts media as either:
- A URL (http:// or https://) pointing to an image or video
- A base64-encoded string (with or without data URL prefix)

When a real classification model is available (ResNet / EfficientNet with
custom weights) it is used.  Otherwise the tool falls back to a rule-based
stub that produces plausible classifications so the MCP pipeline can be
exercised end-to-end.
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

logger = logging.getLogger("mcp-target-classification")

# Directory where cropped target images are saved for verification
_SAVE_DIR = Path("saved_images/target_classification")
_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Classification model loading (lazy singleton)
# ---------------------------------------------------------------------------
_classifier_model = None
_classifier_transforms = None
_USE_MODEL = os.environ.get("USE_CLASSIFIER", "true").lower() in ("1", "true", "yes")

# ---------------------------------------------------------------------------
# Target taxonomy
# ---------------------------------------------------------------------------

# Hierarchical target taxonomy: domain → category → sub-types
TARGET_TAXONOMY: dict[str, dict[str, list[str]]] = {
    "land": {
        "armoured_vehicle": [
            "main_battle_tank", "infantry_fighting_vehicle",
            "armoured_personnel_carrier", "self_propelled_gun",
            "armoured_reconnaissance",
        ],
        "soft_vehicle": [
            "military_truck", "utility_vehicle", "MRAP",
            "command_vehicle", "fuel_tanker",
        ],
        "artillery": [
            "towed_howitzer", "self_propelled_howitzer",
            "multiple_rocket_launcher", "mortar_system",
        ],
        "air_defence": [
            "SAM_launcher", "MANPADS_team", "anti_aircraft_gun",
            "radar_system", "C2_vehicle",
        ],
        "personnel": [
            "infantry_squad", "special_forces", "support_crew",
            "observation_post",
        ],
        "infrastructure": [
            "bridge", "bunker", "command_post", "supply_depot",
            "communications_tower", "checkpoint",
        ],
    },
    "maritime": {
        "surface_combatant": [
            "destroyer", "frigate", "corvette", "patrol_vessel",
            "missile_boat",
        ],
        "submarine": [
            "attack_submarine", "ballistic_missile_submarine",
            "midget_submarine",
        ],
        "auxiliary_vessel": [
            "supply_ship", "tanker", "landing_craft",
            "mine_countermeasure_vessel",
        ],
        "civilian_vessel": [
            "cargo_ship", "container_ship", "fishing_vessel",
            "passenger_ferry", "yacht",
        ],
    },
    "aerial": {
        "fixed_wing": [
            "fighter_aircraft", "bomber", "transport_aircraft",
            "surveillance_aircraft", "tanker_aircraft",
        ],
        "rotary_wing": [
            "attack_helicopter", "transport_helicopter",
            "utility_helicopter", "reconnaissance_helicopter",
        ],
        "UAV": [
            "tactical_UAV", "MALE_UAV", "HALE_UAV",
            "loitering_munition", "micro_UAV",
        ],
    },
}

# Threat-level mapping by category
_THREAT_LEVELS: dict[str, str] = {
    "armoured_vehicle": "high",
    "artillery": "high",
    "air_defence": "critical",
    "surface_combatant": "high",
    "submarine": "critical",
    "fixed_wing": "high",
    "rotary_wing": "medium",
    "UAV": "medium",
    "soft_vehicle": "low",
    "personnel": "low",
    "infrastructure": "medium",
    "auxiliary_vessel": "low",
    "civilian_vessel": "minimal",
}


def _get_classifier():
    """Lazily load a classification model (torchvision ResNet)."""
    global _classifier_model, _classifier_transforms
    if _classifier_model is not None:
        return _classifier_model, _classifier_transforms

    try:
        import torch
        import torchvision.models as models
        import torchvision.transforms as transforms

        # Check for custom weights first, fall back to pretrained ImageNet
        custom_weights = os.environ.get("CLASSIFIER_WEIGHTS", "")
        if custom_weights and os.path.isfile(custom_weights):
            logger.info("Loading custom classifier weights from %s", custom_weights)
            _classifier_model = models.resnet50()
            _classifier_model.load_state_dict(torch.load(custom_weights, map_location="cpu"))
        else:
            logger.info("Loading pretrained ResNet50 for feature extraction …")
            _classifier_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        _classifier_model.eval()

        _classifier_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        logger.info("Classification model loaded successfully.")
        return _classifier_model, _classifier_transforms

    except Exception as exc:
        logger.warning(
            "Failed to load classification model: %s – falling back to stub.", exc,
        )
        return None, None


# ---------------------------------------------------------------------------
# Model-based classification
# ---------------------------------------------------------------------------

# ImageNet class-index → target mapping (selected relevant classes)
_IMAGENET_TO_TARGET: dict[int, tuple[str, str, str, str]] = {
    # (domain, category, sub_type, label)
    # Vehicles
    407: ("land", "soft_vehicle", "military_truck", "ambulance"),
    468: ("land", "soft_vehicle", "utility_vehicle", "cab"),
    555: ("land", "armoured_vehicle", "armoured_personnel_carrier", "fire_engine"),
    569: ("land", "soft_vehicle", "fuel_tanker", "garbage_truck"),
    609: ("land", "soft_vehicle", "utility_vehicle", "jeep"),
    654: ("land", "soft_vehicle", "military_truck", "minibus"),
    675: ("land", "soft_vehicle", "command_vehicle", "moving_van"),
    717: ("land", "soft_vehicle", "utility_vehicle", "pickup"),
    734: ("land", "soft_vehicle", "military_truck", "police_van"),
    751: ("aerial", "fixed_wing", "transport_aircraft", "racer"),
    817: ("land", "soft_vehicle", "military_truck", "sports_car"),
    864: ("land", "soft_vehicle", "military_truck", "tow_truck"),
    867: ("land", "armoured_vehicle", "main_battle_tank", "trailer_truck"),
    # Aircraft
    404: ("aerial", "fixed_wing", "fighter_aircraft", "airliner"),
    895: ("aerial", "fixed_wing", "fighter_aircraft", "warplane"),
    # Maritime
    472: ("maritime", "civilian_vessel", "cargo_ship", "canoe"),
    510: ("maritime", "surface_combatant", "patrol_vessel", "container_ship"),
    554: ("maritime", "auxiliary_vessel", "supply_ship", "fireboat"),
    625: ("maritime", "civilian_vessel", "cargo_ship", "lifeboat"),
    628: ("maritime", "civilian_vessel", "passenger_ferry", "liner"),
    724: ("maritime", "civilian_vessel", "fishing_vessel", "pirate"),
    780: ("maritime", "civilian_vessel", "yacht", "schooner"),
    814: ("maritime", "civilian_vessel", "cargo_ship", "speedboat"),
    871: ("maritime", "surface_combatant", "destroyer", "trimaran"),
    914: ("maritime", "auxiliary_vessel", "supply_ship", "yawl"),
    # Military specific
    413: ("land", "armoured_vehicle", "main_battle_tank", "assault_rifle"),
    474: ("land", "artillery", "towed_howitzer", "cannon"),
    # Infrastructure
    460: ("land", "infrastructure", "bridge", "bridge"),
    483: ("land", "infrastructure", "bunker", "castle"),
    497: ("land", "infrastructure", "command_post", "church"),
    698: ("land", "infrastructure", "supply_depot", "palace"),
    900: ("land", "infrastructure", "communications_tower", "water_tower"),
}


def _classify_with_model(
    crop: PILImage.Image,
) -> dict[str, Any]:
    """Classify a cropped target image using the loaded model."""
    import torch

    model, transform = _get_classifier()
    if model is None:
        return _classify_stub(crop)

    input_tensor = transform(crop).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Get top-5 predictions
    top5_prob, top5_idx = torch.topk(probabilities, 5)

    # Try to map the top prediction to our target taxonomy
    for prob, idx in zip(top5_prob, top5_idx):
        class_idx = idx.item()
        confidence = prob.item()

        if class_idx in _IMAGENET_TO_TARGET:
            domain, category, sub_type, label = _IMAGENET_TO_TARGET[class_idx]
            threat_level = _THREAT_LEVELS.get(category, "unknown")

            return {
                "domain": domain,
                "category": category,
                "sub_type": sub_type,
                "label": label,
                "threat_level": threat_level,
                "confidence": round(confidence, 4),
                "model": "resnet50",
                "top5_classes": [
                    {"class_id": i.item(), "confidence": round(p.item(), 4)}
                    for p, i in zip(top5_prob, top5_idx)
                ],
            }

    # If no mapped class found, return best guess with unknown mapping
    best_idx = top5_idx[0].item()
    best_conf = top5_prob[0].item()

    return {
        "domain": "unknown",
        "category": "unclassified",
        "sub_type": "unclassified",
        "label": f"imagenet_class_{best_idx}",
        "threat_level": "unknown",
        "confidence": round(best_conf, 4),
        "model": "resnet50",
        "top5_classes": [
            {"class_id": i.item(), "confidence": round(p.item(), 4)}
            for p, i in zip(top5_prob, top5_idx)
        ],
    }


# ---------------------------------------------------------------------------
# Stub classification (fallback)
# ---------------------------------------------------------------------------

import random


def _classify_stub(crop: PILImage.Image) -> dict[str, Any]:
    """Generate a plausible classification result (stub)."""
    # Pick a random domain and category
    domain = random.choice(list(TARGET_TAXONOMY.keys()))
    category = random.choice(list(TARGET_TAXONOMY[domain].keys()))
    sub_type = random.choice(TARGET_TAXONOMY[domain][category])
    threat_level = _THREAT_LEVELS.get(category, "unknown")
    confidence = round(random.uniform(0.55, 0.95), 4)

    return {
        "domain": domain,
        "category": category,
        "sub_type": sub_type,
        "label": sub_type.replace("_", " "),
        "threat_level": threat_level,
        "confidence": confidence,
        "model": "stub",
    }


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _get_font(size: int = 14) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size,
        )
    except (IOError, OSError):
        return ImageFont.load_default()


_THREAT_COLOURS: dict[str, str] = {
    "critical": "#FF0000",
    "high": "#FF6600",
    "medium": "#FFCC00",
    "low": "#00CC00",
    "minimal": "#00AAFF",
    "unknown": "#888888",
}


def _draw_classification_overlay(
    image: PILImage.Image,
    bbox: list[int],
    classification: dict[str, Any],
) -> PILImage.Image:
    """Draw the classification result on the image with a highlighted bbox."""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    font = _get_font(14)
    font_small = _get_font(12)

    x1, y1, x2, y2 = bbox
    threat = classification.get("threat_level", "unknown")
    colour = _THREAT_COLOURS.get(threat, "#888888")

    # Draw bounding box with threat-level colour
    draw.rectangle([x1, y1, x2, y2], outline=colour, width=4)

    # Build label lines
    lines = [
        f"{classification.get('label', 'unknown')} ({classification.get('confidence', 0):.2f})",
        f"Type: {classification.get('category', '?')} / {classification.get('sub_type', '?')}",
        f"Domain: {classification.get('domain', '?')}",
        f"Threat: {threat.upper()}",
    ]

    # Draw label background
    line_height = 18
    total_height = line_height * len(lines) + 8
    max_width = max(draw.textlength(line, font=font_small) for line in lines) + 8

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
            font=font_small,
        )

    return annotated


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _classify_single_frame(
    pil_image: PILImage.Image,
    bbox: list[int],
    target_label: str = "",
    frame_index: int | None = None,
) -> tuple[Image, dict[str, Any]]:
    """Classify a target region in a single PIL image."""
    suffix = f"_frame{frame_index}" if frame_index is not None else ""
    w, h = pil_image.size

    # Clamp bbox to image bounds
    x1 = max(0, min(bbox[0], w - 1))
    y1 = max(0, min(bbox[1], h - 1))
    x2 = max(x1 + 1, min(bbox[2], w))
    y2 = max(y1 + 1, min(bbox[3], h))

    logger.info(
        "Classifying target%s, bbox=[%d,%d,%d,%d] (%dx%d) …",
        suffix, x1, y1, x2, y2, w, h,
    )

    crop = pil_image.crop((x1, y1, x2, y2))

    timestamp = int(time.time() * 1000)
    crop_path = _SAVE_DIR / f"{timestamp}{suffix}_crop.jpg"
    crop.save(crop_path, format="JPEG", quality=90)

    if _USE_MODEL:
        classification = _classify_with_model(crop)
    else:
        classification = _classify_stub(crop)

    classification["bbox"] = [x1, y1, x2, y2]
    if target_label:
        classification["detection_label"] = target_label
    if frame_index is not None:
        classification["frame_index"] = frame_index

    logger.info(
        "Classification result%s: %s (%s) – threat: %s, confidence: %.4f",
        suffix,
        classification.get("label"),
        classification.get("sub_type"),
        classification.get("threat_level"),
        classification.get("confidence", 0),
    )

    annotated_image = _draw_classification_overlay(pil_image, [x1, y1, x2, y2], classification)

    annotated_path = _SAVE_DIR / f"{timestamp}{suffix}_classified.jpg"
    annotated_image.save(annotated_path, format="JPEG", quality=90)

    buffer = BytesIO()
    annotated_image.save(buffer, format="JPEG", quality=85)
    mcp_image = Image(data=buffer.getvalue(), format="jpeg")

    return mcp_image, classification


def classify_target_in_image(
    image_source: str,
    bbox: list[int],
    target_label: str = "",
) -> tuple[Image | list[Image], dict[str, Any] | list[dict[str, Any]]]:
    """
    Classify a target region within an image or video.

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
        Bounding box coordinates [x1, y1, x2, y2] identifying the target
        region.  Typically obtained from the target_detection tool.
    target_label : str, optional
        An optional label hint from the detection stage (e.g. "truck").

    Returns
    -------
    tuple[Image | list[Image], dict | list[dict]]
        - For images: a single annotated MCP Image and classification dict.
        - For videos: a list of annotated MCP Images and classification dicts.

    Raises
    ------
    ValueError
        If the source is invalid or the bbox is invalid.
    """
    if len(bbox) != 4:
        raise ValueError(
            f"bbox must be a list of 4 integers [x1, y1, x2, y2], got {bbox}"
        )

    media_type, frames = resolve_media_source(image_source)

    if media_type == "image":
        mcp_image, classification = _classify_single_frame(
            frames[0], bbox, target_label,
        )
        return mcp_image, classification

    # Video — classify the bbox region in each frame
    mcp_images: list[Image] = []
    classifications: list[dict[str, Any]] = []

    for i, frame in enumerate(frames):
        mcp_img, cls = _classify_single_frame(
            frame, bbox, target_label, frame_index=i,
        )
        mcp_images.append(mcp_img)
        classifications.append(cls)

    logger.info(
        "Video classification complete: %d frame(s).", len(frames),
    )

    return mcp_images, classifications


def classify_all_targets(
    image_source: str,
    detections: list[dict[str, Any]],
) -> tuple[Image | list[Image], list[dict[str, Any]]]:
    """
    Classify all detected targets in an image or video.

    Accepts media as either a URL or base64-encoded string.
    For images, iterates over detections and classifies each one.
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

    Returns
    -------
    tuple[Image | list[Image], list[dict]]
        - For images: a single annotated MCP Image.
        - For videos: a list of annotated MCP Images (one per frame).
        - A list of classification dicts (one per detection per frame).

    Raises
    ------
    ValueError
        If the source is invalid.
    """
    media_type, frames = resolve_media_source(image_source)

    all_classifications: list[dict[str, Any]] = []
    mcp_images: list[Image] = []

    for frame_idx, pil_image in enumerate(frames):
        w, h = pil_image.size
        annotated_image = pil_image.copy()

        for det in detections:
            bbox = det.get("bbox", [0, 0, w, h])
            label = det.get("label", "")

            x1 = max(0, min(bbox[0], w - 1))
            y1 = max(0, min(bbox[1], h - 1))
            x2 = max(x1 + 1, min(bbox[2], w))
            y2 = max(y1 + 1, min(bbox[3], h))

            crop = pil_image.crop((x1, y1, x2, y2))

            if _USE_MODEL:
                classification = _classify_with_model(crop)
            else:
                classification = _classify_stub(crop)

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
        annotated_path = _SAVE_DIR / f"{timestamp}{suffix}_all_classified.jpg"
        annotated_image.save(annotated_path, format="JPEG", quality=90)

        buffer = BytesIO()
        annotated_image.save(buffer, format="JPEG", quality=85)
        mcp_images.append(Image(data=buffer.getvalue(), format="jpeg"))

    logger.info(
        "Classified %d target(s) across %d frame(s).",
        len(all_classifications), len(frames),
    )

    if media_type == "image":
        return mcp_images[0], all_classifications
    return mcp_images, all_classifications
