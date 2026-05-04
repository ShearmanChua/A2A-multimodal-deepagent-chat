"""
Object detection tool (stub).

Accepts an image_id (from the image store), draws random bounding boxes with
labels, and returns the annotated image as an MCP-native Image object.
"""

import logging
import random
import time
from io import BytesIO
from pathlib import Path

from PIL import Image as PILImage, ImageDraw, ImageFont
from fastmcp.utilities.types import Image  # MCP Image type

from tools.image_store import get_image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)

logger = logging.getLogger("mcp-object-detection")

# Directory where received and annotated images are saved for verification
_SAVE_DIR = Path("saved_images")
_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Fake class labels used for the random detections
_FAKE_CLASSES = [
    "person", "car", "truck", "bicycle", "dog", "cat",
    "bird", "boat", "airplane", "tree", "building", "bus",
]

# Colours for bounding boxes (one per box, cycled)
_BOX_COLOURS = ["red", "lime", "blue", "yellow", "cyan", "magenta", "orange"]


def _draw_random_detections(
    image: PILImage.Image,
    num_boxes: int = 3,
) -> tuple[PILImage.Image, list[dict]]:
    """
    Draw *num_boxes* random bounding boxes on *image* and return the
    annotated image together with fake detection metadata.
    """
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    # Try to use a nicer font; fall back to the default bitmap font.
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except (IOError, OSError):
        font = ImageFont.load_default()

    w, h = image.size
    detections: list[dict] = []

    for i in range(num_boxes):
        # Generate a random box that is at least 10 % of the image in each dim
        min_side_w = max(int(w * 0.10), 20)
        min_side_h = max(int(h * 0.10), 20)

        x1 = random.randint(0, max(w - min_side_w - 1, 0))
        y1 = random.randint(0, max(h - min_side_h - 1, 0))
        x2 = random.randint(x1 + min_side_w, min(x1 + int(w * 0.5), w))
        y2 = random.randint(y1 + min_side_h, min(y1 + int(h * 0.5), h))

        class_name = random.choice(_FAKE_CLASSES)
        confidence = round(random.uniform(0.50, 0.99), 2)
        colour = _BOX_COLOURS[i % len(_BOX_COLOURS)]

        label = f"{class_name} {confidence:.2f}"

        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=colour, width=3)

        # Draw label background + text
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        draw.rectangle(
            [text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2],
            fill=colour,
        )
        draw.text((x1, y1), label, fill="white", font=font)

        detections.append(
            {
                "class": class_name,
                "confidence": confidence,
                "bbox": [x1, y1, x2, y2],
            }
        )

    return annotated, detections


def detect_objects_in_image(
    image_id: str,
    num_boxes: int = 3,
) -> tuple[Image, list[dict]]:
    """
    Draw random bounding boxes on a stored image (stub for real object
    detection).

    Parameters
    ----------
    image_id : str
        The ID of a previously uploaded image (from the image store).
    num_boxes : int, optional
        Number of random bounding boxes to draw (default 3).

    Returns
    -------
    tuple[Image, list[dict]]
        - The annotated MCP Image with bounding boxes drawn.
        - A list of detection dicts each containing ``class``, ``confidence``,
          and ``bbox`` ([x1, y1, x2, y2]).

    Raises
    ------
    ValueError
        If the ``image_id`` is not found in the store.
    """
    # Retrieve image bytes from the store
    image_bytes = get_image(image_id)
    if image_bytes is None:
        raise ValueError(f"Image with id '{image_id}' not found in the store. Upload it first using the upload_image tool.")

    pil_image = PILImage.open(BytesIO(image_bytes)).convert("RGB")

    logger.info(
        "Running (stub) object detection on image %s (%dx%d) …",
        image_id,
        pil_image.width,
        pil_image.height,
    )

    # Save the received (original) image for verification
    timestamp = int(time.time() * 1000)
    original_path = _SAVE_DIR / f"{timestamp}_{image_id}_original.jpg"
    pil_image.save(original_path, format="JPEG", quality=90)
    logger.info("Saved received image to %s", original_path)

    annotated_image, detections = _draw_random_detections(pil_image, num_boxes)

    logger.info("Generated %d random detection(s).", len(detections))

    # Save the annotated image for verification
    annotated_path = _SAVE_DIR / f"{timestamp}_{image_id}_annotated.jpg"
    annotated_image.save(annotated_path, format="JPEG", quality=90)
    logger.info("Saved annotated image to %s", annotated_path)

    # Encode the annotated image as JPEG bytes for MCP transport
    buffer = BytesIO()
    annotated_image.save(buffer, format="JPEG", quality=85)
    annotated_bytes = buffer.getvalue()

    mcp_image = Image(data=annotated_bytes, format="jpeg")

    return mcp_image, detections
