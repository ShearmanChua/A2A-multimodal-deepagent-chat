from fastmcp import FastMCP
from tools.duckduckgo import search_duckduckgo, search_duckduckgo_images
from tools.target_detection import detect_targets_in_image
from tools.target_classification import classify_target_in_image
from tools.vlm_target_detection import vlm_detect_targets_in_image
from tools.vlm_target_classification import vlm_classify_target_in_image
from tools.minio_store import list_bucket_objects, get_bucket_object

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
)

logger = logging.getLogger("mcp-search")

# Create MCP server
mcp = FastMCP("search-server")

@mcp.tool(
    name="duckduckgo_search",
    description="Search DuckDuckGo and return max results.",
)
def duckduckgo_search(query: str, max_results: int = 5):
    """
    Search DuckDuckGo and return results.
    """
    results = search_duckduckgo(query, max_results)
    logger.info(f"Search results for {query}: {results}")

    return results

@mcp.tool()
def search_images(query: str, max_results: int = 3):
    """
    Search for images using DuckDuckGo.
    Returns images
    """

    results = search_duckduckgo_images(query, max_results)

    return results


@mcp.tool(
    name="target_detection",
    description="Detect targets (vehicles, vessels, aircraft, infrastructure, personnel) in an image or video using YOLOv8. Pass either a URL (http:// or https://) or a base64-encoded image/video string. For videos, frames are extracted and detection runs on each frame. Returns annotated image(s) with bounding boxes and a list of detected targets with type and confidence.",
)
def target_detection(
    image_source: str,
    confidence_threshold: float = 0.25,
    max_detections: int = 20,
):
    """
    Detect targets in an image or video.

    Args:
        image_source: Either a URL (http:// or https://) or a base64-encoded string (with or without data URL prefix).
        confidence_threshold: Minimum confidence to include a detection (default 0.25).
        max_detections: Maximum number of detections per frame (default 20).

    Returns:
        Annotated image(s) with bounding boxes and detection metadata including
        label, target_type, confidence, bbox, and frame_index (for videos).
    """
    result_images, detections = detect_targets_in_image(
        image_source, confidence_threshold, max_detections,
    )
    logger.info(f"Target detection completed: {len(detections)} target(s) found.")

    if isinstance(result_images, list):
        return [*result_images, {"detections": detections}]
    return [result_images, {"detections": detections}]


@mcp.tool(
    name="target_classification",
    description="Classify a single detected target region within an image or video. Pass either a URL (http:// or https://) or a base64-encoded image/video string, along with the bounding box [x1, y1, x2, y2] from target_detection. For videos, the bbox is applied to each extracted frame. Returns a detailed classification including domain, category, sub-type, threat level, and confidence.",
)
def target_classification(
    image_source: str,
    bbox: list[int],
    target_label: str = "",
):
    """
    Classify a detected target in an image or video.

    Args:
        image_source: Either a URL (http:// or https://) or a base64-encoded string (with or without data URL prefix).
        bbox: Bounding box [x1, y1, x2, y2] from target_detection results.
        target_label: Optional label hint from the detection stage.

    Returns:
        Annotated image(s) with classification overlay and classification
        dict(s) containing domain, category, sub_type, label, threat_level,
        and confidence.
    """
    result_images, classification = classify_target_in_image(
        image_source, bbox, target_label,
    )
    if isinstance(classification, list):
        logger.info(f"Target classification completed: {len(classification)} frame(s) classified.")
        if isinstance(result_images, list):
            return [*result_images, {"classifications": classification}]
        return [result_images, {"classifications": classification}]

    logger.info(
        f"Target classification completed: {classification.get('label')} "
        f"(threat: {classification.get('threat_level')})"
    )

    return [result_images, {"classification": classification}]


@mcp.tool(
    name="vlm_target_detection",
    description="(High-quality) Detect targets in an image or video using a Vision Language Model (VLM). This is slower but significantly more accurate than the YOLO-based target_detection tool. The VLM can reason about context, identify camouflaged or partially occluded targets, and provide natural-language descriptions. Pass either a URL (http:// or https://) or a base64-encoded image/video string. For videos, frames are extracted and detection runs on each frame. Requires a VLM endpoint (VLM_ENDPOINT / TOOL_LLM_URL env var).",
)
def vlm_target_detection_tool(
    image_source: str,
    detection_prompt: str = "",
):
    """
    Detect targets in an image or video using a VLM.

    Args:
        image_source: Either a URL (http:// or https://) or a base64-encoded string (with or without data URL prefix).
        detection_prompt: Optional additional instructions to guide detection
                          (e.g. "focus on maritime vessels" or "look for camouflaged vehicles").

    Returns:
        Annotated image(s) with bounding boxes and detection metadata including
        label, target_type, domain, confidence, bbox, description, and
        frame_index (for videos).
    """
    result_images, detections = vlm_detect_targets_in_image(image_source, detection_prompt)
    logger.info(f"VLM target detection completed: {len(detections)} target(s) found.")

    if isinstance(result_images, list):
        return [*result_images, {"detections": detections}]
    return [result_images, {"detections": detections}]


@mcp.tool(
    name="vlm_target_classification",
    description="(High-quality) Classify a single detected target region using a Vision Language Model (VLM). This is slower but produces much richer classifications than the standard target_classification tool, including NATO designations, observable features, camouflage assessment, operational status, and recommended actions. Pass either a URL (http:// or https://) or a base64-encoded image/video string, along with the bounding box [x1, y1, x2, y2] from any detection tool. For videos, the bbox is applied to each extracted frame. Requires a VLM endpoint (VLM_ENDPOINT / TOOL_LLM_URL env var).",
)
def vlm_target_classification_tool(
    image_source: str,
    bbox: list[int],
    target_label: str = "",
    include_context: bool = True,
):
    """
    Classify a detected target using a Vision Language Model.

    Args:
        image_source: Either a URL (http:// or https://) or a base64-encoded string (with or without data URL prefix).
        bbox: Bounding box [x1, y1, x2, y2] from any detection tool's results.
        target_label: Optional label hint from the detection stage.
        include_context: Whether to send the full scene image for additional context (default True).

    Returns:
        Annotated image(s) with classification overlay and detailed classification
        dict(s) containing domain, category, sub_type, label, nato_designation,
        threat_level, confidence, observable_features, camouflage_assessment,
        operational_status, estimated_heading, description, and recommended_actions.
    """
    result_images, classification = vlm_classify_target_in_image(
        image_source, bbox, target_label, include_context,
    )
    if isinstance(classification, list):
        logger.info(f"VLM target classification completed: {len(classification)} frame(s) classified.")
        if isinstance(result_images, list):
            return [*result_images, {"classifications": classification}]
        return [result_images, {"classifications": classification}]

    logger.info(
        f"VLM target classification completed: {classification.get('label')} "
        f"(threat: {classification.get('threat_level')})"
    )

    return [result_images, {"classification": classification}]

# ---------------------------------------------------------------------------
# MinIO object-storage tools
# ---------------------------------------------------------------------------

@mcp.tool(
    name="list_minio_objects",
    description="List objects in a MinIO bucket. Optionally filter by prefix (folder path). Returns object names, sizes, last-modified timestamps, and content types. Use this to discover what files are available before pulling them with get_minio_object.",
)
def list_minio_objects(
    bucket: str = "",
    prefix: str = "",
    recursive: bool = True,
    max_items: int = 100,
) -> list[dict]:
    """
    List objects in a MinIO bucket.

    Args:
        bucket: Bucket name. Leave empty to use the default bucket (MINIO_BUCKET env var).
        prefix: Only return objects whose key starts with this prefix (e.g. "images/" or "2024/").
        recursive: If True, list recursively through sub-directories (default True).
        max_items: Maximum number of items to return (default 100).

    Returns:
        A list of dicts with name, size, last_modified, and content_type for each object.
    """
    results = list_bucket_objects(
        bucket=bucket or None,
        prefix=prefix,
        recursive=recursive,
        max_items=max_items,
    )
    logger.info(f"MinIO list: {len(results)} object(s) returned.")
    return results


@mcp.tool(
    name="get_minio_object",
    description="Download an object from a MinIO bucket and return it to the agent. Images (JPEG, PNG, GIF, WebP, BMP, TIFF) are returned as inline images the agent can see directly. Non-image files are returned as base64-encoded data with metadata. Use list_minio_objects first to discover available objects.",
)
def get_minio_object(
    object_name: str,
    bucket: str = "",
):
    """
    Download an object from a MinIO bucket.

    Args:
        object_name: The full key / path of the object (e.g. "images/photo.jpg").
        bucket: Bucket name. Leave empty to use the default bucket (MINIO_BUCKET env var).

    Returns:
        For images: the image is returned directly as an inline image.
        For other files: a dict with filename, content_type, size, and data_base64.
    """
    result = get_bucket_object(
        object_name=object_name,
        bucket=bucket or None,
    )
    logger.info(f"MinIO get: retrieved '{object_name}'.")
    return result


if __name__ == "__main__":
    # Start MCP server
    mcp.run(transport="http", host="0.0.0.0", port=8000)
