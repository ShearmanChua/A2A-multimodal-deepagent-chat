"""Video frame extraction and media-type helpers."""

from __future__ import annotations

import base64
from pathlib import Path

import cv2

from multimodal_agent.configs.config import IMAGE_EXTENSIONS, MAX_VIDEO_FRAMES, VIDEO_EXTENSIONS


def sample_video_frames_b64(
    video_path: str,
    n_frames: int = MAX_VIDEO_FRAMES,
) -> list[str]:
    """Sample ``n_frames`` evenly-spaced frames from a video file.

    Returns a list of raw base64 strings (JPEG-encoded, no data-URL prefix).
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    indices = [int(i * total / n_frames) for i in range(n_frames)]
    frames: list[str] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        ok, buf = cv2.imencode(".jpg", frame)
        if ok:
            frames.append(base64.b64encode(buf.tobytes()).decode())

    cap.release()
    return frames


def is_video(path: str) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def is_image(path: str) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS
