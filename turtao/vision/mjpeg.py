from __future__ import annotations

import logging
from typing import Any, Callable, Generator

import cv2
import numpy as np

logger = logging.getLogger(__name__)

BOUNDARY = "frame"
JPEG_QUALITY = 70
BBOX_COLOR = (0, 255, 0)
BBOX_THICKNESS = 2


def _encode_jpeg(frame: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ret:
        logger.error("JPEG encoding failed")
        return b""
    return buf.tobytes()


def _draw_bboxes(frame: np.ndarray) -> np.ndarray:
    annotated = frame.copy()
    h, w, _ = annotated.shape
    # Draw center crosshair
    cv2.line(
        annotated, (w // 2 - 15, h // 2), (w // 2 + 15, h // 2), (0, 255, 0), 1
    )
    cv2.line(
        annotated, (w // 2, h // 2 - 15), (w // 2, h // 2 + 15), (0, 255, 0), 1
    )
    return annotated


def generate_mjpeg_stream(
    get_frame: Callable[[], np.ndarray | None],
    quality: int = 70,
) -> Generator[bytes, None, None]:
    """MJPEG multipart stream generator.
    Content-Type: multipart/x-mixed-replace; boundary=frame
    Draws bounding boxes on frame before encoding.
    Handles client disconnect via GeneratorExit."""
    try:
        while True:
            frame = get_frame()
            if frame is None:
                continue

            annotated = _draw_bboxes(frame)
            jpeg_bytes = _encode_jpeg(annotated, quality)
            if not jpeg_bytes:
                continue

            yield (
                b"--" + BOUNDARY.encode() + b"\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(jpeg_bytes)).encode() + b"\r\n"
                b"\r\n"
                + jpeg_bytes
                + b"\r\n"
            )
    except GeneratorExit:
        logger.info("MJPEG client disconnected")
    except Exception as e:
        logger.error("MJPEG stream error: %s", e)
