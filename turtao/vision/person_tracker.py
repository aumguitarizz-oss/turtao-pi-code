from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
import onnxruntime
import supervision as sv
from supervision.tools.byte_tracker import ByteTrack

from turtao.state import AppState, ThreatLabel

logger = logging.getLogger(__name__)

COCO_PERSON = 0
COCO_CELL_PHONE = 67
DETECTION_INTERVAL = 3
PHONE_FACE_DISTANCE_THRESHOLD = 100  # pixels


class PersonTracker:
    def __init__(self, model_path: str = "models/yolov8n.onnx") -> None:
        self._session: onnxruntime.InferenceSession | None = None
        self._tracker = ByteTrack()
        self._active = False
        self._frame_count = 0
        self._last_persons: list[dict[str, Any]] = []
        try:
            self._session = onnxruntime.InferenceSession(
                model_path, providers=["CPUExecutionProvider"]
            )
            logger.info("YOLOv8 ONNX session loaded from %s", model_path)
        except Exception as e:
            logger.error("Failed to load ONNX model %s: %s", model_path, e)

    def set_active(self, active: bool) -> None:
        self._active = active

    def process_frame(self, frame: np.ndarray) -> list[dict[str, Any]]:
        if not self._active or self._session is None:
            return self._last_persons

        self._frame_count += 1
        if self._frame_count % DETECTION_INTERVAL != 0:
            return self._last_persons

        detections = self._infer(frame)
        tracked = self._tracker.update_with_detections(detections)

        persons: list[dict[str, Any]] = []
        phones: list[tuple[int, int, int, int]] = []

        for i in range(len(tracked)):
            cls_id = int(tracked.class_id[i])
            x1, y1, x2, y2 = map(int, tracked.xyxy[i])
            confidence = float(tracked.confidence[i])
            tracker_id = int(tracked.tracker_id[i]) if tracked.tracker_id[i] is not None else -1

            if cls_id == COCO_PERSON:
                persons.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": confidence,
                    "tracker_id": tracker_id,
                })
            elif cls_id == COCO_CELL_PHONE:
                phones.append((x1, y1, x2, y2))

        if phones and persons:
            self._check_phone_spoof(persons, phones)

        self._last_persons = persons
        return persons

    def _infer(self, frame: np.ndarray) -> sv.Detections:
        input_blob = self._preprocess(frame)
        outputs = self._session.run(None, {"images": input_blob})
        return sv.Detections.from_yolov8(outputs)

    @staticmethod
    def _preprocess(frame: np.ndarray) -> np.ndarray:
        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, (640, 640), swapRB=True, crop=False
        )
        return blob.astype(np.float32)

    def _check_phone_spoof(
        self,
        persons: list[dict[str, Any]],
        phones: list[tuple[int, int, int, int]],
    ) -> None:
        for person in persons:
            px1, py1, px2, py2 = person["bbox"]
            face_top = py1
            face_bottom = py1 + (py2 - py1) // 3
            face_left = px1
            face_right = px2
            for ph_x1, ph_y1, ph_x2, ph_y2 in phones:
                ph_cx = (ph_x1 + ph_x2) / 2
                ph_cy = (ph_y1 + ph_y2) / 2
                if (
                    face_left - PHONE_FACE_DISTANCE_THRESHOLD
                    <= ph_cx
                    <= face_right + PHONE_FACE_DISTANCE_THRESHOLD
                    and face_top - PHONE_FACE_DISTANCE_THRESHOLD
                    <= ph_cy
                    <= face_bottom + PHONE_FACE_DISTANCE_THRESHOLD
                ):
                    logger.warning(
                        "Phone spoof detected near person tracker_id=%d",
                        person["tracker_id"],
                    )
