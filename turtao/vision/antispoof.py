from __future__ import annotations

import logging

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    mp = None

logger = logging.getLogger(__name__)

OVERLAP_THRESHOLD = 0.35


from typing import Any


class AntiSpoofDetector:
    def __init__(self) -> None:
        self._hands: Any | None = None
        if mp is None:
            logger.warning("MediaPipe not available, anti-spoofing disabled")
            return
        try:
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            logger.info("MediaPipe Hands initialized for anti-spoof")
        except Exception as e:
            logger.error("Failed to initialize MediaPipe Hands: %s", e)

    def check_occlusion(
        self, frame: np.ndarray, face_bbox: tuple[int, int, int, int]
    ) -> bool:
        """Returns True if hand occludes face (>35% overlap)."""
        if self._hands is None:
            return False

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)

        if not results.multi_hand_landmarks:
            return False

        h, w, _ = frame.shape
        fx, fy, fr, fb = face_bbox
        face_area = max(1, (fr - fx) * (fb - fy))

        for hand_landmarks in results.multi_hand_landmarks:
            xs = [lm.x for lm in hand_landmarks.landmark]
            ys = [lm.y for lm in hand_landmarks.landmark]
            hx = int(min(xs) * w)
            hy = int(min(ys) * h)
            hr = int(max(xs) * w)
            hb = int(max(ys) * h)

            # Compute intersection
            ix = max(fx, hx)
            iy = max(fy, hy)
            ir = min(fr, hr)
            ib = min(fb, hb)

            if ix < ir and iy < ib:
                inter_area = (ir - ix) * (ib - iy)
                overlap = inter_area / face_area
                if overlap > OVERLAP_THRESHOLD:
                    logger.info("Hand occlusion detected: %.2f overlap", overlap)
                    return True

        return False

    def release(self) -> None:
        if self._hands is not None:
            self._hands.close()
            self._hands = None
