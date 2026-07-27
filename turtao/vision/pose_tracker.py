import cv2
import logging
import numpy as np
from turtao.state import AppState

logger = logging.getLogger(__name__)

# Import mediapipe safely
try:
    import mediapipe as mp
except ImportError:
    mp = None
    logger.warning("MediaPipe not available. Pose tracking disabled.")


class PoseTracker:
    def __init__(self, state: AppState) -> None:
        self.state = state
        self._pose = None
        if mp is not None:
            try:
                self.mp_pose = mp.solutions.pose
                self._pose = self.mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=0,  # 0 is the fastest model, optimized for mobile/Pi CPUs
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                logger.info("MediaPipe Pose tracker initialized successfully.")
            except Exception as e:
                logger.error("Failed to initialize MediaPipe Pose tracker: %s", e)

    def process_frame(self, frame: np.ndarray) -> None:
        if self._pose is None:
            return

        try:
            # MediaPipe expects RGB images
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._pose.process(rgb)
            
            landmarks = []
            if results.pose_landmarks:
                h, w = frame.shape[:2]
                for lm in results.pose_landmarks.landmark:
                    # Scale normalized coordinates back to pixel space
                    lx = int(lm.x * w)
                    ly = int(lm.y * h)
                    landmarks.append((lx, ly))
            
            with self.state:
                self.state.pose_landmarks = landmarks
        except Exception:
            logger.exception("Error in MediaPipe Pose processing")

    def close(self) -> None:
        if self._pose is not None:
            try:
                self._pose.close()
            except Exception:
                pass
