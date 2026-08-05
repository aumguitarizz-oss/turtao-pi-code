from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
import onnxruntime
import supervision as sv
from supervision import ByteTrack

logger = logging.getLogger(__name__)

COCO_PERSON = 0
COCO_CELL_PHONE = 67
DETECTION_INTERVAL = 3
PHONE_FACE_DISTANCE_THRESHOLD = 100  # pixels

COCO_CLASSES = [
    "Person", "Bicycle", "Car", "Motorcycle", "Airplane", "Bus", "Train", "Truck", "Boat", "Traffic Light",
    "Fire Hydrant", "Stop Sign", "Parking Meter", "Bench", "Bird", "Cat", "Dog", "Horse", "Sheep", "Cow",
    "Elephant", "Bear", "Zebra", "Giraffe", "Backpack", "Umbrella", "Handbag", "Tie", "Suitcase", "Frisbee",
    "Skis", "Snowboard", "Sports Ball", "Kite", "Baseball Bat", "Baseball Glove", "Skateboard", "Surfboard", "Tennis Racket", "Bottle",
    "Wine Glass", "Cup", "Fork", "Knife", "Spoon", "Bowl", "Banana", "Apple", "Sandwich", "Orange",
    "Broccoli", "Carrot", "Hot Dog", "Pizza", "Donut", "Cake", "Chair", "Couch", "Potted Plant", "Bed",
    "Dining Table", "Toilet", "TV", "Laptop", "Mouse", "Remote", "Keyboard", "Cell Phone", "Microwave", "Oven",
    "Toaster", "Sink", "Refrigerator", "Book", "Clock", "Vase", "Scissors", "Teddy Bear", "Hair Drier", "Toothbrush"
]


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

            class_name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f"Object_{cls_id}"

            persons.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": confidence,
                "tracker_id": tracker_id,
                "class_name": class_name
            })
            
            if cls_id == COCO_CELL_PHONE:
                phones.append((x1, y1, x2, y2))

        if phones and persons:
            self._check_phone_spoof(persons, phones)

        self._last_persons = persons
        return persons

    def _infer(self, frame: np.ndarray) -> sv.Detections:
        input_blob = self._preprocess(frame)
        outputs = self._session.run(None, {"images": input_blob})
        
        # outputs[0] has shape (1, 84, 8400)
        output = outputs[0][0]  # shape (84, 8400)
        output = output.T       # shape (8400, 84)
        
        fh, fw = frame.shape[:2]
        boxes = []
        confidences = []
        class_ids = []
        
        for row in output:
            classes_scores = row[4:]
            class_id = np.argmax(classes_scores)
            confidence = classes_scores[class_id]
            
            # Detect persons (0) and cell phones (67)
            if confidence > 0.25 and class_id in (COCO_PERSON, COCO_CELL_PHONE):
                xc, yc, w, h = row[:4]
                
                # Scale back to original frame resolution (blob input was 640x640)
                x1 = int((xc - w / 2) * (fw / 640.0))
                y1 = int((yc - h / 2) * (fh / 640.0))
                x2 = int((xc + w / 2) * (fw / 640.0))
                y2 = int((yc + h / 2) * (fh / 640.0))
                
                boxes.append([x1, y1, x2, y2])
                confidences.append(float(confidence))
                class_ids.append(int(class_id))
                
        if not boxes:
            return sv.Detections.empty()
            
        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.25, nms_threshold=0.45)
        
        if len(indices) > 0:
            indices = np.array(indices).flatten()
            final_boxes = np.array([boxes[i] for i in indices], dtype=np.float32)
            final_confs = np.array([confidences[i] for i in indices], dtype=np.float32)
            final_classes = np.array([class_ids[i] for i in indices], dtype=np.int32)
            
            return sv.Detections(
                xyxy=final_boxes,
                confidence=final_confs,
                class_id=final_classes
            )
            
        return sv.Detections.empty()

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
            if person.get("class_name") != "Person":
                continue
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
