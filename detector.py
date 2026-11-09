"""
Object Detection Module using Ultralytics YOLOv8.

This module encapsulates loading pre-trained YOLO models, running inference
on image frames, filtering detections by confidence and target classes,
and rendering annotated bounding boxes with labels and confidence scores.
"""

from __future__ import annotations
import os
import logging
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Union

# Suppress noisy Ultralytics logs if desired
os.environ.setdefault("YOLO_VERBOSE", "False")
# Set config directory to avoid permission issues in restricted environments
os.environ.setdefault(
    "YOLO_CONFIG_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", ".config")
)

import cv2
import numpy as np
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("YOLODetector")


@dataclass
class Detection:
    """Represents a single detected object in an image frame."""
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str

    @property
    def xyxy(self) -> List[float]:
        """Return bbox as a list of coordinates [x1, y1, x2, y2]."""
        return list(self.bbox)

    @property
    def width(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(0.0, self.bbox[3] - self.bbox[1])

    @property
    def center(self) -> Tuple[float, float]:
        """Return the (cx, cy) centroid."""
        return (self.bbox[0] + self.bbox[2]) / 2.0, (self.bbox[1] + self.bbox[3]) / 2.0


class YOLODetector:
    """YOLOv8 Object Detector for real-time video streams and images."""

    # Default color palette for visually distinct classes (BGR format for OpenCV)
    PALETTE = [
        (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
        (49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
        (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0),
        (255, 143, 29), (255, 74, 70), (220, 20, 60), (255, 38, 132),
        (205, 0, 205), (138, 43, 226), (75, 0, 130), (0, 0, 255)
    ]

    def __init__(
        self,
        model_path: str = "models/yolov8n.pt",
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        device: Optional[str] = None
    ) -> None:
        """
        Initialize the YOLO object detector.

        Args:
            model_path: Path to the YOLOv8 weights file (default: models/yolov8n.pt).
            conf_threshold: Minimum detection confidence score (0.0 to 1.0).
            iou_threshold: Non-Maximum Suppression (NMS) IoU threshold.
            device: Computing device ('cpu', 'cuda', etc.). Defaults to auto-select.
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.model_path = model_path

        # Resolve weights path
        resolved_path = self._resolve_model_path(model_path)
        logger.info(f"Loading YOLO model from: {resolved_path}")
        self.model = YOLO(resolved_path)

        # Retrieve COCO class dictionary {id: name}
        self.class_names: Dict[int, str] = self.model.names if hasattr(self.model, "names") else {}
        logger.info(f"Loaded model with {len(self.class_names)} COCO classes.")

    @staticmethod
    def _resolve_model_path(model_path: str) -> str:
        """Resolve model path or fallback to base model if missing."""
        if os.path.exists(model_path):
            return model_path

        # Check in models directory relative to this script
        base_dir = os.path.dirname(os.path.abspath(__file__))
        local_in_models = os.path.join(base_dir, "models", os.path.basename(model_path))
        if os.path.exists(local_in_models):
            return local_in_models

        # If not found locally, allow Ultralytics to automatically download standard weights
        logger.warning(f"Model file '{model_path}' not found locally. Falling back to default download.")
        return "yolov8n.pt"

    def get_classes(self) -> Dict[int, str]:
        """Return the mapping of class IDs to class names."""
        return self.class_names

    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: Optional[float] = None,
        classes: Optional[List[int]] = None
    ) -> List[Detection]:
        """
        Perform object detection on a single frame.

        Args:
            frame: Input BGR image (NumPy array).
            conf_threshold: Optional override for confidence threshold.
            classes: Optional list of class IDs to filter (e.g. [0] for 'person').

        Returns:
            List of Detection instances containing bboxes, confidences, and classes.
        """
        if frame is None or frame.size == 0:
            return []

        conf = conf_threshold if conf_threshold is not None else self.conf_threshold

        # Run inference through YOLO
        results = self.model(
            source=frame,
            conf=conf,
            iou=self.iou_threshold,
            classes=classes,
            device=self.device,
            verbose=False
        )

        detections: List[Detection] = []
        if not results or len(results) == 0:
            return detections

        first_res = results[0]
        boxes = first_res.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        # Extract coordinates, confidence, and class id
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)

        for i in range(len(xyxy)):
            box = xyxy[i]
            score = float(confs[i])
            cid = int(cls_ids[i])
            cname = self.class_names.get(cid, f"Class {cid}")

            detection = Detection(
                bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                confidence=score,
                class_id=cid,
                class_name=cname
            )
            detections.append(detection)

        return detections

    def annotate(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        show_confidence: bool = True,
        show_label: bool = True
    ) -> np.ndarray:
        """
        Draw high-contrast, professional bounding boxes and labels on the frame.

        Args:
            frame: Input BGR image (NumPy array).
            detections: List of Detection objects.
            show_confidence: Whether to append confidence percentage to the label.
            show_label: Whether to render class labels.

        Returns:
            Annotated BGR image frame.
        """
        if frame is None:
            return frame

        canvas = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = map(int, det.bbox)
            color = self.PALETTE[det.class_id % len(self.PALETTE)]

            # Draw outer bounding box with clean rounded corner styling
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)

            # Build label text
            parts = []
            if show_label:
                parts.append(det.class_name)
            if show_confidence:
                parts.append(f"{det.confidence * 100:.1f}%")
            label_text = " | ".join(parts)

            if label_text:
                # Calculate text dimensions
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

                # Draw filled label background badge
                badge_y1 = max(0, y1 - th - baseline - 4)
                badge_y2 = y1
                cv2.rectangle(
                    canvas,
                    (x1, badge_y1),
                    (x1 + tw + 6, badge_y2),
                    color,
                    -1
                )

                # Render readable text with high contrast
                cv2.putText(
                    canvas,
                    label_text,
                    (x1 + 3, badge_y2 - baseline),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    lineType=cv2.LINE_AA
                )

        return canvas


# ==============================================================================
# Video & Stream Validation Utilities
# ==============================================================================

def open_video_source(source: Union[int, str]) -> Tuple[bool, str, Optional[cv2.VideoCapture]]:
    """
    Safely open a webcam device index or video file path.

    Args:
        source: Device index (e.g., 0) for webcam, or file path string.

    Returns:
        Tuple of (success_flag, status_message, cv2.VideoCapture instance or None).
    """
    if isinstance(source, str):
        if not os.path.exists(source):
            return False, f"Video file not found at path: '{source}'", None
        if os.path.getsize(source) == 0:
            return False, f"Video file is empty (0 bytes): '{source}'", None

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        source_name = f"Webcam index {source}" if isinstance(source, int) else f"File '{source}'"
        return False, f"Unable to open video stream from {source_name}. Device may be in use or format unsupported.", None

    # Test reading a single frame to ensure actual decodability
    ret, frame = cap.read()
    if not ret or frame is None or frame.size == 0:
        cap.release()
        source_name = f"Webcam index {source}" if isinstance(source, int) else f"File '{source}'"
        return False, f"Could not decode any video frames from {source_name}.", None

    # Reset position to beginning for video files
    if isinstance(source, str):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    return True, "Video source opened successfully.", cap


def get_video_info(cap: cv2.VideoCapture) -> Dict[str, Union[int, float]]:
    """
    Retrieve video metadata such as width, height, FPS, and total frame count.

    Args:
        cap: Opened cv2.VideoCapture instance.

    Returns:
        Dictionary containing metadata properties.
    """
    if cap is None or not cap.isOpened():
        return {"width": 0, "height": 0, "fps": 0.0, "total_frames": 0}

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0 or np.isnan(fps):
        fps = 30.0  # Safe fallback for streams with unannounced frame rate
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "total_frames": max(0, total_frames)
    }
