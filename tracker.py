"""
SORT (Simple Online and Realtime Tracking) Algorithm Implementation.

Implements genuine SORT multi-object tracking using:
1. 7-state Kalman Filter motion estimation (constant velocity model).
2. Intersection-over-Union (IoU) spatial association metric.
3. Hungarian algorithm (Munkres / Linear Sum Assignment) for global optimal bipartite matching.
4. Robust track lifecycle management (birth, active, missed, dead).
5. Visual trajectory history tracking.

Reference:
Bewley, A., Ge, Z., Ott, L., Ramos, F., & Upcroft, B. (2016).
Simple online and realtime tracking. IEEE ICIP.
"""

from __future__ import annotations
import collections
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Union

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter


# ==============================================================================
# Data Structures
# ==============================================================================

@dataclass
class TrackedObject:
    """Represents a confirmed tracked object with unique ID and motion history."""
    track_id: int
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str
    history: List[Tuple[int, int]] = field(default_factory=list)  # Centroid history
    hits: int = 1
    age: int = 1
    time_since_update: int = 0

    @property
    def xyxy(self) -> List[float]:
        return list(self.bbox)

    @property
    def center(self) -> Tuple[int, int]:
        cx = int((self.bbox[0] + self.bbox[2]) / 2.0)
        cy = int((self.bbox[1] + self.bbox[3]) / 2.0)
        return cx, cy


# ==============================================================================
# Geometric Conversion Functions
# ==============================================================================

def bbox_to_z(bbox: Union[List[float], Tuple[float, float, float, float], np.ndarray]) -> np.ndarray:
    """
    Convert bounding box [x1, y1, x2, y2] to measurement vector z:
    z = [u, v, s, r]^T
    where:
        u: center x
        v: center y
        s: scale / area (width * height)
        r: aspect ratio (width / height)
    """
    w = max(1e-6, float(bbox[2] - bbox[0]))
    h = max(1e-6, float(bbox[3] - bbox[1]))
    u = float(bbox[0]) + w / 2.0
    v = float(bbox[1]) + h / 2.0
    s = w * h
    r = w / h
    return np.array([u, v, s, r], dtype=np.float32).reshape((4, 1))


def x_to_bbox(x: np.ndarray) -> np.ndarray:
    """
    Convert state vector x = [u, v, s, r, ...]^T back to bounding box [x1, y1, x2, y2].
    """
    u = float(x[0, 0]) if x.ndim > 1 else float(x[0])
    v = float(x[1, 0]) if x.ndim > 1 else float(x[1])
    s = float(x[2, 0]) if x.ndim > 1 else float(x[2])
    r = float(x[3, 0]) if x.ndim > 1 else float(x[3])

    s = max(1e-6, s)
    r = max(1e-6, r)

    w = np.sqrt(s * r)
    h = s / max(1e-6, w)

    x1 = u - w / 2.0
    y1 = v - h / 2.0
    x2 = u + w / 2.0
    y2 = v + h / 2.0
    return np.array([x1, y1, x2, y2], dtype=np.float32)


# ==============================================================================
# IoU Matrix Computation
# ==============================================================================

def iou_batch(bboxes_a: np.ndarray, bboxes_b: np.ndarray) -> np.ndarray:
    """
    Compute Intersection over Union (IoU) between two sets of bounding boxes.

    Args:
        bboxes_a: Array of shape (N, 4) in [x1, y1, x2, y2] format.
        bboxes_b: Array of shape (M, 4) in [x1, y1, x2, y2] format.

    Returns:
        IoU matrix of shape (N, M).
    """
    if len(bboxes_a) == 0 or len(bboxes_b) == 0:
        return np.zeros((len(bboxes_a), len(bboxes_b)), dtype=np.float32)

    bboxes_a = np.asarray(bboxes_a, dtype=np.float32)
    bboxes_b = np.asarray(bboxes_b, dtype=np.float32)

    # Coordinates of intersection
    x1 = np.maximum(bboxes_a[:, 0:1], bboxes_b[:, 0:1].T)
    y1 = np.maximum(bboxes_a[:, 1:2], bboxes_b[:, 1:2].T)
    x2 = np.minimum(bboxes_a[:, 2:3], bboxes_b[:, 2:3].T)
    y2 = np.minimum(bboxes_a[:, 3:4], bboxes_b[:, 3:4].T)

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    intersection = inter_w * inter_h

    area_a = (bboxes_a[:, 2] - bboxes_a[:, 0]) * (bboxes_a[:, 3] - bboxes_a[:, 1])
    area_b = (bboxes_b[:, 2] - bboxes_b[:, 0]) * (bboxes_b[:, 3] - bboxes_b[:, 1])

    union = area_a[:, None] + area_b[None, :] - intersection
    union = np.maximum(1e-6, union)

    return intersection / union


# ==============================================================================
# KalmanBoxTracker Class
# ==============================================================================

class KalmanBoxTracker:
    """
    Tracks an individual object instance across frames using a 7-state Kalman Filter.

    State representation:
        x = [u, v, s, r, u_dot, v_dot, s_dot]^T
    where (u, v) is center, s is area, r is aspect ratio, and dotted terms are velocities.
    """

    count: int = 0  # Global class tracker counter

    def __init__(
        self,
        bbox: Union[List[float], Tuple[float, float, float, float], np.ndarray],
        class_id: int = 0,
        class_name: str = "object",
        confidence: float = 1.0,
        max_history: int = 40
    ) -> None:
        """Initialize tracker with first detection bounding box."""
        # 7-dimensional state, 4-dimensional measurement
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # State transition matrix F (constant velocity model with dt=1)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)

        # Measurement mapping matrix H
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ], dtype=np.float32)

        # Measurement uncertainty covariance R
        self.kf.R[2:, 2:] *= 10.0

        # State covariance P (high initial uncertainty on unobserved velocities)
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0

        # Process noise covariance Q
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        # Initialize state with initial measurement
        self.kf.x[:4] = bbox_to_z(bbox)

        # Track management attributes
        self.time_since_update: int = 0
        KalmanBoxTracker.count += 1
        self.id: int = KalmanBoxTracker.count
        self.hits: int = 1
        self.hit_streak: int = 1
        self.age: int = 0

        # Metadata
        self.class_id: int = class_id
        self.class_name: str = class_name
        self.confidence: float = confidence

        # Centroid history trail for visualization
        self.history: collections.deque = collections.deque(maxlen=max_history)
        cx = int((bbox[0] + bbox[2]) / 2.0)
        cy = int((bbox[1] + bbox[3]) / 2.0)
        self.history.append((cx, cy))

    def update(
        self,
        bbox: Union[List[float], Tuple[float, float, float, float], np.ndarray],
        class_id: Optional[int] = None,
        class_name: Optional[str] = None,
        confidence: Optional[float] = None
    ) -> None:
        """
        Update tracker state with a new matched measurement.
        """
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

        if class_id is not None:
            self.class_id = class_id
        if class_name is not None:
            self.class_name = class_name
        if confidence is not None:
            self.confidence = confidence

        # Kalman Filter measurement update
        self.kf.update(bbox_to_z(bbox))

        # Append updated centroid to history trail
        curr_box = self.get_state()
        cx = int((curr_box[0] + curr_box[2]) / 2.0)
        cy = int((curr_box[1] + curr_box[3]) / 2.0)
        self.history.append((cx, cy))

    def predict(self) -> np.ndarray:
        """
        Advance state vector one time step and return predicted bounding box.
        """
        # Guard against degenerate negative scale values
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] = 0.0

        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        return self.get_state()

    def get_state(self) -> np.ndarray:
        """Return the current bounding box estimate [x1, y1, x2, y2]."""
        return x_to_bbox(self.kf.x)


# ==============================================================================
# Data Association via Hungarian Algorithm
# ==============================================================================

def associate_detections_to_trackers(
    detections: np.ndarray,
    trackers: np.ndarray,
    iou_threshold: float = 0.3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Associate detections to active trackers using IoU metric and Hungarian Algorithm.

    Args:
        detections: Array of shape (N, 4) of detected bounding boxes.
        trackers: Array of shape (M, 4) of predicted tracker bounding boxes.
        iou_threshold: Minimum IoU overlap required to consider a match.

    Returns:
        matches: Array of shape (K, 2) where column 0 is tracker index, column 1 is detection index.
        unmatched_trackers: Array of indices of trackers without detections.
        unmatched_detections: Array of indices of detections without trackers.
    """
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.empty(0, dtype=int),
            np.arange(len(detections), dtype=int)
        )

    if len(detections) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(trackers), dtype=int),
            np.empty(0, dtype=int)
        )

    # Compute pairwise IoU cost matrix
    iou_matrix = iou_batch(trackers, detections)

    # Hungarian algorithm operates on cost minimization, so cost = -iou (or 1 - iou)
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)

    matched_indices = []
    unmatched_trackers = set(range(len(trackers)))
    unmatched_detections = set(range(len(detections)))

    for r, c in zip(row_ind, col_ind):
        # Reject match if IoU is beneath the threshold
        if iou_matrix[r, c] < iou_threshold:
            continue
        matched_indices.append([r, c])
        unmatched_trackers.discard(r)
        unmatched_detections.discard(c)

    matches = np.array(matched_indices, dtype=int).reshape((-1, 2)) if matched_indices else np.empty((0, 2), dtype=int)
    unmatched_trackers_arr = np.array(sorted(list(unmatched_trackers)), dtype=int)
    unmatched_detections_arr = np.array(sorted(list(unmatched_detections)), dtype=int)

    return matches, unmatched_trackers_arr, unmatched_detections_arr


# ==============================================================================
# SORT Tracker
# ==============================================================================

class SortTracker:
    """
    Multi-Object Tracker based on SORT (Simple Online and Realtime Tracking).
    """

    # High-contrast color palette for unique object track IDs (BGR format)
    TRACK_COLORS = [
        (255, 99, 71),   (50, 205, 50),   (30, 144, 255), (255, 215, 0),
        (255, 105, 180), (0, 255, 255),   (138, 43, 226), (255, 140, 0),
        (0, 250, 154),   (220, 20, 60),   (65, 105, 225), (255, 0, 255),
        (0, 191, 255),   (154, 205, 50),  (255, 20, 147), (72, 209, 204),
        (218, 112, 214), (240, 230, 140), (127, 255, 0), (255, 165, 0)
    ]

    def __init__(
        self,
        max_age: int = 15,
        min_hits: int = 1,
        iou_threshold: float = 0.3
    ) -> None:
        """
        Initialize the SORT tracker.

        Args:
            max_age: Maximum consecutive frames to preserve a track without detections.
                     Allows re-acquisition after temporary occlusion.
            min_hits: Minimum number of detections before a track is published.
                     (1 allows instant tracking; 2-3 reduces false alarms).
            iou_threshold: Minimum IoU overlap required for data association.
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count: int = 0

    def reset(self) -> None:
        """Reset all trackers and frame counters."""
        self.trackers = []
        self.frame_count = 0
        KalmanBoxTracker.count = 0

    def update(
        self,
        detections: List[Any]
    ) -> List[TrackedObject]:
        """
        Update the tracker state with new frame detections.

        Args:
            detections: List of Detection objects (from detector.py) or
                        tuples: (bbox, confidence, class_id, class_name).

        Returns:
            List of TrackedObject instances containing confirmed tracks.
        """
        self.frame_count += 1

        # ----------------------------------------------------------------------
        # Step 1: Predict state for all existing trackers using Kalman Filter
        # ----------------------------------------------------------------------
        trks = np.zeros((len(self.trackers), 4), dtype=np.float32)
        to_del: List[int] = []

        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            trks[t] = pos
            if np.any(np.isnan(pos)) or np.any(np.isinf(pos)):
                to_del.append(t)

        # Remove invalid trackers
        for t in reversed(to_del):
            self.trackers.pop(t)
            trks = np.delete(trks, t, axis=0)

        # ----------------------------------------------------------------------
        # Step 2: Format incoming detections
        # ----------------------------------------------------------------------
        dets = np.zeros((len(detections), 4), dtype=np.float32)
        det_meta: List[Tuple[float, int, str]] = []

        for i, d in enumerate(detections):
            if hasattr(d, "bbox"):
                dets[i] = d.bbox
                det_meta.append((d.confidence, d.class_id, d.class_name))
            elif isinstance(d, (list, tuple)) and len(d) >= 4:
                dets[i] = d[0] if isinstance(d[0], (list, tuple)) else d[:4]
                conf = float(d[1]) if len(d) > 1 and isinstance(d[1], (int, float)) else 1.0
                cid = int(d[2]) if len(d) > 2 and isinstance(d[2], int) else 0
                cname = str(d[3]) if len(d) > 3 and isinstance(d[3], str) else "object"
                det_meta.append((conf, cid, cname))
            else:
                det_meta.append((1.0, 0, "object"))

        # ----------------------------------------------------------------------
        # Step 3: Associate detections with existing trackers (Hungarian Alg)
        # ----------------------------------------------------------------------
        matched, unmatched_trks, unmatched_dets = associate_detections_to_trackers(
            dets, trks, self.iou_threshold
        )

        # ----------------------------------------------------------------------
        # Step 4: Update matched trackers with corresponding detections
        # ----------------------------------------------------------------------
        for m in matched:
            trk_idx = m[0]
            det_idx = m[1]
            conf, cid, cname = det_meta[det_idx]
            self.trackers[trk_idx].update(
                bbox=dets[det_idx],
                class_id=cid,
                class_name=cname,
                confidence=conf
            )

        # ----------------------------------------------------------------------
        # Step 5: Initialize new trackers for unmatched detections
        # ----------------------------------------------------------------------
        for d_idx in unmatched_dets:
            conf, cid, cname = det_meta[d_idx]
            new_trk = KalmanBoxTracker(
                bbox=dets[d_idx],
                class_id=cid,
                class_name=cname,
                confidence=conf
            )
            self.trackers.append(new_trk)

        # ----------------------------------------------------------------------
        # Step 6: Collect confirmed active tracks & purge dead trackers
        # ----------------------------------------------------------------------
        confirmed_tracks: List[TrackedObject] = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            i -= 1
            bbox = trk.get_state()

            # Track is output if:
            # - It was recently updated (time_since_update < 1)
            # - Has sufficient hits or stream is in early warmup frames
            is_active = (trk.time_since_update < 1) and (
                trk.hits >= self.min_hits or self.frame_count <= self.min_hits
            )

            if is_active:
                confirmed_tracks.append(
                    TrackedObject(
                        track_id=trk.id,
                        bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                        confidence=trk.confidence,
                        class_id=trk.class_id,
                        class_name=trk.class_name,
                        history=list(trk.history),
                        hits=trk.hits,
                        age=trk.age,
                        time_since_update=trk.time_since_update
                    )
                )

            # Dead track termination: prune if time_since_update exceeds max_age
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        return confirmed_tracks

    def annotate(
        self,
        frame: np.ndarray,
        tracks: List[TrackedObject],
        show_trails: bool = True,
        show_id: bool = True,
        show_label: bool = True,
        show_confidence: bool = True
    ) -> np.ndarray:
        """
        Render tracked objects, bounding boxes, labels, and motion trails on the frame.

        Args:
            frame: Input BGR image.
            tracks: List of TrackedObject instances.
            show_trails: Whether to render fading trajectory paths.
            show_id: Whether to display tracking ID.
            show_label: Whether to display semantic class name.
            show_confidence: Whether to display detection confidence.

        Returns:
            Annotated BGR frame.
        """
        if frame is None:
            return frame

        canvas = frame.copy()

        for track in tracks:
            # Deterministic color for each unique tracking ID
            color = self.TRACK_COLORS[track.track_id % len(self.TRACK_COLORS)]

            # ------------------------------------------------------------------
            # Draw Motion History Trail
            # ------------------------------------------------------------------
            if show_trails and len(track.history) > 1:
                pts = list(track.history)
                for idx in range(1, len(pts)):
                    thickness = int(np.sqrt(float(idx) / len(pts)) * 3) + 1
                    cv2.line(canvas, pts[idx - 1], pts[idx], color, thickness, lineType=cv2.LINE_AA)

            # ------------------------------------------------------------------
            # Draw Tracking Bounding Box
            # ------------------------------------------------------------------
            x1, y1, x2, y2 = map(int, track.bbox)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)

            # Corner decorative accents for a sleek modern look
            corner_len = min(15, max(5, int((x2 - x1) * 0.15)), max(5, int((y2 - y1) * 0.15)))
            # Top-left corner
            cv2.line(canvas, (x1, y1), (x1 + corner_len, y1), color, 4, lineType=cv2.LINE_AA)
            cv2.line(canvas, (x1, y1), (x1, y1 + corner_len), color, 4, lineType=cv2.LINE_AA)
            # Top-right corner
            cv2.line(canvas, (x2, y1), (x2 - corner_len, y1), color, 4, lineType=cv2.LINE_AA)
            cv2.line(canvas, (x2, y1), (x2, y1 + corner_len), color, 4, lineType=cv2.LINE_AA)
            # Bottom-left corner
            cv2.line(canvas, (x1, y2), (x1 + corner_len, y2), color, 4, lineType=cv2.LINE_AA)
            cv2.line(canvas, (x1, y2), (x1, y2 - corner_len), color, 4, lineType=cv2.LINE_AA)
            # Bottom-right corner
            cv2.line(canvas, (x2, y2), (x2 - corner_len, y2), color, 4, lineType=cv2.LINE_AA)
            cv2.line(canvas, (x2, y2), (x2, y2 - corner_len), color, 4, lineType=cv2.LINE_AA)

            # ------------------------------------------------------------------
            # Draw Badge Label (ID, Class, Confidence)
            # ------------------------------------------------------------------
            badge_parts = []
            if show_id:
                badge_parts.append(f"ID #{track.track_id}")
            if show_label:
                badge_parts.append(track.class_name)
            if show_confidence:
                badge_parts.append(f"{track.confidence * 100:.0f}%")

            label_text = " | ".join(badge_parts)

            if label_text:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.52
                thickness = 1
                (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

                badge_y1 = max(0, y1 - th - baseline - 6)
                badge_y2 = y1
                cv2.rectangle(
                    canvas,
                    (x1, badge_y1),
                    (x1 + tw + 8, badge_y2),
                    color,
                    -1
                )

                cv2.putText(
                    canvas,
                    label_text,
                    (x1 + 4, badge_y2 - baseline - 2),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    lineType=cv2.LINE_AA
                )

        return canvas
