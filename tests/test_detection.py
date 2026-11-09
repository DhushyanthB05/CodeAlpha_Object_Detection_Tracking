"""
Automated Test Suite for Object Detection and Tracking.

Covers:
1. Model loading.
2. Image detection.
3. Video processing.
4. Bounding box generation.
5. Confidence filtering.
6. Object class identification.
7. SORT tracker initialization.
8. Tracking ID assignment.
9. Tracking continuity.
10. Invalid video handling.
"""

import os
import sys
import tempfile
import pytest
import cv2
import numpy as np

# Ensure root directory is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from detector import YOLODetector, Detection, open_video_source, get_video_info
from tracker import SortTracker, KalmanBoxTracker, TrackedObject, iou_batch, associate_detections_to_trackers


@pytest.fixture(scope="session")
def detector() -> YOLODetector:
    """Session-scoped fixture to load YOLO model once for the entire test suite."""
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "yolov8n.pt")
    return YOLODetector(model_path=model_path, conf_threshold=0.25)


@pytest.fixture
def sample_image() -> np.ndarray:
    """Fixture providing a synthetic image with high-contrast shapes."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw simulated objects
    cv2.rectangle(img, (150, 150), (350, 350), (255, 255, 255), -1)
    cv2.circle(img, (500, 240), 60, (200, 200, 200), -1)
    return img


# ==============================================================================
# 1. Model Loading Test
# ==============================================================================

def test_model_loading(detector: YOLODetector):
    """Test that the YOLO model weights are loaded and COCO classes are populated."""
    assert detector.model is not None, "YOLO model instance should not be None"
    classes = detector.get_classes()
    assert isinstance(classes, dict), "Classes should be returned as a dictionary"
    assert len(classes) == 80, f"Standard COCO weights must contain 80 classes, found {len(classes)}"
    assert 0 in classes and classes[0] == "person", "Class ID 0 in COCO should be 'person'"


# ==============================================================================
# 2. Image Detection Test
# ==============================================================================

def test_image_detection(detector: YOLODetector, sample_image: np.ndarray):
    """Test detection execution on a frame returns a list of Detection objects."""
    detections = detector.detect(sample_image)
    assert isinstance(detections, list), "detect() must return a list"

    # Test detection on an empty or None frame
    assert detector.detect(None) == [], "detect(None) should return an empty list"
    assert detector.detect(np.array([])) == [], "detect(empty_array) should return an empty list"


# ==============================================================================
# 3. Bounding Box Generation & Coordinates
# ==============================================================================

def test_bounding_box_generation():
    """Test Detection object coordinates, dimensions, and centroid calculation."""
    det = Detection(bbox=(100.0, 150.0, 300.0, 450.0), confidence=0.85, class_id=2, class_name="car")

    assert det.xyxy == [100.0, 150.0, 300.0, 450.0]
    assert det.width == 200.0, f"Expected width 200.0, got {det.width}"
    assert det.height == 300.0, f"Expected height 300.0, got {det.height}"

    cx, cy = det.center
    assert cx == 200.0, f"Expected center x 200.0, got {cx}"
    assert cy == 300.0, f"Expected center y 300.0, got {cy}"


# ==============================================================================
# 4. Confidence Filtering
# ==============================================================================

def test_confidence_filtering(detector: YOLODetector):
    """Test that setting confidence threshold to 1.0 filters all detections."""
    # Create an image
    img = np.full((320, 320, 3), 128, dtype=np.uint8)

    # Detect with impossible threshold 1.0
    dets_high = detector.detect(img, conf_threshold=1.0)
    assert len(dets_high) == 0, "No detection should exceed 1.0 confidence on uniform image"


# ==============================================================================
# 5. Object Class Identification & Filtering
# ==============================================================================

def test_object_class_identification(detector: YOLODetector):
    """Test class filtering parameter restricts detections to target classes."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)

    # Filter by specific classes (e.g. only person = 0 and car = 2)
    dets = detector.detect(img, classes=[0, 2])
    for d in dets:
        assert d.class_id in [0, 2], f"Detected class {d.class_id} not in filtered list [0, 2]"


# ==============================================================================
# 6. SORT Tracker Initialization & State Matrices
# ==============================================================================

def test_sort_tracker_initialization():
    """Test initialization of SORT tracker and KalmanBoxTracker dimensions."""
    tracker = SortTracker(max_age=15, min_hits=1, iou_threshold=0.3)
    assert tracker.max_age == 15
    assert tracker.min_hits == 1
    assert tracker.iou_threshold == 0.3
    assert len(tracker.trackers) == 0
    assert tracker.frame_count == 0

    # Test single Kalman tracker matrix dimensions
    kb = KalmanBoxTracker(bbox=[100, 100, 200, 200], class_id=0, class_name="person")
    assert kb.kf.dim_x == 7, "State vector dim should be 7 [u, v, s, r, u', v', s']"
    assert kb.kf.dim_z == 4, "Measurement vector dim should be 4 [u, v, s, r]"
    assert kb.kf.x.shape == (7, 1), "State vector shape must be (7, 1)"


# ==============================================================================
# 7. Tracking ID Assignment
# ==============================================================================

def test_tracking_id_assignment():
    """Test that multiple distinct detections in frame 1 receive distinct unique IDs."""
    tracker = SortTracker(max_age=10, min_hits=1, iou_threshold=0.3)
    tracker.reset()

    # Two spatially separated objects
    det1 = Detection(bbox=(50, 50, 100, 100), confidence=0.9, class_id=0, class_name="person")
    det2 = Detection(bbox=(300, 300, 400, 400), confidence=0.85, class_id=2, class_name="car")

    tracks = tracker.update([det1, det2])
    assert len(tracks) == 2, f"Expected 2 tracks, got {len(tracks)}"

    id1 = tracks[0].track_id
    id2 = tracks[1].track_id
    assert id1 != id2, f"Track IDs must be distinct! Got {id1} and {id2}"
    assert id1 > 0 and id2 > 0, "Track IDs must be positive integers"


# ==============================================================================
# 8. Tracking Continuity across Consecutive Frames
# ==============================================================================

def test_tracking_continuity():
    """Verify that an object moving along a trajectory maintains its identity."""
    tracker = SortTracker(max_age=10, min_hits=1, iou_threshold=0.3)
    tracker.reset()

    # Move an object horizontally by 6 pixels per frame for 10 frames
    assigned_ids = []
    for step in range(10):
        x1 = 100 + step * 6
        x2 = 160 + step * 6
        det = Detection(bbox=(x1, 100, x2, 160), confidence=0.9, class_id=0, class_name="person")
        tracks = tracker.update([det])

        assert len(tracks) == 1, f"Frame {step+1}: expected 1 track, got {len(tracks)}"
        assigned_ids.append(tracks[0].track_id)

    # All 10 frames must have the exact same track ID
    unique_ids = set(assigned_ids)
    assert len(unique_ids) == 1, f"Track ID must be persistent across all frames! Found IDs: {unique_ids}"
    assert len(tracks[0].history) >= 10, "Track should accumulate centroid history trail"


# ==============================================================================
# 9. Occlusion and Missed Detections Handling
# ==============================================================================

def test_occlusion_and_missed_detections():
    """Verify that Kalman filter bridges temporary missed detections and maintains ID."""
    tracker = SortTracker(max_age=10, min_hits=1, iou_threshold=0.2)
    tracker.reset()

    # Frame 1: Detection at (200, 200, 260, 260)
    t1 = tracker.update([Detection((200, 200, 260, 260), 0.9, 0, "person")])
    assert len(t1) == 1
    original_id = t1[0].track_id

    # Frame 2 & 3: Temporary occlusion (no detections)
    t2 = tracker.update([])
    assert len(t2) == 0, "No tracks should be published during missed frame"

    t3 = tracker.update([])
    assert len(t3) == 0

    # Frame 4: Object re-emerges slightly shifted to (215, 200, 275, 260)
    t4 = tracker.update([Detection((215, 200, 275, 260), 0.88, 0, "person")])
    assert len(t4) == 1, f"Track should be re-acquired, got {len(t4)} tracks"
    assert t4[0].track_id == original_id, (
        f"Re-acquired object must preserve original ID {original_id}, got {t4[0].track_id}"
    )


# ==============================================================================
# 10. Invalid Video and Camera Handling
# ==============================================================================

def test_invalid_video_handling():
    """Verify open_video_source safely detects and reports non-existent/corrupted sources."""
    # Test non-existent file
    success, msg, cap = open_video_source("non_existent_video_path_xyz123.mp4")
    assert not success, "Non-existent video file should return False"
    assert "not found" in msg.lower()
    assert cap is None

    # Test empty (0 bytes) file
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as empty_file:
        empty_path = empty_file.name

    try:
        success, msg, cap = open_video_source(empty_path)
        assert not success, "0-byte video file should return False"
        assert "empty" in msg.lower() or "decode" in msg.lower()
        assert cap is None
    finally:
        if os.path.exists(empty_path):
            os.remove(empty_path)

    # Test invalid webcam index (e.g., 999 which does not exist)
    success, msg, cap = open_video_source(999)
    assert not success, "Invalid camera index should return False"
    assert cap is None


# ==============================================================================
# 11. Full End-to-End Synthetic Video Processing Test
# ==============================================================================

def test_video_processing_pipeline(detector: YOLODetector):
    """Test reading, metadata retrieval, detection, and tracking through a synthesized video."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as vid_file:
        video_path = vid_file.name

    try:
        # Create a short 10-frame synthetic video
        width, height, fps = 320, 240, 10
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

        for f_idx in range(10):
            frame = np.full((height, width, 3), 40, dtype=np.uint8)
            # Moving rectangle
            rx = 30 + f_idx * 10
            cv2.rectangle(frame, (rx, 50), (rx + 60, 110), (255, 255, 255), -1)
            writer.write(frame)
        writer.release()

        # Open and test properties
        success, msg, cap = open_video_source(video_path)
        assert success, f"Failed to open generated video: {msg}"

        info = get_video_info(cap)
        assert info["width"] == width
        assert info["height"] == height
        assert info["total_frames"] == 10

        # Run detection and tracking across frames
        tracker = SortTracker(max_age=5, min_hits=1)
        frames_read = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frames_read += 1

            # Detect & Track
            dets = detector.detect(frame)
            tracks = tracker.update(dets)

            # Test annotate
            annotated = tracker.annotate(frame, tracks)
            assert annotated.shape == frame.shape

        cap.release()
        assert frames_read == 10, f"Expected 10 frames read, got {frames_read}"

    finally:
        if os.path.exists(video_path):
            os.remove(video_path)
