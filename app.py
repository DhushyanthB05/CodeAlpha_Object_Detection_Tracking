"""
Streamlit Web Application: Real-Time Object Detection and Tracking with YOLO & SORT.

Author: CodeAlpha Artificial Intelligence Internship
Project: Object Detection and Tracking using YOLO, OpenCV, and SORT
"""

from __future__ import annotations
import os
import sys
import time
import tempfile
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np
import streamlit as st

# Import core detection and tracking modules
from detector import YOLODetector, Detection, open_video_source, get_video_info
from tracker import SortTracker, TrackedObject


# ==============================================================================
# Page Configuration & UI Theme
# ==============================================================================

st.set_page_config(
    page_title="AI Object Detection & Tracking | YOLO + SORT",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for high-contrast, modern UI styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px 20px;
        color: #ffffff;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .metric-val {
        font-size: 2.1rem;
        font-weight: 700;
        color: #38bdf8;
        line-height: 1.1;
    }
    .metric-lbl {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        margin-top: 4px;
    }
    .status-badge-running {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        background-color: #064e3b;
        color: #34d399;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .status-badge-stopped {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        background-color: #451a03;
        color: #fb923c;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# Session State Initialization & Model Caching
# ==============================================================================

if "tracking_active" not in st.session_state:
    st.session_state["tracking_active"] = False

if "total_unique_ids" not in st.session_state:
    st.session_state["total_unique_ids"] = set()

if "frame_counter" not in st.session_state:
    st.session_state["frame_counter"] = 0


@st.cache_resource(show_spinner="Loading YOLOv8 Neural Network weights...")
def load_detector(model_path: str = "models/yolov8n.pt") -> YOLODetector:
    """Load and cache the YOLOv8 detector instance."""
    return YOLODetector(model_path=model_path)


# ==============================================================================
# Helper: Synthetic Test Video Generator for Fallback/Testing
# ==============================================================================

def generate_sample_demo_video(output_path: str, duration_sec: int = 5, fps: int = 25) -> str:
    """Generate a clean synthetic video showing moving objects for immediate testing."""
    width, height = 640, 480
    total_frames = duration_sec * fps
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame_idx in range(total_frames):
        # Create dark background
        frame = np.full((height, width, 3), 35, dtype=np.uint8)

        # Draw road / lane lines
        cv2.line(frame, (0, 320), (width, 320), (80, 80, 80), 3)
        cv2.line(frame, (0, 440), (width, 440), (80, 80, 80), 3)
        for rx in range(0, width, 50):
            cv2.line(frame, (rx, 380), (rx + 25, 380), (200, 200, 200), 2)

        # Simulated Object 1: Moving right (Car-like rectangle)
        x1 = int(30 + (frame_idx * 5) % (width - 120))
        y1 = 340
        cv2.rectangle(frame, (x1, y1), (x1 + 100, y1 + 50), (60, 180, 75), -1)
        cv2.putText(frame, "SIMULATED VEHICLE", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Simulated Object 2: Moving left (Person/Bicycle-like shape)
        x2 = int(width - 50 - (frame_idx * 3) % (width - 80))
        y2 = 200
        cv2.rectangle(frame, (x2, y2), (x2 + 40, y2 + 80), (230, 25, 75), -1)
        cv2.putText(frame, "PEDESTRIAN", (x2 - 10, y2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Title on video
        cv2.putText(frame, f"Synthetic Test Stream - Frame {frame_idx + 1}/{total_frames}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        out.write(frame)

    out.release()
    return output_path


# ==============================================================================
# Sidebar UI & Controls
# ==============================================================================

def render_sidebar(detector: YOLODetector) -> Dict:
    """Render interactive configuration options in Streamlit sidebar."""
    st.sidebar.title("🎯 Control Panel")
    st.sidebar.caption("YOLOv8 + SORT Multi-Object Tracker")

    # 1. Source Selection
    st.sidebar.subheader("1. Video Source")
    source_type = st.sidebar.radio(
        "Choose input stream:",
        ["📁 Upload Video File", "📹 Live Webcam", "🧪 Synthetic Demo Stream"],
        index=0
    )

    uploaded_file = None
    webcam_index = 0

    if source_type == "📁 Upload Video File":
        uploaded_file = st.sidebar.file_uploader(
            "Upload an MP4, AVI, or MOV video:",
            type=["mp4", "avi", "mov", "mkv"]
        )
    elif source_type == "📹 Live Webcam":
        webcam_index = st.sidebar.number_input(
            "Webcam Device Index:",
            min_value=0,
            max_value=10,
            value=0,
            step=1,
            help="Default built-in laptop camera is usually index 0."
        )

    # 2. Detection Parameters
    st.sidebar.subheader("2. YOLO Detection Settings")
    conf_thresh = st.sidebar.slider(
        "Confidence Threshold:",
        min_value=0.05,
        max_value=1.00,
        value=0.35,
        step=0.05,
        help="Filter out detections with lower confidence than this threshold."
    )

    # Class filter options
    available_classes = detector.get_classes()
    class_names = sorted(list(available_classes.values()))
    selected_classes = st.sidebar.multiselect(
        "Filter Specific Classes:",
        options=class_names,
        default=[],
        help="Leave empty to detect and track all 80 COCO classes."
    )

    selected_class_ids = None
    if selected_classes:
        name_to_id = {v: k for k, v in available_classes.items()}
        selected_class_ids = [name_to_id[name] for name in selected_classes if name in name_to_id]

    # 3. SORT Tracking Parameters
    st.sidebar.subheader("3. SORT Tracking Settings")
    max_age = st.sidebar.slider(
        "Max Age (Frames):",
        min_value=1,
        max_value=60,
        value=15,
        step=1,
        help="Maximum frames to preserve a track without detections (occlusion handling)."
    )
    min_hits = st.sidebar.slider(
        "Min Consecutive Hits:",
        min_value=1,
        max_value=10,
        value=1,
        step=1,
        help="Minimum detections required before a track is published."
    )
    sort_iou = st.sidebar.slider(
        "SORT IoU Association Threshold:",
        min_value=0.10,
        max_value=0.80,
        value=0.30,
        step=0.05,
        help="Minimum bounding box overlap required for Hungarian matching."
    )

    # 4. Display Preferences
    st.sidebar.subheader("4. Visualization Options")
    show_trails = st.sidebar.checkbox("Show Trajectory Trails", value=True)
    show_id = st.sidebar.checkbox("Show Tracking ID", value=True)
    show_label = st.sidebar.checkbox("Show Class Label", value=True)
    show_conf = st.sidebar.checkbox("Show Confidence Score", value=True)

    # Action Controls
    st.sidebar.markdown("---")
    col_start, col_stop = st.sidebar.columns(2)
    start_clicked = col_start.button("▶️ Start", use_container_width=True, type="primary")
    stop_clicked = col_stop.button("⏹️ Stop", use_container_width=True)

    if start_clicked:
        st.session_state["tracking_active"] = True
    if stop_clicked:
        st.session_state["tracking_active"] = False

    return {
        "source_type": source_type,
        "uploaded_file": uploaded_file,
        "webcam_index": int(webcam_index),
        "conf_thresh": float(conf_thresh),
        "selected_class_ids": selected_class_ids,
        "max_age": int(max_age),
        "min_hits": int(min_hits),
        "sort_iou": float(sort_iou),
        "show_trails": show_trails,
        "show_id": show_id,
        "show_label": show_label,
        "show_conf": show_conf
    }


# ==============================================================================
# Main Application Flow
# ==============================================================================

def main() -> None:
    # Header Banner
    st.title("🎯 Real-Time Object Detection & Tracking")
    st.markdown(
        "**CodeAlpha AI Internship — Project 2** | Integrated **YOLOv8** Object Detection, "
        "**Kalman Filter** State Estimation, and **SORT** Data Association."
    )

    # Load Detector
    try:
        detector = load_detector("models/yolov8n.pt")
    except Exception as e:
        st.error(f"❌ Failed to load YOLO model: {e}")
        return

    # Render Sidebar controls
    cfg = render_sidebar(detector)

    # Top Status & Metrics Dashboard
    metric_cols = st.columns(4)
    with metric_cols[0]:
        active_metric = st.empty()
        active_metric.markdown(
            '<div class="metric-card"><div class="metric-val">0</div>'
            '<div class="metric-lbl">Active Tracks</div></div>',
            unsafe_allow_html=True
        )
    with metric_cols[1]:
        cum_metric = st.empty()
        cum_metric.markdown(
            '<div class="metric-card"><div class="metric-val">0</div>'
            '<div class="metric-lbl">Total Unique IDs</div></div>',
            unsafe_allow_html=True
        )
    with metric_cols[2]:
        fps_metric = st.empty()
        fps_metric.markdown(
            '<div class="metric-card"><div class="metric-val">0.0</div>'
            '<div class="metric-lbl">Processing FPS</div></div>',
            unsafe_allow_html=True
        )
    with metric_cols[3]:
        status_metric = st.empty()
        status_html = '<div class="metric-card"><div class="metric-val" style="font-size:1.4rem; padding-top:6px;">' \
                      + ('<span class="status-badge-running">RUNNING</span>' if st.session_state["tracking_active"] else '<span class="status-badge-stopped">STOPPED</span>') \
                      + '</div><div class="metric-lbl">Tracker Status</div></div>'
        status_metric.markdown(status_html, unsafe_allow_html=True)

    st.markdown("---")

    # Main Workspace Layout: Video Viewport + Live Track Telemetry
    video_col, telemetry_col = st.columns([7, 3])

    with video_col:
        st.subheader("📺 Video Stream Viewport")
        frame_placeholder = st.empty()
        frame_placeholder.info("👈 Configure input source in sidebar and click **Start** to run tracking.")

    with telemetry_col:
        st.subheader("📊 Live Track Telemetry")
        table_placeholder = st.empty()
        table_placeholder.caption("Active track attributes will populate here during inference.")
        chart_placeholder = st.empty()

    # --------------------------------------------------------------------------
    # Processing Loop
    # --------------------------------------------------------------------------
    if st.session_state["tracking_active"]:
        temp_video_path = None
        cap = None

        # Resolve Source
        if cfg["source_type"] == "📁 Upload Video File":
            if cfg["uploaded_file"] is None:
                st.warning("⚠️ Please upload a video file in the sidebar to begin.")
                st.session_state["tracking_active"] = False
                return

            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(cfg["uploaded_file"].read())
            tfile.flush()
            temp_video_path = tfile.name
            source = temp_video_path

        elif cfg["source_type"] == "📹 Live Webcam":
            source = cfg["webcam_index"]

        else:  # Synthetic Demo Stream
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            generate_sample_demo_video(tfile.name, duration_sec=8, fps=25)
            temp_video_path = tfile.name
            source = temp_video_path

        # Open and validate stream
        success, msg, cap = open_video_source(source)
        if not success or cap is None:
            st.error(f"❌ {msg}")
            if cfg["source_type"] == "📹 Live Webcam":
                st.info("💡 **Camera Troubleshooting Tips:**\n"
                        "- Verify your webcam is not currently being used by another application (Zoom, Teams, Camera app).\n"
                        "- If using an external USB camera, test device index 1 or 2 in sidebar.\n"
                        "- You can also use **'📁 Upload Video File'** or **'🧪 Synthetic Demo Stream'** to test the full pipeline.")
            st.session_state["tracking_active"] = False
            return

        # Initialize SORT tracker instance
        tracker = SortTracker(
            max_age=cfg["max_age"],
            min_hits=cfg["min_hits"],
            iou_threshold=cfg["sort_iou"]
        )

        # Tracking state
        fps_history = []
        frame_idx = 0
        all_time_ids = set()

        try:
            while st.session_state["tracking_active"] and cap.isOpened():
                start_time = time.time()
                ret, frame = cap.read()

                if not ret or frame is None or frame.size == 0:
                    # If video file ended, notify user and break
                    st.info("🏁 Video stream playback reached the end.")
                    break

                frame_idx += 1

                # 1. Detect objects via YOLOv8
                detections = detector.detect(
                    frame=frame,
                    conf_threshold=cfg["conf_thresh"],
                    classes=cfg["selected_class_ids"]
                )

                # 2. Update SORT Tracker
                active_tracks = tracker.update(detections)

                # Update cumulative IDs
                for trk in active_tracks:
                    all_time_ids.add(trk.track_id)

                # 3. Annotate frame with bounding boxes, badges, and trails
                annotated_frame = tracker.annotate(
                    frame=frame,
                    tracks=active_tracks,
                    show_trails=cfg["show_trails"],
                    show_id=cfg["show_id"],
                    show_label=cfg["show_label"],
                    show_confidence=cfg["show_conf"]
                )

                # Calculate real-time FPS
                dt = time.time() - start_time
                fps = 1.0 / dt if dt > 0 else 30.0
                fps_history.append(fps)
                if len(fps_history) > 30:
                    fps_history.pop(0)
                avg_fps = sum(fps_history) / len(fps_history)

                # Display annotated frame in Streamlit
                # Convert BGR (OpenCV) to RGB (Streamlit / PIL)
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

                # Update Metric Cards
                active_metric.markdown(
                    f'<div class="metric-card"><div class="metric-val">{len(active_tracks)}</div>'
                    f'<div class="metric-lbl">Active Tracks</div></div>',
                    unsafe_allow_html=True
                )
                cum_metric.markdown(
                    f'<div class="metric-card"><div class="metric-val">{len(all_time_ids)}</div>'
                    f'<div class="metric-lbl">Total Unique IDs</div></div>',
                    unsafe_allow_html=True
                )
                fps_metric.markdown(
                    f'<div class="metric-card"><div class="metric-val">{avg_fps:.1f}</div>'
                    f'<div class="metric-lbl">Processing FPS</div></div>',
                    unsafe_allow_html=True
                )

                # Update Telemetry Table
                if active_tracks:
                    telemetry_data = []
                    class_counts: Dict[str, int] = {}
                    for trk in active_tracks:
                        cx, cy = trk.center
                        telemetry_data.append({
                            "ID": trk.track_id,
                            "Class": trk.class_name,
                            "Conf": f"{trk.confidence * 100:.0f}%",
                            "Pos (X, Y)": f"({cx}, {cy})"
                        })
                        class_counts[trk.class_name] = class_counts.get(trk.class_name, 0) + 1

                    table_placeholder.dataframe(telemetry_data, use_container_width=True, hide_index=True)
                    chart_placeholder.bar_chart(class_counts)
                else:
                    table_placeholder.info("No active tracks detected in current frame.")

        finally:
            if cap is not None:
                cap.release()
            if temp_video_path and os.path.exists(temp_video_path):
                try:
                    os.remove(temp_video_path)
                except OSError:
                    pass

        # Stopped state
        st.session_state["tracking_active"] = False
        status_metric.markdown(
            '<div class="metric-card"><div class="metric-val" style="font-size:1.4rem; padding-top:6px;">'
            '<span class="status-badge-stopped">STOPPED</span></div>'
            '<div class="metric-lbl">Tracker Status</div></div>',
            unsafe_allow_html=True
        )


if __name__ == "__main__":
    main()
