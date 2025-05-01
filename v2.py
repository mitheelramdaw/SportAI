import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import os
import tempfile
from collections import deque, defaultdict
from ultralytics import YOLO

st.set_page_config(page_title="Sportact AI", layout="wide")
st.title("🏟️ Sportact AI – Minimal Arrows, Max Visual Clarity")

# Models
mp_pose = mp.solutions.pose
model = YOLO("yolov8n.pt")

# Player motion tracking
player_history = defaultdict(lambda: deque(maxlen=5))
last_movement_time = defaultdict(lambda: 0)

# Draw small direction arrow at bbox foot
def draw_bbox_arrow(frame, pid, current_point):
    if len(player_history[pid]) >= 2:
        prev = player_history[pid][-2]
        dx, dy = current_point[0] - prev[0], current_point[1] - prev[1]
        if abs(dx) > 1 or abs(dy) > 1:
            tip = (current_point[0] + int(dx * 0.5), current_point[1] + int(dy * 0.5))
            cv2.arrowedLine(frame, current_point, tip, (0, 255, 255), 2, tipLength=0.4)

# Core analysis
def analyse_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    w, h = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    heatmap = np.zeros((h, w), dtype=np.float32)
    frame_count = 0

    with mp_pose.Pose(static_image_mode=False, model_complexity=1) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)[0]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(rgb)

            for i, det in enumerate(results.boxes):
                x1, y1, x2, y2 = map(int, det.xyxy[0])
                cls = int(det.cls[0])
                if cls != 0:
                    continue

                pid = i
                cx = int((x1 + x2) / 2)
                cy = int(y2)  # bottom of bounding box

                player_history[pid].append((cx, cy))
                draw_bbox_arrow(frame, pid, (cx, cy))

                # Draw bounding box & label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                cv2.putText(frame, f"Player {pid}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Heatmap update
                heatmap = cv2.circle(heatmap, (cx, cy), 10, 1, -1)
                last_movement_time[pid] = frame_count

            # Fade heatmap if idle
            for pid in list(last_movement_time.keys()):
                if frame_count - last_movement_time[pid] > fps * 3:
                    heatmap *= 0.95

            # Overlay heatmap
            if np.max(heatmap) > 0:
                heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
                heatmap_img = cv2.applyColorMap(heatmap_norm.astype(np.uint8), cv2.COLORMAP_JET)
                frame = cv2.addWeighted(frame, 0.6, heatmap_img, 0.4, 0)

            out.write(frame)
            frame_count += 1

    cap.release()
    out.release()

# Streamlit UI
uploaded_file = st.file_uploader("🎥 Upload your match video (.mp4)", type=["mp4"])

if uploaded_file:
    st.video(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
        temp_input.write(uploaded_file.read())
        input_path = temp_input.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_output:
        output_path = temp_output.name

    with st.spinner("📊 Analysing your match video..."):
        analyse_video(input_path, output_path)

    st.success("✅ Analysis complete! Preview below:")
    st.video(output_path)

    with open(output_path, "rb") as f:
        st.download_button("📥 Download Analysed Video", f, file_name="sportact_analysis.mp4", mime="video/mp4")
