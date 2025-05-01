import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import os
import tempfile
from collections import deque, defaultdict
from sklearn.cluster import KMeans
from ultralytics import YOLO

st.set_page_config(page_title="Sportact AI – Enhanced", layout="wide")
st.title("🏟️ Sportact AI – Smart Direction & Heatmap Visualisation")

mp_pose = mp.solutions.pose
model = YOLO("yolov8n.pt")

# Tracking state
player_history = defaultdict(lambda: deque(maxlen=30))
last_movement_time = defaultdict(lambda: 0)
frame_counter = 0

# Helper: Get dominant colour for jersey
def get_dominant_colour(crop):
    crop = cv2.resize(crop, (30, 30))
    data = crop.reshape((-1, 3))
    kmeans = KMeans(n_clusters=1).fit(data)
    return tuple(map(int, kmeans.cluster_centers_[0]))

# Draw mini direction arrow from feet
def draw_direction_arrow(frame, pid, current_point):
    if len(player_history[pid]) >= 2:
        prev = player_history[pid][-2]
        dx, dy = current_point[0] - prev[0], current_point[1] - prev[1]
        if abs(dx) > 1 or abs(dy) > 1:
            tip = (current_point[0] + int(dx * 0.6), current_point[1] + int(dy * 0.6))
            cv2.arrowedLine(frame, current_point, tip, (255, 255, 0), 2, tipLength=0.4)

# Draw circle at feet, return feet centre for heatmap
def draw_feet_and_direction(frame, landmarks, pid):
    try:
        l = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        r = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        if l.visibility > 0.5 and r.visibility > 0.5:
            cx = int((l.x + r.x) / 2 * frame.shape[1])
            cy = int((l.y + r.y) / 2 * frame.shape[0])
            center = (cx, cy)
            cv2.circle(frame, center, 10, (0, 255, 0), 2)
            player_history[pid].append(center)
            draw_direction_arrow(frame, pid, center)
            return center
    except:
        pass
    return None

# Main analysis pipeline
def analyse_video(input_path, output_path):
    global frame_counter
    cap = cv2.VideoCapture(input_path)
    w, h = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    heatmaps = defaultdict(lambda: np.zeros((h, w), dtype=np.float32))

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
                if cls != 0: continue  # person class only

                pid = i
                crop = frame[y1:y2, x1:x2]
                jersey_color = get_dominant_colour(crop) if crop.size > 0 else (255, 255, 255)

                # Bounding box & label
                cv2.rectangle(frame, (x1, y1), (x2, y2), jersey_color, 2)
                cv2.putText(frame, f"Player {pid}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, jersey_color, 2)

                # Feet & direction
                if pose_results.pose_landmarks:
                    feet_point = draw_feet_and_direction(frame, pose_results.pose_landmarks.landmark, pid)
                    if feet_point:
                        # Update heatmap
                        heatmaps[pid] = cv2.circle(heatmaps[pid], feet_point, 10, 1, -1)
                        last_movement_time[pid] = frame_counter

            # Fade heatmaps if player idle > 3 seconds
            for pid in list(heatmaps.keys()):
                if frame_counter - last_movement_time[pid] > int(fps * 3):
                    heatmaps[pid] *= 0.95  # fade out

            # Combine and overlay
            if len(heatmaps) > 0:
                heatmap_array = list(heatmaps.values())
                stacked = np.stack(heatmap_array, axis=0)
                total_heat = np.sum(stacked, axis=0)
                total_heat = cv2.normalize(total_heat, None, 0, 255, cv2.NORM_MINMAX)
                heat_img = cv2.applyColorMap(total_heat.astype(np.uint8), cv2.COLORMAP_JET)
                frame = cv2.addWeighted(frame, 0.6, heat_img, 0.4, 0)

            out.write(frame)
            frame_counter += 1

    cap.release()
    out.release()

# Streamlit UI
uploaded_file = st.file_uploader("🎥 Upload match video (.mp4)", type=["mp4"])

if uploaded_file:
    st.video(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
        temp_input.write(uploaded_file.read())
        input_path = temp_input.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_output:
        output_path = temp_output.name

    with st.spinner("🧠 Analysing video with Sportact AI..."):
        analyse_video(input_path, output_path)

    st.success("✅ Analysis Complete – See your enhanced playback:")
    st.video(output_path)

    with open(output_path, "rb") as f:
        st.download_button("📥 Download Enhanced Video", f, file_name="sportact_analysis.mp4", mime="video/mp4")
