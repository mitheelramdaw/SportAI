import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import os
import tempfile
from collections import deque, defaultdict
from sklearn.cluster import KMeans
from ultralytics import YOLO

st.set_page_config(page_title="Sportact Pro AI", layout="wide")
st.title("🏟️ Sportact AI – Full In-Game Analysis")

mp_pose = mp.solutions.pose
model = YOLO("yolov8n.pt")

# Tracking data
player_history = {}
player_stats = defaultdict(lambda: {"path": [], "distance": 0})

# Get dominant jersey colour
def get_dominant_colour(crop):
    crop = cv2.resize(crop, (30, 30))
    data = crop.reshape((-1, 3))
    kmeans = KMeans(n_clusters=1).fit(data)
    color = kmeans.cluster_centers_[0]
    return tuple(map(int, color))

# Heatmap visualisation
def draw_heatmap(frame, heatmap):
    heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_img = cv2.applyColorMap(heatmap_norm.astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(frame, 0.6, heatmap_img, 0.4, 0)

def draw_feet_circle(frame, landmarks, thresh=0.5):
    try:
        l = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        r = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        if l.visibility > thresh and r.visibility > thresh:
            cx = int((l.x + r.x) / 2 * frame.shape[1])
            cy = int((l.y + r.y) / 2 * frame.shape[0])
            cv2.circle(frame, (cx, cy), 15, (0, 255, 0), 3)
    except:
        pass

def analyse_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    w, h = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    heatmap = np.zeros((h, w), dtype=np.float32)

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
                if cls != 0: continue

                pid = i
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                center = (cx, cy)

                # Track movement for stats
                player_stats[pid]["path"].append(center)
                if len(player_stats[pid]["path"]) > 1:
                    d = np.linalg.norm(np.array(center) - np.array(player_stats[pid]["path"][-2]))
                    player_stats[pid]["distance"] += d

                # Arrow
                if pid not in player_history:
                    player_history[pid] = deque(maxlen=5)
                player_history[pid].append(center)
                if len(player_history[pid]) >= 2:
                    prev = player_history[pid][-2]
                    cv2.arrowedLine(frame, prev, center, (0, 255, 255), 3, tipLength=0.4)

                # Dominant colour from torso crop
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    jersey_color = get_dominant_colour(crop)
                else:
                    jersey_color = (255, 255, 255)

                # Draw player box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), jersey_color, 2)
                cv2.putText(frame, f"Player {pid}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, jersey_color, 2)

                # Update heatmap
                cv2.circle(heatmap, center, 15, 1, -1)

            if pose_results.pose_landmarks:
                draw_feet_circle(frame, pose_results.pose_landmarks.landmark)

            frame = draw_heatmap(frame, heatmap)
            out.write(frame)

    cap.release()
    out.release()

# UI
uploaded_file = st.file_uploader("Upload Match Video (.mp4)", type=["mp4"])

if uploaded_file:
    st.video(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
        temp_input.write(uploaded_file.read())
        input_path = temp_input.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_output:
        output_path = temp_output.name

    with st.spinner("🔬 Analysing video... hang tight"):
        analyse_video(input_path, output_path)

    st.success("✅ Done! Here's your AI-powered analysis:")
    st.video(output_path)

    with open(output_path, "rb") as f:
        st.download_button("📥 Download Analysed Video", f, file_name="analysed_video.mp4")

    st.markdown("### 📊 Player Stats")
    for pid, stats in player_stats.items():
        dist_m = stats["distance"] / 50  # approx conversion: pixels to metres
        st.write(f"👤 Player {pid}: Distance covered ~ **{dist_m:.2f} m**")
