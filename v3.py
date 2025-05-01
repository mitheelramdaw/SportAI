# app.py (final version with bounding boxes added for clarity)

import streamlit as st
import cv2
import numpy as np
import tempfile
from collections import defaultdict, deque
from ultralytics import YOLO
from sklearn.cluster import KMeans
import math
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

# (SORT tracker code remains unchanged)

# ------------------------ Sportact Processing ------------------------

model = YOLO("yolov8n.pt")
tracker = Sort()
fps = 30
player_trails = defaultdict(lambda: deque(maxlen=2))
player_stats = defaultdict(lambda: {"distance": 0.0})
team_colours = {}
track_id_to_player_id = {}
next_id = 0

def estimate_speed_and_distance(prev, curr):
    if not prev or not curr:
        return 0.0, 0.0
    dx, dy = curr[0] - prev[0], curr[1] - prev[1]
    px_dist = math.sqrt(dx**2 + dy**2)
    meters = px_dist / 50.0
    if meters > 5:
        return 0.0, 0.0
    return min(round(meters * fps * 3.6, 2), 40.0), round(meters, 2)

def get_dominant_colour(crop):
    crop = cv2.resize(crop, (30, 30))
    data = crop.reshape((-1, 3))
    return tuple(map(int, KMeans(n_clusters=1).fit(data).cluster_centers_[0]))

def assign_team_color(rgb):
    if rgb[0] > rgb[1] and rgb[0] > rgb[2]:
        return (0, 0, 255)
    elif rgb[1] > rgb[0] and rgb[1] > rgb[2]:
        return (0, 255, 255)
    else:
        return (255, 255, 0)

def analyse_video(input_path, output_path):
    global fps, next_id
    cap = cv2.VideoCapture(input_path)
    w, h = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)[0]
        detections = [list(map(float, det.xyxy[0])) + [float(det.conf[0])] for det in results.boxes if int(det.cls[0]) == 0]
        tracks = tracker.update(np.array(detections))

        for tr in tracks:
            x1, y1, x2, y2, track_id = map(int, tr)
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            feet = (cx, int(y2))

            if track_id not in track_id_to_player_id:
                track_id_to_player_id[track_id] = next_id
                next_id += 1

            player_id = track_id_to_player_id[track_id]
            player_trails[player_id].append(feet)

            speed, dist = estimate_speed_and_distance(*player_trails[player_id]) if len(player_trails[player_id]) == 2 else (0.0, 0.0)
            player_stats[player_id]["distance"] += dist

            crop = frame[y1:y2, x1:x2]
            if player_id not in team_colours and crop.size > 0:
                team_colours[player_id] = assign_team_color(get_dominant_colour(crop))

            color = team_colours.get(player_id, (200, 200, 200))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # ✅ BOUNDING BOX
            cv2.circle(frame, feet, 18, color, 2)
            cv2.putText(frame, f"{player_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(frame, f"{speed:.1f} km/h", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
            cv2.putText(frame, f"{player_stats[player_id]['distance']:.1f} m", (x1, y2 + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        out.write(frame)

    cap.release()
    out.release()

# ------------------------ Streamlit UI ------------------------

st.set_page_config(page_title="Sportact AI", layout="wide")
st.title("🏟️ Sportact AI – Final Version with Bounding Boxes")

uploaded_file = st.file_uploader("🎥 Upload your match video (.mp4)", type=["mp4"])
if uploaded_file:
    st.video(uploaded_file)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
        temp_input.write(uploaded_file.read())
        input_path = temp_input.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_output:
        output_path = temp_output.name
    with st.spinner("🧠 Analysing players, tracking, and overlaying bounding boxes..."):
        analyse_video(input_path, output_path)
    st.success("✅ Done! Here's your enhanced match video:")
    st.video(output_path)
    with open(output_path, "rb") as f:
        st.download_button("📥 Download", f, file_name="sportact_final_boxed.mp4", mime="video/mp4")
