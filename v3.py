import streamlit as st
import cv2
import numpy as np
import tempfile
from collections import defaultdict, deque
from ultralytics import YOLO
from sklearn.cluster import KMeans
import math

st.set_page_config(page_title="Sportact Final", layout="wide")
st.title("⚽ Sportact AI – Clean Team-Based Foot Circles & Realistic Speed")

model = YOLO("yolov8n.pt")
fps = 30  # fallback

player_trails = defaultdict(lambda: deque(maxlen=2))
player_stats = defaultdict(lambda: {"distance": 0.0})
team_colours = {}

# Estimate speed and filter unrealistic jumps
def estimate_speed_and_distance(prev, curr, fps):
    if not prev or not curr:
        return 0.0, 0.0
    dx, dy = curr[0] - prev[0], curr[1] - prev[1]
    pixel_dist = math.sqrt(dx**2 + dy**2)
    meters = pixel_dist / 50.0  # ~50 px = 1m
    if meters > 5:  # discard big jumps
        return 0.0, 0.0
    speed_mps = meters * fps
    speed_kph = speed_mps * 3.6
    return min(round(speed_kph, 2), 40.0), round(meters, 2)  # cap speed

# Get dominant jersey colour from crop
def get_dominant_colour(crop):
    crop = cv2.resize(crop, (30, 30))
    data = crop.reshape((-1, 3))
    kmeans = KMeans(n_clusters=1).fit(data)
    color = kmeans.cluster_centers_[0]
    return tuple(map(int, color))

# Group jersey colours into 2 teams
def assign_team_color(rgb):
    if rgb[0] > rgb[1] and rgb[0] > rgb[2]:  # reddish
        return (0, 0, 255)
    elif rgb[1] > rgb[0] and rgb[1] > rgb[2]:  # greenish/yellow
        return (0, 255, 255)
    else:  # bluish
        return (255, 255, 0)

# Main analysis
def analyse_video(input_path, output_path):
    global fps
    cap = cv2.VideoCapture(input_path)
    w, h = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        for i, det in enumerate(results.boxes):
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            cls = int(det.cls[0])
            if cls != 0:
                continue

            pid = i
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            feet_point = (cx, int(y2))

            # Store movement trail
            player_trails[pid].append(feet_point)

            # Distance and speed
            if len(player_trails[pid]) == 2:
                speed, dist = estimate_speed_and_distance(player_trails[pid][0], player_trails[pid][1], fps)
                player_stats[pid]["distance"] += dist
            else:
                speed = 0.0

            # Detect jersey colour
            crop = frame[y1:y2, x1:x2]
            if pid not in team_colours and crop.size > 0:
                dom_color = get_dominant_colour(crop)
                team_colours[pid] = assign_team_color(dom_color)
            circle_color = team_colours.get(pid, (200, 200, 200))

            # Draw circle and overlay info
            cv2.circle(frame, feet_point, 18, circle_color, 2)
            cv2.putText(frame, str(pid), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(frame, f"{speed:.1f} km/h", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
            cv2.putText(frame, f"{player_stats[pid]['distance']:.1f} m", (x1, y2 + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        out.write(frame)

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

    with st.spinner("📊 Analysing match footage..."):
        analyse_video(input_path, output_path)

    st.success("✅ Done! Here’s your clean enhanced playback:")
    st.video(output_path)

    with open(output_path, "rb") as f:
        st.download_button("📥 Download Final Analysis", f, file_name="sportact_final_analysis.mp4", mime="video/mp4")
