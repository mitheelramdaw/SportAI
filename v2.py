import streamlit as st
import cv2
import numpy as np
import tempfile
from collections import defaultdict, deque
from ultralytics import YOLO
import math

st.set_page_config(page_title="Sportact Pro", layout="wide")
st.title("⚽ Sportact AI – Pro Player Tracking with Speed & Distance")

model = YOLO("yolov8n.pt")
player_trails = defaultdict(lambda: deque(maxlen=2))  # store last 2 positions
player_stats = defaultdict(lambda: {"distance": 0.0})  # store total distance
fps = 30  # fallback, gets overwritten dynamically

# Utility: estimate speed & distance between 2 points
def estimate_speed_and_distance(prev, curr, fps):
    if not prev or not curr:
        return 0.0, 0.0
    dx, dy = curr[0] - prev[0], curr[1] - prev[1]
    pixel_dist = math.sqrt(dx**2 + dy**2)
    meters = pixel_dist / 50.0  # 50 px ≈ 1 meter (adjust as needed)
    speed_mps = meters * fps
    return round(speed_mps * 3.6, 2), round(meters, 2)

# Main video analysis
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
            if cls != 0: continue  # only person class

            pid = i
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            base = (cx, int(y2))  # bottom center of box

            # Update stats
            trail = player_trails[pid]
            trail.append(base)
            if len(trail) == 2:
                speed, dist = estimate_speed_and_distance(trail[0], trail[1], fps)
                player_stats[pid]["distance"] += dist
            else:
                speed = 0.0

            # Draw circle at feet
            cv2.circle(frame, base, 18, (0, 255, 255), 2)

            # Draw direction triangle
            if len(trail) == 2:
                dx, dy = trail[1][0] - trail[0][0], trail[1][1] - trail[0][1]
                if abs(dx) > 1 or abs(dy) > 1:
                    tip = (base[0] + int(dx * 0.6), base[1] + int(dy * 0.6))
                    cv2.arrowedLine(frame, base, tip, (0, 255, 0), 2, tipLength=0.4)

            # Label info
            label = f"{pid}"
            cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            # Speed & distance overlay
            cv2.putText(frame, f"{speed:.1f} km/h", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
            cv2.putText(frame, f"{player_stats[pid]['distance']:.1f} m", (x1, y2 + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        out.write(frame)

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

    with st.spinner("🔍 Analysing players... estimating speed and distance"):
        analyse_video(input_path, output_path)

    st.success("✅ Done! Here's the enhanced match video:")
    st.video(output_path)

    with open(output_path, "rb") as f:
        st.download_button("📥 Download Analysed Video", f, file_name="sportact_pro_analysis.mp4", mime="video/mp4")
