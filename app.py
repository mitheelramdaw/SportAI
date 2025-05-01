import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import os
import tempfile
from collections import deque, defaultdict
from sklearn.cluster import KMeans
from ultralytics import YOLO
from sort.sort import Sort  # Using pure Python version
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from io import BytesIO

# Streamlit UI config
st.set_page_config(page_title="Sportact Pro AI", layout="wide")
st.title("🏟️ Sportact AI – Full In-Game Analysis")

# Init models
mp_pose = mp.solutions.pose
model = YOLO("yolov8n.pt")
tracker = Sort(max_age=10, min_hits=3, iou_threshold=0.3)

# Tracking data
player_history = {}
player_stats = defaultdict(lambda: {"path": [], "distance": 0})

def get_dominant_colour(crop):
    crop = cv2.resize(crop, (30, 30))
    data = crop.reshape((-1, 3))
    kmeans = KMeans(n_clusters=1).fit(data)
    color = kmeans.cluster_centers_[0]
    return tuple(map(int, color))

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
            detections = []

            for det in results.boxes:
                x1, y1, x2, y2 = map(int, det.xyxy[0])
                cls = int(det.cls[0])
                if cls == 0:
                    conf = float(det.conf[0])
                    detections.append([x1, y1, x2, y2, conf])

            tracks = tracker.update(np.array(detections))

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(rgb)

            for track in tracks:
                x1, y1, x2, y2, track_id = map(int, track)
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                center = (cx, cy)

                player_stats[track_id]["path"].append(center)
                if len(player_stats[track_id]["path"]) > 1:
                    d = np.linalg.norm(np.array(center) - np.array(player_stats[track_id]["path"][-2]))
                    player_stats[track_id]["distance"] += d

                if track_id not in player_history:
                    player_history[track_id] = deque(maxlen=5)
                player_history[track_id].append(center)
                if len(player_history[track_id]) >= 2:
                    prev = player_history[track_id][-2]
                    cv2.arrowedLine(frame, prev, center, (0, 255, 255), 3, tipLength=0.4)

                crop = frame[y1:y2, x1:x2]
                jersey_color = get_dominant_colour(crop) if crop.size > 0 else (255, 255, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), jersey_color, 2)
                cv2.putText(frame, f"Player {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, jersey_color, 2)

                cv2.circle(heatmap, center, 15, 1, -1)

            if pose_results.pose_landmarks:
                draw_feet_circle(frame, pose_results.pose_landmarks.landmark)

            frame = draw_heatmap(frame, heatmap)
            out.write(frame)

    cap.release()
    out.release()
    return w, h  # Return video resolution for plotting

# Normalize pixel position to pitch dimension
def normalize_to_pitch(x, y, frame_w, frame_h, pitch_w=105, pitch_h=68):
    return (x / frame_w) * pitch_w, (y / frame_h) * pitch_h

# Draw 2D pitch and player trails
def draw_pitch(ax, width=105, height=68):
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_facecolor("green")
    ax.plot([0, 0, width, width, 0], [0, height, height, 0, 0], color="white")
    ax.plot([width / 2, width / 2], [0, height], color="white")
    ax.add_patch(Circle((width / 2, height / 2), 9.15, color="white", fill=False))
    ax.axis("off")

def render_2d_animation(player_stats, frame_w, frame_h):
    fig, ax = plt.subplots(figsize=(10, 6))
    draw_pitch(ax)

    for pid, stats in player_stats.items():
        if len(stats["path"]) < 2:
            continue
        xys = [normalize_to_pitch(x, y, frame_w, frame_h) for x, y in stats["path"]]
        xs, ys = zip(*xys)
        ax.plot(xs, ys, label=f"Player {pid}")
        ax.plot(xs[-1], ys[-1], "o")
    ax.legend()
    buf = BytesIO()
    fig.savefig(buf, format="png")
    st.image(buf.getvalue(), caption="⚽ 2D Tactical Overview (Top-Down View)")

# Streamlit UI
uploaded_file = st.file_uploader("Upload Match Video (.mp4)", type=["mp4"])

if uploaded_file:
    st.video(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
        temp_input.write(uploaded_file.read())
        input_path = temp_input.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_output:
        output_path = temp_output.name

    with st.spinner("🔬 Analysing video... hang tight"):
        w, h = analyse_video(input_path, output_path)

    st.success("✅ Done! Here's your AI-powered analysis:")
    st.video(output_path)

    st.markdown("### 🧠 2D Tactical View")
    render_2d_animation(player_stats, w, h)

    with open(output_path, "rb") as f:
        st.download_button("📥 Download Analysed Video", f, file_name="analysed_video.mp4")

    st.markdown("### 📊 Player Stats")
    for pid, stats in player_stats.items():
        dist_m = stats["distance"] / 50
        st.write(f"👤 Player {int(pid)}: Distance covered ~ **{dist_m:.2f} m**")
