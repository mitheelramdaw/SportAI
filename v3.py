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

# ------------------------ SORT Tracker ------------------------

def iou(bb_test, bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1]) +
              (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
    return o

class KalmanBoxTracker:
    count = 0
    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.eye(7)
        for i in range(4):
            self.kf.F[i, i+3] = 1
        self.kf.H = np.eye(4,7)
        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01
        self.kf.x[:4] = bbox.reshape((4, 1))
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(bbox)

    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self.kf.x

    def get_state(self):
        return self.kf.x[:4].reshape((1, 4))

class Sort:
    def __init__(self, max_age=20, min_hits=1, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t in range(len(self.trackers)):
            pos = self.trackers[t].predict().flatten()
            if pos.shape[0] < 4 or np.any(np.isnan(pos)):
                to_del.append(t)
                continue
            trks[t, :] = [pos[0], pos[1], pos[2], pos[3], 0]

        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:, 1] == t)[0], 0][0]
                trk.update(dets[d, :4])

        for i in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(dets[i, :4]))

        ret = []
        for trk in self.trackers:
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id])).reshape(1, -1))

        return np.concatenate(ret) if len(ret) > 0 else np.empty((0, 5))

def associate_detections_to_trackers(dets, trks, iou_threshold=0.3):
    if len(trks) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(dets)), np.empty((0,), dtype=int)
    iou_matrix = np.zeros((len(dets), len(trks)), dtype=np.float32)
    for d in range(len(dets)):
        for t in range(len(trks)):
            iou_matrix[d, t] = iou(dets[d], trks[t])
    matched_indices = linear_sum_assignment(-iou_matrix)
    matched_indices = np.array(list(zip(*matched_indices)))
    unmatched_dets = [d for d in range(len(dets)) if d not in matched_indices[:, 0]]
    unmatched_trks = [t for t in range(len(trks)) if t not in matched_indices[:, 1]]
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_dets.append(m[0])
            unmatched_trks.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    return np.concatenate(matches) if matches else np.empty((0, 2), dtype=int), np.array(unmatched_dets), np.array(unmatched_trks)

# ------------------------ Analysis Logic ------------------------

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
        if not ret: break
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
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, feet, 18, color, 2)
            cv2.putText(frame, f"{player_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(frame, f"{speed:.1f} km/h", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
            cv2.putText(frame, f"{player_stats[player_id]['distance']:.1f} m", (x1, y2 + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        out.write(frame)

    cap.release()
    out.release()

# ------------------------ Streamlit UI ------------------------

st.set_page_config(page_title="Sportact AI", layout="wide")
st.title("🏟️ Sportact AI – Complete System")

uploaded_file = st.file_uploader("🎥 Upload match video (.mp4)", type=["mp4"])
if uploaded_file:
    st.video(uploaded_file)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
        temp_input.write(uploaded_file.read())
        input_path = temp_input.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_output:
        output_path = temp_output.name
    with st.spinner("🔍 Analysing..."):
        analyse_video(input_path, output_path)
    st.success("✅ Done!")
    st.video(output_path)
    with open(output_path, "rb") as f:
        st.download_button("📥 Download", f, file_name="sportact_analysis.mp4", mime="video/mp4")
