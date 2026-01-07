# Human Fall Detection - Video Scanner
import cv2
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import math

# --- INITIALIZATION ---
# Loading the detector model from TF Hub
model = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")
movenet = model.signatures["serving_default"]

# Video Path - Change this to your video file path
VIDEO_PATH = "E:/human-fall-detection/uploads/20240912_102048.mp4"

# Configuration
THRESHOLD = 0.3
EDGES = {
    (0, 1): "m", (0, 2): "c", (1, 3): "m", (2, 4): "c",
    (0, 5): "m", (0, 6): "c", (5, 7): "m", (7, 9): "m",
    (6, 8): "c", (8, 10): "c", (5, 6): "y", (5, 11): "m",
    (6, 12): "c", (11, 12): "y", (11, 13): "m", (13, 15): "m",
    (12, 14): "c", (14, 16): "c",
}

# State tracking
movement_history = {i: [] for i in range(1, 7)}

# --- CORE FUNCTIONS ---

def draw_keypoints(frame, keypoints, threshold=THRESHOLD):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (255, 0, 0), -1)

def draw_skeleton(frame, keypoints, edges, threshold=THRESHOLD):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        if (c1 > threshold) & (c2 > threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

def calculate_aspect_ratio(frame, keypoints):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
    x1, y1, x2, y2 = x, y, 0, 0
    for kp in shaped:
        ky, kx, kp_conf = kp
        x1, y1 = min(x1, kx), min(y1, ky)
        x2, y2 = max(x2, kx), max(y2, ky)
    if (y2 - y1) == 0: return 1.0
    return (x2 - x1) / (y2 - y1)

def calculate_angle(frame, keypoints):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
    x1, y1, x2, y2 = x, y, 0, 0
    for kp in shaped:
        ky, kx, kp_conf = kp
        x1, y1 = min(x1, kx), min(y1, ky)
        x2, y2 = max(x2, kx), max(y2, ky)
    centroid = (x1 + x2) / 2, (y1 + y2) / 2
    angle = math.atan2(centroid[1] - y1, centroid[0] - x1) * 180 / math.pi
    return angle

def get_posture(frame, keypoints, threshold=THRESHOLD):
    conf_scores = keypoints[:, 2]
    if np.mean(conf_scores) < threshold: return "UNKNOWN"
    aspect_ratio = calculate_aspect_ratio(frame, keypoints)
    angle = calculate_angle(frame, keypoints)
    if aspect_ratio > 1.5 and angle < 45: return "FALLING"
    elif 0.8 < aspect_ratio <= 1.5: return "SITTING"
    else: return "STANDING"

def analyze_temporal_states(person_idx, current_posture):
    history = movement_history[person_idx]
    if len(history) < 30:
        return "NORMAL" if current_posture != "FALLING" else "FALL (EMERGENCY)"

    recent_coords = [h[:2] for h in history[-30:]]
    dx, dy = np.std([c[0] for c in recent_coords]), np.std([c[1] for c in recent_coords])
    movement_intensity = dx + dy

    if current_posture == "FALLING":
        rapid_drop = any(h[2] == "STANDING" for h in history[-20:])
        if rapid_drop or movement_intensity > 10.0 or movement_intensity < 2.0 or movement_intensity > 15.0:
            return "FALL (EMERGENCY)"
        
        was_sitting = any(h[2] == "SITTING" for h in history[-40:])
        if was_sitting and movement_intensity < 8.0:
            return "NORMAL"

    return "NORMAL"

# --- MAIN LOOP ---

def process_video():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {VIDEO_PATH}")
        return

    print(f"Scanning Video: {VIDEO_PATH}...")
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        
        # Prepare image for MoveNet
        img = frame.copy()
        img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 160, 320)
        img = tf.cast(img, dtype=tf.int32)

        # Detection
        result = movenet(img)
        keypoints_with_scores = result["output_0"].numpy()[:, :, :51].reshape(6, 17, 3)

        # Handle detected people
        final_class = "NORMAL" 
        for i, person in enumerate(keypoints_with_scores):
            if np.mean(person[:, 2]) < THRESHOLD: continue
            
            # Tracking coordinates
            y_h, x_h, _ = frame.shape
            nose_y = person[0][0] * y_h
            nose_x = person[0][1] * x_h
            
            # Base posture
            base_p = get_posture(frame, person)
            
            # Update history
            movement_history[i+1].append((nose_x, nose_y, base_p))
            if len(movement_history[i+1]) > 100: movement_history[i+1].pop(0)
            
            # Classification
            final_class = analyze_temporal_states(i+1, base_p)
            
            # Drawing
            color = (0, 0, 255) if final_class == "FALL (EMERGENCY)" else (0, 255, 0)
            draw_skeleton(frame, person, EDGES)
            draw_keypoints(frame, person)
            
            # Label
            cv2.putText(frame, f"Frame: {frame_count} | {final_class}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Bounding box
            shaped = np.squeeze(np.multiply(person, [y_h, x_h, 1]))
            x1, y1, x2, y2 = x_h, y_h, 0, 0
            for kp in shaped:
                if kp[2] > THRESHOLD:
                    x1, y1, x2, y2 = min(x1, kp[1]), min(y1, kp[0]), max(x2, kp[1]), max(y2, kp[0])
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # Show Output
        cv2.imshow("Video Classification Monitor", frame)
        
        # Printing for console access
        print(f"Frame {frame_count:04d}: Classification = {final_class}")

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    print("Video Scan Complete.")

if __name__ == "__main__":
    process_video()
