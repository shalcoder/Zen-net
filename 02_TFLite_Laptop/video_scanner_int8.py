# Human Fall Detection - Video Scanner (TFLite Int8 SinglePose)
import cv2
import tensorflow as tf
import numpy as np
import math
import os

# --- INITIALIZATION ---
# Load TFLite Int8 Model
POSSIBLE_PATHS = [
    "02_TFLite_Laptop/model_thunder_int8.tflite", 
    "model_thunder_int8.tflite",
    "../02_TFLite_Laptop/model_thunder_int8.tflite",
    "02_TFLite_Laptop/model_int8.tflite"
]
MODEL_PATH = None
for p in POSSIBLE_PATHS:
    if os.path.exists(p):
        MODEL_PATH = p
        break

if not MODEL_PATH:
    raise FileNotFoundError("Could not find model_int8.tflite in standard locations.")

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape'] # [1, 192, 192, 3] usually
input_index = input_details[0]['index']
output_index = output_details[0]['index']

# Video Path
VIDEO_PATHS = [
    "E:/human-fall-detection/uploads/20240912_104833.mp4",
    "uploads/B_D_0007_resized.mp4",
    "uploads/20240912_102048.mp4"
]
VIDEO_PATH = None
for p in VIDEO_PATHS:
    if os.path.exists(p):
        VIDEO_PATH = p
        break
        
if not VIDEO_PATH:
    # Fallback just in case
    VIDEO_PATH = "uploads/20240912_101626.mp4" 

# Configuration
THRESHOLD = 0.15 # Lowered to catch low-confidence sleeping poses
EDGES = {
    (0, 1): "m", (0, 2): "c", (1, 3): "m", (2, 4): "c",
    (0, 5): "m", (0, 6): "c", (5, 7): "m", (7, 9): "m",
    (6, 8): "c", (8, 10): "c", (5, 6): "y", (5, 11): "m",
    (6, 12): "c", (11, 12): "y", (11, 13): "m", (13, 15): "m",
    (12, 14): "c", (14, 16): "c",
}

# State tracking (Single Person)
movement_history = [] 
last_centroid = None 
last_area = None # Track size to prevent background snapping
MAX_JUMP_THRESHOLD = 0.15 # Tightened from 0.25

# --- CORE FUNCTIONS (Reused) ---

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

def calculate_aspect_ratio(frame, keypoints, threshold=THRESHOLD):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
    points = [kp for kp in shaped if kp[2] > threshold]
    
    if len(points) < 5: return 1.0
    
    x1 = min([p[1] for p in points])
    y1 = min([p[0] for p in points])
    x2 = max([p[1] for p in points])
    y2 = max([p[0] for p in points])
    
    if (y2 - y1) == 0: return 1.0
    return (x2 - x1) / (y2 - y1)

def calculate_angle(frame, keypoints, threshold=THRESHOLD):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
    points = [kp for kp in shaped if kp[2] > threshold]
    
    if len(points) < 5: return 90.0
    
    x1 = min([p[1] for p in points])
    y1 = min([p[0] for p in points])
    x2 = max([p[1] for p in points])
    y2 = max([p[0] for p in points])
    
    centroid = (x1 + x2) / 2, (y1 + y2) / 2
    angle = math.atan2(centroid[1] - y1, centroid[0] - x1) * 180 / math.pi
    return angle

def get_posture(frame, keypoints, threshold=THRESHOLD):
    conf_scores = keypoints[:, 2]
    if np.mean(conf_scores) < threshold: return "UNKNOWN", 0, 0
    aspect_ratio = calculate_aspect_ratio(frame, keypoints)
    angle = calculate_angle(frame, keypoints)
    
    # --- PROXIMITY PROTECTION ---
    # Only allow LYING if we can see hips (ID 11,12) and head is not above shoulders
    has_hips = keypoints[11, 2] > 0.3 or keypoints[12, 2] > 0.3
    shoulder_y = (keypoints[5, 0] + keypoints[6, 0]) / 2
    nose_y = keypoints[0, 0]
    is_horizontal = nose_y > shoulder_y - 0.05

    if aspect_ratio > 0.7: 
        if angle < 65 and has_hips and is_horizontal:
             return "LYING/RELAXED", aspect_ratio, angle
        else:
             return "SITTING", aspect_ratio, angle
    elif 0.40 < aspect_ratio <= 0.7: 
        return "SITTING", aspect_ratio, angle
    else: 
        return "STANDING", aspect_ratio, angle

def analyze_temporal_states(current_posture):
    history = movement_history
    if len(history) < 30:
        return "NORMAL" if current_posture != "FALLING" else "FALL (EMERGENCY)"

    recent_coords = [h[:2] for h in history[-30:]]
    # Simple temporal analysis
    if current_posture == "FALLING":
         return "FALL (EMERGENCY)"
    return "NORMAL"

# --- MAIN LOOP ---

def process_video():
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video not found at {VIDEO_PATH}")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    print(f"Scanning Video (Int8 TFLite): {VIDEO_PATH}...")
    frame_count = 0
    global last_centroid
    last_centroid = None # Reset for each video run

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        
        # Prepare image for MoveNet with PAD-RESIZE (Letterbox)
        # This prevents squashing and keeps keypoints aligned
        h, w = input_shape[1], input_shape[2]
        old_h, old_w = frame.shape[:2]
        ratio = min(w/old_w, h/old_h)
        new_w, new_h = int(old_w * ratio), int(old_h * ratio)
        
        # Resize to fit inside box
        resized = cv2.resize(frame, (new_w, new_h))
        
        # Create padded black background 256x256
        padded = np.zeros((h, w, 3), dtype=np.uint8)
        # Center the image
        off_x = (w - new_w) // 2
        off_y = (h - new_h) // 2
        padded[off_y:off_y+new_h, off_x:off_x+new_w] = resized
        
        img = np.expand_dims(padded, axis=0)
        # Quantized model expects input to be uint8 (0-255)
        if input_details[0]['dtype'] == np.uint8:
            img = img.astype(np.uint8)
        else:
             img = img.astype(np.float32)

        # Inference
        interpreter.set_tensor(input_index, img)
        interpreter.invoke()
        keypoints_with_scores = interpreter.get_tensor(output_index) 
        
        person = keypoints_with_scores[0, 0, :, :] # (17, 3)
        
        # MAP KEYPOINTS BACK to original frame by removing padding and scaling
        # keypoints are 0..1 relative to the PADDED image
        # person[:, 0] is Y, person[:, 1] is X
        person_mapped = person.copy()
        person_mapped[:, 0] = (person[:, 0] * h - off_y) / new_h
        person_mapped[:, 1] = (person[:, 1] * w - off_x) / new_w
        
        # 1. Total Confidence Check
        if np.mean(person_mapped[:, 2]) > THRESHOLD:
            y_h, x_h, _ = frame.shape
            points = [kp for kp in person_mapped if kp[2] > THRESHOLD]
            
            # --- GHOST PROTECTION (Head/Torso Check) ---
            head_pts = person_mapped[0:5, 2]   # Nose, Eyes, Ears
            body_pts = person_mapped[5:11, 2]  # Shoulders, Elbows, Wrists
            if np.max(head_pts) < 0.25 and np.max(body_pts) < 0.25:
                continue # This is likely background noise/objects

            base_p, ar, ang = get_posture(frame, person_mapped)

            # --- JUMP & SCALE FILTER (Refined for Falls) ---
            current_centroid = (np.mean([p[1] for p in points]), np.mean([p[0] for p in points])) if points else (0.5, 0.5)
            x1, y1 = min([p[1] for p in points]), min([p[0] for p in points])
            x2, y2 = max([p[1] for p in points]), max([p[0] for p in points])
            current_area = (x2 - x1) * (y2 - y1)
            
            if last_centroid is not None:
                dist = math.sqrt((current_centroid[0]-last_centroid[0])**2 + (current_centroid[1]-last_centroid[1])**2)
                
                # If a jump happens, only allow it if it looks like a human falling (LYING)
                if dist > MAX_JUMP_THRESHOLD and "LYING" not in base_p:
                    continue # Ignore "ghost" jumps to background objects
            
            last_centroid = current_centroid
            last_area = current_area

            # Update history with GLOBAL centroid (average of all active points)
            centroid_x, centroid_y = current_centroid[0] * x_h, current_centroid[1] * y_h
            movement_history.append((centroid_x, centroid_y, base_p))
            if len(movement_history) > 100: movement_history.pop(0)

            # Classification logic with GLOBAL Stillness Detection
            movement_v = 0.0
            if "LYING" in base_p or "SITTING" in base_p:
                 if len(movement_history) >= 15:
                      recent = movement_history[-15:]
                      movement_v = np.std([r[0] for r in recent]) + np.std([r[1] for r in recent])
                      
                      # Strict Sleep Check: must be truly motionless (< 22)
                      if movement_v < 22.0: 
                           final_class = "NORMAL (SLEEPING)"
                      elif movement_v > 90.0 and "LYING" in base_p: 
                           final_class = "FALL DETECTED"
                      elif "LYING" in base_p: 
                           final_class = "FAINT DETECTED"
                      else:
                           final_class = "NORMAL"
                 else:
                      final_class = "NORMAL"
            else:
                 final_class = "NORMAL"
            
            print(f"Frame {frame_count:04d}: {final_class} | AR: {ar:.2f} | Ang: {ang:.1f} | Move: {movement_v:.1f}")
            
            # Visualize with better colors
            if "FALL" in final_class: color = (0, 0, 255) # Red
            elif "FAINT" in final_class: color = (0, 165, 255) # Orange
            elif "SLEEPING" in final_class: color = (0, 255, 0) # Green (Safe)
            else: color = (0, 255, 0) # Green
            draw_skeleton(frame, person_mapped, EDGES)
            draw_keypoints(frame, person_mapped)
            
            # Detailed Debug Info
            display_posture = base_p
            if "SLEEPING" in final_class:
                 display_posture = "RESTING"
                 
            cv2.putText(frame, f"{final_class}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, f"Posture: {display_posture}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"AR: {ar:.2f} | Ang: {ang:.1f}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # BBox
            shaped = np.squeeze(np.multiply(person_mapped, [y_h, x_h, 1]))
            x1, y1, x2, y2 = x_h, y_h, 0, 0
            for kp in shaped:
                 if kp[2] > THRESHOLD:
                     x1, y1, x2, y2 = min(x1, kp[1]), min(y1, kp[0]), max(x2, kp[1]), max(y2, kp[0])
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        if frame_count == 1:
            print("--- Analysis Started: Tracking Posture for Sleeping/Fall Safety ---")

        cv2.imshow("Int8 Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    print("Video Scan Complete.")

if __name__ == "__main__":
    process_video()
