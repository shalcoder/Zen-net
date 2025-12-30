# TFLite Video Classifier Test
import cv2
import tensorflow as tf
import numpy as np
import math

# --- CONFIGURATION ---
MODEL_PATH = "movenet_multipose_lighting_quant.tflite"
VIDEO_PATH = "E:/human-fall-detection/20240912_102048.mp4"
THRESHOLD = 0.3

# Definition of body connections
EDGES = {
    (0, 1): "m", (0, 2): "c", (1, 3): "m", (2, 4): "c",
    (0, 5): "m", (0, 6): "c", (5, 7): "m", (7, 9): "m",
    (6, 8): "c", (8, 10): "c", (5, 6): "y", (5, 11): "m",
    (6, 12): "c", (11, 12): "y", (11, 13): "m", (13, 15): "m",
    (12, 14): "c", (14, 16): "c",
}

# State tracking
movement_history = {i: [] for i in range(1, 7)}

# --- HELPER FUNCTIONS (Same as before) ---
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

# --- MAIN EXECUTION ---
def run_tflite_inference():
    # 1. Load TFLite Model
    print(f"Loading TFLite model: {MODEL_PATH}")
    try:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    input_index = interpreter.get_input_details()[0]['index']
    
    # Explicitly set the input shape to MoveNet's standard resolution
    # The default shape might be [1, 1, 1, 3] (dynamic), so we force it.
    TARGET_HEIGHT = 160
    TARGET_WIDTH = 320
    interpreter.resize_tensor_input(input_index, [1, TARGET_HEIGHT, TARGET_WIDTH, 3])
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_shape = input_details[0]['shape'] 
    output_index = output_details[0]['index']
    
    target_h, target_w = input_shape[1], input_shape[2]
    print(f"Model configured with input shape: {input_shape}")

    # 2. Open Video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error opening video: {VIDEO_PATH}")
        return

    print("Starting TFLite Inference... Press 'q' to exit.")
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        
        # 3. Preprocess Input
        # Resize/Pad to match model input [1, 160, 320, 3] and type (int32 or float32)
        img = frame.copy()
        
        # Efficient resizing and padding using TF (same as original script to match logic)
        # Note: In production (RPi), you'd use cv2.resize/copyMakeBorder to avoid heavy TF import
        input_img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), target_h, target_w)
        
        # Check input type expected by model
        if input_details[0]['dtype'] == np.int32:
            input_img = tf.cast(input_img, dtype=tf.int32)
        elif input_details[0]['dtype'] == np.uint8:
            input_img = tf.cast(input_img, dtype=tf.uint8)
        else:
             # Typically float32
             input_img = tf.cast(input_img, dtype=tf.float32)

        # 4. Run Inference
        interpreter.set_tensor(input_index, input_img.numpy())
        interpreter.invoke()
        
        # 5. Get Output
        # MoveNet Multipose Output: [1, 6, 56] 
        # (6 people, 17 keypoints * 3 [y,x,s] + 5 bbox info)
        keypoints_with_scores = interpreter.get_tensor(output_index)
        
        # Just like original script, extract first 51 columns (17 keypoints * 3)
        # Reshape to [6, 17, 3]
        keypoints_with_scores = keypoints_with_scores[:, :, :51].reshape(6, 17, 3)

        # 6. Process Predictions
        final_class = "NORMAL"
        for i, person in enumerate(keypoints_with_scores):
            if np.mean(person[:, 2]) < THRESHOLD: continue
            
            # Posture checks
            y_h, x_h, _ = frame.shape
            
            # TFLite output is normalized [0,1], multiply by dims
            # Tracking coordinates
            nose_y = person[0][0] * y_h
            nose_x = person[0][1] * x_h
            
            base_p = get_posture(frame, person)
            
            # Update history
            movement_history[i+1].append((nose_x, nose_y, base_p))
            if len(movement_history[i+1]) > 100: movement_history[i+1].pop(0)
            
            final_class = analyze_temporal_states(i+1, base_p)
            
            # Vis
            color = (0, 0, 255) if final_class == "FALL (EMERGENCY)" else (0, 255, 0)
            draw_skeleton(frame, person, EDGES)
            draw_keypoints(frame, person)
            
            cv2.rectangle(frame, (0, 0), (0,0), color, 2) # Just setup color
            cv2.putText(frame, f"{final_class}", (int(nose_x), int(nose_y)-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Global Label
        color = (0, 0, 255) if final_class == "FALL (EMERGENCY)" else (0, 255, 0)
        cv2.putText(frame, f"TFLITE | Frame: {frame_count} | {final_class}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("TFLite Model Test", frame)
        print(f"Frame {frame_count}: {final_class}")
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    print("TFLite Test Complete.")

if __name__ == "__main__":
    run_tflite_inference()
