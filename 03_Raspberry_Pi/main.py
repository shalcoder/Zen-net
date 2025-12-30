import cv2
import numpy as np
import math
import time

# --- CONFIGURATION ---
MODEL_PATH = "movenet_multipose_lighting_quant.tflite"
THRESHOLD = 0.3
TARGET_HEIGHT = 160
TARGET_WIDTH = 320

# TFLite Runtime import (RPi specific)
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        print("Error: Could not import tflite_runtime or tensorflow.lite")
        exit(1)

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

def main():
    print("Initializing Raspberry Pi Fall Detection...")
    
    # 1. Load TFLite Model
    try:
        interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        input_index = interpreter.get_input_details()[0]['index']
        # Resize input just in case
        interpreter.resize_tensor_input(input_index, [1, TARGET_HEIGHT, TARGET_WIDTH, 3])
        interpreter.allocate_tensors()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_index = input_details[0]['index']
    output_index = output_details[0]['index']

    # 2. Open Webcam (Using V4L2 backend suitable for Pi Camera)
    cap = cv2.VideoCapture(0)
    
    # Set lower resolution for better FPS on Pi
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Monitoring started. Press 'q' to exit.")
    
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret: break

        # 3. Preprocess Input (Without using heavy TF library)
        img = frame.copy()
        # Resize to 320x160 (Note: TF resize_with_pad maintains aspect ratio, here we do simple resize for speed on Pi)
        # For significantly better accuracy, you should implement letterbox resizing with numpy
        img_resized = cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT))
        input_data = np.expand_dims(img_resized, axis=0)
        
        # Cast to expected type
        if input_details[0]['dtype'] == np.float32:
            input_data = input_data.astype(np.float32)
        elif input_details[0]['dtype'] == np.int32:
            input_data = input_data.astype(np.int32)
        elif input_details[0]['dtype'] == np.uint8:
            input_data = input_data.astype(np.uint8)

        # 4. Inference
        interpreter.set_tensor(input_index, input_data)
        interpreter.invoke()
        keypoints_with_scores = interpreter.get_tensor(output_index)
        
        # 5. Process
        keypoints_with_scores = keypoints_with_scores[:, :, :51].reshape(6, 17, 3)
        
        # 6. Visualization
        for i, person in enumerate(keypoints_with_scores):
            if np.mean(person[:, 2]) < THRESHOLD: continue
            
            y_h, x_h, _ = frame.shape
            nose_y = person[0][0] * y_h
            nose_x = person[0][1] * x_h
            
            base_p = get_posture(frame, person)
            
            movement_history[i+1].append((nose_x, nose_y, base_p))
            if len(movement_history[i+1]) > 100: movement_history[i+1].pop(0)
            
            final_class = analyze_temporal_states(i+1, base_p)
            
            color = (0, 0, 255) if final_class == "FALL (EMERGENCY)" else (0, 255, 0)
            draw_skeleton(frame, person, EDGES)
            draw_keypoints(frame, person)
            
            cv2.rectangle(frame, (0, 0), (0,0), color, 2)
            cv2.putText(frame, final_class, (int(nose_x), int(nose_y)-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            if final_class == "FALL (EMERGENCY)":
                cv2.putText(frame, "!!! FALL DETECTED !!!", (10, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Calculate FPS
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, height - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Pi Fall Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
