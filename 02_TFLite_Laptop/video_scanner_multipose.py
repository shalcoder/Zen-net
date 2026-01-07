# Human Fall Detection - Video Scanner (TFLite MultiPose Dynamic)
import cv2
import tensorflow as tf
import numpy as np
import math
import os

# --- INITIALIZATION ---
MODEL_PATH = "02_TFLite_Laptop/model_quant_dynamic.tflite"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "model_quant_dynamic.tflite"

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)

# Resize input tensor to expected shape [1, 192, 320, 3]
interpreter.resize_tensor_input(0, [1, 192, 320, 3])
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_index = input_details[0]['index']
output_index = output_details[0]['index']

# Video Path
VIDEO_PATH = "uploads/20240912_102048.mp4"
if not os.path.exists(VIDEO_PATH):
    VIDEO_PATH = "E:/human-fall-detection/uploads/20240912_102048.mp4"

# Configuration
THRESHOLD = 0.25
EDGES = {
    (0, 1): "m", (0, 2): "c", (1, 3): "m", (2, 4): "c",
    (0, 5): "m", (0, 6): "c", (5, 7): "m", (7, 9): "m",
    (6, 8): "c", (8, 10): "c", (5, 6): "y", (5, 11): "m",
    (6, 12): "c", (11, 12): "y", (11, 13): "m", (13, 15): "m",
    (12, 14): "c", (14, 16): "c",
}

# --- FUNCTIONS ---

def draw_skeleton(frame, keypoints, edges, threshold=THRESHOLD):
    y, x, c = frame.shape
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = keypoints[p1]
        y2, x2, c2 = keypoints[p2]
        if (c1 > threshold) & (c2 > threshold):
            cv2.line(frame, (int(x1*x), int(y1*y)), (int(x2*x), int(y2*y)), (0, 0, 255), 2)

def calculate_aspect_ratio(keypoints):
    x_coords = keypoints[:, 1]
    y_coords = keypoints[:, 0]
    conf = keypoints[:, 2]
    
    valid_x = x_coords[conf > THRESHOLD]
    valid_y = y_coords[conf > THRESHOLD]
    
    if len(valid_x) < 5: return 1.0
    
    w = np.max(valid_x) - np.min(valid_x)
    h = np.max(valid_y) - np.min(valid_y)
    return w / h if h > 0 else 1.0

def calculate_angle(keypoints):
    x_coords = keypoints[:, 1]
    y_coords = keypoints[:, 0]
    conf = keypoints[:, 2]
    
    valid_x = x_coords[conf > THRESHOLD]
    valid_y = y_coords[conf > THRESHOLD]
    
    if len(valid_x) < 5: return 90.0
    
    x1, y1 = np.min(valid_x), np.min(valid_y)
    x2, y2 = np.max(valid_x), np.max(valid_y)
    cx, cy = (x1+x2)/2, (y1+y2)/2
    
    return math.atan2(cy - y1, cx - x1) * 180 / math.pi

def get_posture(keypoints):
    ar = calculate_aspect_ratio(keypoints)
    ang = calculate_angle(keypoints)
    
    if ar > 1.2 and ang < 60: return "LYING/FALL", ar, ang
    elif 0.6 < ar <= 1.2: return "SITTING", ar, ang
    else: return "STANDING", ar, ang

def process():
    cap = cv2.VideoCapture(VIDEO_PATH)
    print(f"Starting MultiPose Analysis: {VIDEO_PATH}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Resize for MultiPose (expects multiples of 32, e.g. 256x256 or 160x320)
        # Multipose Lightning Hub default is 160x320. 
        # But for RPi we can use 256x256 or similar.
        input_img = cv2.resize(frame, (320, 192)) # Multipose standard
        input_img = np.expand_dims(input_img, axis=0).astype(np.int32)
        
        interpreter.set_tensor(input_index, input_img)
        interpreter.invoke()
        output = interpreter.get_tensor(output_index) # [1, 6, 56]
        
        # Parse output
        # Output 56 values: [ymin, xmin, ymax, xmax, score, 17 * (y, x, s)]
        people = output[0] # [6, 56]
        
        for i in range(1): # Track the main person (highest score person 0)
            person_data = people[i]
            overall_score = person_data[4]
            if overall_score < THRESHOLD: continue
            
            keypoints = person_data[5:].reshape(17, 3)
            
            posture_class, ar, ang = get_posture(keypoints)
            
            # Classification
            final_status = "FALL (WARNING)" if "LYING" in posture_class else "NORMAL"
            color = (0, 0, 255) if "FALL" in final_status else (0, 255, 0)
            
            # Draw
            draw_skeleton(frame, keypoints, EDGES)
            cv2.putText(frame, f"{final_status} | {posture_class}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, f"AR: {ar:.2f} | Ang: {ang:.1f} | Score: {overall_score:.2f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Logs
            print(f"Status: {final_status} | AR: {ar:.2f} | Ang: {ang:.1f}")

        cv2.imshow("MultiPose Monitor", frame)
        if cv2.waitKey(50) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process()
