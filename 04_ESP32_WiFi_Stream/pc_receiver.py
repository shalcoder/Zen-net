
# Script to connect to ESP32-CAM and run Fall Detection
import cv2
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

# ====== CONFIGURATION ======
# Replace with the IP printed in your Arduino Serial Monitor
ESP32_URL = "http://192.168.1.100:81/stream" 

# Load Model
print("Loading Model...")
model = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")
movenet = model.signatures["serving_default"]

# Helper Functions (Simplified from fall.py)
def loop_through_people(frame, keypoints, threshold=0.1):
    for person in keypoints:
        if np.mean(person[:, 2]) < threshold: continue
        y, x, c = frame.shape
        shaped = np.squeeze(np.multiply(person, [y, x, 1]))
        
        # Draw Skeleton
        for kp in shaped:
            if kp[2] > threshold: cv2.circle(frame, (int(kp[1]), int(kp[0])), 4, (0,255,0), -1)
            
        # Draw Label (Simplified Logic)
        cv2.putText(frame, "Person Detected", (int(shaped[0][1]), int(shaped[0][0])), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

# Main Loop
cap = cv2.VideoCapture(ESP32_URL)

if not cap.isOpened():
    print(f"Error: Could not connect to ESP32 at {ESP32_URL}")
    print("Check if ESP32 is powered on and IP is correct.")
    exit()

print(f"Connected to {ESP32_URL}")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Stream lost...")
        break
        
    # Resize for Model
    img = frame.copy()
    img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 160, 320)
    img = tf.cast(img, dtype=tf.int32)
    
    # Inference
    result = movenet(img)
    keypoints = result["output_0"].numpy()[:, :, :51].reshape(6, 17, 3)
    
    # Visuals
    loop_through_people(frame, keypoints)
    
    cv2.imshow("ESP32 Fall Monitor", frame)
    if cv2.waitKey(1) == 27: break

cap.release()
cv2.destroyAllWindows()
