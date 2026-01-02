
import cv2
import numpy as np
import time
import requests
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

# --- CONFIGURATION ---
BLYNK_AUTH = "YOUR_BLYNK_AUTH_TOKEN"
BLYNK_URL = f"https://blynk.cloud/external/api/update?token={BLYNK_AUTH}"

# TFLite Settings
MODEL_PATH = "model.tflite" # Ensure this file is present
THRESHOLD = 0.3

# Wellness Tracking
MAX_SITTING_TIME_SEC = 3600 # 1 Hour
current_sitting_start = None

# IoT State
wearable_triggered = False
wearable_last_heartbeat = time.time()
camera_active_until = 0

# --- HTTP SERVER (To receive Trigger from ESP32 Wearable) ---
class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global wearable_triggered, wearable_last_heartbeat, camera_active_until
        
        if self.path == '/trigger_fall':
            print("[IOT] WEARABLE DETECTED IMPACT! Verifying with Vision...")
            wearable_triggered = True
            camera_active_until = time.time() + 15 # Activate Camera for 15s verification
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"VERIFYING")
            
        elif self.path == '/heartbeat':
            wearable_last_heartbeat = time.time()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")

def start_server():
    server = HTTPServer(('0.0.0.0', 5000), RequestHandler)
    server.serve_forever()

# --- BLYNK ALERTS ---
def send_alert(message, pin="v1"):
    try:
        url = f"{BLYNK_URL}&{pin}={message}"
        requests.get(url, timeout=2)
        print(f"[CLOUD] Alert Sent: {message}")
    except:
        print("[CLOUD] Failed to send alert (Check Internet)")

# --- MOVENET UTILS (Simplified for brevity) ---
# TFLite Runtime Import
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        print("Error: Install tflite-runtime")
        exit(1)

# Edges for skeleton (Same as before)
EDGES = { (0, 1): "m", (0, 2): "c", (1, 3): "m", (2, 4): "c", (0, 5): "m", (0, 6): "c", (5, 7): "m", (7, 9): "m", (6, 8): "c", (8, 10): "c", (5, 6): "y", (5, 11): "m", (6, 12): "c", (11, 12): "y", (11, 13): "m", (13, 15): "m", (12, 14): "c", (14, 16): "c" }

def main():
    global current_sitting_start
    
    # 1. Start IoT Listener
    t = threading.Thread(target=start_server)
    t.daemon = True
    t.start()
    print("✅ IoT Server Started on Port 5000")
    
    # 2. Load Model
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_index = input_details[0]['index']
    output_index = output_details[0]['index']
    interpreter.resize_tensor_input(input_index, [1, 160, 320, 3])
    interpreter.allocate_tensors()
    print("✅ MoveNet AI Model Loaded")

    # 3. Open Camera
    cap = cv2.VideoCapture(0) # USB Cam
    
    print("🚀 System Armed. Waiting for Wearable Trigger or Wellness Checks...")
    
    while True:
        # --- HEALTH CHECK (Self-Healing) ---
        if time.time() - wearable_last_heartbeat > 60:
            print("⚠️ WARNING: Wearable Sensor Offline!")
            send_alert("SENSOR_OFFLINE", "v10")
            wearable_last_heartbeat = time.time() # Reset to avoid spam

        # --- LOGIC SWITCHER ---
        # Only run heavy AI if triggered OR periodically (every 5 seconds) for Wellness check
        # This saves Huge Energy (Green IoT)
        
        should_run_ai = False
        if time.time() < camera_active_until:
            should_run_ai = True # Emergency Mode
        elif int(time.time()) % 5 == 0:
            should_run_ai = True # Periodic Wellness Check
            
        if not should_run_ai:
             time.sleep(0.1)
             continue

        ret, frame = cap.read()
        if not ret: break

        # AI Inference
        img = cv2.resize(frame, (320, 160))
        input_data = np.expand_dims(img, axis=0)
        
        # Type handling
        if input_details[0]['dtype'] == np.uint8:
             input_data = input_data.astype(np.uint8)
        else:
             input_data = input_data.astype(np.float32)

        interpreter.set_tensor(input_index, input_data)
        interpreter.invoke()
        keypoints = interpreter.get_tensor(output_index)
        keypoints = keypoints[:, :, :51].reshape(6, 17, 3)[0] # First person
        
        # Posture Analysis (Simplified)
        posture = "STANDING"
        conf = np.mean(keypoints[:, 2])
        
        if conf > 0.3:
            y_coords = keypoints[:, 0]
            aspect_ratio = (np.max(keypoints[:, 1]) - np.min(keypoints[:, 1])) / (np.max(y_coords) - np.min(y_coords))
            
            if aspect_ratio > 1.5: 
                posture = "Recumbent (Lying)" 
            elif aspect_ratio > 0.8:
                posture = "SITTING"
            
            # --- EMERGENCY LOGIC ---
            if posture == "Recumbent (Lying)" or posture == "FALLING":
                 # Dual Verification: Did wearable trigger AND camera see it?
                 if time.time() < camera_active_until:
                      print("🚨 CONFIRMED FALL! Sending Emergency Alert!")
                      send_alert("EMERGENCY_FALL_CONFIRMED")
                      camera_active_until = 0 # Reset
                 else:
                      print("Visual Lying detected (No impact trigger) -> Assuming Rest.")
            
            # --- WELLNESS LOGIC (Fatigue & Slump) ---
            if posture == "SITTING":
                # Slump Detection: Check distance between Nose (0) and avg Shoulders (5,6)
                nose_y = keypoints[0][0]
                shoulder_y = (keypoints[5][0] + keypoints[6][0]) / 2
                
                # In normalized coords (0-1), a small diff means head is down
                neck_length = abs(shoulder_y - nose_y)
                
                # Threshold dependent on distance, but usually < 0.05 means head is dropped
                if neck_length < 0.05: 
                     print("💤 FATIGUE: Slumping detected (Head Drop)")
                     send_alert("FATIGUE_SLUMP_DETECTED", "v6")
                     posture = "FATIGUE (SLUMP)"

                if current_sitting_start is None: current_sitting_start = time.time()
                elif time.time() - current_sitting_start > MAX_SITTING_TIME_SEC: 
                    print("💤 FATIGUE ALERT: User sitting too long.")
                    send_alert("FATIGUE_WARNING_MOVE_AROUND", "v5")
                    current_sitting_start = None # Reset
            else:
                current_sitting_start = None

        # Display (Optional - for Demo)
        cv2.putText(frame, f"STATUS: {posture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Guardian Eye", frame)
        if cv2.waitKey(1) == 27: break

    cap.release()

if __name__ == "__main__":
    main()
