
import requests
import time
import random

# YOUR RENDER URL
RENDER_URL = "https://guardian-ai-backend-7zfj.onrender.com"

def send_mock_data():
    print(f"🚀 Sending test data to {RENDER_URL}...")
    
    # 1. Send Camera Data (Normal)
    cam_payload = {
        "vision_status": "Normal",
        "posture_class": "STANDING",
        "risk_score": 12.5
    }
    
    # 2. Send Wearable Data (Normal)
    mpu_payload = {
        "posture_class": "SITTING",
        "risk_score": 8.2
    }

    try:
        # Upload Camera
        r1 = requests.post(f"{RENDER_URL}/upload_telemetry_cam", json=cam_payload)
        print(f"📸 Camera Response: {r1.status_code} - {r1.json()}")
        
        # Upload Wearable
        r2 = requests.post(f"{RENDER_URL}/upload_telemetry_mpu", json=mpu_payload)
        print(f"⌚ Wearable Response: {r2.status_code} - {r2.json()}")
        
        print("\n✅ TEST COMPLETE! Check your dashboard now.")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    send_mock_data()
