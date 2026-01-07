from fastapi import FastAPI, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from datetime import datetime, timezone
import pandas as pd
import requests

from database.db_manager import SessionLocal, UserTelemetry, init_db
from analytics.fatigue_logic import FatigueComputer

# Blynk Configuration
BLYNK_TEMPLATE_ID = "TMPL3Q_roROB9"
BLYNK_TEMPLATE_NAME = "Elderly Care EdgeAI"
BLYNK_AUTH_TOKEN = "eGCQr0mI4f416sHgYr7b55TzdthN-Ru9"
BLYNK_DEVICE_NAME = "ESPCAM_node"
BLYNK_EVENT_CODE = "fall_alert"

def send_blynk_alert(message: str):
    """Send fall alert via Blynk cloud"""
    try:
        # Blynk Event API
        url = f"https://blynk.cloud/external/api/logEvent?token={BLYNK_AUTH_TOKEN}&code={BLYNK_EVENT_CODE}"
        response = requests.get(url, timeout=3)
        if response.status_code == 200:
            print(f"✅ Blynk alert sent: {message}")
        else:
            print(f"⚠️ Blynk alert failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Blynk error: {e}")

app = FastAPI()
fatigue = FatigueComputer()

# Initialize DB tables
init_db()

class MPUTelemetry(BaseModel):
    posture_class: str  # STANDING, SITTING, FALLING
    risk_score: float   # 0-100

class CamTelemetry(BaseModel):
    vision_status: str  # Normal or Fall
    risk_score: float   # 0-100

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def now_utc():
    return datetime.now(timezone.utc)

@app.post("/upload_telemetry_mpu")
def upload_mpu(data: MPUTelemetry, db: Session = Depends(get_db)):
    impact = 2.8 if data.posture_class == "FALLING" else 1.0
    
    # Calculate fatigue using sophisticated algorithm
    history = db.query(UserTelemetry).filter(
        UserTelemetry.device_id == "ESP32_WEARABLE"
    ).order_by(UserTelemetry.timestamp.desc()).limit(50).all()
    
    fatigue_idx = 0.0
    if history:
        df_history = pd.DataFrame([{
            "timestamp": h.timestamp,
            "posture_class": h.posture_class,
            "slump_metric": h.slump_metric,
            "accel_magnitude": h.accel_magnitude
        } for h in history])
        fatigue_idx = fatigue.calculate_fatigue_score(df_history)

    db.add(UserTelemetry(
        timestamp=now_utc(),
        device_id="ESP32_WEARABLE",
        posture_class=data.posture_class,
        accel_magnitude=impact,
        slump_metric=0.0,
        fatigue_index=fatigue_idx,
        vision_status="Normal",
        alert_status="IMU_DATA",
        risk_score=data.risk_score
    ))
    db.commit()
    return {"status": "ok", "fusion": "IMU_DATA", "fatigue": fatigue_idx}

@app.post("/upload_telemetry_cam")
def upload_cam(data: CamTelemetry, db: Session = Depends(get_db)):
    posture = "FALLING" if data.vision_status == "Fall" else "STANDING"
    
    # Send Blynk alert if fall is detected
    if data.vision_status == "Fall":
        send_blynk_alert(f"⚠️ FALL DETECTED by {BLYNK_DEVICE_NAME}! Risk: {data.risk_score:.1f}%")
    
    # Calculate fatigue using sophisticated algorithm
    history = db.query(UserTelemetry).filter(
        UserTelemetry.device_id == "RPI_CAM_01"
    ).order_by(UserTelemetry.timestamp.desc()).limit(50).all()
    
    fatigue_idx = 0.0
    if history:
        df_history = pd.DataFrame([{
            "timestamp": h.timestamp,
            "posture_class": h.posture_class,
            "slump_metric": h.slump_metric,
            "accel_magnitude": h.accel_magnitude
        } for h in history])
        fatigue_idx = fatigue.calculate_fatigue_score(df_history)

    db.add(UserTelemetry(
        timestamp=now_utc(),
        device_id="RPI_CAM_01",
        posture_class=posture,
        accel_magnitude=1.0,
        slump_metric=0.0,
        fatigue_index=fatigue_idx,
        vision_status=data.vision_status,
        alert_status="CAM_DATA",
        risk_score=data.risk_score
    ))
    db.commit()
    return {"status": "ok", "fusion": "CAM_DATA", "fatigue": fatigue_idx}

@app.post("/upload_telemetry_fusion")
def fusion_engine(device_id: str, posture: str, accel: float, vision: str, risk: float, db: Session = Depends(get_db)):
    verified_fall = (posture == "FALLING" or vision == "Fall")
    imu_spike = accel > 2.2

    if verified_fall and imu_spike:
        alert = "CRITICAL: VERIFIED FALL"
        fatigue_idx = 0.0
    else:
        alert = fatigue.get_status_label(accel)
        fatigue_idx = fatigue.calculate_fatigue_score(pd.DataFrame([{
            "timestamp": now_utc(),
            "posture_class": posture,
            "accel_magnitude": accel,
            "slump_metric": 0.0
        }]))

    db.add(UserTelemetry(
        timestamp=now_utc(),
        device_id=device_id,
        posture_class=posture,
        accel_magnitude=accel,
        slump_metric=0.0,
        fatigue_index=fatigue_idx,
        vision_status=vision,
        alert_status=alert,
        risk_score=risk
    ))
    db.commit()
    return {"status": "ok", "fusion_alert": alert}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)