from fastapi import FastAPI, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from datetime import datetime, timezone
import pandas as pd
import requests
import os

from database.db_manager import SessionLocal, UserTelemetry, init_db
from analytics.fatigue_logic import FatigueComputer

# --- BLYNK CONFIGURATION ---
BLYNK_TEMPLATE_ID = os.getenv("BLYNK_TEMPLATE_ID", "TMPL3Q_roROB9")
BLYNK_TEMPLATE_NAME = os.getenv("BLYNK_TEMPLATE_NAME", "Elderly Care EdgeAI")
BLYNK_AUTH_TOKEN = os.getenv("BLYNK_AUTH_TOKEN", "eGCQr0mI4f416sHgYr7b55TzdthN-Ru9")
BLYNK_DEVICE_NAME = os.getenv("BLYNK_DEVICE_NAME", "ESPCAM_node")
BLYNK_EVENT_CODE = os.getenv("BLYNK_EVENT_CODE", "fall_alert")

# --- EMAIL CONFIGURATION (Gmail SMTP) ---
import smtplib
from email.mime.text import MIMEText

EMAIL_SENDER = os.environ.get("EMAIL_SENDER", "vishalm26012006@gmail.com")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD", "your_app_password")
EMAIL_RECEIVER = os.environ.get("EMAIL_RECEIVER", "vishal.2006dev@gmail.com")

def send_email_alert(risk_score: float):
    """Send emergency email alert using Gmail SMTP"""
    print(f"Attempting to send email from {EMAIL_SENDER} to {EMAIL_RECEIVER}...")
    
    if not EMAIL_PASSWORD or "your_app" in EMAIL_PASSWORD:
        print("ERROR: Email Password invalid or missing. Check Environment Variables.")
        return

    subject = f"🚨 EMERGENCY: Fall Detected! Risk {risk_score:.1f}%"
    body = f"""
    Guardian AI Alert System
    ------------------------
    EVENT: Verified Fall Detected
    RISK LEVEL: {risk_score:.1f}%
    STATUS: Confirmed by Vision + Wearable Sensors
    
    Please check the dashboard immediately.
    """
    
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER

    try:
        # Use Port 587 with STARTTLS (More reliable than 465)
        with smtplib.SMTP('smtp.gmail.com', 587, timeout=10) as server:
            server.starttls() # Upgrade connection to secure
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        print(f"SUCCESS: Email Alert Sent to {EMAIL_RECEIVER}!")
    except Exception as e:
        print(f"FAILURE: Email sending error: {e}")

def send_blynk_alert(message: str):
    """Send fall alert via Blynk cloud"""
    if BLYNK_AUTH_TOKEN:
        try:
            url = f"https://blynk.cloud/external/api/logEvent?token={BLYNK_AUTH_TOKEN}&code={BLYNK_EVENT_CODE}"
            response = requests.get(url, timeout=3) # Re-added timeout for robustness
            if response.status_code == 200:
                print(f"Blynk Alert Sent! Response: {response.status_code}")
            else:
                print(f"Failed to send Blynk alert: {response.status_code} {response.text}")
        except Exception as e:
            print(f"Blynk connection error: {e}")
    else:
        print("Blynk token not configured, skipping alert.")

app = FastAPI()
fatigue = FatigueComputer()

# Initialize DB tables
init_db()

@app.get("/")
def root():
    """API root endpoint"""
    return {
        "status": "online",
        "service": "Guardian AI Fall Detection Backend",
        "version": "1.0.0",
        "endpoints": {
            "mpu": "/upload_telemetry_mpu",
            "camera": "/upload_telemetry_cam",
            "docs": "/docs"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "guardian-ai-backend"}

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

from fastapi import BackgroundTasks

@app.post("/upload_telemetry_mpu")
def upload_mpu(data: MPUTelemetry, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    impact = 2.8 if data.posture_class == "FALLING" else 1.0
    
    # Check for direct MPU Fall (Posture=FALLING)
    is_fall = (data.posture_class == "FALLING")
    
    if is_fall:
        print(f"Wearable Fall Detected! Impact: {impact:.2f}g")
        # Send in BACKGROUND so we don't block the ESP32 response
        background_tasks.add_task(send_email_alert, risk_score=data.risk_score if data.risk_score > 0 else 90.0)

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
        alert_status="MPU_FALL" if is_fall else "IMU_DATA",
        risk_score=data.risk_score
    ))
    db.commit()
    return {"status": "ok", "fusion": "IMU_DATA", "fatigue": fatigue_idx}

@app.post("/upload_telemetry_cam")
def upload_cam(data: CamTelemetry, db: Session = Depends(get_db)):
    posture = "FALLING" if data.vision_status == "Fall" else "STANDING"
    
    # 1. Check for recent high impact from wearable (within last 5 seconds)
    recent_impact = db.query(UserTelemetry).filter(
        UserTelemetry.device_id.like("%ESP32%"),
        UserTelemetry.accel_magnitude > 2.2,
        UserTelemetry.timestamp >= (now_utc() - timedelta(seconds=5))
    ).first()

    verified_fall = False
    if data.posture_class == "FALLING":
        # Dual Verification Logic
        if recent_impact:
            verified_fall = True
            print(f"CONFIRMED FALL: Vision + Wearable Impact ({recent_impact.accel_magnitude:.2f}g)")
            
            # Send BOTH alerts
            send_blynk_alert(f"EMERGENCY: Confirmed Fall! Risk: {data.risk_score:.1f}%")
            send_email_alert(data.risk_score)
        else:
             print(f"Vision Fall detected, but no wearable impact found. Alert suppressed.")

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
        alert_status="Blynk OK" if verified_fall else "None",
        risk_score=data.risk_score
    ))
    db.commit()
    return {"status": "ok", "fusion": "CAM_DATA", "fatigue": fatigue_idx}

@app.get("/get_telemetry")
def get_telemetry(limit: int = 100, db: Session = Depends(get_db)):
    """Fetch the latest telemetry records for the dashboard"""
    data = db.query(UserTelemetry).order_by(UserTelemetry.timestamp.desc()).limit(limit).all()
    # Convert SQLAlchemy objects to serializable dictionaries
    result = []
    for d in data:
        item = {column.name: getattr(d, column.name) for column in d.__table__.columns}
        # Convert datetime to string for JSON serialization
        if item['timestamp']:
            item['timestamp'] = item['timestamp'].isoformat()
        result.append(item)
    return result

@app.post("/upload_telemetry_fusion")
def fusion_engine(device_id: str, posture: str, accel: float, vision: str, risk: float, db: Session = Depends(get_db)):
    verified_fall = (posture == "FALLING" or vision == "Fall")
    imu_spike = accel > 2.2

    if verified_fall and imu_spike:
        alert = "CRITICAL: VERIFIED FALL"
        print(f"Received MPU Data: {data}")
        
        # Calculate Fatigue (using new logic)
        fatigue_val = 0.0
        fatigue_idx = fatigue.calculate_fatigue_score(pd.DataFrame([{
            "timestamp": now_utc(),
            "posture_class": posture,
            "accel_magnitude": accel,
            "slump_metric": 0.0
        }]))
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