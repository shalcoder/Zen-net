# Guardian AI Dashboard

This is the central command center for the Fall Detection & Fatigue System.
It serves two purposes:
1.  **Receiver API**: Listens for data from Raspberry Pi / ESP32.
2.  **Dashboard**: Visualizes Fatigue, Posture, and Alerts in real-time.

## 🏗️ Architecture
*   **Database**: SQLite (`guardian_system.db`) - Lightweight, zero-config.
*   **Backend**: FastAPI - Handles high-frequency telemetry ingestion.
*   **Analytics**: Implements "Fatigue accumulation based on posture and movement analysis" (ASPA logic).
*   **Frontend**: Streamlit - Auto-refreshing visual interface.

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Receiver API (Background)
This listens on Port 8000 for data from your devices.
```bash
python receiver_server.py
```
*Leave this terminal window open.*

### 3. Start the Dashboard (Frontend)
Open a new terminal:
```bash
streamlit run dashboard_app.py
```
*This will open `http://localhost:8501` in your browser.*

---

## 🔗 Connectivity Guide (Hackathon Final)

### **Wireless Fusion Mode (ESP32 + Cam)**
The dashboard uses a **Multi-Modal Verification** engine. It cross-references AI Vision (Camera) with Physical Impact (MPU6050) to eliminate false alarms.

#### **Device Payload Format**
Send a `POST` request to `http://<YOUR_IP>:8000/upload_telemetry` with:

```json
{
  "device_id": "GUARDIAN_IOT_01",
  "posture_class": "FALLING", 
  "vision_status": "Fall",
  "accel_magnitude": 2.85, 
  "slump_metric": 1.0, 
  "risk_score": 95.0
}
```

### **1. AI Vision (Raspberry Pi/Laptop)**
*   Send `posture_class` and `vision_status`.
*   The backend will set the initial alert level.

### **2. Wearable (ESP32 + MPU6050)**
*   Send `accel_magnitude` (The total 3D Vector length).
*   **Verification Rule**: If `accel_magnitude` > 2.2, the system escalates the Vision alert to **"CRITICAL: VERIFIED FALL"**.

---
*Developed for Guardian AI Hackathon 2026*
