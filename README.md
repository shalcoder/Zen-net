# Zen-net — Guardian AI: Pure AIoT Multi-Model Fall Detection System

> **Pure AIoT project** combining computer-vision pose estimation with optional wearable IMU inference for fall detection and fatigue monitoring.  
> **Multi-model / multi-modal**: vision pose model (MoveNet) + optional ESP32/MPU6050 wearable impact classifier (Edge Impulse) running across multiple deployment targets (TF Hub, TFLite, INT8).  
> **Logging only in cloud**: all telemetry and alert events are posted by edge devices to a central FastAPI receiver and stored in a cloud-accessible SQLite store; the Streamlit dashboard reads exclusively from that centralized store.

---

## Overview

**Guardian AI** is a practical human fall-detection and fatigue-monitoring system built for Hackathon 2026. It classifies posture states in real time (Standing, Sitting, Falling/Lying) using MoveNet keypoint estimation from a camera, optionally fusing data from an ESP32 wearable to verify impacts and eliminate false alarms.

Key properties:
- **AIoT** — edge devices (Raspberry Pi, ESP32) perform on-device AI inference and send telemetry to a cloud receiver.
- **Multi-model / multi-modal** — vision pose model (MoveNet MultiPose) + wearable IMU classifier (Edge Impulse on ESP32/MPU6050) + multiple quantization targets (TF Hub full-precision, TFLite dynamic-range, TFLite INT8).
- **Logging only in cloud** — edge nodes never store events locally; all events flow to the central FastAPI receiver which persists them in SQLite. The dashboard reads from that single source of truth.

---

## System Architecture

```
┌──────────────────────────────┐      POST /upload_telemetry
│  Edge Node (Camera + AI)     │ ──────────────────────────────►
│  Raspberry Pi / Laptop       │                                 ┌────────────────────────┐
│  MoveNet (TFLite INT8)       │                                 │  Cloud Receiver (API)  │
└──────────────────────────────┘                                 │  FastAPI + SQLite      │
                                                                 │  receiver_server.py    │
┌──────────────────────────────┐      POST /upload_telemetry     └──────────┬─────────────┘
│  Wearable (ESP32 + MPU6050)  │ ──────────────────────────────►            │
│  Edge Impulse classifier     │                                             │ reads
└──────────────────────────────┘                                             ▼
                                                                 ┌────────────────────────┐
                                                                 │  Dashboard (UI)        │
                                                                 │  Streamlit             │
                                                                 │  dashboard_app.py      │
                                                                 └────────────────────────┘
```

### Processing pipeline

1. **Vision (Camera)** — MoveNet detects body keypoints; heuristics classify posture using bounding-box aspect ratio, joint angles, and temporal motion vectors.
2. **Wearable verification (optional)** — ESP32 + MPU6050 runs an Edge Impulse impact classifier; the backend escalates alerts when `accel_magnitude` > 2.2 g.
3. **Cloud receiver** — FastAPI endpoint ingests JSON telemetry from all devices and persists records in SQLite (`guardian_system.db`).
4. **Dashboard** — Streamlit reads from the centralized SQLite store and renders live status, trend charts, and alerts. **No local device logging.**

---

## Tech Stack

| Layer | Technology |
|---|---|
| **AI / ML** | TensorFlow, TF Hub, TensorFlow Lite (INT8 quantization), MoveNet MultiPose Lightning |
| **Computer Vision** | OpenCV |
| **Backend / Receiver** | FastAPI |
| **Dashboard** | Streamlit |
| **Storage (cloud)** | SQLite (`guardian_system.db`) — central telemetry store |
| **IoT / Wearable** | ESP32, MPU6050, Edge Impulse inference sketch |
| **Deployment targets** | Laptop / PC (TF Hub full-precision), Raspberry Pi 4 (TFLite INT8) |

---

## Repo Map

```
Zen-net/
├── 01_Normal_TF_Hub/          # Full TF Hub pipeline — best for PC/laptop
│   ├── main_live.py           # Real-time webcam fall detection
│   ├── video_scanner.py       # Offline video file scanner
│   └── requirements.txt
│
├── 02_TFLite_Laptop/          # TFLite conversion & testing tools
│   ├── convert_to_tflite.py   # Converts TF Hub model → quantized TFLite
│   ├── test_model.py          # Run .tflite model on a video file
│   ├── video_scanner_int8.py  # INT8-optimised scanner with extra filtering
│   └── requirements.txt
│
├── 03_Raspberry_Pi/           # Edge-deployment runtime (TFLite + tflite-runtime)
│   ├── main.py                # Simple local visualisation
│   ├── main_guardian.py       # Guardian system: dual-verify + posts telemetry to API
│   └── requirements.txt
│
├── 06_Dashboard_App/          # Cloud receiver + monitoring dashboard
│   ├── receiver_server.py     # FastAPI telemetry ingestion + alert logic
│   ├── dashboard_app.py       # Streamlit real-time UI
│   └── requirements.txt
│
├── 1d_esp32_model/            # ESP32 + MPU6050 Edge Impulse inference sketch
│
├── dataset/                   # Place test videos here (e.g. queda.mp4)
├── uploads/                   # Additional video assets
├── download_int8_model.py     # Downloads quantized MoveNet TFLite from TF Hub
└── requirements.txt           # Root-level dependencies
```

---

## Quick Start

### Option A — Laptop / PC (TensorFlow Hub, full-precision)

Best for development and quick demos.

```bash
cd 01_Normal_TF_Hub
pip install -r requirements.txt
python main_live.py          # real-time webcam
# or
python video_scanner.py      # scan a video file
```

> **Note:** Python 3.9 is recommended for this module (TensorFlow + NumPy 1.x compatibility).

---

### Option B — Laptop TFLite tests (pre-deployment validation)

```bash
cd 02_TFLite_Laptop
pip install -r requirements.txt
python convert_to_tflite.py  # produces movenet_multipose_lightning_quant.tflite
python test_model.py         # test on ../dataset/queda.mp4
python video_scanner_int8.py # INT8 variant with additional filtering
```

Optional — download a pre-quantized model from TF Hub:

```bash
python ../download_int8_model.py
```

---

### Option C — Raspberry Pi 4 (edge deployment)

Full instructions: [`03_Raspberry_Pi/README.md`](03_Raspberry_Pi/README.md)

```bash
cd 03_Raspberry_Pi
sudo apt-get install python3-opencv python3-venv libatlas-base-dev -y
python3 -m venv fall_env
source fall_env/bin/activate
pip install -r requirements.txt
python main_guardian.py      # posts telemetry to the cloud receiver
```

---

### Option D — Cloud Receiver + Dashboard

All telemetry from edge devices is sent here. **Logging only in cloud.**

```bash
cd 06_Dashboard_App
pip install -r requirements.txt
```

Terminal 1 — start the receiver API (listens on port 8000):

```bash
python receiver_server.py
```

Terminal 2 — start the dashboard UI (opens on port 8501):

```bash
streamlit run dashboard_app.py
```

- Receiver API: `http://localhost:8000`
- Dashboard UI: `http://localhost:8501`

#### Telemetry payload format

Edge devices send a `POST` request to `http://<RECEIVER_IP>:8000/upload_telemetry`:

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

Multi-modal verification rule: if `accel_magnitude` > 2.2 g, the backend escalates the alert to **"CRITICAL: VERIFIED FALL"**.

---

## Notes

- Several scripts contain **hard-coded paths** (e.g. `E:/human-fall-detection/...`). Update `VIDEO_PATH` / `MODEL_PATH` to match your environment before running.
- Python version guidance: **Python 3.9** for `01_Normal_TF_Hub` and `02_TFLite_Laptop` (TFLite Converter + NumPy 1.x). Separate virtual environments per module are recommended.

---

## Credits

Built as **Guardian AI** (Hackathon 2026). Organized as a multi-target AIoT repo: PC → TFLite → Raspberry Pi → Cloud Dashboard.
