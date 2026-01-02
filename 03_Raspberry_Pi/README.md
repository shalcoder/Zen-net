# Raspberry Pi Deployment

This folder contains the optimized code for running Fall Detection on a Raspberry Pi 4.

## Files
- `main.py`: The lightweight script for real-time inference using `tflite-runtime`.
- `requirements.txt`: Dependencies for RPi.

## Setup on Raspberry Pi (Python 3.11 / Bookworm)

**1. Prepare the Environment (Crucial for OS Bookworm)**
Raspberry Pi OS now enforces managed environments. You **must** use a virtual environment.

```bash
# Update System
sudo apt-get update
sudo apt-get install python3-opencv python3-venv libatlas-base-dev -y

# Create Virtual Environment (Run this inside the 03_Raspberry_Pi folder)
python3 -m venv fall_env

# Activate it
source fall_env/bin/activate
```

**2. Install Dependencies**
*With the environment activated (you see `(fall_env)` in terminal):*
```bash
# Upgrade pip to handle new wheels
pip install --upgrade pip

# Install Libraries
pip install -r requirements.txt
```
*Note: If `tflite-runtime` fails to find a version, run:*
`pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite-runtime`

**3. Connect Camera**
Ensure your USB Webcam is plugged in.

## Usage

**Run Monitor:**
```bash
python3 main.py
```
- A window will appear showing the camera feed and detection status.
- Press `q` to exit.
