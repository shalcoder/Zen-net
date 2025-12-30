# Raspberry Pi Deployment

This folder contains the optimized code for running Fall Detection on a Raspberry Pi 4.

## Files
- `main.py`: The lightweight script for real-time inference using `tflite-runtime`.
- `requirements.txt`: Dependencies for RPi.

## Setup on Raspberry Pi

1. **Transfer Files**: Copy this entire folder (and the TFLite model from `models/`) to your Pi.
2. **Install Dependencies**:
   ```bash
   sudo apt-get update
   sudo apt-get install python3-opencv
   pip3 install -r requirements.txt
   ```
   *Note: Python 3.9 is recommended. If on Bookworm, consider using a venv.*
   *If `tflite-runtime` fails, try:* `pip3 install tflite-runtime --extra-index-url https://google-coral.github.io/py-repo/`

3. **Connect Camera**: Ensure your USB Webcam is plugged in.

## Usage

**Run Monitor:**
```bash
python3 main.py
```
- A window will appear showing the camera feed and detection status.
- Press `q` to exit.
