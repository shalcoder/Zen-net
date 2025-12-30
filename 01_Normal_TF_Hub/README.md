# Normal TF Hub Fall Detection

This folder contains the original TensorFlow Hub implementation using MoveNet Multipose Lightning.

## Files
- `main_live.py`: Runs real-time fall detection using your Webcam.
- `video_scanner.py`: Scans a video file (`../dataset/queda.mp4`) frame-by-frame.

## Setup
1. **Prerequisite:** Ensure you are using **Python 3.9**.
   - Issues were found with Python 3.10+ and Numpy 2.0.
   
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

**Live Webcam Detection:**
```bash
python main_live.py
```
- Press `Esc` to exit and view analysis graphs.

**Video File Scanning:**
```bash
python video_scanner.py
```
- Ensure `queda.mp4` lives in `../dataset/` (Check/Update path in `video_scanner.py` line 14).
