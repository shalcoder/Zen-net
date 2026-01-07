# Raspberry Pi Fall Detection - Complete Setup Guide

This guide will take you from a fresh Raspberry Pi to a fully functional "Guardian AI" Fall Detection System.

## 📦 Prerequisites
*   **Hardware**:
    *   Raspberry Pi 4 (Recommended 4GB or 8GB RAM).
    *   USB Webcam (Any standard Logitech/Generic webcam).
    *   MicroSD Card (32GB+).
*   **Software**:
    *   Raspberry Pi OS (Bookworm 64-bit recommended).

---

## 🛠️ Step 1: Initial Setup (If brand new)
1.  **Flash OS**: Use [Raspberry Pi Imager](https://www.raspberrypi.com/software/) to flash **Raspberry Pi OS (64-bit)** to your SD Card.
2.  **WiFi**: Configure your WiFi name and password in the Imager settings (Shift+Ctrl+X) before flashing.
3.  **Boot**: Insert SD card, power on, and wait for desktop/terminal.
4.  **Terminal**: Open the Terminal icon (Black box).

---

## 📥 Step 2: Get the Code
Run these commands in your terminal to download the project:

```bash
# Install Git
sudo apt install git -y

# Clone the repository
git clone https://github.com/shalcoder/Zen-net.git

# Enter the project folder
cd Zen-net/03_Raspberry_Pi
```

---

## ⚙️ Step 3: Architecture Setup (Crucial!)
Raspberry Pi OS "Bookworm" requires a **Virtual Environment (venv)** to prevent system breakage.

**1. Update & Install System Tools:**
```bash
sudo apt-get update
sudo apt-get install python3-opencv python3-venv libatlas-base-dev -y
```

**2. Create the Safe Environment:**
```bash
python3 -m venv fall_env
```

**3. Activate the Environment:**
*You must do this every time you open a new terminal!*
```bash
source fall_env/bin/activate
```
*(You will see `(fall_env)` appear at the start of your command line).*

**4. Install Python Libraries:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
*If `tflite-runtime` fails, run this backup command:*
```bash
pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite-runtime
```

---

## 🔌 Step 4: Hardware Check
1.  Plug your USB Webcam into a USB port.
2.  Verify it is detected:
    ```bash
    ls /dev/video*
    ```
    *You should see `/dev/video0` listed.*

---

## 🚀 Step 5: Run the AI
Make sure your environment is valid (`source fall_env/bin/activate`)!

### Option A: Simple Monitor (Testing)
Just visualize the skeleton and fall detection locally.
```bash
python main.py
```

### Option B: The "Guardian" System (Hackathon Demo) 🏆 
Features: **Dual-Verification**, **Fatigue Detection** (Slump), and **IoT Alerts** (Blynk).
1.  **Edit the code** to add your Blynk Token:
    ```bash
    nano main_guardian.py
    ```
    *Change `BLYNK_AUTH = "YOUR_TOKEN"` inside the file. Press `Ctrl+X`, then `Y`, then `Enter` to save.*
2.  **Run it:**
    ```bash
    python main_guardian.py
    ```

---

## ❓ Troubleshooting

**"No module named cv2"**
*   Pass check: Did you verify `(fall_env)` is visible? Run `source fall_env/bin/activate`.

**"Camera not found"**
*   Unplug and replug the camera.
*   Try `python main.py` again.
*   Ensure no other app (like Zoom or Browser) is using the camera.

**System is slow (Low FPS)**
*   Ensure you are using a **Raspberry Pi 4**.
*   The code is optimized for 640x480 resolution. Increasing it will slow down inference.
