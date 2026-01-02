# ESP32-CAM "Wireless Eye" System

This architecture turns your **ESP32-CAM (approx. $5)** into a wireless IP camera that streams video directly to your PC, Laptop, or Raspberry Pi. The heavy AI processing (MoveNet) happens on the receiving device, bypassing the ESP32's hardware limitations.

## 📂 Folder Contents
- **`CameraWebServer.ino`**: The main firmware file. Open this in Arduino IDE.
- **`app_httpd.cpp` / `app_httpd.h`**: The logic that handles the MJPEG video stream.
- **`pc_receiver.py`**: The "Brain". This python script receives the video feed and runs the Fall Detection AI.

---

## 🛠️ Step 1: Flashing the ESP32-CAM

1. **Install Arduino IDE**
   - Download from: [arduino.cc/en/software](https://www.arduino.cc/en/software)

2. **Install ESP32 Board Support**
   - Go to **File** -> **Preferences**.
   - Paste this into "Additional Boards Manager URLs": 
     `https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json`
   - Go to **Tools** -> **Board** -> **Boards Manager**.
   - Search for **"esp32"** (by Espressif Systems) and click **Install**.

3. **Configure the Code**
   - Open `CameraWebServer.ino` from this folder.
   - **Important**: Find lines 7-8 and enter your home WiFi details:
     ```cpp
     const char* ssid = "YOUR_WIFI_NAME";
     const char* password = "YOUR_WIFI_PASSWORD";
     ```

4. **Select Your Board**
   - Go to **Tools** -> **Board** -> **ESP32 Arduino** -> Select **AI Thinker ESP32-CAM**.
   - **Note**: If "AI Thinker" isn't listed, select "ESP32 Wrover Module".

5. **Upload**
   - Connect your ESP32-CAM to your PC via a USB-TTL (FTDI) adapter.
   - **Crucial**: Connect **IO0 to GND** (this puts it in flash mode).
   - Click the **Upload (Arrow)** button in Arduino IDE.
   - Press the tiny **RESET button** on the ESP32-CAM if it says "Connecting...".

6. **Getting the IP Address**
   - Once uploaded, **disconnect IO0 from GND**.
   - Open **Tools** -> **Serial Monitor**.
   - Set baud rate to **115200**.
   - Press the **RESET button** on the ESP32-CAM again.
   - You should see: **"Camera Ready! Use this URL: http://192.168.X.X:81/stream"**
   - **Copy this URL.**

---

## 🧠 Step 2: Running the AI Monitor (On PC/Laptop/Pi)

1. Open `pc_receiver.py` in your code editor.
2. Find line 9 (`ESP32_URL`) and paste the URL you copied above:
   ```python
   ESP32_URL = "http://192.168.1.50:81/stream" # <-- Your actual IP here
   ```
3. Install the AI dependencies (if not already installed):
   ```bash
   pip install tensorflow opencv-python numpy tensorflow_hub
   ```
4. Run the monitor:
   ```bash
   python pc_receiver.py
   ```

A window named **"ESP32 Fall Monitor"** should appear, showing the live feed from your tiny camera with detection overlays!
