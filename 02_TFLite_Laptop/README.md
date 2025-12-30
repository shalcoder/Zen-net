# TFLite Conversion & Testing (Laptop)

This folder contains scripts to convert the MoveNet model to TFLite and test it on your PC before deploying to Raspberry Pi.

## Files
- `convert_to_tflite.py`: Converts the TF Hub model to a Quantized TFLite model.
- `test_model.py`: Tests the generated `.tflite` model on a video file.
- `model.tflite`: The converted model (ensure it is generated or copied here).

## Setup
1. **Prerequisite:** Ensure you are using **Python 3.9**.
   - Vital for TFLite Converter compatibility.
   
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

**1. Convert Model:**
```bash
python convert_to_tflite.py
```
- This will generate `movenet_multipose_lighting_quant.tflite`.

**2. Test Model:**
```bash
python test_model.py
```
- Runs the TFLite model on `../dataset/queda.mp4` (Check/Update path in `test_model.py`).
