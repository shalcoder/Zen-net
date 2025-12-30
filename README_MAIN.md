# Human Fall Detection System

This repository contains a comprehensive Fall Detection System using MoveNet (Pose Estimation) with logic for detecting:
- **Normal Activities**: Standing, Walking, Sitting, Intentional Lying.
- **Emergency Falls**: Rapid collapse, Fainted/Immobile states, Distress/Thrashing.

## Project Structure

### 1. [01_Normal_TF_Hub](01_Normal_TF_Hub/)
The original, full-power implementation using TensorFlow Hub.
- Best for: Laptops/PCs with strong CPUs/GPUs.
- Features: Live Webcam & Video Analysis, Matplotlib Graphs.

### 2. [02_TFLite_Laptop](02_TFLite_Laptop/)
Scripts to convert and test the model in TFLite format on a PC.
- Best for: Optimizing the model before deployment.
- Features: INT8/Dynamic Range Quantization conversion.

### 3. [03_Raspberry_Pi](03_Raspberry_Pi/)
The deployment-ready code for Raspberry Pi 4.
- Best for: Edge consumption (Low RAM/CPU usage).
- Features: Uses `tflite-runtime`, minimal dependencies.

### 4. [dataset](dataset/)
Place your test videos (e.g., `queda.mp4`) here.

### 5. [models](models/)
Contains the converted `.tflite` models.

## Quick Start
1. Navigate to the folder matching your platform.
2. Install the specific `requirements.txt`.
3. Follow the `README.md` inside that folder.
