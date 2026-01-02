
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import sys

# Function to generate representative data (Essential for INT8)
def representative_data_gen():
    # Use the video again for calibration
    video_path = "../dataset/queda.mp4" 
    # Note: Ensure path is correct relative to where you run this script
    # If file not found, we generate random noise, but real data is better for accuracy.
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Warning: Video {video_path} not found. Using random noise for calibration.")
        for _ in range(20):
             yield [np.random.randint(0, 255, size=(1, 160, 320, 3)).astype(np.uint8)]
        return

    print("Calibrating with video data...")
    for _ in range(50):
        ret, frame = cap.read()
        if not ret: break
        
        # Resize to MoveNet Standard (160 height, 320 width)
        img = cv2.resize(frame, (320, 160))
        img = np.expand_dims(img, axis=0) # [1, 160, 320, 3]
        
        # MoveNet Multipose usually expects int32 for the TFHub model, 
        # but for Full Int8 TFLite, we usually feed uint8 or int8.
        # The Converter handles the cast if we set inference_input_type properly.
        # We pass standard uint8 image data here.
        yield [img.astype(np.uint8)]
        
    cap.release()

def convert():
    print("Loading MoveNet from TFHub...")
    model = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")
    concrete_func = model.signatures["serving_default"]

    print("Configuring TFLite Converter for Microcontrollers (Full INT8)...")
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    
    # 1. OPTIMIZE FOR SIZE/LATENCY
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # 2. FULL INTEGER QUANTIZATION (Required for ESP32/TFLM)
    converter.representative_dataset = representative_data_gen
    
    # Enforce integer only operations (No float fallback allowed)
    # This is the strict check. If MoveNet has unsupported ops, this might fail or require "SELECT_TF_OPS".
    # ESP32 TFLM does *not* support SELECT_TF_OPS easily. 
    # We try to enforce builtins. 
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    
    # 3. SET INPUT/OUTPUT TYPES
    converter.inference_input_type = tf.uint8  # Standard for ESP32-CAM images
    converter.inference_output_type = tf.uint8 # Output usually dequantized manually

    try:
        print("Converting... (This might fail if model uses unsupported ops)")
        tflite_model = converter.convert()
        
        # Save TFLite file
        output_path = "movenet_esp32_int8.tflite"
        with open(output_path, "wb") as f:
            f.write(tflite_model)
        
        print(f"Success! Model saved to {output_path}")
        print(f"Size: {len(tflite_model)/1024:.2f} KB")
        
        # Generate C Header file for Arduino/ESP32
        print("Generating C header file for Arduino...")
        hex_array = ", ".join([f"0x{b:02x}" for b in tflite_model])
        c_code = f"""
#ifndef MOVENET_MODEL_H
#define MOVENET_MODEL_H

const unsigned char movenet_model[] = {{
  {hex_array}
}};

const int movenet_model_len = {len(tflite_model)};

#endif
        """
        with open("movenet_model.h", "w") as f:
            f.write(c_code)
        print("Header file 'movenet_model.h' generated.")

    except Exception as e:
        print("\n[ERROR] Optimization Failed!")
        print(f"Reason: {e}")
        print("\nEXPLANATION FOR USER:")
        print("The standard MoveNet model uses complex operations (like StridedSlice) that likely")
        print("cannot be fully quantized to pure INT8 without TF Select ops.")
        print("Standard ESP32-CAM libraries (TFLM) do not support these operations out of the box.")
        print("RECOMMENDATION: Use the 'Video Streaming' method instead.")

if __name__ == "__main__":
    convert()
