
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np

def representative_data_gen():
    # Use the existing video for calibration data
    video_path = "queda.mp4"
    cap = cv2.VideoCapture(video_path)
    
    num_calibration_steps = 100
    for _ in range(num_calibration_steps):
        if not cap.isOpened():
            break
        ret, frame = cap.read()
        if not ret:
            break
            
        # Resize to model input size (192, 192, 3) or (160, 320) depending on variant
        # Multipose Lightning 1 input is [1, 160, 320, 3] usually, let's verify or use the resizing from fall.py
        # fall.py uses: tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 160, 320)
        
        input_size = 320
        # Resize and pad logic similar to fall.py to ensure distribution matches
        img = frame.copy()
        img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 160, 320)
        img = tf.cast(img, dtype=tf.int32) # Model expects int32 input
        
        # TFLite converter expects float input for calibration if the input is quantized, 
        # but the source model takes int32? 
        # Actually, MoveNet Multipose Lightning expects [1, 160, 320, 3] int32 tensor.
        # But for representative dataset, we usually yield the input tensors.
        
        yield [img]
        
    cap.release()

def convert():
    print("Loading model from TF Hub...")
    model = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")
    concrete_func = model.signatures["serving_default"]

    print("Setting up TFLite Converter...")
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    
    # Dynamic Range Quantization with TF Select Ops (to handle StridedSlice)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS # Required for MoveNet's slicing operations
    ]
    
    print("Converting model with Dynamic Range Quantization (TF Select Enabled)...")
    tflite_model = converter.convert()
    
    output_path = "movenet_multipose_lighting_quant.tflite"
    with open(output_path, "wb") as f:
        f.write(tflite_model)
        
    print(f"Success! Model saved to {output_path}")
    print(f"Size: {len(tflite_model) / 1024 / 1024:.2f} MB")
        

if __name__ == "__main__":
    convert()
