
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np

def representative_data_gen():
    # Use the existing video for calibration data, or random fallback
    video_path = "queda.mp4"
    cap = cv2.VideoCapture(video_path)
    
    num_calibration_steps = 100
    
    # Check if video opened successfully
    use_random = not cap.isOpened()
    if use_random:
        print("Warning: 'queda.mp4' not found. Using random noise for calibration (Not recommended for high accuracy).")
    
    for _ in range(num_calibration_steps):
        if use_random:
            # MoveNet Lightning expects [1, 192, 192, 3] usually.
            # Generate int32 random noise [0, 255]
            img = tf.random.uniform((1, 192, 192, 3), minval=0, maxval=255, dtype=tf.int32)
        else:
            ret, frame = cap.read()
            if not ret:
                break
            # Resize and process
            # Cast to int32 because model expects int32
            img_tf = tf.expand_dims(frame, axis=0)
            img_tf = tf.image.resize_with_pad(img_tf, 192, 192)
            img = tf.cast(img_tf, tf.int32)
            
        yield [img]
        
    if not use_random:
        cap.release()

def convert():
    print("Loading model from TF Hub...")
    model = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")
    concrete_func = model.signatures["serving_default"]

    print("Setting up TFLite Converter...")
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    
    # --- DYNAMIC RANGE QUANTIZATION (Best for RPi4 CPU) ---
    # Full Int8 fails due to MoveNet's custom slicing ops. 
    # Dynamic Range gives 4x size reduction and good speedup on CPU.
    
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Allow custom ops (Flex/Select) for MoveNet
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    
    # Remove strict int8 input/output requirements to avoid signature conflicts.
    # The models input will remain int32 (as per Hub signature) and output float32.
    
    print("Converting model with Dynamic Range Quantization...")
    
    tflite_model = converter.convert()
    
    output_path = "model_quant_dynamic.tflite"
    with open(output_path, "wb") as f:
        f.write(tflite_model)
        
    print(f"Success! Model saved to {output_path}")
    print(f"Size: {len(tflite_model) / 1024 / 1024:.2f} MB")
     
        

if __name__ == "__main__":
    convert()
