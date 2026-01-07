
import requests
import tarfile
import io
import os
import shutil

# URL for MoveNet SinglePose Lightning INT8 (Quantized)
# This model is specifically designed for CPU/NPU execution and does NOT require GPU.
URL = "https://tfhub.dev/google/movenet/singlepose/lightning/tflite/int8/4?tf-hub-format=compressed"
SAVE_DIR = os.path.join(os.path.dirname(__file__), "02_TFLite_Laptop")
MODEL_FILENAME = "movenet_singlepose_lightning_int8.tflite"

def download_and_extract():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        
    print(f"Downloading model from {URL}...")
    try:
        response = requests.get(URL)
        response.raise_for_status()
        
        print("Extracting...")
        with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as tar:
            # The tar usually contains a single .tflite file with a generic name or specific name
            tar.extractall(path=SAVE_DIR)
            
            # Find the .tflite file
            for member in tar.getnames():
                if member.endswith(".tflite"):
                    extracted_path = os.path.join(SAVE_DIR, member)
                    new_path = os.path.join(SAVE_DIR, MODEL_FILENAME)
                    os.rename(extracted_path, new_path)
                    print(f"Success! Model saved to: {new_path}")
                    return new_path
                    
    except Exception as e:
        print(f"Error downloading model: {e}")
        return None

if __name__ == "__main__":
    download_and_extract()
