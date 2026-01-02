# ESP32 Standalone Fall Detection - Colab Training Script
# This notebook:
# 1. Downloads the Kaggle Dataset via API.
# 2. Extracts frames from videos to create an image dataset.
# 3. Trains a MobileNetV2 (Alpha 0.35) model.
# 4. Converts it to TFLite (INT8) for ESP32.
# 5. Generates the 'model_data.h' for Arduino.

# --- CELL 1: Setup & Kaggle API ---
# Upload your 'kaggle.json' to Colab first!
import os

# Install Kaggle
# !pip install -q kaggle

# Setup Kaggle Config (Reliable File Method)
# PLEASE UPLOAD 'kaggle.json' TO THE COLAB FILES TAB BEFORE RUNNING!
if not os.path.exists('kaggle.json'):
    print("❌ ERROR: 'kaggle.json' not found in Colab files!")
    print("Please upload it to the Files sidebar first.")
    # Halt execution if file is missing (in a notebook this might not stop cell execution fully but prints clearly)
else:
    # Force clean old config to fix "KeyError: username"
    #!rm -rf /root/.kaggle
    #!mkdir /root/.kaggle
    #!cp kaggle.json /root/.kaggle/
    #!chmod 600 /root/.kaggle/kaggle.json
    #print("✅ Kaggle Config Refreshed.")

# --- CELL 2: Download Dataset ---
# Downloading 'payutch/fall-video-dataset'
#!kaggle datasets download -d payutch/fall-video-dataset
#!unzip -q fall-video-dataset.zip -d raw_dataset
   print("Dataset Downloaded and Extracted.")

# --- CELL 3: Data Processing (Video -> Images) ---
import cv2
import glob
import numpy as np
from tqdm import tqdm

# Create Image Dataset Structure
#!mkdir -p dataset/fall
#!mkdir -p dataset/normal

def extract_frames(video_path, output_folder, label, gap=5):
    cap = cv2.VideoCapture(video_path)
    count = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        if count % gap == 0: # Save every 5th frame to reduce redundancy
            # Resize to ESP32 standard (96x96)
            # Center crop or Resize? Resize is simpler for now.
            resized = cv2.resize(frame, (96, 96))
            fname = f"{label}_{os.path.basename(video_path)}_{saved}.jpg"
            cv2.imwrite(os.path.join(output_folder, fname), resized)
            saved += 1
        count += 1
    cap.release()

# Process Fall Videos
fall_videos = glob.glob("raw_dataset/Fall/Raw_Video/*.mp4")
print(f"Processing {len(fall_videos)} Fall Videos...")
for v in tqdm(fall_videos):
    extract_frames(v, "dataset/fall", "fall", gap=10) # Less frequent for fall (shorter usually)

# Process Normal Videos
normal_videos = glob.glob("raw_dataset/No_Fall/Raw_Video/*.mp4")
print(f"Processing {len(normal_videos)} Normal Videos...")
for v in tqdm(normal_videos):
    extract_frames(v, "dataset/normal", "normal", gap=15) # More frequent skip for long ADL

print("Data Extraction Complete.")

# --- CELL 4: Model Training (MobileNetV2) ---
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 96
BATCH_SIZE = 32
EPOCHS = 15

# Augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    'dataset',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    'dataset',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# Tiny MobileNetV2
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    alpha=0.35, # Smallest alpha for ESP32
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("Starting Training...")
model.fit(train_generator, epochs=EPOCHS, validation_data=val_generator)

# Fine Tuning
base_model.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_generator, epochs=5, validation_data=val_generator)

# --- CELL 5: Convert to TFLite (INT8 for ESP32) ---
def representative_data_gen():
    for _ in range(100):
        x, _ = next(train_generator)
        yield [x]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

# Save
with open('fall_detection_esp32.tflite', 'wb') as f:
    f.write(tflite_model)

print(f"Model Converted! Size: {len(tflite_model)/1024:.2f} KB")

# --- CELL 6: Generate Arduino Header ---
hex_array = ", ".join([f"0x{b:02x}" for b in tflite_model])
c_code = f"""
#ifndef MODEL_H
#define MODEL_H

// Auto-generated by Colab Script
const unsigned char model_data[] = {{
  {hex_array}
}};
const int model_data_len = {len(tflite_model)};

#endif
"""

with open('model_data.h', 'w') as f:
    f.write(c_code)

print("SUCCESS! Download 'model_data.h' and put it in your Arduino sketch folder.")
from google.colab import files
files.download('model_data.h')
