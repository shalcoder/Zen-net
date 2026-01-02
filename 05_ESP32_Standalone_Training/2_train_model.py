
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

IMG_SIZE = 96
BATCH_SIZE = 32
EPOCHS = 10
DATA_DIR = "dataset"

def train_and_convert():
    # 1. Load Data
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    
    train_generator = datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training'
    )
    
    val_generator = datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation'
    )
    
    if train_generator.samples == 0:
        print("Error: No images found. Please run 1_extract_data.py first!")
        return

    # 2. Build MobileNetV2 (Tiny Version)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        alpha=0.35, # The smallest alpha
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False # Transfer Learning
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1, activation='sigmoid') # Binary Classification
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # 3. Train
    print("Training Model...")
    model.fit(train_generator, epochs=EPOCHS, validation_data=val_generator)
    
    # 4. Fine Tuning (Optional but recommended)
    base_model.trainable = True
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_generator, epochs=5, validation_data=val_generator)

    # 5. Convert to TFLite (INT8)
    print("Converting to TFLite (INT8)...")
    
    def representative_data_gen():
        # Use simple generator logic for calibration
        for _ in range(100):
            x, _ = next(train_generator)
            yield [x.astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    
    # Enforce Integer Ops for ESP32
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_model = converter.convert()
    
    with open("fall_detection_esp32.tflite", "wb") as f:
        f.write(tflite_model)
        
    # 6. Generate C Header
    hex_array = ", ".join([f"0x{b:02x}" for b in tflite_model])
    c_code = f"""
#ifndef MODEL_H
#define MODEL_H
const unsigned char model_data[] = {{ {hex_array} }};
const int model_data_len = {len(tflite_model)};
#endif
    """
    with open("model_data.h", "w") as f:
        f.write(c_code)
        
    print("Success! 'model_data.h' generated for Arduino.")

if __name__ == "__main__":
    train_and_convert()
