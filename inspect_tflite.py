import tensorflow as tf
import numpy as np

model_path = "02_TFLite_Laptop/model_thunder_int8.tflite"

try:
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Check all tensors for quantization
    total_tensors = 0
    quantized_tensors = 0
    float_tensors = 0
    
    print(f"--- Model Inspection: {model_path} ---")

    for i, detail in enumerate(input_details):
        dtype = detail['dtype']
        shape = detail['shape']
        print(f"Input {i}: {detail['name']}, Type: {dtype}, Shape: {shape}")

    for i, detail in enumerate(output_details):
        dtype = detail['dtype']
        shape = detail['shape']
        print(f"Output {i}: {detail['name']}, Type: {dtype}, Shape: {shape}")

    # Heuristic: Check if majority of tensors are int8 or float
    for tensor_details in interpreter.get_tensor_details():
        total_tensors += 1
        curr_type = tensor_details['dtype']
        if curr_type == np.int8 or curr_type == np.uint8:
            quantized_tensors += 1
        elif curr_type == np.float32:
            float_tensors += 1
            
    print(f"\nTensor Stats:")
    print(f"Total Tensors: {total_tensors}")
    print(f"Float32 Tensors: {float_tensors}")
    print(f"Int8/Uint8 Tensors: {quantized_tensors}")
    
    if quantized_tensors > float_tensors:
         print("\nVERDICT: This is likely an INT8 QUANTIZED model.")
    elif float_tensors > 0:
         print("\nVERDICT: This is likely a FLOAT32 model.")

    if input_details[0]['dtype'] == np.float32:
        print("Input Layer expects: FLOAT32")
    else:
        print(f"Input Layer expects: {input_details[0]['dtype']}")

except Exception as e:
    print(f"Error inspecting model: {e}")
