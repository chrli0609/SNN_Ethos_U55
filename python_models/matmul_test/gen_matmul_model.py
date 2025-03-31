import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

import sys

# Create the model
model = keras.Sequential([
    layers.Input(shape=(28, 28)),  # Input shape (28,28)
    layers.Flatten(),              # Flatten into a 1D vector of 784 elements
    layers.Dense(1008, activation="relu")  # Fully connected layer with 1008 outputs
])

# Save the model in float32 format
saved_model_dir = sys.argv[1]
model.save(saved_model_dir)

# Function to generate a representative dataset for quantization
def representative_data_gen():
    for _ in range(100):
        yield [np.random.randint(0, 256, size=(1, 28, 28), dtype=np.uint8).astype(np.float32)]  # Convert uint8 -> float32

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Enable quantization
converter.representative_dataset = representative_data_gen  # Provide sample dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]  # Ensure only INT8 operations
converter.inference_input_type = tf.uint8  # Input type as uint8
converter.inference_output_type = tf.uint8  # Output type as uint8
converter.fully_quantize = True  # Ensure full quantization

# Convert the model
tflite_model = converter.convert()

# Save the quantized TFLite model
tflite_model_path = sys.argv[2]
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print("✅ Fully Quantized TFLite model saved at:", tflite_model_path)

