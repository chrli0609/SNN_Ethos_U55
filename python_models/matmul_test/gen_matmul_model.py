import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import sys


IN_DIM = 16
OUT_DIM = 32


# Create the model
model = keras.Sequential([
    layers.Input(shape=(IN_DIM,)),  # Input shape is 1D with IN_DIM elements
    layers.Dense(OUT_DIM, activation=None)  # Fully connected layer with OUT_DIM outputs
])



##### Set weight and biases: ######

# Get the initial weights and biases
weights = model.get_weights()
print("Original weight shapes:", [w.shape for w in weights])
print(model.get_weights())

# Create new weights manually
#new_weight_matrix = np.random.randn(IN_DIM, OUT_DIM).astype(np.float32)  # Weight matrix shape (IN_DIM, OUT_DIM)
#new_bias_vector = np.zeros(OUT_DIM, dtype=np.float32)  # Bias vector shape (OUT_DIM,)

#new_weight_matrix = np.zeros((IN_DIM, OUT_DIM))
#new_bias_vector = np.zeros((OUT_DIM))

new_weight_matrix = np.ones((IN_DIM, OUT_DIM))
new_bias_vector = np.ones((OUT_DIM))

#new_weight_matrix = np.tile(np.arange(1, 5), (32, 4)).T  # Repeat [1, 2, 3, 4] across 32 rows
# Convert weight matrix to float32
#new_weight_matrix = np.tile(np.arange(1, 5), (32, 4)).reshape(IN_DIM, OUT_DIM).astype(np.float32)

# Convert bias vector to float32
#new_bias_vector = np.array([0, 1, 2, 3] * 8).astype(np.float32)



print("new_weight_matrix.shape", new_weight_matrix.shape)
#new_bias_vector = np.tile(np.arange(0, 4), 4*2)


# Set the new weights
model.set_weights([new_weight_matrix, new_bias_vector])

# Verify the update
updated_weights = model.get_weights()
print("Updated weight shapes:", [w.shape for w in updated_weights])
print(model.get_weights())

##########################################




# Set input
test_in = tf.convert_to_tensor(np.ones((1, IN_DIM)))
# Check what the expected output is
test_out = model(test_in)
print("test_in", test_in)
print("test_out", test_out)






# Save the model in float32 format
saved_model_dir = sys.argv[1]
model.save(saved_model_dir)

# Function to generate a representative dataset for quantization
def representative_data_gen():
    for _ in range(100):
        yield [np.random.randint(0, 256, size=(1, IN_DIM), dtype=np.uint8).astype(np.float32)]  # Convert uint8 -> float32

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

