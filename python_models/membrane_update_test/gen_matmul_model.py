import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import sys





class LIFLayer(layers.Layer):
    def __init__(self, num_neurons, beta=0.9, threshold=1.0, **kwargs):
        super(LIFLayer, self).__init__(**kwargs)
        self.num_neurons = num_neurons
        self.beta = beta
        self.threshold = threshold
    
    def build(self, input_shape):
        # Initialize weights (shape: [input_dim, num_neurons])
        self.w = self.add_weight(shape=(input_shape[0][-1], self.num_neurons),
                                 initializer="random_normal",
                                 trainable=True,
                                 name="weights")
        # Initialize biases (shape: [num_neurons])
        self.b = self.add_weight(shape=(self.num_neurons,),
                                 initializer="zeros",
                                 trainable=True,
                                 name="biases")
    
    def call(self, inputs):
        x, mem = inputs  # Separate input spikes and membrane potential
        #x = tf.cast(x, tf.float32)
        #mem = tf.cast(mem, tf.float32)
        
        # Compute LIF dynamics
        #spk = tf.cast(mem > self.threshold, tf.float32)
        spk = tf.nn.relu(self.threshold - tf.nn.relu(self.threshold - mem))
        mem = self.beta * mem + tf.matmul(x, self.w) + self.b - spk * self.threshold
        
        return mem, spk  # Output spikes and updated membrane potential

# Create the model
input_spikes = keras.Input(shape=(784,))  # Input spikes
input_mem = keras.Input(shape=(1008,))  # Membrane potential
lif_layer, new_mem = LIFLayer(1008)([input_spikes, input_mem])
model = keras.Model(inputs=[input_spikes, input_mem], outputs=[lif_layer, new_mem])

# Summary
model.summary()

# Debugging: Print model weight shapes
weights = model.get_weights()
print("Model expects", len(weights), "weight tensors.")
for i, w in enumerate(weights):
    print(f"Weight {i}: shape {w.shape}")

# Set new weights and biases
new_weight_matrix = np.zeros((784, 1008), dtype=np.float32)
new_bias_vector = np.zeros((1008,), dtype=np.float32)
model.set_weights([new_weight_matrix, new_bias_vector])



















# Save the model in float32 format
saved_model_dir = sys.argv[1]

print("model", model)
print("saved_model_dir", saved_model_dir)
model.save(saved_model_dir)




# Function to generate a representative dataset for quantization
def representative_data_gen():
    for _ in range(100):
        input_spikes = np.random.randint(0, 256, size=(1, 784), dtype=np.uint8).astype(np.float32)
        input_mem = np.random.rand(1, 1008).astype(np.float32)  # Random membrane potentials
        yield [input_spikes, input_mem]

# Convert the model to TensorFlow Lite format
saved_model_dir = sys.argv[1]
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
