import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
import numpy as np


import sys
#from IPython.display import Image

# Define a simple CNN model
model = keras.Sequential([
#    layers.Conv2D(filters=16, kernel_size=(2, 2), strides=(2, 2), input_shape=(8, 8, 16))  # First conv layer
    layers.MaxPooling2D(pool_size=(2, 2)),  # Downsample the feature maps
#    layers.Conv2D(64, (3, 3), activation='relu'),  # Second conv layer
#    layers.MaxPooling2D(pool_size=(2, 2)),  # Downsample again
#    layers.Flatten(),  # Flatten the output for the dense layer
#    layers.Dense(128, activation='relu'),  # Fully connected layer
#    layers.Dense(10, activation='softmax')  # Output layer (10 classes)
])

# Compile the model
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.build((None, 8,8,16))
model.summary()

tf.saved_model.save(model, sys.argv[1])


#input_tensor = tf.random.normal((1, 8, 8, 16))  # Random tensor inpu
np_input = np.full((1, 8, 8, 16), 1, dtype=int)
input_tensor = tf.constant(np_input)
print("Input shape:", input_tensor.shape, '\n', input_tensor)
output_tensor = model(input_tensor)
print("Output shape:", output_tensor.shape, '\n', output_tensor)


