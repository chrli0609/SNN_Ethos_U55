import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
import numpy as np


import sys


# Define input shapes
input_shape = (4, 4, 16)
addition_shape = (4, 4, 16)

# Define inputs
x_input = keras.Input(shape=input_shape)
y_input = keras.Input(shape=addition_shape)

# Elementwise addition
added = layers.Add()([x_input, y_input])

# Create the model
model = keras.Model(inputs=[x_input, y_input], outputs=added)

# Display the model summary
model.summary()



# Example input
import numpy as np
#x_test = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]])
#y_test = np.array([[16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]])
x_test = np.ones((1, 4, 4, 16))
y_test = np.ones((1, 4, 4, 16)) * 2
print("IFM1:", x_test)
print("IFM2:", y_test)
print("Output:", model([x_test, y_test]).numpy())

tf.saved_model.save(model, sys.argv[1])
