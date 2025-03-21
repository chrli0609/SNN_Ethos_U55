import sys

import tensorflow as tf
import numpy as np


# Model Architecture parameters
INPUT_SIZE = 1
HIDDEN_SIZES = [2]
OUTPUT_SIZE = 1

#Batch Size
BATCH_SIZE = 1




# Example model
class SpikeNeuron(tf.Module):
	def __init__(self):
		super().__init__()
		self.mem = tf.Variable(0.0)  # Single membrane potential for the neuron

		# Neuron parameters
		self.R = 5.1
		self.C = 5e-3
		self.time_step = 1e-3
		self.threshold = 1.0



	@tf.function
	def __call__(self, i_app):

		tau_mem = self.R * self.C

		# Aggregate input (sum of weighted inputs)
		total_input = tf.reduce_sum(i_app)

		# Compute spike (binary output in uint8)
		#spk = tf.cast(tf.greater(self.mem, self.threshold), dtype=tf.float32)

		if tf.greater(self.mem, self.threshold):
			spk = tf.constant([1.0])
		else:
			spk = tf.constant([0.0])


		self.mem.assign(self.mem + (self.time_step / tau_mem) * (-self.mem + total_input))


		#return tf.reshape(spk, [-1])
		#print("spk", spk)
		#print("spk.size", tf.size(spk))
		return spk


class SpikeMLP(tf.Module):
	def __init__(self, input_size, hidden_sizes, output_size):
		super().__init__()

		# Define layer sizes
		layer_sizes = [input_size] + hidden_sizes + [output_size]

        # Initialize layers and weights
		self.layers = []
		self.weights = []

		for i in range(len(layer_sizes) - 1):
			num_neurons = layer_sizes[i + 1]
			self.layers.append([SpikeNeuron() for _ in range(num_neurons)])  # Create neurons
			self.weights.append(
				tf.Variable(tf.random.normal([layer_sizes[i], num_neurons], stddev=0.1))
			)  # Weight matrix

	@tf.function(input_signature=[tf.TensorSpec(shape=[BATCH_SIZE, INPUT_SIZE], dtype=tf.float32)])
	def __call__(self, x):
	

		for i in range(len(self.layers)):
			x = tf.matmul(x, self.weights[i])  # Linear transformation

			# Apply spiking activation per neuron
			new_x = []
			for neuron in self.layers[i]:
				tmp = neuron(x)
				#print("neuron(x):", tmp)
				new_x.append(tmp)
		

			#print("x", x)
			x = tf.stack(new_x, axis=1)  # Combine outputs into a tensor
			#print("post stack x", x)


		return x




# Instantiate model



model = SpikeMLP(INPUT_SIZE, HIDDEN_SIZES, OUTPUT_SIZE)


# Save model
tf.saved_model.save(model, sys.argv[1])

