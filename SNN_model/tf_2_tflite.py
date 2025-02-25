import tensorflow as tf
import numpy as np

import sys

#saved_model_dir = "saved_model/my_model"
#tflite_model_path = "tflite_model/tflite_model.tflite"
saved_model_dir = sys.argv[1]
tflite_model_path = sys.argv[2]


# Load the SavedModel
model = tf.saved_model.load(saved_model_dir)

print("signatures", model.signatures)


def representative_data_gen():
	for _ in range(100):
		sample_input = np.random.rand(128, 28*28).astype(np.float32)
		yield [sample_input]



# Convert model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_typ = tf.uint8
converter.nference_output_type = tf.uint8

tflite_model = converter.convert()



#save quantized TFLite Model


with open(tflite_model_path, "wb") as f:
	f.write(tflite_model)


print("Quantized TFLite model saved at:", tflite_model_path)



