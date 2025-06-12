import numpy as np
from extra_func import next_multiple_of_8


print("FC1")

INPUT_LAYER_SIZE_INIT = 64
OUTPUT_LAYER_SIZE_INIT = 64

def next_multiple_of_8(x: int) -> int:
    return ((x + 7) // 8) * 8


INPUT_LAYER_SIZE = next_multiple_of_8(INPUT_LAYER_SIZE_INIT)
OUTPUT_LAYER_SIZE = next_multiple_of_8(OUTPUT_LAYER_SIZE_INIT)

print("INPUT_LAYER_SIZE", INPUT_LAYER_SIZE)
print("OUTPUT_LAYER_SIZE", OUTPUT_LAYER_SIZE)


in_padding = INPUT_LAYER_SIZE - INPUT_LAYER_SIZE_INIT
out_padding = OUTPUT_LAYER_SIZE - OUTPUT_LAYER_SIZE_INIT


# Unique
# Define Weights


## Get weights and biases
weights_init = np.load("model_params/fc1_weights.npy")
bias_init = np.load("model_params/fc1_biases.npy")

print("weights_init:", weights_init.shape, "\n", weights_init)

#np.save("weights_init.npy", weights_init)
#np.save("bias_init.npy", weights_init)



# Append by how much we are missing
weights_padded = np.pad(weights_init, ((0, out_padding), (0, in_padding)), mode='constant')
bias_padded = np.pad(bias_init, (0, out_padding), mode='constant')

print("weights_padded:", weights_padded.shape, "\n", weights_padded)
print("biases_padded:", bias_padded.shape, "\n", bias_padded)

#np.save("weights_padded.npy", weights_padded)
#np.save("bias_padded.npy", weights_padded)


# Reshape weights
weights_reshaped = weights_padded.reshape(OUTPUT_LAYER_SIZE, 1, 1, INPUT_LAYER_SIZE)

print("weights_reshaped:", weights_reshaped.shape, "\n", weights_reshaped)

#np.save("weights_reshaped.npy", weights_reshaped)



weights_volume_ohwi = weights_reshaped
bias_list = bias_padded

print("Max weight value:", weights_volume_ohwi.max())
print("Min weight value", weights_volume_ohwi.min())
print("Max bias value", bias_list.max())
print("Min bias value", bias_list.min())

# Unique
# Define Weights
##### Set LIF Param values #######

# Generate Beta values
beta_list = []
for i in range(OUTPUT_LAYER_SIZE):
    beta_list.append(0.95)

# Generate Vth values
vth_list = []
for i in range(OUTPUT_LAYER_SIZE):
    vth_list.append(1)

    

