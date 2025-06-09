import numpy as np
from extra_func import next_multiple_of_8

print("FC0")

INPUT_LAYER_SIZE_INIT = 28*28
OUTPUT_LAYER_SIZE_INIT = 64

# Constraints on input/output

'''
1. Output size % 8 = 0
2. All DMA src and dst addresses MUST be 16-byte aligned, if we have use dma to place Weights in DTCM --> only output_size needs to be % 8
'''






INPUT_LAYER_SIZE = INPUT_LAYER_SIZE_INIT
OUTPUT_LAYER_SIZE = next_multiple_of_8(OUTPUT_LAYER_SIZE_INIT)
print("OUTPUT_LAYER_INIT", OUTPUT_LAYER_SIZE_INIT)
print("OUTPUT_LAYER_SIZE", OUTPUT_LAYER_SIZE)

print("INPUT_LAYER_SIZE", INPUT_LAYER_SIZE)
print("OUTPUT_LAYER_SIZE", OUTPUT_LAYER_SIZE)


in_padding = INPUT_LAYER_SIZE - INPUT_LAYER_SIZE_INIT
out_padding = OUTPUT_LAYER_SIZE - OUTPUT_LAYER_SIZE_INIT


# Unique
# Define Weights


## Get weights and biases
weights_init = np.load("model_params/fc0_weights.npy")
bias_init = np.load("model_params/fc0_biases.npy")

# Append by how much we are missing
print("weights_init:", weights_init.shape, "\n", weights_init)
weights_padded = np.pad(weights_init, ((0, out_padding), (0, in_padding)), mode='constant')
bias_padded = np.pad(bias_init, (0, out_padding), mode='constant')

print("weights_padded:", weights_padded.shape, "\n", weights_padded)
print("biases_padded:", bias_padded.shape, "\n", bias_padded)

weights_reshaped = weights_padded.reshape(OUTPUT_LAYER_SIZE, 1, 1, INPUT_LAYER_SIZE)


print("weights_reshaped:", weights_reshaped.shape, "\n", weights_reshaped)

weights_volume_ohwi = weights_reshaped
bias_list = bias_padded

print("Max weight value:", weights_volume_ohwi.max())
print("Min weight value", weights_volume_ohwi.min())
print("Max bias value", bias_list.max())
print("Min bias value", bias_list.min())



## Define Weights
#ALL_WEIGHT_VALUES = -0.1
#ALL_BIAS_VALUES = 0

#weights_volume_ohwi = ALL_WEIGHT_VALUES * np.ones((OUTPUT_LAYER_SIZE, 1, 1, INPUT_LAYER_SIZE))

##Biases
#bias_list = []
#for i in range(OUTPUT_LAYER_SIZE):
##    #bias_list.append(np.int64(i%4))
    #bias_list.append(np.int64(ALL_BIAS_VALUES))



# Unique
# Define Weights
##### Set LIF Param values #######

# Generate Beta values
beta_list = []
for i in range(OUTPUT_LAYER_SIZE):
    beta_list.append(20)

# Generate Vth values
vth_list = []
for i in range(OUTPUT_LAYER_SIZE):
    vth_list.append(1)

    

