import numpy as np


print("FC2")

INPUT_LAYER_SIZE_INIT = 32 
OUTPUT_LAYER_SIZE_INIT = 32

def next_multiple_of_8(x: int) -> int:
    return ((x + 7) // 8) * 8


INPUT_LAYER_SIZE = next_multiple_of_8(INPUT_LAYER_SIZE_INIT)
OUTPUT_LAYER_SIZE = next_multiple_of_8(OUTPUT_LAYER_SIZE_INIT)

print("INPUT_LAYER_SIZE", INPUT_LAYER_SIZE)
print("OUTPUT_LAYER_SIZE", OUTPUT_LAYER_SIZE)


#in_padding = INPUT_LAYER_SIZE - INPUT_LAYER_SIZE_INIT
#out_padding = OUTPUT_LAYER_SIZE - OUTPUT_LAYER_SIZE_INIT


## Unique
## Define Weights


### Get weights and biases
#weights_init = np.load("model_params/fc2_weights.npy")
#bias_init = np.load("model_params/fc2_bias.npy")

#print("weights_init:", weights_init.shape, "\n", weights_init)

##np.save("weights_init.npy", weights_init)
##np.save("bias_init.npy", weights_init)



## Append by how much we are missing
#weights_padded = np.pad(weights_init, ((0, out_padding), (0, in_padding)), mode='constant')
#bias_padded = np.pad(bias_init, (0, out_padding), mode='constant')

#print("weights_padded:", weights_padded.shape, "\n", weights_padded)
#print("biases_padded:", bias_padded.shape, "\n", bias_padded)

##np.save("weights_padded.npy", weights_padded)
##np.save("bias_padded.npy", weights_padded)


## Reshape weights
#weights_reshaped = weights_padded.reshape(OUTPUT_LAYER_SIZE, 1, 1, INPUT_LAYER_SIZE)

#print("weights_reshaped:", weights_reshaped.shape, "\n", weights_reshaped)

##np.save("weights_reshaped.npy", weights_reshaped)



#weights_volume_ohwi = weights_reshaped
#bias_list = bias_padded

# Unique
# Define Weights
ALL_WEIGHT_VALUES = 0.1
ALL_BIAS_VALUES = 0

weights_volume_ohwi = ALL_WEIGHT_VALUES * np.ones((OUTPUT_LAYER_SIZE, 1, 1, INPUT_LAYER_SIZE))

#Biases
bias_list = []
for i in range(OUTPUT_LAYER_SIZE):
#    #bias_list.append(np.int64(i%4))
    bias_list.append(np.int64(ALL_BIAS_VALUES))


# Unique
# Define Weights
##### Set LIF Param values #######

# Generate Beta values
beta_list = []
for i in range(OUTPUT_LAYER_SIZE):
    beta_list.append(0.9)

# Generate Vth values
vth_list = []
for i in range(OUTPUT_LAYER_SIZE):
    vth_list.append(1)

    

