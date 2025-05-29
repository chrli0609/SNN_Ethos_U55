import numpy as np


INPUT_LAYER_SIZE = 1000
OUTPUT_LAYER_SIZE = 10


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

    

