import numpy as np
from pathlib import Path


from config_file import INIT_LAYER_SIZES_LIST, LAYER_BASE_NAME, NUM_TIME_STEPS

MANUAL_INPUT_VAL = 10
MANUAL_WEIGHT_VAL = 0.1
FOLDER_NAME = Path("model_params")

# We assume this is 16
MODEL_IN_SIZE = INIT_LAYER_SIZES_LIST[0]
# Make this 2 just for easier debugging

for i in range(len(INIT_LAYER_SIZES_LIST)-1):
    input_size = INIT_LAYER_SIZES_LIST[i]
    output_size = INIT_LAYER_SIZES_LIST[i+1]

    weights = np.full((output_size, input_size), MANUAL_WEIGHT_VAL)
    biases = np.full(output_size, MANUAL_WEIGHT_VAL)

    np.save(FOLDER_NAME / Path(LAYER_BASE_NAME+str(i)+"_weights.npy"), weights)
    np.save(FOLDER_NAME / Path(LAYER_BASE_NAME+str(i)+"_biases.npy"), biases)


# Set testing data
test_input_0 = np.full((NUM_TIME_STEPS, 1, 1, 4, 4), MANUAL_INPUT_VAL)
test_target_0 = np.full((1), 4)

test_pattern_folder = Path("test_patterns")
np.save(test_pattern_folder / Path("test_input_0"), test_input_0)
np.save(test_pattern_folder / Path("test_target_0"), test_target_0)
