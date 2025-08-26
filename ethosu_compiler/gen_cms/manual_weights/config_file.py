from ethosu.vela.api import NpuAccelerator



MODEL_NAME = "manual_weights"
LAYER_BASE_NAME = "fc_lif_layer_"

# For easy debugging
NUM_TIME_STEPS = 2
NUM_LAYERS = 2

ALL_BETA_VALUE = 0.9
ALL_VTH_VALUE = 1



INIT_LAYER_SIZES_LIST = [
    16,
    16,
    10,
]

MEM_STORE_LOC_LIST = [
    "model_params_sram1",
    "model_params_sram1",
]


WEIGHTS_AND_BIASES_ON_SRAM_LIST = [
    False,
    False,
]


LIF_PARAMS_ON_SRAM_LIST = [
    False,
    False
]

'''
Set Accelerator
'''
ACCELERATOR = NpuAccelerator.Ethos_U55_256



'''
Set Test Pattern to use
'''
TEST_PATTERN_NUM = 0





'''
For setting quantization params
'''


TIME_NOT_UPDATED_MAX_VAL = 16
TIME_NOT_UPDATED_MIN_VAL = 0

#IN_CURR_MAX_VAL = 9
#IN_CURR_MIN_VAL = -9

#V_MEM_MAX_VAL = 9
#V_MEM_MIN_VAL = -6

DECAY_ACC_MAX_VAL = 0
DECAY_ACC_MIN_VAL = -1
DECAY_MAX_VAL = 0.95
DECAY_MIN_VAL = 0

#DECAYED_MEM_MAX_VAL = 7
#DECAYED_MEM_MIN_VAL = -4

IN_CURR_MAX_VAL_LIST = [6, 6]
IN_CURR_MIN_VAL_LIST = [-6, -6]

DECAYED_MEM_MAX_VAL_LIST = [6, 6]
DECAYED_MEM_MIN_VAL_LIST = [-4, -4]

V_MEM_MAX_VAL_LIST = [6, 6]
V_MEM_MIN_VAL_LIST = [-6, -6]