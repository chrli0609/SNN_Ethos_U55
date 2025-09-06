from ethosu.vela.api import NpuAccelerator



MODEL_NAME = "784x64x64x10_ver_2"
LAYER_BASE_NAME = "fc_lif_layer_"

NUM_TIME_STEPS = 25
NUM_LAYERS = 3

ALL_BETA_VALUE = 0.95
ALL_VTH_VALUE = 1



INIT_LAYER_SIZES_LIST = [
    28*28,
    64,
    64,
    10
]

MEM_STORE_LOC_LIST = [
    #"model_params_sram1",
    #"model_params_sram1",
    #"model_params_sram1"
    "model_params_dtcm",
    "model_params_dtcm",
    "model_params_dtcm"
]

WEIGHTS_AND_BIASES_ON_SRAM_LIST = [
    False,
    False,
    False
    #True,
    #True,
    #True
]


LIF_PARAMS_ON_SRAM_LIST = [
    False,
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
#TEST_PATTERN_NUM = 4





'''
For setting quantization params
'''


TIME_NOT_UPDATED_MAX_VAL = 16
TIME_NOT_UPDATED_MIN_VAL = 0

IN_CURR_MAX_VAL = 9
IN_CURR_MIN_VAL = -9

V_MEM_MAX_VAL = 9
V_MEM_MIN_VAL = -6

DECAY_ACC_MAX_VAL = 0
DECAY_ACC_MIN_VAL = -1
DECAY_MAX_VAL = 0.95
DECAY_MIN_VAL = 0

DECAYED_MEM_MAX_VAL = 7
DECAYED_MEM_MIN_VAL = -4





IN_CURR_MAX_VAL_LIST = [IN_CURR_MAX_VAL, IN_CURR_MAX_VAL, IN_CURR_MAX_VAL]
IN_CURR_MIN_VAL_LIST = [IN_CURR_MIN_VAL, IN_CURR_MIN_VAL, IN_CURR_MIN_VAL]
DECAYED_MEM_MAX_VAL_LIST = [DECAYED_MEM_MAX_VAL, DECAYED_MEM_MAX_VAL, DECAYED_MEM_MAX_VAL]
DECAYED_MEM_MIN_VAL_LIST = [DECAYED_MEM_MIN_VAL, DECAYED_MEM_MIN_VAL, DECAYED_MEM_MIN_VAL]
V_MEM_MAX_VAL_LIST = [V_MEM_MAX_VAL, V_MEM_MAX_VAL, V_MEM_MAX_VAL]
V_MEM_MIN_VAL_LIST = [V_MEM_MIN_VAL, V_MEM_MIN_VAL, V_MEM_MIN_VAL]




