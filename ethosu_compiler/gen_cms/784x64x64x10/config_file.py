from ethosu.vela.api import NpuAccelerator



MODEL_NAME = "784x64x64x10"
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
    "model_params_dtcm",
    "model_params_dtcm",
    "model_params_dtcm"
]

WEIGHTS_AND_BIASES_ON_SRAM_LIST = [
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

