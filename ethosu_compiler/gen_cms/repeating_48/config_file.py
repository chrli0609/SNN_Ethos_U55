from ethosu.vela.api import NpuAccelerator



#MODEL_NAME = "784x48x48x48x48x10"
LAYER_BASE_NAME = "fc_lif_layer_"

NUM_TIME_STEPS = 25
#NUM_LAYERS = 5

ALL_BETA_VALUE = 0.95
ALL_VTH_VALUE = 1


HIDDEN_SIZES = 48

MEM_STORE_LOC = "model_params_sram1"
#MEM_STORE_LOC = "model_params_sram0"



WEIGHTS_AND_BIASES_ON_SRAM = False

LIF_PARAMS_ON_SRAM = False

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

IN_CURR_MAX_VAL = 6
V_MEM_MAX_VAL = 9
DECAYED_MEM_MAX_VAL = 7

IN_CURR_MIN_VAL = -6
V_MEM_MIN_VAL = -6
DECAYED_MEM_MIN_VAL = -4


DECAY_ACC_MAX_VAL = 0
DECAY_ACC_MIN_VAL = -1

DECAY_MAX_VAL = 0.95
DECAY_MIN_VAL = 0

