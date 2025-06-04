from pathlib import Path
import os
import sys
CURR_WORKING_DIR = Path(os.getcwd())
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from ethosu.vela.api import NpuAccelerator




# Only set to true if running check_npu_op_time.ps1
SWEEP_NUM_NEURONS = False

# Set to true to show debug printouts
DEBUG_MODE = False


CURR_WORKING_DIR_TO_MODEL_DIR = Path("../../../snn_on_alif_e7/my_snn_bare_metal/nn_models/")



'''
Command line argument: LAYER_0_OUTPUT_SIZE
'''

MODEL_NAME = "spk_mnist_mlp"
LAYER_0_CMS_NAME = "fc_lif_layer_0"
LAYER_1_CMS_NAME = "fc_lif_layer_1"


'''
Set Accelerator
'''
ACCELERATOR = NpuAccelerator.Ethos_U55_256



'''
Set Test Pattern to use
'''
TEST_PATTERN = '0'


import fc_lif
import layer0
import layer1

if SWEEP_NUM_NEURONS:
    if len(sys.argv) > 1:
            try:
                layer0.OUTPUT_LAYER_SIZE = int(sys.argv[1])
            except:
                print("Expected Integer command line argument but received:", sys.argv[1])
                exit()
    else:
        print("AN ERROR HAS OCCURRRED, INCORRECT COMMAND LINE ARGUMENTS SET WHEN CALLING main.py")







# Assign header filenames
from extra_func import get_header_filepath
header_out_filepath_layer0 = get_header_filepath(LAYER_0_CMS_NAME, MODEL_NAME, CURR_WORKING_DIR, CURR_WORKING_DIR_TO_MODEL_DIR)
header_out_filepath_layer1 = get_header_filepath(LAYER_1_CMS_NAME, MODEL_NAME, CURR_WORKING_DIR, CURR_WORKING_DIR_TO_MODEL_DIR)


# Ensure Layer Input/Output Dimensions match
if not SWEEP_NUM_NEURONS:
    if (layer0.OUTPUT_LAYER_SIZE != layer1.INPUT_LAYER_SIZE):
        print("Error: Layer Dimensions don't match")
    
    



fc_lif.gen_fc_lif(
    INPUT_LAYER_SIZE=layer0.INPUT_LAYER_SIZE,
    OUTPUT_LAYER_SIZE=layer0.OUTPUT_LAYER_SIZE,


    weights_volume_ohwi=layer0.weights_volume_ohwi,
    bias_list=layer0.bias_list,
    beta_list=layer0.beta_list,
    vth_list=layer0.vth_list,    

    cms_name=LAYER_0_CMS_NAME,
    DEBUG_MODE=DEBUG_MODE,
    ACCELERATOR=ACCELERATOR,
    header_out_filepath=header_out_filepath_layer0
)

fc_lif.gen_fc_lif(
    INPUT_LAYER_SIZE=layer1.INPUT_LAYER_SIZE,
    OUTPUT_LAYER_SIZE=layer1.OUTPUT_LAYER_SIZE,


    weights_volume_ohwi=layer1.weights_volume_ohwi,
    bias_list=layer1.bias_list,
    beta_list=layer1.beta_list,
    vth_list=layer1.vth_list,    

    cms_name=LAYER_1_CMS_NAME,
    DEBUG_MODE=DEBUG_MODE,
    ACCELERATOR=ACCELERATOR,
    header_out_filepath=header_out_filepath_layer1
)




# Generate file for test patterns
from write_test_patterns_to_h_file import test_patterns_2_h_file

test_pattern_header_filepath = CURR_WORKING_DIR / CURR_WORKING_DIR_TO_MODEL_DIR / Path(MODEL_NAME) / Path("test_patterns") / Path("pattern_"+TEST_PATTERN+".h")
test_patterns_2_h_file( ".data_sram0",
                        Path("test_patterns/test_input_"+TEST_PATTERN+".npy"),
                        Path("test_patterns/test_target_"+TEST_PATTERN+".npy"),
                        test_pattern_header_filepath
                       )




