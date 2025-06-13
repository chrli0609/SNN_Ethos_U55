from pathlib import Path
import os
import sys

from ethosu.vela.api import NpuAccelerator

import fc_lif

from extra_func import get_header_filepath, get_connectivity_filepath, process_weights_and_biases, align_input_output_sizes_to_8
from write_connectivity_h_file import clear_connectivity_file, write_tensor_declarations, write_init_func, write_init_func_array


CURR_WORKING_DIR = Path(os.getcwd())
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Only set to true if running check_npu_op_time.ps1
SWEEP_NUM_NEURONS = False

# Set to true to show debug printouts
DEBUG_MODE = False


CURR_WORKING_DIR_TO_MODEL_DIR = Path("../../../snn_on_alif_e7/my_snn_bare_metal/nn_models/")



MODEL_NAME = "nmnist_784x64x64x10"
NUM_LAYERS = 3
LAYER_BASE_NAME = "fc_lif_layer_"

INIT_LAYER_SIZES_LIST = [
    28*28,
    64,
    64,
    10
]

MEM_STORE_LOC_LIST = [
    "model_params_sram1",
    "model_params_sram1",
    "model_params_sram1"
]



'''
Set Accelerator
'''
ACCELERATOR = NpuAccelerator.Ethos_U55_256



'''
Set Test Pattern to use
'''
TEST_PATTERN = '0'







# Write connectivity file
connectivity_filepath = get_connectivity_filepath(MODEL_NAME, CURR_WORKING_DIR, CURR_WORKING_DIR_TO_MODEL_DIR)
clear_connectivity_file(connectivity_filepath)
write_tensor_declarations(connectivity_filepath, LAYER_BASE_NAME, NUM_LAYERS, MEM_STORE_LOC_LIST)

for layer_num in range(NUM_LAYERS):

    # Get layer names
    layer_name = f"{LAYER_BASE_NAME}{layer_num}"

    if layer_num == 0:
        is_first_layer = True
    else:
        is_first_layer = False

    aligned_input_size, aligned_output_size, in_padding, out_padding = align_input_output_sizes_to_8(INIT_LAYER_SIZES_LIST[layer_num], INIT_LAYER_SIZES_LIST[layer_num+1], is_first_layer)


    weights_volume_ohwi, bias_list = process_weights_and_biases(Path("model_params") / Path(layer_name + "_weights.npy"),
                                                                Path("model_params") / Path(layer_name + "_biases.npy"),
                                                                aligned_input_size,
                                                                aligned_output_size,
                                                                in_padding, out_padding)

    ##### Set LIF Param values #######

    # Generate Beta values
    beta_list = []
    for j in range(aligned_output_size):
        beta_list.append(0.95)

    # Generate Vth values
    vth_list = []
    for k in range(aligned_output_size):
        vth_list.append(1)


    #layer_name = f"layer{layer_num}"
    #layer = importlib.import_module(f"layers.{layer_name}")
    header_out_filepath = get_header_filepath(LAYER_BASE_NAME+str(layer_num), MODEL_NAME, CURR_WORKING_DIR, CURR_WORKING_DIR_TO_MODEL_DIR)

    fc_lif.gen_fc_lif(
        INPUT_LAYER_SIZE=aligned_input_size,
        OUTPUT_LAYER_SIZE=aligned_output_size,

        weights_volume_ohwi=weights_volume_ohwi,
        bias_list=bias_list,
        beta_list=beta_list,
        vth_list=vth_list,

        cms_name=LAYER_BASE_NAME+str(layer_num),
        is_last_layer=(layer_num == NUM_LAYERS - 1),

        DEBUG_MODE=DEBUG_MODE,
        ACCELERATOR=ACCELERATOR,
        header_out_filepath=header_out_filepath,
    )
    


    write_init_func(connectivity_filepath, LAYER_BASE_NAME+str(layer_num))

write_init_func_array(connectivity_filepath, LAYER_BASE_NAME, NUM_LAYERS)



    
    






# Generate file for test patterns
from nmnist_write_test_patterns_to_h_file import test_patterns_2_h_file

test_pattern_header_filepath = CURR_WORKING_DIR / CURR_WORKING_DIR_TO_MODEL_DIR / Path(MODEL_NAME) / Path("test_patterns") / Path("pattern_"+TEST_PATTERN+".h")
test_patterns_2_h_file( ".data_sram0",
                        Path("test_patterns/test_input_"+TEST_PATTERN+".npy"),
                        Path("test_patterns/test_target_"+TEST_PATTERN+".npy"),
                        test_pattern_header_filepath
                       )
