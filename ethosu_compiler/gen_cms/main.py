from pathlib import Path
import os
import importlib
import argparse


from fc_lif import gen_fc_lif
from extra_func import get_header_filepath, get_connectivity_filepath, process_weights_and_biases, align_input_output_sizes_to_8
from write_connectivity_h_file import clear_file_and_write_preamble, write_init_func, write_init_func_array


CURR_WORKING_DIR = Path(os.getcwd())
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Only set to true if running check_npu_op_time.ps1
SWEEP_NUM_NEURONS = False

# Set to true to show debug printouts
DEBUG_MODE = False

CURR_WORKING_DIR_TO_MODEL_DIR = Path("../../snn_on_alif_e7/my_snn_bare_metal/nn_models/")





# Parse CLI args
parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="Model folder name, e.g. 784x64x64x10 or 784x48x48x48x48x10")
args = parser.parse_args()


# Import config file for the correct model
model_module = importlib.import_module(f"{args.model}.config_file")



# Write connectivity file
connectivity_filepath = get_connectivity_filepath(model_module.MODEL_NAME, CURR_WORKING_DIR, CURR_WORKING_DIR_TO_MODEL_DIR)
connectivity_filepath.parent.mkdir(parents=True, exist_ok=True)


clear_file_and_write_preamble(connectivity_filepath, args.model, model_module.LAYER_BASE_NAME, model_module.NUM_LAYERS, model_module.NUM_TIME_STEPS)
#write_tensor_declarations(connectivity_filepath, model_module.LAYER_BASE_NAME, model_module.NUM_LAYERS, model_module.MEM_STORE_LOC_LIST)


for layer_num in range(model_module.NUM_LAYERS):

    # Get layer names
    layer_name = f"{model_module.LAYER_BASE_NAME}{layer_num}"

    if layer_num == 0:
        is_first_layer = True
    else:
        is_first_layer = False

    aligned_input_size, aligned_output_size, in_padding, out_padding = align_input_output_sizes_to_8(model_module.INIT_LAYER_SIZES_LIST[layer_num], model_module.INIT_LAYER_SIZES_LIST[layer_num+1], is_first_layer)


    weights_volume_ohwi, bias_list = process_weights_and_biases(Path(args.model) / Path("model_params") / Path(layer_name + "_weights.npy"),
                                                                Path(args.model) / Path("model_params") / Path(layer_name + "_biases.npy"),
                                                                aligned_input_size,
                                                                aligned_output_size,
                                                                in_padding, out_padding)

    print("Sizes: Input\t", aligned_input_size, "Output\t", aligned_output_size)
    ##### Set LIF Param values #######

    # Generate Beta values
    beta_list = []
    for j in range(aligned_output_size):
        beta_list.append(model_module.ALL_BETA_VALUE)

    # Generate Vth values
    vth_list = []
    for k in range(aligned_output_size):
        vth_list.append(model_module.ALL_VTH_VALUE)


    header_out_filepath = get_header_filepath(model_module.LAYER_BASE_NAME+str(layer_num), model_module.MODEL_NAME, CURR_WORKING_DIR, CURR_WORKING_DIR_TO_MODEL_DIR)

    # Create layer dir if doesnt already exist
    header_out_filepath.parent.mkdir(parents=True, exist_ok=True)

    gen_fc_lif(
        INPUT_LAYER_SIZE=aligned_input_size,
        OUTPUT_LAYER_SIZE=aligned_output_size,

        weights_volume_ohwi=weights_volume_ohwi,
        bias_list=bias_list,
        beta_list=beta_list,
        vth_list=vth_list,

        #cms_name=model_module.LAYER_BASE_NAME+str(layer_num),
        layer_base_name=model_module.LAYER_BASE_NAME,
        layer_num=layer_num,
        weights_and_biases_on_sram=model_module.WEIGHTS_AND_BIASES_ON_SRAM_LIST[layer_num],
        lif_params_on_sram=model_module.LIF_PARAMS_ON_SRAM_LIST[layer_num],
        is_last_layer=(layer_num == model_module.NUM_LAYERS - 1),
        NUM_TIME_STEPS=model_module.NUM_TIME_STEPS,

        ### For setting quantization params ### 
        #IN_SPK_MAX_VAL=model_module.IN_SPK_MAX_VAL,
        #IN_SPK_MIN_VAL=model_module.IN_SPK_MIN_VAL,
        #Must be symmetric
        #WEIGHT_MAX_VAL=model_module.WEIGHT_MAX_VAL,
        #WEIGHT_MIN_VAL=model_module.WEIGHT_MIN_VAL,
        #LN_BETA_MAX_VAL=model_module.LN_BETA_MAX_VAL,
        #LN_BETA_MIN_VAL=model_module.LN_BETA_MIN_VAL,
        TIME_NOT_UPDATED_MAX_VAL=model_module.TIME_NOT_UPDATED_MAX_VAL,
        TIME_NOT_UPDATED_MIN_VAL=model_module.TIME_NOT_UPDATED_MIN_VAL,
        IN_CURR_MAX_VAL=model_module.IN_CURR_MAX_VAL_LIST[layer_num],
        IN_CURR_MIN_VAL=model_module.IN_CURR_MIN_VAL_LIST[layer_num],
        V_MEM_MAX_VAL=model_module.V_MEM_MAX_VAL_LIST[layer_num],
        V_MEM_MIN_VAL=model_module.V_MEM_MIN_VAL_LIST[layer_num],
        DECAY_ACC_MAX_VAL=model_module.DECAY_ACC_MAX_VAL,
        DECAY_ACC_MIN_VAL=model_module.DECAY_ACC_MIN_VAL,
        DECAY_MAX_VAL=model_module.DECAY_MAX_VAL,
        DECAY_MIN_VAL=model_module.DECAY_MIN_VAL,
        DECAYED_MEM_MAX_VAL=model_module.DECAYED_MEM_MAX_VAL_LIST[layer_num],
        DECAYED_MEM_MIN_VAL=model_module.DECAYED_MEM_MIN_VAL_LIST[layer_num],
        #VTH_MAX_VAL=model_module.VTH_MAX_VAL,
        #VTH_MIN_VAL=model_module.VTH_MIN_VAL,
        #V_MEM_SUB_VTH_MAX_VAL=model_module.V_MEM_SUB_VTH_MAX_VAL,
        #V_MEM_SUB_VTH_MIN_VAL=model_module.V_MEM_SUB_VTH_MIN_VAL,
        #OUT_SPK_MAX_VAL=model_module.OUT_SPK_MAX_VAL,
        #OUT_SPK_MIN_VAL=model_module.OUT_SPK_MIN_VAL,

        DEBUG_MODE=DEBUG_MODE,
        ACCELERATOR=model_module.ACCELERATOR,
        mem_store_loc=model_module.MEM_STORE_LOC_LIST[layer_num],
        header_out_filepath=header_out_filepath,
        connectivity_filepath=connectivity_filepath
    )
    


    #write_init_func(connectivity_filepath, model_module.LAYER_BASE_NAME, layer_num)



write_init_func_array(connectivity_filepath, model_module.LAYER_BASE_NAME, model_module.NUM_LAYERS)



    
    




print("about to write test_patterns_2_h_file from nmnist_write...")

# Generate file for test patterns
from nmnist_write_test_patterns_to_h_file import test_patterns_2_h_file

test_pattern_header_filepath = CURR_WORKING_DIR / CURR_WORKING_DIR_TO_MODEL_DIR / Path(model_module.MODEL_NAME) / Path("test_patterns") / Path("pattern_"+str(model_module.TEST_PATTERN_NUM)+".h")

# Create dir of doesnt already exist
test_pattern_header_filepath.parent.mkdir(parents=True, exist_ok=True)

test_patterns_2_h_file(45,  ".data_sram0",
                        (Path(args.model) / Path("test_patterns/test_input_"+str(model_module.TEST_PATTERN_NUM)+".npy")),
                        (Path(args.model) / Path("test_patterns/test_target_"+str(model_module.TEST_PATTERN_NUM)+".npy")),
                        test_pattern_header_filepath
                       )





