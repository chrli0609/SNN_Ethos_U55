
from pathlib import Path
import os
import importlib
import argparse



from extra_func import align_input_output_sizes_to_8


# Parse CLI args
#parser = argparse.ArgumentParser()
#parser.add_argument("--model", required=True, help="Model folder name, e.g. 784x64x64x10 or 784x48x48x48x48x10")
#args = parser.parse_args()

models = ["784x80x10", "784x64x64x10", "784x48x48x48x48x10"]
dcache_flush_time = [2697.177778, 3733.644444, 5628.244444]



for i in range (len(models)):

    # Import config file for the correct model
    #model_module = importlib.import_module(f"{args.model}.config_file")
    model_module = importlib.import_module(f"{models[i]}.config_file")





    tot_tensor_arena_size = 0
    tot_weight_tensor_size = 0
    tot_lif_params_tensor_size = 0
    tot_input_tensor_size = 0
    tot_output_tensor_size = 0


    for layer_num in range(model_module.NUM_LAYERS):

        if layer_num == 0:
            is_first_layer = True
        else:
            is_first_layer = False


        aligned_input_size, aligned_output_size, in_padding, out_padding = align_input_output_sizes_to_8(model_module.INIT_LAYER_SIZES_LIST[layer_num], model_module.INIT_LAYER_SIZES_LIST[layer_num+1], is_first_layer)



        tot_weight_tensor_size += aligned_input_size * aligned_output_size
        tot_tensor_arena_size += 3*aligned_output_size + 16
        tot_input_tensor_size += aligned_input_size
        tot_output_tensor_size += aligned_output_size
        tot_lif_params_tensor_size += 2*aligned_output_size








    print("tot_tensor_arena",tot_tensor_arena_size)
    print("tot_weight_tensor_size",tot_weight_tensor_size)
    print("tot_lif_params_tensor_size",tot_lif_params_tensor_size)
    print("tot_input_tensor_size",tot_input_tensor_size)
    print("tot_output_tensor_size",tot_output_tensor_size)
    print("_______________________")
    tot_const_tensor_size = tot_weight_tensor_size+tot_lif_params_tensor_size
    tot_non_const_tensor_size = tot_tensor_arena_size+tot_input_tensor_size+tot_output_tensor_size
    ratio_non_const = tot_non_const_tensor_size/dcache_flush_time[i]
    ratio_all = (tot_non_const_tensor_size + tot_const_tensor_size) / dcache_flush_time[i]
    print("tot const:", tot_const_tensor_size)
    print("total non const:", tot_non_const_tensor_size)
    print("ratio non const :", ratio_non_const)
    print("ratio all:", ratio_all)
    print("_______________________\n\n\n\n")

