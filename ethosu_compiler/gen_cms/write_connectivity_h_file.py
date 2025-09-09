from typing import List


#from extra_func import get_lif_param_methods_declare_str, get_lut_methods_declare_str
from constants import *
from common import get_arr_dec_str, get_arr_def_str, get_method_def_str, get_arr_name, get_method_name, get_layer_name, get_arr_def_str_dump_string_contents
from common import MethodAccessType


def clear_file_and_write_preamble(connectivity_h_filepath, model_name, base_name, num_layers, num_time_steps, test_pattern_nr):
    with open(connectivity_h_filepath, 'w') as f:
        f.write("#pragma once\n\n")

        f.write('#include "../include/nn_data_structure.h"\n')
        f.write('#include "../model.h"\n\n\n')
        f.write("#define MODEL_NAME \"" + model_name + "\"\n\n\n")
        #f.write("const char* MODEL_NAME = \"" + model_name + "\";\n\n\n")


        for layer_num in range(num_layers):
            f.write("#include \"layers/" + base_name + str(layer_num) + ".h\"\n")
        f.write("\n\n\n")

        
        f.write("#define MLP_INPUT_LAYER_SIZE\t" + base_name.upper() + str(0) + "_INPUT_LAYER_SIZE\n")
        f.write("#define MLP_OUTPUT_LAYER_SIZE\t" + base_name.upper() + str(num_layers-1) + "_OUTPUT_LAYER_SIZE\n\n")

        f.write("#define MLP_NUM_LAYERS " + str(num_layers) + "\n")
        f.write("#define MLP_NUM_TIME_STEPS " + str(num_time_steps) + "\n\n\n")


        # test_pattern
        f.write("// for test patterns\n")
        f.write(f'#include "test_patterns/pattern_{test_pattern_nr}.h"\n\n')
        f.write(f"#define NUM_TEST_SAMPLES test_input_{test_pattern_nr}_NUM_SAMPLES\n\n")

        f.write("volatile int8_t* get_test_target() {\n\treturn test_target_"+str(test_pattern_nr)+";\n}\n")
        f.write("volatile int8_t (*get_test_inputs())[MLP_NUM_TIME_STEPS][MLP_INPUT_LAYER_SIZE] {\n\treturn test_input_"+str(test_pattern_nr)+";\n}\n")






def write_init_func(connectivity_h_filepath, mem_alloc, base_name, layer_num):
    layer_name = get_layer_name(base_name, layer_num)

    with open(connectivity_h_filepath, 'a') as f:
        # Define function
        f.write("\n\nNNLayer* Init_" + layer_name + "() {\n\n")



        ### Write Memory Regions ###
        mem_regions_arr_name_list = []
        mem_regions_size_list = []
        mem_regions_region_number_list = []
        #for region_name, region in mem_alloc.regions.items():
        for region in mem_alloc.get_sorted_mem_regions():

            # Assumption is that only input layer defines its input memory region
            if (region.name == INPUT_REGION_NAME and layer_num != 0):
                mem_regions_arr_name_list.append(get_method_name(OUTPUT_REGION_NAME, get_layer_name(base_name, layer_num-1), MethodAccessType.POINTER) + "()")
                mem_regions_size_list.append(get_method_name(OUTPUT_REGION_NAME, get_layer_name(base_name, layer_num-1), MethodAccessType.LEN) + "()")
                mem_regions_region_number_list.append(region.number)

            else:
                mem_regions_arr_name_list.append(get_method_name(region.name, layer_name, MethodAccessType.POINTER) + "()")
                mem_regions_size_list.append(get_method_name(region.name, layer_name, MethodAccessType.LEN) + "()")
                mem_regions_region_number_list.append(region.number)

            
            #f.write(get_method_def_str(region.dtype, region_name, layer_name, MethodAccessType.POINTER))
            #f.write(get_method_def_str(None, region_name, layer_name, MethodAccessType.LEN))

        
        print("mem_regons_arr_name_list", mem_regions_arr_name_list)
        print("mem_regions_size_list", mem_regions_size_list)
        print("mem_regions_region_number_list", mem_regions_region_number_list)

        f.write(get_arr_def_str("int8_t*", "region_ptrs", layer_name, mem_regions_arr_name_list))
        f.write(get_arr_def_str("size_t", "region_sizes", layer_name, mem_regions_size_list))
        f.write(get_arr_def_str("size_t", "region_numbers", layer_name, mem_regions_region_number_list))



        # Call NNLayer_Init
        f.write("\tNNLayer* " + layer_name + " = NNLayer_Init(\n")
        f.write("\t\t" + str(len(mem_alloc.tensors)) + ",\n")   # Num Tensors
        f.write("\t\t" + str(len(mem_alloc.regions)) + "\n")   # Num Regions
        f.write("\t);\n\n")


        # Call NNLayer_Assign
        f.write("\tNNLayer_Assign(\n")
        
        # Arguments
        f.write("\t\t" + layer_name + ",\n")
        f.write("\t\t" + get_method_name("cms", layer_name, MethodAccessType.POINTER) + "(),\n")
        f.write("\t\t" + get_method_name("cms", layer_name, MethodAccessType.LEN) + "(),\n")

        f.write("\t\t" + get_arr_name("region_ptrs", layer_name) + ",\n")
        f.write("\t\t" + get_arr_name("region_sizes", layer_name) + ",\n")
        f.write("\t\t" + get_arr_name("region_numbers", layer_name) + ",\n")
        f.write("\t\t" + str(len(mem_alloc.regions)) + ",\n")

        f.write('\t\t"' + mem_alloc.input_tensor.name + '\",\n')
        f.write("\t\t" + str(mem_alloc.input_size) + ",\n")
        #out_spk        
        #f.write("\t\t" + get_method_name(OUTPUT_REGION_NAME, layer_name, MethodAccessType.POINTER) + "(),\n")
        f.write('\t\t"' + mem_alloc.output_tensor.name + '",\n')
        f.write("\t\t" + str(mem_alloc.output_size) + ",\n")



        # custom tensors
        f.write("\t\t" + get_method_name("name", layer_name, MethodAccessType.POINTER) + "(),\n")
        f.write("\t\t" + get_method_name("relative_addr", layer_name, MethodAccessType.POINTER) + "(),\n")
        f.write("\t\t" + get_method_name("region", layer_name, MethodAccessType.POINTER) + "(),\n")
        f.write("\t\t" + get_method_name("size", layer_name, MethodAccessType.POINTER) + "(),\n")
        f.write("\t\t" + get_method_name("scale", layer_name, MethodAccessType.POINTER) + "(),\n")
        f.write("\t\t" + get_method_name("zero_point", layer_name, MethodAccessType.POINTER) + "(),\n")
        f.write("\t\t" + str(len(mem_alloc.tensors)) + ",\n")



        f.write("\t\t" + layer_name.upper() + "_IS_LAST_LAYER" + "\n")
    
        f.write("\t);\n")
        f.write("\treturn " + layer_name + ";\n")

        f.write("}\n\n\n")
            


def write_init_func_array(connectivity_h_filepath, base_name, num_layers):



    with open(connectivity_h_filepath, 'a') as f:
        f.write("\n")
        f.write("NNLayer* (*init_layers_func[MLP_NUM_LAYERS]) (void) = {\n")
        
        for layer_num in range(num_layers):
            f.write("\tInit_" + base_name + str(layer_num) + ",\n")
        
        f.write("};")