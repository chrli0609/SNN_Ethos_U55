from typing import List


#from extra_func import get_lif_param_methods_declare_str, get_lut_methods_declare_str
from constants import *
from common import get_arr_dec_str, get_arr_def_str, get_method_def_str, get_arr_name, get_method_name, get_layer_name, get_arr_def_str_dump_string_contents
from common import MethodAccessType


def clear_file_and_write_preamble(connectivity_h_filepath, model_name, base_name, num_layers, num_time_steps):
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
        f.write("#define MLP_NUM_TIME_STEPS " + str(num_time_steps) + "\n")


#def get_func_calls(funcname_first_part, layer_name, content_name, type):

    #return funcname_first_part + layer_name + content_name + type + "(),\n"

#def get_array_definition(arr_name: str, arr_len: int, arr_value_type: str, arr_values_list: List[str]) -> str:
    #ret_str = f"{arr_value_type} {arr_name} [{arr_len}] = {{\n"

    #for element in arr_values_list:
        #ret_str += f"\t{element}\n"
    #ret_str += "}}\n"

    #return  ret_str


#def write_tensor_declarations(layer_file_filepath, mem_alloc, base_name, layer_num, mem_store_loc):

    #with open(layer_file_filepath, 'a') as f:
        #f.write("\n")
        

        #layer_name = base_name + str(layer_num)
        ##f.write("\tInit_" + base_name + str(layer_num) + ",\n")
        #if layer_num == 0:
            ##f.write('static int8_t ' + base_name + str(layer_num) + "_in_spk[" + str(base_name).upper() + str(layer_num) + '_INPUT_LAYER_SIZE] __attribute__((section("' +  mem_store_loc_list[layer_num] + '"))) __attribute__((aligned(16)));\n')
            #f.write(get_arr_dec_str("static int8_t", "in_spk", layer_name, layer_name.upper()+"_INPUT_LAYER_SIZE", [f'section("{mem_store_loc}")', f"aligned(16)"]))
            #f.write(get_method_def_str("const int8_t", "in_spk", layer_name, MethodAccessType.POINTER))
            #f.write(get_method_def_str(None, "in_spk", layer_name, MethodAccessType.LEN))

        ##f.write('static int8_t ' + base_name + str(layer_num) + "_tensor_arena[" + str(base_name).upper() + str(layer_num) + '_TENSOR_ARENA_SIZE] __attribute__((section("' +  mem_store_loc_list[layer_num] + '"))) __attribute__((aligned(16)));\n')
        ##f.write('static int8_t ' + base_name + str(layer_num) + "_out_spk[" + str(base_name).upper() + str(layer_num) + '_OUTPUT_LAYER_SIZE] __attribute__((section("' + mem_store_loc_list[layer_num] + '"))) __attribute__((aligned(16)));\n')
        ##f.write(get_arr_dec_str("static int8_t", "tensor_arena", layer_name, layer_name.upper()+"_TENSOR_ARENA_SIZE", [f'section("{mem_store_loc_list[layer_num]}")', f"aligned(16)"]))
        ##f.write(get_method_def_str("const int8_t*", "tensor_arena", layer_name, MethodAccessType.POINTER))
        #f.write(get_arr_dec_str("static int8_t", "out_spk", layer_name, layer_name.upper()+"_OUTPUT_LAYER_SIZE", [f'section("{mem_store_loc}")', f"aligned(16)"]))
        #f.write(get_method_def_str("const int8_t", "out_spk", layer_name, MethodAccessType.POINTER))
        #f.write(get_method_def_str(None, "out_spk", layer_name, MethodAccessType.LEN))

        #for region_name, region in mem_alloc.regions.items():
            #if mem_alloc.is_custom_region(region_name) and not region.is_on_mram and region_name != "INPUT_REGION" and region_name != "OUTPUT_REGION":
                #f.write(get_arr_dec_str("static int8_t", region_name, layer_name, layer_name.upper()+"_OUTPUT_LAYER_SIZE", [f'section("{mem_store_loc}")', f"aligned(16)"]))
                #f.write(get_method_def_str("const int8_t", region_name, layer_name, MethodAccessType.POINTER))
                #f.write(get_method_def_str(None, region_name, layer_name, MethodAccessType.LEN))
                

        #f.write("\n")
        
        #f.write("\n")
        
        ##f.write("};")





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

        #f.write("\t\t" + get_method_name("region_ptrs", layer_name, MethodAccessType.POINTER) + "(),\n")
        #f.write("\t\t" + get_method_name("region_sizes", layer_name, MethodAccessType.POINTER) + "(),\n")
        #f.write("\t\t" + get_method_name("region_numbers", layer_name, MethodAccessType.POINTER) + "(),\n")
        f.write("\t\t" + get_arr_name("region_ptrs", layer_name) + ",\n")
        f.write("\t\t" + get_arr_name("region_sizes", layer_name) + ",\n")
        f.write("\t\t" + get_arr_name("region_numbers", layer_name) + ",\n")
        f.write("\t\t" + str(len(mem_alloc.regions)) + ",\n")

        # in_spk
        #if layer_num == 0:
            ##f.write("\t\t" + get_method_name("in_spk", layer_name, MethodAccessType.POINTER) + "(),\n")
            #f.write("\t\t" + get_method_name(INPUT_REGION_NAME, layer_name, MethodAccessType.POINTER) + "(),\n")
        #else:
            ##f.write("\t\t" + get_method_name("out_spk", get_layer_name(base_name, layer_num-1), MethodAccessType.POINTER) + "(),\n")
            #f.write("\t\t" + get_method_name(OUTPUT_REGION_NAME, get_layer_name(base_name, layer_num-1), MethodAccessType.POINTER) + "(),\n")
        f.write('\t\t"' + mem_alloc.input_tensor.name + '\",\n')
        f.write("\t\t" + str(mem_alloc.input_size) + ",\n")
        #out_spk        
        #f.write("\t\t" + get_method_name(OUTPUT_REGION_NAME, layer_name, MethodAccessType.POINTER) + "(),\n")
        f.write('\t\t"' + mem_alloc.output_tensor.name + '",\n')
        f.write("\t\t" + str(mem_alloc.output_size) + ",\n")


        ##v_mem sizes
        #f.write("\t\t" + f"{layer_name.upper()}_OUTPUT_LAYER_SIZE,\n")
        #f.write("\t\t1,\n")
        #f.write("\t\t1,\n")

        # custom tensors
        f.write("\t\t" + get_method_name("name", layer_name, MethodAccessType.POINTER) + "(),\n")
        f.write("\t\t" + get_method_name("relative_addr", layer_name, MethodAccessType.POINTER) + "(),\n")
        f.write("\t\t" + get_method_name("region", layer_name, MethodAccessType.POINTER) + "(),\n")
        f.write("\t\t" + get_method_name("size", layer_name, MethodAccessType.POINTER) + "(),\n")
        f.write("\t\t" + get_method_name("scale", layer_name, MethodAccessType.POINTER) + "(),\n")
        f.write("\t\t" + get_method_name("zero_point", layer_name, MethodAccessType.POINTER) + "(),\n")
        f.write("\t\t" + str(len(mem_alloc.tensors)) + ",\n")

        ## quant params
        #f.write("\t\t" + layer_name.upper() + "_IN_SPK_SCALE" + ",\n")
        #f.write("\t\t" + layer_name.upper() + "_IN_SPK_ZERO_POINT" + ",\n")

        #f.write("\t\t" + layer_name.upper() + "_V_MEM_SCALE" + ",\n")
        #f.write("\t\t" + layer_name.upper() + "_V_MEM_ZERO_POINT" + ",\n")
        #f.write("\t\t" + layer_name.upper() + "_TIME_NOT_UPDATED_SCALE" + ",\n")
        #f.write("\t\t" + layer_name.upper() + "_TIME_NOT_UPDATED_ZERO_POINT" + ",\n")

        #f.write("\t\t" + layer_name.upper() + "_OUT_SPK_SCALE" + ",\n")
        #f.write("\t\t" + layer_name.upper() + "_OUT_SPK_ZERO_POINT" + "\n")


        f.write("\t\t" + layer_name.upper() + "_IS_LAST_LAYER" + "\n")

        


        #Arguments (old)
        #f.write("\t\t" + get_func_calls("Get", layer_name, "CMS", "Pointer"))
        #f.write("\t\t" + get_func_calls("Get", layer_name, "CMS", "Len"))
        #f.write("\t\t" + get_func_calls("Get", layer_name, "Weights", "Pointer"))
        #f.write("\t\t" + get_func_calls("Get", layer_name, "Weights", "Len"))


        #f.write("\t\t" + layer_name.upper() + "_IS_LAST_LAYER" + ",\n")
        #f.write("\t\t" + layer_name.upper() + "_OUT_SPK_SUM_ADDR" + ",\n")
        #f.write("\t\t" + layer_name.upper() + "_OUT_SPK_SCALE" + ",\n")
        #f.write("\t\t" + layer_name.upper() + "_OUT_SPK_SUM_ZERO_POINT" + ",\n")

        #f.write("\t\t" + layer_name.upper() + "_MAX_NUM_TENSORS_TO_TRACK" + ",\n")

        #f.write("\t\t" + layer_name.upper() + "_TENSOR_ARENA_SIZE" + ",\n")

        #f.write("\t\t" + layer_name + "_tensor_arena,\n")



        ## If is first layer
        #if (layer_num == 0):
            #f.write("\t\t" + layer_name + "_in_spk,\n")
        #else:
            #f.write("\t\t" + base_name + str(layer_num-1) + "_out_spk,\n")

        #f.write("\t\t" + base_name + str(layer_num) + "_out_spk,\n")


        #f.write("\t\t" + layer_name.upper() + "_BIAS_ADDR" + ",\n")
        #f.write("\t\t" + layer_name.upper() + "_WEIGHT_ADDR" + ",\n")
        #f.write("\t\t" + layer_name.upper() + "_V_MEM_ADDR" + ",\n")
        #f.write("\t\t" + layer_name.upper() + "_TIME_NOT_UPDATED_ADDR" + ",\n")
        #f.write("\t\t" + layer_name.upper() + "_UPDATE_NXT_LAYER_ADDR" + ",\n")
        
    


        #f.write("\t\t" + layer_name.upper() + "_INPUT_LAYER_SIZE" + ",\n")
        #f.write("\t\t" + layer_name.upper() + "_OUTPUT_LAYER_SIZE" + ",\n")
        #f.write("\t\t" + layer_name.upper() + "_BIAS_LEN" + ",\n")
        #f.write("\t\t" + layer_name.upper() + "_WEIGHT_LEN" + ",\n")


        #f.write("\t\t" + layer_name.upper() + "_IN_SPK_SCALE" + ",\n")
        #f.write("\t\t" + layer_name.upper() + "_IN_SPK_ZERO_POINT" + ",\n")

        #f.write("\t\t" + layer_name.upper() + "_V_MEM_SCALE" + ",\n")
        #f.write("\t\t" + layer_name.upper() + "_V_MEM_ZERO_POINT" + ",\n")
        #f.write("\t\t" + layer_name.upper() + "_TIME_NOT_UPDATED_SCALE" + ",\n")
        #f.write("\t\t" + layer_name.upper() + "_TIME_NOT_UPDATED_ZERO_POINT" + ",\n")

        #f.write("\t\t" + layer_name.upper() + "_OUT_SPK_SCALE" + ",\n")
        #f.write("\t\t" + layer_name.upper() + "_OUT_SPK_ZERO_POIN" + "T\n")
    
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