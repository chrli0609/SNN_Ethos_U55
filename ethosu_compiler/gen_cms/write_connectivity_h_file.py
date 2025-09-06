


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

        f.write(f'#include "test_patterns/pattern_{test_pattern_nr}.h"\n\n')
        f.write(f"#define NUM_TEST_SAMPLES test_input_{test_pattern_nr}_NUM_SAMPLES\n\n")

        f.write("volatile int8_t* get_test_target() {\n\treturn test_target_"+str(test_pattern_nr)+";\n}\n")
        f.write("volatile int8_t (*get_test_inputs())[MLP_NUM_TIME_STEPS][MLP_INPUT_LAYER_SIZE] {\n\treturn test_input_"+str(test_pattern_nr)+";\n}\n")



def get_func_calls(funcname_first_part, layer_name, content_name, type):

    return funcname_first_part + layer_name + content_name + type + "(),\n"



def write_tensor_declarations(connectivity_h_filepath, base_name, num_layers, mem_store_loc_list):

    with open(connectivity_h_filepath, 'a') as f:
        f.write("\n")
        
        for layer_num in range(num_layers):
            #f.write("\tInit_" + base_name + str(layer_num) + ",\n")
            if layer_num == 0:
                f.write('static int8_t ' + base_name + str(layer_num) + "_in_spk[" + str(base_name).upper() + str(layer_num) + '_INPUT_LAYER_SIZE] __attribute__((section("' +  mem_store_loc_list[layer_num] + '"))) __attribute__((aligned(16)));\n')

            f.write('static int8_t ' + base_name + str(layer_num) + "_tensor_arena[" + str(base_name).upper() + str(layer_num) + '_TENSOR_ARENA_SIZE] __attribute__((section("' +  mem_store_loc_list[layer_num] + '"))) __attribute__((aligned(16)));\n')
            f.write('static int8_t ' + base_name + str(layer_num) + "_out_spk[" + str(base_name).upper() + str(layer_num) + '_OUTPUT_LAYER_SIZE] __attribute__((section("' + mem_store_loc_list[layer_num] + '"))) __attribute__((aligned(16)));\n')

            f.write("\n")
        
        f.write("\n")
        
        #f.write("};")





def write_init_func(connectivity_h_filepath, base_name, layer_num):
    layer_name = base_name + str(layer_num)

    with open(connectivity_h_filepath, 'a') as f:
        f.write("NNLayer* Init_" + layer_name + "() {\n\n")
        f.write("\tNNLayer* " + layer_name + " = FC_LIF_Layer_Init(\n")
        
        #Arguments
        f.write("\t\t" + get_func_calls("Get", layer_name, "CMS", "Pointer"))
        f.write("\t\t" + get_func_calls("Get", layer_name, "CMS", "Len"))
        f.write("\t\t" + get_func_calls("Get", layer_name, "Weights", "Pointer"))
        f.write("\t\t" + get_func_calls("Get", layer_name, "Weights", "Len"))
        f.write("\t\t" + get_func_calls("Get", layer_name, "LIFParam", "Pointer"))
        f.write("\t\t" + get_func_calls("Get", layer_name, "LIFParam", "Len"))
        f.write("\t\t" + get_func_calls("Get", layer_name, "LUT", "Pointer"))
        f.write("\t\t" + get_func_calls("Get", layer_name, "LUT", "Len"))


        f.write("\t\t" + layer_name.upper() + "_IS_LAST_LAYER" + ",\n")
        f.write("\t\t" + layer_name.upper() + "_OUT_SPK_SUM_ADDR" + ",\n")
        f.write("\t\t" + layer_name.upper() + "_OUT_SPK_SCALE" + ",\n")
        f.write("\t\t" + layer_name.upper() + "_OUT_SPK_SUM_ZERO_POINT" + ",\n")

        f.write("\t\t" + layer_name.upper() + "_MAX_NUM_TENSORS_TO_TRACK" + ",\n")

        f.write("\t\t" + layer_name.upper() + "_TENSOR_ARENA_SIZE" + ",\n")

        f.write("\t\t" + layer_name + "_tensor_arena,\n")



        # If is first layer
        if (layer_num == 0):
            f.write("\t\t" + layer_name + "_in_spk,\n")
        else:
            f.write("\t\t" + base_name + str(layer_num-1) + "_out_spk,\n")

        f.write("\t\t" + base_name + str(layer_num) + "_out_spk,\n")


        f.write("\t\t" + layer_name.upper() + "_BIAS_ADDR" + ",\n")
        f.write("\t\t" + layer_name.upper() + "_WEIGHT_ADDR" + ",\n")
        f.write("\t\t" + layer_name.upper() + "_V_MEM_ADDR" + ",\n")
        f.write("\t\t" + layer_name.upper() + "_TIME_NOT_UPDATED_ADDR" + ",\n")
        f.write("\t\t" + layer_name.upper() + "_UPDATE_NXT_LAYER_ADDR" + ",\n")
        
    


        f.write("\t\t" + layer_name.upper() + "_INPUT_LAYER_SIZE" + ",\n")
        f.write("\t\t" + layer_name.upper() + "_OUTPUT_LAYER_SIZE" + ",\n")
        f.write("\t\t" + layer_name.upper() + "_BIAS_LEN" + ",\n")
        f.write("\t\t" + layer_name.upper() + "_WEIGHT_LEN" + ",\n")


        f.write("\t\t" + layer_name.upper() + "_IN_SPK_SCALE" + ",\n")
        f.write("\t\t" + layer_name.upper() + "_IN_SPK_ZERO_POINT" + ",\n")

        f.write("\t\t" + layer_name.upper() + "_V_MEM_SCALE" + ",\n")
        f.write("\t\t" + layer_name.upper() + "_V_MEM_ZERO_POINT" + ",\n")
        f.write("\t\t" + layer_name.upper() + "_TIME_NOT_UPDATED_SCALE" + ",\n")
        f.write("\t\t" + layer_name.upper() + "_TIME_NOT_UPDATED_ZERO_POINT" + ",\n")

        f.write("\t\t" + layer_name.upper() + "_OUT_SPK_SCALE" + ",\n")
        f.write("\t\t" + layer_name.upper() + "_OUT_SPK_ZERO_POIN" + "T\n")
    
        f.write("\t);")
        f.write("\t return " + layer_name + ";\n")

        f.write("}\n\n\n")
            


def write_init_func_array(connectivity_h_filepath, base_name, num_layers):



    with open(connectivity_h_filepath, 'a') as f:
        f.write("\n")
        f.write("NNLayer* (*init_layers_func[MLP_NUM_LAYERS]) (void) = {\n")
        
        for layer_num in range(num_layers):
            f.write("\tInit_" + base_name + str(layer_num) + ",\n")
        
        f.write("};")