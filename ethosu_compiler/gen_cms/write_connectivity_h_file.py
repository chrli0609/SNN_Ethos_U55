


def clear_connectivity_file(connectivity_h_filepath):
    with open(connectivity_h_filepath, 'w') as f:
        f.write("")
        f.write("")
        f.write('#include "../../include/nn_data_structure.h"\n')
        f.write('#include "model.h"\n\n\n')


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





def write_init_func(connectivity_h_filepath, layer_name):


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

        f.write("\t\t" + layer_name.upper() + "_NUM_NON_CONST_TENSORS" + ",\n")

        f.write("\t\t" + layer_name.upper() + "_TENSOR_ARENA_SIZE" + ",\n")

        f.write("\t\t" + layer_name + "_tensor_arena,\n")

        try:
            layer_num = int(layer_name[-1])
        except:
            print("Layer CMS name should end with layer number")
            exit(1)


        # If is first layer
        if (layer_num == 0):
            f.write("\t\t" + layer_name + "_in_spk,\n")
        else:
            f.write("\t\t" + layer_name[:-1] + str(layer_num-1) + "_out_spk,\n")

        f.write("\t\t" + layer_name + "_out_spk,\n")


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