from ethosu.vela.api import *

from cms_interpreter import register_cms_2_assembly


from pathlib import Path




def align_input_output_sizes_to_16(init_input_size, init_output_size, is_first_layer):

    '''
    Output Feature map constraints:
        - OFM_BLOCK_WIDTH:
            - Multiple of MIN_BLOCK_WIDTH
            - 1 < OFM_BLOCK_WIDTH < 64
        - OFM_BLOCK_HEIGHT:
            - Multiple of MIN_BLOCK_HEIGHT
            - 1 < OFM_BLOCK_HEIGHT < 32
        - OFM_BLOCK_DEPTH:
            - Multiple of MIN_BLOCK_DEPTH
            - 1 < OFM_BLOCK_DEPTH < 128
            - If OFM_BLOCK_DEPTH not multiple of 16: OFM_DEPTH <= OFM_BLOCK_DEPTH

            
    For Ethos U55 with configuration: 256:
        - MIN_BLOCK_WIDTH = 2
        - MIN_BLOCK_HEIGHT = 2
        - MIN_BLOCK_DEPTH = 8
    '''

    

    if is_first_layer:
        aligned_input_size = init_input_size
    else:
        #aligned_input_size = next_multiple_of_8(init_input_size)
        aligned_input_size = next_multiple(init_input_size, 16)

    #aligned_output_size = next_multiple_of_8(init_output_size)
    aligned_output_size = next_multiple(init_output_size, 16)

    in_padding = aligned_input_size - init_input_size
    out_padding = aligned_output_size - init_output_size


    return aligned_input_size, aligned_output_size, in_padding, out_padding


import numpy as np

def process_weights_and_biases(weights_filepath, biases_filepath, input_layer_size, output_layer_size, in_padding, out_padding):
    ## Get weights and biases
    weights_init = np.load(weights_filepath)
    bias_init = np.load(biases_filepath)

    # Append by how much we are missing
    weights_padded = np.pad(weights_init, ((0, out_padding), (0, in_padding)), mode='constant')
    bias_padded = np.pad(bias_init, (0, out_padding), mode='constant')


    # Reshape weights
    weights_reshaped = weights_padded.reshape(output_layer_size, 1, 1, input_layer_size)


    weights_volume_ohwi = weights_reshaped
    bias_list = bias_padded

    print("Max weight value:", weights_volume_ohwi.max())
    print("Min weight value", weights_volume_ohwi.min())
    print("Max bias value", bias_list.max())
    print("Min bias value", bias_list.min())

    return weights_volume_ohwi, bias_list



def get_connectivity_filepath(model_name, current_working_directory, current_to_model_directory):
    connectivity_filepath = current_working_directory / current_to_model_directory / Path(model_name) / Path("connectivity.h")

    return connectivity_filepath


'''
Header filepath: current_working_directory / current_to_model_directory / model_name / layers / layer_name.h
'''
def get_header_filepath(layer_name, model_name, current_working_directory, current_to_model_directory):

    header_out_filepath = current_working_directory / current_to_model_directory / Path(model_name) / Path("layers") / Path(layer_name+ ".h")
    return header_out_filepath


def next_multiple(n, m=16):
    return n + (-n % m)

# For finding the next integer that is divisible by 8, since it is a requirement of the NPU that  the output size of each layer must be divisible by 8
def next_multiple_of_8(x: int) -> int:
    return ((x + 7) // 8) * 8

from config_ops import create_feature_map, gen_weights_and_biases
def get_int8_fc_weights_and_biases(
        weights_volume_ohwi, 
        bias_list,
        input_size,
        output_size,

        weight_scale, weight_zero_point,
        ifm_scale, ofm_scale,

        accelerator,
        debug_mode
    ):

    UNSET = 0

    ifm = create_feature_map(
        height=1, width=1, depth=input_size,
        region = UNSET,
        layout=NpuLayout.NHWC,
        data_type=NpuDataType.INT8,
        fm_elem_size=1,
        fm_addr=UNSET,
        scale=ifm_scale,
        zero_point=UNSET,
    )



    ifm2 = None


    ofm = create_feature_map(
        height=1, width=1, depth=output_size,
        region=UNSET,
        #layout=NpuLayout.NHCWB16,
        layout=NpuLayout.NHWC,
        data_type=NpuDataType.INT8,
        fm_elem_size=1,
        fm_addr=UNSET,
        scale = ofm_scale,
        zero_point = UNSET
    )



    # Kernel
    kernel = NpuKernel(
        w=1, h=1, 
        stride_x=1, stride_y=1, dilation_x=1, dilation_y=1
    )

    my_op = NpuConv2DOperation()
    my_op.ifm               =   ifm
    my_op.ifm2              =   None
    my_op.ofm               =   ofm
    my_op.kernel            =   kernel
    block_config = get_block_config(my_op, accelerator)



    block_traversal = NpuBlockTraversal.DEPTH_FIRST


    if ifm.data_type == NpuDataType.INT8:
        weight_ifm_bitdepth = 8 #int8
    elif ifm.data_type == NpuDataType.INT16:
        weight_ifm_bitdepth = 16 #int16

    
    weight_byte_arr, bias_byte_arr = gen_weights_and_biases(
                            accelerator=accelerator,
                            weights_volume_ohwi=weights_volume_ohwi,
                            dilation_xy=(1,1),
                            ifm_bitdepth=weight_ifm_bitdepth,
                            ofm_block_depth=block_config[2],
                            op_type=NpuOperationType.Conv2D,
                            block_traversal=block_traversal,

                            #ONLY FOR 1 DIM FMs!!!!
                            bias_list=bias_list,

                            ifm_scale=ifm.quantization.scale_f32,
                            weight_scale=weight_scale,
                            weight_zero_point=weight_zero_point,    # should always be zero
                            ofm_scale=ofm.quantization.scale_f32,


                            is_debug_mode=debug_mode
    )

    return weight_byte_arr, bias_byte_arr


def check_block_config_legal(block_config, my_op, accelerator):
    # Check that block config is legal
    available_block_configs = npu_find_block_configs(my_op, accelerator)

    block_config_legal = False
    for i in range(len(available_block_configs)):
        if block_config == available_block_configs[i]:
            block_config_legal = True

    if not block_config_legal:
        print("ERROR: Illegal BLOCK_CONFIG found for", my_op, "\n\t", block_config, "\n\n\t But expected one of the following:\n\t", available_block_configs,
              "\n IFM:", my_op.ifm.name,
              "\n OFM:", my_op.ofm.name)
        exit()
    #else:
        #print("//Current Block Config is legal:", block_config)



def get_block_config(my_op, accelerator):
    available_block_configs = npu_find_block_configs(my_op, accelerator)
    print("available_blk_configs:", available_block_configs)
    return available_block_configs[-1]
    #return available_block_configs[0]


def float_to_int_safe(x: float) -> int:
    if not x.is_integer():
        raise ValueError(f"Cannot convert {x} to int: it has nonzero decimals.")
    return int(x)


def get_includes_str():
    return "#include <stddef.h>\n#include <stdint.h>\n\n\n\n\n"




def get_lif_param_methods_definition_str(base_name):
    return "\n\n\n\n\nstatic inline const int8_t* Get" + base_name + "LIFParamPointer()\n{\n\treturn lif_param_" + base_name + ";\n}\nstatic inline size_t Get" + base_name + "LIFParamLen()\n{\n\treturn sizeof(lif_param_" + base_name + ");\n}\n\n"

def get_lut_methods_definition_str(base_name):
    return "\n\n\n\n\nstatic inline const int8_t* Get" + base_name + "LUTPointer()\n{\n\treturn lut_" + base_name + ";\n}\nstatic inline size_t Get" + base_name + "LUTLen()\n{\n\treturn sizeof(lut_" + base_name + ");\n}\n\n"

def get_cms_methods_definition_str(base_name):
    return "\n\n\n\n\nstatic inline const uint8_t * Get" + base_name + "CMSPointer()\n{\n\treturn cms_" + base_name + ";\n}\n\nstatic inline size_t Get"+ base_name +"CMSLen()\n{\n\treturn sizeof(cms_" + base_name + ");\n}\n\n"

def get_weight_methods_definition_str(base_name):
    return "\n\n\n\nstatic inline const int8_t * Get" + base_name + "WeightsPointer()\n{\n\treturn weight_" + base_name + ";\n}\n\nstatic inline size_t Get"+ base_name +"WeightsLen()\n{\n\treturn sizeof(weight_" + base_name + ");\n}\n\n"



# Dont need method declarations, can delete if needed
def get_lif_param_methods_declare_str(base_name):
    return "\n\n\n\n\nstatic inline const int8_t* Get" + base_name + "LIFParamPointer();\n\n\nsize_t Get" + base_name + "LIFParamLen();\n\n\n"

def get_lut_methods_declare_str(base_name):
    return "\n\n\n\n\nstatic inline const int8_t* Get" + base_name + "LUTPointer();\n\n\nsize_t Get" + base_name + "LUTLen();\n\n\n"

def get_cms_methods_declare_str(base_name):
    return "\n\n\n\n\nstatic inline const uint8_t * Get" + base_name + "CMSPointer();\n\n\nsize_t Get"+ base_name +"CMSLen();\n\n\n"

def get_weight_methods_declare_str(base_name):
    return "\n\n\n\nstatic inline const int8_t * Get" + base_name + "WeightsPointer();\n\nsize_t Get"+ base_name +"WeightsLen();\n\n\n"




def get_cms_arr_def_str(base_name):
    return "\n\n\n\nstatic const uint8_t cms_" + base_name + "[] __attribute__((aligned(16))) = \n{\n"

def get_weights_arr_def_str(base_name):
    return "\n\n\n\nstatic const int8_t weight_" + base_name + "[] __attribute__((aligned(16))) = \n{\n"

def get_lut_arr_def_str(base_name):
    return "\n\n\n\nstatic const int8_t lut_" + base_name + "[] __attribute__((aligned(16))) = \n{\n" 

def get_lif_param_arr_def_str(base_name, lif_params_on_sram):
    ret_str = "\n\n\n\nstatic "

    if (not lif_params_on_sram):
        ret_str += "const "

    ret_str += "int8_t lif_param_" + base_name + "[] "

    if (lif_params_on_sram):
        ret_str += "__attribute__((section(\"model_params_sram1\")))"

    ret_str += " __attribute__((aligned(16))) = \n{\n" 


    return ret_str



    


def get_macro_def_str(my_dict):

    
    
    ret_str = "\n"
    # Skip first because first op is DMA
    for key, value in my_dict.items():
        ret_str += "#define "
        ret_str += key + " " + str(value) + "\n"
    
    
    return ret_str + "\n"



        
def format_bytearr_for_printout(byte_arr):
    formatted = ",\n".join(
    ", ".join(f"0x{b:02x}" for b in byte_arr[i:i+4])
    for i in range(0, len(byte_arr), 4)
    )

    return formatted + ", "



def parse_formatted_hex(formatted_str):
    """
    Parse a formatted string of hexadecimal bytes back into a byte array.
    This is the inverse of format_bytearr_for_printout.
    
    Args:
        formatted_str (str): A string formatted as "0xXX, 0xXX, ..., 0xXX, "
    
    Returns:
        bytearray: The original byte array
    """
    # Remove any whitespace and newlines
    cleaned_str = formatted_str.replace("\n", "").replace(" ", "")
    
    # Remove trailing comma if it exists
    if cleaned_str.endswith(","):
        cleaned_str = cleaned_str[:-1]
    
    # Split by commas
    hex_values = cleaned_str.split(",")
    
    # Filter out any empty strings
    hex_values = [val for val in hex_values if val]
    
    # Convert each hex string to integer and then to byte
    byte_arr = bytearray()
    for hex_val in hex_values:
        if hex_val.startswith("0x"):
            byte_val = int(hex_val, 16)
            byte_arr.append(byte_val)
    
    return byte_arr
    



def write_cms_to_files(header_filepath, lif_params_on_sram, cms_driver_payload_byte_array, register_cms, base_name, sizes_dict, addr_dict, quant_params_dict, lif_params_arr_contents_str, lut_arr_contents_str, weight_byte_arr=None, bias_byte_arr=None):
    
    formatted_cms = format_bytearr_for_printout(cms_driver_payload_byte_array)
    
    if weight_byte_arr:
        formatted_biases = format_bytearr_for_printout(bias_byte_arr)
        formatted_weights = format_bytearr_for_printout(weight_byte_arr)
        
    with open(header_filepath, 'w') as f:
        f.write("#pragma once\n")

        f.write(get_includes_str())


        # Define layer constants as macros
        f.write("// Tensor sizes\n")
        f.write(get_macro_def_str(sizes_dict))
        f.write("// Input/output addresses (Relative Addressing)\n")
        f.write(get_macro_def_str(addr_dict))
        f.write("//Quantization Params\n")
        f.write(get_macro_def_str(quant_params_dict))


        #f.write(get_cms_methods_declare_str(base_name))
        #f.write(get_weight_methods_declare_str(base_name))
        #f.write(get_lut_methods_declare_str(base_name))
        #f.write(get_lif_param_methods_declare_str(base_name))


        
    #with open(imp_filepath, 'w') as f:
        #f.write("#include \"include/"+base_name+".h\"")

        f.write("\n\n\n\n\n\n")


        f.write(get_cms_arr_def_str(base_name)+"\n")
        f.write(formatted_cms)
        f.write("\n\n};\n\n\n")

        f.write(get_cms_methods_definition_str(base_name))


        if weight_byte_arr:
            f.write(get_weights_arr_def_str(base_name)+"\n")
            f.write("//biases\n")
            f.write(formatted_biases + "\n")
            f.write("//weights\n")
            f.write(formatted_weights)
            f.write("\n\n};\n\n\n")

            f.write(get_weight_methods_definition_str(base_name))

        
        # Write LIF Params
        if lif_params_arr_contents_str:
            f.write(get_lif_param_arr_def_str(base_name, lif_params_on_sram) + "\n")
            f.write(lif_params_arr_contents_str)
            f.write("\n\n};\n\n\n")

            f.write(get_lif_param_methods_definition_str(base_name))

        # Write LUT
        if lut_arr_contents_str:
            f.write(get_lut_arr_def_str(base_name) + "\n")
            f.write(lut_arr_contents_str)
            f.write("\n\n};\n\n\n")

            f.write(get_lut_methods_definition_str(base_name))



        f.write(register_cms_2_assembly(register_cms))






def gen_cms(npu_op_list, accelerator, debug_mode=False):
    register_command_stream = npu_generate_register_command_stream(npu_op_list, accelerator)
    if debug_mode:
        print("\nCommands:\n")
        print(register_cms_2_assembly(register_command_stream))


    driver_payload_byte_array = npu_create_driver_payload(register_command_stream, accelerator)


    return driver_payload_byte_array, register_command_stream




'''

This function is not used anymore, weight/bias length and addressing check is now done in def_fullyconnected()
'''

def check_weight_and_bias_len_correct(cms_name, addr_dict, weight_byte_arr, bias_byte_arr):

    # Assume that the SRAM Looks like this:
    '''
    In_spk
    Bias
    Weights
    v_mem
    time_not_updated
    tmp1
    tmp2
    out_spk
    '''


    bias_addr = addr_dict[cms_name.upper()+"_BIAS_ADDR"]
    weight_addr = addr_dict[cms_name.upper()+"_WEIGHT_ADDR"]
    v_mem_addr = addr_dict[cms_name.upper()+"_V_MEM_ADDR"]

    if (weight_addr - bias_addr) != len(bias_byte_arr):
        print("Incorrect bias length or bias addressing is incorrect")
        print("\t BIAS_LEN set:", len(bias_byte_arr), "but addressing implies bias_len:", (weight_addr - bias_addr))
        print("exiting...")
        exit()
    
    if (v_mem_addr - weight_addr) != len(weight_byte_arr):
        print("Incorrect weight length or weight addressing is incorrect")
        print("\t WEIGHT_LEN set:", len(weight_byte_arr), "but addressing implies weight_len:", (v_mem_addr - weight_addr))
        print("exiting...")
        exit()
    





def get_arr_string_1D(array_name, array):

    ret_str = "static const float " + array_name + " [" + str(len(array)) + "] = {\n"

    for i in range(len(array)):
        ret_str += str(array[i]) + ", "

        if i+1 % 4 == 0:
            ret_str += "\n"

    ret_str += "\n};\n\n\n"

    return ret_str

def get_arr_string_2D(array_name, array, mem_store_loc):

    ret_str = "static const float " + array_name + " [" + str(len(array)*len(array[0])) + "] __attribute__((section(\"" + mem_store_loc +"\"))) = {\n"

    for i in range(len(array)):


        for j in range(len(array[i])):
            ret_str += str(array[i][j]) + ", "

            if i+1 % 4 == 0:
                ret_str += "\n"
        

    ret_str += "\n};\n\n\n"

    return ret_str


def get_non_const_arr_string1D(array_name, arr_size, mem_store_loc):

    ret_str = "static float " + array_name + "[" + str(arr_size) + "] __attribute__((section(\"" + mem_store_loc +"\")));\n"

    return ret_str


def write_to_cpu_file(cpu_filepath, base_name, num_layers, layer_sizes_list, mem_store_loc_list, num_time_steps, weights_arr_list, biases_arr_list, vth_arr_list, beta_arr_list):
    
    with open(cpu_filepath, 'w') as f:


        f.write("#pragma once\n\n\n\n\n")
        f.write("#include \"../model.h\"\n\n\n\n")

        f.write("#define CPU_NN_MODEL_NUM_LAYERS " + str(num_layers) + "\n")
        f.write("#define CPU_NN_MODEL_NUM_TIME_STEPS " + str(num_time_steps) + "\n\n\n\n\n\n")


        # Define const arrays
        for layer_num in range(num_layers):
            f.write(get_arr_string_2D("weights" + base_name + str(layer_num), weights_arr_list[layer_num], "nn_model"))
            f.write(get_arr_string_1D("biases" + base_name + str(layer_num), biases_arr_list[layer_num]))
            f.write(get_arr_string_1D("beta" + base_name + str(layer_num), beta_arr_list[layer_num]))
            f.write(get_arr_string_1D("vth" + base_name + str(layer_num), vth_arr_list[layer_num]))
        

        # Define non const arrays
        for layer_num in range(num_layers):
            if layer_num == 0:
                f.write(get_non_const_arr_string1D("in_spk" + base_name + str(layer_num), layer_sizes_list[layer_num], mem_store_loc_list[layer_num]))

            f.write("\n")
            f.write(get_non_const_arr_string1D("v_mem" + base_name + str(layer_num), layer_sizes_list[layer_num+1], mem_store_loc_list[layer_num]))
            f.write(get_non_const_arr_string1D("time_since_last_update" + base_name + str(layer_num), layer_sizes_list[layer_num+1], mem_store_loc_list[layer_num]))
            f.write(get_non_const_arr_string1D("out_spk" + base_name + str(layer_num), layer_sizes_list[layer_num+1], mem_store_loc_list[layer_num]))
            f.write("\n")

            if layer_num == num_layers-1:
                f.write(get_non_const_arr_string1D("out_spk_sum" + base_name + str(layer_num), layer_sizes_list[layer_num+1], mem_store_loc_list[layer_num]))
                f.write("\n\n\n")

                




        # Define the functions initing each layer
        for layer_num in range(num_layers):
            layer_name = base_name + str(layer_num)
            f.write("NNLayer_CPU* Init_cpu" + layer_name + "() {\n")
            f.write("\tNNLayer_CPU* " + layer_name + " = FC_LIF_Layer_CPU_Init(\n")
            f.write("\t\tweights"+layer_name + ",\n")
            f.write("\t\tbiases"+layer_name + ",\n")
            f.write("\t\tbeta"+layer_name + ",\n")
            f.write("\t\tvth"+layer_name + ",\n")

            if layer_num == 0:
                f.write("\t\tin_spk"+layer_name+",\n")
            else:
                f.write("\t\tout_spk"+base_name + str(layer_num-1) + ",\n")

            f.write("\t\tout_spk"+layer_name + ",\n")
            f.write("\t\tv_mem"+layer_name + ",\n")
            f.write("\t\ttime_since_last_update" + layer_name + ",\n")

            f.write("\t\t"+ str(layer_sizes_list[layer_num]) + ",\n")
            f.write("\t\t"+ str(layer_sizes_list[layer_num+1]) + ",\n")
            if layer_num == num_layers-1:
                f.write("\t\tout_spk_sum"+layer_name + "\n")
            else:
                f.write("\t\tNULL\n")
        
            f.write("\t);\n")

            f.write("\treturn " + layer_name +";\n")

            f.write("}\n\n")

            




        # Write the array pointing to the init functions for each layer
        f.write("\n")
        f.write("NNLayer_CPU* (*init_cpu_layers_func[" + str(num_layers) +"]) (void) = {\n")
        
        for layer_num in range(num_layers):
            f.write("\tInit_cpu" + base_name + str(layer_num) + ",\n")
        
        f.write("};")


