from ethosu.vela.api import *

from cms_interpreter import register_cms_2_assembly



def check_block_config_legal(block_config, my_op, accelerator):
    # Check that block config is legal
    available_block_configs = npu_find_block_configs(my_op, accelerator)

    block_config_legal = False
    for i in range(len(available_block_configs)):
        if block_config == available_block_configs[i]:
            block_config_legal = True

    if not block_config_legal:
        print("ERROR: BLOCK_CONFIG is not legal, found\n\t", block_config, "\n\n\t But expected one of the following:\n\t", available_block_configs)
        exit()
    else:
        print("//Current Block Config is legal:", block_config)






def float_to_int_safe(x: float) -> int:
    if not x.is_integer():
        raise ValueError(f"Cannot convert {x} to int: it has nonzero decimals.")
    return int(x)


def get_includes_str():
    return "#include <stddef.h>\n#include <stdint.h>\n\n\n\n\n"

def get_tensor_arena_size_str(basename):
    from mem_u_int8 import TENSOR_ARENA_SIZE, INPUT_LAYER_SIZE, OUTPUT_LAYER_SIZE

    ret_str = "#define " + basename.upper() + "_TENSOR_ARENA_SIZE " + str(TENSOR_ARENA_SIZE) + "\n"
    ret_str += "#define " + basename.upper() + "_INPUT_LAYER_SIZE " + str(INPUT_LAYER_SIZE) + "\n"
    ret_str += "#define " + basename.upper() + "_OUTPUT_LAYER_SIZE " + str(OUTPUT_LAYER_SIZE) + "\n\n"

    return ret_str



def get_lut_methods_definition_str(base_name):
    return "\n\n\n\n\nconst int8_t* Get" + base_name + "LUTPointer()\n{\n\treturn lut_" + base_name + ";\n}\nsize_t Get" + base_name + "LUTLen()\n{\n\treturn sizeof(lut_" + base_name + ");\n}\n\n"


def get_cms_methods_definition_str(base_name):
    return "\n\n\n\n\nconst uint8_t * Get" + base_name + "CMSPointer()\n{\n\treturn cms_" + base_name + ";\n}\n\nsize_t Get"+ base_name +"CMSLen()\n{\n\treturn sizeof(cms_" + base_name + ");\n}\n\n"

def get_weight_methods_definition_str(base_name):
    return "\n\n\n\nconst int8_t * Get" + base_name + "WeightsPointer()\n{\n\treturn weight_" + base_name + ";\n}\n\nsize_t Get"+ base_name +"WeightsLen()\n{\n\treturn sizeof(weight_" + base_name + ");\n}\n\n"


def get_lut_methods_declare_str(base_name):
    return "\n\n\n\n\nconst int8_t* Get" + base_name + "LUTPointer();\n\n\nsize_t Get" + base_name + "LUTLen();\n\n\n"

def get_cms_methods_declare_str(base_name):
    return "\n\n\n\n\nconst uint8_t * Get" + base_name + "CMSPointer();\n\n\nsize_t Get"+ base_name +"CMSLen();\n\n\n"

def get_weight_methods_declare_str(base_name):
    return "\n\n\n\nconst int8_t * Get" + base_name + "WeightsPointer();\n\nsize_t Get"+ base_name +"WeightsLen();\n\n\n"




def get_cms_arr_def_str(base_name):
    return "\n\n\n\nstatic const uint8_t cms_" + base_name + "[] __attribute__((aligned(16))) = \n{\n"

def get_weights_arr_def_str(base_name):
    return "\n\n\n\nstatic const int8_t weight_" + base_name + "[] __attribute__((aligned(16))) = \n{\n"

def get_lut_arr_def_str(base_name):
    return "\n\n\n\nstatic const int8_t lut_" + base_name + "[] __attribute__((aligned(16))) = \n{\n" 


def get_addr_macros(addr_dict):
    ret_str = "// Input/output addresses (Relative Addressing)\n"
    for key, value in addr_dict.items():
        ret_str += "#define "
        ret_str += key + " " + str(value) + "\n"
    
    return ret_str + "\n"
    


def get_quant_vars(quant_params_dict):

    
    
    ret_str = "//Quantization Params\n"
    # Skip first because first op is DMA
    for key, value in quant_params_dict.items():
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
    



def write_cms_to_files(header_filepath, imp_filepath, cms_driver_payload_byte_array, register_cms, base_name, addr_dict, quant_params_dict, lut_arr_contents_str, weight_byte_arr=None, bias_byte_arr=None):
    
    formatted_cms = format_bytearr_for_printout(cms_driver_payload_byte_array)
    
    if weight_byte_arr:
        formatted_biases = format_bytearr_for_printout(bias_byte_arr)
        formatted_weights = format_bytearr_for_printout(weight_byte_arr)
        print(bias_byte_arr)
        print(weight_byte_arr)
        

    with open(header_filepath, 'w') as f:
        f.write("#pragma once\n")

        f.write(get_includes_str())

        f.write(get_tensor_arena_size_str(base_name))

        f.write(get_addr_macros(addr_dict))
        f.write(get_quant_vars(quant_params_dict))

        f.write(get_cms_methods_declare_str(base_name))
        f.write(get_weight_methods_declare_str(base_name))
        f.write(get_lut_methods_declare_str(base_name))


        

    with open(imp_filepath, 'w') as f:
        f.write("#include \"include/my_mem_u.h\"")
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






def check_weight_and_bias_len_correct(cms_name, addr_dict, weight_byte_arr, bias_byte_arr):

    # Assume that the SRAM Looks like this:
    '''
    In_spk
    Bias
    Weights
    ln_beta
    v_th
    v_mem
    time_not_updated
    tmp1
    tmp2
    out_spk
    '''


    bias_addr = addr_dict[cms_name.upper()+"_BIAS_ADDR"]
    weight_addr = addr_dict[cms_name.upper()+"_WEIGHT_ADDR"]
    ln_beta_addr = addr_dict[cms_name.upper()+"_LN_BETA_ADDR"]

    if (weight_addr - bias_addr) != len(bias_byte_arr):
        print("Incorrect bias length or bias addressing is incorrect")
        print("\t BIAS_LEN set:", len(bias_byte_arr), "but addressing implies bias_len:", (weight_addr - bias_addr))
        exit()
    
    if (ln_beta_addr - weight_addr) != len(weight_byte_arr):
        print("Incorrect weight length or weight addressing is incorrect")
        print("\t WEIGHT_LEN set:", len(weight_byte_arr), "but addressing implies weight_len:", (ln_beta_addr - weight_addr))
        exit()
    
