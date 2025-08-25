from typing import List, Optional
from enum import Enum



# For writing to file
class MethodAccessType(str, Enum):
    POINTER = "Pointer"
    LEN = "Len"


def get_cms_methods_definition_str(base_name):
    return "\n\n\n\n\nstatic inline const uint8_t * Get" + base_name + "CMSPointer()\n{\n\treturn cms_" + base_name + ";\n}\n\nstatic inline size_t Get"+ base_name +"CMSLen()\n{\n\treturn sizeof(cms_" + base_name + ");\n}\n\n"


def format_bytearr_for_printout(byte_arr):
    formatted = ",\n".join(
    ", ".join(f"0x{b:02x}" for b in byte_arr[i:i+4])
    for i in range(0, len(byte_arr), 4)
    )

    return formatted + ", "



def get_includes_str():
    return "#include <stddef.h>\n#include <stdint.h>\n\n\n\n\n"



def get_macro_def_str(my_dict):

    
    
    ret_str = "\n"
    # Skip first because first op is DMA
    for key, value in my_dict.items():
        ret_str += "#define "
        ret_str += key + " " + str(value) + "\n"
    
    
    return ret_str + "\n"


def get_layer_name(base_name:str, layer_num: int) -> str:
    return f"{base_name}{layer_num}"

def get_arr_name(arr_name: str, layer_name: str) -> str:
    return f"{arr_name}_{layer_name}"

def get_method_name(arr_name:str, layer_name: str, access_type: MethodAccessType) -> str:
    return f"Get{arr_name}{layer_name}{access_type}"


#def get_arr_def_str(dtype: str, arr_name: str, layer_name: str) -> str:
    #return f"\n\n\n\nstatic {dtype} {get_arr_name(arr_name, layer_name)} [] __attribute__((aligned(16))) = \n{{\n"

def get_arr_dec_str(dtype: str, arr_name: str, layer_name:str, arr_len, attribute_list: List[str] = []) -> str:
    ret_str = f"\n{dtype} {get_arr_name(arr_name, layer_name)} [{arr_len}]"
    for attribute in attribute_list:
        ret_str += f" __attribute__(({attribute}))"
    ret_str += ";\n"

    return ret_str
    

# Already preprocessed and just dump string into array
def get_arr_def_str_dump_string_contents(dtype: str, arr_name: str, layer_name: str, array_content_str: str, attribute_list: List[str]=[]) -> str:

    if not "const " in dtype:
        print("Warning: Dumping string to array but is not of const type. Should most likely use const type here")



    ret_str = f"\n{dtype} {get_arr_name(arr_name, layer_name)}[]"

    for attribute in attribute_list:
        ret_str += f" __attribute__(({attribute}))"

    ret_str += " =\n"
    ret_str += "{\n"

    ret_str += array_content_str
    ret_str += "};\n"

    return  ret_str

# 
def get_arr_def_str(dtype: str, arr_name: str, layer_name: str, arr_values_list: List, attribute_list: List[str]=[]) -> str:

    ret_str = f"\n{dtype} {get_arr_name(arr_name, layer_name)} [{len(arr_values_list)}]"
    
    for attribute in attribute_list:
        ret_str += f" __attribute__(({attribute}))"

    ret_str += " =\n{\n"

    for element in arr_values_list:
        ret_str += f"\t{element},\n"
    
    ret_str += "};\n"

    return  ret_str

def get_method_def_str(dtype: str, arr_name: str, layer_name: str, access_type: MethodAccessType) -> str:

    if access_type == MethodAccessType.POINTER:
        return_type_str = f"{dtype}*"
        return_line_str = f"\treturn {get_arr_name(arr_name, layer_name)};\n"
    elif access_type == MethodAccessType.LEN:
        return_type_str = f"const size_t"
        return_line_str = f"\treturn sizeof({get_arr_name(arr_name, layer_name)});\n"
    else:
        raise ValueError(f"Unexpected Type found for MethodAccessType: Found {access_type}")

    ret_str = f"\nstatic inline {return_type_str} {get_method_name(arr_name, layer_name, access_type)}()\n"
    ret_str += "{\n" + return_line_str + "}\n\n"

    return ret_str
