
from common import format_bytearr_for_printout, get_includes_str, get_macro_def_str, get_arr_def_str, get_arr_dec_str, get_method_def_str, get_arr_def_str_dump_string_contents, get_arr_name
from common import MethodAccessType

from constants import *

from cms_interpreter import register_cms_2_assembly
from data_structs import MemoryAllocator


def write_cms_to_files(header_filepath, mem_alloc: MemoryAllocator, cms_driver_payload_byte_array, register_cms, mem_store_loc, layer_name, layer_num, sizes_dict, addr_dict, quant_params_dict):
    
    formatted_cms = format_bytearr_for_printout(cms_driver_payload_byte_array)
    
        
    with open(header_filepath, 'w') as f:
        f.write("#pragma once\n")

        f.write(get_includes_str())


        # Define layer constants as macros
        f.write("// Tensor sizes\n")
        f.write(get_macro_def_str(sizes_dict))



        

        f.write("\n\n\n\n\n\n")

        # Note: Start here, and make the arr definitions and method definitions independent of what
        # type of custom regions are used
        f.write(get_arr_def_str_dump_string_contents("static const uint8_t", "cms", layer_name, formatted_cms, ["aligned(16)"]))

        #f.write(get_cms_methods_definition_str(layer_name))
        f.write(get_method_def_str("const uint8_t", "cms", layer_name, MethodAccessType.POINTER))
        f.write(get_method_def_str(None, "cms", layer_name, MethodAccessType.LEN))


        #### Write Memory Regions ###
        #mem_regions_arr_name_list = []
        #mem_regions_size_list = []
        #mem_regions_region_number_list = []
        ##for region_name, region in mem_alloc.regions.items():

        for region in mem_alloc.get_sorted_mem_regions():
            # MRAM Regions
            if region.is_on_mram:
                f.write(get_arr_def_str_dump_string_contents("static "+region.dtype, region.name, layer_name, region.arr_values_str, ["aligned(16)"]))

            # SRAM Regions
            else:
                # Only Declare Array for Input Region if its the first layer
                if (region.name == INPUT_REGION_NAME and layer_num != 0):
                    continue
                
                f.write(get_arr_dec_str("static "+region.dtype, region.name, layer_name, region.size, [f'section("{mem_store_loc}")', f"aligned(16)"]))
            
            
            f.write(get_method_def_str(region.dtype, region.name, layer_name, MethodAccessType.POINTER))
            f.write(get_method_def_str(None, region.name, layer_name, MethodAccessType.LEN))

        



        ### Write tensors ###
        tensor_name_list = []
        tensor_relative_addr_list = []
        tensor_region_list = []
        tensor_sizes_list = []
        tensor_scale_list = []
        tensor_zero_point_list = []

        for tensor in mem_alloc.tensors:
            tensor_name_list.append(f'"{tensor.name}"')
            tensor_relative_addr_list.append(tensor.address)
            tensor_region_list.append(tensor.region)
            tensor_sizes_list.append(tensor.get_size())
            tensor_scale_list.append(tensor.quantization.scale_f32)
            tensor_zero_point_list.append(tensor.quantization.zero_point)

        f.write(get_arr_def_str("char*", "name", layer_name, tensor_name_list))
        f.write(get_arr_def_str("size_t", "relative_addr", layer_name, tensor_relative_addr_list))
        f.write(get_arr_def_str("size_t", "region", layer_name, tensor_region_list))
        f.write(get_arr_def_str("size_t", "size", layer_name, tensor_sizes_list))
        f.write(get_arr_def_str("float", "scale", layer_name, tensor_scale_list))
        f.write(get_arr_def_str("int", "zero_point", layer_name, tensor_zero_point_list))

            
        f.write(get_method_def_str("char*", "name", layer_name, MethodAccessType.POINTER))
        f.write(get_method_def_str("size_t", "relative_addr", layer_name, MethodAccessType.POINTER))
        f.write(get_method_def_str("size_t", "region", layer_name, MethodAccessType.POINTER))
        f.write(get_method_def_str("size_t", "size", layer_name, MethodAccessType.POINTER))
        f.write(get_method_def_str("float", "scale", layer_name, MethodAccessType.POINTER))
        f.write(get_method_def_str("int", "zero_point", layer_name, MethodAccessType.POINTER))



        f.write(register_cms_2_assembly(register_cms))