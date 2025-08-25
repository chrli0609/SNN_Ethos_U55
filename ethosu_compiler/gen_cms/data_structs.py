from dataclasses import dataclass, field
from typing import List, Dict, Optional


import numpy as np


from constants import *
from extra_func import next_multiple
from config_ops import symmetric_zero_point_quant, zero_point_quant
from ethosu.vela.api import NpuShape3D, NpuDataType, NpuLayout, NpuFeatureMap, NpuQuantization, NpuTileBox

@dataclass
class FeatureMap:
    address: int
    size: int

#@dataclass
class Region:
    def __init__(self, number: str, name: str, dtype: str, is_on_mram: str):
        self.number = number
        self.name = name
        self.dtype = dtype
        self.is_on_mram = is_on_mram
        self.memory_map = {}
        self.size = 0
        self.array_values_str = None
        self.previous_fm = None
    #number: int
    #name: str
    #dtype: str
    #is_on_mram: bool
    #memory_map: dict[str, int] = field(default_factory=dict)
    #size: int = 0

    ## Only will be set if is_on_mram == True
    #arr_values_str: str = None

    #previous_fm: FeatureMap = None # only used when constructing

    def __str__(self):
        return f"\tnumber: {self.number}\n\tdtype: {self.dtype}\n\tis_on_mram: {self.is_on_mram}\n\tmemory_map: {self.memory_map}\n\tsize: {self.size}"


class Tensor():
    def __init__(self):
        self.name: str = None
        self.region: int = None
        self.address = None
        self.shape = NpuShape3D
        self.data_type: NpuDataType
        self.tensor_values: List[float] = []

        self.quantization: Optional[NpuQuantization] = None

    def get_size(self) -> int:
        return self.shape.height * self.shape.width * self.shape.depth
    
    #def get_relative_addr(fm: NpuFeatureMap) -> int:
        #return fm.tiles.addresses[0]
    
    def get_attr_name(self, attr_value):
        for attr_name, value in self.__dict__.items():
            if value is attr_value:  # identity check
                return attr_name
        return None


class MemoryAllocator:

    def __init__(self,  input_size:     int, 
                        output_size:    int,
                        custom_memory_regions:        Dict[str, Region],
                ):
        
        self.input_size = input_size
        self.output_size = output_size

        self.regions = {
            MODEL_REGION_NAME : Region(0, MODEL_REGION_NAME, "const int8_t", True),
            SRAM_SCRATCH_REGION_NAME : Region(1, SRAM_SCRATCH_REGION_NAME, "int8_t", False),
            SRAM_FAST_SCRATCH_REGION_NAME : Region(2, SRAM_FAST_SCRATCH_REGION_NAME, "int8_t", False),
            INPUT_REGION_NAME : Region(5, INPUT_REGION_NAME, "int8_t", False),
            OUTPUT_REGION_NAME : Region(6, OUTPUT_REGION_NAME, "int8_t", False),

        }
        # Check that there are no invalid region numbers
        for key, value in custom_memory_regions.items():

            # The range of legal custom regions are 3 - 7
            if value.number <= 2 or value.number > 7:
                raise Exception(f"Custom Region '{key}' has out-of-range value: {value.number}, expected value between 3 and 7")
            else:
                self.regions[key] = value
            

        self.tensors = set()

        # Must define the input output tensors, expect type Tensor
        self.input_tensor = None
        self.output_tensor = None


    def alloc(self, region_name: str, tensor_name: str, fm_size: int, fm_format: int = NpuLayout.NHWC) -> int:



        current_region = self.regions[region_name]



        # If Layout format for an FM is NHCWB16, then 16-byte alignment is required on address
        if current_region.previous_fm == None:
            address_padding = 0
            new_address = 0
        else:
            if fm_format == NpuLayout.NHCWB16:
                address_padding = next_multiple(current_region.previous_fm.size, 16) - current_region.previous_fm.size
            else:
                address_padding = 0
            
            new_address = current_region.previous_fm.address + current_region.previous_fm.size + address_padding
            




        self.regions[region_name].memory_map[tensor_name] = new_address

        # Increment total memory region size by the current fm size
        # Must also include the padding
        self.regions[region_name].size += fm_size + address_padding

        # Store FM that was just added as the previous FM
        self.regions[region_name].previous_fm = FeatureMap(new_address, fm_size)




        
        

    def create_feature_map_v2(self,
                        height: int, width: int, depth: int,
                        region_name: str,
                        segment_name: str,
                        layout,  # Pass NpuLayout.NHWC or similar
                        data_type,  # Pass NpuDataType.INT8 or similar
                        #fm_elem_size: int,
                        max_fm_value,
                        min_fm_value,
                        is_symmetric_quant,
                        tensor_name,
                        ) -> NpuFeatureMap:
        """
        Create an NpuFeatureMap with the given parameters.
        """
        fm = NpuFeatureMap()
        fm.shape = NpuShape3D(height=height, width=width, depth=depth)


        fm.region = self.regions[region_name].number
    
        if layout is not None:
            fm.layout = layout
    
        if data_type is not None:
            fm.data_type = data_type
    
        if is_symmetric_quant:
            scale, zero_point = symmetric_zero_point_quant(max_fm_value, min_fm_value)
        else:
            scale, zero_point = zero_point_quant(max_fm_value, min_fm_value)
        #if scale is not None and zero_point is not None:
        fm.quantization = NpuQuantization(scale_f32=scale, zero_point=zero_point)
    

        # Stride is purely dependent on FM dimensions
        #stride_y = fm_elem_size*depth*width
        #stride_x = fm_elem_size*depth
        #stride_c = fm_elem_size
        #if stride_y is not None and stride_x is not None and stride_c is not None:
            #fm.strides = NpuShape3D(height=stride_y, width=stride_x, depth=stride_c)
    

        fm_addr = self.regions[region_name].memory_map[segment_name]
        # Default tile setup for single tile (most common case)
        fm.tiles = NpuTileBox(
            height_0=height, 
            height_1=0, 
            width_0=width, 
            addresses=[fm_addr, 0, 0, 0]
        )
    
        fm.name = tensor_name


        #### Create Tensor  ###
        tensor = Tensor()
        tensor.name = fm.name
        tensor.address = fm_addr
        tensor.region = fm.region
        if fm.data_type != NpuDataType.INT8:
            raise ValueError("Error: Currently Do not support any data type other than INT8")
        tensor.data_type = "int8_t"
        tensor.shape = fm.shape
        tensor.quantization = NpuQuantization(scale, zero_point)

        self.tensors.add(tensor)
    
        return fm
    
    def create_weight_and_bias_fm(self,
                                  height: int, width: int, depth: int,
                                  region_name: str,
                                  segment_name: str,
                                  data_type: NpuDataType,
                                  tensor_values: List[float],
                                  tensor_name: str) -> Tensor:
        tensor = Tensor()
        tensor.shape = NpuShape3D(height=height, width=width, depth=depth)


        tensor.region = self.regions[region_name].number
    
    
        if data_type != NpuDataType.INT8:
            raise ValueError("Currently Only support INT8")
        tensor.data_type = data_type
    
        #scale, zero_point = symmetric_zero_point_quant(max_fm_value, min_fm_value)
        #if scale is not None and zero_point is not None:

        max_weight_val = np.max(tensor_values)
        min_weight_val = np.min(tensor_values)
        # Get the one with the largest absolute value (since the npu only supports symmetric quantization for weights)
        if abs(min_weight_val) > abs(max_weight_val):
            largest_weight_abs_val = abs(min_weight_val)
        else:
            largest_weight_abs_val = abs(max_weight_val)

        max_fm_value = largest_weight_abs_val# + MIN_DIFF 
        min_fm_value = -(largest_weight_abs_val)# + MIN_DIFF)
        scale, zero_point = symmetric_zero_point_quant(max_fm_value, min_fm_value)
        tensor.quantization = NpuQuantization(scale_f32=scale, zero_point=zero_point)
    

        # Stride is purely dependent on FM dimensions
        #stride_y = fm_elem_size*depth*width
        #stride_x = fm_elem_size*depth
        #stride_c = fm_elem_size
        #if stride_y is not None and stride_x is not None and stride_c is not None:
            #fm.strides = NpuShape3D(height=stride_y, width=stride_x, depth=stride_c)
    

        fm_addr = self.regions[region_name].memory_map[segment_name]
        tensor.address = fm_addr
        ## Default tile setup for single tile (most common case)
        #fm.tiles = NpuTileBox(
            #height_0=height, 
            #height_1=0, 
            #width_0=width, 
            #addresses=[fm_addr, 0, 0, 0]
        #)
    
        tensor.name = tensor_name

        tensor.tensor_values = tensor_values

        self.tensors.add(tensor)

    
        return tensor
    
    def set_input_tensor(self, input_tensor_name) -> None:
        for tensor in self.tensors:
            if tensor.name == input_tensor_name:
                self.input_tensor = tensor
                return None

        raise Exception(f"Assigned input tensor is not among the registered tensors")

    def set_output_tensor(self, output_tensor_name) -> None:
        for tensor in self.tensors:
            if tensor.name == output_tensor_name:
                self.output_tensor = tensor
                return None

        raise Exception(f"Assigned output tensor is not among the registered tensors:\nRegistered Tensors {self.tensors}")

    

    def is_custom_region(self, region_name):
        if self.regions[region_name].number >= 3:
            return True
        else:
            return False
        
    def print_mem_regions(self):
        for region_name, region in self.regions.items():
            print(f"{region_name}")
            print(region)

    def get_sorted_mem_regions(self):
        #for region in sorted(self.regions.values(), key=lambda r: r.number):
        return sorted(self.regions.values(), key=lambda r: r.number)




