import math
import os, sys

from dataclasses import dataclass, field


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ethosu.vela.api import *





from config_ops import *
from extra_func import *




@dataclass
class FeatureMap:
    address: int
    size: int

@dataclass
class Region:
    number: int
    memory_map: dict[str, int] = field(default_factory=dict)
    size: int = 0
    previous_fm: FeatureMap = None # only used when constructing


class WeightFeatureMap():
    def __init__(self):
        self.name: str = None
        self.region: int = None
        self.address = None
        self.shape = NpuShape3D
        self.data_type: NpuDataType
        self.tensor_values: List[float] = []



class MemoryAllocator:

    def __init__(self,  input_size:     int, 
                        output_size:    int,
                        regions:        Dict[str, Region],
                ):
        
        self.input_size = input_size
        self.output_size = output_size

        # Check that there are no invalid region numbers
        for key, value in regions.items():
            if value.number < 0 or value.number > 7:
                raise Exception(f"Region '{key}' has out-of-range value: {value.number}, expected value between 0 and 7")

        self.regions = regions
        self.tensors = set()


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

        self.tensors.add(fm)
    
        return fm
    
    def create_weight_and_bias_fm(self,
                                  height: int, width: int, depth: int,
                                  region_name: str,
                                  segment_name: str,
                                  data_type: NpuDataType,
                                  tensor_values: List[float],
                                  tensor_name: str) -> WeightFeatureMap:
        fm = WeightFeatureMap()
        fm.shape = NpuShape3D(height=height, width=width, depth=depth)


        fm.region = self.regions[region_name].number
    
    
        if data_type is not None:
            fm.data_type = data_type
    
        #scale, zero_point = symmetric_zero_point_quant(max_fm_value, min_fm_value)
        #if scale is not None and zero_point is not None:

        #max_weight_val = np.max(tensor_values)
        #min_weight_val = np.min(tensor_values)
        ## Get the one with the largest absolute value (since the npu only supports symmetric quantization for weights)
        #if abs(min_weight_val) > abs(max_weight_val):
            #largest_weight_abs_val = abs(min_weight_val)
        #else:
            #largest_weight_abs_val = abs(max_weight_val)

        #max_fm_value = largest_weight_abs_val# + MIN_DIFF 
        #min_fm_value = -(largest_weight_abs_val)# + MIN_DIFF)
        #scale, zero_point = symmetric_zero_point_quant(max_fm_value, min_fm_value)
        #fm.quantization = NpuQuantization(scale_f32=scale, zero_point=zero_point)
    

        # Stride is purely dependent on FM dimensions
        #stride_y = fm_elem_size*depth*width
        #stride_x = fm_elem_size*depth
        #stride_c = fm_elem_size
        #if stride_y is not None and stride_x is not None and stride_c is not None:
            #fm.strides = NpuShape3D(height=stride_y, width=stride_x, depth=stride_c)
    

        fm_addr = self.regions[region_name].memory_map[segment_name]
        fm.address = fm_addr
        ## Default tile setup for single tile (most common case)
        #fm.tiles = NpuTileBox(
            #height_0=height, 
            #height_1=0, 
            #width_0=width, 
            #addresses=[fm_addr, 0, 0, 0]
        #)
    
        fm.name = tensor_name

        fm.tensor_values = tensor_values

    
        return fm




def get_elementwise_op(op_type, ifm, ifm2, ofm,
                    accelerator,
                    ifm2_scalar=None,
                    activation_op=NpuActivationOp.NONE_OR_RELU,
                    activation_max_val=None,
                    activation_min_val=None,
                    reversed_operands=False,
                    rescale=None,
                    fused_quantize=False,
                    rounding_mode=NpuRoundingMode.TFL,
                    ifm_upscale=NpuResamplingMode.NONE,
                    accumulator=NpuAccumulatorType.Default
                ):

    activation = create_activation(
        activation_op=activation_op,
        min_val=activation_max_val,
        max_val=activation_min_val
        )


    ew_op = NpuElementWiseOperation(op_type)

    #elementwise operation
    ew_op.reversed_operands = reversed_operands 
    ew_op.rescale = rescale

    #NpuBlockOperation
    ew_op.ifm = ifm

    # If it is scalar instead of tensor, then set the quantization to scale=1, zp=0
    if ifm2_scalar == None:
        ew_op.ifm2 = ifm2
        ew_op.ifm2_scalar = None
    elif ifm2 == None:
        ew_op.ifm2_scalar = ifm2_scalar   #set if ifm2 is a scalar

        ifm2_fm = NpuFeatureMap()
        ifm2_fm.quantization = NpuQuantization(1,0)
        ifm2_fm.name="None"
        ew_op.ifm2 = ifm2_fm
        

    ew_op.ofm = ofm
    ew_op.kernel = None
    ew_op.weights = []
    ew_op.biases = []
    ew_op.padding = None
    ew_op.activation = activation


    block_config = get_block_config(ew_op, accelerator)
    ew_op.block_config = block_config

    ew_op.rounding_mode = rounding_mode
    ew_op.fused_quantize = fused_quantize
    ew_op.ifm_upscale = ifm_upscale
    ew_op.accumulator_type = accumulator


    #check_block_config_legal(block_config, ew_op, ACCELERATOR)

    return ew_op







def get_elementwise_op_with_lut(op_type, ifm, ifm2, pre_lut_ofm, post_lut_ofm, 
                                mem_alloc,
                                lut_index,
                                lut_function,
                                lut_region_name,
                                accelerator, debug_mode=False,


                                activation_max_val=None,
                                activation_min_val=None,
                                reversed_operands=False,
                                rescale=None,
                                fused_quantize=False,
                                rounding_mode=NpuRoundingMode.TFL,
                                ifm_upscale=NpuResamplingMode.NONE,
                                accumulator=NpuAccumulatorType.Default
                                
                                ):



    #DMA for LUT



    # Handle LUT generation and DMA

    dma_lut_op, lut_values = create_lut_and_dma(approximated_func=lut_function, lut_index=lut_index, lut_region=mem_alloc.regions[lut_region_name].number, data_type=pre_lut_ofm.data_type, 
                    scale_pre_lut=pre_lut_ofm.quantization.scale_f32, zero_point_pre_lut=pre_lut_ofm.quantization.zero_point,
                    scale_post_lut=post_lut_ofm.quantization.scale_f32, zero_point_post_lut=post_lut_ofm.quantization.zero_point,
                    accelerator=accelerator,
                    debug_mode=debug_mode
    )

    activation = create_activation(
        activation_op=NpuActivationOp.TABLE_LOOKUP,
        min_val=activation_max_val,
        max_val=activation_min_val,
        lookup_table_index=lut_index
    )


    ew_lut_op = NpuElementWiseOperation(op_type)
    
    #elementwise operation
    ew_lut_op.reversed_operands = reversed_operands
    ew_lut_op.rescale = rescale

    #NpuBlockOperation
    ew_lut_op.ifm = ifm
    ew_lut_op.ifm2 = ifm2
    ew_lut_op.ifm2_scalar = None   #set if ifm2 is a scalar
    ew_lut_op.ofm = pre_lut_ofm
    ew_lut_op.kernel = None
    ew_lut_op.weights = []
    ew_lut_op.biases = []
    ew_lut_op.padding = None
    ew_lut_op.activation = activation
    
    block_config = get_block_config(ew_lut_op, accelerator)
    ew_lut_op.block_config = block_config
    ew_lut_op.rounding_mode = rounding_mode
    ew_lut_op.fused_quantize = fused_quantize
    ew_lut_op.ifm_upscale = ifm_upscale
    ew_lut_op.accumulator_type = accumulator

    #check_block_config_legal(block_config, ew_lut_op, ACCELERATOR)

    return dma_lut_op, ew_lut_op, lut_values, lut_index



def get_pooling_op(op_type,
                   ifm: NpuFeatureMap, ofm: NpuFeatureMap,
                   kernel_size: Tuple[int, int],
                   accelerator,
                   kernel_stride: Tuple[int, int] = (1,1),
                   kernel_dilation: Tuple[int, int] = (1,1),
                   padding: NpuPadding = NpuPadding(0,0,0,0),

                   activation_op=NpuActivationOp.NONE_OR_RELU,
                   activation_max_val=None,
                   activation_min_val=None,

                   rescale=None,
                   rounding_mode=NpuRoundingMode.TFL,
                   fused_quantize=False,
                   ifm_upscale=NpuResamplingMode.NONE,
                   accumulator=NpuAccumulatorType.Default
                   ):



    kernel = NpuKernel(
        w=kernel_size[0], h=kernel_size[1], stride_x=kernel_stride[0], stride_y=kernel_stride[1],
        dilation_x=kernel_dilation[0], dilation_y=kernel_dilation[1]
    )


    #block_config = NpuShape3D(2, 2, 8)


    activation = create_activation(
        activation_op=activation_op,
        min_val=activation_max_val,
        max_val=activation_min_val
    )

    pooling_op = NpuPoolingOperation(op_type)
    
    #Pooling operation
    pooling_op.rescale = rescale

    #NpuBlockOperation
    pooling_op.ifm = ifm
    pooling_op.ifm2 = None
    pooling_op.ifm2_scalar = None   #set if ifm2 is a scalar
    pooling_op.ofm = ofm
    pooling_op.kernel = kernel
    pooling_op.weights = []
    pooling_op.biases = []
    pooling_op.padding = padding
    pooling_op.activation = activation

    block_config = get_block_config(pooling_op, accelerator)
    pooling_op.block_config = block_config
    pooling_op.rounding_mode = rounding_mode
    pooling_op.fused_quantize = fused_quantize
    pooling_op.ifm_upscale = ifm_upscale
    pooling_op.accumulator_type = accumulator

    #check_block_config_legal(block_config, pooling_op, ACCELERATOR)

    return pooling_op





def get_fully_connected_op(ifm, ofm, #weights_volume_ohwi, bias_list, 
                           weight_tensor: WeightFeatureMap, bias_tensor: WeightFeatureMap,
                           mem_alloc,
                           #weight_region_name, weight_fm_name,
                           #bias_region_name, bias_fm_name,
                           accelerator, debug_mode=False,

                            activation_op=NpuActivationOp.NONE_OR_RELU,
                            activation_max_val=None,
                            activation_min_val=None,
                            fused_quantize=False,
                            rounding_mode=NpuRoundingMode.TFL,
                            ifm_upscale=NpuResamplingMode.NONE,
                            accumulator=NpuAccumulatorType.Int32
                           ):


    # Kernel (1 x 1 x Input Channels) for fully connected
    kernel = NpuKernel(
        w=1, h=1, 
        stride_x=1, stride_y=1, dilation_x=1, dilation_y=1
    )


    my_op = NpuConv2DOperation()
    my_op.ifm               =   ifm
    my_op.ifm2              =   None
    my_op.ifm2_scalar       =   None
    my_op.ofm               =   ofm
    block_config = get_block_config(my_op, accelerator)

    

    block_traversal = NpuBlockTraversal.DEPTH_FIRST


    # Define Weights
    if ifm.data_type == NpuDataType.INT8:
        weight_ifm_bitdepth = 8 #int8
    elif ifm.data_type == NpuDataType.INT16:
        weight_ifm_bitdepth = 16 #int16




    # Set Weights Quantization params, it must be symmetric quantization 
    max_weight_val = np.max(weight_tensor.tensor_values)
    min_weight_val = np.min(weight_tensor.tensor_values)
    # Get the one with the largest absolute value (since the npu only supports symmetric quantization for weights)
    if abs(min_weight_val) > abs(max_weight_val):
        largest_weight_abs_val = abs(min_weight_val)
    else:
        largest_weight_abs_val = abs(max_weight_val)
    weight_scale, weight_zero_point = symmetric_zero_point_quant(largest_weight_abs_val, -largest_weight_abs_val)


    weight_byte_arr, bias_byte_arr = gen_weights_and_biases(accelerator=accelerator,
                        weights_volume_ohwi=weight_tensor.tensor_values,
                            dilation_xy=(1,1),
                            ifm_bitdepth=weight_ifm_bitdepth,
                            ofm_block_depth=block_config[2],
                            op_type=NpuOperationType.Conv2D,
                            block_traversal=block_traversal,

                            #ONLY FOR 1 DIM FMs!!!!
                            bias_list=bias_tensor.tensor_values,

                            ifm_scale=ifm.quantization.scale_f32,
                            weight_scale=weight_scale,
                            weight_zero_point=weight_zero_point,
                            ofm_scale=ofm.quantization.scale_f32,


                            is_debug_mode=debug_mode
    )

    weight_n_bias_len = len(bias_byte_arr) + len(weight_byte_arr)
    if debug_mode:
        print("weight_n_bias_len", weight_n_bias_len)
        print("\tbias_len:", len(bias_byte_arr))
        print("\tweight_len", len(weight_byte_arr))

        

    ## Make sure that init is the same as current weights
    #if (weight_byte_arr != weight_byte_arr_init):
        #print("Error: weight_byte_arr != weight_byte_arr_init")
        #exit()
    #if (bias_byte_arr != bias_byte_arr_init):
        #print("Error: bias_byte_arr != bias_byte_arr_init")
        #exit()

    

        


    #BIAS_ADDR = WEIGHT_N_BIAS_ADDR
    #WEIGHT_ADDR = BIAS_ADDR + len(bias_byte_arr)
    
    
    #if (weights_and_biases_on_sram):
        #WEIGHT_N_BIAS_ADDR = BIAS_ADDR #Bias before weights

        ##DMA    
        #dma_src = NpuAddressRange(region=WEIGHT_AND_BIASES_REGION, address=0, length=weight_n_bias_len)
        #dma_dst = NpuAddressRange(region=SRAM_SCRATCH_REGION, address=WEIGHT_N_BIAS_ADDR, length=weight_n_bias_len)
        #dma_op = NpuDmaOperation(src=dma_src, dest=dma_dst)


        #weights = NpuAddressRange(region=SRAM_SCRATCH_REGION, address=WEIGHT_ADDR, length=len(weight_byte_arr))
        #biases = NpuAddressRange(region=SRAM_SCRATCH_REGION, address=BIAS_ADDR, length=len(bias_byte_arr))
        
    #else:

    dma_op = None

    weights = NpuAddressRange(region=weight_tensor.region,
                              address=weight_tensor.address,
                              length=len(weight_byte_arr))
    biases = NpuAddressRange(region=bias_tensor.region,
                             address=bias_tensor.address,
                             length=len(bias_byte_arr))




    

    padding = NpuPadding(top=0, left=0, bottom=0, right=0)



    


    activation = create_activation(
        activation_op=activation_op,
        min_val=activation_max_val,
        max_val=activation_min_val,
    )

    





    my_op.kernel            =   kernel
    my_op.weights           =   [weights]
    my_op.biases            =   [biases]
    my_op.padding           =   padding
    my_op.activation        =   activation

    my_op.block_config      =   block_config
    my_op.rounding_mode     =   rounding_mode
    my_op.fused_quantize    =   fused_quantize
    my_op.ifm_upscale       =   ifm_upscale
    my_op.accumulator_type  =   accumulator
    my_op.block_traversal   =   block_traversal


    #check_block_config_legal(block_config, my_op, ACCELERATOR)




    # Add the tensors to mem_alloc for storage
    


    return my_op, weight_byte_arr, bias_byte_arr, 





# Create Dicts for writing to C files
def generate_dict_for_writing_defines_to_C_files(cms_name, mem_alloc: MemoryAllocator, 
                                                 is_last_layer, input_size, output_size, 
                                                 weight_byte_arr, bias_byte_arr):


    sizes_dict = {

        cms_name.upper()+"_IS_LAST_LAYER"           : int(is_last_layer),    #num tensors in sram scratchpad
        #cms_name.upper()+"_MAX_NUM_TENSORS_TO_TRACK"   : len(mem_alloc.regions["SRAM_SCRATCH_REGION"].memory_map),    #num tensors in sram scratchpad
        cms_name.upper()+"_MAX_NUM_TENSORS_TO_TRACK"   : 8,    #num tensors in sram scratchpad

        cms_name.upper()+"_TENSOR_ARENA_SIZE "  : mem_alloc.regions["SRAM_SCRATCH_REGION"].size,

        cms_name.upper()+"_INPUT_LAYER_SIZE "   : input_size,             
        cms_name.upper()+"_OUTPUT_LAYER_SIZE "  : output_size,

        cms_name.upper()+"_WEIGHT_LEN" : len(weight_byte_arr),
        cms_name.upper()+"_BIAS_LEN" : len(bias_byte_arr)        
    }


    addr_dict = {}

    #temporary
    addr_dict[cms_name.upper()+"_OUT_SPK_SUM_ADDR"] = -1

    for region_name, region in mem_alloc.regions.items():
        for fm_name, fm_addr in region.memory_map.items():
            addr_dict[cms_name.upper()+"_"+fm_name.upper()+"_ADDR"] = fm_addr


        
    quant_param_dict = {}

    #Just temporary
    addr_dict[cms_name.upper()+"_OUT_SPK_SUM_SCALE"] = -1
    addr_dict[cms_name.upper()+"_OUT_SPK_SUM_ZERO_POINT"] = -1

    for fm in mem_alloc.tensors:
        quant_param_dict[cms_name.upper()+"_"+fm.name.upper()+"_SCALE"] = fm.quantization.scale_f32
        quant_param_dict[cms_name.upper()+"_"+fm.name.upper()+"_ZERO_POINT"] = fm.quantization.zero_point


    return sizes_dict, addr_dict, quant_param_dict









def gen_fc_lif(INPUT_LAYER_SIZE, OUTPUT_LAYER_SIZE, 
        weights_volume_ohwi, bias_list, beta_list, vth_list,
        cms_name, weights_and_biases_on_sram, lif_params_on_sram, is_last_layer, NUM_TIME_STEPS,

        TIME_NOT_UPDATED_MAX_VAL,
        TIME_NOT_UPDATED_MIN_VAL,

        IN_CURR_MAX_VAL,
        IN_CURR_MIN_VAL,

        V_MEM_MAX_VAL,
        V_MEM_MIN_VAL,

        DECAY_ACC_MAX_VAL,
        DECAY_ACC_MIN_VAL,

        DECAY_MAX_VAL,
        DECAY_MIN_VAL,

        DECAYED_MEM_MAX_VAL,
        DECAYED_MEM_MIN_VAL,
         
        DEBUG_MODE, ACCELERATOR, header_out_filepath):




    '''
    Set FM Quantization Params
    '''


    #TIME_NOT_UPDATED_MAX_VAL = 16
    #TIME_NOT_UPDATED_MIN_VAL = 0

    #IN_CURR_MAX_VAL = 9
    #IN_CURR_MIN_VAL = -9

    #V_MEM_MAX_VAL = 9
    #V_MEM_MIN_VAL = -6

    #DECAY_ACC_MAX_VAL = 0
    #DECAY_ACC_MIN_VAL = -1
    #DECAY_MAX_VAL = 0.95
    #DECAY_MIN_VAL = 0

    #DECAYED_MEM_MAX_VAL = 7
    #DECAYED_MEM_MIN_VAL = -4

    # Must be  >= 0
    MIN_DIFF = 0.1


    ###Must be symmetric
    #max_weight_val = np.max(weights_volume_ohwi)
    #min_weight_val = np.min(weights_volume_ohwi)
    ## Get the one with the largest absolute value (since the npu only supports symmetric quantization for weights)
    #if abs(min_weight_val) > abs(max_weight_val):
        #largest_weight_abs_val = abs(min_weight_val)
    #else:
        #largest_weight_abs_val = abs(max_weight_val)

    #WEIGHT_MAX_VAL = largest_weight_abs_val + MIN_DIFF 
    #WEIGHT_MIN_VAL = -(largest_weight_abs_val + MIN_DIFF)






    # Take the natural log of each value
    ln_beta_values = [math.log(v) for v in beta_list]
    max_ln_beta_value = max(ln_beta_values)
    min_ln_beta_value =  min(ln_beta_values)

    LN_BETA_MAX_VAL = max_ln_beta_value
    LN_BETA_MIN_VAL = min_ln_beta_value

    if (max_ln_beta_value == min_ln_beta_value):
        LN_BETA_MAX_VAL += MIN_DIFF
        LN_BETA_MIN_VAL -= MIN_DIFF
    




    max_vth_value = max(vth_list)
    min_vth_value = min(vth_list)
    VTH_MAX_VAL = max_vth_value
    VTH_MIN_VAL = min_vth_value

    if (min_vth_value == max_vth_value):
        VTH_MAX_VAL += MIN_DIFF
        VTH_MIN_VAL -= MIN_DIFF





    if (is_last_layer):
        # MAX VAL == NUM_TIME_STEPS
        OUT_SPK_SUM_MAX_VAL = NUM_TIME_STEPS
        OUT_SPK_SUM_MIN_VAL = 0



    ###########
    # Autoset Params (depends on the previously set quantization params)

    # Must be same for input and output quantization
    IN_SPK_MAX_VAL = 127
    IN_SPK_MIN_VAL = -128
    OUT_SPK_MAX_VAL = 127
    OUT_SPK_MIN_VAL = -128

    # Only need to differentiate between > 0 and < 0
    V_MEM_SUB_VTH_MAX_VAL = 1
    V_MEM_SUB_VTH_MIN_VAL = -1


    # Reset is either 0 or VTH --> same quantization params as VTH
    RESET_MAX_VAL = VTH_MAX_VAL
    RESET_MIN_VAL = 0


    # Only need to differentiate between 0 and anything else
    UPDATE_NXT_LAYER_MAX_VAL = 1
    UPDATE_NXT_LAYER_MIN_VAL = 0

    ###########



    # Need to make sure that in_spk and out_spk have the same quantization parameters
    if (IN_SPK_MAX_VAL != OUT_SPK_MAX_VAL or IN_SPK_MIN_VAL != OUT_SPK_MIN_VAL):
        print("IN_SPK and OUT_SPK do not match")
        exit()






    '''
    Generate Quantization Parameters from the Value range set above for each tensor
    '''
    IN_SPK_SCALE, IN_SPK_ZERO_POINT = zero_point_quant(IN_SPK_MAX_VAL, IN_SPK_MIN_VAL)


    # Layer params
    LN_BETA_SCALE, LN_BETA_ZERO_POINT = zero_point_quant(LN_BETA_MAX_VAL, LN_BETA_MIN_VAL)
    VTH_SCALE, VTH_ZERO_POINT = zero_point_quant(VTH_MAX_VAL, VTH_MIN_VAL)

    # TMP Feature maps
    IN_CURR_SCALE, IN_CURR_ZERO_POINT = zero_point_quant(IN_CURR_MAX_VAL, IN_CURR_MIN_VAL)




    '''
    Quantize and Decrypt Weight, bias and Time constant (beta), and Vth
    '''
    # Generate Weights and Bias list
    weight_byte_arr_init, bias_byte_arr_init = get_int8_fc_weights_and_biases(weights_volume_ohwi, bias_list, INPUT_LAYER_SIZE, OUTPUT_LAYER_SIZE, IN_SPK_SCALE, IN_CURR_SCALE, ACCELERATOR, DEBUG_MODE)


    # Generate LIF Params Quant List
    LN_BETA_QUANT_LIST = generate_ln_beta_values(beta_list=beta_list, ln_beta_scale=LN_BETA_SCALE, ln_beta_zero_point=LN_BETA_ZERO_POINT)    
    VTH_QUANT_LIST = quantize_vth_values(vth_list=vth_list, vth_scale=VTH_SCALE, vth_zero_point=VTH_ZERO_POINT)




    # Assign Memory segments in SRAM Scratch (region 1)
    mem_alloc = MemoryAllocator(INPUT_LAYER_SIZE,
                                OUTPUT_LAYER_SIZE,
                                {
                                "WEIGHTS_AND_BIASES_REGION" : Region(0),
                                "SRAM_SCRATCH_REGION" : Region(1),
                                "PARAMS_REGION" : Region(3),
                                "LUT_REGION" : Region(4),
                                "INPUT_REGION" : Region(5),
                                "OUTPUT_REGION" : Region(6),
                                }
                                )



    mem_alloc.alloc("SRAM_SCRATCH_REGION", "TMP1", OUTPUT_LAYER_SIZE, NpuLayout.NHCWB16)
    mem_alloc.alloc("SRAM_SCRATCH_REGION", "TMP2", OUTPUT_LAYER_SIZE, NpuLayout.NHCWB16)
    mem_alloc.alloc("SRAM_SCRATCH_REGION", "V_MEM", OUTPUT_LAYER_SIZE, NpuLayout.NHCWB16)

    if is_last_layer:
        mem_alloc.alloc("SRAM_SCRATCH_REGION", "OUT_SPK_SUM", OUTPUT_LAYER_SIZE, NpuLayout.NHCWB16)

    mem_alloc.alloc("SRAM_SCRATCH_REGION", "TIME_NOT_UPDATED", 1)
    mem_alloc.alloc("SRAM_SCRATCH_REGION", "UPDATE_NXT_LAYER", 1)

      
    # WEIGHTS AND BIAS REGION
    mem_alloc.alloc("WEIGHTS_AND_BIASES_REGION", "BIAS", len(bias_byte_arr_init))
    mem_alloc.alloc("WEIGHTS_AND_BIASES_REGION", "WEIGHT", len(weight_byte_arr_init))


    #if "OUT_SPK_SUM" in mem_alloc.regions["SRAM_SCRATCH_REGION"].memory_map:
        #OUT_SPK_SUM_ADDR = mem_alloc.regions["SRAM_SCRATCH_REGION"].memory_map["OUT_SPK_SUM"]
    #else:
        #OUT_SPK_SUM_ADDR = -1




    # Assign Memory segments for region 3
    mem_alloc.alloc("PARAMS_REGION", "LN_BETA", OUTPUT_LAYER_SIZE)
    mem_alloc.alloc("PARAMS_REGION", "VTH", OUTPUT_LAYER_SIZE)


    # Assign Memory segments for region 4
    DECAY_LUT_INDEX = 0
    CHECK_SPK_LUT_INDEX = 1



    # Assign Memory segment for region 5
    mem_alloc.alloc("INPUT_REGION", "IN_SPK", INPUT_LAYER_SIZE)

    # Assign Memory segment for region 6
    mem_alloc.alloc("OUTPUT_REGION", "OUT_SPK", OUTPUT_LAYER_SIZE)









    ln_beta_fm = mem_alloc.create_feature_map_v2(
        height=1, width=1, depth=OUTPUT_LAYER_SIZE,
        region_name="PARAMS_REGION",
        segment_name="LN_BETA",
        layout=NpuLayout.NHWC,
        data_type=NpuDataType.INT8,
        max_fm_value=LN_BETA_MAX_VAL,
        min_fm_value=LN_BETA_MIN_VAL,
        is_symmetric_quant=False,
        tensor_name="LN_BETA"
    )


    time_not_updated_fm = mem_alloc.create_feature_map_v2(
        height=1, width=1, depth=1,
        region_name="SRAM_SCRATCH_REGION",
        segment_name="TIME_NOT_UPDATED",
        layout=NpuLayout.NHWC,
        data_type=NpuDataType.INT8,
        max_fm_value=TIME_NOT_UPDATED_MAX_VAL,
        min_fm_value=TIME_NOT_UPDATED_MIN_VAL,
        is_symmetric_quant=False,
        tensor_name="TIME_NOT_UPDATED"
    )

    decay_acc_fm = mem_alloc.create_feature_map_v2(
        height=1, width=1, depth=OUTPUT_LAYER_SIZE,
        region_name="SRAM_SCRATCH_REGION",
        segment_name="TMP1",
        layout=NpuLayout.NHCWB16,
        data_type=NpuDataType.INT8,
        max_fm_value=DECAY_ACC_MAX_VAL,
        min_fm_value=DECAY_ACC_MIN_VAL,
        is_symmetric_quant=False,
        tensor_name="DECAY_ACC"
    )

    in_spk_fm = mem_alloc.create_feature_map_v2(
        height=1, width=1, depth=INPUT_LAYER_SIZE,
        region_name="INPUT_REGION",
        segment_name="IN_SPK",
        layout=NpuLayout.NHWC,
        data_type=NpuDataType.INT8,
        max_fm_value=IN_SPK_MAX_VAL,
        min_fm_value=IN_SPK_MIN_VAL,
        is_symmetric_quant=False,
        tensor_name="IN_SPK"
    )
    

    weight_tensor = mem_alloc.create_weight_and_bias_fm(
        height=1, width=1, depth=len(weight_byte_arr_init),
        region_name="WEIGHTS_AND_BIASES_REGION",
        segment_name="WEIGHT",
        data_type=NpuDataType.INT8,
        tensor_values=weights_volume_ohwi,
        tensor_name="WEIGHT"
    )

    
    bias_tensor = mem_alloc.create_weight_and_bias_fm(
        height=1, width=1, depth=len(bias_byte_arr_init),
        region_name="WEIGHTS_AND_BIASES_REGION",
        segment_name="BIAS",
        data_type=NpuDataType.INT8,
        tensor_values=bias_list,
        tensor_name="WEIGHT"
    )
        
        


    in_curr_fm = mem_alloc.create_feature_map_v2(
        height=1, width=1, depth=OUTPUT_LAYER_SIZE,
        region_name="SRAM_SCRATCH_REGION",
        segment_name="TMP2",
        layout=NpuLayout.NHCWB16,
        data_type=NpuDataType.INT8,
        max_fm_value=IN_CURR_MAX_VAL,
        min_fm_value=IN_CURR_MIN_VAL,
        is_symmetric_quant=False,
        tensor_name="IN_CURR"
    )

    v_mem_fm = mem_alloc.create_feature_map_v2(
        height=1, width=1, depth=OUTPUT_LAYER_SIZE,
        region_name="SRAM_SCRATCH_REGION",
        segment_name="V_MEM",
        layout=NpuLayout.NHWC,
        data_type=NpuDataType.INT8,
        max_fm_value=V_MEM_MAX_VAL,
        min_fm_value=V_MEM_MIN_VAL,
        is_symmetric_quant=False,
        tensor_name="V_MEM"
    )

    decay_fm = mem_alloc.create_feature_map_v2(
        height=1, width=1, depth=OUTPUT_LAYER_SIZE,
        region_name="SRAM_SCRATCH_REGION",
        segment_name="TMP1",
        layout=NpuLayout.NHCWB16,
        #layout=NpuLayout.NHWC,
        data_type=NpuDataType.INT8,
        max_fm_value=DECAY_MAX_VAL,
        min_fm_value=DECAY_MIN_VAL,
        is_symmetric_quant=False,
        tensor_name="DECAY"
    )



    decayed_mem_fm = mem_alloc.create_feature_map_v2(
        height=1, width=1, depth=OUTPUT_LAYER_SIZE,
        region_name="SRAM_SCRATCH_REGION",
        segment_name="TMP1",
        layout=NpuLayout.NHCWB16,
        #layout=NpuLayout.NHWC,
        data_type=NpuDataType.INT8,
        max_fm_value=DECAYED_MEM_MAX_VAL,
        min_fm_value=DECAYED_MEM_MIN_VAL,
        is_symmetric_quant=False,
        tensor_name="DECAYED_MEM"
    )


    vth_fm = mem_alloc.create_feature_map_v2(
        height=1, width=1, depth=OUTPUT_LAYER_SIZE,
        region_name="PARAMS_REGION",
        segment_name="VTH",
        layout=NpuLayout.NHWC,
        data_type=NpuDataType.INT8,
        max_fm_value=VTH_MAX_VAL,
        min_fm_value=VTH_MIN_VAL,
        is_symmetric_quant=False,
        tensor_name="VTH"
    )


    v_mem_sub_vth_fm = mem_alloc.create_feature_map_v2(
        height=1, width=1, depth=OUTPUT_LAYER_SIZE,
        region_name="OUTPUT_REGION",
        segment_name="OUT_SPK",
        layout=NpuLayout.NHWC,
        data_type=NpuDataType.INT8,
        max_fm_value=V_MEM_SUB_VTH_MAX_VAL,
        min_fm_value=V_MEM_SUB_VTH_MIN_VAL,
        is_symmetric_quant=False,
        tensor_name="v_mem_sub_vth"
    )


    out_spk_fm = mem_alloc.create_feature_map_v2(
        height=1, width=1, depth=OUTPUT_LAYER_SIZE,
        region_name="OUTPUT_REGION",
        segment_name="OUT_SPK",
        layout=NpuLayout.NHWC,
        data_type=NpuDataType.INT8,
        max_fm_value=OUT_SPK_MAX_VAL,
        min_fm_value=OUT_SPK_MIN_VAL,
        is_symmetric_quant=False,
        tensor_name="OUT_SPK"
    )



    reset_fm = mem_alloc.create_feature_map_v2(
        height=1, width=1, depth=OUTPUT_LAYER_SIZE,
        region_name="SRAM_SCRATCH_REGION",
        segment_name="TMP2",
        layout=NpuLayout.NHCWB16,
        #layout=NpuLayout.NHWC,
        data_type=NpuDataType.INT8,
        max_fm_value=RESET_MAX_VAL,
        min_fm_value=RESET_MIN_VAL,
        is_symmetric_quant=False,
        tensor_name="RESET"
    )


    update_nxt_layer_fm = mem_alloc.create_feature_map_v2(
        height=1, width=1, depth=1,
        region_name="SRAM_SCRATCH_REGION",
        segment_name="UPDATE_NXT_LAYER",
        layout=NpuLayout.NHWC,
        data_type=NpuDataType.INT8,
        max_fm_value=UPDATE_NXT_LAYER_MAX_VAL,
        min_fm_value=UPDATE_NXT_LAYER_MIN_VAL,
        is_symmetric_quant=False,
        tensor_name="UPDATE_NXT_LAYER"
    )
    


    if is_last_layer:
        out_spk_sum_fm = mem_alloc.create_feature_map_v2(
            height=1, width=1, depth=OUTPUT_LAYER_SIZE,
            region_name="SRAM_SCRATCH_REGION",
            segment_name="OUT_SPK_SUM",
            layout=NpuLayout.NHWC,
            data_type=NpuDataType.INT8,
            max_fm_value=OUT_SPK_SUM_MAX_VAL,
            min_fm_value=OUT_SPK_SUM_MIN_VAL,
            is_symmetric_quant=False,
            tensor_name="OUT_SPK_SUM"
        )








    # Define the individual NPU Operations
    dma_lut_op, exp_mul_lnb_time_op, decay_lut_values, decay_lut_index = get_elementwise_op_with_lut(NpuElementWiseOp.MUL, ln_beta_fm, time_not_updated_fm, decay_acc_fm,
                                                                                                        decay_fm, 
                                                                                                        mem_alloc, 
                                                                                                        DECAY_LUT_INDEX, math.exp, "LUT_REGION",
                                                                                                        ACCELERATOR)


    mul_decay_op = get_elementwise_op(NpuElementWiseOp.MUL, v_mem_fm, decay_fm, decayed_mem_fm, ACCELERATOR)

    add_decayed_mem_in_curr = get_elementwise_op(NpuElementWiseOp.ADD, decayed_mem_fm, in_curr_fm, v_mem_fm, ACCELERATOR)


    def check_positive(x_real):
        if x_real > 0: y_real = 1
        else: y_real = 0
        return y_real

    check_spk_lut_dma_op, check_spk_sub_v_mem_updated_vth, check_spk_lut_values, check_spk_lut_index = get_elementwise_op_with_lut(NpuElementWiseOp.SUB, v_mem_fm, vth_fm, v_mem_sub_vth_fm,
                                                                                                                                    out_spk_fm,
                                                                                                                                    mem_alloc,
                                                                                                                                    CHECK_SPK_LUT_INDEX,
                                                                                                                                    check_positive,
                                                                                                                                    "LUT_REGION",
                                                                                                                                    ACCELERATOR)

    reset_mul_vth_out_spk_op = get_elementwise_op(NpuElementWiseOp.MUL, vth_fm, out_spk_fm, reset_fm, ACCELERATOR)

    sub_v_mem_reset_op = get_elementwise_op(NpuElementWiseOp.SUB, v_mem_fm, reset_fm, v_mem_fm, ACCELERATOR)

    update_nxt_layer_reduce_sum_out_spk = get_pooling_op(NpuPoolingOp.REDUCE_SUM, out_spk_fm, update_nxt_layer_fm, kernel_size=(1, 1), accelerator=ACCELERATOR)

    reset_time_op = get_elementwise_op(NpuElementWiseOp.MUL, time_not_updated_fm, None, time_not_updated_fm, ACCELERATOR, ifm2_scalar=0)





    fully_connected_op, weight_byte_arr, bias_byte_arr = get_fully_connected_op(in_spk_fm, in_curr_fm, 
                                                                                weight_tensor, bias_tensor,
                                                                                #weights_volume_ohwi, bias_list, 
                                                                                mem_alloc, 
                                                                                #"WEIGHTS_AND_BIASES_REGION", "WEIGHT",
                                                                                #"WEIGHTS_AND_BIASES_REGION", "BIAS",
                                                                                ACCELERATOR)
        

    npu_op_list = [dma_lut_op, exp_mul_lnb_time_op, fully_connected_op, mul_decay_op, add_decayed_mem_in_curr, check_spk_lut_dma_op, check_spk_sub_v_mem_updated_vth, reset_mul_vth_out_spk_op, sub_v_mem_reset_op, update_nxt_layer_reduce_sum_out_spk, reset_time_op]



    if (is_last_layer):
        incr_out_spk_sum_op = get_elementwise_op(NpuElementWiseOp.ADD, out_spk_sum_fm,  out_spk_fm, out_spk_sum_fm, ACCELERATOR)
        npu_op_list.append(incr_out_spk_sum_op)




    '''
    Wrap up and prepare for writing to C header files
    '''

    # Merge
    lut_arr_contents_str = merge_lut_values_to_str([(decay_lut_values, decay_lut_index), (check_spk_lut_values, check_spk_lut_index)])
    lif_params_arr_contents_str = merge_lif_params_to_str(LN_BETA_QUANT_LIST, VTH_QUANT_LIST)
    cms_bytearr, register_cms = gen_cms(npu_op_list, ACCELERATOR, DEBUG_MODE)

    # Generate Dicts for writing to C
    sizes_dict, addr_dict, quant_param_dict = generate_dict_for_writing_defines_to_C_files(cms_name=cms_name, mem_alloc=mem_alloc, is_last_layer=is_last_layer,
                                                                                            input_size=INPUT_LAYER_SIZE, output_size=OUTPUT_LAYER_SIZE,
                                                                                            weight_byte_arr=weight_byte_arr, bias_byte_arr=bias_byte_arr)

    write_cms_to_files(header_out_filepath, lif_params_on_sram, cms_bytearr, register_cms, cms_name, sizes_dict, addr_dict, quant_param_dict, lif_params_arr_contents_str, lut_arr_contents_str, weight_byte_arr=weight_byte_arr, bias_byte_arr=bias_byte_arr)
    




