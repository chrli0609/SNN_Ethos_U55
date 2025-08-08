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


class MemoryAllocator:

    def __init__(self,  input_size:     int, 
                        output_size:    int,
                        regions:        Dict[str, Region]
                ):
        
        self.input_size = input_size
        self.output_size = output_size

        # Check that there are no invalid region numbers
        for key, value in regions.items():
            if value.number < 0 or value.number > 7:
                raise Exception(f"Region '{key}' has out-of-range value: {value.number}, expected value between 0 and 7")

        self.regions = regions


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



        
        

def create_feature_map_v2(mem_alloc: MemoryAllocator,
                      height: int, width: int, depth: int,
                      region_name: str,
                      tensor_name: str,
                      layout,  # Pass NpuLayout.NHWC or similar
                      data_type,  # Pass NpuDataType.INT8 or similar
                      #fm_elem_size: int,
                      max_fm_value,
                      min_fm_value,
                      is_symmetric_quant
                    ) -> NpuFeatureMap:
    """
    Create an NpuFeatureMap with the given parameters.
    """
    fm = NpuFeatureMap()
    fm.shape = NpuShape3D(height=height, width=width, depth=depth)


    fm.region = mem_alloc.regions[region_name].number
    
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
    

    fm_addr = mem_alloc.regions[region_name].memory_map[tensor_name]
    # Default tile setup for single tile (most common case)
    fm.tiles = NpuTileBox(
        height_0=height, 
        height_1=0, 
        width_0=width, 
        addresses=[fm_addr, 0, 0, 0]
    )
    
    fm.name = tensor_name
    
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


    #ifm = create_feature_map(
        #height=1, width=1, depth=OUTPUT_LAYER_SIZE,
        #region=OUTPUT_REGION,
        #layout=NpuLayout.NHWC,
        #data_type=NpuDataType.INT8, fm_elem_size=1,
        #fm_addr=OUT_SPK_ADDR,
        #scale=OUT_SPK_SCALE, zero_point=OUT_SPK_ZERO_POINT,
        #name="out_spk"
    #)

    #ofm = create_feature_map(
        #height=1, width=1, depth=1,
        #region=SRAM_SCRATCH_REGION,
        #layout=NpuLayout.NHWC,
        #data_type=NpuDataType.INT8, fm_elem_size=1,
        #fm_addr=UPDATE_NXT_LAYER_ADDR,
        #scale=UPDATE_NXT_LAYER_SCALE,
        #zero_point=UPDATE_NXT_LAYER_ZERO_POINT,
        #name="update_nxt_layer"
    #)


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





def get_fully_connected_op(ifm, ofm, weights_volume_ohwi, bias_list, 
                           mem_alloc,
                           weight_region_name, weight_fm_name,
                           bias_region_name, bias_fm_name,
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
    max_weight_val = np.max(weights_volume_ohwi)
    min_weight_val = np.min(weights_volume_ohwi)
    # Get the one with the largest absolute value (since the npu only supports symmetric quantization for weights)
    if abs(min_weight_val) > abs(max_weight_val):
        largest_weight_abs_val = abs(min_weight_val)
    else:
        largest_weight_abs_val = abs(max_weight_val)
    weight_scale, weight_zero_point = symmetric_zero_point_quant(largest_weight_abs_val, -largest_weight_abs_val)


    weight_byte_arr, bias_byte_arr = gen_weights_and_biases(accelerator=accelerator,
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

    weights = NpuAddressRange(region=mem_alloc.regions[weight_region_name].number,
                              address=mem_alloc.regions[weight_region_name].memory_map[weight_fm_name],
                              length=len(weight_byte_arr))
    biases = NpuAddressRange(region=mem_alloc.regions[bias_region_name].number,
                             address=mem_alloc.regions[bias_region_name].memory_map[bias_fm_name],
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





    return my_op, weight_byte_arr, bias_byte_arr, 








if __name__ == "__main__":
    output_size = 48
    mem_alloc = MemoryAllocator(48, output_size, 10, 100, 
            {"WEIGHT_AND_BIAS": Region(0), 
             "SRAM_SCRATCH": Region(1), 
             "PARAMS": Region(3), 
             "LUT": Region(4), 
             "INPUT": Region(5), 
             "OUTPUT": Region(6),
             })
    
    mem_alloc.alloc("SRAM_SCRATCH", "TMP1", output_size)
    mem_alloc.alloc("SRAM_SCRATCH", "TMP2", output_size)
    mem_alloc.alloc("SRAM_SCRATCH", "V_MEM_ADDR", output_size)
    mem_alloc.alloc("SRAM_SCRATCH", "TIME_NOT_UPDATED", 1)
    mem_alloc.alloc("SRAM_SCRATCH", "UPDATE_NXT_LAYER", 1)


    print("regions", mem_alloc.regions)



    #create_feature_map_v2(mem_alloc,
                        #1, 1, 48,
                        #"SRAM_SCRATCH",
                        #"TMP1",
                        #NpuLayout.NHWC,
                        #TIME_NOT_UPDATED_MAX_VAL,
                        #TIME_NOT_UPDATED_MIN_vAL,
                        #"time_not_updated"
                        #)
    




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

        # Assign Memory Regions (0 - 7)



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

    # Must be same for input and output quantization
    IN_SPK_MAX_VAL = 127
    IN_SPK_MIN_VAL = -128
    OUT_SPK_MAX_VAL = 127
    OUT_SPK_MIN_VAL = -128

    # Only need to differentiate between > 0 and < 0
    V_MEM_SUB_VTH_MAX_VAL = 1
    V_MEM_SUB_VTH_MIN_VAL = -1

    ##Must be symmetric
    max_weight_val = np.max(weights_volume_ohwi)
    min_weight_val = np.min(weights_volume_ohwi)
    # Get the one with the largest absolute value (since the npu only supports symmetric quantization for weights)
    if abs(min_weight_val) > abs(max_weight_val):
        largest_weight_abs_val = abs(min_weight_val)
    else:
        largest_weight_abs_val = abs(max_weight_val)

    WEIGHT_MAX_VAL = largest_weight_abs_val + MIN_DIFF 
    WEIGHT_MIN_VAL = -(largest_weight_abs_val + MIN_DIFF)






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

    # Layer status
    V_MEM_SCALE, V_MEM_ZERO_POINT = zero_point_quant(V_MEM_MAX_VAL, V_MEM_MIN_VAL)
    TIME_NOT_UPDATED_SCALE, TIME_NOT_UPDATED_ZERO_POINT = zero_point_quant(TIME_NOT_UPDATED_MAX_VAL, TIME_NOT_UPDATED_MIN_VAL)

    # TMP Feature maps
    DECAY_SCALE, DECAY_ZERO_POINT = zero_point_quant(DECAY_MAX_VAL, DECAY_MIN_VAL)
    DECAY_ACC_SCALE, DECAY_ACC_ZERO_POINT = zero_point_quant(DECAY_ACC_MAX_VAL, DECAY_ACC_MIN_VAL)
    IN_CURR_SCALE, IN_CURR_ZERO_POINT = zero_point_quant(IN_CURR_MAX_VAL, IN_CURR_MIN_VAL)
    DECAYED_MEM_SCALE, DECAYED_MEM_ZERO_POINT = zero_point_quant(DECAYED_MEM_MAX_VAL, DECAYED_MEM_MIN_VAL)
    V_MEM_SUB_VTH_SCALE, V_MEM_SUB_VTH_ZERO_POINT = zero_point_quant(V_MEM_SUB_VTH_MAX_VAL, V_MEM_SUB_VTH_MIN_VAL)


    RESET_SCALE, RESET_ZERO_POINT = zero_point_quant(RESET_MAX_VAL, RESET_MIN_VAL)

    WEIGHT_SCALE, WEIGHT_ZERO_POINT = symmetric_zero_point_quant(WEIGHT_MAX_VAL, WEIGHT_MIN_VAL)
    BIAS_SCALE, BIAS_ZERO_POINT = IN_SPK_SCALE*WEIGHT_SCALE, 0
    # Bias at most 40 bit value
    #print("BIAS:\n\tMax val:", 2**39 / BIAS_SCALE, "\n\tMin val:", -2**39 / BIAS_SCALE)

    # Output Feature Map
    UPDATE_NXT_LAYER_SCALE, UPDATE_NXT_LAYER_ZERO_POINT = zero_point_quant(UPDATE_NXT_LAYER_MAX_VAL, UPDATE_NXT_LAYER_MIN_VAL)
    OUT_SPK_SCALE, OUT_SPK_ZERO_POINT = zero_point_quant(OUT_SPK_MAX_VAL, OUT_SPK_MIN_VAL)

    if (is_last_layer):
        OUT_SPK_SUM_SCALE, OUT_SPK_SUM_ZERO_POINT = zero_point_quant(OUT_SPK_SUM_MAX_VAL, OUT_SPK_SUM_MIN_VAL)
    else:
        OUT_SPK_SUM_SCALE, OUT_SPK_SUM_ZERO_POINT = 0, 0




    '''
    Quantize and Decrypt Weight, bias and Time constant (beta), and Vth
    '''
    # Generate Weights and Bias list
    weight_byte_arr_init, bias_byte_arr_init = get_int8_fc_weights_and_biases(weights_volume_ohwi, bias_list, INPUT_LAYER_SIZE, OUTPUT_LAYER_SIZE, IN_SPK_SCALE, IN_CURR_SCALE, ACCELERATOR, DEBUG_MODE)


    # Generate LIF Params Quant List
    LN_BETA_QUANT_LIST = generate_ln_beta_values(beta_list=beta_list, ln_beta_scale=LN_BETA_SCALE, ln_beta_zero_point=LN_BETA_ZERO_POINT)    
    VTH_QUANT_LIST = quantize_vth_values(vth_list=vth_list, vth_scale=VTH_SCALE, vth_zero_point=VTH_ZERO_POINT)



    # Can at most only ever be 8 tensors we have to track on SRAM Sratch region: bias, weight, tmp1, tmp2, v_mem, out_spk_sum, time_not_updataed, update_nxt_layer
    MAX_NUM_TENSORS_TO_TRACK = 8

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


    WEIGHT_AND_BIASES_REGION = mem_alloc.regions["WEIGHTS_AND_BIASES_REGION"].number
    SRAM_SCRATCH_REGION = mem_alloc.regions["SRAM_SCRATCH_REGION"].number
    PARAMS_REGION = mem_alloc.regions["PARAMS_REGION"].number
    LUT_REGION = mem_alloc.regions["LUT_REGION"].number
    INPUT_REGION = mem_alloc.regions["INPUT_REGION"].number
    OUTPUT_REGION = mem_alloc.regions["OUTPUT_REGION"].number

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



    TMP1_ADDR = mem_alloc.regions["SRAM_SCRATCH_REGION"].memory_map["TMP1"]
    TMP2_ADDR = mem_alloc.regions["SRAM_SCRATCH_REGION"].memory_map["TMP2"]
    V_MEM_ADDR = mem_alloc.regions["SRAM_SCRATCH_REGION"].memory_map["V_MEM"]


    if "OUT_SPK_SUM" in mem_alloc.regions["SRAM_SCRATCH_REGION"].memory_map:
        OUT_SPK_SUM_ADDR = mem_alloc.regions["SRAM_SCRATCH_REGION"].memory_map["OUT_SPK_SUM"]
    else:
        OUT_SPK_SUM_ADDR = -1

    TIME_NOT_UPDATED_ADDR = mem_alloc.regions["SRAM_SCRATCH_REGION"].memory_map["TIME_NOT_UPDATED"]
    UPDATE_NXT_LAYER_ADDR = mem_alloc.regions["SRAM_SCRATCH_REGION"].memory_map["UPDATE_NXT_LAYER"]


    BIAS_ADDR = mem_alloc.regions["WEIGHTS_AND_BIASES_REGION"].memory_map["BIAS"]
    WEIGHT_ADDR = mem_alloc.regions["WEIGHTS_AND_BIASES_REGION"].memory_map["WEIGHT"]

    TENSOR_ARENA_SIZE = mem_alloc.regions["SRAM_SCRATCH_REGION"].size



        #TENSOR_ARENA_SIZE	=	3*next_multiple(OUTPUT_LAYER_SIZE, 16) + 2 

    def is_16_bit_aligned(val):
        if val == next_multiple(val, 16):
            return True
        else:
            return False

    print("====================================================")
    print("TMP1", TMP1_ADDR, is_16_bit_aligned(TMP1_ADDR))
    print("TMP2", TMP2_ADDR, is_16_bit_aligned(TMP2_ADDR))
    print("V_MEM", V_MEM_ADDR, is_16_bit_aligned(V_MEM_ADDR))
    print("OUT_SPK_SUM", OUT_SPK_SUM_ADDR, is_16_bit_aligned(OUT_SPK_SUM_ADDR))
    print("TIME_NOT_UPDATED:", TIME_NOT_UPDATED_ADDR, is_16_bit_aligned(TIME_NOT_UPDATED_ADDR))
    print("UPDATE_NXT_LAYER:", UPDATE_NXT_LAYER_ADDR, is_16_bit_aligned(UPDATE_NXT_LAYER_ADDR))
    print("----------------------------------------------------")
    print("TENSOR_ARENA_SIZE:", TENSOR_ARENA_SIZE, is_16_bit_aligned(TENSOR_ARENA_SIZE))
    print("====================================================")





    ##############
    # Assign Tmp tensors here!
    #DECAY_ADDR = TMP1_ADDR
    #IN_CURR_ADDR = TMP2_ADDR
    #DECAYED_MEM_ADDR = TMP1_ADDR
    #RESET_ADDR = TMP2_ADDR

    ##############



    # Assign Memory segments for region 3
    mem_alloc.alloc("PARAMS_REGION", "LN_BETA", OUTPUT_LAYER_SIZE)
    mem_alloc.alloc("PARAMS_REGION", "VTH", OUTPUT_LAYER_SIZE)

    #LN_BETA_ADDR = mem_alloc.regions["PARAMS_REGION"].memory_map["LN_BETA"]
    #VTH_ADDR = mem_alloc.regions["PARAMS_REGION"].memory_map["VTH"]

    #LN_BETA_ADDR = 0
    #VTH_ADDR = LN_BETA_ADDR + OUTPUT_LAYER_SIZE


    # Assign Memory segments for region 4

    DECAY_LUT_INDEX = 0
    CHECK_SPK_LUT_INDEX = 1



    # Assign Memory segment for region 5
    mem_alloc.alloc("INPUT_REGION", "IN_SPK", INPUT_LAYER_SIZE)
    #IN_SPK_ADDR = mem_alloc.regions["INPUT_REGION"].memory_map["IN_SPK"]
    #IN_SPK_ADDR = 0

    # Assign Memory segment for region 6
    mem_alloc.alloc("OUTPUT_REGION", "OUT_SPK", OUTPUT_LAYER_SIZE)
    #OUT_SPK_ADDR = mem_alloc.regions["OUTPUT_REGION"].memory_map["OUT_SPK"]
    #OUT_SPK_ADDR = 0









    ln_beta_fm = create_feature_map_v2(mem_alloc,
        height=1, width=1, depth=OUTPUT_LAYER_SIZE,
        region_name="PARAMS_REGION",
        tensor_name="LN_BETA",
        layout=NpuLayout.NHWC,
        data_type=NpuDataType.INT8,
        max_fm_value=LN_BETA_MAX_VAL,
        min_fm_value=LN_BETA_MIN_VAL,
        is_symmetric_quant=False
    )


    time_not_updated_fm = create_feature_map_v2(mem_alloc,
        height=1, width=1, depth=1,
        region_name="SRAM_SCRATCH_REGION",
        tensor_name="TIME_NOT_UPDATED",
        layout=NpuLayout.NHWC,
        data_type=NpuDataType.INT8,
        max_fm_value=TIME_NOT_UPDATED_MAX_VAL,
        min_fm_value=TIME_NOT_UPDATED_MIN_VAL,
        is_symmetric_quant=False
    )

    decay_acc_fm = create_feature_map_v2(mem_alloc,
        height=1, width=1, depth=OUTPUT_LAYER_SIZE,
        region_name="SRAM_SCRATCH_REGION",
        tensor_name="TMP1",
        layout=NpuLayout.NHCWB16,
        data_type=NpuDataType.INT8,
        max_fm_value=DECAY_ACC_MAX_VAL,
        min_fm_value=DECAY_ACC_MIN_VAL,
        is_symmetric_quant=False
    )

    in_spk_fm = create_feature_map_v2(mem_alloc,
        height=1, width=1, depth=INPUT_LAYER_SIZE,
        region_name="INPUT_REGION",
        tensor_name="IN_SPK",
        layout=NpuLayout.NHWC,
        data_type=NpuDataType.INT8,
        max_fm_value=IN_SPK_MAX_VAL,
        min_fm_value=IN_SPK_MIN_VAL,
        is_symmetric_quant=False

    )


    in_curr_fm = create_feature_map_v2(mem_alloc,
        height=1, width=1, depth=OUTPUT_LAYER_SIZE,
        region_name="SRAM_SCRATCH_REGION",
        tensor_name="TMP2",
        layout=NpuLayout.NHCWB16,
        data_type=NpuDataType.INT8,
        max_fm_value=IN_CURR_MAX_VAL,
        min_fm_value=IN_CURR_MIN_VAL,
        is_symmetric_quant=False
    )

    v_mem_fm = create_feature_map_v2(mem_alloc,
        height=1, width=1, depth=OUTPUT_LAYER_SIZE,
        region_name="SRAM_SCRATCH_REGION",
        tensor_name="V_MEM",
        layout=NpuLayout.NHWC,
        data_type=NpuDataType.INT8,
        max_fm_value=V_MEM_MAX_VAL,
        min_fm_value=V_MEM_MIN_VAL,
        is_symmetric_quant=False
    )

    decay_fm = create_feature_map_v2(mem_alloc,
        height=1, width=1, depth=OUTPUT_LAYER_SIZE,
        region_name="SRAM_SCRATCH_REGION",
        tensor_name="TMP1",
        layout=NpuLayout.NHCWB16,
        #layout=NpuLayout.NHWC,
        data_type=NpuDataType.INT8,
        max_fm_value=DECAY_MAX_VAL,
        min_fm_value=DECAY_MIN_VAL,
        is_symmetric_quant=False
    )



    decayed_mem_fm = create_feature_map_v2(mem_alloc,
        height=1, width=1, depth=OUTPUT_LAYER_SIZE,
        region_name="SRAM_SCRATCH_REGION",
        tensor_name="TMP1",
        layout=NpuLayout.NHCWB16,
        #layout=NpuLayout.NHWC,
        data_type=NpuDataType.INT8,
        max_fm_value=DECAYED_MEM_MAX_VAL,
        min_fm_value=DECAYED_MEM_MIN_VAL,
        is_symmetric_quant=False
    )


    vth_fm = create_feature_map_v2(mem_alloc,
        height=1, width=1, depth=OUTPUT_LAYER_SIZE,
        region_name="PARAMS_REGION",
        tensor_name="VTH",
        layout=NpuLayout.NHWC,
        data_type=NpuDataType.INT8,
        max_fm_value=VTH_MAX_VAL,
        min_fm_value=VTH_MIN_VAL,
        is_symmetric_quant=False
    )


    v_mem_sub_vth_fm = create_feature_map_v2(mem_alloc,
        height=1, width=1, depth=OUTPUT_LAYER_SIZE,
        region_name="OUTPUT_REGION",
        tensor_name="OUT_SPK",
        layout=NpuLayout.NHWC,
        data_type=NpuDataType.INT8,
        max_fm_value=V_MEM_SUB_VTH_MAX_VAL,
        min_fm_value=V_MEM_SUB_VTH_MIN_VAL,
        is_symmetric_quant=False
    )


    out_spk_fm = create_feature_map_v2(mem_alloc,
        height=1, width=1, depth=OUTPUT_LAYER_SIZE,
        region_name="OUTPUT_REGION",
        tensor_name="OUT_SPK",
        layout=NpuLayout.NHWC,
        data_type=NpuDataType.INT8,
        max_fm_value=OUT_SPK_MAX_VAL,
        min_fm_value=OUT_SPK_MIN_VAL,
        is_symmetric_quant=False
    )



    reset_fm = create_feature_map_v2(mem_alloc,
        height=1, width=1, depth=OUTPUT_LAYER_SIZE,
        region_name="SRAM_SCRATCH_REGION",
        tensor_name="TMP2",
        layout=NpuLayout.NHCWB16,
        #layout=NpuLayout.NHWC,
        data_type=NpuDataType.INT8,
        max_fm_value=RESET_MAX_VAL,
        min_fm_value=RESET_MIN_VAL,
        is_symmetric_quant=False
    )


    update_nxt_layer_fm = create_feature_map_v2(mem_alloc,
        height=1, width=1, depth=1,
        region_name="SRAM_SCRATCH_REGION",
        tensor_name="UPDATE_NXT_LAYER",
        layout=NpuLayout.NHWC,
        data_type=NpuDataType.INT8,
        max_fm_value=UPDATE_NXT_LAYER_MAX_VAL,
        min_fm_value=UPDATE_NXT_LAYER_MIN_VAL,
        is_symmetric_quant=False
    )
    


    if is_last_layer:
        out_spk_sum_fm = create_feature_map_v2(mem_alloc,
            height=1, width=1, depth=OUTPUT_LAYER_SIZE,
            region_name="SRAM_SCRATCH_REGION",
            tensor_name="OUT_SPK_SUM",
            layout=NpuLayout.NHWC,
            data_type=NpuDataType.INT8,
            max_fm_value=OUT_SPK_SUM_MAX_VAL,
            min_fm_value=OUT_SPK_SUM_MIN_VAL,
            is_symmetric_quant=False
        )




    # Create Dicts for writing to C files
    def generate_dict_for_writing_defines_to_C_files(cms_name, weight_byte_arr, bias_byte_arr):


        sizes_dict = {

            cms_name.upper()+"_IS_LAST_LAYER"           : int(is_last_layer),    #num tensors in sram scratchpad
            cms_name.upper()+"_MAX_NUM_TENSORS_TO_TRACK"   : MAX_NUM_TENSORS_TO_TRACK,    #num tensors in sram scratchpad

            cms_name.upper()+"_TENSOR_ARENA_SIZE "  : TENSOR_ARENA_SIZE,
            cms_name.upper()+"_INPUT_LAYER_SIZE "   : INPUT_LAYER_SIZE,             
            cms_name.upper()+"_OUTPUT_LAYER_SIZE "  : OUTPUT_LAYER_SIZE,

            cms_name.upper()+"_WEIGHT_LEN" : len(weight_byte_arr),
            cms_name.upper()+"_BIAS_LEN" : len(bias_byte_arr)        
        }


        addr_dict = {}
        for region_name, region in mem_alloc.regions.items():
            for fm_name, fm_addr in region.memory_map.items():
                addr_dict[cms_name.upper()+"_"+fm_name.upper()+"_ADDR"] = fm_addr




        quant_param_dict = {
            cms_name.upper()+"_IN_SPK_SCALE" : IN_SPK_SCALE,
            cms_name.upper()+"_IN_SPK_ZERO_POINT" : IN_SPK_ZERO_POINT,


            cms_name.upper()+"_BIAS_SCALE" : BIAS_SCALE,
            cms_name.upper()+"_BIAS_ZERO_POINT" : BIAS_ZERO_POINT,
            cms_name.upper()+"_WEIGHT_SCALE" : WEIGHT_SCALE,
            cms_name.upper()+"_WEIGHT_ZERO_POINT" : WEIGHT_ZERO_POINT,


            cms_name.upper()+"_LN_BETA_SCALE" : LN_BETA_SCALE,
            cms_name.upper()+"_LN_BETA_ZERO_POINT" : LN_BETA_ZERO_POINT,
            cms_name.upper()+"_VTH_SCALE" : VTH_SCALE,
            cms_name.upper()+"_VTH_ZERO_POINT" : VTH_ZERO_POINT,
            cms_name.upper()+"_V_MEM_SCALE" : V_MEM_SCALE,
            cms_name.upper()+"_V_MEM_ZERO_POINT" : V_MEM_ZERO_POINT,
            cms_name.upper()+"_TIME_NOT_UPDATED_SCALE" : TIME_NOT_UPDATED_SCALE,
            cms_name.upper()+"_TIME_NOT_UPDATED_ZERO_POINT" : TIME_NOT_UPDATED_ZERO_POINT,

            cms_name.upper()+"_DECAY_SCALE" : DECAY_SCALE,
            cms_name.upper()+"_DECAY_ZERO_POINT" : DECAY_ZERO_POINT,
            cms_name.upper()+"_IN_CURR_SCALE" : IN_CURR_SCALE,
            cms_name.upper()+"_IN_CURR_ZERO_POINT" : IN_CURR_ZERO_POINT,
            cms_name.upper()+"_DECAYED_MEM_SCALE" : DECAYED_MEM_SCALE,
            cms_name.upper()+"_DECAYED_MEM_ZERO_POINT" : DECAYED_MEM_ZERO_POINT,


            # Output
            cms_name.upper()+"_UPDATE_NXT_LAYER_SCALE" : UPDATE_NXT_LAYER_SCALE,
            cms_name.upper()+"_UPDATE_NXT_LAYER_ZERO_POINT" : UPDATE_NXT_LAYER_ZERO_POINT,
            cms_name.upper()+"_OUT_SPK_SCALE" : OUT_SPK_SCALE,
            cms_name.upper()+"_OUT_SPK_ZERO_POINT" : OUT_SPK_ZERO_POINT,
        }


        if not is_last_layer:
            addr_dict[cms_name.upper()+"_OUT_SPK_SUM_ADDR"] = OUT_SPK_SUM_ADDR
        quant_param_dict[cms_name.upper()+"_OUT_SPK_SUM_SCALE"] = OUT_SPK_SUM_SCALE
        quant_param_dict[cms_name.upper()+"_OUT_SPK_SUM_ZERO_POINT"] = OUT_SPK_SUM_ZERO_POINT

        return sizes_dict, addr_dict, quant_param_dict




    def layer_merge_and_write(cms_name, header_out_filepath, lif_params_on_sram):

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





        fully_connected_op, weight_byte_arr, bias_byte_arr = get_fully_connected_op(in_spk_fm, in_curr_fm, weights_volume_ohwi, bias_list, 
                                                                                    mem_alloc, 
                                                                                    "WEIGHTS_AND_BIASES_REGION", "WEIGHT",
                                                                                    "WEIGHTS_AND_BIASES_REGION", "BIAS",
                                                                                    ACCELERATOR)
        

        npu_op_list = [dma_lut_op, exp_mul_lnb_time_op, fully_connected_op, mul_decay_op, add_decayed_mem_in_curr, check_spk_lut_dma_op, check_spk_sub_v_mem_updated_vth, reset_mul_vth_out_spk_op, sub_v_mem_reset_op, update_nxt_layer_reduce_sum_out_spk, reset_time_op]



        if (is_last_layer):
            #incr_out_spk_sum_op = def_increment_spk_list(out_spk_sum_fm,  out_spk_fm, out_spk_sum_fm)
            incr_out_spk_sum_op = get_elementwise_op(NpuElementWiseOp.ADD, out_spk_sum_fm,  out_spk_fm, out_spk_sum_fm, ACCELERATOR)
            #incr_out_spk_sum_op = def_increment_spk_list(OUT_SPK_SUM_ADDR,  OUT_SPK_SUM_SCALE,   OUT_SPK_SUM_ZERO_POINT,
            #                                             OUT_SPK_ADDR,      OUT_SPK_SCALE,      OUT_SPK_ZERO_POINT)
            
            npu_op_list.append(incr_out_spk_sum_op)




        # Merge
        lut_arr_contents_str = merge_lut_values_to_str([(decay_lut_values, decay_lut_index), (check_spk_lut_values, check_spk_lut_index)])
        lif_params_arr_contents_str = merge_lif_params_to_str(LN_BETA_QUANT_LIST, VTH_QUANT_LIST)
        cms_bytearr, register_cms = gen_cms(npu_op_list, ACCELERATOR, DEBUG_MODE)

        # Generate Dicts for writing to C
        sizes_dict, addr_dict, quant_param_dict = generate_dict_for_writing_defines_to_C_files(cms_name=cms_name, weight_byte_arr=weight_byte_arr, bias_byte_arr=bias_byte_arr)



        write_cms_to_files(header_out_filepath, lif_params_on_sram, cms_bytearr, register_cms, cms_name, sizes_dict, addr_dict, quant_param_dict, lif_params_arr_contents_str, lut_arr_contents_str, weight_byte_arr=weight_byte_arr, bias_byte_arr=bias_byte_arr)
    

    layer_merge_and_write(cms_name, header_out_filepath, lif_params_on_sram)




def gen_fc_lif_old(INPUT_LAYER_SIZE, OUTPUT_LAYER_SIZE, 
         weights_volume_ohwi, bias_list, beta_list, vth_list,
         cms_name, weights_and_biases_on_sram, lif_params_on_sram, is_last_layer, NUM_TIME_STEPS,


        #IN_SPK_MAX_VAL,
        #IN_SPK_MIN_VAL,


        #Must be symmetric
        #WEIGHT_MAX_VAL,
        #WEIGHT_MIN_VAL,


        #LN_BETA_MAX_VAL,
        #LN_BETA_MIN_VAL,

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


        #VTH_MAX_VAL,
        #VTH_MIN_VAL,

        #V_MEM_SUB_VTH_MAX_VAL,
        #V_MEM_SUB_VTH_MIN_VAL,

        #OUT_SPK_MAX_VAL,
        #OUT_SPK_MIN_VAL,

         
         
         DEBUG_MODE, ACCELERATOR, header_out_filepath):



    # Assign Memory Regions (0 - 7)

    WEIGHT_AND_BIASES_REGION = 0
    SRAM_SCRATCH_REGION = 1
    PARAMS_REGION = 3
    LUT_REGION = 4
    INPUT_REGION = 5
    OUTPUT_REGION = 6


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

    # Must be same for input and output quantization
    IN_SPK_MAX_VAL = 127
    IN_SPK_MIN_VAL = -128
    OUT_SPK_MAX_VAL = 127
    OUT_SPK_MIN_VAL = -128

    # Only need to differentiate between > 0 and < 0
    V_MEM_SUB_VTH_MAX_VAL = 1
    V_MEM_SUB_VTH_MIN_VAL = -1

    ##Must be symmetric
    max_weight_val = np.max(weights_volume_ohwi)
    min_weight_val = np.min(weights_volume_ohwi)
    # Get the one with the largest absolute value (since the npu only supports symmetric quantization for weights)
    if abs(min_weight_val) > abs(max_weight_val):
        largest_weight_abs_val = abs(min_weight_val)
    else:
        largest_weight_abs_val = abs(max_weight_val)

    WEIGHT_MAX_VAL = largest_weight_abs_val + MIN_DIFF 
    WEIGHT_MIN_VAL = -(largest_weight_abs_val + MIN_DIFF)






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

    # Layer status
    V_MEM_SCALE, V_MEM_ZERO_POINT = zero_point_quant(V_MEM_MAX_VAL, V_MEM_MIN_VAL)
    TIME_NOT_UPDATED_SCALE, TIME_NOT_UPDATED_ZERO_POINT = zero_point_quant(TIME_NOT_UPDATED_MAX_VAL, TIME_NOT_UPDATED_MIN_VAL)

    # TMP Feature maps
    DECAY_SCALE, DECAY_ZERO_POINT = zero_point_quant(DECAY_MAX_VAL, DECAY_MIN_VAL)
    DECAY_ACC_SCALE, DECAY_ACC_ZERO_POINT = zero_point_quant(DECAY_ACC_MAX_VAL, DECAY_ACC_MIN_VAL)
    IN_CURR_SCALE, IN_CURR_ZERO_POINT = zero_point_quant(IN_CURR_MAX_VAL, IN_CURR_MIN_VAL)
    DECAYED_MEM_SCALE, DECAYED_MEM_ZERO_POINT = zero_point_quant(DECAYED_MEM_MAX_VAL, DECAYED_MEM_MIN_VAL)
    V_MEM_SUB_VTH_SCALE, V_MEM_SUB_VTH_ZERO_POINT = zero_point_quant(V_MEM_SUB_VTH_MAX_VAL, V_MEM_SUB_VTH_MIN_VAL)


    RESET_SCALE, RESET_ZERO_POINT = zero_point_quant(RESET_MAX_VAL, RESET_MIN_VAL)

    WEIGHT_SCALE, WEIGHT_ZERO_POINT = symmetric_zero_point_quant(WEIGHT_MAX_VAL, WEIGHT_MIN_VAL)
    BIAS_SCALE, BIAS_ZERO_POINT = IN_SPK_SCALE*WEIGHT_SCALE, 0
    # Bias at most 40 bit value
    #print("BIAS:\n\tMax val:", 2**39 / BIAS_SCALE, "\n\tMin val:", -2**39 / BIAS_SCALE)

    # Output Feature Map
    UPDATE_NXT_LAYER_SCALE, UPDATE_NXT_LAYER_ZERO_POINT = zero_point_quant(UPDATE_NXT_LAYER_MAX_VAL, UPDATE_NXT_LAYER_MIN_VAL)
    OUT_SPK_SCALE, OUT_SPK_ZERO_POINT = zero_point_quant(OUT_SPK_MAX_VAL, OUT_SPK_MIN_VAL)

    if (is_last_layer):
        OUT_SPK_SUM_SCALE, OUT_SPK_SUM_ZERO_POINT = zero_point_quant(OUT_SPK_SUM_MAX_VAL, OUT_SPK_SUM_MIN_VAL)
    else:
        OUT_SPK_SUM_SCALE, OUT_SPK_SUM_ZERO_POINT = 0, 0




    '''
    Quantize and Decrypt Weight, bias and Time constant (beta), and Vth
    '''
    # Generate Weights and Bias list
    weight_byte_arr_init, bias_byte_arr_init = get_int8_fc_weights_and_biases(weights_volume_ohwi, bias_list, INPUT_LAYER_SIZE, OUTPUT_LAYER_SIZE, WEIGHT_SCALE, WEIGHT_ZERO_POINT, IN_SPK_SCALE, IN_CURR_SCALE, ACCELERATOR, DEBUG_MODE)


    # Generate LIF Params Quant List
    LN_BETA_QUANT_LIST = generate_ln_beta_values(beta_list=beta_list, ln_beta_scale=LN_BETA_SCALE, ln_beta_zero_point=LN_BETA_ZERO_POINT)    
    VTH_QUANT_LIST = quantize_vth_values(vth_list=vth_list, vth_scale=VTH_SCALE, vth_zero_point=VTH_ZERO_POINT)



    # Can at most only ever be 8 tensors we have to track on SRAM Sratch region: bias, weight, tmp1, tmp2, v_mem, out_spk_sum, time_not_updataed, update_nxt_layer
    MAX_NUM_TENSORS_TO_TRACK = 8

    # Assign Memory segments in SRAM Scratch (region 1)
    
    if (not is_last_layer):

        if (not weights_and_biases_on_sram):


            # SRAM SCRATCH REGION (tensors that use NHCWB16 format needs to be 16-byte aligned)
            TMP1_ADDR           	=	0
            TMP2_ADDR           	=	TMP1_ADDR               +   next_multiple(OUTPUT_LAYER_SIZE, 16)
            V_MEM_ADDR          	=	TMP2_ADDR               +   next_multiple(OUTPUT_LAYER_SIZE, 16)
            TIME_NOT_UPDATED_ADDR	=	V_MEM_ADDR              +   next_multiple(OUTPUT_LAYER_SIZE, 16)
            UPDATE_NXT_LAYER_ADDR	=	TIME_NOT_UPDATED_ADDR   +   1


            #TMP1_ADDR           	=	0
            #TMP2_ADDR           	=	TMP1_ADDR               +   OUTPUT_LAYER_SIZE
            #V_MEM_ADDR          	=	TMP2_ADDR               +   OUTPUT_LAYER_SIZE
            #TIME_NOT_UPDATED_ADDR	=	V_MEM_ADDR              +   OUTPUT_LAYER_SIZE
            #UPDATE_NXT_LAYER_ADDR	=	TIME_NOT_UPDATED_ADDR   +   1

            #TENSOR_ARENA_SIZE	=	3*OUTPUT_LAYER_SIZE + 16 
            TENSOR_ARENA_SIZE	=	3*next_multiple(OUTPUT_LAYER_SIZE, 16) + 2 


            OUT_SPK_SUM_ADDR = -1



            # WEIGHTS AND BIAS REGION
            BIAS_ADDR = 0
            WEIGHT_ADDR = BIAS_ADDR + len(bias_byte_arr_init)


        else:

            BIAS_ADDR           	=   0 
            WEIGHT_ADDR         	=	BIAS_ADDR               +   len(bias_byte_arr_init) # Bias len
            TMP1_ADDR           	=	WEIGHT_ADDR             +   len(weight_byte_arr_init) #weight len
            TMP2_ADDR           	=	TMP1_ADDR               +   next_multiple(OUTPUT_LAYER_SIZE, 16)
            V_MEM_ADDR          	=	TMP2_ADDR               +   next_multiple(OUTPUT_LAYER_SIZE, 16)
            TIME_NOT_UPDATED_ADDR	=	V_MEM_ADDR              +   next_multiple(OUTPUT_LAYER_SIZE, 16)
            UPDATE_NXT_LAYER_ADDR	=	TIME_NOT_UPDATED_ADDR   +   1


            TENSOR_ARENA_SIZE	=	3*OUTPUT_LAYER_SIZE + len(bias_byte_arr_init) +len(weight_byte_arr_init) + 16

            OUT_SPK_SUM_ADDR = -1
    
    else:

        if (not weights_and_biases_on_sram):
            # SRAM SCRATCH REGION
            TMP1_ADDR           	=	0
            TMP2_ADDR           	=	TMP1_ADDR               +   next_multiple(OUTPUT_LAYER_SIZE, 16)
            V_MEM_ADDR          	=	TMP2_ADDR               +   next_multiple(OUTPUT_LAYER_SIZE, 16)
            OUT_SPK_SUM_ADDR        =   V_MEM_ADDR              +   next_multiple(OUTPUT_LAYER_SIZE, 16)
            TIME_NOT_UPDATED_ADDR   =   OUT_SPK_SUM_ADDR        +   next_multiple(OUTPUT_LAYER_SIZE, 16)
            UPDATE_NXT_LAYER_ADDR	=	TIME_NOT_UPDATED_ADDR   +   1

            TENSOR_ARENA_SIZE	=	4*OUTPUT_LAYER_SIZE + 16


            # WEIGHTS AND BIAS REGION
            BIAS_ADDR = 0
            WEIGHT_ADDR = BIAS_ADDR + len(bias_byte_arr_init)

        

        else:

            BIAS_ADDR           	=   0 
            WEIGHT_ADDR         	=	BIAS_ADDR               +   len(bias_byte_arr_init) # Bias len
            TMP1_ADDR           	=	WEIGHT_ADDR             +   len(weight_byte_arr_init) #weight len
            TMP2_ADDR           	=	TMP1_ADDR               +   next_multiple(OUTPUT_LAYER_SIZE, 16)
            V_MEM_ADDR          	=	TMP2_ADDR               +   next_multiple(OUTPUT_LAYER_SIZE, 16)  
            OUT_SPK_SUM_ADDR        =   V_MEM_ADDR              +   next_multiple(OUTPUT_LAYER_SIZE, 16)
            TIME_NOT_UPDATED_ADDR   =   OUT_SPK_SUM_ADDR        +   next_multiple(OUTPUT_LAYER_SIZE, 16)
            UPDATE_NXT_LAYER_ADDR	=	TIME_NOT_UPDATED_ADDR   +   1

            TENSOR_ARENA_SIZE	=	4*OUTPUT_LAYER_SIZE + len(bias_byte_arr_init) +len(weight_byte_arr_init) + 1





    def is_16_bit_aligned(val):
        if val == next_multiple(val, 16):
            return True
        else:
            return False

    print("====================================================")
    print("TMP1", TMP1_ADDR, is_16_bit_aligned(TMP1_ADDR))
    print("TMP2", TMP2_ADDR, is_16_bit_aligned(TMP2_ADDR))
    print("V_MEM", V_MEM_ADDR, is_16_bit_aligned(V_MEM_ADDR))
    print("OUT_SPK_SUM", OUT_SPK_SUM_ADDR, is_16_bit_aligned(OUT_SPK_SUM_ADDR))
    print("TIME_NOT_UPDATED:", TIME_NOT_UPDATED_ADDR, is_16_bit_aligned(TIME_NOT_UPDATED_ADDR))
    print("UPDATE_NXT_LAYER:", UPDATE_NXT_LAYER_ADDR, is_16_bit_aligned(UPDATE_NXT_LAYER_ADDR))
    print("----------------------------------------------------")
    print("TENSOR_ARENA_SIZE:", TENSOR_ARENA_SIZE, is_16_bit_aligned(TENSOR_ARENA_SIZE))
    print("====================================================")




    ##############
    # Assign Tmp tensors here!
    DECAY_ADDR = TMP1_ADDR
    IN_CURR_ADDR = TMP2_ADDR
    DECAYED_MEM_ADDR = TMP1_ADDR
    RESET_ADDR = TMP2_ADDR

    ##############



    # Assign Memory segments for region 3
    LN_BETA_ADDR = 0
    VTH_ADDR = LN_BETA_ADDR + OUTPUT_LAYER_SIZE


    # Assign Memory segments for region 4

    DECAY_LUT_INDEX = 0
    CHECK_SPK_LUT_INDEX = 1



    # Assign Memory segment for region 5
    IN_SPK_ADDR = 0

    # Assign Memory segment for region 6
    OUT_SPK_ADDR = 0











    # Create Dicts for writing to C files
    def generate_dict_for_writing_defines_to_C_files(cms_name, weight_byte_arr, bias_byte_arr):


        sizes_dict = {

            cms_name.upper()+"_IS_LAST_LAYER"           : int(is_last_layer),    #num tensors in sram scratchpad
            cms_name.upper()+"_MAX_NUM_TENSORS_TO_TRACK"   : MAX_NUM_TENSORS_TO_TRACK,    #num tensors in sram scratchpad

            cms_name.upper()+"_TENSOR_ARENA_SIZE "  : TENSOR_ARENA_SIZE,
            cms_name.upper()+"_INPUT_LAYER_SIZE "   : INPUT_LAYER_SIZE,             
            cms_name.upper()+"_OUTPUT_LAYER_SIZE "  : OUTPUT_LAYER_SIZE,

            cms_name.upper()+"_WEIGHT_LEN" : len(weight_byte_arr),
            cms_name.upper()+"_BIAS_LEN" : len(bias_byte_arr)        
        }

        addr_dict = {
            # Input Feature map
            cms_name.upper()+"_IN_SPK_ADDR" : IN_SPK_ADDR,

            # Layer params
            cms_name.upper()+"_BIAS_ADDR" : BIAS_ADDR,
            cms_name.upper()+"_WEIGHT_ADDR" : WEIGHT_ADDR,
            cms_name.upper()+"_LN_BETA_ADDR" : LN_BETA_ADDR,
            cms_name.upper()+"_VTH_ADDR" : VTH_ADDR,

            # Layer status
            cms_name.upper()+"_V_MEM_ADDR" : V_MEM_ADDR,
            cms_name.upper()+"_TIME_NOT_UPDATED_ADDR" : TIME_NOT_UPDATED_ADDR,

            # TMP Feature maps
            cms_name.upper()+"_IN_CURR_ADDR" : IN_CURR_ADDR,
            cms_name.upper()+"_DECAY_ADDR" : DECAY_ADDR,
            cms_name.upper()+"_DECAYED_MEM_ADDR" : DECAYED_MEM_ADDR,


            # Output Feature Map
            cms_name.upper()+"_UPDATE_NXT_LAYER_ADDR" : UPDATE_NXT_LAYER_ADDR,
            cms_name.upper()+"_OUT_SPK_ADDR" : OUT_SPK_ADDR

        }



        quant_param_dict = {
            cms_name.upper()+"_IN_SPK_SCALE" : IN_SPK_SCALE,
            cms_name.upper()+"_IN_SPK_ZERO_POINT" : IN_SPK_ZERO_POINT,


            cms_name.upper()+"_BIAS_SCALE" : BIAS_SCALE,
            cms_name.upper()+"_BIAS_ZERO_POINT" : BIAS_ZERO_POINT,
            cms_name.upper()+"_WEIGHT_SCALE" : WEIGHT_SCALE,
            cms_name.upper()+"_WEIGHT_ZERO_POINT" : WEIGHT_ZERO_POINT,


            cms_name.upper()+"_LN_BETA_SCALE" : LN_BETA_SCALE,
            cms_name.upper()+"_LN_BETA_ZERO_POINT" : LN_BETA_ZERO_POINT,
            cms_name.upper()+"_VTH_SCALE" : VTH_SCALE,
            cms_name.upper()+"_VTH_ZERO_POINT" : VTH_ZERO_POINT,
            cms_name.upper()+"_V_MEM_SCALE" : V_MEM_SCALE,
            cms_name.upper()+"_V_MEM_ZERO_POINT" : V_MEM_ZERO_POINT,
            cms_name.upper()+"_TIME_NOT_UPDATED_SCALE" : TIME_NOT_UPDATED_SCALE,
            cms_name.upper()+"_TIME_NOT_UPDATED_ZERO_POINT" : TIME_NOT_UPDATED_ZERO_POINT,

            cms_name.upper()+"_DECAY_SCALE" : DECAY_SCALE,
            cms_name.upper()+"_DECAY_ZERO_POINT" : DECAY_ZERO_POINT,
            cms_name.upper()+"_IN_CURR_SCALE" : IN_CURR_SCALE,
            cms_name.upper()+"_IN_CURR_ZERO_POINT" : IN_CURR_ZERO_POINT,
            cms_name.upper()+"_DECAYED_MEM_SCALE" : DECAYED_MEM_SCALE,
            cms_name.upper()+"_DECAYED_MEM_ZERO_POINT" : DECAYED_MEM_ZERO_POINT,


            # Output
            cms_name.upper()+"_UPDATE_NXT_LAYER_SCALE" : UPDATE_NXT_LAYER_SCALE,
            cms_name.upper()+"_UPDATE_NXT_LAYER_ZERO_POINT" : UPDATE_NXT_LAYER_ZERO_POINT,
            cms_name.upper()+"_OUT_SPK_SCALE" : OUT_SPK_SCALE,
            cms_name.upper()+"_OUT_SPK_ZERO_POINT" : OUT_SPK_ZERO_POINT,
        }


        #if (is_last_layer):
        addr_dict[cms_name.upper()+"_OUT_SPK_SUM_ADDR"] = OUT_SPK_SUM_ADDR
        quant_param_dict[cms_name.upper()+"_OUT_SPK_SUM_SCALE"] = OUT_SPK_SUM_SCALE
        quant_param_dict[cms_name.upper()+"_OUT_SPK_SUM_ZERO_POINT"] = OUT_SPK_SUM_ZERO_POINT

        return sizes_dict, addr_dict, quant_param_dict






    def def_decay_lut():

        IFM2_IS_FIRST_OPERAND = False

        ifm = create_feature_map(
            height=1, width=1, depth=OUTPUT_LAYER_SIZE,
            region=PARAMS_REGION,
            layout=NpuLayout.NHWC,
            data_type=NpuDataType.INT8,
            fm_elem_size=1,
            fm_addr=LN_BETA_ADDR,
            scale=LN_BETA_SCALE,
            zero_point=LN_BETA_ZERO_POINT,
            name="ln_beta"
        )


        ifm2 = create_feature_map(
            height=1, width=1, depth=1,
            region=SRAM_SCRATCH_REGION,
            layout=NpuLayout.NHWC,
            data_type=NpuDataType.INT8,
            fm_elem_size=1,
            fm_addr=TIME_NOT_UPDATED_ADDR,
            scale=TIME_NOT_UPDATED_SCALE,
            zero_point=TIME_NOT_UPDATED_ZERO_POINT,
            name="time_not_updated"
        )


        ofm = create_feature_map(
            height=1, width=1, depth=OUTPUT_LAYER_SIZE,
            region=SRAM_SCRATCH_REGION,
            layout=NpuLayout.NHCWB16,
            #layout=NpuLayout.NHWC,
            data_type=NpuDataType.INT8,
            fm_elem_size=1,
            fm_addr=DECAY_ADDR,
            scale=DECAY_ACC_SCALE,
            zero_point=DECAY_ACC_ZERO_POINT,
            name="decay"
        )





        #DMA for LUT



        # Handle LUT generation and DMA
        decay_lut_index = DECAY_LUT_INDEX

        import math
        dma_lut_op, decay_lut_values = create_lut_and_dma(approximated_func=math.exp, lut_index=decay_lut_index, lut_region=LUT_REGION, data_type=ofm.data_type, 
                        scale_pre_lut=DECAY_ACC_SCALE, zero_point_pre_lut=DECAY_ACC_ZERO_POINT,
                        scale_post_lut=DECAY_SCALE, zero_point_post_lut=DECAY_ZERO_POINT,
                        accelerator=ACCELERATOR,
                        debug_mode=DEBUG_MODE
        )

        activation = create_activation(
            activation_op=NpuActivationOp.TABLE_LOOKUP,
            min_val=None,
            max_val=None,
            lookup_table_index=decay_lut_index
        )


        exp_mul_lnb_time_op = NpuElementWiseOperation(NpuElementWiseOp.MUL)
    
        #elementwise operation
        exp_mul_lnb_time_op.reversed_operands = IFM2_IS_FIRST_OPERAND
        exp_mul_lnb_time_op.rescale = None

        #NpuBlockOperation
        exp_mul_lnb_time_op.ifm = ifm
        exp_mul_lnb_time_op.ifm2 = ifm2
        exp_mul_lnb_time_op.ifm2_scalar = None   #set if ifm2 is a scalar
        exp_mul_lnb_time_op.ofm = ofm
        exp_mul_lnb_time_op.kernel = None
        exp_mul_lnb_time_op.weights = []
        exp_mul_lnb_time_op.biases = []
        exp_mul_lnb_time_op.padding = None
        exp_mul_lnb_time_op.activation = activation
    
        block_config = get_block_config(exp_mul_lnb_time_op, ACCELERATOR)
        exp_mul_lnb_time_op.block_config = block_config
        exp_mul_lnb_time_op.rounding_mode = NpuRoundingMode.TFL
        exp_mul_lnb_time_op.fused_quantize = False
        exp_mul_lnb_time_op.ifm_upscale = NpuResamplingMode.NONE
        exp_mul_lnb_time_op.accumulator_type = NpuAccumulatorType.Default

        #check_block_config_legal(block_config, exp_mul_lnb_time_op, ACCELERATOR)

        return dma_lut_op, exp_mul_lnb_time_op, decay_lut_values, decay_lut_index


    def def_fullyconnected(IN_SPK_ADDR, IN_CURR_ADDR, weights_and_biases_on_sram):




        ifm = create_feature_map(
        height=1, width=1, depth=INPUT_LAYER_SIZE,
        region=INPUT_REGION,
        layout=NpuLayout.NHWC,
        data_type=NpuDataType.INT8,
        fm_elem_size=1,
        fm_addr=IN_SPK_ADDR,
        scale = IN_SPK_SCALE,
        zero_point = IN_SPK_ZERO_POINT,
        name="in_spk"
        )



        ifm2 = None


        ofm = create_feature_map(
            height=1, width=1, depth=OUTPUT_LAYER_SIZE,
            region=SRAM_SCRATCH_REGION,
            layout=NpuLayout.NHCWB16,
            #layout=NpuLayout.NHWC,
            data_type=NpuDataType.INT8,
            fm_elem_size=1,
            fm_addr=IN_CURR_ADDR,
            scale = IN_CURR_SCALE,
            zero_point = IN_CURR_ZERO_POINT,
            name="in_curr"
        )



        # Kernel
        kernel = NpuKernel(
            w=1, h=1, 
            stride_x=1, stride_y=1, dilation_x=1, dilation_y=1
        )


        my_op = NpuConv2DOperation()
        my_op.ifm               =   ifm
        my_op.ifm2              =   ifm2
        my_op.ifm2_scalar       =   None
        my_op.ofm               =   ofm
        block_config = get_block_config(my_op, ACCELERATOR)

    

        block_traversal = NpuBlockTraversal.DEPTH_FIRST


        # Define Weights
        if ifm.data_type == NpuDataType.INT8:
            weight_ifm_bitdepth = 8 #int8
        elif ifm.data_type == NpuDataType.INT16:
            weight_ifm_bitdepth = 16 #int16






        weight_byte_arr, bias_byte_arr = gen_weights_and_biases(accelerator=ACCELERATOR,
                            weights_volume_ohwi=weights_volume_ohwi,
                                dilation_xy=(1,1),
                                ifm_bitdepth=weight_ifm_bitdepth,
                                ofm_block_depth=block_config[2],
                                op_type=NpuOperationType.Conv2D,
                                block_traversal=block_traversal,

                                #ONLY FOR 1 DIM FMs!!!!
                                bias_list=bias_list,

                                ifm_scale=ifm.quantization.scale_f32,
                                weight_scale=WEIGHT_SCALE,
                                weight_zero_point=WEIGHT_ZERO_POINT,
                                ofm_scale=ofm.quantization.scale_f32,


                                is_debug_mode=DEBUG_MODE
        )

        weight_n_bias_len = len(bias_byte_arr) + len(weight_byte_arr)
        if DEBUG_MODE:
            print("weight_n_bias_len", weight_n_bias_len)
            print("\tbias_len:", len(bias_byte_arr))
            print("\tweight_len", len(weight_byte_arr))

        

        # Make sure that init is the same as current weights
        if (weight_byte_arr != weight_byte_arr_init):
            print("Error: weight_byte_arr != weight_byte_arr_init")
            exit()
        if (bias_byte_arr != bias_byte_arr_init):
            print("Error: bias_byte_arr != bias_byte_arr_init")
            exit()

    

        


        #BIAS_ADDR = WEIGHT_N_BIAS_ADDR
        #WEIGHT_ADDR = BIAS_ADDR + len(bias_byte_arr)
    
    
        if (weights_and_biases_on_sram):
            WEIGHT_N_BIAS_ADDR = BIAS_ADDR #Bias before weights

            #DMA    
            dma_src = NpuAddressRange(region=WEIGHT_AND_BIASES_REGION, address=0, length=weight_n_bias_len)
            dma_dst = NpuAddressRange(region=SRAM_SCRATCH_REGION, address=WEIGHT_N_BIAS_ADDR, length=weight_n_bias_len)
            dma_op = NpuDmaOperation(src=dma_src, dest=dma_dst)


            weights = NpuAddressRange(region=SRAM_SCRATCH_REGION, address=WEIGHT_ADDR, length=len(weight_byte_arr))
            biases = NpuAddressRange(region=SRAM_SCRATCH_REGION, address=BIAS_ADDR, length=len(bias_byte_arr))
        
        else:

            dma_op = None

            weights = NpuAddressRange(region=WEIGHT_AND_BIASES_REGION, address=WEIGHT_ADDR, length=len(weight_byte_arr))
            biases = NpuAddressRange(region=WEIGHT_AND_BIASES_REGION, address=BIAS_ADDR, length=len(bias_byte_arr))




    

        padding = NpuPadding(top=0, left=0, bottom=0, right=0)



    


        activation = create_activation(
            activation_op=NpuActivationOp.NONE_OR_RELU,
            min_val=None,
            max_val=None,
        )

    


        fused_quantize = False





        my_op.kernel            =   kernel
        my_op.weights           =   [weights]
        my_op.biases            =   [biases]
        my_op.padding           =   padding
        my_op.activation        =   activation

        my_op.block_config      =   block_config
        my_op.rounding_mode     =   NpuRoundingMode.TFL
        my_op.fused_quantize    =   fused_quantize
        my_op.ifm_upscale       =   NpuResamplingMode.NONE
        my_op.accumulator_type  =   NpuAccumulatorType.Int32
        my_op.block_traversal   =   block_traversal


        #check_block_config_legal(block_config, my_op, ACCELERATOR)





        return my_op, dma_op, weight_byte_arr, bias_byte_arr, 






    def def_mul_decay_Vmem():
    
        IFM2_IS_FIRST_OPERAND = False


        ifm = create_feature_map(
            height=1, width=1, depth=OUTPUT_LAYER_SIZE,
            region=SRAM_SCRATCH_REGION,
            layout=NpuLayout.NHWC,
            data_type=NpuDataType.INT8,
            fm_elem_size=1,
            fm_addr=V_MEM_ADDR,
            scale = V_MEM_SCALE,
            zero_point = V_MEM_ZERO_POINT,
            name="v_mem"
        )

        ifm2 = create_feature_map(
            height=1, width=1, depth=OUTPUT_LAYER_SIZE,
            region=SRAM_SCRATCH_REGION,
            layout=NpuLayout.NHCWB16,
            #layout=NpuLayout.NHWC,
            data_type=NpuDataType.INT8,
            fm_elem_size=1,
            fm_addr=DECAY_ADDR,
            scale = DECAY_SCALE,
            zero_point = DECAY_ZERO_POINT,
            name="decay"
        )



        ofm = create_feature_map(
            height=1, width=1, depth=OUTPUT_LAYER_SIZE,
            region=SRAM_SCRATCH_REGION,
            layout=NpuLayout.NHCWB16,
            #layout=NpuLayout.NHWC,
            data_type=NpuDataType.INT8,
            fm_elem_size=1,
            fm_addr=DECAYED_MEM_ADDR,
            scale = DECAYED_MEM_SCALE,
            zero_point = DECAYED_MEM_ZERO_POINT,
            name="decayed_mem"
        )

        #block_config = NpuShape3D(2, 2, 32)

        activation = create_activation(
            activation_op=NpuActivationOp.NONE_OR_RELU,
            min_val=None,
            max_val=None,
        )




        mul_decay_op = NpuElementWiseOperation(NpuElementWiseOp.MUL)
    
        #elementwise operation
        mul_decay_op.reversed_operands = IFM2_IS_FIRST_OPERAND
        mul_decay_op.rescale = None

        #NpuBlockOperation
        mul_decay_op.ifm = ifm
        mul_decay_op.ifm2 = ifm2
        mul_decay_op.ifm2_scalar = None   #set if ifm2 is a scalar
        mul_decay_op.ofm = ofm
        mul_decay_op.kernel = None
        mul_decay_op.weights = []
        mul_decay_op.biases = []
        mul_decay_op.padding = None
        mul_decay_op.activation = activation

        block_config = get_block_config(mul_decay_op, ACCELERATOR)
        mul_decay_op.block_config = block_config
        mul_decay_op.rounding_mode = NpuRoundingMode.TFL
        mul_decay_op.fused_quantize = False
        mul_decay_op.ifm_upscale = NpuResamplingMode.NONE
        mul_decay_op.accumulator_type = NpuAccumulatorType.Default


        #check_block_config_legal(block_config, mul_decay_op, ACCELERATOR)



        return mul_decay_op




    def def_add_decayed_mem_in_curr():
        IFM2_IS_FIRST_OPERAND = False

        ifm = create_feature_map(
            height=1, width=1, depth=OUTPUT_LAYER_SIZE,
            region=SRAM_SCRATCH_REGION,
            layout=NpuLayout.NHCWB16,
            #layout=NpuLayout.NHWC,
            data_type=NpuDataType.INT8,
            fm_elem_size=1,
            fm_addr=DECAYED_MEM_ADDR,
            scale = DECAYED_MEM_SCALE,
            zero_point = DECAYED_MEM_ZERO_POINT,
            name="decayed_mem"
        )

        ifm2 = create_feature_map(
            height=1, width=1, depth=OUTPUT_LAYER_SIZE,
            region=SRAM_SCRATCH_REGION,
            layout=NpuLayout.NHCWB16,
            #layout=NpuLayout.NHWC,
            data_type=NpuDataType.INT8,
            fm_elem_size=1,
            fm_addr=IN_CURR_ADDR,
            scale = IN_CURR_SCALE,
            zero_point = IN_CURR_ZERO_POINT,
            name="in_curr"
        )




        ofm = create_feature_map(
            height=1, width=1, depth=OUTPUT_LAYER_SIZE,
            region=SRAM_SCRATCH_REGION,
            layout=NpuLayout.NHCWB16,
            #layout=NpuLayout.NHWC,
            data_type=NpuDataType.INT8,
            fm_elem_size=1,
            fm_addr=V_MEM_ADDR,
            scale=V_MEM_SCALE,
            zero_point=V_MEM_ZERO_POINT,
            name="updated_mem"
        )

        #block_config = NpuShape3D(2, 2, 32)

        activation = create_activation(
            activation_op=NpuActivationOp.NONE_OR_RELU,
            min_val=None,
            max_val=None,
        )




        add_decayed_mem_in_curr = NpuElementWiseOperation(NpuElementWiseOp.ADD)

        #elementwise operation
        add_decayed_mem_in_curr.reversed_operands = IFM2_IS_FIRST_OPERAND
        add_decayed_mem_in_curr.rescale = None

        #NpuBlockOperation
        add_decayed_mem_in_curr.ifm = ifm
        add_decayed_mem_in_curr.ifm2 = ifm2
        add_decayed_mem_in_curr.ifm2_scalar = None   #set if ifm2 is a scalar
        add_decayed_mem_in_curr.ofm = ofm
        add_decayed_mem_in_curr.kernel = None
        add_decayed_mem_in_curr.weights = []
        add_decayed_mem_in_curr.biases = []
        add_decayed_mem_in_curr.padding = None
        add_decayed_mem_in_curr.activation = activation

        block_config = get_block_config(add_decayed_mem_in_curr, ACCELERATOR)
        add_decayed_mem_in_curr.block_config = block_config
        add_decayed_mem_in_curr.rounding_mode = NpuRoundingMode.TFL
        add_decayed_mem_in_curr.fused_quantize = False
        add_decayed_mem_in_curr.ifm_upscale = NpuResamplingMode.NONE
        add_decayed_mem_in_curr.accumulator_type = NpuAccumulatorType.Default


        #check_block_config_legal(block_config, add_decayed_mem_in_curr, ACCELERATOR)



        return add_decayed_mem_in_curr





    def def_check_spk_sub_v_mem_updated_vth():
        IFM2_IS_FIRST_OPERAND = False 

        ifm = create_feature_map(
            height=1, width=1, depth=OUTPUT_LAYER_SIZE,
            region=SRAM_SCRATCH_REGION,
            layout=NpuLayout.NHCWB16,
            #layout=NpuLayout.NHWC,
            data_type=NpuDataType.INT8,
            fm_elem_size=1,
            fm_addr=V_MEM_ADDR,
            scale = V_MEM_SCALE,
            zero_point = V_MEM_ZERO_POINT,
            name="v_mem_updated"
        )

        ifm2 = create_feature_map(
            height=1, width=1, depth=OUTPUT_LAYER_SIZE,
            region=PARAMS_REGION,
            layout=NpuLayout.NHWC,
            data_type=NpuDataType.INT8,
            fm_elem_size=1,
            fm_addr=VTH_ADDR,
            scale = VTH_SCALE,
            zero_point = VTH_ZERO_POINT,
            name="vth"
        )




        ofm = create_feature_map(
            height=1, width=1, depth=OUTPUT_LAYER_SIZE,
            region=OUTPUT_REGION,
            layout=NpuLayout.NHWC,
            data_type=NpuDataType.INT8,
            fm_elem_size=1,
            fm_addr=OUT_SPK_ADDR,
            scale=V_MEM_SUB_VTH_SCALE,
            zero_point=V_MEM_SUB_VTH_ZERO_POINT,
            name="out_spk"
        )

        #block_config = NpuShape3D(2, 2, 32)


        check_spk_lut_index = CHECK_SPK_LUT_INDEX
        activation = create_activation(
            activation_op=NpuActivationOp.TABLE_LOOKUP,
            min_val=None,
            max_val=None,
            lookup_table_index=check_spk_lut_index
        )

        #activation = create_activation(
            #activation_op=NpuActivationOp.NONE_OR_RELU,
            #min_val=None,
            #max_val=None
        #)


        # Define function that lut will approximate
        def check_positive(x_real):
            if x_real > 0:
                y_real = 1
            else:
                y_real = 0
        
            return y_real

        # It might be problematic to have the same scaling before and after LUT, currently is working though
        # if scale = 1, zero_point = 0, and dif = (v_mem - vth), where dif < 1, then it will only spike if dif is rounded to 1 (and not 0)
        check_spk_lut_dma_op, check_spk_lut_values = create_lut_and_dma(approximated_func=check_positive, lut_index=check_spk_lut_index, lut_region=LUT_REGION, data_type=ofm.data_type, 
                        scale_pre_lut=V_MEM_SUB_VTH_SCALE, zero_point_pre_lut=V_MEM_SUB_VTH_ZERO_POINT,
                        scale_post_lut=OUT_SPK_SCALE, zero_point_post_lut=OUT_SPK_ZERO_POINT,
                        accelerator=ACCELERATOR,
                        debug_mode=DEBUG_MODE
        )




        check_spk_sub_v_mem_updated_vth_op = NpuElementWiseOperation(NpuElementWiseOp.SUB)

        #elementwise operation
        check_spk_sub_v_mem_updated_vth_op.reversed_operands = IFM2_IS_FIRST_OPERAND
        check_spk_sub_v_mem_updated_vth_op.rescale = None

        #NpuBlockOperation
        check_spk_sub_v_mem_updated_vth_op.ifm = ifm
        check_spk_sub_v_mem_updated_vth_op.ifm2 = ifm2
        check_spk_sub_v_mem_updated_vth_op.ifm2_scalar = None   #set if ifm2 is a scalar
        check_spk_sub_v_mem_updated_vth_op.ofm = ofm
        check_spk_sub_v_mem_updated_vth_op.kernel = None
        check_spk_sub_v_mem_updated_vth_op.weights = []
        check_spk_sub_v_mem_updated_vth_op.biases = []
        check_spk_sub_v_mem_updated_vth_op.padding = None
        check_spk_sub_v_mem_updated_vth_op.activation = activation
    
        block_config = get_block_config(check_spk_sub_v_mem_updated_vth_op, ACCELERATOR)
        check_spk_sub_v_mem_updated_vth_op.block_config = block_config
        check_spk_sub_v_mem_updated_vth_op.rounding_mode = NpuRoundingMode.TFL
        check_spk_sub_v_mem_updated_vth_op.fused_quantize = False
        check_spk_sub_v_mem_updated_vth_op.ifm_upscale = NpuResamplingMode.NONE
        check_spk_sub_v_mem_updated_vth_op.accumulator_type = NpuAccumulatorType.Int32


        #check_block_config_legal(block_config, check_spk_sub_v_mem_updated_vth_op, ACCELERATOR)



        return check_spk_lut_dma_op, check_spk_sub_v_mem_updated_vth_op, check_spk_lut_values, check_spk_lut_index



    def def_mul_vth_out_spk():

        IFM2_IS_FIRST_OPERAND = False


        ifm = create_feature_map(
            height=1, width=1, depth=OUTPUT_LAYER_SIZE,
            region=PARAMS_REGION,
            layout=NpuLayout.NHWC,
            data_type=NpuDataType.INT8,
            fm_elem_size=1,
            fm_addr=VTH_ADDR,
            scale = VTH_SCALE,
            zero_point = VTH_ZERO_POINT,
            name="vth"
        )

        ifm2 = create_feature_map(
            height=1, width=1, depth=OUTPUT_LAYER_SIZE,
            region=OUTPUT_REGION,
            layout=NpuLayout.NHWC,
            data_type=NpuDataType.INT8,
            fm_elem_size=1,
            fm_addr=OUT_SPK_ADDR,
            scale = OUT_SPK_SCALE,
            zero_point = OUT_SPK_ZERO_POINT,
            name="out_spk"
        )


        # Same scaling as VTH (since reset is either 0 or 1)
        ofm = create_feature_map(
            height=1, width=1, depth=OUTPUT_LAYER_SIZE,
            region=SRAM_SCRATCH_REGION,
            layout=NpuLayout.NHCWB16,
            #layout=NpuLayout.NHWC,
            data_type=NpuDataType.INT8,
            fm_elem_size=1,
            fm_addr=RESET_ADDR,
            scale = RESET_SCALE,
            zero_point = RESET_ZERO_POINT,
            name="reset"
        )


        #block_config = NpuShape3D(2, 2, 32)

        activation = create_activation(
            activation_op=NpuActivationOp.NONE_OR_RELU,
            min_val=None,
            max_val=None,
        )




        mul_vth_out_spk_op = NpuElementWiseOperation(NpuElementWiseOp.MUL)
    
        #elementwise operation
        mul_vth_out_spk_op.reversed_operands = IFM2_IS_FIRST_OPERAND
        mul_vth_out_spk_op.rescale = None

        #NpuBlockOperation
        mul_vth_out_spk_op.ifm = ifm
        mul_vth_out_spk_op.ifm2 = ifm2
        mul_vth_out_spk_op.ifm2_scalar = None   #set if ifm2 is a scalar
        mul_vth_out_spk_op.ofm = ofm
        mul_vth_out_spk_op.kernel = None
        mul_vth_out_spk_op.weights = []
        mul_vth_out_spk_op.biases = []
        mul_vth_out_spk_op.padding = None
        mul_vth_out_spk_op.activation = activation


        block_config = get_block_config(mul_vth_out_spk_op, ACCELERATOR)
        mul_vth_out_spk_op.block_config = block_config

        mul_vth_out_spk_op.rounding_mode = NpuRoundingMode.TFL
        mul_vth_out_spk_op.fused_quantize = False
        mul_vth_out_spk_op.ifm_upscale = NpuResamplingMode.NONE
        mul_vth_out_spk_op.accumulator_type = NpuAccumulatorType.Default


        #check_block_config_legal(block_config, mul_vth_out_spk_op, ACCELERATOR)



        return mul_vth_out_spk_op

    

    def def_sub_mem_updated_reset():

        IFM2_IS_FIRST_OPERAND = False


        ifm = create_feature_map(
            height=1, width=1, depth=OUTPUT_LAYER_SIZE,
            region=SRAM_SCRATCH_REGION,
            layout=NpuLayout.NHCWB16,
            #layout=NpuLayout.NHWC,
            data_type=NpuDataType.INT8,
            fm_elem_size=1,
            fm_addr=V_MEM_ADDR,
            scale = V_MEM_SCALE,
            zero_point = V_MEM_ZERO_POINT,
            name="v_mem"
        )


        # Same scaling as VTH (since reset is either 0 or 1)
        ifm2 = create_feature_map(
            height=1, width=1, depth=OUTPUT_LAYER_SIZE,
            region=SRAM_SCRATCH_REGION,
            layout=NpuLayout.NHCWB16,
            #layout=NpuLayout.NHWC,
            data_type=NpuDataType.INT8,
            fm_elem_size=1,
            fm_addr=RESET_ADDR,
            scale = RESET_SCALE,
            zero_point = RESET_ZERO_POINT,
            name="reset"
        )


        ofm = create_feature_map(
            height=1, width=1, depth=OUTPUT_LAYER_SIZE,
            region=SRAM_SCRATCH_REGION,
            layout=NpuLayout.NHWC,
            data_type=NpuDataType.INT8,
            fm_elem_size=1,
            fm_addr=V_MEM_ADDR,
            scale = V_MEM_SCALE,
            zero_point = V_MEM_ZERO_POINT,
            name="v_mem_post_reset"
        )

        #block_config = NpuShape3D(2, 2, 32)

        activation = create_activation(
            activation_op=NpuActivationOp.NONE_OR_RELU,
            min_val=None,
            max_val=None,
        )




        sub_v_mem_reset_op = NpuElementWiseOperation(NpuElementWiseOp.SUB)
    
        #elementwise operation
        sub_v_mem_reset_op.reversed_operands = IFM2_IS_FIRST_OPERAND
        sub_v_mem_reset_op.rescale = None

        #NpuBlockOperation
        sub_v_mem_reset_op.ifm = ifm
        sub_v_mem_reset_op.ifm2 = ifm2
        sub_v_mem_reset_op.ifm2_scalar = None   #set if ifm2 is a scalar
        sub_v_mem_reset_op.ofm = ofm
        sub_v_mem_reset_op.kernel = None
        sub_v_mem_reset_op.weights = []
        sub_v_mem_reset_op.biases = []
        sub_v_mem_reset_op.padding = None
        sub_v_mem_reset_op.activation = activation

        block_config = get_block_config(sub_v_mem_reset_op, ACCELERATOR)
        sub_v_mem_reset_op.block_config = block_config
        sub_v_mem_reset_op.rounding_mode = NpuRoundingMode.TFL
        sub_v_mem_reset_op.fused_quantize = False
        sub_v_mem_reset_op.ifm_upscale = NpuResamplingMode.NONE
        sub_v_mem_reset_op.accumulator_type = NpuAccumulatorType.Default


        #check_block_config_legal(block_config, sub_v_mem_reset_op, ACCELERATOR)



        return sub_v_mem_reset_op
    


    def def_update_nxt_layer_reduce_sum_out_spk():


        ifm = create_feature_map(
            height=1, width=1, depth=OUTPUT_LAYER_SIZE,
            region=OUTPUT_REGION,
            layout=NpuLayout.NHWC,
            data_type=NpuDataType.INT8, fm_elem_size=1,
            fm_addr=OUT_SPK_ADDR,
            scale=OUT_SPK_SCALE, zero_point=OUT_SPK_ZERO_POINT,
            name="out_spk"
        )

        ofm = create_feature_map(
            height=1, width=1, depth=1,
            region=SRAM_SCRATCH_REGION,
            layout=NpuLayout.NHWC,
            data_type=NpuDataType.INT8, fm_elem_size=1,
            fm_addr=UPDATE_NXT_LAYER_ADDR,
            scale=UPDATE_NXT_LAYER_SCALE,
            zero_point=UPDATE_NXT_LAYER_ZERO_POINT,
            name="update_nxt_layer"
        )


        kernel = NpuKernel(
            w=1, h=1, stride_x=1, stride_y=1,
            dilation_x=1, dilation_y=1
        )

        padding = NpuPadding(top=0, left=0, bottom=0, right=0)
   
        #block_config = NpuShape3D(2, 2, 8)


        activation = create_activation(
            activation_op=NpuActivationOp.NONE_OR_RELU,
            min_val=None,
            max_val=None
        )

        update_nxt_layer_reduce_sum_op = NpuPoolingOperation(NpuPoolingOp.REDUCE_SUM)
    
        #Pooling operation
        update_nxt_layer_reduce_sum_op.rescale = None

        #NpuBlockOperation
        update_nxt_layer_reduce_sum_op.ifm = ifm
        update_nxt_layer_reduce_sum_op.ifm2 = None
        update_nxt_layer_reduce_sum_op.ifm2_scalar = None   #set if ifm2 is a scalar
        update_nxt_layer_reduce_sum_op.ofm = ofm
        update_nxt_layer_reduce_sum_op.kernel = kernel
        update_nxt_layer_reduce_sum_op.weights = []
        update_nxt_layer_reduce_sum_op.biases = []
        update_nxt_layer_reduce_sum_op.padding = padding
        update_nxt_layer_reduce_sum_op.activation = activation

        block_config = get_block_config(update_nxt_layer_reduce_sum_op, ACCELERATOR)
        update_nxt_layer_reduce_sum_op.block_config = block_config
        update_nxt_layer_reduce_sum_op.rounding_mode = NpuRoundingMode.TFL
        update_nxt_layer_reduce_sum_op.fused_quantize = False
        update_nxt_layer_reduce_sum_op.ifm_upscale = NpuResamplingMode.NONE
        update_nxt_layer_reduce_sum_op.accumulator_type = NpuAccumulatorType.Default

        #check_block_config_legal(block_config, update_nxt_layer_reduce_sum_op, ACCELERATOR)

        return update_nxt_layer_reduce_sum_op




    def def_reset_time():

        IFM2_IS_FIRST_OPERAND = False


        ifm = create_feature_map(
            height=1, width=1, depth=1,
            region=SRAM_SCRATCH_REGION,
            layout=NpuLayout.NHWC,
            data_type=NpuDataType.INT8,
            fm_elem_size=1,
            fm_addr=TIME_NOT_UPDATED_ADDR,
            scale = TIME_NOT_UPDATED_SCALE,
            zero_point = TIME_NOT_UPDATED_ZERO_POINT,
            name="time_not_updated"
        )

        ifm2 = NpuFeatureMap()
        ifm2.quantization = NpuQuantization(1, 0)
        ifm2_scalar = 0

        ofm = create_feature_map(
            height=1, width=1, depth=1,
            region=SRAM_SCRATCH_REGION,
            layout=NpuLayout.NHWC,
            data_type=NpuDataType.INT8,
            fm_elem_size=1,
            fm_addr=TIME_NOT_UPDATED_ADDR,
            scale = TIME_NOT_UPDATED_SCALE,
            zero_point = TIME_NOT_UPDATED_ZERO_POINT,
            name="time_not_updated"
        )

        #block_config = NpuShape3D(2, 2, 32)

        activation = create_activation(
            activation_op=NpuActivationOp.NONE_OR_RELU,
            min_val=0,
            max_val=0,
        )




        reset_time_op = NpuElementWiseOperation(NpuElementWiseOp.MUL)
    
        #elementwise operation
        reset_time_op.reversed_operands = IFM2_IS_FIRST_OPERAND
        reset_time_op.rescale = None

        #NpuBlockOperation
        reset_time_op.ifm = ifm
        reset_time_op.ifm2 = ifm2
        reset_time_op.ifm2_scalar = ifm2_scalar   #set if ifm2 is a scalar
        reset_time_op.ofm = ofm
        reset_time_op.kernel = None
        reset_time_op.weights = []
        reset_time_op.biases = []
        reset_time_op.padding = None
        reset_time_op.activation = activation

        block_config = get_block_config(reset_time_op, ACCELERATOR)
        reset_time_op.block_config = block_config
        reset_time_op.rounding_mode = NpuRoundingMode.TFL
        reset_time_op.fused_quantize = True 
        reset_time_op.ifm_upscale = NpuResamplingMode.NONE
        reset_time_op.accumulator_type = NpuAccumulatorType.Default


        #check_block_config_legal(block_config, reset_time_op, ACCELERATOR)



        return reset_time_op
    



    def def_increment_spk_list(OUT_SPK_SUM_ADDR,    OUT_SPK_SUM_SCALE,  OUT_SPK_SUM_ZERO_POINT,
                               OUT_SPK_ADDR,        OUT_SPK_SCALE,      OUT_SPK_ZERO_POINT):

        IFM2_IS_FIRST_OPERAND = False


        ifm = create_feature_map(
            height=1, width=1, depth=OUTPUT_LAYER_SIZE,
            region=SRAM_SCRATCH_REGION,
            layout=NpuLayout.NHWC,
            data_type=NpuDataType.INT8,
            fm_elem_size=1,
            fm_addr=OUT_SPK_SUM_ADDR,
            scale = OUT_SPK_SUM_SCALE,
            zero_point = OUT_SPK_SUM_ZERO_POINT,
            name="out_spk_sum"
        )


        # Same scaling as VTH (since reset is either 0 or 1)
        ifm2 = create_feature_map(
            height=1, width=1, depth=OUTPUT_LAYER_SIZE,
            region=OUTPUT_REGION,
            layout=NpuLayout.NHWC,
            data_type=NpuDataType.INT8,
            fm_elem_size=1,
            fm_addr=OUT_SPK_ADDR,
            scale = OUT_SPK_SCALE,
            zero_point = OUT_SPK_ZERO_POINT,
            name="out_spk"
        )


        ofm = create_feature_map(
            height=1, width=1, depth=OUTPUT_LAYER_SIZE,
            region=SRAM_SCRATCH_REGION,
            layout=NpuLayout.NHWC,
            data_type=NpuDataType.INT8,
            fm_elem_size=1,
            fm_addr=OUT_SPK_SUM_ADDR,
            scale = OUT_SPK_SUM_SCALE,
            zero_point = OUT_SPK_SUM_ZERO_POINT,
            name="out_spk_sum"
        )

        #block_config = NpuShape3D(2, 2, 32)

        activation = create_activation(
            activation_op=NpuActivationOp.NONE_OR_RELU,
            min_val=None,
            max_val=None,
        )




        incr_out_spk_sum_op = NpuElementWiseOperation(NpuElementWiseOp.ADD)
    
        #elementwise operation
        incr_out_spk_sum_op.reversed_operands = IFM2_IS_FIRST_OPERAND
        incr_out_spk_sum_op.rescale = None

        #NpuBlockOperation
        incr_out_spk_sum_op.ifm = ifm
        incr_out_spk_sum_op.ifm2 = ifm2
        incr_out_spk_sum_op.ifm2_scalar = None   #set if ifm2 is a scalar
        incr_out_spk_sum_op.ofm = ofm
        incr_out_spk_sum_op.kernel = None
        incr_out_spk_sum_op.weights = []
        incr_out_spk_sum_op.biases = []
        incr_out_spk_sum_op.padding = None
        incr_out_spk_sum_op.activation = activation

        block_config = get_block_config(incr_out_spk_sum_op, ACCELERATOR)
        incr_out_spk_sum_op.block_config = block_config
        incr_out_spk_sum_op.rounding_mode = NpuRoundingMode.TFL
        incr_out_spk_sum_op.fused_quantize = False
        incr_out_spk_sum_op.ifm_upscale = NpuResamplingMode.NONE
        incr_out_spk_sum_op.accumulator_type = NpuAccumulatorType.Default


        #check_block_config_legal(block_config, incr_out_spk_sum_op, ACCELERATOR)



        return incr_out_spk_sum_op
    



    def layer_merge_and_write(cms_name, header_out_filepath, lif_params_on_sram):

        # Define the individual NPU Operations
        dma_lut_op, exp_mul_lnb_time_op, decay_lut_values, decay_lut_index = def_decay_lut()


        mul_decay_op = def_mul_decay_Vmem()
        add_decayed_mem_in_curr = def_add_decayed_mem_in_curr()
        check_spk_lut_dma_op, check_spk_sub_v_mem_updated_vth, check_spk_lut_values, check_spk_lut_index = def_check_spk_sub_v_mem_updated_vth()
        reset_mul_vth_out_spk_op = def_mul_vth_out_spk()
        sub_v_mem_reset_op = def_sub_mem_updated_reset()
        update_nxt_layer_reduce_sum_out_spk = def_update_nxt_layer_reduce_sum_out_spk()
        reset_time_op = def_reset_time()




        #npu_op_list = [dma_lut_op, exp_mul_lnb_time_op, dma_op, fully_connected_op, mul_decay_op, add_decayed_mem_in_curr, check_spk_lut_dma_op, check_spk_sub_v_mem_updated_vth, reset_mul_vth_out_spk_op, sub_v_mem_reset_op, update_nxt_layer_reduce_sum_out_spk, reset_time_op]
        npu_op_list = [dma_lut_op, exp_mul_lnb_time_op]

        if (not weights_and_biases_on_sram):
            fully_connected_op, _ , weight_byte_arr, bias_byte_arr = def_fullyconnected(IN_SPK_ADDR, IN_CURR_ADDR, weights_and_biases_on_sram)
            npu_op_list.append(fully_connected_op)
        else:
            fully_connected_op, dma_op, weight_byte_arr, bias_byte_arr = def_fullyconnected(IN_SPK_ADDR, IN_CURR_ADDR, weights_and_biases_on_sram)
            npu_op_list.extend([dma_op, fully_connected_op])

        
        npu_op_list.extend([mul_decay_op, add_decayed_mem_in_curr, check_spk_lut_dma_op, check_spk_sub_v_mem_updated_vth, reset_mul_vth_out_spk_op, sub_v_mem_reset_op, update_nxt_layer_reduce_sum_out_spk, reset_time_op])



        if (is_last_layer):
            incr_out_spk_sum_op = def_increment_spk_list(OUT_SPK_SUM_ADDR,  OUT_SPK_SUM_SCALE,   OUT_SPK_SUM_ZERO_POINT,
                                                         OUT_SPK_ADDR,      OUT_SPK_SCALE,      OUT_SPK_ZERO_POINT)
            
            npu_op_list.append(incr_out_spk_sum_op)




        # Merge
        lut_arr_contents_str = merge_lut_values_to_str([(decay_lut_values, decay_lut_index), (check_spk_lut_values, check_spk_lut_index)])
        lif_params_arr_contents_str = merge_lif_params_to_str(LN_BETA_QUANT_LIST, VTH_QUANT_LIST)
        cms_bytearr, register_cms = gen_cms(npu_op_list, ACCELERATOR, DEBUG_MODE)

        # Generate Dicts for writing to C
        sizes_dict, addr_dict, quant_param_dict = generate_dict_for_writing_defines_to_C_files(cms_name=cms_name, weight_byte_arr=weight_byte_arr, bias_byte_arr=bias_byte_arr)



        write_cms_to_files(header_out_filepath, lif_params_on_sram, cms_bytearr, register_cms, cms_name, sizes_dict, addr_dict, quant_param_dict, lif_params_arr_contents_str, lut_arr_contents_str, weight_byte_arr=weight_byte_arr, bias_byte_arr=bias_byte_arr)
    

    layer_merge_and_write(cms_name, header_out_filepath, lif_params_on_sram)





