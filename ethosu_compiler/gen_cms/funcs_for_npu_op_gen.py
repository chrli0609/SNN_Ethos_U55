import math
import os, sys



sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ethosu.vela.api import *



from constants import *
from data_structs import MemoryAllocator, Tensor 
from config_ops import *
from extra_func import get_block_config






def get_int8_fc_weights_and_biases(
        weights_volume_ohwi, 
        bias_list,
        input_size,
        output_size,

        ifm_scale, ofm_scale,

        accelerator,
        debug_mode
    ):

    from config_ops import create_feature_map
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

    from config_ops import symmetric_zero_point_quant
    # Set Weights Quantization params, it must be symmetric quantization 
    max_weight_val = np.max(weights_volume_ohwi)
    min_weight_val = np.min(weights_volume_ohwi)
    # Get the one with the largest absolute value (since the npu only supports symmetric quantization for weights)
    if abs(min_weight_val) > abs(max_weight_val):
        largest_weight_abs_val = abs(min_weight_val)
    else:
        largest_weight_abs_val = abs(max_weight_val)
    weight_scale, weight_zero_point = symmetric_zero_point_quant(largest_weight_abs_val, -largest_weight_abs_val)

    
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
                    #accumulator=0
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
                           weight_tensor: Tensor, bias_tensor: Tensor,
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
        #cms_name.upper()+"_MAX_NUM_TENSORS_TO_TRACK"   : len(mem_alloc.regions[SRAM_SCRATCH_REGION_NAME].memory_map),    #num tensors in sram scratchpad
        cms_name.upper()+"_MAX_NUM_TENSORS_TO_TRACK"   : 8,    #num tensors in sram scratchpad

        cms_name.upper()+"_TENSOR_ARENA_SIZE "  : mem_alloc.regions[SRAM_SCRATCH_REGION_NAME].size,

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


    for fm in mem_alloc.tensors:
        quant_param_dict[cms_name.upper()+"_"+fm.name.upper()+"_SCALE"] = fm.quantization.scale_f32
        quant_param_dict[cms_name.upper()+"_"+fm.name.upper()+"_ZERO_POINT"] = fm.quantization.zero_point


    return sizes_dict, addr_dict, quant_param_dict