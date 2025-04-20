from ethosu.vela.api import *

#from config_ops import create_feature_map, create_activation
from config_ops import *
from extra_func import *


import numpy as np


INPUT_LAYER_SIZE = 16
OUTPUT_LAYER_SIZE = 32

DEBUG_MODE = True



ACCELERATOR = NpuAccelerator.Ethos_U55_256
CMS_NAME = "my_mem_u"





TENSOR_ARENA_SIZE	=	704
		
IN_SPK_ADDR	=	0
BIAS_ADDR	=	16
WEIGHT_ADDR	=	336
LN_BETA_ADDR	=	480
VTH_ADDR	=	512
V_MEM_ADDR	=	544
TIME_NOT_UPDATED_ADDR	=	576
TMP1_ADDR	=	608
TMP2_ADDR	=	640
OUT_SPK_ADDR	=	672



# Tensor arena allocation
'''
Start Address	Tensor name	Size
0	In_spk	16
16	Bias	320
336	Weights	144
480	curr (tmp)	32
512	v_mem	32
544	decay	32

576	decayed_mem (tmp)	32

'''



# Set FM Quantization Params

IN_SPK_MAX_VAL = 1
IN_SPK_MIN_VAL = 0

#Must be symmetric
WEIGHT_MAX_VAL = 127/100
WEIGHT_MIN_VAL = -128/100

LN_BETA_MAX_VAL = 0
LN_BETA_MIN_VAL = -4

TIME_NOT_UPDATED_MAX_VAL = 16
TIME_NOT_UPDATED_MIN_VAL = 0

IN_CURR_MAX_VAL = 4
IN_CURR_MIN_VAL = -0.5

V_MEM_MAX_VAL = 4
V_MEM_MIN_VAL = 0

DECAY_MAX_VAL = 1
DECAY_MIN_VAL = 0

DECAYED_MEM_MAX_VAL = 1
DECAYED_MEM_MIN_VAL = 0

VTH_MAX_VAL = 2
VTH_MIN_VAL = 0.5


CHECK_SPK_SUB_INNER_MAX_VAL = 3
CHECK_SPK_SUB_INNER_MIN_VAL = 0




##############
# Assign Tmp tensors here!
DECAY_ADDR = TMP1_ADDR
IN_CURR_ADDR = TMP2_ADDR
DECAYED_MEM_ADDR = TMP1_ADDR

##############


ADDR_DICT = {
    # Input Feature map
    CMS_NAME.upper()+"_IN_SPK_ADDR" : IN_SPK_ADDR,

    # Layer params
    CMS_NAME.upper()+"_BIAS_ADDR" : BIAS_ADDR,
    CMS_NAME.upper()+"_WEIGHT_ADDR" : WEIGHT_ADDR,
    CMS_NAME.upper()+"_LN_BETA_ADDR" : LN_BETA_ADDR,
    CMS_NAME.upper()+"_VTH_ADDR" : VTH_ADDR,

    # Layer status
    CMS_NAME.upper()+"_V_MEM_ADDR" : V_MEM_ADDR,
    CMS_NAME.upper()+"_TIME_NOT_UPDATED_ADDR" : TIME_NOT_UPDATED_ADDR,

    # TMP Feature maps
    CMS_NAME.upper()+"_IN_CURR_ADDR" : IN_CURR_ADDR,
    CMS_NAME.upper()+"_DECAY_ADDR" : DECAY_ADDR,
    CMS_NAME.upper()+"_DECAYED_MEM_ADDR" : DECAYED_MEM_ADDR

    # Output Feature Map

}


IN_SPK_SCALE, IN_SPK_ZERO_POINT = zero_point_quant(IN_SPK_MAX_VAL, IN_SPK_MIN_VAL)


# Layer params
LN_BETA_SCALE, LN_BETA_ZERO_POINT = zero_point_quant(LN_BETA_MAX_VAL, LN_BETA_MIN_VAL)
VTH_SCALE, VTH_ZERO_POINT = zero_point_quant(VTH_MAX_VAL, VTH_MIN_VAL)

# Layer status
V_MEM_SCALE, V_MEM_ZERO_POINT = zero_point_quant(V_MEM_MAX_VAL, V_MEM_MIN_VAL)
TIME_NOT_UPDATED_SCALE, TIME_NOT_UPDATED_ZERO_POINT = zero_point_quant(TIME_NOT_UPDATED_MAX_VAL, TIME_NOT_UPDATED_MIN_VAL)

# TMP Feature maps
DECAY_SCALE, DECAY_ZERO_POINT = zero_point_quant(DECAY_MAX_VAL, DECAY_MIN_VAL)
IN_CURR_SCALE, IN_CURR_ZERO_POINT = zero_point_quant(IN_CURR_MAX_VAL, IN_CURR_MIN_VAL)
DECAYED_MEM_SCALE, DECAYED_MEM_ZERO_POINT = zero_point_quant(DECAYED_MEM_MAX_VAL, DECAYED_MEM_MIN_VAL)

WEIGHT_SCALE, WEIGHT_ZERO_POINT = symmetric_zero_point_quant(WEIGHT_MAX_VAL, WEIGHT_MIN_VAL)
BIAS_SCALE, BIAS_ZERO_POINT = IN_SPK_SCALE*WEIGHT_SCALE/IN_CURR_SCALE, 0

CHECK_SPK_SUB_INNER_SCALE, CHECK_SPK_SUB_INNER_ZERO_POINT = zero_point_quant(CHECK_SPK_SUB_INNER_MAX_VAL, CHECK_SPK_SUB_INNER_MIN_VAL)

QUANT_PARAM_DICT = {
    CMS_NAME.upper()+"_IN_SPK_SCALE" : IN_SPK_SCALE,
    CMS_NAME.upper()+"_IN_SPK_ZERO_POINT" : IN_SPK_ZERO_POINT,


    CMS_NAME.upper()+"_BIAS_SCALE" : BIAS_SCALE,
    CMS_NAME.upper()+"_BIAS_ZERO_POINT" : BIAS_ZERO_POINT,
    CMS_NAME.upper()+"_WEIGHT_SCALE" : WEIGHT_SCALE,
    CMS_NAME.upper()+"_WEIGHT_ZERO_POINT" : WEIGHT_ZERO_POINT,


    CMS_NAME.upper()+"_LN_BETA_SCALE" : LN_BETA_SCALE,
    CMS_NAME.upper()+"_LN_BETA_ZERO_POINT" : LN_BETA_ZERO_POINT,
    CMS_NAME.upper()+"_VTH_SCALE" : VTH_SCALE,
    CMS_NAME.upper()+"_VTH_ZERO_POINT" : VTH_ZERO_POINT,
    CMS_NAME.upper()+"_V_MEM_SCALE" : V_MEM_SCALE,
    CMS_NAME.upper()+"_V_MEM_ZERO_POINT" : V_MEM_ZERO_POINT,
    CMS_NAME.upper()+"_TIME_NOT_UPDATED_SCALE" : TIME_NOT_UPDATED_SCALE,
    CMS_NAME.upper()+"_TIME_NOT_UPDATED_ZERO_POINT" : TIME_NOT_UPDATED_ZERO_POINT,

    CMS_NAME.upper()+"_DECAY_SCALE" : DECAY_SCALE,
    CMS_NAME.upper()+"_DECAY_ZERO_POINT" : DECAY_ZERO_POINT,
    CMS_NAME.upper()+"_IN_CURR_SCALE" : IN_CURR_SCALE,
    CMS_NAME.upper()+"_IN_CURR_ZERO_POINT" : IN_CURR_ZERO_POINT,
    CMS_NAME.upper()+"_DECAYED_MEM_SCALE" : DECAYED_MEM_SCALE,
    CMS_NAME.upper()+"_DECAYED_MEM_ZERO_POINT" : DECAYED_MEM_ZERO_POINT,
}





def def_decay_lut():

    IFM2_IS_FIRST_OPERAND = False

    ifm = create_feature_map(
        height=1, width=1, depth=OUTPUT_LAYER_SIZE,
        region=1,
        layout=NpuLayout.NHWC,
        data_type=NpuDataType.INT8,
        fm_elem_size=1,
        fm_addr=LN_BETA_ADDR,
        scale=LN_BETA_SCALE,
        zero_point=LN_BETA_ZERO_POINT,
        name="ln_beta"
    )


    ifm2 = create_feature_map(
        height=1, width=1, depth=OUTPUT_LAYER_SIZE,
        region=1,
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
        region=1,
        layout=NpuLayout.NHWC,
        data_type=NpuDataType.INT8,
        fm_elem_size=1,
        fm_addr=DECAY_ADDR,
        scale=DECAY_SCALE,
        zero_point=DECAY_ZERO_POINT,
        name="decay"
    )

    #DMA for LUT
    decay_lut_index = 0
    decay_lut_len = 256

    if ofm.data_type == NpuDataType.INT8 or ofm.data_type == NpuDataType.UINT8:
        lut_slot_size = 256
    else:
        print("Not using INT8 --> dont know how big each LUT is")
        exit()

    from ethosu.vela.architecture_features import create_default_arch
    from ethosu.vela.architecture_features import Accelerator
    from ethosu.vela.architecture_features import ArchitectureFeatures
    from ethosu.vela.register_command_stream_util import BASE_PTR_INDEX_MEM2MEM

    import math
    import tensorflow as tf


    default_arch = create_default_arch(Accelerator.from_npu_accelerator(ACCELERATOR))
    # LUT stored at the end of the Shared Buffer
    lut_shared_buffer_start_addr = default_arch.shram_lut_address

    # LUT storage = start of LUT segment in shared buffer + which lut it is
    decay_lut_addr = lut_shared_buffer_start_addr + decay_lut_index * lut_slot_size
    dma_src = NpuAddressRange(region=3, address=0, length=decay_lut_len)
    dma_dst = NpuAddressRange(region=BASE_PTR_INDEX_MEM2MEM, address=decay_lut_addr, length=decay_lut_len)
    #dma_dst = NpuAddressRange(region=1, address=IN_SPK_ADDR, length=decay_lut_len)
    dma_lut_op = NpuDmaOperation(src=dma_src, dest=dma_dst)


    block_config = NpuShape3D(2, 2, 32)

    activation = create_activation(
        activation_op=NpuActivationOp.TABLE_LOOKUP,
        min_val=None,
        max_val=None,
        lookup_table_index=decay_lut_index
    )


    #int8 --> [-128, 127], uint8 --> [0, 255]
    decay_lut = []
    #scale_in = ifm.quantization.scale_f32 * ifm2.quantization.scale_f32
    scale_out = ofm.quantization.scale_f32
    zero_point_out = ofm.quantization.zero_point
    y_quant_max = 127
    y_quant_min = -128
    for x_quant in range(-128, 128):
        print("x_quant", x_quant)
        x_real = scale * (x_quant - zero_point)
        y_real = math.exp(x_real)
        print("x_real", x_real)
        print("y_real", y_real)
        y_quant = tf.round(y_real / scale) + zero_point
        print("y_quant", y_quant)

        # Cap
        if y_quant < y_quant_min:
            y_quant = y_quant_min
        elif y_quant > y_quant_max:
            y_quant = y_quant_max

        decay_lut.append(y_quant)


    
    #now print it so i can copy it in
    #for val in decay_lut:
        #print(str(int(tf.get_static_value(val))) + ",")

    
   # activation = create_activation(
        #activation_op=NpuActivationOp.NONE_OR_RELU,
        #min_val=None,
        #max_val=None
        ##lookup_table_index=decay_lut_index
    #)


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
    exp_mul_lnb_time_op.block_config = block_config
    exp_mul_lnb_time_op.rounding_mode = NpuRoundingMode.TFL
    exp_mul_lnb_time_op.fused_quantize = False
    exp_mul_lnb_time_op.ifm_upscale = NpuResamplingMode.NONE
    exp_mul_lnb_time_op.accumulator_type = NpuAccumulatorType.Default


    check_block_config_legal(block_config, exp_mul_lnb_time_op, ACCELERATOR)

    return dma_lut_op, exp_mul_lnb_time_op


def def_fullyconnected():




    ifm = create_feature_map(
    height=1, width=1, depth=INPUT_LAYER_SIZE,
    region=1,
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
        region=1,
        layout=NpuLayout.NHWC,
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

    block_config = NpuShape3D(2, 2, 32)
    block_traversal = NpuBlockTraversal.DEPTH_FIRST


    # Define Weights


    weights_volume_ohwi = np.ones((ofm.shape.depth, kernel.height, kernel.width, ifm.shape.depth))
    #weights_volume_ohwi=np.zeros((ofm.shape.depth, kernel.height, kernel.width, ifm.shape.depth), dtype=np.int8)
    #print("weights_volume_ohwi", weights_volume_ohwi.shape)
    #print(weights_volume_ohwi)
    #weights_volume_ohwi = np.tile(np.arange(1, 5), (32, 4)).reshape(ofm.shape.depth, kernel.height, kernel.width, ifm.shape.depth).astype(np.int8)  # Repeat [1, 2, 3, 4] across 32 rows
    #weights_volume_ohwi = np.ones((OFM_DEPTH, KERNEL_HEIGHT, KERNEL_WIDTH, IFM_DEPTH)).astype(np.uint8)
    if ifm.data_type == NpuDataType.INT8:
        weight_ifm_bitdepth = 8 #int16
    elif ifm.data_type == NpuDataType.INT16:
        weight_ifm_bitdepth = 16 #int16


    #scale = int(1/0.003937007859349251)
    #print("scale", scale)

    #Biases
    bias_list = []
    for i in range(ofm.shape.depth):
    #    #bias_list.append(np.int64(i%4))
        bias_list.append(np.int64(0))




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
                            ifm_zero_point=ifm.quantization.zero_point,
                            weight_scale=WEIGHT_SCALE,
                            weight_zero_point=WEIGHT_ZERO_POINT,
                            ofm_scale=ofm.quantization.scale_f32,
                            ofm_zero_point=ofm.quantization.zero_point,


                            is_debug_mode=DEBUG_MODE
    )

    weight_n_bias_len = len(bias_byte_arr) + len(weight_byte_arr)
    if DEBUG_MODE:
        print("weight_n_bias_len", weight_n_bias_len)
        print("\tbias_len:", len(bias_byte_arr))
        print("\tweight_len", len(weight_byte_arr))

    

    ADDR_DICT[CMS_NAME.upper()+"_WEIGHT_LEN"] = len(weight_byte_arr)
    ADDR_DICT[CMS_NAME.upper()+"_BIAS_LEN"]  = len(bias_byte_arr)
        


    #BIAS_ADDR = WEIGHT_N_BIAS_ADDR
    #WEIGHT_ADDR = BIAS_ADDR + len(bias_byte_arr)
    
    
    WEIGHT_N_BIAS_ADDR = BIAS_ADDR #Bias before weights

    #DMA    
    dma_src = NpuAddressRange(region=0, address=0, length=weight_n_bias_len)
    dma_dst = NpuAddressRange(region=1, address=WEIGHT_N_BIAS_ADDR, length=weight_n_bias_len)
    dma_op = NpuDmaOperation(src=dma_src, dest=dma_dst)





    

    padding = NpuPadding(top=0, left=0, bottom=0, right=0)


    weights = NpuAddressRange(region=1, address=WEIGHT_ADDR, length=len(weight_byte_arr))
    biases = NpuAddressRange(region=1, address=BIAS_ADDR, length=len(bias_byte_arr))

    


    activation = create_activation(
        activation_op=NpuActivationOp.NONE_OR_RELU,
        min_val=None,
        max_val=None,
    )

    #activation = create_activation(
        #activation_op=NpuActivationOp.TABLE_LOOKUP,
        #min_val=None,
        #max_val=None,
        #lookup_table_index=0
    #)
    


    fused_quantize = False












    my_op = NpuConv2DOperation()
    my_op.ifm               =   ifm
    my_op.ifm2              =   ifm2
    my_op.ifm2_scalar       =   None
    my_op.ofm               =   ofm
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


    check_block_config_legal(block_config, my_op, ACCELERATOR)





    return my_op, dma_op, weight_byte_arr, bias_byte_arr, 





def def_mul_decay_Vmem():
    
    IFM2_IS_FIRST_OPERAND = False


    ifm = create_feature_map(
        height=1, width=1, depth=OUTPUT_LAYER_SIZE,
        region=1,
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
        region=1,
        layout=NpuLayout.NHWC,
        data_type=NpuDataType.INT8,
        fm_elem_size=1,
        fm_addr=DECAY_ADDR,
        scale = DECAY_SCALE,
        zero_point = DECAY_ZERO_POINT,
        name="decay"
    )



    ofm = create_feature_map(
        height=1, width=1, depth=OUTPUT_LAYER_SIZE,
        region=1,
        #layout=NpuLayout.NHCWB16,
        layout=NpuLayout.NHWC,
        data_type=NpuDataType.INT8,
        fm_elem_size=1,
        fm_addr=DECAYED_MEM_ADDR,
        scale = DECAYED_MEM_SCALE,
        zero_point = DECAYED_MEM_ZERO_POINT,
        name="decayed_mem"
    )

    block_config = NpuShape3D(2, 2, 32)

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
    mul_decay_op.block_config = block_config
    mul_decay_op.rounding_mode = NpuRoundingMode.TFL
    mul_decay_op.fused_quantize = False
    mul_decay_op.ifm_upscale = NpuResamplingMode.NONE
    mul_decay_op.accumulator_type = NpuAccumulatorType.Default


    check_block_config_legal(block_config, mul_decay_op, ACCELERATOR)



    return mul_decay_op




def def_add_decayed_mem_in_curr():
    IFM2_IS_FIRST_OPERAND = False

    ifm = create_feature_map(
        height=1, width=1, depth=OUTPUT_LAYER_SIZE,
        region=1,
        layout=NpuLayout.NHWC,
        data_type=NpuDataType.INT8,
        fm_elem_size=1,
        fm_addr=DECAYED_MEM_ADDR,
        scale = DECAYED_MEM_SCALE,
        zero_point = DECAYED_MEM_ZERO_POINT,
        name="decayed_mem"
    )

    ifm2 = create_feature_map(
        height=1, width=1, depth=OUTPUT_LAYER_SIZE,
        region=1,
        layout=NpuLayout.NHWC,
        data_type=NpuDataType.INT8,
        fm_elem_size=1,
        fm_addr=IN_CURR_ADDR,
        scale = IN_CURR_SCALE,
        zero_point = IN_CURR_ZERO_POINT,
        name="in_curr"
    )




    ofm = create_feature_map(
        height=1, width=1, depth=OUTPUT_LAYER_SIZE,
        region=1,
        #layout=NpuLayout.NHCWB16,
        layout=NpuLayout.NHWC,
        data_type=NpuDataType.INT8,
        fm_elem_size=1,
        fm_addr=V_MEM_ADDR,
        scale=V_MEM_SCALE,
        zero_point=V_MEM_SCALE,
        name="updated_mem"
    )

    block_config = NpuShape3D(2, 2, 32)

    activation = create_activation(
        activation_op=NpuActivationOp.NONE_OR_RELU,
        min_val=None,
        max_val=None,
    )




    add_decayed_mem_in_curr = NpuElementWiseOperation(NpuElementWiseOp.MUL)

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
    add_decayed_mem_in_curr.block_config = block_config
    add_decayed_mem_in_curr.rounding_mode = NpuRoundingMode.TFL
    add_decayed_mem_in_curr.fused_quantize = False
    add_decayed_mem_in_curr.ifm_upscale = NpuResamplingMode.NONE
    add_decayed_mem_in_curr.accumulator_type = NpuAccumulatorType.Default


    check_block_config_legal(block_config, mul_decay_op, ACCELERATOR)



    return add_decayed_mem_in_curr


'''
def def_relu_sub_inner_vth_vmem():
    IFM2_IS_FIRST_OPERAND = False

    ifm = create_feature_map(
        height=1, width=1, depth=OUTPUT_LAYER_SIZE,
        region=1,
        layout=NpuLayout.NHWC,
        data_type=NpuDataType.INT8,
        fm_elem_size=1,
        fm_addr=VTH_ADDR,
        scale = VTH_SCALE,
        zero_point = VTH_ZERO_POINT,
        name="v_th"
    )

    ifm2 = create_feature_map(
        height=1, width=1, depth=OUTPUT_LAYER_SIZE,
        region=1,
        layout=NpuLayout.NHWC,
        data_type=NpuDataType.INT8,
        fm_elem_size=1,
        fm_addr=V_MEM_ADDR,
        scale = V_MEM_SCALE,
        zero_point = V_MEM_ZERO_POINT,
        name="v_mem_post_update_pre_rst"
    )


    ofm = create_feature_map(
        height=1, width=1, depth=OUTPUT_LAYER_SIZE,
        region=1,
        #layout=NpuLayout.NHCWB16,
        layout=NpuLayout.NHWC,
        data_type=NpuDataType.INT8,
        fm_elem_size=1,
        fm_addr=CHECK_SPK_SUB_INNER_ADDR,
        scale=CHECK_SPK_SUB_INNER_SCALE,
        zero_point=CHECK_SPK_SUB_INNER_ZERO_POINT,
        name="check_spk_sub_inner"
    )

    block_config = NpuShape3D(2, 2, 32)


    # ReLu
    activation = create_activation(
        activation_op=NpuActivationOp.NONE_OR_RELU,
        min_val=0,
        max_val=None,
    )




    check_spk_sub_inner_op = NpuElementWiseOperation(NpuElementWiseOp.SUB)

    #elementwise operation
    check_spk_sub_inner_op.reversed_operands = IFM2_IS_FIRST_OPERAND
    check_spk_sub_inner_op.rescale = None

    #NpuBlockOperation
    check_spk_sub_inner_op.ifm = ifm
    check_spk_sub_inner_op.ifm2 = ifm2
    check_spk_sub_inner_op.ifm2_scalar = None   #set if ifm2 is a scalar
    check_spk_sub_inner_op.ofm = ofm
    check_spk_sub_inner_op.kernel = None
    check_spk_sub_inner_op.weights = []
    check_spk_sub_inner_op.biases = []
    check_spk_sub_inner_op.padding = None
    check_spk_sub_inner_op.activation = activation
    check_spk_sub_inner_op.block_config = block_config
    check_spk_sub_inner_op.rounding_mode = NpuRoundingMode.TFL
    check_spk_sub_inner_op.fused_quantize = False
    check_spk_sub_inner_op.ifm_upscale = NpuResamplingMode.NONE
    check_spk_sub_inner_op.accumulator_type = NpuAccumulatorType.Default


    check_block_config_legal(block_config, check_spk_sub_inner_op, ACCELERATOR)



    return check_spk_sub_inner_op
'''







'''
def def_relu_sub_outer_vth_vmem():
    IFM2_IS_FIRST_OPERAND = False

    ifm = create_feature_map(
        height=1, width=1, depth=OUTPUT_LAYER_SIZE,
        region=1,
        layout=NpuLayout.NHWC,
        data_type=NpuDataType.INT8,
        fm_elem_size=1,
        fm_addr=VTH_ADDR,
        scale = VTH_SCALE,
        zero_point = VTH_ZERO_POINT,
        name="v_th"
    )

    ifm2 = create_feature_map(
        height=1, width=1, depth=OUTPUT_LAYER_SIZE,
        region=1,
        layout=NpuLayout.NHWC,
        data_type=NpuDataType.INT8,
        fm_elem_size=1,
        fm_addr=SPK_SUB_TMP_ADDR,
        scale = SPK_SUB_TMP_SCALE,
        zero_point = SPK_SUB_TMP_ZERO_POINT,
        name="spk_sub_tmp"
    )


    ofm = create_feature_map(
        height=1, width=1, depth=OUTPUT_LAYER_SIZE,
        region=1,
        #layout=NpuLayout.NHCWB16,
        layout=NpuLayout.NHWC,
        data_type=NpuDataType.INT8,
        fm_elem_size=1,
        fm_addr=CHECK_SPK_SUB_INNER_ADDR,
        scale=CHECK_SPK_SUB_INNER_SCALE,
        zero_point=CHECK_SPK_SUB_INNER_ZERO_POINT,
        name="check_spk_sub_inner"
    )

    block_config = NpuShape3D(2, 2, 32)


    # ReLu
    activation = create_activation(
        activation_op=NpuActivationOp.NONE_OR_RELU,
        min_val=0,
        max_val=None,
    )




    check_spk_sub_inner_op = NpuElementWiseOperation(NpuElementWiseOp.SUB)

    #elementwise operation
    check_spk_sub_inner_op.reversed_operands = IFM2_IS_FIRST_OPERAND
    check_spk_sub_inner_op.rescale = None

    #NpuBlockOperation
    check_spk_sub_inner_op.ifm = ifm
    check_spk_sub_inner_op.ifm2 = ifm2
    check_spk_sub_inner_op.ifm2_scalar = None   #set if ifm2 is a scalar
    check_spk_sub_inner_op.ofm = ofm
    check_spk_sub_inner_op.kernel = None
    check_spk_sub_inner_op.weights = []
    check_spk_sub_inner_op.biases = []
    check_spk_sub_inner_op.padding = None
    check_spk_sub_inner_op.activation = activation
    check_spk_sub_inner_op.block_config = block_config
    check_spk_sub_inner_op.rounding_mode = NpuRoundingMode.TFL
    check_spk_sub_inner_op.fused_quantize = False
    check_spk_sub_inner_op.ifm_upscale = NpuResamplingMode.NONE
    check_spk_sub_inner_op.accumulator_type = NpuAccumulatorType.Default


    check_block_config_legal(block_config, check_spk_sub_inner_op, ACCELERATOR)



    return check_spk_sub_inner_op
'''









if __name__ == '__main__':
    dma_lut_op, exp_mul_lnb_time_op = def_decay_lut()
    fully_connected_op, dma_op, weight_byte_arr, bias_byte_arr = def_fullyconnected()
    mul_decay_op = def_mul_decay_Vmem()
    add_decayed_mem_in_curr = def_add_decayed_mem_in_curr()
    
    #check_spk_sub_inner_op = def_relu_sub_inner_vth_vmem()

    #npu_op_list = [dma_op, fully_connected_op, mul_decay_op, add_decayed_mem_in_curr]
    #npu_op_list = [dma_lut_op, exp_mul_lnb_time_op, dma_op, fully_connected_op]
    npu_op_list = [dma_lut_op, exp_mul_lnb_time_op]


    check_weight_and_bias_len_correct(CMS_NAME, ADDR_DICT, weight_byte_arr, bias_byte_arr)

    cms_bytearr, register_cms = gen_cms(npu_op_list, ACCELERATOR, DEBUG_MODE)

    header_out_filepath = "../../snn_on_alif_e7/simple_code_test/include/" + CMS_NAME + ".h"
    imp_out_filepath = "../../snn_on_alif_e7/simple_code_test/nn_ops/" + CMS_NAME + ".c"
    write_cms_to_files(header_out_filepath, imp_out_filepath, npu_op_list, cms_bytearr, register_cms, CMS_NAME, ADDR_DICT, QUANT_PARAM_DICT, weight_byte_arr=weight_byte_arr, bias_byte_arr=bias_byte_arr)
    