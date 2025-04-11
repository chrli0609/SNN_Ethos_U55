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



#TENSOR_ARENA_SIZE = 16 + 464 + 32 


TENSOR_ARENA_SIZE	=	608
		
		
IN_SPK_ADDR	=	0
BIAS_ADDR	=	16
WEIGHT_ADDR	=	336
IN_CURR_ADDR	=	480
V_MEM_ADDR	=	512
DECAY_ADDR	=	544
DECAYED_MEM_ADDR	=	576

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

IN_CURR_MAX_VAL = 10
IN_CURR_MIN_VAL = 0

V_MEM_MAX_VAL = 3
V_MEM_MIN_VAL = 0

DECAY_MAX_VAL = 1
DECAY_MIN_VAL = 0

DECAYED_MEM_MAX_VAL = 3
DECAYED_MEM_MIN_VAL = 0









ADDR_DICT = {
    CMS_NAME.upper()+"_IN_SPK_ADDR" : IN_SPK_ADDR,
    CMS_NAME.upper()+"_WEIGHT_ADDR" : WEIGHT_ADDR,
    CMS_NAME.upper()+"_BIAS_ADDR" : BIAS_ADDR,
    CMS_NAME.upper()+"_IN_CURR_ADDR" : IN_CURR_ADDR,
    CMS_NAME.upper()+"_V_MEM_ADDR" : V_MEM_ADDR,
    CMS_NAME.upper()+"_DECAY_ADDR" : DECAY_ADDR,
    CMS_NAME.upper()+"_DECAYED_MEM_ADDR" : DECAYED_MEM_ADDR
}



def def_fullyconnected():



    #IN_SPK_ADDR = 0
    

    #WEIGHT_N_BIAS_ADDR = 0 + INPUT_LAYER_SIZE
    #IN_CURR_ADDR = WEIGHT_N_BIAS_ADDR + 464






    ifm = create_feature_map(
    height=1, width=1, depth=INPUT_LAYER_SIZE,
    region=1,
    layout=NpuLayout.NHWC,
    data_type=NpuDataType.INT8,
    fm_elem_size=1,
    fm_addr=IN_SPK_ADDR,
    max_fm_value = IN_SPK_MAX_VAL,
    min_fm_value = IN_SPK_MIN_VAL,
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
        max_fm_value = IN_CURR_MAX_VAL,
        min_fm_value = IN_CURR_MIN_VAL,
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
    weights_volume_ohwi=np.ones((ofm.shape.depth, kernel.height, kernel.width, ifm.shape.depth), dtype=np.int8)
    #print("weights_volume_ohwi", weights_volume_ohwi.shape)
    #print(weights_volume_ohwi)
    #weights_volume_ohwi = np.tile(np.arange(1, 5), (32, 4)).reshape(ofm.shape.depth, kernel.height, kernel.width, ifm.shape.depth).astype(np.int8)  # Repeat [1, 2, 3, 4] across 32 rows
    #weights_volume_ohwi = np.ones((OFM_DEPTH, KERNEL_HEIGHT, KERNEL_WIDTH, IFM_DEPTH)).astype(np.uint8)
    weight_ifm_bitdepth = 8 #int8_t


    #Biases
    bias_list = []
    scale_list = []
    shift_list = []
    for i in range(ofm.shape.depth):
        #bias_list.append(np.int64(i%4))
        bias_list.append(np.int64(1))
        scale_list.append(1)
        shift_list.append(0)




    weight_byte_arr, bias_byte_arr = gen_weights_and_biases(accelerator=ACCELERATOR,
                           weights_volume_ohwi=weights_volume_ohwi,
                            dilation_xy=(1,1),
                            ifm_bitdepth=weight_ifm_bitdepth,
                            ofm_block_depth=block_config[2],
                            op_type=NpuOperationType.Conv2D,
                            block_traversal=block_traversal,

                            bias_list=bias_list,
                            scale_list=scale_list,
                            shift_list=shift_list
    )

    weight_n_bias_len = len(bias_byte_arr) + len(weight_byte_arr)
    if DEBUG_MODE:
        print("weight_n_bias_len", weight_n_bias_len)

    

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
    my_op.accumulator_type  =   NpuAccumulatorType.Default
    my_op.block_traversal   =   block_traversal


    check_block_config_legal(block_config, my_op, ACCELERATOR)





    return my_op, dma_op, weight_byte_arr, bias_byte_arr, 





def def_mul_decay_Vmem():
    
    IFM2_IS_FIRST_OPERAND = False

    #V_MEM_ADDR = 0x00
    #DECAY_ADDR = 0x00
    #DECAYED_MEM_ADDR = 0x00


    ifm = create_feature_map(
        height=1, width=1, depth=OUTPUT_LAYER_SIZE,
        region=1,
        layout=NpuLayout.NHWC,
        data_type=NpuDataType.INT8,
        fm_elem_size=1,
        fm_addr=V_MEM_ADDR,
        max_fm_value=V_MEM_MAX_VAL,
        min_fm_value=V_MEM_MIN_VAL,
        name="v_mem"
    )

    ifm2 = create_feature_map(
        height=1, width=1, depth=OUTPUT_LAYER_SIZE,
        region=1,
        layout=NpuLayout.NHWC,
        data_type=NpuDataType.INT8,
        fm_elem_size=1,
        fm_addr=DECAY_ADDR,
        max_fm_value=DECAY_MAX_VAL,
        min_fm_value=DECAY_MIN_VAL,
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
        max_fm_value=DECAYED_MEM_MAX_VAL,
        min_fm_value=DECAYED_MEM_MIN_VAL,
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
        max_fm_value=DECAYED_MEM_MAX_VAL,
        min_fm_value=DECAYED_MEM_MIN_VAL,
        name="decayed_mem"
    )

    ifm2 = create_feature_map(
        height=1, width=1, depth=OUTPUT_LAYER_SIZE,
        region=1,
        layout=NpuLayout.NHWC,
        data_type=NpuDataType.INT8,
        fm_elem_size=1,
        fm_addr=IN_CURR_ADDR,
        max_fm_value=IN_CURR_MAX_VAL,
        min_fm_value=IN_CURR_MIN_VAL,
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
        max_fm_value=V_MEM_MAX_VAL,
        min_fm_value=V_MEM_MIN_VAL,
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





if __name__ == '__main__':
    fully_connected_op, dma_op, weight_byte_arr, bias_byte_arr = def_fullyconnected()
    mul_decay_op = def_mul_decay_Vmem()
    add_decayed_mem_in_curr = def_add_decayed_mem_in_curr()

    npu_op_list = [dma_op, fully_connected_op, mul_decay_op, add_decayed_mem_in_curr]


    cms_bytearr, register_cms = gen_cms(npu_op_list, ACCELERATOR, DEBUG_MODE)

    header_out_filepath = "../../snn_on_alif_e7/simple_code_test/include/" + CMS_NAME + ".h"
    imp_out_filepath = "../../snn_on_alif_e7/simple_code_test/nn_ops/" + CMS_NAME + ".c"
    write_cms_to_files(header_out_filepath, imp_out_filepath, npu_op_list, cms_bytearr, register_cms, CMS_NAME, ADDR_DICT, weight_byte_arr=weight_byte_arr, bias_byte_arr=bias_byte_arr)
    