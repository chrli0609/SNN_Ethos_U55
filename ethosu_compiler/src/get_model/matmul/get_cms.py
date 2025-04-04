import numpy as np

from ethosu.vela.api import *

import sys
import os
sys.path.insert(0, os.path.abspath('../'))
from cms_interpreter import register_cms_2_assembly

from extra_func import float_to_int_safe
from extra_func import print_includes
from extra_func import print_methods

from get_weights import *



DEBUG_MODE = True

#Accelerator
ACCELERATOR = NpuAccelerator.Ethos_U55_256

DATA_TYPE = NpuDataType.INT8

# Fully Connected Layer
OP_TYPE = NpuConv2DOperation()

# PARAMS



#IFM
IFM_HEIGHT = 1
IFM_WIDTH = 1
IFM_DEPTH = 16


#OFM
OFM_HEIGHT = 1
OFM_WIDTH = 1
OFM_DEPTH = 32



#Kernel
KERNEL_HEIGHT = 1
KERNEL_WIDTH = 1


# Set weights and biases

#Weights
#weights_volume_ohwi=np.ones((OFM_DEPTH, KERNEL_HEIGHT, KERNEL_WIDTH, IFM_DEPTH), dtype=np.uint8)
#print("weights_volume_ohwi", weights_volume_ohwi.shape)
#print(weights_volume_ohwi)
weights_volume_ohwi = np.tile(np.arange(1, 5), (32, 4)).reshape(OFM_DEPTH, KERNEL_HEIGHT, KERNEL_WIDTH, IFM_DEPTH).astype(np.uint8)  # Repeat [1, 2, 3, 4] across 32 rows
#weights_volume_ohwi = np.ones((OFM_DEPTH, KERNEL_HEIGHT, KERNEL_WIDTH, IFM_DEPTH)).astype(np.uint8)
WEIGHTS_DILATION_XY = (1,1)
weight_ifm_bitdepth = 8 #uint8_t


#Biases
bias_list = []
scale_list = []
shift_list = []
for i in range(OFM_DEPTH):
    #bias_list.append(np.int64(i%4))
    bias_list.append(np.int64(1))
    scale_list.append(1)
    shift_list.append(0)


if DEBUG_MODE:
    print("/*\nweights:")
    print(weights_volume_ohwi)
    print("biases:\n", bias_list)
    print("*/")







KERNEL = NpuKernel(
    w = KERNEL_WIDTH,
    h = KERNEL_HEIGHT,
    stride_x = 1,
    stride_y = 1,
    dilation_x = 1,
    dilation_y = 1
)

ACTIVATION_TYPE = NpuActivationOp.NONE_OR_RELU
ACTIVATION_MAX = 255
ACTIVATION_MIN = 0

PADDING = NpuPadding(0, 0, 0, 0)
BLOCK_CONFIG = NpuShape3D(2, 2, 16)
BLOCK_TRAVERSAL = NpuBlockTraversal.DEPTH_FIRST


#########





# Print includes
print_includes()


### Generate weights and biases

# How should block traversal look like does it:
# a) depend on BLOCK_TRAVERSAL OF OP
# b) depend on the operation type (i.e. CONV2D vs DepthwiseCONV)
'''
#a)
if BLOCK_TRAVERSAL == NpuBlockTraversal.DEPTH_FIRST:
    is_depthwise = True
elif BLOCK_TRAVERSAL == NpuBlockTraversal.PART_KERNEL_FIRST:
    is_depthwise = False
else:
    print("Error: Unrecognized BLOCK_TRAVERSAL METHOD")
    exit()
'''
#b)
#NpuOperationType.Conv2D
if OP_TYPE.op_type == NpuOperationType.Conv2D:
    is_depthwise = False
elif OP_TYPE.op_type == NpuOperationType.ConvDepthWise:
    is_depthwise = True
else:
    print("Error: Unrecognized Operation type")
    exit()



weight_len, bias_len = gen_weights_and_biases(
    accelerator=ACCELERATOR,
    weights_volume_ohwi=weights_volume_ohwi,
    dilation_xy=WEIGHTS_DILATION_XY,
    ifm_bitdepth=weight_ifm_bitdepth,
    ofm_block_depth=BLOCK_CONFIG[2],
    is_depthwise=is_depthwise,
    block_traversal=BLOCK_TRAVERSAL,

    bias_list=bias_list,
    scale_list=scale_list,
    shift_list=shift_list,

    is_debug_mode=DEBUG_MODE
)










##### Memory Management ######

#Tensor Arena Size:
input_tensor_size = IFM_HEIGHT * IFM_WIDTH * IFM_DEPTH
output_tensor_size = OFM_HEIGHT * OFM_WIDTH * OFM_DEPTH
weight_tensor_size = weight_len + bias_len


required_size = input_tensor_size + output_tensor_size + weight_tensor_size


if required_size % 2 != 0:
    print("Tensor Arena Size must be 16 bit aligned, instead found required_size =", required_size)

tensor_arena_size = required_size

input_addr = tensor_arena_size - input_tensor_size
output_addr = input_addr - output_tensor_size
weight_addr = output_addr - weight_tensor_size




#DMA Mem
DMA_SRC_REGION = 0
DMA_SRC_ADDR = 0x00

DMA_LEN = weight_tensor_size

DMA_DST_REGION = 1
DMA_DST_ADDR = weight_addr


#Weights & Biases
BIAS_REGION = 1
BIAS_ADDR = weight_addr
BIAS_LEN_BYTE = bias_len

WEIGHT_ADDR = weight_addr + bias_len
WEIGHT_REGION = 1
WEIGHT_LEN_BYTE = weight_len



#Input output
INPUT_ADDR = input_addr
OUTPUT_ADDR = output_addr


# Print define statements
print("#define MATMUL_TENSOR_ARENA_SIZE", tensor_arena_size)
print("#define MATMUL_INPUT_TENSOR_SIZE", input_tensor_size)
print("#define MATMUL_OUTPUT_TENSOR_SIZE", output_tensor_size, "\n\n\n\n")
print("/*")




#IFM
my_ifm = NpuFeatureMap()
my_ifm.data_type = DATA_TYPE
my_ifm.region = 1
my_ifm.shape = NpuShape3D(height=IFM_HEIGHT, width=IFM_WIDTH, depth=IFM_DEPTH)
# Only 1 tile used
my_ifm.tiles = NpuTileBox(height_0=my_ifm.shape.height, height_1=0, width_0=my_ifm.shape.width, addresses=[INPUT_ADDR, 0, 0, 0])
my_ifm.quantization = NpuQuantization(scale_f32= 1, zero_point= 0)
my_ifm.layout = NpuLayout.NHWC
my_ifm_elem_size = my_ifm.data_type.value[0]/8
my_ifm.strides = NpuShape3D(
    float_to_int_safe(my_ifm_elem_size*IFM_DEPTH*IFM_WIDTH),
    float_to_int_safe(my_ifm_elem_size*IFM_DEPTH),
    float_to_int_safe(my_ifm_elem_size)
)

#print("IFM stride x/y/c:", my_ifm.strides)



#OFM
my_ofm = NpuFeatureMap()
my_ofm.data_type = DATA_TYPE
my_ofm.region = 1
my_ofm.shape = NpuShape3D(height=OFM_HEIGHT, width=OFM_WIDTH, depth=OFM_DEPTH)
# Only 1 tile used
my_ofm.tiles = NpuTileBox(height_0=my_ofm.shape.height, height_1=0, width_0=my_ofm.shape.width, addresses=[OUTPUT_ADDR, 0, 0, 0])
my_ofm.quantization = NpuQuantization(scale_f32=1, zero_point= 0)
my_ofm.layout = NpuLayout.NHWC

my_ofm_elem_size = my_ofm.data_type.value[0]/8
my_ofm.strides = NpuShape3D(
    float_to_int_safe(my_ofm_elem_size*OFM_DEPTH*OFM_WIDTH),
    float_to_int_safe(my_ofm_elem_size*OFM_DEPTH),
    float_to_int_safe(my_ofm_elem_size)
)

#print("OFM stride x/y/c:", my_ofm.strides)









weights = NpuAddressRange(WEIGHT_REGION, WEIGHT_ADDR, WEIGHT_LEN_BYTE)
biases = NpuAddressRange(BIAS_REGION, BIAS_ADDR, BIAS_LEN_BYTE)






#Activation
my_activation = NpuActivation(ACTIVATION_TYPE)
my_activation.min = ACTIVATION_MIN
my_activation.max = ACTIVATION_MAX









#DMA
dma_src = NpuAddressRange(region=DMA_SRC_REGION, address=DMA_SRC_ADDR, length=DMA_LEN)
dma_dst = NpuAddressRange(region=DMA_DST_REGION, address=DMA_DST_ADDR, length=DMA_LEN)
dma_op = NpuDmaOperation(src=dma_src, dest=dma_dst)





# MATMUL
matmul_op = NpuConv2DOperation()

#NpuBlockOperation
matmul_op.ifm = my_ifm
matmul_op.ifm2 = None
matmul_op.ifm2_scalar = None
matmul_op.ofm = my_ofm
matmul_op.kernel = KERNEL
matmul_op.weights = [weights]
matmul_op.biases = [biases]
matmul_op.padding = PADDING
matmul_op.activation = my_activation
matmul_op.block_config = BLOCK_CONFIG
matmul_op.rounding_mode = NpuRoundingMode.TFL
matmul_op.fused_quantize = False
matmul_op.ifm_upscale = NpuResamplingMode.NONE
matmul_op.accumulator_type = NpuAccumulatorType.Default
matmul_op.block_traversal = BLOCK_TRAVERSAL




register_command_stream = npu_generate_register_command_stream([dma_op, matmul_op], ACCELERATOR)
if DEBUG_MODE:
    print("\nCommands:\n")
    register_cms_2_assembly(register_command_stream)


driver_payload_byte_array = npu_create_driver_payload(register_command_stream, ACCELERATOR)



# Print CMS
formatted_cms = ", ".join(f"0x{b:02x}" for b in driver_payload_byte_array)
print("*/")
print("\n\n\n\nstatic const uint8_t cms_matmul[] __attribute__((aligned(16))) = \n{")
print(formatted_cms)
print("};\n\n\n")



# Print Methods
print_methods("MatMul", "matmul")




# Check that block config is legal
available_block_configs = npu_find_block_configs(matmul_op, ACCELERATOR)

block_config_legal = False
for i in range(len(available_block_configs)):
    if BLOCK_CONFIG == available_block_configs[i]:
        block_config_legal = True

if not block_config_legal:
    print("ERROR: BLOCK_CONFIG is not legal, found\n\t", BLOCK_CONFIG, "\n\n\t But expected one of the following:\n\t", available_block_configs)
else:
    print("//Current Block Config is legal:", BLOCK_CONFIG)






