import numpy as np

from ethosu.vela.api import *



#Accelerator
accelerator = NpuAccelerator.Ethos_U55_256


'''
#DMA Mem
DMA_SRC_REGION = 0
DMA_SRC_ADDR = 0xa0 

DMA_DST_REGION = 1
DMA_DST_ADDR = 0x800

#len(Bias + weights) = 168, to closest 16-bit alignment --> 176
DMA_LEN = 1008

#Input output
INPUT_ADDR = 0x400
OUTPUT_ADDR = 0x00

#Weights & Biases
WEIGHT_ADDR = 0x8a0
WEIGHT_REGION = 1
WEIGHT_LEN_BYTE = 848

BIAS_ADDR = 0x800
BIAS_REGION = 1
#40 bytes: closes 16 bit alignment --> 16*3 = 48
BIAS_LEN_BYTE = 160 #in hex: 0x30
'''

#Tensor Arena Size:
input_tensor_size = 8*8*16
output_tensor_size = 4*4*16
weight_tensor_size = 304 #16*2*2*16 #compressed
weight_len = 144
bias_len = 160

required_size = input_tensor_size + output_tensor_size + weight_tensor_size


if required_size % 2 != 0:
    print("Tensor Arena Size must be 16 bit aligned, instead found required_size =", required_size)

tensor_arena_size = required_size

input_addr = tensor_arena_size - input_tensor_size
output_addr = input_addr - output_tensor_size
weight_addr = output_addr - weight_tensor_size




#DMA Mem
DMA_SRC_REGION = 0
DMA_SRC_ADDR = weight_addr

#len(Bias + weights) = 168, to closest 16-bit alignment --> 176
#DMA_LEN = 768
#DMA_LEN = 0x300
DMA_LEN = weight_tensor_size #(192)

DMA_DST_REGION = 1
DMA_DST_ADDR = weight_addr


#Weights & Biases
BIAS_REGION = 1
BIAS_ADDR = weight_addr
#40 bytes: closest 16 bit alignment --> 16*3 = 48
#BIAS_LEN_BYTE = 640
BIAS_LEN_BYTE = bias_len #(160)

WEIGHT_ADDR = weight_addr + bias_len
WEIGHT_REGION = 1
WEIGHT_LEN_BYTE = weight_len #(32)





#Input output
INPUT_ADDR = input_addr
OUTPUT_ADDR = output_addr





#IFM
my_ifm = NpuFeatureMap()
my_ifm.data_type = NpuDataType.INT8
my_ifm.region = 1
my_ifm.shape = NpuShape3D(height=8, width=8, depth=16)
# Only 1 tile used
my_ifm.tiles = NpuTileBox(height_0=my_ifm.shape.height, height_1=0, width_0=my_ifm.shape.width, addresses=[INPUT_ADDR, 0, 0, 0])
my_ifm.quantization = NpuQuantization(scale_f32= 1, zero_point= 0)
my_ifm.layout = NpuLayout.NHWC
my_ifm.strides = NpuShape3D(128, 16, 128)




#OFM


my_ofm = NpuFeatureMap()
my_ofm.data_type = NpuDataType.INT8
my_ofm.region = 1
my_ofm.shape = NpuShape3D(height=4, width=4, depth=16)
# Only 1 tile used
my_ofm.tiles = NpuTileBox(height_0=my_ofm.shape.height, height_1=0, width_0=my_ofm.shape.width, addresses=[OUTPUT_ADDR, 0, 0, 0])
my_ofm.quantization = NpuQuantization(scale_f32=1, zero_point= 0)
my_ofm.layout = NpuLayout.NHWC
my_ofm.strides = NpuShape3D(64, 16, 64)


#Kernel
my_kernel = NpuKernel(
    w = 2,
    h = 2,
    stride_x = 2,
    stride_y = 2,
    dilation_x = 1,
    dilation_y = 1
)


#Padding
my_padding = NpuPadding(0, 0, 0, 0)

#Weights
'''
weights_volume=np.zeros((16, 2, 2, 16), dtype=np.uint8)
dilation_xy=(1,1)
ifm_bitdepth = 8
ofm_block_depth = 16
is_depthwise = False
block_traversal = NpuBlockTraversal.PART_KERNEL_FIRST

weight_byte_array = npu_encode_weights(
    accelerator=accelerator,
    weights_volume=weights_volume,
    dilation_xy=dilation_xy,
    ifm_bitdepth=ifm_bitdepth,
    ofm_block_depth=ofm_block_depth,
    is_depthwise=is_depthwise,
    block_traversal=block_traversal
)


#Biases
bias = np.int64(0)
scale=1077964381
shift=31

bias_byte_array = npu_encode_bias(
    bias=bias,
    scale=scale,
    shift=shift
)
'''


weights = NpuAddressRange(WEIGHT_REGION, WEIGHT_ADDR, WEIGHT_LEN_BYTE)



biases = NpuAddressRange(BIAS_REGION, BIAS_ADDR, BIAS_LEN_BYTE)






#Activation
my_activation = NpuActivation(NpuActivationOp.NONE_OR_RELU)
my_activation.min = 0
my_activation.max = 256


#Block Config
my_block_config = NpuShape3D(4, 4, 16)













#DMA


dma_src = NpuAddressRange(region=DMA_SRC_REGION, address=DMA_SRC_ADDR, length=DMA_LEN)
dma_dst = NpuAddressRange(region=DMA_DST_REGION, address=DMA_DST_ADDR, length=DMA_LEN)
dma_op = NpuDmaOperation(src=dma_src, dest=dma_dst)





# CONV2D
conv2d_op = NpuConv2DOperation()

#NpuBlockOperation
conv2d_op.ifm = my_ifm
conv2d_op.ifm2 = None
conv2d_op.ifm2_scalar = None
conv2d_op.ofm = my_ofm
conv2d_op.kernel = my_kernel
conv2d_op.weights = [weights]
conv2d_op.biases = [biases]
conv2d_op.padding = my_padding
conv2d_op.activation = my_activation
conv2d_op.block_config = my_block_config
conv2d_op.rounding_mode = NpuRoundingMode.TFL
conv2d_op.fused_quantize = False
conv2d_op.ifm_upscale = NpuResamplingMode.NONE
conv2d_op.accumulator_type = NpuAccumulatorType.Default
accelerator = accelerator





register_command_stream = npu_generate_register_command_stream([dma_op, conv2d_op], accelerator)


driver_payload_byte_array = npu_create_driver_payload(register_command_stream, accelerator)


formatted_cms = ", ".join(f"0x{b:02x}" for b in driver_payload_byte_array)
print(formatted_cms)



