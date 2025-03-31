from ethosu.vela.api import *
from cms_interpreter import register_cms_2_assembly
from extra_func import float_to_int_safe




#Accelerator
accelerator = NpuAccelerator.Ethos_U55_256


#IFM1
IFM1_HEIGHT = 4
IFM1_WIDTH = 4
IFM1_DEPTH = 16

#IFM2
IFM2_HEIGHT = 4
IFM2_WIDTH = 4
IFM2_DEPTH = 16

#OFM
OFM_HEIGHT = 4
OFM_WIDTH = 4
OFM_DEPTH = 16


#Data type
DATA_TYPE = NpuDataType.INT8




######### Memory management ########
#Tensor Arena Size:
input1_tensor_size = IFM1_HEIGHT * IFM1_WIDTH * IFM1_DEPTH
input2_tensor_size = IFM2_HEIGHT * IFM2_WIDTH * IFM2_DEPTH
output_tensor_size = OFM_HEIGHT * OFM_WIDTH * OFM_DEPTH



required_size = input1_tensor_size + input2_tensor_size + output_tensor_size


if required_size % 2 != 0:
    print("Tensor Arena Size must be 16 bit aligned, instead found required_size =", required_size)

tensor_arena_size = required_size

input1_addr = tensor_arena_size - input1_tensor_size
input2_addr = input1_addr - input2_tensor_size
output_addr = input2_addr - output_tensor_size




print("Allocated size for Arena Tensor:", tensor_arena_size)
print("input1_addr", input1_addr)
print("input2_addr", input2_addr)
print("output_addr", output_addr)



IFM1_ADDR = input1_addr
IFM2_ADDR = input2_addr
OFM_ADDR = output_addr
####################################


IFM2_IS_FIRST_OPERAND = False





#Activation
my_activation = NpuActivation(NpuActivationOp.NONE_OR_RELU)
my_activation.min = 0
my_activation.max = 256










#IFM
my_ifm = NpuFeatureMap()
my_ifm.data_type = DATA_TYPE
my_ifm.region = 1
my_ifm.shape = NpuShape3D(height=IFM1_HEIGHT, width=IFM1_WIDTH, depth=IFM1_DEPTH)
# Only 1 tile used
my_ifm.tiles = NpuTileBox(height_0=my_ifm.shape.height, height_1=0, width_0=my_ifm.shape.width, addresses=[IFM1_ADDR, 0, 0, 0])
my_ifm.quantization = NpuQuantization(scale_f32= 1, zero_point= 0)
my_ifm.layout = NpuLayout.NHWC
my_ifm_elem_size = my_ifm.data_type.value[0]/8
my_ifm.strides = NpuShape3D(
    float_to_int_safe(my_ifm_elem_size*IFM1_DEPTH*IFM1_WIDTH),
    float_to_int_safe(my_ifm_elem_size*IFM1_DEPTH),
    float_to_int_safe(my_ifm_elem_size)
)




#IFM2
my_ifm2 = NpuFeatureMap()
my_ifm2.data_type = DATA_TYPE
my_ifm2.region = 1
my_ifm2.shape = NpuShape3D(height=IFM2_HEIGHT, width=IFM2_WIDTH, depth=IFM2_DEPTH)
# Only 1 tile used
my_ifm2.tiles = NpuTileBox(height_0=my_ifm2.shape.height, height_1=0, width_0=my_ifm2.shape.width, addresses=[IFM2_ADDR, 0, 0, 0])
my_ifm2.quantization = NpuQuantization(scale_f32= 1, zero_point= 0)
my_ifm2.layout = NpuLayout.NHWC
my_ifm2_elem_size = my_ifm2.data_type.value[0]/8
my_ifm2.strides = NpuShape3D(
    float_to_int_safe(my_ifm2_elem_size*IFM2_DEPTH*IFM2_WIDTH),
    float_to_int_safe(my_ifm2_elem_size*IFM2_DEPTH),
    float_to_int_safe(my_ifm2_elem_size)
)



#OFM
my_ofm = NpuFeatureMap()
my_ofm.data_type = NpuDataType.INT8
my_ofm.region = 1
my_ofm.shape = NpuShape3D(height=OFM_HEIGHT, width=OFM_WIDTH, depth=OFM_DEPTH)
#my_ofm.shape = NpuShape3D(height=4, width=4, depth=16)
# Only 1 tile used
my_ofm.tiles = NpuTileBox(height_0=my_ofm.shape.height, height_1=0, width_0=my_ofm.shape.width, addresses=[OFM_ADDR, 0, 0, 0])
my_ofm.quantization = NpuQuantization(scale_f32= 1, zero_point= 0)
my_ofm.layout = NpuLayout.NHWC
my_ofm_elem_size = my_ofm.data_type.value[0]/8
my_ofm.strides = NpuShape3D(
    float_to_int_safe(my_ofm_elem_size*OFM_DEPTH*OFM_WIDTH),
    float_to_int_safe(my_ofm_elem_size*OFM_DEPTH),
    float_to_int_safe(my_ofm_elem_size)
)



# add_op
add_op = NpuElementWiseOperation(NpuElementWiseOp.ADD)

#elementwise operation
add_op.reversed_operands = IFM2_IS_FIRST_OPERAND
add_op.rescale = None

#NpuBlockOperation
add_op.ifm = my_ifm
add_op.ifm2 = my_ifm2
add_op.ifm2_scalar = None   #set if ifm2 is a scalar
add_op.ofm = my_ofm
add_op.kernel = None
add_op.weights = []
add_op.biases = []
add_op.padding = None
add_op.activation = my_activation

# Block Config
print("Show available block_configs")
print(npu_find_block_configs(add_op, accelerator))

#Block Config
my_block_config = NpuShape3D(4, 4, 16)

add_op.block_config = my_block_config
add_op.rounding_mode = NpuRoundingMode.TFL
add_op.fused_quantize = False
add_op.ifm_upscale = NpuResamplingMode.NONE
add_op.accumulator_type = NpuAccumulatorType.Default







register_command_stream = npu_generate_register_command_stream([add_op], accelerator)

#print("\nregister command stream\n", register_command_stream)
register_cms_2_assembly(register_command_stream)



driver_payload_byte_array = npu_create_driver_payload(register_command_stream, accelerator)





formatted_cms = ", ".join(f"0x{b:02x}" for b in driver_payload_byte_array)
print("\nDriver payload:")
print(formatted_cms)






