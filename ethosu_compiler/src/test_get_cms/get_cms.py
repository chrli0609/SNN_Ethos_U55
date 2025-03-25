from ethosu.vela.api import *


#my_op = NpuOperation(NpuOperationType.Conv2D)
my_op = NpuConv2DOperation()

#IFM
my_ifm = NpuFeatureMap()
my_ifm.shape = NpuShape3D(height=1, width=1, depth=1)



#NpuBlockOperation
my_op.ifm = []
my_op.ifm2 = 
# my_ope = 
my_op.ifm2_scalar = 
my_op.ofm = 
my_op.kernel = 
# my_opights = 
# my_opst = 
my_op.weights = 
# my_opases = 
# my_opst = 
my_op.biases = 
my_op.padding = 
# my_optional = 
my_op.activation = 
# my_ope = 
# my_ope = 
# my_op = 
# my_ope = 
my_op.block_config = 
my_op.rounding_mode = 
# my_opt = 
my_op.fused_quantize = 
# my_opM = 
my_op.ifm_upscale = 
my_op.accumulator_type = 
accelerator = NpuAccelerator.Ethos_U55_256

print(npu_generate_register_command_stream([my_op], accelerator))

