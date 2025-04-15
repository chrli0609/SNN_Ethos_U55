from enum import Enum, auto
from typing import List, NamedTuple, Optional, Tuple, Dict, Any, Union


class NpuAccumulatorType(Enum):
    """
    Accumulator dtype of NPU operation
    """

    Default = auto()
    Int32 = auto()
    Int40 = auto()



from ethosu.vela.api import *
from extra_func import float_to_int_safe

# Helper functions/builders for easier creation of NPU operations





def gen_weights_and_biases(
        accelerator,
        weights_volume_ohwi,
        dilation_xy,
        ifm_bitdepth,
        ofm_block_depth,
        op_type,
        block_traversal,

        ofm_size,

        #ifm_scale,
        #ifm_zero_point,
        #ofm_scale,
        #ofm_zero_point,
        weight_scale,
        weight_zero_point,

        is_debug_mode=False,
):
    #NpuOperationType.Conv2D
    if op_type == NpuOperationType.Conv2D:
        is_depthwise = False
    elif op_type == NpuOperationType.ConvDepthWise:
        is_depthwise = True
    else:
        print("Error: Unrecognized Operation type")
        exit()


    num_biases = ofm_size

    #num_biases = len(bias_list)
    #ofm_depth = weights_volume_ohwi.shape[0]


    # Make checks
    #if num_biases != ofm_depth:
    #    print("Error: Incorrect Dim(Bias), expected len(bias_tensor) == len(weights_volume_ohwi.shape[0]), instead got:\n\tlen(bias_tenosr):", num_biases, "\n\tweights_tensor_ohwi.shape[0]:", ofm_depth)
    #    exit()





    # Quantize the values
    weights_volume_quantized = quantize_array_float_to_int8(weights_volume_ohwi, weight_scale, weight_zero_point)


    weight_bytearr = npu_encode_weights(
            accelerator=accelerator,
            weights_volume=weights_volume_quantized,
            dilation_xy=dilation_xy,
            ifm_bitdepth=ifm_bitdepth,
            ofm_block_depth=ofm_block_depth,
            is_depthwise=is_depthwise,
            block_traversal=block_traversal,
        )




    #print("bias_quant params: scale", ofm_scale, " zero_point", ofm_zero_point)
    #bias_quantized = quantize_array_float_to_int8(np.array(bias_list, dtype='float32'), ofm_scale, ofm_zero_point).astype(np.int64)
    

    bias_bytearr_list = []
    for i in range(num_biases):

        # convert floating point scale value to fixed point
        scale, shift = float_to_fixed(weight_scale)
        bias_bytearr_list.append(npu_encode_bias(
            bias=np.int64(0),
            scale=scale,
            shift=shift+1,
        ))


    bias_byte_arr = bytearray()
    for i in range(num_biases):
        bias_byte_arr.extend(bias_bytearr_list[i])
        


    tot_tensor_byte_len = len(weight_bytearr) + len(bias_byte_arr)
    if is_debug_mode:
        print("//len(biases):", len(bias_byte_arr), " as hex: (" + hex(len(bias_byte_arr)) + ")\n")
        print("//len(weights):", len(weight_bytearr), " as hex: (" + hex(len(weight_bytearr)) + ")\n")
        print("//tot len:", tot_tensor_byte_len, " as hex: (" + hex(tot_tensor_byte_len) + ")\n\n")
        #print("quantized biases:", bias_quantized)
        print("quantized weights:\n\tweight_scale:", 
              weight_scale, "scale_fixed_point:", scale, "shift:", shift, "converted back to float", fixed_to_float(scale, shift), 
              "\n\tzero_point:", weight_zero_point, "\n", weights_volume_quantized
              )



    return weight_bytearr, bias_byte_arr


def symmetric_zero_point_quant(max_val, min_val):
    zero_point = 0

    scale = (max_val - min_val) / 255

    return scale, zero_point



def zero_point_quant(max_val, min_val):
    #print("max_val", max_val, "min_val", min_val)
    scale = (max_val - min_val) / 255
    zero_point = -round(min_val / scale) - 128


    #print("scale", scale, "zero_point", zero_point)

    return scale, zero_point





def symmetric_zero_point_quant_int16(max_val, min_val):
    zero_point = 0

    scale = (max_val - min_val) / 255

    return scale, zero_point



def zero_point_quant_int16(max_val, min_val):
    #print("max_val", max_val, "min_val", min_val)
    scale = (max_val - min_val) / (32767 - (-32768))
    zero_point = -round(min_val / scale)
    print("This part is not working")
    exit()


    #print("scale", scale, "zero_point", zero_point)

    return scale, zero_point









import numpy as np
def quantize_array_float_to_int8(input_array, scale, zero_point):
    """
    Quantize an array of floats to int8 values.

    Parameters:
    - input_array: NumPy array of floats
    - scale: quantization scale (float)
    - zero_point: quantization zero point (int)

    Returns:
    - output_array: NumPy array of int8 values
    """
    input_array = np.asarray(input_array, dtype=np.float32)
    quantized = np.round(input_array / scale).astype(np.int32) + zero_point
    quantized = np.clip(quantized, -128, 127)
    return quantized.astype(np.int8)


def quantize_array_float_to_int64(input_array, scale, zero_point):
    """
    Quantize an array of floats to int64 values.

    Parameters:
    - input_array: NumPy array of floats
    - scale: quantization scale (float)
    - zero_point: quantization zero point (int)

    Returns:
    - output_array: NumPy array of int64 values
    """
    input_array = np.asarray(input_array, dtype=np.float32)
    quantized = np.round(input_array / scale).astype(np.int64) + zero_point
    return quantized



def float_to_fixed(value):
    """
    Convert a floating-point number to fixed-point representation.
    
    Returns:
        tuple: (int_part, shift) where:
            - int_part is a 32-bit signed integer
            - shift is a 6-bit unsigned integer (0-63)
    
    The original value can be recovered with: int_part * 2^(-shift)
    """
    import math
    
    # Handle special cases
    if value == 0:
        return 0, 0
    
    # Determine if value is negative
    sign = -1 if value < 0 else 1
    value = abs(value)
    
    # Find appropriate shift to maximize precision
    # We want to use as much of the 32-bit range as possible
    max_shift = 63  # 6-bit shift value can represent 0-63
    max_int = 0x7FFFFFFF  # Maximum 31-bit positive value (saving 1 bit for sign)
    
    # Find the exponent in the binary representation
    exponent = math.floor(math.log2(value)) if value != 0 else 0
    
    # Calculate initial shift and int_part
    shift = 0
    int_part = 0
    
    if exponent >= 31:
        # Value is too large, use maximum precision
        shift = 0
        int_part = sign * max_int
    elif exponent < -max_shift:
        # Value is too small, use minimum non-zero representation
        shift = max_shift
        int_part = sign * 1
    else:
        # Normal case: maximize precision
        shift = max(0, -exponent + 30)
        shift = min(shift, max_shift)  # Ensure shift is within 6-bit range
        
        # Calculate the integer part with the calculated shift
        scaled_value = value * (2 ** shift)
        
        # Round to nearest integer
        int_part = int(scaled_value + 0.5) if sign > 0 else int(scaled_value - 0.5)
        
        # Ensure int_part fits in 32-bit signed int
        if abs(int_part) > max_int:
            if shift > 0:
                # Try to adjust shift to make int_part fit
                shift -= 1
                scaled_value = value * (2 ** shift)
                int_part = int(scaled_value + 0.5) if sign > 0 else int(scaled_value - 0.5)
                
                # If still too large, cap at max value
                if abs(int_part) > max_int:
                    int_part = sign * max_int
            else:
                # Can't reduce shift further, cap at max value
                int_part = sign * max_int
    
    # Apply sign to int_part
    int_part = sign * abs(int_part)
    
    # Ensure int_part fits in 32 bits (-2^31 to 2^31-1)
    int_part = max(min(int_part, 0x7FFFFFFF), -0x80000000)
    
    # Ensure shift fits in 6 bits (0-63)
    shift = max(min(shift, 63), 0)
    
    return int_part, shift


def fixed_to_float(int_part, shift):
    """
    Convert a fixed-point representation back to a floating-point number.
    
    Args:
        int_part: 32-bit signed integer
        shift: 6-bit unsigned integer (0-63)
    
    Returns:
        float: The original floating-point value
    """
    return int_part * (2 ** (-shift))


def create_feature_map(height: int, width: int, depth: int, 
                      region, 
                      layout,  # Pass NpuLayout.NHWC or similar
                      data_type,  # Pass NpuDataType.INT8 or similar
                      #scale: Optional[float] = None,
                      #zero_point: Optional[int] = None,
                      fm_elem_size: int,
                      fm_addr,
                      #max_fm_value,
                      #min_fm_value,
                      scale,
                      zero_point,
                      name: Optional[str] = None) -> NpuFeatureMap:
    """
    Create an NpuFeatureMap with the given parameters.
    """
    fm = NpuFeatureMap()
    fm.shape = NpuShape3D(height=height, width=width, depth=depth)
    fm.region = region
    
    if layout is not None:
        fm.layout = layout
    
    if data_type is not None:
        fm.data_type = data_type
    
    #scale, zero_point = zero_point_quant(max_fm_value, min_fm_value)
    if scale is not None and zero_point is not None:
        fm.quantization = NpuQuantization(scale_f32=scale, zero_point=zero_point)
    

    # Stride is purely dependent on FM dimensions
    stride_y = fm_elem_size*depth*width
    stride_x = fm_elem_size*depth
    stride_c = fm_elem_size
    if stride_y is not None and stride_x is not None and stride_c is not None:
        fm.strides = NpuShape3D(height=stride_y, width=stride_x, depth=stride_c)
    
    # Default tile setup for single tile (most common case)
    fm.tiles = NpuTileBox(
        height_0=height, 
        height_1=0, 
        width_0=width, 
        addresses=[fm_addr, 0, 0, 0]
    )
    
    if name is not None:
        fm.name = name
    
    return fm



def create_activation(activation_op, min_val, max_val, lookup_table_index=None):
    activation = NpuActivation(activation_op)
    activation.min = min_val
    activation.max = max_val
    if lookup_table_index:
        activation.lookup_table_index = lookup_table_index

    return activation



#def create_address_range(region: int, address: int, length: int) -> NpuAddressRange:
#    """Create an NpuAddressRange with the given parameters."""
#    return NpuAddressRange(region=region, address=address, length=length)

#def create_kernel(width: int, height: int, 
#                 stride_x: int = 1, stride_y: int = 1,
#                 dilation_x: int = 1, dilation_y: int = 1) -> NpuKernel:
#    """Create an NpuKernel with the given parameters."""
#    return NpuKernel(
#        w=width, 
#        h=height, 
#        stride_x=stride_x, 
#        stride_y=stride_y, 
#        dilation_x=dilation_x, 
#        dilation_y=dilation_y
#    )

#def create_block_config(height: int, width: int, depth: int) -> NpuShape3D:
#    """Create a block config with the given parameters."""
#    return NpuShape3D(height=height, width=width, depth=depth)

def create_padding(top: int, left: int, bottom: int, right: int) -> NpuPadding:
    """Create a padding configuration with the given parameters."""
    return NpuPadding(top=top, left=left, bottom=bottom, right=right)


class NpuOperationBuilder:
    """Base builder class for NPU operations."""
    
    def __init__(self, operation_type):
        self.operation = self._create_operation(operation_type)
    
    def _create_operation(self, operation_type):
        """Create an operation of the specific type."""
        if operation_type == NpuOperationType.Conv2D:
            return NpuConv2DOperation()
        elif operation_type == NpuOperationType.ElementWise:
            # This requires sub_op_type, which will be set later
            return None
        elif operation_type == NpuOperationType.Pooling:
            # This requires pooling_op_type, which will be set later
            return None
        elif operation_type == NpuOperationType.ConvDepthWise:
            return NpuConvDepthWiseOperation()
        else:
            return None
    
    def with_name(self, name: str):
        """Set the operation name."""
        self.operation.name = name
        return self
    
    def with_ifm(self, ifm: NpuFeatureMap):
        """Set the input feature map."""
        self.operation.ifm = ifm
        return self
    
    def with_ofm(self, ofm: NpuFeatureMap):
        """Set the output feature map."""
        self.operation.ofm = ofm
        return self
    
    def with_kernel(self, kernel: NpuKernel):
        """Set the kernel."""
        self.operation.kernel = kernel
        return self
    
    def with_padding(self, padding: NpuPadding):
        """Set the padding."""
        self.operation.padding = padding
        return self
    
    def with_block_config(self, block_config: NpuShape3D):
        """Set the block configuration."""
        self.operation.block_config = block_config
        return self
    
    def with_weights(self, weights: List[NpuAddressRange]):
        """Set the weights."""
        self.operation.weights = weights
        return self
    
    def with_biases(self, biases: List[NpuAddressRange]):
        """Set the biases."""
        self.operation.biases = biases
        return self
    
    def with_activation(self, activation: NpuActivation):
        """Set the activation function."""
        self.operation.activation = activation
        return self
    
    def with_rounding_mode(self, rounding_mode: NpuRoundingMode):
        """Set the rounding mode."""
        self.operation.rounding_mode = rounding_mode
        return self
    
    def with_fused_quantize(self, fused_quantize: bool):
        """Set whether to fuse with quantize operation."""
        self.operation.fused_quantize = fused_quantize
        return self
    
    def with_ifm_upscale(self, ifm_upscale: NpuResamplingMode):
        """Set the IFM upscaling mode."""
        self.operation.ifm_upscale = ifm_upscale
        return self
    
    def with_accumulator_type(self, accumulator_type: NpuAccumulatorType):
        """Set the accumulator type."""
        self.operation.accumulator_type = accumulator_type
        return self
    
    def build(self):
        """Build and return the operation."""
        return self.operation


class NpuConv2DBuilder(NpuOperationBuilder):
    """Builder for Conv2D operations."""
    
    def __init__(self):
        super().__init__(NpuOperationType.Conv2D)
    
    def with_block_traversal(self, block_traversal: NpuBlockTraversal):
        """Set the block traversal mode."""
        self.operation.block_traversal = block_traversal
        return self


class NpuElementWiseBuilder:
    """Builder for ElementWise operations."""
    
    def __init__(self, elementwise_op_type: NpuElementWiseOp):
        self.operation = NpuElementWiseOperation(elementwise_op_type)
    
    def with_name(self, name: str):
        """Set the operation name."""
        self.operation.name = name
        return self
    
    def with_ifm(self, ifm: NpuFeatureMap):
        """Set the first input feature map."""
        self.operation.ifm = ifm
        return self
    
    def with_ifm2(self, ifm2: NpuFeatureMap):
        """Set the second input feature map."""
        self.operation.ifm2 = ifm2
        return self
    
    def with_ifm2_scalar(self, scalar: float):
        """Set the scalar value for the second input."""
        self.operation.ifm2_scalar = scalar
        return self
    
    def with_ofm(self, ofm: NpuFeatureMap):
        """Set the output feature map."""
        self.operation.ofm = ofm
        return self
    
    def with_reversed_operands(self, reversed_operands: bool):
        """Set whether operands should be reversed."""
        self.operation.reversed_operands = reversed_operands
        return self
    
    def with_rescale(self, rescale: Optional[Tuple]):
        """Set the rescale parameters."""
        self.operation.rescale = rescale
        return self
    
    def with_block_config(self, block_config: NpuShape3D):
        """Set the block configuration."""
        self.operation.block_config = block_config
        return self
    
    def with_rounding_mode(self, rounding_mode: NpuRoundingMode):
        """Set the rounding mode."""
        self.operation.rounding_mode = rounding_mode
        return self
    
    def with_ifm_upscale(self, ifm_upscale: NpuResamplingMode):
        """Set the IFM upscaling mode."""
        self.operation.ifm_upscale = ifm_upscale
        return self
    
    def build(self):
        """Build and return the operation."""
        return self.operation


class NpuPoolingBuilder:
    """Builder for Pooling operations."""
    
    def __init__(self, pooling_op_type: NpuPoolingOp):
        self.operation = NpuPoolingOperation(pooling_op_type)
    
    def with_name(self, name: str):
        """Set the operation name."""
        self.operation.name = name
        return self
    
    def with_ifm(self, ifm: NpuFeatureMap):
        """Set the input feature map."""
        self.operation.ifm = ifm
        return self
    
    def with_ofm(self, ofm: NpuFeatureMap):
        """Set the output feature map."""
        self.operation.ofm = ofm
        return self
    
    def with_kernel(self, kernel: NpuKernel):
        """Set the kernel."""
        self.operation.kernel = kernel
        return self
    
    def with_padding(self, padding: NpuPadding):
        """Set the padding."""
        self.operation.padding = padding
        return self
    
    def with_rescale(self, rescale: Optional[float]):
        """Set the rescale factor."""
        self.operation.rescale = rescale
        return self
    
    def with_block_config(self, block_config: NpuShape3D):
        """Set the block configuration."""
        self.operation.block_config = block_config
        return self
    
    def with_rounding_mode(self, rounding_mode: NpuRoundingMode):
        """Set the rounding mode."""
        self.operation.rounding_mode = rounding_mode
        return self
    
    def with_ifm_upscale(self, ifm_upscale: NpuResamplingMode):
        """Set the IFM upscaling mode."""
        self.operation.ifm_upscale = ifm_upscale
        return self
    
    def build(self):
        """Build and return the operation."""
        return self.operation


# Examples of usage

def create_example_conv2d():
    """Create an example Conv2D operation."""
    # Create the input feature map
    ifm = create_feature_map(
        height=1, width=1, depth=784,
        region=1,
        layout=NpuLayout.NHWC,
        data_type=NpuDataType.INT8,
        scale=0.007842868566513062,
        zero_point=0,
        stride_y=784, stride_x=784, stride_c=1,
        name="serving_default_x:0_npu"
    )
    
    # Set tile address manually if needed
    ifm.tiles = NpuTileBox(height_0=1, height_1=1, width_0=1, addresses=[0, 0, 0, 0])
    
    # Create the output feature map
    ofm = create_feature_map(
        height=1, width=1, depth=1008,
        region=1,
        layout=NpuLayout.NHCWB16,
        data_type=NpuDataType.INT8,
        scale=0.30126750469207764,
        zero_point=-12,
        stride_y=1008, stride_x=16, stride_c=16,
        name="MatMul2"
    )
    
    # Set tile address manually if needed
    ofm.tiles = NpuTileBox(height_0=1, height_1=1, width_0=1, addresses=[0xbd0, 0, 0, 0])
    
    # Create the kernel
    kernel = create_kernel(width=1, height=1, stride_x=1, stride_y=1, dilation_x=1, dilation_y=1)
    
    # Create the padding
    padding = create_padding(top=0, left=0, bottom=0, right=0)
    
    # Create the weights address range
    weights = [create_address_range(region=0, address=0x52c0, length=100976)]
    
    # Create the block config
    block_config = create_block_config(height=2, width=2, depth=128)
    
    # Build the operation
    operation = (NpuConv2DBuilder()
                .with_name("MatMul2")
                .with_ifm(ifm)
                .with_ofm(ofm)
                .with_kernel(kernel)
                .with_padding(padding)
                .with_weights(weights)
                .with_block_config(block_config)
                .with_block_traversal(NpuBlockTraversal.DEPTH_FIRST)
                .with_rounding_mode(NpuRoundingMode.TFL)
                .with_ifm_upscale(NpuResamplingMode.NONE)
                .build())
    
    return operation

def create_example_elementwise():
    """Create an example ElementWise operation."""
    # Create the first input feature map
    ifm = create_feature_map(
        height=1, width=1, depth=1008,
        region=1,
        layout=NpuLayout.NHWC,
        data_type=NpuDataType.INT8,
        scale=0.00392152089625597,
        zero_point=-128,
        stride_y=1008, stride_x=1008, stride_c=1,
        name="serving_default_mem:0_npu"
    )
    
    # Set tile address manually if needed
    ifm.tiles = NpuTileBox(height_0=1, height_1=1, width_0=1, addresses=[0x3f0, 0, 0, 0])
    
    # Create the second input feature map
    ifm2 = create_feature_map(
        height=1, width=1, depth=1008,
        region=1,
        layout=NpuLayout.NHWC,
        data_type=NpuDataType.INT8,
        scale=0.003921563737094402,
        zero_point=-128,
        stride_y=1008, stride_x=1008, stride_c=1,
        name="serving_default_decay:0_npu"
    )
    
    # Set tile address manually if needed
    ifm2.tiles = NpuTileBox(height_0=1, height_1=1, width_0=1, addresses=[0x7e0, 0, 0, 0])
    
    # Create the output feature map
    ofm = create_feature_map(
        height=1, width=1, depth=1008,
        region=1,
        layout=NpuLayout.NHCWB16,
        data_type=NpuDataType.INT8,
        scale=0.30488383769989014,
        zero_point=-13,
        stride_y=1008, stride_x=16, stride_c=16,
        name="Mul"
    )
    
    # Set tile address manually if needed
    ofm.tiles = NpuTileBox(height_0=1, height_1=1, width_0=1, addresses=[0, 0, 0, 0])
    
    # Create the block config
    block_config = create_block_config(height=2, width=2, depth=128)
    
    # Build the operation
    operation = (NpuElementWiseBuilder(NpuElementWiseOp.MUL)
                .with_name("Mul")
                .with_ifm(ifm)
                .with_ifm2(ifm2)
                .with_ofm(ofm)
                .with_block_config(block_config)
                .with_rounding_mode(NpuRoundingMode.TFL)
                .with_ifm_upscale(NpuResamplingMode.NONE)
                .build())
    
    return operation


# Alternative approach: Dictionary-based creation

def create_npu_operation_from_dict(config: Dict[str, Any]) -> NpuOperation:
    """
    Create an NPU operation from a dictionary configuration.
    This allows for a more declarative style of creating operations.
    """
    op_type = config.get('op_type')
    
    if op_type == 'Conv2D':
        operation = NpuConv2DOperation()
        
        # Set block traversal if provided
        if 'block_traversal' in config:
            operation.block_traversal = config['block_traversal']
            
    elif op_type == 'ElementWise':
        sub_op_type = config.get('sub_op_type')
        if not sub_op_type:
            raise ValueError("ElementWise operation requires sub_op_type")
        
        operation = NpuElementWiseOperation(sub_op_type)
        
        # Set ElementWise specific properties
        if 'reversed_operands' in config:
            operation.reversed_operands = config['reversed_operands']
        
        if 'rescale' in config:
            operation.rescale = config['rescale']
            
        # Set ifm2 if provided
        if 'ifm2' in config:
            operation.ifm2 = config['ifm2']
        
        if 'ifm2_scalar' in config:
            operation.ifm2_scalar = config['ifm2_scalar']
            
    elif op_type == 'Pooling':
        sub_op_type = config.get('sub_op_type')
        if not sub_op_type:
            raise ValueError("Pooling operation requires sub_op_type")
        
        operation = NpuPoolingOperation(sub_op_type)
        
        # Set Pooling specific properties
        if 'rescale' in config:
            operation.rescale = config['rescale']
            
    else:
        raise ValueError(f"Unsupported operation type: {op_type}")
    
    # Set common properties
    if 'name' in config:
        operation.name = config['name']
    
    if 'ifm' in config:
        operation.ifm = config['ifm']
    
    if 'ofm' in config:
        operation.ofm = config['ofm']
    
    if 'kernel' in config:
        operation.kernel = config['kernel']
    
    if 'padding' in config:
        operation.padding = config['padding']
    
    if 'weights' in config:
        operation.weights = config['weights']
    
    if 'biases' in config:
        operation.biases = config['biases']
    
    if 'activation' in config:
        operation.activation = config['activation']
    
    if 'block_config' in config:
        operation.block_config = config['block_config']
    
    if 'rounding_mode' in config:
        operation.rounding_mode = config['rounding_mode']
    
    if 'fused_quantize' in config:
        operation.fused_quantize = config['fused_quantize']
    
    if 'ifm_upscale' in config:
        operation.ifm_upscale = config['ifm_upscale']
    
    if 'accumulator_type' in config:
        operation.accumulator_type = config['accumulator_type']
    
    return operation


# Example of using the dictionary-based approach
def create_example_dict_conv2d():
    """Create an example Conv2D operation using the dictionary approach."""
    # Create feature maps
    ifm = create_feature_map(
        height=1, width=1, depth=784,
        region=1,
        layout=NpuLayout.NHWC,
        data_type=NpuDataType.INT8,
        scale=0.007842868566513062,
        zero_point=0,
        name="serving_default_x:0_npu"
    )
    
    ofm = create_feature_map(
        height=1, width=1, depth=1008,
        region=1,
        layout=NpuLayout.NHCWB16,
        data_type=NpuDataType.INT8,
        scale=0.30126750469207764,
        zero_point=-12,
        name="MatMul2"
    )
    
    # Create the configuration dictionary
    config = {
        'op_type': 'Conv2D',
        'name': 'MatMul2',
        'ifm': ifm,
        'ofm': ofm,
        'kernel': create_kernel(width=1, height=1),
        'padding': create_padding(top=0, left=0, bottom=0, right=0),
        'weights': [create_address_range(region=0, address=0x52c0, length=100976)],
        'block_config': create_block_config(height=2, width=2, depth=128),
        'block_traversal': NpuBlockTraversal.DEPTH_FIRST,
        'rounding_mode': NpuRoundingMode.TFL,
        'ifm_upscale': NpuResamplingMode.NONE
    }
    
    # Create the operation
    return create_npu_operation_from_dict(config)


# Example of how to use these utilities
def main():
    # Create a Conv2D operation using the builder
    conv2d_op = create_example_conv2d()
    print(f"Created Conv2D operation: {conv2d_op.name}")
    print(f"  IFM shape: {conv2d_op.ifm.shape}")
    print(f"  OFM shape: {conv2d_op.ofm.shape}")
    print(f"  Block traversal: {conv2d_op.block_traversal}")
    
    # Create an ElementWise operation using the builder
    elementwise_op = create_example_elementwise()
    print(f"\nCreated ElementWise operation: {elementwise_op.name}")
    print(f"  IFM shape: {elementwise_op.ifm.shape}")
    print(f"  IFM2 shape: {elementwise_op.ifm2.shape}")
    print(f"  OFM shape: {elementwise_op.ofm.shape}")
    print(f"  Sub op type: {elementwise_op.sub_op_type}")
    
    # Create a Conv2D operation using the dictionary approach
    dict_conv2d_op = create_example_dict_conv2d()
    print(f"\nCreated Conv2D operation (dict approach): {dict_conv2d_op.name}")
    print(f"  IFM shape: {dict_conv2d_op.ifm.shape}")
    print(f"  OFM shape: {dict_conv2d_op.ofm.shape}")


if __name__ == "__main__":
    main()