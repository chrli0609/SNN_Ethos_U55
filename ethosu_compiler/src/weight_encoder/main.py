import numpy as np

from ethosu.vela.api import npu_encode_weights
from ethosu.vela.api import npu_encode_bias

from ethosu.vela.api import NpuAccelerator
from ethosu.vela.api import NpuBlockTraversal




def main():

    # weights
    accelerator=NpuAccelerator.Ethos_U55_256
    #weight_tensor = np.zeros((16,2,2,16))
    #weights_hwio = np.zeros((16, 2, 2, 16))
    #weights_ohwi = np.transpose(weights_hwio, (3, 0, 1, 2))
    weights_ohwi = np.zeros((16, 16, 2, 2), dtype=np.uint8)
    print(weights_ohwi.shape)
    weights_volume=weights_ohwi

    dilation_xy=(1,1)
    ifm_bitdepth = 8
    ofm_block_depth = 16
    is_depthwise = False
    #block_traversal=NpuBlockTraversal.PART_KERNEL_FIRST
    block_traversal=NpuBlockTraversal.DEPTH_FIRST

    #bias
    #bias_tensor = np.zeros(16)

    print("check input vals")
    print("accelerator:", accelerator.__class__, "\n", accelerator)
    print("weights_volume:", weights_volume.__class__, "\n", weights_volume.shape)
    print("dilation_xy:", dilation_xy.__class__, "\n", dilation_xy)
    print("ifm_bitdepth:", ifm_bitdepth.__class__, "\n", ifm_bitdepth)
    print("ofm_block_depth:", ofm_block_depth.__class__, "\n", ofm_block_depth)
    print("is_depthwise:", is_depthwise.__class__, "\n", is_depthwise)
    print("block_traversal:", block_traversal.__class__, "\n", block_traversal)


    #are_these_weights = encode_weights(
    are_these_weights = npu_encode_weights(
        accelerator=NpuAccelerator.Ethos_U55_256,
        weights_volume=weights_ohwi,
        dilation_xy=(1,1),
        ifm_bitdepth=ifm_bitdepth,
        ofm_block_depth=ofm_block_depth,
        is_depthwise=is_depthwise,
        block_traversal=NpuBlockTraversal.PART_KERNEL_FIRST,
    )


    #are_these_biases = encode_bias(
    are_these_biases = npu_encode_bias(
        bias=np.int64(0),
        scale=1077964381,
        shift=31
    )

    formatted_weights = ", ".join(f"0x{b:02x}" for b in are_these_weights)
    formatted_biases = ", ".join(f"0x{b:02x}" for b in are_these_biases)


    print("weights?\n", formatted_weights)
    print("biases?\n", formatted_biases)





'''

### encode_weights() ###

check input vals
accelerator: <enum 'Accelerator'>
 Accelerator.Ethos_U55_256
weights_volume: <class 'numpy.ndarray'>
 [[[[0 0 0 ... 0 0 0]
   [0 0 0 ... 0 0 0]]

  [[0 0 0 ... 0 0 0]
   [0 0 0 ... 0 0 0]]]


 [[[0 0 0 ... 0 0 0]
   [0 0 0 ... 0 0 0]]

  [[0 0 0 ... 0 0 0]
   [0 0 0 ... 0 0 0]]]


 [[[0 0 0 ... 0 0 0]
   [0 0 0 ... 0 0 0]]

  [[0 0 0 ... 0 0 0]
   [0 0 0 ... 0 0 0]]]


 ...


 [[[0 0 0 ... 0 0 0]
   [0 0 0 ... 0 0 0]]

  [[0 0 0 ... 0 0 0]
   [0 0 0 ... 0 0 0]]]


 [[[0 0 0 ... 0 0 0]
   [0 0 0 ... 0 0 0]]

  [[0 0 0 ... 0 0 0]
   [0 0 0 ... 0 0 0]]]


 [[[0 0 0 ... 0 0 0]
   [0 0 0 ... 0 0 0]]

  [[0 0 0 ... 0 0 0]
   [0 0 0 ... 0 0 0]]]]
dilation_xy: <class 'vela.operation.PointXY'>
 PointXY(x=1, y=1)
ifm_bitdepth: <class 'int'>
 8
ofm_block_depth: <class 'int'>
 16
is_depthwise: <class 'bool'>
 False
block_traversal: <enum 'NpuBlockTraversal'>
 NpuBlockTraversal.PART_KERNEL_FIRST
<vela.scheduler.SchedulerOpInfo object at 0x7f3bc81c8e20>: npu_weights_tensor is being modified: None -> <nng.Tensor 'sequential/conv2d/Conv2D_reshape_npu_npu_encoded_weights' shape=[1, 1, 1, 192] dtype=uint8>
######################




### encode_bias() ####
check input vals
bias: <class 'numpy.int64'>
 0
scale: <class 'int'>
 1077964381
shift: <class 'int'>
 31
######################

'''

main()
