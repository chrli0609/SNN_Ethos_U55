import numpy as np

from architecture_features import Accelerator
from api import NpuBlockTraversal

from api import npu_encode_weights
from api import npu_encode_bias


#from weight_compressor import encode_weights
#from weight_compressor import encode_bias





def main():

    # weights
    weight_tensor = np.zeros((2,2,16,16))
    ifm_bitdepth = 8
    ofm_block_depth = 16
    is_depthwise = False

    #bias
    #bias_tensor = np.zeros(16)


    #are_these_weights = encode_weights(
    are_these_weights = npu_encode_weights(
        accelerator=Accelerator.Ethos_U55_256,
        weights_volume=weight_tensor,
        dilation_xy=(1,1),
        ifm_bitdepth=ifm_bitdepth,
        ofm_block_depth=ofm_block_depth,
        is_depthwise=is_depthwise,
        block_traversal=NpuBlockTraversal.PART_KERNEL_FIRST,
    )


    #are_these_biases = encode_bias(
    are_these_biases = npu_encode_bias(
        bias=0,
        scale=1077964381,
        shift=31
    )


    print("weights?\n", are_these_weights)
    print("biases?\n", are_these_biases)





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