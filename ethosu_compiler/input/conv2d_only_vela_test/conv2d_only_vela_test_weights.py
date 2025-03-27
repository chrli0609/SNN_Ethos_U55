import numpy as np

from ethosu.vela.api import NpuAccelerator
from ethosu.vela.api import NpuBlockTraversal




accelerator=NpuAccelerator.Ethos_U55_256
weights_volume=np.zeros((16, 2, 2, 16), dtype=np.uint8)
dilation_xy=(1,1)
ifm_bitdepth = 8
ofm_block_depth = 16
is_depthwise = False
block_traversal = NpuBlockTraversal.PART_KERNEL_FIRST

bias = np.int64(0)
scale=1077964381
shift=31