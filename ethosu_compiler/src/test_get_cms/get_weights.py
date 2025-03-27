import numpy as np

from ethosu.vela.api import NpuAccelerator
from ethosu.vela.api import NpuBlockTraversal

from ethosu.vela.api import npu_encode_weights
from ethosu.vela.api import npu_encode_bias




accelerator=NpuAccelerator.Ethos_U55_256
#weights_volume_ohwi=np.zeros((16, 2, 2, 16), dtype=np.uint8)
#weights_volume_ohwi = (np.arange(1024) % 256).astype(np.uint8).reshape(16, 2, 2, 16)
weights_volume_ohwi=np.ones((16, 2, 2, 16), dtype=np.uint8)
#weights_volume_ohwi = np.full((16, 2, 2, 16), 2, dtype=np.uint8)
print("weights:\n", weights_volume_ohwi)


dilation_xy=(1,1)
ifm_bitdepth = 8
ofm_block_depth = 16
is_depthwise = False
block_traversal = NpuBlockTraversal.PART_KERNEL_FIRST

#bias = np.int64(549755813888)

NUM_FILTERS = 16
bias_list = []
scale_list = []
shift_list = []
for i in range(NUM_FILTERS):
    bias_list.append(np.int64(0))
    scale_list.append(1)
    shift_list.append(0)
#scale=1077964381
#scale=0
#shift=31
#shift=63

#I should have 16 biases, right?




weight_bytearr = npu_encode_weights(
        accelerator=accelerator,
        weights_volume=weights_volume_ohwi,
        dilation_xy=dilation_xy,
        ifm_bitdepth=ifm_bitdepth,
        ofm_block_depth=ofm_block_depth,
        is_depthwise=is_depthwise,
        block_traversal=block_traversal,
    )



bias_bytearr_list = []
for i in range(NUM_FILTERS):
    bias_bytearr_list.append(npu_encode_bias(
        bias=bias_list[i],
        scale=scale_list[i],
        shift=shift_list[i]
    ))

formatted_weights = ", ".join(f"0x{b:02x}" for b in weight_bytearr)

len_bias_bytearr = 0
formatted_biases = ''
for i in range(NUM_FILTERS):
    len_bias_bytearr += len(bias_bytearr_list[i])
    formatted_biases += ", ".join(f"0x{b:02x}" for b in bias_bytearr_list[i]) + ",\n"


print("len(biases):", len_bias_bytearr, " as hex: (" + hex(len_bias_bytearr) + ")\n")
print("len(weights):", len(weight_bytearr), " as hex: (" + hex(len(weight_bytearr)) + ")\n")
tot_byte_len = len_bias_bytearr+len(weight_bytearr)
print("tot len:", tot_byte_len, " as hex: (" + hex(tot_byte_len) + ")\n\n")

print("biases:\n", formatted_biases)
print("weights:\n", formatted_weights)

