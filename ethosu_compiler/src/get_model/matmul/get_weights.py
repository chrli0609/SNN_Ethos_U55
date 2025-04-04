import numpy as np

from ethosu.vela.api import NpuAccelerator
from ethosu.vela.api import NpuBlockTraversal

from ethosu.vela.api import npu_encode_weights
from ethosu.vela.api import npu_encode_bias




def gen_weights_and_biases(
        accelerator,
        weights_volume_ohwi,
        dilation_xy,
        ifm_bitdepth,
        ofm_block_depth,
        is_depthwise,
        block_traversal,

        bias_list,
        scale_list,
        shift_list,

        is_debug_mode=False,


):

    num_biases = len(bias_list)
    ifm_depth = weights_volume_ohwi.shape[3]


    # Make checks
    #if num_biases != ifm_depth:
    #    print("Error: Incorrect Dim(Bias), expected len(bias_tensor) == len(weights_volume_ohwi.shape[3]), instead got:\n\tlen(bias_tenosr):", num_biases, "\n\tweights_tensor_ohwi.shape[3]:", ifm_depth)
    #    exit()




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
    for i in range(num_biases):
        bias_bytearr_list.append(npu_encode_bias(
            bias=bias_list[i],
            scale=scale_list[i],
            shift=shift_list[i]
        ))

    formatted_weights = ", ".join(f"0x{b:02x}" for b in weight_bytearr)

    len_bias_bytearr = 0
    formatted_biases = ''
    for i in range(num_biases):
        len_bias_bytearr += len(bias_bytearr_list[i])
        formatted_biases += ", ".join(f"0x{b:02x}" for b in bias_bytearr_list[i]) + ",\n"


    tot_tensor_byte_len = len_bias_bytearr+len(weight_bytearr)
    if is_debug_mode:
        print("//len(biases):", len_bias_bytearr, " as hex: (" + hex(len_bias_bytearr) + ")\n")
        print("//len(weights):", len(weight_bytearr), " as hex: (" + hex(len(weight_bytearr)) + ")\n")
        print("//tot len:", tot_tensor_byte_len, " as hex: (" + hex(tot_tensor_byte_len) + ")\n\n")


    print("static const uint8_t weights_matmul[] __attribute__((aligned(16))) = \n{\n")
    print("//biases:\n", formatted_biases)
    print("//weights:\n", formatted_weights)
    print("};\n\n\n\n\n")


    return len(weight_bytearr), len_bias_bytearr