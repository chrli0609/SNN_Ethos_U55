import sys

##################### my own parser ##########################
from register_command_stream_generator import CommandStreamEmitter
from ethos_u55_regs import cmd0
from ethos_u55_regs import cmd1



# Necessary cmd_stream generation functions
'''
emit.cmd0_with_param
emit.cmd1_with_address
emit.cmd1_with_offset

'''

cmd0_dict = {
    "NPU_OP_STOP": cmd0.NPU_OP_STOP,
    "NPU_OP_IRQ": cmd0.NPU_OP_IRQ,
    "NPU_OP_CONV": cmd0.NPU_OP_CONV,
    "NPU_OP_DEPTHWISE": cmd0.NPU_OP_DEPTHWISE,
    "NPU_OP_POOL": cmd0.NPU_OP_POOL,
    "NPU_OP_ELEMENTWISE": cmd0.NPU_OP_ELEMENTWISE,
    "NPU_OP_DMA_START": cmd0.NPU_OP_DMA_START,
    "NPU_OP_DMA_WAIT": cmd0.NPU_OP_DMA_WAIT,
    "NPU_OP_KERNEL_WAIT": cmd0.NPU_OP_KERNEL_WAIT,
    "NPU_OP_PMU_MASK": cmd0.NPU_OP_PMU_MASK,
    "NPU_SET_IFM_PAD_TOP": cmd0.NPU_SET_IFM_PAD_TOP,
    "NPU_SET_IFM_PAD_LEFT": cmd0.NPU_SET_IFM_PAD_LEFT,
    "NPU_SET_IFM_PAD_RIGHT": cmd0.NPU_SET_IFM_PAD_RIGHT,
    "NPU_SET_IFM_PAD_BOTTOM": cmd0.NPU_SET_IFM_PAD_BOTTOM,
    "NPU_SET_IFM_DEPTH_M1": cmd0.NPU_SET_IFM_DEPTH_M1,
    "NPU_SET_IFM_PRECISION": cmd0.NPU_SET_IFM_PRECISION,
    "NPU_SET_IFM_UPSCALE": cmd0.NPU_SET_IFM_UPSCALE,
    "NPU_SET_IFM_ZERO_POINT": cmd0.NPU_SET_IFM_ZERO_POINT,
    "NPU_SET_IFM_WIDTH0_M1": cmd0.NPU_SET_IFM_WIDTH0_M1,
    "NPU_SET_IFM_HEIGHT0_M1": cmd0.NPU_SET_IFM_HEIGHT0_M1,
    "NPU_SET_IFM_HEIGHT1_M1": cmd0.NPU_SET_IFM_HEIGHT1_M1,
    "NPU_SET_IFM_IB_END": cmd0.NPU_SET_IFM_IB_END,
    "NPU_SET_IFM_REGION": cmd0.NPU_SET_IFM_REGION,
    "NPU_SET_OFM_WIDTH_M1": cmd0.NPU_SET_OFM_WIDTH_M1,
    "NPU_SET_OFM_HEIGHT_M1": cmd0.NPU_SET_OFM_HEIGHT_M1,
    "NPU_SET_OFM_DEPTH_M1": cmd0.NPU_SET_OFM_DEPTH_M1,
    "NPU_SET_OFM_PRECISION": cmd0.NPU_SET_OFM_PRECISION,
    "NPU_SET_OFM_BLK_WIDTH_M1": cmd0.NPU_SET_OFM_BLK_WIDTH_M1,
    "NPU_SET_OFM_BLK_HEIGHT_M1": cmd0.NPU_SET_OFM_BLK_HEIGHT_M1,
    "NPU_SET_OFM_BLK_DEPTH_M1": cmd0.NPU_SET_OFM_BLK_DEPTH_M1,
    "NPU_SET_OFM_ZERO_POINT": cmd0.NPU_SET_OFM_ZERO_POINT,
    "NPU_SET_OFM_WIDTH0_M1": cmd0.NPU_SET_OFM_WIDTH0_M1,
    "NPU_SET_OFM_HEIGHT0_M1": cmd0.NPU_SET_OFM_HEIGHT0_M1,
    "NPU_SET_OFM_HEIGHT1_M1": cmd0.NPU_SET_OFM_HEIGHT1_M1,
    "NPU_SET_OFM_REGION": cmd0.NPU_SET_OFM_REGION,
    "NPU_SET_KERNEL_WIDTH_M1": cmd0.NPU_SET_KERNEL_WIDTH_M1,
    "NPU_SET_KERNEL_HEIGHT_M1": cmd0.NPU_SET_KERNEL_HEIGHT_M1,
    "NPU_SET_KERNEL_STRIDE": cmd0.NPU_SET_KERNEL_STRIDE,
    "NPU_SET_PARALLEL_MODE": cmd0.NPU_SET_PARALLEL_MODE,
    "NPU_SET_ACC_FORMAT": cmd0.NPU_SET_ACC_FORMAT,
    "NPU_SET_ACTIVATION": cmd0.NPU_SET_ACTIVATION,
    "NPU_SET_ACTIVATION_MIN": cmd0.NPU_SET_ACTIVATION_MIN,
    "NPU_SET_ACTIVATION_MAX": cmd0.NPU_SET_ACTIVATION_MAX,
    "NPU_SET_WEIGHT_REGION": cmd0.NPU_SET_WEIGHT_REGION,
    "NPU_SET_SCALE_REGION": cmd0.NPU_SET_SCALE_REGION,
    "NPU_SET_AB_START": cmd0.NPU_SET_AB_START,
    "NPU_SET_BLOCKDEP": cmd0.NPU_SET_BLOCKDEP,
    "NPU_SET_DMA0_SRC_REGION": cmd0.NPU_SET_DMA0_SRC_REGION,
    "NPU_SET_DMA0_DST_REGION": cmd0.NPU_SET_DMA0_DST_REGION,
    "NPU_SET_DMA0_SIZE0": cmd0.NPU_SET_DMA0_SIZE0,
    "NPU_SET_DMA0_SIZE1": cmd0.NPU_SET_DMA0_SIZE1,
    "NPU_SET_IFM2_BROADCAST": cmd0.NPU_SET_IFM2_BROADCAST,
    "NPU_SET_IFM2_SCALAR": cmd0.NPU_SET_IFM2_SCALAR,
    "NPU_SET_IFM2_PRECISION": cmd0.NPU_SET_IFM2_PRECISION,
    "NPU_SET_IFM2_ZERO_POINT": cmd0.NPU_SET_IFM2_ZERO_POINT,
    "NPU_SET_IFM2_WIDTH0_M1": cmd0.NPU_SET_IFM2_WIDTH0_M1,
    "NPU_SET_IFM2_HEIGHT0_M1": cmd0.NPU_SET_IFM2_HEIGHT0_M1,
    "NPU_SET_IFM2_HEIGHT1_M1": cmd0.NPU_SET_IFM2_HEIGHT1_M1,
    "NPU_SET_IFM2_IB_START": cmd0.NPU_SET_IFM2_IB_START,
    "NPU_SET_IFM2_REGION": cmd0.NPU_SET_IFM2_REGION,
}



cmd1_dict = {
    "NPU_SET_IFM_BASE0": cmd1.NPU_SET_IFM_BASE0,
    "NPU_SET_IFM_BASE1": cmd1.NPU_SET_IFM_BASE1,
    "NPU_SET_IFM_BASE2": cmd1.NPU_SET_IFM_BASE2,
    "NPU_SET_IFM_BASE3": cmd1.NPU_SET_IFM_BASE3,
    "NPU_SET_IFM_STRIDE_X": cmd1.NPU_SET_IFM_STRIDE_X,
    "NPU_SET_IFM_STRIDE_Y": cmd1.NPU_SET_IFM_STRIDE_Y,
    "NPU_SET_IFM_STRIDE_C": cmd1.NPU_SET_IFM_STRIDE_C,
    "NPU_SET_OFM_BASE0": cmd1.NPU_SET_OFM_BASE0,
    "NPU_SET_OFM_BASE1": cmd1.NPU_SET_OFM_BASE1,
    "NPU_SET_OFM_BASE2": cmd1.NPU_SET_OFM_BASE2,
    "NPU_SET_OFM_BASE3": cmd1.NPU_SET_OFM_BASE3,
    "NPU_SET_OFM_STRIDE_X": cmd1.NPU_SET_OFM_STRIDE_X,
    "NPU_SET_OFM_STRIDE_Y": cmd1.NPU_SET_OFM_STRIDE_Y,
    "NPU_SET_OFM_STRIDE_C": cmd1.NPU_SET_OFM_STRIDE_C,
    "NPU_SET_WEIGHT_BASE": cmd1.NPU_SET_WEIGHT_BASE,
    "NPU_SET_WEIGHT_LENGTH": cmd1.NPU_SET_WEIGHT_LENGTH,
    "NPU_SET_SCALE_BASE": cmd1.NPU_SET_SCALE_BASE,
    "NPU_SET_SCALE_LENGTH": cmd1.NPU_SET_SCALE_LENGTH,
    "NPU_SET_OFM_SCALE": cmd1.NPU_SET_OFM_SCALE,
    "NPU_SET_OPA_SCALE": cmd1.NPU_SET_OPA_SCALE,
    "NPU_SET_OPB_SCALE": cmd1.NPU_SET_OPB_SCALE,
    "NPU_SET_DMA0_SRC": cmd1.NPU_SET_DMA0_SRC,
    "NPU_SET_DMA0_DST": cmd1.NPU_SET_DMA0_DST,
    "NPU_SET_DMA0_LEN": cmd1.NPU_SET_DMA0_LEN,
    "NPU_SET_DMA0_SKIP0": cmd1.NPU_SET_DMA0_SKIP0,
    "NPU_SET_DMA0_SKIP1": cmd1.NPU_SET_DMA0_SKIP1,
    "NPU_SET_IFM2_BASE0": cmd1.NPU_SET_IFM2_BASE0,
    "NPU_SET_IFM2_BASE1": cmd1.NPU_SET_IFM2_BASE1,
    "NPU_SET_IFM2_BASE2": cmd1.NPU_SET_IFM2_BASE2,
    "NPU_SET_IFM2_BASE3": cmd1.NPU_SET_IFM2_BASE3,
    "NPU_SET_IFM2_STRIDE_X": cmd1.NPU_SET_IFM2_STRIDE_X,
    "NPU_SET_IFM2_STRIDE_Y": cmd1.NPU_SET_IFM2_STRIDE_Y,
    "NPU_SET_IFM2_STRIDE_C": cmd1.NPU_SET_IFM2_STRIDE_C,
    "NPU_SET_WEIGHT1_BASE": cmd1.NPU_SET_WEIGHT1_BASE,
    "NPU_SET_WEIGHT1_LENGTH": cmd1.NPU_SET_WEIGHT1_LENGTH,
    "NPU_SET_SCALE1_BASE": cmd1.NPU_SET_SCALE1_BASE,
    "NPU_SET_SCALE1_LENGTH": cmd1.NPU_SET_SCALE1_LENGTH,
}


def process_cmd0(emit, npu_op_str):
    
    npu_op_parts = npu_op_str.split(',')

    #make sure only have two parts (command and parameter)
    if len(npu_op_parts) != 2:
        print("ERROR: found more than 2 elements in cmd0 command", npu_op_parts)
        return -1
    
    npu_cmd_str = npu_op_parts[0]
    parameter_str = npu_op_parts[1]

    #convert command to enum type:
    try:
        npu_cmd = cmd0_dict[npu_cmd_str]
    except:
        print("ERROR: npu_cmd not found:", npu_cmd_str)
        return -1
    
    #convert parameter to value (should it be binary or hex or integer?)
    try:
        parameter = int(parameter_str, 16)
    except:
        print("ERROR: parameter could not be converted to hex, parameter:", parameter_str)


    #Add the command stream
    emit.cmd0_with_param(npu_cmd, parameter)

    return npu_op_parts


def process_cmd1(emit, npu_op_str):
    npu_op_parts = npu_op_str.split(',')

    #make sure only have two parts (command and parameter, payload)
    if len(npu_op_parts) != 3:
        print("ERROR: found more than 2 elements in cmd0 command", npu_op_parts)
        return -1
    
    npu_cmd_str = npu_op_parts[0]
    parameter_str = npu_op_parts[1]
    payload_str = npu_op_parts[2]

    #convert command to enum type:
    try:
        npu_cmd = cmd1_dict[npu_cmd_str]
    except:
        print("ERROR: npu_cmd not found:", npu_cmd_str)
        return -1
    
    #convert parameter to value (should it be binary or hex or integer?)
    try:
        parameter = int(parameter_str, 16)
    except:
        print("ERROR: parameter could not be converted to hex, parameter:", parameter_str)
        return -1

    try:
        payload = int(payload_str, 16)
    except:
        print("ERROR: payload could not be converted to hex, parameter:", payload_str)
        return -1


    emit.cmd1_with_offset(npu_cmd, payload, parameter)

    return npu_op_parts



def parse_assembly(input_name, emit):


    with open(input_name, 'r', encoding='utf8') as file:
        for line in file:

            #remove newline
            line = line.strip('\n')
            #remove white spaces
            line = line.replace(' ', '')
            #remove all characters after '#' to allow for comments
            line = line.split('#', 1)[0]
            print("line:", line)

            #decide if its cmd0 or cmd1
            tmp = line.split('.')

            #Make sure only 1 '.' in code
            if len(tmp) > 2:
                print("ERROR: should only have 1 '.' in each line, found:", len(tmp))

            cmd_type = tmp[0]
            npu_op_str = tmp[1]


            if cmd_type == 'cmd0':
                if(process_cmd0(emit, npu_op_str) == -1):
                    return -1

            elif cmd_type == 'cmd1':
                if (process_cmd1(emit, npu_op_str) == -1):
                    return -1
            else:
                print("ERROR: Unrecognized command type, expected either cmd0 or cmd1, got:", cmd_type)
                return -1
    
    return emit


def get_header():
    return "#include <cstddef>\n#include <cstdint>\n\n\n\n\nstatic const uint8_t command_stream[] __attribute__((aligned(16))) =\n\t{\n"

def get_end_stuff():
    return "};\n\n\n\n\nconst uint8_t * GetModelPointer()\n{\n\treturn command_stream;\n}\n\nsize_t GetModelLen()\n{\n\treturn sizeof(command_stream);\n}"
    

def pretty_string(split_up_hex_vals_list, next_is_payload=False):
    
    out_str = ''

    for i in range(4): #32 bits split to 1 byte chunks --> 32/(8) = 4
        out_str += "0x" + str(format(split_up_hex_vals_list[i], '02x')) + ', '

    if next_is_payload:
        out_str += '\t'
    else:
        out_str += '\n'

    return out_str

def split_to_two_byte_and_little_endian(hex_val_8_bit):
    split_up = []
    for i in range(4):
        split_up.append(hex_val_8_bit >> 8*i & 0xFF)

    return split_up


def write_to_file(emit, output_tfl_filename):
    print("output_tfl_filename:", output_tfl_filename)
    print("emitting emit.cmd_stream:\n", emit.cmd_stream)


    out_str = ''

    # Print header:
    out_str += get_header()



    #Command stream in C array format
    for i in range(len(emit.cmd_stream)):
        cmd_tuple = emit.cmd_stream[i]


        if len(cmd_tuple) == 1:
            split_up_cmd = split_to_two_byte_and_little_endian(cmd_tuple[0])
            out_str += pretty_string(split_up_cmd, next_is_payload=False)

        elif len(cmd_tuple) == 2:
            split_up_cmd = split_to_two_byte_and_little_endian(cmd_tuple[0])
            split_up_payload = split_to_two_byte_and_little_endian(cmd_tuple[1])

            out_str += pretty_string(split_up_cmd, next_is_payload=True)
            out_str += pretty_string(split_up_payload, next_is_payload=False)

        else:
            print("ERROR: Expected 1 or 2 elements in command stream but got:", len(cmd_tuple))

    #End stuff
    out_str += get_end_stuff()
        
    with open(output_tfl_filename, "w") as text_file:
        text_file.write(out_str)
    





def assembly_2_machine_code(input_name, output_name):

   	#if compiler_options.timing:
   	#     start = time.time()

    #os.makedirs(compiler_options.output_dir, exist_ok=True)
    #output_basename = os.path.join(compiler_options.output_dir, os.path.splitext(os.path.basename(input_name))[0])
    #DebugDatabase.show_warnings = enable_debug_db


    emit = CommandStreamEmitter()

    emit = parse_assembly(input_name, emit)
    if emit == -1:
        print("ERROR: parse_assembly failed")
        return -1


    #output_tfl_filename = output_basename + "_translated.hpp"
    
    write_to_file(emit, output_name)


    return 0



if __name__ == "__main__":
	assembly_2_machine_code(sys.argv[1], sys.argv[2])

