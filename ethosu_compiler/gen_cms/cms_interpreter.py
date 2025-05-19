from prettytable import PrettyTable

from typing import List
import struct

cmd0_dict = {
    0x000: "NPU_OP_STOP",
    0x001: "NPU_OP_IRQ",
    0x002: "NPU_OP_CONV",
    0x003: "NPU_OP_DEPTHWISE",
    0x005: "NPU_OP_POOL",
    0x006: "NPU_OP_ELEMENTWISE",
    0x010: "NPU_OP_DMA_START",
    0x011: "NPU_OP_DMA_WAIT",
    0x012: "NPU_OP_KERNEL_WAIT",
    0x013: "NPU_OP_PMU_MASK",
    0x100: "NPU_SET_IFM_PAD_TOP",
    0x101: "NPU_SET_IFM_PAD_LEFT",
    0x102: "NPU_SET_IFM_PAD_RIGHT",
    0x103: "NPU_SET_IFM_PAD_BOTTOM",
    0x104: "NPU_SET_IFM_DEPTH_M1",
    0x105: "NPU_SET_IFM_PRECISION",
    0x107: "NPU_SET_IFM_UPSCALE",
    0x109: "NPU_SET_IFM_ZERO_POINT",
    0x10A: "NPU_SET_IFM_WIDTH0_M1",
    0x10B: "NPU_SET_IFM_HEIGHT0_M1",
    0x10C: "NPU_SET_IFM_HEIGHT1_M1",
    0x10D: "NPU_SET_IFM_IB_END",
    0x10F: "NPU_SET_IFM_REGION",
    0x111: "NPU_SET_OFM_WIDTH_M1",
    0x112: "NPU_SET_OFM_HEIGHT_M1",
    0x113: "NPU_SET_OFM_DEPTH_M1",
    0x114: "NPU_SET_OFM_PRECISION",
    0x115: "NPU_SET_OFM_BLK_WIDTH_M1",
    0x116: "NPU_SET_OFM_BLK_HEIGHT_M1",
    0x117: "NPU_SET_OFM_BLK_DEPTH_M1",
    0x118: "NPU_SET_OFM_ZERO_POINT",
    0x11A: "NPU_SET_OFM_WIDTH0_M1",
    0x11B: "NPU_SET_OFM_HEIGHT0_M1",
    0x11C: "NPU_SET_OFM_HEIGHT1_M1",
    0x11F: "NPU_SET_OFM_REGION",
    0x120: "NPU_SET_KERNEL_WIDTH_M1",
    0x121: "NPU_SET_KERNEL_HEIGHT_M1",
    0x122: "NPU_SET_KERNEL_STRIDE",
    0x124: "NPU_SET_ACC_FORMAT",
    0x125: "NPU_SET_ACTIVATION",
    0x126: "NPU_SET_ACTIVATION_MIN",
    0x127: "NPU_SET_ACTIVATION_MAX",
    0x128: "NPU_SET_WEIGHT_REGION",
    0x129: "NPU_SET_SCALE_REGION",
    0x12D: "NPU_SET_AB_START",
    0x12F: "NPU_SET_BLOCKDEP",
    0x130: "NPU_SET_DMA0_SRC_REGION",
    0x131: "NPU_SET_DMA0_DST_REGION",
    0x180: "NPU_SET_IFM2_BROADCAST",
    0x181: "NPU_SET_IFM2_SCALAR",
    0x185: "NPU_SET_IFM_PRECISION",
    0x189: "NPU_SET_IFM2_ZERO_POINT",
    0x18A: "NPU_SET_IFM2_WIDTH0_M1",
    0x18B: "NPU_SET_IFM2_HEIGHT0_M1",
    0x18C: "NPU_SET_IFM2_HEIGHT1_M1",
    0x18D: "NPU_SET_IFM2_IB_START",
    0x18F: "NPU_SET_IFM2_REGION"
}


cmd1_dict = {
    0x000: "NPU_SET_IFM_BASE0",
    0x001: "NPU_SET_IFM_BASE1",
    0x002: "NPU_SET_IFM_BASE2",
    0x003: "NPU_SET_IFM_BASE3",
    0x004: "NPU_SET_IFM_STRIDE_X",
    0x005: "NPU_SET_IFM_STRIDE_Y",
    0x006: "NPU_SET_IFM_STRIDE_C",
    0x010: "NPU_SET_OFM_BASE0",
    0x011: "NPU_SET_OFM_BASE1",
    0x012: "NPU_SET_OFM_BASE2",
    0x013: "NPU_SET_OFM_BASE3",
    0x014: "NPU_SET_OFM_STRIDE_X",
    0x015: "NPU_SET_OFM_STRIDE_Y",
    0x016: "NPU_SET_OFM_STRIDE_C",
    0x020: "NPU_SET_WEIGHT_BASE",
    0x021: "NPU_SET_WEIGHT_LENGTH",
    0x022: "NPU_SET_SCALE_BASE",
    0x023: "NPU_SET_SCALE_LENGTH",
    0x024: "NPU_SET_OFM_SCALE",
    0x025: "NPU_SET_OPA_SCALE",
    0x026: "NPU_SET_OPB_SCALE",
    0x030: "NPU_SET_DMA0_SRC",
    0x031: "NPU_SET_DMA0_DST",
    0x032: "NPU_SET_DMA0_LEN",
    0x033: "NPU_SET_DMA0_SKIP0",
    0x034: "NPU_SET_DMA0_SKIP1",
    0x080: "NPU_SET_IFM2_BASE0",
    0x081: "NPU_SET_IFM2_BASE1",
    0x082: "NPU_SET_IFM2_BASE2",
    0x083: "NPU_SET_IFM2_BASE3",
    0x084: "NPU_SET_IFM2_STRIDE_X",
    0x085: "NPU_SET_IFM2_STRIDE_Y",
    0x086: "NPU_SET_IFM2_STRIDE_C",
    0x090: "NPU_SET_WEIGHT1_BASE",
    0x091: "NPU_SET_WEIGHT1_LENGTH",
    0x092: "NPU_SET_SCALE1_BASE",
    0x093: "NPU_SET_SCALE1_LENGTH",
}




def register_cms_2_assembly(register_cms):
    

    ret_str = '/*\n'

    # Create a PrettyTable instance
    ret_str += "Register Command Stream:\n"
    table = PrettyTable(["Command Name", "Parameter", "Payload Data"])
    table.align = "l"

    i = 0
    while i < len(register_cms):

        command = register_cms[i]




        bin_cmd = bin(command)[2:].zfill(32)[::-1]

        #print("bin_cmd:", format(int(bin_cmd, 2), '08x'), "\t", bin_cmd)
        
        #print("bin_cmd[10:16]:", bin_cmd[10:16])
        
        cmd = (bin_cmd[:10])[::-1]
        parameter = (bin_cmd[16:])[::-1]

        if (bin_cmd[10:16])[::-1] == '000000': #cmd0 
            cmd_name = cmd0_dict[int(cmd, 2)]
            has_payload = False
        elif (bin_cmd[10:16])[::-1] == '010000': #cmd1
            cmd_name = cmd1_dict[int(cmd, 2)]
            has_payload = True
        else:
            print("Do not recognize command type, neither cmd0 nor cmd1")
            exit()

            

        
        #print(f"{cmd_name}\t\t{int(parameter, 2)}", end='')
        #if has_payload:
        #    print("\t\t" + format(register_cms[i+1], '08x'))
        #else:
        #    print()



        

        
        
        row = [cmd_name, int(parameter, 2)]  # Convert binary parameter to integer

        if has_payload:
            row.append("0x" + format(register_cms[i + 1], '08x') + " ("+str(register_cms[i + 1])+")")  # Format payload as hex
        else:
            row.append("-")  # Placeholder if no payload

        table.add_row(row)

       

        if has_payload:
            i += 2
        else:
            i += 1





     # Print the table
    ret_str += str(table)
    ret_str += "\n*/\n"

    return ret_str






def extract_register_command_stream(driver_payload: bytes) -> List[int]:
    """
    Extract the register command stream from a driver payload.
    This is the opposite of npu_create_driver_payload().
    
    Args:
        driver_payload (bytes): The driver payload created by npu_create_driver_payload()
        
    Returns:
        List[int]: The extracted register command stream as a list of 32-bit integers
    """
    # First, convert bytes to a list of 32-bit integers (little-endian)
    words = list(struct.unpack(f"<{len(driver_payload)//4}I", driver_payload))
    
    # Check for the COP1 fourcc at the beginning
    if len(words) < 1:
        raise ValueError("Driver payload is too short")
    
    # Check for the "COP1" fourcc
    fourcc_value = words[0]
    fourcc = chr(fourcc_value & 0xFF) + chr((fourcc_value >> 8) & 0xFF) + \
             chr((fourcc_value >> 16) & 0xFF) + chr((fourcc_value >> 24) & 0xFF)
    if fourcc != "COP1":
        raise ValueError(f"Invalid fourcc in driver payload: {fourcc}")
    
    # Look for the command stream header (DACommands.CmdStream = 0x02)
    cmd_stream_idx = None
    for i in range(1, len(words)):
        cmd_id = words[i] & 0xFF
        if cmd_id == 0x02:  # CmdStream command
            cmd_stream_idx = i
            break
    
    if cmd_stream_idx is None:
        raise ValueError("Could not find command stream header in driver payload")
    
    # Extract the length from the command stream header
    cs_header = words[cmd_stream_idx]
    length_high = ((cs_header >> 8) & 0xFF) << 16
    length_low = (cs_header >> 16) & 0xFFFF
    length = length_high | length_low
    
    # The command stream starts after the header
    cmd_stream_start = cmd_stream_idx + 1
    cmd_stream_end = cmd_stream_start + length
    
    if cmd_stream_end > len(words):
        raise ValueError(f"Command stream length ({length}) exceeds available data")
    
    # Extract the command stream
    register_command_stream = words[cmd_stream_start:cmd_stream_end]
    
    return register_command_stream






def strip_cmds_from_register_command_stream(register_cms, list_of_cmds_to_strip):
    

    new_cmd_table = '/*\n'
    new_register_cms = []

    # Create a PrettyTable instance
    new_cmd_table += "Register Command Stream:\n"
    table = PrettyTable(["Command Name", "Parameter", "Payload Data"])
    table.align = "l"

    i = 0
    while i < len(register_cms):


        command = register_cms[i]




        bin_cmd = bin(command)[2:].zfill(32)[::-1]

        #print("bin_cmd:", format(int(bin_cmd, 2), '08x'), "\t", bin_cmd)
        
        #print("bin_cmd[10:16]:", bin_cmd[10:16])
        
        cmd = (bin_cmd[:10])[::-1]
        parameter = (bin_cmd[16:])[::-1]

        if (bin_cmd[10:16])[::-1] == '000000': #cmd0 
            cmd_name = cmd0_dict[int(cmd, 2)]
            has_payload = False
        elif (bin_cmd[10:16])[::-1] == '010000': #cmd1
            cmd_name = cmd1_dict[int(cmd, 2)]
            has_payload = True
        else:
            print("Do not recognize command type, neither cmd0 nor cmd1")
            exit()

            

        #skip if in list
        if (cmd_name in list_of_cmds_to_strip):
            if has_payload:
                i += 2
            else:
                i += 1
            continue

        # If not in list --> keep it
        new_register_cms.append(command)


        

        
        
        row = [cmd_name, int(parameter, 2)]  # Convert binary parameter to integer

        if has_payload:
            row.append("0x" + format(register_cms[i + 1], '08x') + " ("+str(register_cms[i + 1])+")")  # Format payload as hex
        else:
            row.append("-")  # Placeholder if no payload

        table.add_row(row)

       

        if has_payload:
            i += 2
        else:
            i += 1





     # Print the table
    new_cmd_table += str(table)
    new_cmd_table += "\n*/\n"



    return new_register_cms, new_cmd_table

