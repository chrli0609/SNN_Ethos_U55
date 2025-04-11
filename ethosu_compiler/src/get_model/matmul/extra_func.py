


def float_to_int_safe(x: float) -> int:
    if not x.is_integer():
        raise ValueError(f"Cannot convert {x} to int: it has nonzero decimals.")
    return int(x)



def get_includes_str():
    return "#include <stddef.h>\n#include <stdint.h>\n\n\n\n\n"


def get_methods_str(base_name):
    return "\n\n\n\n\nconst uint8_t * Get" + base_name + "CMSPointer()\n{\n\treturn cms_" + base_name + ";\n}\n\nsize_t Get"+ base_name +"CMSLen()\n{\n\treturn sizeof(cms_" + base_name + ");\n}\n\n\n\n\n\nconst uint8_t * Get" + base_name + "WeightsPointer()\n{\n\treturn weights_" + base_name + ";\n}\n\nsize_t Get"+ base_name +"WeightsLen()\n{\n\treturn sizeof(weights_" + base_name + ");\n}\n\n"


def get_array_def_str(array_type, base_name):
    return "\n\n\n\nstatic const uint8_t " + array_type + "_" + base_name + "[] __attribute__((aligned(16))) = \n{"

def write_cms_to_file(filepath, driver_payload_byte_array, base_name):
    # Print CMS
    formatted_cms = ", ".join(f"0x{b:02x}" for b in driver_payload_byte_array)
    print("*/")
    print("\n\n\n\nstatic const uint8_t cms_matmul[] __attribute__((aligned(16))) = \n{")
    print(formatted_cms)
    print("};\n\n\n")
    with open(filepath) as f:

        f.write(get_array_def_str("cms", base_name))
        f.write(formatted_cms)
        f.write("};\n\n\n")


    