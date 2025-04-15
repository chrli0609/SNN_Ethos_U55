


def float_to_int_safe(x: float) -> int:
    if not x.is_integer():
        raise ValueError(f"Cannot convert {x} to int: it has nonzero decimals.")
    return int(x)


def get_includes_str():
    return "#include <stddef.h>\n#include <stdint.h>\n\n\n\n\n"


def get_methods_str(op_name, base_name):
    return "\n\n\n\n\nconst uint8_t * Get" + op_name + "CMSPointer()\n{\n\treturn cms_" + base_name + ";\n}\n\nsize_t Get"+ op_name +"CMSLen()\n{\n\treturn sizeof(cms_" + base_name + ");\n}\n\n\n\n\n\nconst uint8_t * Get" + op_name + "WeightsPointer()\n{\n\treturn weights_" + base_name + ";\n}\n\nsize_t Get"+ op_name +"WeightsLen()\n{\n\treturn sizeof(weights_" + base_name + ");\n}\n\n"



def write_to_file(filepath, content):
    with open(filepath) as f:
        f.write(content)