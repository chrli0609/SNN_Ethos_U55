import importlib
import sys
import os

from cms_translator.my_translator import assembly_2_machine_code
from weight_encoder.weight_encoder import encode_weights_and_biases






def main(input_name, output_name, params):
    
    if (assembly_2_machine_code(input_name, output_name) != 0):
        print("An error has occurred when translating from cms to cms code")
        exit()

    

    if (encode_weights_and_biases(
        output_name,

        params.accelerator,
        params.weights_volume,
        params.dilation_xy,
        params.ifm_bitdepth,
        params.ofm_block_depth,
        params.is_depthwise,
        params.block_traversal,

        params.bias,
        params.scale,
        params.shift
        != 0)):
        print("An error occurred when generating and writing weight stream")
        exit()


def load_module_from_path(module_path):
    module_name = os.path.splitext(os.path.basename(module_path))[0]  # Extract "conv2d_doc_ex_weights"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None:
        print(f"Error: Module '{module_path}' not found.")
        sys.exit(1)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python3 main.py <cms_file_input> <weights_params_module> <output_file>")
        sys.exit(1)

    input_name = sys.argv[1]
    output_name = sys.argv[3]
    params_module_name = sys.argv[2]  # Get the module name from command line
    #try:
    #    params = importlib.import_module(params_module_name)  # Dynamically import
    #except ModuleNotFoundError:
    #    print(f"Error: Module '{params_module_name}' not found.")
    #    sys.exit(1)


    if not os.path.isfile(params_module_name):
        print(f"Error: Module '{params_module_name}' not found.")
        sys.exit(1)

    module = load_module_from_path(params_module_name)


    # Call the function with dynamically loaded parameters
    main(input_name, output_name, module)