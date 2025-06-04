#!/usr/bin/env python3
"""
Script to convert NumPy .npy files to C header file with int8_t arrays
"""

import numpy as np
import sys
import os



def numpy_to_c_array_3d(arr, var_name, mem_section_name, num_samples_macro_name, output_file):
    """Convert 3D numpy array to C array format and write to file"""
    depth, rows, cols = arr.shape
    
    # Write array declaration
    output_file.write(f"static volatile __attribute__((section(\"{mem_section_name}\"))) int8_t {var_name}[{num_samples_macro_name}][{rows}][{cols}] = {{\n")
    
    # Write array data
    for d in range(depth):
        output_file.write("    {\n")
        for i in range(rows):
            output_file.write("        {")
            for j in range(cols):
                # Convert to int8_t (clamp to [-128, 127] range)
                val_dequant = arr[d, i, j]
                val = int(np.clip(val_dequant, -128, 127))
                # Check if integer value
                if (val_dequant != int(val_dequant)):
                    print("WE HAVE FLOATING POINT INPUT VALUE!!!!!!")
                output_file.write(f"{val}")
                if j < cols - 1:
                    output_file.write(", ")
            output_file.write("}")
            if i < rows - 1:
                output_file.write(",")
            output_file.write("\n")
        output_file.write("    }")
        if d < depth - 1:
            output_file.write(",")
        output_file.write("\n")
    
    output_file.write("};\n\n")



def numpy_to_c_array_2d(arr, var_name, mem_section_name, num_samples_macro_name, output_file):
    """Convert 2D numpy array to C array format and write to file"""
    rows, cols = arr.shape
    
    # Write array declaration
    output_file.write(f"static volatile __attribute__((section(\"{mem_section_name}\"))) int8_t {var_name}[{num_samples_macro_name}][{cols}] = {{\n")
    
    # Write array data
    for i in range(rows):
        output_file.write("    {")
        for j in range(cols):

            # Convert to int8_t (clamp to [-128, 127] range)
            val_dequant = arr[i,j]
            val = int(np.clip(val_dequant, -128, 127))

            # Check if integer value
            if (val_dequant != int(val_dequant)):
                print("WE HAVE FLOATING POINT INPUT VALUE!!!!!!")
            output_file.write(f"{val}")
            if j < cols - 1:
                output_file.write(", ")
        output_file.write("}")
        if i < rows - 1:
            output_file.write(",")
        output_file.write("\n")
    
    output_file.write("};\n\n")

def numpy_to_c_array_1d(arr, var_name, mem_section_name, num_samples_macro_name, output_file):
    """Convert 1D numpy array to C array format and write to file"""
    length = arr.shape[0]
    
    # Write array declaration
    output_file.write(f"static volatile __attribute__((section(\"{mem_section_name}\"))) int8_t {var_name}[{num_samples_macro_name}] = {{\n")
    
    # Write array data
    output_file.write("    ")
    for i in range(length):
        # Convert to int8_t (clamp to [-128, 127] range)
        val = int(np.clip(arr[i], -128, 127))
        output_file.write(f"{val}")
        if i < length - 1:
            output_file.write(", ")
        # Add line breaks every 16 elements for readability
        if (i + 1) % 16 == 0 and i < length - 1:
            output_file.write("\n    ")
    
    output_file.write("\n};\n\n")

def test_patterns_2_h_file(mem_section_name, input_file_path, target_file_path, output_file_path):
    # File paths
    #input_file_path = "test_input.npy"
    #target_file_path = "test_target.npy"
    #output_file_path = "test_data.h"
    
    num_samples_macro_name = str(input_file_path.stem) + "_NUM_SAMPLES"
    num_samples = 10
    
    # Allow command line arguments to override default paths
    if len(sys.argv) >= 2:
        mem_section_name = sys.argv[1]
    if len(sys.argv) >= 3:
        input_file_path = sys.argv[2]
    if len(sys.argv) >= 4:
        target_file_path = sys.argv[3]
    if len(sys.argv) >= 5:
        output_file_path = sys.argv[4]
    
    try:
        # Load numpy arrays
        print(f"Loading {input_file_path}...")
        test_input_untransposed = np.load(input_file_path)
        print("Transpose test_inputs to get samples on rows instead of columns")
        test_input_unstripped = np.transpose(test_input_untransposed, (1, 0, 2))  # Rearrange axes
        print(f"Loaded test_input with shape: {test_input_unstripped.shape}")
        print("Extract only the first {num_samples} samples")
        test_input = test_input_unstripped[:num_samples]
        
        print(f"Loading {target_file_path}...")
        test_target_unstripped = np.load(target_file_path)
        print(f"Loaded test_target with shape: {test_target_unstripped.shape}")
        print(f"Extract only the first {num_samples} samples")
        test_target = test_target_unstripped[:num_samples]
        



        # Validate dimensions
        test_input_num_rows = test_input.shape[0]
        test_target_len = test_target.shape[0]

        
        if test_target_len != test_input_num_rows:
            print(f"Warning: test_target shape {test_target.shape} and test_input shape {test_input.shape} don't match")
        
        # Check data range and warn if values will be clipped
        input_min, input_max = test_input.min(), test_input.max()
        target_min, target_max = test_target.min(), test_target.max()

        
        if input_min < -128 or input_max > 127:
            print(f"Warning: test_input values range [{input_min}, {input_max}] will be clipped to [-128, 127]")
        
        if target_min < -128 or target_max > 127:
            print(f"Warning: test_target values range [{target_min}, {target_max}] will be clipped to [-128, 127]")
        
        # Generate C header file
        print(f"Writing C header to {output_file_path}...")
        with open(output_file_path, 'w') as f:
            # Write header comment
            f.write("/*\n")
            f.write(" * Generated C header file from NumPy arrays\n")
            f.write(f" * {input_file_path} shape: {test_input.shape}\n")
            f.write(f" * {target_file_path} shape: {test_target.shape}\n")
            f.write(" */\n\n")
            
            f.write("#ifndef TEST_DATA_H\n")
            f.write("#define TEST_DATA_H\n\n")
            
            f.write("#include <stdint.h>\n\n")

            f.write("#define " +  num_samples_macro_name + " " + str(test_input_num_rows) + "\n\n")

            
            # Convert and write test_target array
            numpy_to_c_array_1d(test_target, str(target_file_path.stem), mem_section_name, num_samples_macro_name, f)

            # Convert and write test_input array
            #numpy_to_c_array_2d(test_input, str(input_file_path.stem), mem_section_name, num_samples_macro_name, f)
            # Transpose first so we get different samples on the rows
            numpy_to_c_array_3d(test_input, str(input_file_path.stem), mem_section_name, num_samples_macro_name, f)
            
            
            f.write("#endif // TEST_DATA_H\n")
        
        print(f"Successfully generated {output_file_path}")
        print(f"File size: {os.path.getsize(output_file_path)} bytes")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("NumPy to C Header Converter")
    print("Usage: python script.py [mem_section_name] [input_file.npy] [target_file.npy] [output_file.h]")
    print("Default: \".data_sram0\" test_input.npy test_target.npy test_data.h\n")
    test_patterns_2_h_file()