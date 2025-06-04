import re
import numpy as np
from collections import defaultdict

def parse_neural_network_file(filename):
    """
    Parse a neural network data file and extract arrays for each time step and layer.
    Handles cases where numbers are split across lines.
    
    Returns:
        dict: Nested dictionary with structure:
              {time_step: {layer_name: {data_type: [array_values]}}}
    """
    data = defaultdict(lambda: defaultdict(dict))
    
    with open(filename, 'r') as file:
        content = file.read()
    
    # Split by time steps
    time_step_pattern = r'time step:\s*(\d+)'
    time_steps = re.split(time_step_pattern, content)
    
    # Remove empty first element if it exists
    if time_steps[0].strip() == '':
        time_steps = time_steps[1:]
    
    # Process each time step
    for i in range(0, len(time_steps), 2):
        if i + 1 >= len(time_steps):
            break
            
        time_step = int(time_steps[i])
        time_step_content = time_steps[i + 1]
        
        # Find all layer data patterns with improved regex
        # This pattern captures everything until the next Layer or time step
        layer_pattern = r'(Layer\d+)->(input|v_mem|output):\s*\n(.*?)(?=\n\s*Layer\d+->|\n\s*time step:|\Z)'
        matches = re.findall(layer_pattern, time_step_content, re.DOTALL)
        
        for layer_name, data_type, array_str in matches:
            # Enhanced parsing to handle numbers split across lines
            array_str = array_str.strip()
            if array_str:
                # Parse the array string with better handling of line breaks
                values = parse_array_string(array_str)
                data[time_step][layer_name][data_type] = np.array(values)
            else:
                data[time_step][layer_name][data_type] = np.array([])
    
    return data

def parse_array_string(array_str):
    """
    Parse an array string that may have numbers split across lines.
    Handles cases like:
    - "0.000, 1.000, 0.000, 0.0"
    - "0, 1.000 ..."
    - Numbers continuing on the next line
    
    Args:
        array_str (str): The raw string containing array data
        
    Returns:
        list: List of float values
    """
    values = []
    
    # Remove any trailing ellipsis (...) which might indicate continuation
    array_str = re.sub(r'\.{3,}\s*$', '', array_str)
    
    # Split into lines and process each line
    lines = array_str.split('\n')
    current_number = ""
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Remove any leading/trailing commas and whitespace
        line = line.strip(' ,')
        
        # Split by commas but be careful about incomplete numbers
        parts = line.split(',')
        
        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue
                
            # If we have a current_number from previous line, complete it
            if current_number:
                # Try to combine with current part
                combined = current_number + part
                try:
                    value = float(combined)
                    values.append(value)
                    current_number = ""
                    continue
                except ValueError:
                    # If combination fails, try to parse current_number alone
                    try:
                        value = float(current_number)
                        values.append(value)
                        current_number = ""
                    except ValueError:
                        pass
            
            # Try to parse the current part as a complete number
            try:
                value = float(part)
                values.append(value)
            except ValueError:
                # Check if this might be an incomplete number
                # (ends with decimal point or looks like start of number)
                if (part.endswith('.') or 
                    re.match(r'^-?\d*\.?\d*$', part) and 
                    part not in ['', '.', '-']):
                    current_number = part
                else:
                    # Try to extract any valid numbers from the part using regex
                    numbers = re.findall(r'-?\d*\.?\d+', part)
                    for num_str in numbers:
                        try:
                            value = float(num_str)
                            values.append(value)
                        except ValueError:
                            continue
    
    # Handle any remaining incomplete number
    if current_number:
        try:
            value = float(current_number)
            values.append(value)
        except ValueError:
            print(f"Warning: Could not parse incomplete number: '{current_number}'")
    
    return values

def compare_neural_network_data(file1, file2, tolerance=0.02):
    """
    Compare neural network data from two files and report differences.
    
    Args:
        file1 (str): Path to first file
        file2 (str): Path to second file
        tolerance (float): Maximum allowed difference before reporting error
    
    Returns:
        dict: Summary of comparison results
    """
    print(f"Parsing {file1}...")
    data1 = parse_neural_network_file(file1)
    
    print(f"Parsing {file2}...")
    data2 = parse_neural_network_file(file2)
    
    errors = []
    total_comparisons = 0
    total_errors = 0
    
    # Get all time steps from both files
    all_time_steps = set(data1.keys()) | set(data2.keys())
    
    for time_step in sorted(all_time_steps):
        if time_step not in data1:
            errors.append(f"Time step {time_step} missing in file1")
            continue
        if time_step not in data2:
            errors.append(f"Time step {time_step} missing in file2")
            continue
        
        # Get all layers from both files for this time step
        all_layers = set(data1[time_step].keys()) | set(data2[time_step].keys())
        
        for layer in sorted(all_layers):
            if layer not in data1[time_step]:
                errors.append(f"Layer {layer} missing in file1 at time step {time_step}")
                continue
            if layer not in data2[time_step]:
                errors.append(f"Layer {layer} missing in file2 at time step {time_step}")
                continue
            
            # Get all data types for this layer
            all_data_types = set(data1[time_step][layer].keys()) | set(data2[time_step][layer].keys())
            
            for data_type in sorted(all_data_types):
                if data_type not in data1[time_step][layer]:
                    errors.append(f"{layer}->{data_type} missing in file1 at time step {time_step}")
                    continue
                if data_type not in data2[time_step][layer]:
                    errors.append(f"{layer}->{data_type} missing in file2 at time step {time_step}")
                    continue
                
                array1 = data1[time_step][layer][data_type]
                array2 = data2[time_step][layer][data_type]
                
                # Compare array lengths
                if len(array1) != len(array2):
                    errors.append(f"Array length mismatch for {layer}->{data_type} at time step {time_step}: "
                                f"file1={len(array1)}, file2={len(array2)}")
                    print(f"length not same: file1={len(array1)},\tfile2={len(array2)}\t\tfor time step: {time_step}, layer: {layer}, data_type: {data_type}")
                    continue
                
                # Skip empty arrays
                if len(array1) == 0:
                    print("empty array")
                    continue
                
                # Compare values element by element
                total_comparisons += len(array1)
                diff = np.abs(array1 - array2)
                error_indices = np.where(diff > tolerance)[0]
                
                if len(error_indices) > 0:
                    total_errors += len(error_indices)
                    for idx in error_indices:
                        errors.append(f"Value mismatch at {layer}->{data_type}[{idx}] (time step {time_step}): "
                                    f"file1={array1[idx]:.6f}, file2={array2[idx]:.6f}, diff={diff[idx]:.6f}")
    
    return {
        'errors': errors,
        'total_errors': total_errors,
        'total_comparisons': total_comparisons,
        'tolerance': tolerance
    }

def test_parse_array_string():
    """
    Test function for the enhanced array string parsing.
    """
    test_cases = [
        "0.000, 1.000, 0.000, 0.0",
        "0, 1.000\n2.500, 3.0",
        "1.2\n3, 4.5, 5.",
        "1.0, 2.0, 3.\n0, 4.5",
        "0.000, 1.000, 0.000, 0.0\n0, 1.000",
        "-1.5, 2.3, -0.0\n1, -2.7"
    ]
    
    print("Testing array string parsing:")
    print("-" * 40)
    for i, test_str in enumerate(test_cases, 1):
        result = parse_array_string(test_str)
        print(f"Test {i}:")
        print(f"  Input: {repr(test_str)}")
        print(f"  Output: {result}")
        print()

def main():
    """
    Main function to run the comparison.
    Modify the file paths and tolerance as needed.
    """
    # Uncomment the next line to test the parsing function
    # test_parse_array_string()
    
    # Modify these paths to your actual files
    file1_path = "../snn_on_alif_e7/scripts/output_pat_3.txt"
    file2_path = "spk_mnist_mlp/output_pat_3.txt"
    tolerance = 0.02
    
    try:
        results = compare_neural_network_data(file1_path, file2_path, tolerance)
        
        print("\n" + "="*80)
        print("COMPARISON RESULTS")
        print("="*80)
        
        print(f"Total comparisons made: {results['total_comparisons']}")
        print(f"Total errors found: {results['total_errors']}")
        print(f"Tolerance used: {results['tolerance']}")
        
        if results['total_errors'] == 0:
            print("\n✅ SUCCESS: All values match within tolerance!")
        else:
            error_rate = (results['total_errors'] / results['total_comparisons']) * 100
            print(f"\n❌ ERRORS DETECTED: {error_rate:.2f}% of values exceed tolerance")
            
            print(f"\nFirst {min(20, len(results['errors']))} errors:")
            print("-" * 60)
            for i, error in enumerate(results['errors'][:20]):
                print(f"{i+1:2d}. {error}")
            
            if len(results['errors']) > 20:
                print(f"\n... and {len(results['errors']) - 20} more errors")
                
                # Show summary by layer and data type
                layer_errors = defaultdict(int)
                for error in results['errors']:
                    if 'Value mismatch at' in error:
                        layer_info = error.split('Value mismatch at ')[1].split(' (time step')[0]
                        layer_errors[layer_info] += 1
                
                if layer_errors:
                    print(f"\nError summary by layer/data type:")
                    print("-" * 40)
                    for layer_info, count in sorted(layer_errors.items()):
                        print(f"  {layer_info}: {count} errors")
    
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
        print("Please make sure both input files exist and update the file paths in the script.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
