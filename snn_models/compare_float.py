#!/usr/bin/env python3
"""
Script to compare floating point values from two text files.
Warns when differences exceed a specified tolerance.
"""

def read_float_file(filename):
    """Read floating point values from a comma-separated text file."""
    try:
        with open(filename, 'r') as file:
            content = file.read().strip()
            # Split by comma and convert to float, filtering out empty strings
            values = [float(x.strip()) for x in content.split(',') if x.strip()]
            return values
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except ValueError as e:
        print(f"Error parsing values in '{filename}': {e}")
        return None

def compare_files(file1, file2, tolerance=0.02):
    """Compare floating point values from two files with given tolerance."""
    
    # Read values from both files
    values1 = read_float_file(file1)
    values2 = read_float_file(file2)
    
    if values1 is None or values2 is None:
        return
    
    # Check if files have same number of elements
    if len(values1) != len(values2):
        print(f"Warning: Files have different lengths!")
        print(f"  {file1}: {len(values1)} elements")
        print(f"  {file2}: {len(values2)} elements")
        print("Comparing only the first {} elements.\n".format(min(len(values1), len(values2))))
    
    # Compare elements
    differences_found = 0
    min_length = min(len(values1), len(values2))
    
    print(f"Comparing {min_length} elements with tolerance = {tolerance}")
    print("-" * 50)
    
    for i in range(min_length):
        diff = abs(values1[i] - values2[i])
        if diff > tolerance:
            differences_found += 1
            print(f"WARNING: Element {i} exceeds tolerance!")
            print(f"  File 1: {values1[i]:.5f}")
            print(f"  File 2: {values2[i]:.5f}")
            print(f"  Difference: {diff:.5f} (tolerance: {tolerance})")
            print()
    
    # Summary
    print("-" * 50)
    if differences_found == 0:
        print("✓ All elements are within tolerance.")
    else:
        print(f"⚠ Found {differences_found} element(s) exceeding tolerance of {tolerance}")
    
    print(f"Total elements compared: {min_length}")

def main():
    """Main function to run the comparison."""
    
    # File names - modify these as needed
    file1 = "file1.txt"
    file2 = "file2.txt"
    tolerance = 0.1
    
    print("Float File Comparison Tool")
    print("=" * 30)
    print(f"File 1: {file1}")
    print(f"File 2: {file2}")
    print(f"Tolerance: {tolerance}")
    print()
    
    compare_files(file1, file2, tolerance)

if __name__ == "__main__":
    main()