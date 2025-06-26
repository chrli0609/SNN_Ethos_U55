#!/usr/bin/env python3
"""
Script to copy and manipulate neural network layer files.

Usage:
    python script.py <model> <layer_to_copy> <layer_to_push_back> <num_copies_to_make> [--modify-percent <percent>]

Example:
    python script.py model_1 2 5 3 --modify-percent 10
"""

import argparse
import os
import shutil
import numpy as np
import glob
import re
from pathlib import Path


def parse_layer_number(filename):
    """Extract layer number from filename like 'fc_lif_layer_X_weights.npy'"""
    match = re.search(r'fc_lif_layer_(\d+)_', filename)
    if match:
        return int(match.group(1))
    return None


def create_new_filename(filename, new_layer_num):
    """Create new filename with updated layer number"""
    return re.sub(r'fc_lif_layer_(\d+)_', f'fc_lif_layer_{new_layer_num}_', filename)


def modify_array_values(file_path, percent_increase):
    """Load numpy array, increase values by percentage, and save back"""
    try:
        arr = np.load(file_path)
        #modified_arr = arr * (1 + percent_increase / 100.0)
        modified_arr = arr + np.abs(arr) * (percent_increase / 100.0)

        np.save(file_path, modified_arr)
        print(f"  Modified values in {file_path} by {percent_increase}%")
    except Exception as e:
        print(f"  Error modifying {file_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Copy and manipulate neural network layer files')
    parser.add_argument('model', type=str, help='Model directory name (e.g., model_1)')
    parser.add_argument('layer_to_copy', type=int, help='Layer number to copy')
    parser.add_argument('layer_to_push_back', type=int, help='Layer number to push back')
    parser.add_argument('num_copies_to_make', type=int, help='Number of copies to make')
    parser.add_argument('--modify-percent', type=float, default=0, 
                       help='Percentage to increase values in copied files (default: 0)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.num_copies_to_make <= 0:
        print("Error: num_copies_to_make must be positive")
        return
    
    if args.layer_to_copy >= args.layer_to_push_back:
        print("Error: layer_to_copy should be less than layer_to_push_back")
        return
    
    # Set up paths
    model_dir = Path(args.model)
    params_dir = model_dir / 'model_params'
    
    if not params_dir.exists():
        print(f"Error: Directory {params_dir} does not exist")
        return
    
    print(f"Working in directory: {params_dir}")
    print(f"Copying layer {args.layer_to_copy}, pushing back layer {args.layer_to_push_back}")
    print(f"Making {args.num_copies_to_make} copies")
    
    # Find all .npy files in the directory
    npy_files = list(params_dir.glob('*.npy'))
    
    if not npy_files:
        print("Error: No .npy files found in the directory")
        return
    
    # Step 1: Rename the layer_to_push_back files
    print(f"\nStep 1: Renaming layer {args.layer_to_push_back} files...")
    new_push_back_layer = args.layer_to_push_back + args.num_copies_to_make
    
    files_to_rename = []
    for file_path in npy_files:
        layer_num = parse_layer_number(file_path.name)
        if layer_num == args.layer_to_push_back:
            files_to_rename.append(file_path)
    
    if not files_to_rename:
        print(f"Warning: No files found for layer {args.layer_to_push_back}")
    else:
        for file_path in files_to_rename:
            new_name = create_new_filename(file_path.name, new_push_back_layer)
            new_path = file_path.parent / new_name
            
            print(f"  Renaming: {file_path.name} -> {new_name}")
            file_path.rename(new_path)
    
    # Step 2: Copy the layer_to_copy files
    print(f"\nStep 2: Copying layer {args.layer_to_copy} files...")
    
    # Refresh file list after renaming
    npy_files = list(params_dir.glob('*.npy'))
    
    files_to_copy = []
    for file_path in npy_files:
        layer_num = parse_layer_number(file_path.name)
        if layer_num == args.layer_to_copy:
            files_to_copy.append(file_path)
    
    if not files_to_copy:
        print(f"Error: No files found for layer {args.layer_to_copy}")
        return
    
    copied_files = []
    for i in range(args.num_copies_to_make):
        new_layer_num = args.layer_to_copy + 1 + i
        print(f"  Creating copies for layer {new_layer_num}...")
        
        for file_path in files_to_copy:
            new_name = create_new_filename(file_path.name, new_layer_num)
            new_path = file_path.parent / new_name
            
            print(f"    Copying: {file_path.name} -> {new_name}")
            shutil.copy2(file_path, new_path)
            copied_files.append(new_path)
    
    # Step 3: Modify copied files if requested
    if args.modify_percent != 0:
        print(f"\nStep 3: Modifying copied files by {args.modify_percent}%...")
        for file_path in copied_files:
            modify_array_values(file_path, args.modify_percent)
    
    print(f"\nOperation completed successfully!")
    print(f"- Renamed {len(files_to_rename)} files from layer {args.layer_to_push_back} to layer {new_push_back_layer}")
    print(f"- Created {len(copied_files)} new files")
    if args.modify_percent != 0:
        print(f"- Modified copied files by {args.modify_percent}%")


if __name__ == '__main__':
    main()