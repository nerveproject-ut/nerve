#!/usr/bin/env python3
"""
Convert HDF5 event representations to PNG images for standard YOLO training.

Usage:
    python hdf5_to_png.py --input /path/to/hdf5_dataset --output /path/to/png_dataset
"""

import os
import h5py
import numpy as np
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm


def normalize_for_display(data):
    """
    Normalize multi-channel event data to 0-255 range for visualization.
    
    Args:
        data: numpy array of shape (C, H, W) or (H, W, C)
    
    Returns:
        Normalized array ready for saving as image
    """
    # Ensure channel-first format (C, H, W)
    if data.shape[-1] in [1, 3, 10, 20]:  # Likely channel-last
        data = np.transpose(data, (2, 0, 1))
    
    channels, height, width = data.shape
    
    # Normalize each channel independently
    normalized = np.zeros_like(data, dtype=np.float32)
    for c in range(channels):
        channel_data = data[c]
        min_val = channel_data.min()
        max_val = channel_data.max()
        
        if max_val > min_val:
            normalized[c] = (channel_data - min_val) / (max_val - min_val)
        else:
            normalized[c] = channel_data
    
    # Convert to uint8
    normalized = (normalized * 255).astype(np.uint8)
    
    # Handle different channel counts
    if channels == 1:
        # Single channel - save as grayscale
        return normalized[0]
    elif channels == 3:
        # RGB - transpose to (H, W, C) for OpenCV
        return np.transpose(normalized, (1, 2, 0))
    elif channels <= 10:
        # Multi-channel (e.g., 5, 10 bins) - create visualization
        # Option 1: Average across channels
        return np.mean(normalized, axis=0).astype(np.uint8)
        # Option 2: Use first 3 channels as RGB (uncomment if preferred)
        # if channels >= 3:
        #     return np.transpose(normalized[:3], (1, 2, 0))
    else:
        # Many channels (e.g., 20 for voxel_grid) - average
        return np.mean(normalized, axis=0).astype(np.uint8)


def convert_hdf5_to_png(hdf5_file, output_path, key='events', normalize=True, verbose=False):
    """
    Convert a single HDF5 file to PNG image(s).
    
    Args:
        hdf5_file: Path to HDF5 file
        output_path: Output PNG file path
        key: HDF5 dataset key ('events', '1mp', 'clip')
        normalize: Whether to normalize data
        verbose: Print progress
    
    Returns:
        True if successful, False otherwise
    """
    try:
        with h5py.File(hdf5_file, 'r') as f:
            # Try different possible keys
            data_key = None
            for possible_key in [key, 'events', '1mp', 'clip']:
                if possible_key in f:
                    data_key = possible_key
                    break
            
            if data_key is None:
                if verbose:
                    print(f"⚠️  Warning: No valid key found in {hdf5_file}")
                return False
            
            data = f[data_key][:]
            
            # Handle different data shapes
            if len(data.shape) == 4:
                # Shape: (T, C, H, W) - multiple frames
                # For single_frame mode, T should be 1
                if data.shape[0] == 1:
                    data = data[0]  # Remove time dimension
                else:
                    # Take first frame for multi-frame sequences
                    data = data[0]
                    if verbose:
                        print(f"ℹ️  Using first frame from {data.shape[0]} frames")
            
            # Now data should be (C, H, W)
            if len(data.shape) != 3:
                if verbose:
                    print(f"⚠️  Unexpected shape: {data.shape}")
                return False
            
            # Normalize and convert
            if normalize:
                img_data = normalize_for_display(data)
            else:
                # Assume data is already in correct range
                if data.shape[0] == 3:  # RGB
                    img_data = np.transpose(data, (1, 2, 0))
                else:
                    img_data = data[0]  # Use first channel
            
            # Save as PNG
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, img_data)
            
            return True
            
    except Exception as e:
        if verbose:
            print(f"❌ Error converting {hdf5_file}: {e}")
        return False


def convert_dataset(input_dir, output_dir, source_name, splits, key='events', verbose=False):
    """
    Convert entire PEGMA HDF5 dataset to PNG images.
    
    Args:
        input_dir: Input directory with HDF5 files
        output_dir: Output directory for PNG images
        source_name: Source name (e.g., 'prophesee')
        splits: List of splits to process
        key: HDF5 dataset key
        verbose: Print progress
    """
    stats = {'converted': 0, 'failed': 0}
    
    for split in splits:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Processing {split} split")
            print('='*60)
        
        # Source and destination directories
        source_dir = os.path.join(input_dir, split, 'data', source_name)
        dest_dir = os.path.join(output_dir, split, 'data', source_name)
        
        if not os.path.exists(source_dir):
            if verbose:
                print(f"⚠️  Warning: {source_dir} not found, skipping")
            continue
        
        # Find all HDF5 files
        h5_files = [f for f in os.listdir(source_dir) if f.endswith('.h5')]
        
        if len(h5_files) == 0:
            if verbose:
                print(f"⚠️  No HDF5 files found in {source_dir}")
            continue
        
        if verbose:
            print(f"Found {len(h5_files)} HDF5 files")
        
        # Convert each file
        for h5_file in tqdm(h5_files, desc=f"Converting {split}", disable=not verbose):
            input_path = os.path.join(source_dir, h5_file)
            
            # Create output filename (replace .h5 with .png)
            png_file = Path(h5_file).stem + '.png'
            output_path = os.path.join(dest_dir, png_file)
            
            # Convert
            success = convert_hdf5_to_png(input_path, output_path, key, verbose=False)
            
            if success:
                stats['converted'] += 1
            else:
                stats['failed'] += 1
        
        # Copy annotations directory
        source_ann = os.path.join(input_dir, split, 'annotations')
        dest_ann = os.path.join(output_dir, split, 'annotations')
        
        if os.path.exists(source_ann):
            import shutil
            os.makedirs(os.path.dirname(dest_ann), exist_ok=True)
            if os.path.exists(dest_ann):
                shutil.rmtree(dest_ann)
            shutil.copytree(source_ann, dest_ann)
            if verbose:
                print(f"✓ Copied annotations to {dest_ann}")
    
    return stats


def fix_coco_json_files(output_dir, source, splits, verbose=False):
    """
    Fix COCO JSON files to reference .png instead of .h5 files.
    
    Args:
        output_dir: Output directory containing coco_labels/
        source: Source name (prophesee, davis, etc.)
        splits: List of splits to fix
        verbose: Print progress
    """
    import json
    
    output_path = Path(output_dir)
    fixed_count = 0
    
    for split in splits:
        coco_file = output_path / 'coco_labels' / split / f'{source}.json'
        
        if not coco_file.exists():
            if verbose:
                print(f"  Skipping {split}: COCO JSON not found")
            continue
        
        if verbose:
            print(f"  Processing {split}...")
        
        # Load COCO JSON
        with open(coco_file, 'r') as f:
            coco_data = json.load(f)
        
        # Fix image filenames
        split_fixed = 0
        for img in coco_data['images']:
            if img['file_name'].endswith('.h5'):
                img['file_name'] = img['file_name'].replace('.h5', '.png')
                split_fixed += 1
        
        # Save fixed JSON
        with open(coco_file, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        if verbose:
            print(f"    ✓ Fixed {split_fixed} image references")
        
        fixed_count += split_fixed
    
    if verbose:
        print(f"\n  Total: Fixed {fixed_count} COCO annotations (.h5 → .png)")


def main():
    parser = argparse.ArgumentParser(
        description='Convert HDF5 event representations to PNG images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert entire dataset
  python hdf5_to_png.py -i /pegma_output -o /png_output -s prophesee

  # Convert with specific key
  python hdf5_to_png.py -i /input -o /output -s prophesee --key 1mp

  # Convert single file
  python hdf5_to_png.py --single-file input.h5 --output output.png
        """
    )
    
    parser.add_argument('--input', '-i', type=str,
                       help='Input directory with HDF5 files')
    parser.add_argument('--output', '-o', type=str,
                       help='Output directory for PNG images')
    parser.add_argument('--source', '-s', type=str, default='prophesee',
                       help='Source name (prophesee, davis, etc.)')
    parser.add_argument('--splits', type=str, nargs='+', default=['train', 'val'],
                       help='Dataset splits to process')
    parser.add_argument('--key', type=str, default='events',
                       help='HDF5 dataset key (events, 1mp, clip)')
    parser.add_argument('--single-file', type=str,
                       help='Convert a single HDF5 file instead of directory')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Single file mode
    if args.single_file:
        if not args.output:
            args.output = Path(args.single_file).stem + '.png'
        
        if args.verbose:
            print(f"Converting single file: {args.single_file} → {args.output}")
        
        success = convert_hdf5_to_png(args.single_file, args.output, args.key, verbose=args.verbose)
        
        if success:
            print(f"✓ Converted to {args.output}")
            return 0
        else:
            print(f"❌ Conversion failed")
            return 1
    
    # Directory mode
    if not args.input or not args.output:
        parser.print_help()
        return 1
    
    if args.verbose:
        print("="*60)
        print("HDF5 to PNG Converter")
        print("="*60)
        print(f"Input: {args.input}")
        print(f"Output: {args.output}")
        print(f"Source: {args.source}")
        print(f"Splits: {args.splits}")
        print(f"HDF5 Key: {args.key}")
    
    # Convert dataset
    stats = convert_dataset(
        args.input,
        args.output,
        args.source,
        args.splits,
        args.key,
        args.verbose
    )
    
    if args.verbose:
        print("\n" + "="*60)
        print("Conversion Summary")
        print("="*60)
        print(f"✓ Converted: {stats['converted']} files")
        if stats['failed'] > 0:
            print(f"❌ Failed: {stats['failed']} files")
        print("="*60)
    
    # Automatically fix COCO JSON files to reference .png instead of .h5
    if stats['converted'] > 0:
        if args.verbose:
            print("\n" + "="*60)
            print("Fixing COCO JSON Annotations")
            print("="*60)
        
        fix_coco_json_files(args.output, args.source, args.splits, args.verbose)
    
    return 0


if __name__ == '__main__':
    exit(main())

