#!/usr/bin/env python3
"""
Fix COCO JSON annotations to reference PNG files instead of HDF5 files.
Use this after converting HDF5 event representations to PNG.
"""

import json
import argparse
import os
from pathlib import Path


def fix_coco_json(coco_file, output_file=None, verbose=False):
    """
    Replace .h5 extensions with .png in COCO JSON file.
    
    Args:
        coco_file: Path to COCO JSON file
        output_file: Path to save fixed JSON (default: overwrite input)
        verbose: Print progress
    """
    if verbose:
        print(f"Loading COCO JSON from: {coco_file}")
    
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)
    
    # Fix image filenames
    fixed_count = 0
    for img in coco_data['images']:
        if img['file_name'].endswith('.h5'):
            img['file_name'] = img['file_name'].replace('.h5', '.png')
            fixed_count += 1
    
    if verbose:
        print(f"Fixed {fixed_count} image references (.h5 → .png)")
    
    # Save to output file
    if output_file is None:
        output_file = coco_file
    
    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    if verbose:
        print(f"Saved fixed JSON to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Fix COCO JSON to reference PNG files instead of HDF5'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to PEGMA output directory (containing coco_labels/)'
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'val', 'test'],
        help='Dataset splits to fix (default: train val test)'
    )
    parser.add_argument(
        '--source',
        type=str,
        default='prophesee',
        help='Dataset source name (default: prophesee)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return 1
    
    # Fix each split
    for split in args.splits:
        coco_file = input_dir / 'coco_labels' / split / f'{args.source}.json'
        
        if not coco_file.exists():
            if args.verbose:
                print(f"Skipping {split}: {coco_file} not found")
            continue
        
        if args.verbose:
            print(f"\n{'='*60}")
            print(f"Processing {split} split")
            print(f"{'='*60}")
        
        fix_coco_json(str(coco_file), verbose=args.verbose)
    
    print(f"\n✅ Done! COCO JSON files now reference .png instead of .h5")
    print(f"\nYou can now train YOLOX with:")
    print(f"  python yoloX/tools/train.py -f custom_exp/exp__vtei.py -b 16 --fp16 -expn vtei_exp --cache")


if __name__ == '__main__':
    main()

