#!/usr/bin/env python3
"""
Fix PEGMA dataset structure for YOLOv8 training.

YOLOv8 expects:
- images/ directory with actual images (NOT a symlink)
- labels/ directory with .txt files
- YOLOv8 finds labels by replacing 'images' with 'labels' in the path

This script:
1. Removes the symlink and creates a real images/ directory
2. Moves images from data/{source}/ to images/{source}/
3. Ensures labels are in labels/{source}/
4. Updates data.yaml accordingly

Usage:
    python fix_yolov8_structure.py /path/to/dataset --source davis
"""

import os
import shutil
import argparse
from pathlib import Path


def fix_dataset_structure(dataset_path, source='davis', verbose=True):
    """Fix a PEGMA dataset structure for YOLOv8 training.
    
    If source is 'all', automatically detect and fix all sources (davis, ti_radar, etc.)
    """
    dataset_path = Path(dataset_path).resolve()
    
    # If source is 'all', detect all available sources
    if source == 'all':
        sources = set()
        for split in ['train', 'val', 'test']:
            split_dir = dataset_path / split
            if split_dir.exists():
                # Check data/ directory for sources
                data_dir = split_dir / 'data'
                if data_dir.exists():
                    for d in data_dir.iterdir():
                        if d.is_dir():
                            sources.add(d.name)
                # Also check images/ directory
                images_dir = split_dir / 'images'
                if images_dir.exists() and not images_dir.is_symlink():
                    for d in images_dir.iterdir():
                        if d.is_dir():
                            sources.add(d.name)
        
        if not sources:
            print("❌ No sources found in dataset!")
            return False
        
        if verbose:
            print(f"Auto-detected sources: {sorted(sources)}")
        
        # Fix each source
        all_success = True
        for src in sorted(sources):
            if verbose:
                print(f"\n{'='*60}\nProcessing source: {src}\n{'='*60}")
            success = fix_dataset_structure(dataset_path, source=src, verbose=verbose)
            all_success = all_success and success
        
        return all_success
    
    if verbose:
        print("=" * 60)
        print("Fixing PEGMA dataset structure for YOLOv8")
        print("=" * 60)
        print(f"Dataset: {dataset_path}")
        print(f"Source: {source}")
    
    splits = []
    for split in ['train', 'val', 'test']:
        split_dir = dataset_path / split
        if split_dir.exists():
            # Check if data/source or images/source exists
            if (split_dir / 'data' / source).exists() or (split_dir / 'images' / source).exists():
                splits.append(split)
    
    if not splits:
        print("❌ No valid splits found!")
        return False
    
    if verbose:
        print(f"Splits found: {splits}")
    
    # Process each split
    for split in splits:
        split_dir = dataset_path / split
        if verbose:
            print(f"\n--- Processing {split} ---")
        
        data_dir = split_dir / 'data' / source
        images_link = split_dir / 'images'
        images_dir = split_dir / 'images' / source
        labels_dir = split_dir / 'labels' / source
        
        # Step 1: Handle images/ symlink or directory
        if images_link.is_symlink():
            # Remove the symlink
            images_link.unlink()
            if verbose:
                print(f"  Removed symlink: {split}/images")
        
        # Step 2: Create images directory structure
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 3: Move or copy images from data/ to images/
        if data_dir.exists() and data_dir.is_dir():
            image_files = list(data_dir.glob('*.png')) + list(data_dir.glob('*.jpg')) + list(data_dir.glob('*.jpeg'))
            if image_files:
                if verbose:
                    print(f"  Moving {len(image_files)} images from data/{source}/ to images/{source}/")
                for img_file in image_files:
                    dest = images_dir / img_file.name
                    if not dest.exists():
                        shutil.move(str(img_file), str(dest))
                
                # Remove empty data/source directory
                try:
                    data_dir.rmdir()
                except OSError:
                    pass  # Directory not empty or doesn't exist
                
                # Try to remove data/ if empty
                try:
                    (split_dir / 'data').rmdir()
                except OSError:
                    pass
        elif not list(images_dir.glob('*')):
            if verbose:
                print(f"  ⚠️  No images found in {data_dir} or {images_dir}")
        
        # Step 4: Clean up cache files
        for cache_file in split_dir.rglob('*.cache'):
            cache_file.unlink()
            if verbose:
                print(f"  Removed cache: {cache_file.relative_to(dataset_path)}")
        
        # Step 5: Verify labels exist
        if labels_dir.exists():
            label_count = len(list(labels_dir.glob('*.txt')))
            non_empty = sum(1 for f in labels_dir.glob('*.txt') if f.stat().st_size > 0)
            if verbose:
                print(f"  Labels: {label_count} files ({non_empty} with objects)")
        else:
            if verbose:
                print(f"  ⚠️  No labels directory at {split}/labels/{source}")
    
    # Update data.yaml
    yaml_path = dataset_path / 'data.yaml'
    if yaml_path.exists():
        with open(yaml_path, 'r') as f:
            content = f.read()
        
        # Ensure paths use images/ not data/
        new_content = content.replace('/data/', '/images/')
        
        with open(yaml_path, 'w') as f:
            f.write(new_content)
        
        if verbose:
            print(f"\n✓ Updated data.yaml")
            print("\nNew data.yaml content:")
            print("-" * 40)
            print(new_content)
            print("-" * 40)
    
    if verbose:
        print("\n" + "=" * 60)
        print("✓ Dataset structure fixed!")
        print("=" * 60)
        print("\nExpected structure:")
        print(f"  {dataset_path}/")
        print(f"  ├── train/")
        print(f"  │   ├── images/{source}/  <- actual images")
        print(f"  │   └── labels/{source}/  <- YOLO txt labels")
        print(f"  ├── val/")
        print(f"  │   ├── images/{source}/")
        print(f"  │   └── labels/{source}/")
        print(f"  └── data.yaml")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Fix PEGMA dataset structure for YOLOv8')
    parser.add_argument('dataset', type=str, help='Path to PEGMA dataset')
    parser.add_argument('--source', '-s', type=str, default='all',
                       help='Source name (davis, ti_radar, etc.) or "all" to auto-detect and fix all sources')
    parser.add_argument('--dry-run', '-n', action='store_true',
                       help='Show what would be done without making changes')
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("DRY RUN - No changes will be made")
    
    success = fix_dataset_structure(args.dataset, args.source)
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())

