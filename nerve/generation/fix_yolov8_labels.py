#!/usr/bin/env python3
"""
Fix existing PEGMA datasets for YOLOv8 training.

This script:
1. Generates YOLO txt labels from COCO JSON annotations
2. Creates 'images' symlink for YOLOv8 compatibility
3. Updates data.yaml to use correct paths

Usage:
    python fix_yolov8_labels.py /path/to/dataset --source davis
"""

import os
import json
import argparse
from pathlib import Path


def generate_yolo_txt_labels(coco_json_path, output_labels_dir, verbose=False):
    """Convert COCO JSON to YOLO txt format."""
    coco_path = Path(coco_json_path)
    labels_dir = Path(output_labels_dir)
    
    if not coco_path.exists():
        print(f"  ⚠️  COCO file not found: {coco_path}")
        return 0
    
    with open(coco_path, 'r') as f:
        coco = json.load(f)
    
    categories = {cat['id']: cat for cat in coco.get('categories', [])}
    sorted_cat_ids = sorted(categories.keys())
    coco_id_to_yolo_idx = {coco_id: idx for idx, coco_id in enumerate(sorted_cat_ids)}
    
    images = {img['id']: img for img in coco.get('images', [])}
    
    anns_by_image = {}
    for ann in coco.get('annotations', []):
        img_id = ann['image_id']
        if img_id not in anns_by_image:
            anns_by_image[img_id] = []
        anns_by_image[img_id].append(ann)
    
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    created = 0
    for img_id, img_info in images.items():
        filename = img_info['file_name']
        img_w = img_info['width']
        img_h = img_info['height']
        
        txt_filename = Path(filename).stem + '.txt'
        txt_path = labels_dir / txt_filename
        
        anns = anns_by_image.get(img_id, [])
        
        yolo_lines = []
        for ann in anns:
            cat_id = ann['category_id']
            if cat_id not in coco_id_to_yolo_idx:
                continue
            
            bbox = ann['bbox']
            x, y, w, h = bbox
            
            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            width = w / img_w
            height = h / img_h
            
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            width = max(0, min(1, width))
            height = max(0, min(1, height))
            
            class_idx = coco_id_to_yolo_idx[cat_id]
            yolo_lines.append(f"{class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        with open(txt_path, 'w') as f:
            f.write('\n'.join(yolo_lines))
        
        if yolo_lines:
            created += 1
    
    if verbose:
        print(f"  Created {created} label files with objects")
    
    return created


def fix_dataset(dataset_path, source='davis', verbose=True):
    """Fix a PEGMA dataset for YOLOv8 training."""
    dataset_path = Path(dataset_path).resolve()
    
    if verbose:
        print("=" * 60)
        print("Fixing PEGMA dataset for YOLOv8")
        print("=" * 60)
        print(f"Dataset: {dataset_path}")
        print(f"Source: {source}")
    
    splits = []
    for split in ['train', 'val', 'test']:
        if (dataset_path / split / 'data' / source).exists():
            splits.append(split)
    
    if not splits:
        print("❌ No splits found!")
        return False
    
    if verbose:
        print(f"Splits found: {splits}")
    
    # Process each split
    total_labels = 0
    for split in splits:
        if verbose:
            print(f"\n--- Processing {split} ---")
        
        # Generate YOLO txt labels
        coco_path = dataset_path / split / 'annotations' / f'{source}.json'
        labels_dir = dataset_path / split / 'labels' / source
        
        if coco_path.exists():
            count = generate_yolo_txt_labels(coco_path, labels_dir, verbose=verbose)
            total_labels += count
        else:
            print(f"  ⚠️  No annotations found: {coco_path}")
        
        # Create 'images' symlink
        images_link = dataset_path / split / 'images'
        if not images_link.exists():
            try:
                images_link.symlink_to('data')
                if verbose:
                    print(f"  Created symlink: {split}/images -> data")
            except OSError as e:
                print(f"  ⚠️  Could not create symlink: {e}")
    
    # Update data.yaml
    yaml_path = dataset_path / 'data.yaml'
    if yaml_path.exists():
        with open(yaml_path, 'r') as f:
            content = f.read()
        
        # Replace data/ with images/ for YOLOv8 compatibility
        new_content = content.replace('/data/', '/images/')
        
        with open(yaml_path, 'w') as f:
            f.write(new_content)
        
        if verbose:
            print(f"\n✓ Updated data.yaml to use images/ paths")
    
    if verbose:
        print("\n" + "=" * 60)
        print(f"✓ Fixed! Created {total_labels} label files total")
        print("=" * 60)
        print(f"\nYour dataset is now ready for YOLOv8 training.")
        print(f"Use: python train.py -f your_yolov8_experiment.py")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Fix PEGMA dataset for YOLOv8 training')
    parser.add_argument('dataset', type=str, help='Path to PEGMA dataset')
    parser.add_argument('--source', '-s', type=str, default='davis',
                       help='Source name (prophesee, davis, etc.)')
    
    args = parser.parse_args()
    
    success = fix_dataset(args.dataset, args.source)
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())

