#!/usr/bin/env python3
"""
Convert PEGMA COCO-format dataset to standard YOLO format (images + txt labels).

Usage:
    python convert_to_yolo.py --input /path/to/pegma_output --output /path/to/yolo_dataset
"""

import os
import json
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm


def convert_coco_to_yolo(coco_file, output_dir, image_dir, filter_classes=None, verbose=False):
    """
    Convert COCO annotations to YOLO format.
    
    Args:
        coco_file: Path to COCO JSON file
        output_dir: Output directory for YOLO labels
        image_dir: Directory containing images
        filter_classes: List of class names to include (None = include all)
        verbose: Print progress
    """
    with open(coco_file, 'r') as f:
        coco = json.load(f)
    
    # Build category mapping
    all_categories = {cat['id']: cat['name'] for cat in coco['categories']}
    
    # Filter categories if specified
    if filter_classes:
        # Keep only specified classes
        categories = {cat_id: name for cat_id, name in all_categories.items() 
                     if name in filter_classes}
        if verbose:
            print(f"Filtering to classes: {filter_classes}")
            print(f"Found {len(categories)} matching classes: {list(categories.values())}")
    else:
        categories = all_categories
    
    # Create mapping from COCO category IDs to YOLO class indices (0-based)
    cat_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(sorted(categories.keys()))}
    
    if verbose:
        print(f"Output classes ({len(categories)}): {list(categories.values())}")
    
    # Build image id to filename mapping
    images = {img['id']: img['file_name'] for img in coco['images']}
    
    # Build annotations by image
    annotations_by_image = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    # Get image dimensions
    image_dims = {img['id']: (img['width'], img['height']) for img in coco['images']}
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert each image's annotations
    converted = 0
    for img_id, anns in tqdm(annotations_by_image.items(), desc="Converting labels", disable=not verbose):
        img_filename = images.get(img_id)
        if not img_filename:
            continue
        
        # Get label filename (same as image but .txt)
        label_filename = Path(img_filename).stem + '.txt'
        label_path = os.path.join(output_dir, label_filename)
        
        # Get image dimensions
        img_w, img_h = image_dims.get(img_id, (640, 640))
        
        # Convert annotations
        yolo_lines = []
        for ann in anns:
            cat_id = ann['category_id']
            
            # Skip if this category is not in our filtered list
            if cat_id not in cat_id_to_idx:
                continue
            
            bbox = ann['bbox']  # [x, y, width, height]
            
            # Convert to YOLO format (normalized)
            x_center = (bbox[0] + bbox[2] / 2) / img_w
            y_center = (bbox[1] + bbox[3] / 2) / img_h
            width = bbox[2] / img_w
            height = bbox[3] / img_h
            
            # Get class index (remapped to 0-based for filtered classes)
            class_idx = cat_id_to_idx[cat_id]
            
            # Create YOLO line
            yolo_lines.append(f"{class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        # Write label file
        with open(label_path, 'w') as f:
            f.write('\n'.join(yolo_lines))
        
        converted += 1
    
    if verbose:
        print(f"Converted {converted} label files")
    
    return categories


def organize_yolo_dataset(pegma_dir, output_dir, source_name, splits, filter_classes=None, verbose=False):
    """
    Organize PEGMA output into standard YOLO structure.
    
    Args:
        pegma_dir: PEGMA output directory
        output_dir: Output YOLO dataset directory
        source_name: Source name (e.g., 'prophesee')
        splits: List of splits (train, val, test)
        filter_classes: List of class names to include (None = include all)
        verbose: Print progress
    """
    os.makedirs(output_dir, exist_ok=True)
    
    categories = None
    
    for split in splits:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Processing {split} split")
            print('='*60)
        
        # Create directories
        images_dir = os.path.join(output_dir, 'images', split)
        labels_dir = os.path.join(output_dir, 'labels', split)
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        
        # Source directories
        source_images = os.path.join(pegma_dir, split, 'data', source_name)
        source_annotations = os.path.join(pegma_dir, split, 'annotations', f'{source_name}.json')
        
        if not os.path.exists(source_images):
            if verbose:
                print(f"⚠️  Warning: {source_images} not found, skipping")
            continue
        
        if not os.path.exists(source_annotations):
            if verbose:
                print(f"⚠️  Warning: {source_annotations} not found, skipping")
            continue
        
        # Copy images
        if verbose:
            print(f"\nCopying images from {source_images}")
        
        image_files = [f for f in os.listdir(source_images) if f.endswith(('.png', '.jpg', '.jpeg'))]
        for img_file in tqdm(image_files, desc="Copying images", disable=not verbose):
            src = os.path.join(source_images, img_file)
            dst = os.path.join(images_dir, img_file)
            shutil.copy2(src, dst)
        
        # Convert labels
        if verbose:
            print(f"\nConverting labels from {source_annotations}")
        
        split_categories = convert_coco_to_yolo(
            source_annotations,
            labels_dir,
            source_images,
            filter_classes=filter_classes,
            verbose=verbose
        )
        
        if categories is None:
            categories = split_categories
    
    return categories


def create_yaml_config(output_dir, categories, verbose=False):
    """Create YOLO dataset.yaml configuration file."""
    
    yaml_content = f"""# YOLO Dataset Configuration
# Generated by PEGMA convert_to_yolo.py

path: {os.path.abspath(output_dir)}  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')
test: images/test  # test images (optional)

# Classes
nc: {len(categories)}  # number of classes
names: {list(categories.values())}  # class names
"""
    
    yaml_path = os.path.join(output_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    if verbose:
        print(f"\n✓ Created {yaml_path}")
        print(yaml_content)
    
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description='Convert PEGMA output to standard YOLO format')
    
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Input PEGMA dataset directory')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output YOLO dataset directory')
    parser.add_argument('--source', '-s', type=str, default='prophesee',
                       help='Source name (prophesee, davis, etc.)')
    parser.add_argument('--splits', type=str, nargs='+', default=['train', 'val', 'test'],
                       help='Dataset splits to process')
    parser.add_argument('--classes', '-c', type=str, nargs='+', default=None,
                       help='Filter to specific class names (e.g., --classes person car)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        print("="*60)
        print("PEGMA to YOLO Converter")
        print("="*60)
        print(f"Input: {args.input}")
        print(f"Output: {args.output}")
        print(f"Source: {args.source}")
        print(f"Splits: {args.splits}")
        if args.classes:
            print(f"Classes filter: {args.classes}")
    
    # Convert dataset
    categories = organize_yolo_dataset(
        args.input,
        args.output,
        args.source,
        args.splits,
        filter_classes=args.classes,
        verbose=args.verbose
    )
    
    if categories:
        # Create YAML config
        create_yaml_config(args.output, categories, args.verbose)
        
        if args.verbose:
            print("\n" + "="*60)
            print("✓ Conversion complete!")
            print("="*60)
            print(f"\nYour YOLO dataset is ready at: {args.output}")
            print("\nTo train with YOLOv8:")
            print(f"  yolo detect train data={os.path.join(args.output, 'dataset.yaml')} model=yolov8n.pt epochs=100")
    else:
        print("❌ No data was converted. Check your input paths.")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

