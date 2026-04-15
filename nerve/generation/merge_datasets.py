"""
Merge partial datasets created by parallel workers into a single dataset.
This script handles index renumbering and file copying.

@Author  :   Pietro Martinello (modified for parallel processing)
@Contact :   martin66@imec.be / pietromartinello.dev@gmail.com
"""

import os
import shutil
import json
import argparse
from tqdm import tqdm


def get_arguments():
    """Parse all the arguments provided from the CLI."""
    parser = argparse.ArgumentParser(description='Merge partial datasets into a single dataset')
    
    parser.add_argument("--source", "-s", type=str, required=True, 
                        help="Path of the source dataset to merge from.")
    parser.add_argument("--target", "-t", type=str, required=True, 
                        help="Path of the target dataset to merge into.")
    parser.add_argument("--add", "-a", action='store_true', 
                        help="Add to existing dataset.")
    parser.add_argument("--clean", action='store_true', 
                        help="Create new dataset (overwrite if exists).")
    
    return parser.parse_args()


def merge_annotations(source_file, target_file, image_id_offset, annotation_id_offset):
    """
    Merge annotations from source into target, renumbering IDs.
    
    Returns:
        tuple: (new_image_id_offset, new_annotation_id_offset, num_images_added)
    """
    with open(source_file, 'r') as f:
        source_data = json.load(f)
    
    if os.path.isfile(target_file):
        with open(target_file, 'r') as f:
            target_data = json.load(f)
    else:
        # Create new target structure from source template
        target_data = {
            "info": source_data.get("info", {}),
            "licenses": source_data.get("licenses", []),
            "images": [],
            "annotations": [],
            "categories": source_data.get("categories", [])
        }
    
    # Track old ID to new ID mapping for images
    image_id_map = {}
    
    # Add images with renumbered IDs
    for img in source_data['images']:
        old_id = img['id']
        new_id = image_id_offset
        image_id_map[old_id] = new_id
        
        new_img = img.copy()
        new_img['id'] = new_id
        target_data['images'].append(new_img)
        
        image_id_offset += 1
    
    # Add annotations with renumbered IDs
    for ann in source_data['annotations']:
        new_ann = ann.copy()
        new_ann['id'] = annotation_id_offset
        new_ann['image_id'] = image_id_map[ann['image_id']]
        target_data['annotations'].append(new_ann)
        
        annotation_id_offset += 1
    
    # Write merged data
    with open(target_file, 'w') as f:
        json.dump(target_data, f)
    
    num_images = len(source_data['images'])
    return image_id_offset, annotation_id_offset, num_images, image_id_map


def copy_data_files(source_data_dir, target_data_dir, data_type, image_id_map):
    """
    Copy data files from source to target, updating filenames if necessary.
    """
    source_type_dir = os.path.join(source_data_dir, data_type)
    target_type_dir = os.path.join(target_data_dir, data_type)
    
    if not os.path.isdir(source_type_dir):
        return
    
    os.makedirs(target_type_dir, exist_ok=True)
    
    # Copy all files
    files = [f for f in os.listdir(source_type_dir) if os.path.isfile(os.path.join(source_type_dir, f))]
    
    for filename in tqdm(files, desc=f"Copying {data_type} files", leave=False):
        source_path = os.path.join(source_type_dir, filename)
        target_path = os.path.join(target_type_dir, filename)
        shutil.copy2(source_path, target_path)


def main():
    args = get_arguments()
    source_dir = args.source
    target_dir = args.target
    is_adding = args.add
    clean = args.clean
    
    assert os.path.isdir(source_dir), f"Source directory does not exist: {source_dir}"
    assert not (clean and is_adding), "Cannot use both --clean and --add flags"
    
    # Setup target directory
    if clean and os.path.isdir(target_dir):
        shutil.rmtree(target_dir)
    
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)
        os.makedirs(os.path.join(target_dir, 'annotations'))
        os.makedirs(os.path.join(target_dir, 'data'))
        is_adding = False  # First dataset, not adding
    
    # Determine starting offsets
    source_annotations_dir = os.path.join(source_dir, 'annotations')
    target_annotations_dir = os.path.join(target_dir, 'annotations')
    source_data_dir = os.path.join(source_dir, 'data')
    target_data_dir = os.path.join(target_dir, 'data')
    
    # Get list of data types from source annotations
    annotation_files = [f for f in os.listdir(source_annotations_dir) if f.endswith('.json')]
    
    if not annotation_files:
        print("No annotation files found in source dataset")
        return
    
    print(f"Merging {len(annotation_files)} data streams...")
    
    for ann_file in annotation_files:
        data_type = ann_file.replace('.json', '')
        source_ann_path = os.path.join(source_annotations_dir, ann_file)
        target_ann_path = os.path.join(target_annotations_dir, ann_file)
        
        print(f"\nProcessing {data_type}...")
        
        # Determine offsets
        if is_adding and os.path.isfile(target_ann_path):
            with open(target_ann_path, 'r') as f:
                target_data = json.load(f)
            image_id_offset = target_data['images'][-1]['id'] + 1 if target_data['images'] else 0
            annotation_id_offset = target_data['annotations'][-1]['id'] + 1 if target_data['annotations'] else 1
        else:
            image_id_offset = 0
            annotation_id_offset = 1
        
        # Merge annotations
        _, _, num_images, image_id_map = merge_annotations(
            source_ann_path, 
            target_ann_path, 
            image_id_offset, 
            annotation_id_offset
        )
        
        print(f"  Added {num_images} images")
        
        # Copy data files
        copy_data_files(source_data_dir, target_data_dir, data_type, image_id_map)
    
    print("\n✓ Merge completed successfully")


if __name__ == '__main__':
    main()















