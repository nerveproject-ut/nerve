import os
import numpy as np
import json
from importlib.resources import files


def GetCategories() -> list:
    categories_file = files("nerve.data").joinpath("categories.json")
    categories = json.loads(categories_file.read_text(encoding="utf-8"))
    return categories


def GetCategoryNameToIdMap()->dict:
    """
    Returns a mapping from category name to COCO category ID.
    E.g., {'person': 1, 'bicycle': 2, 'car': 3, ...}
    """
    categories = GetCategories()
    return {cat['name']: cat['id'] for cat in categories}


def GetCategoryIdToNameMap()->dict:
    """
    Returns a mapping from COCO category ID to category name.
    E.g., {1: 'person', 2: 'bicycle', 3: 'car', ...}
    """
    categories = GetCategories()
    return {cat['id']: cat['name'] for cat in categories}


def ResolveClassNamesToIds(class_names: list) -> list:
    """
    Convert a list of class names to COCO category IDs.
    
    Args:
        class_names: List of class names (e.g., ['person', 'car'])
        
    Returns:
        List of COCO category IDs (e.g., [1, 3])
    """
    name_to_id = GetCategoryNameToIdMap()
    ids = []
    for name in class_names:
        if name in name_to_id:
            ids.append(name_to_id[name])
        else:
            raise ValueError(f"Unknown class name: '{name}'. Available classes: {list(name_to_id.keys())}")
    return ids


def GetFilteredCategories(class_ids: list) -> list:
    """
    Get filtered categories for the specified class IDs.
    
    Keeps original COCO category IDs - does NOT remap them.
    This preserves compatibility with pre-trained models, COCO evaluation tools,
    and standard COCO conventions.
    
    Args:
        class_ids: List of COCO category IDs to include (e.g., [1] for person only)
        
    Returns:
        List of category dicts with original COCO IDs preserved
    """
    all_categories = GetCategories()
    
    # Filter to only requested categories
    filtered = [cat for cat in all_categories if cat['id'] in class_ids]
    
    # Sort by original ID to maintain consistent ordering
    filtered.sort(key=lambda x: x['id'])
    
    return filtered


def GetCocoIdToYoloIndexMap(class_ids: list) -> dict:
    """
    Create a mapping from COCO category IDs to YOLO class indices.
    
    YOLO uses 0-indexed sequential class indices based on position in the class list.
    COCO uses fixed category IDs (person=1, car=3, etc. with gaps).
    
    This mapping is used when converting COCO annotations to YOLO format.
    
    Args:
        class_ids: List of COCO category IDs being used (e.g., [1, 3] for person and car)
        
    Returns:
        Dict mapping COCO ID -> YOLO index (e.g., {1: 0, 3: 1})
    """
    # Sort to ensure consistent ordering
    sorted_ids = sorted(class_ids)
    return {coco_id: yolo_idx for yolo_idx, coco_id in enumerate(sorted_ids)}



def GetYolo2COCO_CategoryMapping()->dict:
    """
    Apparently, YOLO class order is misaligned and in a different order respect the one provided by COCO.
    This dict maps from yolo categories to COCO categories (when possible). 
    """
    # Apparently, YOLO class order is misaligned and in a different order respect the one provided by COCO.
    # In order to keep the supercategories (available in COCO categories, not available in YOLO ones),
    # let's code a mapping from YOLO categories to COCO ones.


    mapping_file = files("nerve.data").joinpath("yolo2coco_categories_map.json")
    mapping = json.loads(mapping_file.read_text(encoding="utf-8"))
    
    # NOTE that keys of this dict were integers originally, which got converted to strings automatically from json.
    # Let's remap it to integers.
    corrected_mapping = {}
    for k in mapping.keys():
        corrected_mapping[int(k)] = mapping[k]
    return corrected_mapping