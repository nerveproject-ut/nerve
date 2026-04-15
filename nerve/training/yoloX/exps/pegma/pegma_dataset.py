#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
PEGMA Dataset for YOLOX with full capabilities:
- PEGMA folder structure support
- Radar fusion (DVS + radar point cloud)
- Distance estimation labels
- Source prefix handling

This unifies the capabilities of:
- Old DVS_Radar_Dataset (radar fusion, distance)
- New PEGMACOCODataset (PEGMA folder structure)
"""

import os
import copy
import cv2
import numpy as np
from pycocotools.coco import COCO

# Import YOLOX components
import sys
from pathlib import Path
yolox_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(yolox_root))

from yolox.data import COCODataset, TrainTransform, ValTransform
from yolox.data.datasets.coco import remove_useless_info
from yolox.data.datasets.datasets_wrapper import CacheDataset


class PEGMADataset(COCODataset):
    """
    PEGMA Dataset for YOLOX with radar fusion and distance estimation.
    
    Handles PEGMA directory structure:
        {data_dir}/{split}/annotations/{json_file}
        {data_dir}/{split}/images/{source}/*.png
        {data_dir}/{split}/images/ti_radar/*.png  (for radar fusion)
    
    Features:
        - Radar point cloud fusion into image's 3rd channel
        - Distance labels (6th column in annotations)
        - Distance-based sample filtering
        - Source prefix handling (davis/filename.png -> filename.png)
    """
    
    def __init__(
        self,
        data_dir: str,
        json_file: str = "davis.json",
        name: str = "davis",  # Source name (davis, prophesee, etc.)
        split: str = "train",  # train, val, or test
        img_size: tuple = (416, 416),
        preproc=None,
        cache: bool = False,
        cache_type: str = "ram",
        # Radar/Distance parameters
        use_also_radar: bool = False,
        include_distance: bool = False,
        min_dist: float = 0.0,
        max_dist: float = 10.0,
    ):
        """
        Initialize PEGMA dataset.
        
        Args:
            data_dir: Root directory of dataset
            json_file: COCO annotation filename
            name: Source folder name (davis, prophesee, etc.)
            split: Data split (train, val, test)
            img_size: Target image size (height, width)
            preproc: Preprocessing/augmentation transform
            cache: Whether to cache images
            cache_type: Cache type ('ram' or 'disk')
            use_also_radar: If True, fuse radar point cloud into 3rd channel
            include_distance: If True, load distance labels
            min_dist: Minimum distance threshold (meters)
            max_dist: Maximum distance threshold (meters)
        """
        # Store custom parameters
        self.data_dir = data_dir
        self.json_file = json_file
        self.split = split
        self.source = name
        self.img_size = img_size
        self.preproc = preproc
        
        # Radar/Distance parameters
        self.use_also_radar = use_also_radar
        self.include_distance = include_distance
        self.min_distance = min_dist
        self.max_distance = max_dist
        
        # Load COCO annotations
        ann_path = os.path.join(self.data_dir, split, "annotations", json_file)
        print(f"Loading annotations from: {ann_path}")
        
        if not os.path.exists(ann_path):
            raise FileNotFoundError(f"Annotation file not found: {ann_path}")
        
        self.coco = COCO(ann_path)
        remove_useless_info(self.coco)
        
        self.ids = self.coco.getImgIds()
        self.class_ids = sorted(self.coco.getCatIds())
        self.cats = self.coco.loadCats(self.coco.getCatIds())
        self._classes = tuple([c["name"] for c in self.cats])
        self.name = name
        
        # Load annotations (uses parent's _load_coco_annotations or our custom version)
        if self.include_distance:
            self.annotations = self._load_coco_annotations_with_distance()
        else:
            self.annotations = self._load_coco_annotations()
        
        # Filter by distance if enabled
        if self.include_distance:
            self._filter_by_distance()
        
        self.num_imgs = len(self.ids)
        
        # Build path_filename list for CacheDataset
        # Handle source prefix in filenames
        img_base = os.path.join(split, "images", name)
        path_filename = []
        for anno in self.annotations:
            fname = anno[3]
            fname = self._strip_source_prefix(fname)
            path_filename.append(os.path.join(img_base, fname))
        
        # Initialize CacheDataset (parent of COCODataset)
        CacheDataset.__init__(
            self,
            input_dimension=img_size,
            num_imgs=self.num_imgs,
            data_dir=data_dir,
            cache_dir_name=f"cache_{split}_{name}",
            path_filename=path_filename,
            cache=cache,
            cache_type=cache_type
        )
        
        print(f"PEGMA Dataset initialized:")
        print(f"  - Split: {split}")
        print(f"  - Source: {name}")
        print(f"  - Images: {self.num_imgs}")
        print(f"  - Classes: {len(self.class_ids)}")
        print(f"  - Radar fusion: {use_also_radar}")
        print(f"  - Distance labels: {include_distance}")
        if include_distance:
            print(f"  - Distance range: [{min_dist}, {max_dist}]m")
    
    def _strip_source_prefix(self, filename: str) -> str:
        """
        Strip source folder prefix from filename if present.
        
        Handles:
            "davis/image.png" -> "image.png"
            "davis\\image.png" -> "image.png"
            "image.png" -> "image.png"
        """
        if filename.startswith(f"{self.source}/"):
            return filename[len(self.source) + 1:]
        if filename.startswith(f"{self.source}\\"):
            return filename[len(self.source) + 1:]
        return filename
    
    def _load_coco_annotations_with_distance(self):
        """
        Load COCO annotations with distance labels.
        
        Returns annotations as list of tuples:
            (labels, img_info, resized_info, file_name)
        
        Where labels has shape (N, 6): [x1, y1, x2, y2, class_id, distance]
        """
        annotations = []
        
        for img_id in self.ids:
            im_ann = self.coco.loadImgs(img_id)[0]
            width = im_ann["width"]
            height = im_ann["height"]
            
            anno_ids = self.coco.getAnnIds(imgIds=[int(img_id)], iscrowd=False)
            objs = self.coco.loadAnns(anno_ids)
            
            valid_objs = []
            for obj in objs:
                x1 = np.max((0, obj["bbox"][0]))
                y1 = np.max((0, obj["bbox"][1]))
                x2 = np.min((width, x1 + np.max((0, obj["bbox"][2]))))
                y2 = np.min((height, y1 + np.max((0, obj["bbox"][3]))))
                
                if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                    obj["clean_bbox"] = [x1, y1, x2, y2]
                    valid_objs.append(obj)
            
            num_objs = len(valid_objs)
            
            # 6 columns: x1, y1, x2, y2, class_id, distance
            res = np.zeros((num_objs, 6))
            
            for ix, obj in enumerate(valid_objs):
                cls = self.class_ids.index(obj["category_id"])
                res[ix, 0:4] = obj["clean_bbox"]
                res[ix, 4] = cls
                # Get distance from annotation (default 0 if not present)
                res[ix, 5] = obj.get("avg_distance", 0.0)
            
            # Scale bboxes to target size
            r = min(self.img_size[0] / height, self.img_size[1] / width)
            res[:, :4] *= r
            
            img_info = (height, width)
            resized_info = (int(height * r), int(width * r))
            
            file_name = im_ann.get("file_name", f"{img_id:012d}.jpg")
            
            annotations.append((res, img_info, resized_info, file_name))
        
        return annotations
    
    def _filter_by_distance(self):
        """Filter out samples with distances outside the valid range."""
        idx_to_remove = []
        
        for idx, (ann, _, _, _) in enumerate(self.annotations):
            for subject in range(len(ann)):
                dist = ann[subject, 5]
                if dist < self.min_distance or dist > self.max_distance:
                    idx_to_remove.append(idx)
                    break
        
        if idx_to_remove:
            print(f"Filtering {len(idx_to_remove)} samples with distance out of range "
                  f"[{self.min_distance}, {self.max_distance}]")
            
            # Remove in reverse order to preserve indices
            for idx in sorted(idx_to_remove, reverse=True):
                self.ids.pop(idx)
                self.annotations.pop(idx)
        
        assert len(self.ids) == len(self.annotations)
    
    def load_image(self, index: int) -> np.ndarray:
        """
        Load image with optional radar fusion.
        
        PEGMA structure:
            {data_dir}/{split}/images/{source}/{filename}.png
            {data_dir}/{split}/images/ti_radar/point_cloud_img__{frame_num}.png
        
        Args:
            index: Image index
            
        Returns:
            Image array (H, W, 3) with radar fused into 3rd channel if enabled
        """
        file_name = self.annotations[index][3]
        file_name = self._strip_source_prefix(file_name)
        
        # Construct image path
        img_file = os.path.join(
            self.data_dir,
            self.split,
            "images",
            self.source,
            file_name
        )
        
        img = cv2.imread(img_file)
        assert img is not None, f"File not found: {img_file}"
        
        if not self.use_also_radar:
            return img
        
        # Fuse radar point cloud into 3rd channel
        # Extract frame number from filename (e.g., histogram_img__0000000171.png -> 0000000171)
        file_stem = os.path.splitext(file_name)[0]
        frame_num = file_stem.split('_')[-1]
        
        # Construct radar path
        radar_file = os.path.join(
            self.data_dir,
            self.split,
            "images",
            "ti_radar",
            f"point_cloud_img__{frame_num}.png"
        )
        
        if not os.path.exists(radar_file):
            print(f"Warning: Radar file not found: {radar_file}")
            return img
        
        radar_frame = cv2.imread(radar_file)
        if radar_frame is None:
            print(f"Warning: Failed to load radar file: {radar_file}")
            return img
        
        # Dilate radar points to prevent loss during resizing
        radar_frame = cv2.dilate(radar_frame, np.ones((5, 5), np.uint8))
        
        # Resize radar to match image if needed
        if radar_frame.shape[:2] != img.shape[:2]:
            radar_frame = cv2.resize(
                radar_frame,
                (img.shape[1], img.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
        
        # Fuse: replace R channel (index 2 in BGR) with radar data
        img[:, :, 2] = radar_frame[:, :, 0]
        
        return img
    
    def load_resized_img(self, index: int) -> np.ndarray:
        """Load and resize image."""
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR if not self.use_also_radar else cv2.INTER_NEAREST
        ).astype(np.uint8)
        return resized_img
    
    def load_anno(self, index: int):
        """Load annotations for given index."""
        return self.annotations[index][0]
    
    def pull_item(self, index: int):
        """
        Get single image and label for training.
        
        Returns:
            img: Resized image array
            label: Annotations (N, 5) or (N, 6) if include_distance
            img_info: Original image size (h, w)
            img_id: Image ID
        """
        id_ = self.ids[index]
        label, origin_image_size, _, _ = self.annotations[index]
        img = self.load_resized_img(index)
        
        # For standard detection, only return first 5 columns
        if not self.include_distance:
            label = label[:, :5]
        
        return img, copy.deepcopy(label), origin_image_size, np.array([id_])
    
    def __len__(self):
        return self.num_imgs
    
    @property
    def input_dim(self):
        """
        Return input dimension for YOLOX DataLoader compatibility.
        Required by YoloBatchSampler.
        """
        return self.img_size


# Convenience functions
def create_pegma_train_dataset(
    data_dir: str,
    json_file: str = "davis.json",
    source: str = "davis",
    img_size: tuple = (416, 416),
    use_radar: bool = False,
    include_distance: bool = False,
    min_dist: float = 0.0,
    max_dist: float = 10.0,
    flip_prob: float = 0.5,
    hsv_prob: float = 0.0,
    cache: bool = False,
) -> PEGMADataset:
    """Create training dataset with augmentation."""
    return PEGMADataset(
        data_dir=data_dir,
        json_file=json_file,
        name=source,
        split="train",
        img_size=img_size,
        preproc=TrainTransform(
            max_labels=50,
            flip_prob=flip_prob,
            hsv_prob=hsv_prob
        ),
        cache=cache,
        use_also_radar=use_radar,
        include_distance=include_distance,
        min_dist=min_dist,
        max_dist=max_dist,
    )


def create_pegma_val_dataset(
    data_dir: str,
    json_file: str = "davis.json",
    source: str = "davis",
    img_size: tuple = (416, 416),
    use_radar: bool = False,
    include_distance: bool = False,
    min_dist: float = 0.0,
    max_dist: float = 10.0,
) -> PEGMADataset:
    """Create validation dataset."""
    return PEGMADataset(
        data_dir=data_dir,
        json_file=json_file,
        name=source,
        split="val",
        img_size=img_size,
        preproc=ValTransform(legacy=False),
        use_also_radar=use_radar,
        include_distance=include_distance,
        min_dist=min_dist,
        max_dist=max_dist,
    )


# For testing
if __name__ == '__main__':
    # Test the dataset
    data_dir = "/scratch-shared/tmp.8EGdXT6jjc/clean_pipeline_yolox_yolov8_distance"
    
    print("Testing PEGMADataset...")
    
    # Test with distance
    dataset = PEGMADataset(
        data_dir=data_dir,
        json_file="davis.json",
        name="davis",
        split="train",
        img_size=(416, 416),
        use_also_radar=True,
        include_distance=True,
        min_dist=0.0,
        max_dist=10.0,
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        img, label, img_info, img_id = dataset.pull_item(0)
        print(f"Image shape: {img.shape}")
        print(f"Label shape: {label.shape}")
        print(f"First label: {label[0] if len(label) > 0 else 'No labels'}")

