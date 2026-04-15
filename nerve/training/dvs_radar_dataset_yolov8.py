"""
Custom YOLOv8 Dataset for DVS+Radar with optional distance estimation.
Similar to dvs_radar_dataset.py but adapted for YOLOv8 framework.
"""

import os
import cv2
import numpy as np
import torch
import yaml
from pathlib import Path
from pycocotools.coco import COCO

try:
    from ultralytics.data import YOLODataset
    from ultralytics.utils import LOGGER
except ImportError:
    raise ImportError("YOLOv8 (ultralytics) is not installed. Install with: pip install ultralytics")


class YOLOv8_DVS_Radar_Dataset(YOLODataset):
    """
    Custom YOLOv8 dataset that supports:
    - DVS + Radar data fusion
    - Optional distance labels for distance estimation
    - Filtering samples based on distance range
    """
    
    def __init__(
        self,
        img_path,
        imgsz=640,
        cache=False,
        augment=True,
        hyp=None,
        prefix='',
        rect=False,
        batch_size=None,
        stride=32,
        pad=0.5,
        single_cls=False,
        classes=None,
        fraction=1.0,
        # Custom parameters
        use_also_radar=False,
        include_distance=False,
        min_dist=0.0,
        max_dist=10.0,
        json_file=None,
        data_dir=None,
        data_dict=None
    ):
        """
        Args:
            img_path: Path to images or COCO JSON file
            use_also_radar: If True, fuse radar point cloud data with DVS
            include_distance: If True, include distance labels (6th dimension)
            min_dist: Minimum distance threshold (meters)
            max_dist: Maximum distance threshold (meters)
            json_file: COCO annotation file name
            data_dir: Root directory of dataset
        """
        self.use_also_radar = use_also_radar
        self.include_distance = include_distance
        self.min_distance = min_dist
        self.max_distance = max_dist
        self.json_file = json_file
        self.data_dir = data_dir
        self.coco = None  # Initialize COCO object placeholder
        
        # Set data dict BEFORE parent init - the parent class needs this
        if data_dict is not None:
            self.data = data_dict
        elif isinstance(img_path, str) and img_path.endswith('.yaml'):
            with open(img_path, 'r') as f:
                self.data = yaml.safe_load(f)
        elif isinstance(img_path, dict):
            self.data = img_path
        else:
            # Create a minimal data dict with all required fields
            self.data = YOLOv8_DVS_Radar_Dataset._get_default_data_dict()
        
        # Initialize COCO object early if we have json_file
        # This is needed for cache_labels which runs during parent __init__
        if json_file and data_dir:
            self._init_coco(data_dir, json_file)
        
        # Store data dict temporarily (parent may overwrite self.data)
        _temp_data = self.data
        
        # Initialize parent class
        super().__init__(
            img_path=img_path,
            imgsz=imgsz,
            cache=cache,
            augment=augment,
            hyp=hyp,
            prefix=prefix,
            rect=rect,
            batch_size=batch_size,
            stride=stride,
            pad=pad,
            single_cls=single_cls,
            classes=classes,
            fraction=fraction,
            data=_temp_data,
        )
        
        # Restore data dict if parent overwrote it
        if self.data is None:
            self.data = _temp_data
    
    def _init_coco(self, data_dir, json_file):
        """Initialize COCO object from annotation file."""
        # Try multiple path patterns for COCO annotations:
        # 1. Direct path: data_dir/json_file (e.g., dataset/train/annotations/davis.json)
        # 2. Legacy path: data_dir/coco_labels/json_file
        # 3. Absolute path: if json_file is already absolute
        
        possible_paths = []
        
        # If json_file is absolute, use it directly
        if os.path.isabs(json_file):
            possible_paths.append(json_file)
        else:
            # Try direct path first (json_file is relative like "train/annotations/davis.json")
            possible_paths.append(os.path.join(data_dir, json_file))
            # Try legacy path with coco_labels subdirectory
            possible_paths.append(os.path.join(data_dir, "coco_labels", json_file))
        
        coco_path = None
        for path in possible_paths:
            if os.path.exists(path):
                coco_path = path
                break
        
        if coco_path:
            # Temporarily suppress COCO's print output
            import sys
            import io
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                self.coco = COCO(coco_path)
            finally:
                sys.stdout = old_stdout
            LOGGER.info(f"Loaded COCO annotations from: {coco_path}")
        else:
            LOGGER.warning(f"COCO annotation file not found. Tried: {possible_paths}")
        
    def get_img_files(self, img_path):
        """Override to handle COCO-style dataset with custom filtering."""
        if self.json_file and self.data_dir:
            # Initialize COCO if not already done
            if self.coco is None:
                self._init_coco(self.data_dir, self.json_file)
            
            if self.coco is not None:
                img_ids = self.coco.getImgIds()
                
                # Filter based on distance if needed
                if self.include_distance:
                    img_ids = self._filter_by_distance(img_ids)
                
                # Get image paths
                # Handle path construction carefully - COCO file_name may include subfolder
                # E.g., file_name='davis/image.png' and img_path='train/images/davis'
                # We need to avoid creating 'train/images/davis/davis/image.png'
                img_files = []
                for img_id in img_ids:
                    img_info = self.coco.loadImgs(img_id)[0]
                    file_name = img_info['file_name']
                    
                    # Check if file_name starts with a subfolder that's already in img_path
                    # e.g., file_name='davis/img.png', img_path='train/images/davis'
                    file_parts = file_name.split('/')
                    img_path_parts = img_path.rstrip('/').split('/')
                    
                    if len(file_parts) > 1 and file_parts[0] == img_path_parts[-1]:
                        # file_name starts with the same folder that img_path ends with
                        # Use only the filename part (strip the redundant folder prefix)
                        file_name = '/'.join(file_parts[1:])
                    
                    # Construct full path
                    img_file = os.path.join(self.data_dir, img_path, file_name)
                    
                    if os.path.exists(img_file):
                        img_files.append(img_file)
                    elif len(img_files) == 0:
                        # Debug: log first missing file to help diagnose path issues
                        LOGGER.warning(f"First image not found at: {img_file}")
                
                LOGGER.info(f"Loaded {len(img_files)} images from COCO dataset")
                return img_files
        
        # Fall back to parent implementation
        return super().get_img_files(img_path)
    
    def _filter_by_distance(self, img_ids):
        """Filter image IDs based on distance range."""
        filtered_ids = []
        removed_count = 0
        
        for img_id in img_ids:
            ann_ids = self.coco.getAnnIds(imgIds=[int(img_id)], iscrowd=False)
            annotations = self.coco.loadAnns(ann_ids)
            
            valid = True
            for ann in annotations:
                if 'avg_distance' in ann:
                    dist = ann['avg_distance']
                    if dist < self.min_distance or dist > self.max_distance:
                        valid = False
                        removed_count += 1
                        break
            
            if valid:
                filtered_ids.append(img_id)
        
        if removed_count > 0:
            LOGGER.info(f"Filtered out {removed_count} images with distance out of range "
                       f"[{self.min_distance}, {self.max_distance}]")
        
        return filtered_ids
    
    def cache_labels(self, path=Path('./labels.cache')):
        """
        Override cache_labels to handle COCO format and YOLO txt labels.
        
        This method handles:
        1. COCO JSON annotations (preferred for distance datasets)
        2. YOLO txt labels with proper path conversion (handles data/ symlink issue)
        
        Returns dict format expected by YOLODataset parent class.
        """
        # Ensure self.data is set before accessing it
        if not hasattr(self, 'data') or self.data is None:
            self.data = YOLOv8_DVS_Radar_Dataset._get_default_data_dict()
        
        labels = None
        
        # Check if we have COCO annotations available
        if self.coco is not None:
            # Load labels directly from COCO - skip parent's .txt file scanning
            LOGGER.info(f"Loading labels from COCO annotations (skipping YOLO .txt scan)")
            labels = self._load_labels_from_coco()
        else:
            # No COCO - try to load YOLO txt labels with custom path handling
            LOGGER.info("Loading labels from YOLO txt files")
            labels = self._load_labels_from_yolo_txt()
        
        if labels:
            # Return in the format expected by YOLODataset parent class
            # Count images with/without labels
            nf = sum(1 for lb in labels if len(lb.get('bboxes', [])) > 0)  # found
            nm = 0  # missing (we handle this differently)
            ne = 0  # empty annotations  
            nc = 0  # corrupt
            
            return {
                'labels': labels,
                'hash': None,
                'results': (nf, nm, ne, nc, len(labels)),
                'msgs': [],
                'version': '1.0.0'
            }
        
        # Last resort: fall back to parent implementation
        LOGGER.warning("Custom label loading failed, falling back to parent implementation")
        return super().cache_labels(path)
    
    def _load_labels_from_yolo_txt(self):
        """
        Load labels from YOLO txt files.
        Handles the symlink issue where images are in /data/ but labels are in /labels/.
        """
        labels = []
        
        if not hasattr(self, 'im_files') or not self.im_files:
            LOGGER.warning("No image files found for YOLO label loading")
            return labels
        
        num_with_labels = 0
        num_without_labels = 0
        
        for img_file in self.im_files:
            img_path = Path(img_file)
            
            # Convert image path to label path
            # Standard: replace /images/ with /labels/, .jpg/.png with .txt
            # Also handle: /data/ → /labels/ for symlink case
            label_path = None
            
            # Try standard conversion first (/images/ → /labels/)
            img_str = str(img_path)
            for img_dir in ['/images/', '/data/']:
                if img_dir in img_str:
                    label_str = img_str.replace(img_dir, '/labels/')
                    label_str = str(Path(label_str).with_suffix('.txt'))
                    if os.path.exists(label_str):
                        label_path = label_str
                        break
            
            # Read label file
            if label_path and os.path.exists(label_path):
                try:
                    with open(label_path, 'r') as f:
                        lines = f.read().strip().split('\n')
                    
                    bboxes = []
                    classes = []
                    distances = []
                    
                    for line in lines:
                        if not line.strip():
                            continue
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls = int(parts[0])
                            # YOLO format: class cx cy w h [optional: distance]
                            # Convert from xywh (center) to xyxy format for consistency
                            cx, cy, bw, bh = map(float, parts[1:5])
                            # Convert center xywh to xyxy (normalized)
                            x1 = cx - bw / 2
                            y1 = cy - bh / 2
                            x2 = cx + bw / 2
                            y2 = cy + bh / 2
                            bboxes.append([x1, y1, x2, y2])
                            classes.append(cls)
                            # Check for distance (6th column)
                            if len(parts) >= 6:
                                distances.append(float(parts[5]))
                            else:
                                distances.append(0.0)
                    
                    if bboxes:
                        num_with_labels += 1
                        try:
                            img = cv2.imread(img_file)
                            h, w = img.shape[:2] if img is not None else (640, 640)
                        except Exception:
                            h, w = 640, 640
                        
                        labels.append({
                            'im_file': img_file,
                            'shape': (h, w),
                            'cls': np.array(classes, dtype=np.float32).reshape(-1, 1),
                            'bboxes': np.array(bboxes, dtype=np.float32),
                            'segments': [],
                            'keypoints': None,
                            'normalized': True,
                            'bbox_format': 'xyxy',  # Standard format - transforms convert to xywh
                            'distances': np.array(distances, dtype=np.float32)
                        })
                        continue
                except Exception as e:
                    LOGGER.warning(f"Error reading label file {label_path}: {e}")
            
            # No label found or empty - create background entry
            num_without_labels += 1
            try:
                img = cv2.imread(img_file)
                h, w = img.shape[:2] if img is not None else (640, 640)
            except Exception:
                h, w = 640, 640
            
            labels.append({
                'im_file': img_file,
                'shape': (h, w),
                'cls': np.zeros((0, 1), dtype=np.float32),
                'bboxes': np.zeros((0, 4), dtype=np.float32),
                'segments': [],
                'keypoints': None,
                'normalized': True,
                'bbox_format': 'xyxy',  # Consistent with labels that have boxes
                'distances': np.zeros((0,), dtype=np.float32)
            })
        
        LOGGER.info(f"Loaded YOLO labels: {num_with_labels} images with labels, {num_without_labels} background images")
        return labels
    
    def _load_labels_from_coco(self):
        """
        Load labels from COCO annotations.
        Returns labels in the format expected by YOLODataset.
        """
        labels = []
        
        if self.coco is None:
            LOGGER.warning("COCO object not initialized, cannot load labels")
            return labels
        
        if not hasattr(self, 'im_files') or not self.im_files:
            LOGGER.warning("No image files found for COCO label loading")
            return labels
        
        num_with_labels = 0
        num_without_labels = 0
        
        # Build filename-to-ID lookup dictionary ONCE for O(1) lookups
        # This avoids O(n²) nested loop over all images
        img_ids = self.coco.getImgIds()
        filename_to_id = {}
        for iid in img_ids:
            img_info = self.coco.loadImgs(iid)[0]
            coco_filename = Path(img_info['file_name']).name
            filename_to_id[coco_filename] = iid
        LOGGER.info(f"Built filename lookup for {len(filename_to_id)} COCO images")
        
        for img_file in self.im_files:
            img_name = Path(img_file).name
            
            # O(1) lookup instead of O(n) iteration
            img_id = filename_to_id.get(img_name)
            
            if img_id is None:
                # Image not in COCO annotations - create empty label
                num_without_labels += 1
                try:
                    img = cv2.imread(img_file)
                    h, w = img.shape[:2] if img is not None else (640, 640)
                except Exception:
                    h, w = 640, 640
                
                labels.append({
                    'im_file': img_file,
                    'shape': (h, w),
                    'cls': np.zeros((0, 1), dtype=np.float32),
                    'bboxes': np.zeros((0, 4), dtype=np.float32),
                    'segments': [],
                    'keypoints': None,
                    'normalized': True,
                    'bbox_format': 'xyxy',  # Consistent with labels that have boxes
                    'distances': np.zeros((0,), dtype=np.float32)
                })
                continue
            
            # Load annotations for this image
            img_info = self.coco.loadImgs(img_id)[0]
            ann_ids = self.coco.getAnnIds(imgIds=[int(img_id)], iscrowd=False)
            annotations = self.coco.loadAnns(ann_ids)
            
            w, h = img_info['width'], img_info['height']
            
            bboxes = []
            classes = []
            distances = []
            
            for ann in annotations:
                # COCO bbox format is [x, y, w, h] (top-left corner + width/height)
                x, y, w_box, h_box = ann['bbox']
                
                # Clamp to image boundaries
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(w, x + w_box)
                y2 = min(h, y + h_box)
                
                # Calculate actual width and height after clamping
                actual_w = x2 - x1
                actual_h = y2 - y1
                
                if ann['area'] > 0 and actual_w > 0 and actual_h > 0:
                    # Convert to normalized xyxy format (x1, y1, x2, y2)
                    # This matches standard YOLODataset format - transforms will convert to xywh
                    nx1 = x1 / w  # normalized x1
                    ny1 = y1 / h  # normalized y1
                    nx2 = x2 / w  # normalized x2
                    ny2 = y2 / h  # normalized y2
                    bboxes.append([nx1, ny1, nx2, ny2])
                    
                    # Get class (single_cls mode maps all to 0)
                    if hasattr(self, 'data') and self.data and self.data.get('nc', None) == 1:
                        cls = 0
                    else:
                        cat_id = ann['category_id']
                        try:
                            cls = list(self.coco.getCatIds()).index(cat_id)
                        except ValueError:
                            continue
                    classes.append(cls)
                    
                    # Get distance if available
                    dist = ann.get('avg_distance', 0.0)
                    distances.append(dist)
            
            if len(bboxes) > 0:
                num_with_labels += 1
                bboxes_array = np.array(bboxes, dtype=np.float32)
            else:
                num_without_labels += 1
                bboxes_array = np.zeros((0, 4), dtype=np.float32)
            
            labels.append({
                'im_file': img_file,
                'shape': (h, w),
                'cls': np.array(classes, dtype=np.float32).reshape(-1, 1) if classes else np.zeros((0, 1), dtype=np.float32),
                'bboxes': bboxes_array,
                'segments': [],
                'keypoints': None,
                'normalized': True,
                'bbox_format': 'xyxy',  # Standard YOLODataset format - transforms convert to xywh
                'distances': np.array(distances, dtype=np.float32) if distances else np.zeros((0,), dtype=np.float32)
            })
        
        LOGGER.info(f"Loaded COCO labels: {num_with_labels} images with labels, {num_without_labels} background images")
        
        return labels
    
    def build_transforms(self, hyp=None):
        """
        Override build_transforms to ensure self.data exists.
        The parent's __init__ calls this method which expects self.data to be set.
        """
        # Ensure self.data is set before parent's build_transforms accesses it
        if not hasattr(self, 'data') or self.data is None:
            self.data = YOLOv8_DVS_Radar_Dataset._get_default_data_dict()
        
        return super().build_transforms(hyp=hyp)
    
    @staticmethod
    def _get_default_data_dict():
        """Get default data dictionary with required fields."""
        return {
            'names': {0: 'person'},
            'nc': 1,
            'kpt_shape': (0, 0),
            'flip_idx': []  # Required for keypoints augmentation
        }
    
    def load_image(self, i):
        """
        Load image with optional radar fusion.
        Similar to dvs_radar_dataset.py load_image method.
        
        Dataset structure:
        - images/train/davis/ : DVS only (needs fusion with ti_radar/)
        - images/train/davis_radar/ : DVS + Radar pre-fused (no additional fusion needed)
        - images/train/ti_radar/ : Radar point clouds only
        """
        # Load base image using parent method
        result = super().load_image(i)
        
        # Handle different return formats from parent
        if isinstance(result, tuple) and len(result) == 3:
            im, f, fn = result
        else:
            # Unexpected format, return as-is
            return result
        
        if not self.use_also_radar:
            return im, f, fn
        
        # Get the actual file path
        file_path = self.im_files[i] if hasattr(self, 'im_files') and i < len(self.im_files) else f
        parent_dir = Path(file_path).parent
        folder_name = parent_dir.name  # e.g., 'davis', 'davis_radar', 'ti_radar'
        
        # Check if we're loading from pre-fused data
        if 'radar' in folder_name and folder_name != 'ti_radar':
            # This is pre-fused data (e.g., davis_radar, prophesee_radar)
            # No additional fusion needed - radar is already in the image
            return im, f, fn
        
        # We're loading from DVS-only folder (e.g., 'davis' or 'prophesee')
        # Need to manually fuse with radar point cloud
        file_name = Path(file_path).name
        file_number = file_name.split('.')[0].split('_')[-1]
        
        # Construct radar file path
        # Radar point clouds are in ti_radar folder at the same level
        grandparent_dir = parent_dir.parent
        radar_dir = grandparent_dir / 'ti_radar'
        radar_path = radar_dir / f'point_cloud_img__{file_number}.png'
        
        if not radar_path.exists():
            LOGGER.warning(f"Radar file not found: {radar_path}")
            return im, f, fn
        
        # Load radar frame (grayscale but stored as 3-channel)
        radar_frame = cv2.imread(str(radar_path))
        if radar_frame is None:
            LOGGER.warning(f"Failed to load radar frame: {radar_path}")
            return im, f, fn
        
        # Dilate radar points to prevent loss during resize
        radar_frame = cv2.dilate(radar_frame, np.ones((5, 5), np.uint8))
        
        # Resize radar to match image size if needed
        if radar_frame.shape[:2] != im.shape[:2]:
            radar_frame = cv2.resize(
                radar_frame, 
                (im.shape[1], im.shape[0]), 
                interpolation=cv2.INTER_NEAREST
            )
        
        # Fuse: replace R channel (index 2 in BGR) with radar data
        im[:, :, 2] = radar_frame[:, :, 0]
        
        return im, f, fn
    
    def get_labels(self):
        """Load labels with optional distance information."""
        self.label_files = []
        
        # Check if we have COCO annotations
        if self.coco is not None:
            return self._load_labels_from_coco()
        
        # Try YOLO txt labels with custom path handling
        labels = self._load_labels_from_yolo_txt()
        if labels:
            return labels
        
        # Fall back to parent implementation for standard YOLOv8 labels
        return super().get_labels()
    
    def __getitem__(self, index):
        """Get item with optional distance labels."""
        data = super().__getitem__(index)
        
        # Handle distance information if include_distance is enabled
        # The distances should already be in the data dict if they were in the labels
        # (they flow through the transform pipeline along with bboxes and cls)
        if self.include_distance:
            # Check if distances came through the transforms
            if 'distances' in data:
                # Convert numpy to tensor if needed
                if isinstance(data['distances'], np.ndarray):
                    data['distances'] = torch.from_numpy(data['distances']).float()
                elif not isinstance(data['distances'], torch.Tensor):
                    data['distances'] = torch.tensor(data['distances'], dtype=torch.float32)
            else:
                # Distances not in data - this shouldn't happen if labels were loaded correctly
                # Fall back to getting from original label (for non-augmented case)
                distances = np.zeros((0,), dtype=np.float32)
                if hasattr(self, 'labels') and self.labels and index < len(self.labels):
                    label = self.labels[index]
                    if 'distances' in label:
                        distances = label['distances']
                data['distances'] = torch.from_numpy(distances).float()
            
            # Ensure distances match bboxes count (sanity check)
            # If they don't match, we have an augmentation issue
            if 'bboxes' in data and len(data['distances']) != len(data['bboxes']):
                # Mismatch - this means augmentation didn't properly handle distances
                # Pad or truncate distances to match bboxes
                num_bboxes = len(data['bboxes'])
                num_dists = len(data['distances'])
                
                if num_dists < num_bboxes:
                    # Pad with -1 (invalid distance marker)
                    padding = torch.full((num_bboxes - num_dists,), -1.0, dtype=torch.float32)
                    data['distances'] = torch.cat([data['distances'], padding])
                elif num_dists > num_bboxes:
                    # Truncate
                    data['distances'] = data['distances'][:num_bboxes]
        
        return data
    
    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function that handles distances in addition to standard YOLOv8 fields.
        Distances are concatenated like bboxes/cls (one distance value per object).
        
        Handles both numpy arrays and tensors (transforms may not convert during validation).
        """
        new_batch = {}
        
        # Get union of all keys across batch items to handle varying key sets
        # Access values by key name, not by positional .values() iteration
        all_keys = set()
        for b in batch:
            all_keys.update(b.keys())
        
        # First, build batch_idx with proper offsets (one index per object per image)
        batch_idx_list = []
        for batch_i, b in enumerate(batch):
            # Get number of objects in this image from bboxes
            if 'bboxes' in b:
                bboxes = b['bboxes']
                if isinstance(bboxes, np.ndarray):
                    n_objects = bboxes.shape[0]
                elif isinstance(bboxes, torch.Tensor):
                    n_objects = bboxes.shape[0]
                else:
                    n_objects = 0
            else:
                n_objects = 0
            # Create batch_idx entries for all objects in this image
            batch_idx_list.append(torch.full((n_objects,), batch_i, dtype=torch.long))
        
        for k in all_keys:
            # FIX: Extract values by key name from each batch item
            value = [b.get(k) for b in batch]
            
            # Skip keys where all values are None
            if all(v is None for v in value):
                continue
                
            if k == 'img':
                # Convert to tensor if needed, then stack
                # Filter out None values (shouldn't happen for img, but be safe)
                value = [torch.from_numpy(v) if isinstance(v, np.ndarray) else v for v in value if v is not None]
                value = torch.stack(value, 0)
            elif k == 'batch_idx':
                # Use pre-computed batch_idx with proper offsets
                value = torch.cat(batch_idx_list, 0) if batch_idx_list else torch.zeros(0, dtype=torch.long)
            elif k == 'bboxes':
                # Bboxes need shape [N, 4] even when empty
                converted = []
                for v in value:
                    if isinstance(v, np.ndarray):
                        t = torch.from_numpy(v).float()
                    elif isinstance(v, torch.Tensor):
                        t = v.float()
                    else:
                        t = torch.zeros((0, 4), dtype=torch.float32)
                    # Ensure 2D with 4 columns
                    if t.dim() == 1 and t.numel() == 0:
                        t = t.view(0, 4)
                    elif t.dim() == 1:
                        t = t.view(-1, 4)
                    converted.append(t)
                value = torch.cat(converted, 0) if converted else torch.zeros((0, 4), dtype=torch.float32)
            elif k == 'cls':
                # Cls needs shape [N, 1] even when empty
                converted = []
                for v in value:
                    if isinstance(v, np.ndarray):
                        t = torch.from_numpy(v).float()
                    elif isinstance(v, torch.Tensor):
                        t = v.float()
                    else:
                        t = torch.zeros((0, 1), dtype=torch.float32)
                    # Ensure 2D with 1 column
                    if t.dim() == 1:
                        t = t.view(-1, 1)
                    converted.append(t)
                value = torch.cat(converted, 0) if converted else torch.zeros((0, 1), dtype=torch.float32)
            elif k == 'distances':
                # Distances need shape [N] (1D)
                converted = []
                for v in value:
                    if isinstance(v, np.ndarray):
                        t = torch.from_numpy(v).float()
                    elif isinstance(v, torch.Tensor):
                        t = v.float()
                    else:
                        t = torch.zeros(0, dtype=torch.float32)
                    # Ensure 1D
                    if t.dim() > 1:
                        t = t.view(-1)
                    converted.append(t)
                value = torch.cat(converted, 0) if converted else torch.zeros(0, dtype=torch.float32)
            elif k in ['masks', 'keypoints']:
                # Convert numpy arrays to tensors before concatenating
                converted = []
                for v in value:
                    if isinstance(v, np.ndarray):
                        converted.append(torch.from_numpy(v))
                    elif isinstance(v, torch.Tensor):
                        converted.append(v)
                    elif v is not None:
                        converted.append(torch.tensor(v))
                if converted and all(c.numel() > 0 for c in converted):
                    value = torch.cat(converted, 0)
                else:
                    value = None  # No valid masks/keypoints
            new_batch[k] = value
        
        return new_batch


def create_yolov8_dataloader(
    path,
    imgsz=640,
    batch_size=16,
    stride=32,
    hyp=None,
    augment=False,
    cache=False,
    pad=0.5,
    rect=False,
    rank=-1,
    workers=8,
    close_mosaic=False,
    prefix='',
    shuffle=False,
    seed=0,
    # Custom parameters
    use_also_radar=False,
    include_distance=False,
    min_dist=0.0,
    max_dist=10.0,
    json_file=None,
    data_dir=None
):
    """
    Create a custom dataloader for YOLOv8 with DVS+Radar support.
    """
    from ultralytics.data import build_dataloader
    from torch.utils.data import DataLoader
    
    dataset = YOLOv8_DVS_Radar_Dataset(
        img_path=path,
        imgsz=imgsz,
        batch_size=batch_size,
        augment=augment,
        hyp=hyp,
        rect=rect,
        cache=cache,
        stride=stride,
        pad=pad,
        prefix=prefix,
        use_also_radar=use_also_radar,
        include_distance=include_distance,
        min_dist=min_dist,
        max_dist=max_dist,
        json_file=json_file,
        data_dir=data_dir
    )
    
    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])
    
    return dataset, DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and not rect,
        num_workers=nw,
        collate_fn=getattr(dataset, 'collate_fn', None),
        pin_memory=True
    )

