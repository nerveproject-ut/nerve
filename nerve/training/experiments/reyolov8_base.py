"""
ReYOLOv8-specific base configuration.
Extends BaseConfig with ReYOLOv8-specific parameters for event camera data.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports

from nerve.training.experiments.base import BaseConfig


class ReYOLOv8Base(BaseConfig):
    """
    Base configuration class for ReYOLOv8 experiments.
    Inherits from BaseConfig and adds ReYOLOv8-specific parameters for event camera data.
    """
    
    MODEL_TYPE = 'reyolov8'
    
    def __init__(self, dataset_path: str = None, data_yaml: str = None):
        super().__init__(dataset_path, data_yaml)
        
        # ReYOLOv8 Model Parameters
        self.model = 'ReYOLOv8n.yaml'  # Model architecture
        self.pretrained = None  # Path to pretrained weights
        
        # ReYOLOv8 Sequence Parameters
        self.channels = 10  # Number of input channels (e.g., 10 for VTEI, +1 if radar)
        self.clip_length = 11  # Number of frames per clip
        self.clip_stride = 11  # Stride between clips
        
        # Channel selection: None = use all, int = first N channels, list = specific indices
        # Example: select_channels=5 uses first 5 channels (ignores radar if dataset has 6)
        self.select_channels = None
        
        # ReYOLOv8 Training Parameters
        self.imgsz = 416
        self.optimizer = 'SGD'
        self.lrf = 0.01
        self.cos_lr = False
        self.warmup_bias_lr = 0.1
        self.warmup_momentum = 0.8
        self.nbs = 64
        
        self.val_interval = 10  # Validate every N epochs
        self.save_period = -1
        
        # Event-Specific Augmentation
        # ReYOLOv8 uses specialized augmentations for event data
        self.flip = 0.5  # Horizontal flip probability
        self.invert = 0.0  # Polarity inversion probability
        self.suppress = 0.0  # Random polarity suppression probability
        self.positive = 0.0  # Positive polarity suppression probability
        self.zoom_out = 0.0  # Zoom out probability
        
        # Standard augmentations (usually disabled for event data)
        self.hsv_h = 0.0
        self.hsv_s = 0.0
        self.hsv_v = 0.0
        self.degrees = 0.0
        self.translate = 0.0
        self.scale = 0.0
        self.shear = 0.0
        self.perspective = 0.0
        self.flipud = 0.0
        self.mosaic = 0.0
        self.mixup = 0.0
        
        # ReYOLOv8 Loss
        self.box_loss_gain = 7.5
        self.cls_loss_gain = 0.5
        self.dfl_loss_gain = 1.5
        
        # Validation
        self.conf_threshold = 0.001
        self.iou_threshold = 0.6
        self.max_det = 300
        self.single_cls = True
        
        # Output
        self.project = 'runs/reyolov8'
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]
    
    def get_model_config_path(self):
        """Get full path to model YAML configuration file."""
        if not self.model.endswith('.yaml'):
            self.model = f'{self.model}.yaml'
        
        model_path = Path(self.model)
        if model_path.is_absolute() and model_path.exists():
            return str(model_path)
        
        # Try to find in ReYOLOv8 model directory
        reyolov8_dir = Path(__file__).parent.parent / 'reyolov8'
        model_path = reyolov8_dir / 'ultralytics' / 'models' / 'v8' / 'Recurrent' / self.model
        
        if model_path.exists():
            return str(model_path)
        
        return self.model
    
    def to_dict(self) -> dict:
        """Convert config to dictionary for ReYOLOv8 training."""
        return {
            # Model type
            'model_type': self.MODEL_TYPE,
            
            # Data
            'data': self.data_yaml,
            'dataset_path': self.dataset_path,
            
            # Model
            'model': self.get_model_config_path(),
            'pretrained': self.pretrained,
            'nc': self.num_classes,
            
            # ReYOLOv8 specific
            'channels': self.channels,
            'clip_length': self.clip_length,
            'clip_stride': self.clip_stride,
            'select_channels': self.select_channels,
            
            # Training
            'epochs': self.epochs,
            'batch': self.batch_size,
            'imgsz': self.imgsz,
            'workers': self.workers,
            'val_epoch': self.val_interval,
            'save_period': self.save_period,
            
            # Optimization
            'optimizer': self.optimizer,
            'lr0': self.lr0,
            'lrf': self.lrf,
            'momentum': self.momentum,
            'weight_decay': self.weight_decay,
            'warmup_epochs': self.warmup_epochs,
            'warmup_momentum': self.warmup_momentum,
            'warmup_bias_lr': self.warmup_bias_lr,
            'nbs': self.nbs,
            'cos_lr': self.cos_lr,
            
            # Event augmentation
            'flip': self.flip,
            'invert': self.invert,
            'suppress': self.suppress,
            'positive': self.positive,
            'zoom_out': self.zoom_out,
            
            # Standard augmentation
            'hsv_h': self.hsv_h,
            'hsv_s': self.hsv_s,
            'hsv_v': self.hsv_v,
            'degrees': self.degrees,
            'translate': self.translate,
            'scale': self.scale,
            'shear': self.shear,
            'perspective': self.perspective,
            'flipud': self.flipud,
            'mosaic': self.mosaic,
            'mixup': self.mixup,
            
            # Loss
            'box': self.box_loss_gain,
            'cls': self.cls_loss_gain,
            'dfl': self.dfl_loss_gain,
            
            # Distance
            'include_radar': self.include_radar,
            'process_distance': self.process_distance,
            'distance_from_head': self.distance_from_head,
            'min_dist': self.min_distance,
            'max_dist': self.max_distance,
            'nbins': self.nbins,
            'distance_loss_multiplier': self.distance_loss_multiplier,
            
            # Validation
            'conf': self.conf_threshold,
            'iou': self.iou_threshold,
            'max_det': self.max_det,
            'single_cls': self.single_cls,
            
            # Output
            'project': self.project,
            'name': self.exp_name,
            'exist_ok': self.exist_ok,
            
            # Device
            'device': self.device,
            
            # Other
            'verbose': self.verbose,
            'seed': self.seed,
            'deterministic': self.deterministic,
            
            # W&B
            'use_wandb': self.use_wandb,
            'wandb_project': self.wandb_project,
            'wandb_name': self.wandb_name or self.exp_name,
        }
    
    def __str__(self):
        """String representation with ReYOLOv8-specific info."""
        base_str = super().__str__()
        lines = base_str.split('\n')
        # Insert ReYOLOv8-specific info before the closing line
        insert_idx = len(lines) - 1
        lines.insert(insert_idx, f"Channels: {self.channels}")
        lines.insert(insert_idx + 1, f"Clip: length={self.clip_length}, stride={self.clip_stride}")
        lines.insert(insert_idx + 2, f"Event Augmentation: flip={self.flip}, suppress={self.suppress}")
        return '\n'.join(lines)


