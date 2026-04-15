"""
YOLOv8-specific base configuration.
Extends BaseConfig with YOLOv8-specific parameters.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports

from nerve.training.experiments.base import BaseConfig


class YOLOv8Base(BaseConfig):
    """
    Base configuration class for YOLOv8 experiments.
    Inherits from BaseConfig and adds YOLOv8-specific parameters.
    """
    
    MODEL_TYPE = 'yolov8'
    
    def __init__(self, dataset_path: str = None, data_yaml: str = None):
        super().__init__(dataset_path, data_yaml)
        
        # YOLOv8 Model Parameters
        self.model = 'yolov8n.yaml'  # Model architecture file
        self.pretrained = 'yolov8n.pt'  # Pretrained weights (or None)
        self.freeze = None  # Freeze first N layers (None = no freezing)
        
        # YOLOv8 Training Parameters
        self.imgsz = 416
        self.optimizer = 'SGD'  # 'SGD', 'Adam', 'AdamW'
        self.lrf = 0.01  # Final learning rate factor
        self.cos_lr = False  # Cosine LR scheduler
        self.warmup_bias_lr = 0.1
        self.warmup_momentum = 0.8
        self.nbs = 64  # Nominal batch size for weight decay adjustment
        
        self.val_interval = 1  # Validate every N epochs
        self.save_period = -1  # Save checkpoint every x epochs (-1 = disabled)
        
        # YOLOv8 Augmentation
        self.degrees = 0.0  # Rotation
        self.translate = 0.0  # Translation
        self.scale = 0.0  # Scale
        self.shear = 0.0  # Shear
        self.perspective = 0.0  # Perspective
        self.flipud = 0.0  # Flip up-down
        self.close_mosaic = 10  # Disable mosaic for last N epochs (Ultralytics default: 10)
        self.erasing = 0.0  # Random erasing probability (Ultralytics default: 0.4, harmful for event cameras)
        self.auto_augment = ''  # Auto augmentation policy (Ultralytics default: 'randaugment', harmful for event cameras)
        
        # YOLOv8 Loss
        self.box_loss_gain = 7.5
        self.cls_loss_gain = 0.5
        self.dfl_loss_gain = 1.5
        
        # Validation
        self.conf_threshold = 0.001
        self.iou_threshold = 0.6
        self.max_det = 300
        self.single_cls = True
        
        # Output
        self.project = 'runs/detect'
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]
    
    def to_dict(self) -> dict:
        """Convert config to dictionary for YOLOv8 training."""
        return {
            # Model type
            'model_type': self.MODEL_TYPE,
            
            # Data
            'data': self.data_yaml,
            'dataset_path': self.dataset_path,
            
            # Model
            'model': self.model,
            'pretrained': self.pretrained,
            'nc': self.num_classes,
            'freeze': self.freeze,
            
            # Training
            'epochs': self.epochs,
            'batch': self.batch_size,
            'imgsz': self.imgsz,
            'workers': self.workers,
            'val': True,  # Enable validation (ultralytics expects boolean, not interval)
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
            
            # Augmentation
            'hsv_h': self.hsv_h,
            'hsv_s': self.hsv_s,
            'hsv_v': self.hsv_v,
            'degrees': self.degrees,
            'translate': self.translate,
            'scale': self.scale,
            'shear': self.shear,
            'perspective': self.perspective,
            'fliplr': self.flip_prob,
            'flipud': self.flipud,
            'mosaic': self.mosaic,
            'mixup': self.mixup,
            'close_mosaic': self.close_mosaic,
            'erasing': self.erasing,
            'auto_augment': self.auto_augment,
            
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


