"""
YOLOX-specific base configuration.
Extends BaseConfig with YOLOX-specific parameters.
Parameters matched to original custom_tiny_exp.py implementation.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports

from nerve.training.experiments.base import BaseConfig


class YOLOXBase(BaseConfig):
    """
    Base configuration class for YOLOX experiments.
    Inherits from BaseConfig and adds YOLOX-specific parameters.
    Parameters matched to original custom_tiny_exp.py.
    
    If data_yaml is provided, source and annotation paths are automatically
    extracted from it. You can still override them manually if needed.
    """
    
    MODEL_TYPE = 'yolox'
    
    def __init__(self, dataset_path: str = None, data_yaml: str = None):
        super().__init__(dataset_path, data_yaml)
        
        # YOLOX Model Parameters
        # Matching old custom_tiny_exp.py
        self.depth = 0.33
        self.width = 0.375  # OLD: 0.375 (not 0.25 nano)
        self.act = 'silu'
        
        # YOLOX Training Parameters
        self.input_size = (416, 416)
        self.test_size = (416, 416)
        self.mosaic_scale = (0.5, 1.5)
        self.random_size = (10, 20)
        
        self.max_labels = 20  # Max number of labels per image (OLD: 20)
        self.eval_interval = 2  # Validate every N epochs (OLD: 2)
        self.save_history_ckpt = True
        self.warmup_epochs = 2  # OLD: 2
        self.no_aug_epochs = 0  # OLD: 0
        
        # Override base class defaults to match old implementation
        self.epochs = 100
        self.workers = 8  # OLD: data_num_workers = 8
        
        # YOLOX Augmentation
        # Matching old custom_tiny_exp.py
        self.use_mosaic = False
        self.enable_mixup = False
        self.mosaic_prob = 1.0
        self.flip_prob = 1.0  # OLD: 1.0 (not 0.5)
        self.mixup_prob = 0
        self.hsv_prob = 0  # Disabled for event cameras (maps to hsv_h)
        
        # YOLOX Evaluation
        # Higher test_conf for cleaner visualizations (reduces false positive clutter)
        # For AP calculation, predictions are sorted by confidence anyway
        self.test_conf = 0.25  # Changed from 0.01 to reduce visual FPs
        self.nmsthre = 0.65
        
        # YOLOX Loss
        self.use_l1_loss = True  # OLD: True
        
        # Annotation Paths
        # NOTE: These should be just the filename, not the full path.
        # The YOLOX dataset code constructs: data_dir/split/annotations/{ann_file}
        self.train_ann = "davis.json"
        self.val_ann = "davis.json"
        self.test_ann = "davis.json"
        
        # Image Source Folder
        # The folder name under data_dir/split/images/ containing the images
        # Must match the annotation file (e.g., "davis" for davis.json, "davis_radar" for davis_radar.json)
        self.source = "davis"
        
        # Output
        self.project = 'YOLOX_outputs'
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]
        
        # YOLOX Exp File
        # Path to a YOLOX-format experiment file (required for YOLOX training)
        self.yolox_exp_file = None
        
        # Auto-configure from data.yaml AFTER defaults are set
        # This allows data.yaml values to override the hardcoded defaults
        self._parse_data_yaml()
    
    def _parse_data_yaml(self):
        """
        Parse data.yaml to auto-configure source and annotation paths.
        
        Expected data.yaml format:
            path: /path/to/dataset
            train: train/images/davis_radar
            val: val/images/davis_radar
            test: test/images/davis_radar
            train_ann: train/annotations/davis_radar.json
            val_ann: val/annotations/davis_radar.json
            test_ann: test/annotations/davis_radar.json
            nc: 1
            names:
              0: person
        """
        if not self.data_yaml or not os.path.exists(self.data_yaml):
            return
        
        try:
            import yaml
            with open(self.data_yaml, 'r') as f:
                data = yaml.safe_load(f)
            
            # Extract dataset path if not already set
            if not self.dataset_path and data.get('path'):
                self.dataset_path = data['path']
            
            # Extract source from train path (last component: "train/images/davis_radar" -> "davis_radar")
            if data.get('train'):
                train_path = data['train']
                self.source = os.path.basename(train_path.rstrip('/'))
            
            # Extract annotation filenames (basename only: "train/annotations/davis_radar.json" -> "davis_radar.json")
            if data.get('train_ann'):
                self.train_ann = os.path.basename(data['train_ann'])
            if data.get('val_ann'):
                self.val_ann = os.path.basename(data['val_ann'])
            if data.get('test_ann'):
                self.test_ann = os.path.basename(data['test_ann'])
            
            # Extract class info
            if data.get('nc'):
                self.num_classes = data['nc']
            if data.get('names'):
                if isinstance(data['names'], dict):
                    self.class_names = [data['names'][i] for i in sorted(data['names'].keys())]
                elif isinstance(data['names'], list):
                    self.class_names = data['names']
            
            print(f"Auto-configured from data.yaml:")
            print(f"  source: {self.source}")
            print(f"  train_ann: {self.train_ann}")
            print(f"  num_classes: {self.num_classes}")
            
        except Exception as e:
            print(f"Warning: Could not parse data.yaml: {e}")
            print("Using default source and annotation paths.")
    
    def get_model(self):
        """
        Build and return the YOLOX model.
        Should be called after setting include_radar and process_distance.
        """
        from yoloX.yolox.models import YOLOPAFPN, YOLOXHead
        from custom_yolo import Customized_YOLOX, YOLOX_custom_distance_head
        import torch.nn as nn
        
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        
        in_channels = [256, 512, 1024]
        backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, act=self.act)
        
        if self.process_distance and self.distance_from_head:
            head = YOLOX_custom_distance_head(
                self.num_classes, self.width, 
                in_channels=in_channels, act=self.act
            )
            head.distance_loss_multiplier = self.distance_loss_multiplier
        else:
            head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, act=self.act)
        
        model = Customized_YOLOX(backbone, head, self.distance_from_head if self.process_distance else False)
        model.apply(init_yolo)
        model.head.initialize_biases(1e-2)
        model.head.use_l1 = self.use_l1_loss
        model.train()
        
        return model
    
    def to_dict(self) -> dict:
        """Convert config to dictionary for YOLOX training."""
        return {
            # Model type
            'model_type': self.MODEL_TYPE,
            
            # Data
            'dataset_path': self.dataset_path,
            'data_yaml': self.data_yaml,
            'train_ann': self.train_ann,
            'val_ann': self.val_ann,
            'test_ann': self.test_ann,
            'source': self.source,
            
            # Model
            'depth': self.depth,
            'width': self.width,
            'act': self.act,
            'num_classes': self.num_classes,
            
            # Training
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'input_size': self.input_size,
            'test_size': self.test_size,
            'max_labels': self.max_labels,
            'eval_interval': self.eval_interval,
            'workers': self.workers,
            'warmup_epochs': self.warmup_epochs,
            'no_aug_epochs': self.no_aug_epochs,
            'save_history_ckpt': self.save_history_ckpt,
            
            # Augmentation
            'use_mosaic': self.use_mosaic,
            'enable_mixup': self.enable_mixup,
            'mosaic_prob': self.mosaic_prob,
            'mixup_prob': self.mixup_prob,
            'mosaic_scale': self.mosaic_scale,
            'random_size': self.random_size,
            'flip_prob': self.flip_prob,
            'hsv_prob': self.hsv_prob,
            
            # Evaluation
            'test_conf': self.test_conf,
            'nmsthre': self.nmsthre,
            
            # Distance
            'include_radar': self.include_radar,
            'process_distance': self.process_distance,
            'distance_from_head': self.distance_from_head,
            'min_dist': self.min_distance,
            'max_dist': self.max_distance,
            'nbins': self.nbins,
            'distance_loss_multiplier': self.distance_loss_multiplier,
            
            # Loss
            'use_l1_loss': self.use_l1_loss,
            
            # Output
            'project': self.project,
            'exp_name': self.exp_name,
            'name': self.exp_name,  # Alias for YOLOX
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
            
            # YOLOX exp file (required for YOLOX training)
            'yolox_exp_file': self.yolox_exp_file,
        }
