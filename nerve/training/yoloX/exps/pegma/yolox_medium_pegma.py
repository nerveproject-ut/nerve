#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
YOLOX-M (Medium) experiment for PEGMA datasets.
Based on the original custom_tiny_exp.py implementation.

Model size: YOLOX-Medium (depth=0.67, width=0.75, ~25.3M params)
Comparable to RVT-Base (~18.5M params) for fair benchmarking.

Supports:
- Standard object detection
- Radar fusion (optional)
- Distance estimation (optional)

Usage:
    # Standard detection (from yoloX directory)
    python tools/train.py -f exps/pegma/yolox_medium_pegma.py -d 0 -b 8
    
    # Or via unified trainer (from deep directory)
    python train.py -f experiments/templates/my_yolox_detection.py
"""

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist

# Add YOLOX root to path
yolox_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(yolox_root))

from yolox.exp import Exp as BaseExp
from yolox.data import TrainTransform, ValTransform, DataLoader
from yolox.evaluators import COCOEvaluator

# Import PEGMA dataset
from exps.pegma.pegma_dataset import PEGMADataset


class Exp(BaseExp):
    """
    YOLOX-M (Medium) experiment for PEGMA datasets.
    
    Model size reference:
        - YOLOX-Nano:  depth=0.33, width=0.25  (~0.91M params)
        - YOLOX-Tiny:  depth=0.33, width=0.375 (~5.06M params)
        - YOLOX-S:     depth=0.33, width=0.50  (~9.0M params)
        - YOLOX-M:     depth=0.67, width=0.75  (~25.3M params)  <-- This config
        - YOLOX-L:     depth=1.00, width=1.00  (~54.2M params)
    """
    
    def __init__(self):
        super(Exp, self).__init__()
        
        # ==================== DATASET CONFIGURATION ====================
        # Path to PEGMA dataset root
        self.data_dir = "/scratch-shared/tmp.8EGdXT6jjc/yolox_yolov8_distance_shist"
        
        # Source name (folder containing images: davis, prophesee, etc.)
        self.source = "davis"
        
        # Annotation files (matching old format)
        self.train_ann = f"{self.source}.json"
        self.val_ann = f"{self.source}.json"
        self.test_ann = f"{self.source}.json"
        
        # Number of classes
        self.num_classes = 1
        
        # ==================== MODEL CONFIGURATION ====================
        # YOLOX-M (Medium) - comparable to RVT-Base for fair benchmarking
        self.depth = 0.67   # Medium depth
        self.width = 0.75   # Medium width
        self.input_size = (416, 416)
        self.test_size = (416, 416)
        self.random_size = (10, 20)
        self.mosaic_scale = (0.5, 1.5)
        
        # ==================== TRAINING CONFIGURATION ====================
        # Matching old custom_tiny_exp.py exactly
        self.max_epoch = 100
        self.warmup_epochs = 2  # OLD: 2 (not 3)
        self.no_aug_epochs = 0  # OLD: 0 (not 10)
        self.eval_interval = 2  # OLD: 2 (not 5)
        self.save_history_ckpt = True
        
        self.max_labels = 20  # Max number of labels per image
        self.data_num_workers = 8
        
        # Learning rate (from base class default)
        self.basic_lr_per_img = 0.01 / 64.0
        
        # ==================== AUGMENTATION CONFIG ====================
        # Matching old custom_tiny_exp.py exactly
        self.use_mosaic = False
        self.enable_mixup = False
        self.mosaic_prob = 1.0
        self.flip_prob = 1.0  # OLD: 1.0 (not 0.5)
        self.mixup_prob = 0
        self.hsv_prob = 0  # Disabled for event cameras
        
        # ==================== EVALUATION CONFIG ====================
        # For visualization: use higher threshold to reduce false positives
        # For AP calculation: COCO evaluation uses all predictions sorted by confidence
        self.test_conf = 0.25  # Increased from 0.01 - reduces visual false positives
        self.nmsthre = 0.65
        
        # ==================== RADAR/DISTANCE SETTINGS ====================
        # Set to True to fuse radar point cloud data
        self.use_radar = False
        
        # Distance estimation settings (set to True for distance experiments)
        self.include_distance = False
        self.min_distance = 0.0
        self.max_distance = 10.0
        
        # Training settings
        self.use_l1_loss = True
        self.distance_loss_multiplier = 1.0
        
        # Experiment name (used as subdirectory under output_dir)
        self.exp_name = "yolox_m_detection"
        
        # Output directory - use runs/ to match YOLOv8 structure
        # Final path will be: output_dir/exp_name/ (e.g., runs/yolox_m_detection/)
        # This is set relative to the deep/ directory where train.py is run
        deep_dir = Path(__file__).parent.parent.parent.parent
        self.output_dir = str(deep_dir / "runs")
    
    def get_model(self, sublinear=False):
        """Build YOLOX model (matching old custom_tiny_exp.py)."""
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        
        if getattr(self, "model", None) is None:
            from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
            
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(
                self.depth, self.width, in_channels=in_channels,
                act=self.act
            )
            head = YOLOXHead(
                self.num_classes, self.width, in_channels=in_channels,
                act=self.act
            )
            self.model = YOLOX(backbone, head)
        
        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        self.model.head.use_l1 = self.use_l1_loss
        self.model.train()
        return self.model
    
    def get_dataset(self, cache: bool = False, cache_type: str = "ram"):
        """Get PEGMA training dataset."""
        return PEGMADataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            name=self.source,
            split="train",
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=self.max_labels,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob
            ),
            cache=cache,
            cache_type=cache_type,
            use_also_radar=self.use_radar,
            include_distance=self.include_distance,
            min_dist=self.min_distance,
            max_dist=self.max_distance,
        )
    
    def get_eval_dataset(self, **kwargs):
        """Get PEGMA validation dataset."""
        testdev = kwargs.get("testdev", False)
        legacy = kwargs.get("legacy", False)
        
        return PEGMADataset(
            data_dir=self.data_dir,
            json_file=self.test_ann if testdev else self.val_ann,
            name=self.source,
            split="test" if testdev else "val",
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
            use_also_radar=self.use_radar,
            include_distance=self.include_distance,
            min_dist=self.min_distance,
            max_dist=self.max_distance,
        )
    
    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        """Get validation dataloader."""
        valdataset = self.get_eval_dataset(testdev=testdev, legacy=legacy)
        
        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(valdataset, shuffle=False)
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)
        
        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
            "batch_size": batch_size,
        }
        
        return DataLoader(valdataset, **dataloader_kwargs)
    
    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        """Get evaluator for validation."""
        return COCOEvaluator(
            dataloader=self.get_eval_loader(batch_size, is_distributed, testdev, legacy),
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )
    
    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img: str = None):
        """
        Get dataloader with PROPER use_mosaic support.
        
        The base class ignores self.use_mosaic - this override fixes that.
        """
        from yolox.data import (
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
            worker_init_reset_seed,
        )
        from yolox.utils import wait_for_the_master
        
        # Create dataset if needed
        if self.dataset is None:
            with wait_for_the_master():
                assert cache_img is None, \
                    "cache_img must be None if you didn't create self.dataset before launch"
                self.dataset = self.get_dataset(cache=False, cache_type=cache_img)
        
        # IMPORTANT: Use self.use_mosaic to control whether mosaic is enabled
        # The base class uses `not no_aug` which ignores user's use_mosaic setting
        enable_mosaic = self.use_mosaic and not no_aug
        
        print(f"[get_data_loader] use_mosaic={self.use_mosaic}, no_aug={no_aug}, enable_mosaic={enable_mosaic}")
        print(f"[get_data_loader] mosaic_prob={self.mosaic_prob}, enable_mixup={self.enable_mixup}, mixup_prob={self.mixup_prob}")
        
        self.dataset = MosaicDetection(
            dataset=self.dataset,
            mosaic=enable_mosaic,  # ? NOW respects self.use_mosaic!
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=self.max_labels,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob
            ),
            degrees=self.degrees if hasattr(self, 'degrees') else 10.0,
            translate=self.translate if hasattr(self, 'translate') else 0.1,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale if hasattr(self, 'mixup_scale') else (0.5, 1.5),
            shear=self.shear if hasattr(self, 'shear') else 2.0,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
        )
        
        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
        
        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if hasattr(self, 'seed') and self.seed else 0)
        
        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=enable_mosaic,  # ? Also use enable_mosaic here
        )
        
        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed
        
        train_loader = DataLoader(self.dataset, **dataloader_kwargs)
        
        return train_loader


# For standalone testing
if __name__ == '__main__':
    exp = Exp()
    print(f"Experiment: {exp.exp_name}")
    print(f"Data dir: {exp.data_dir}")
    print(f"Source: {exp.source}")
    print(f"Model: YOLOX-M (depth={exp.depth}, width={exp.width})")
    print(f"Training: max_epoch={exp.max_epoch}, warmup={exp.warmup_epochs}, no_aug={exp.no_aug_epochs}")
    print(f"Eval: test_conf={exp.test_conf}, nmsthre={exp.nmsthre}")
    print(f"Aug: flip={exp.flip_prob}, hsv={exp.hsv_prob}, mosaic={exp.mosaic_prob}")
    print(f"Use radar: {exp.use_radar}")
    print(f"Include distance: {exp.include_distance}")
    
    # Test dataset loading
    try:
        dataset = exp.get_dataset()
        print(f"\n? Dataset loaded successfully: {len(dataset)} images")
    except Exception as e:
        print(f"\n? Dataset loading failed: {e}")
