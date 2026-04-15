#!/usr/bin/env python3
# coding:utf-8
"""
YOLOX Distance Estimation experiment for PEGMA datasets.
Based on the original exp__dist__radar.py implementation.

Supports radar fusion and distance estimation.

Usage:
    # Distance estimation (from yoloX directory)
    python tools/train.py -f exps/pegma/yolox_nano_pegma_distance.py -d 0 -b 16
    
    # Or via unified trainer (from deep directory)
    python train.py -f experiments/templates/my_yolox_distance.py
"""

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist

from yolox.exp import Exp as BaseExp
from yolox.data import DataLoader
from yolox.data.data_augment_distance import TrainTransformWithDistance, ValTransformWithDistance
from yolox.evaluators import COCOEvaluator

try:
    from nerve.training.distance_coco_evaluator import Distance_COCO_Evaluator
    HAS_DISTANCE_EVALUATOR = True
except ImportError:
    HAS_DISTANCE_EVALUATOR = False

from nerve.training.yolox_exps.pegma_dataset import PEGMADataset

from yolox.data.datasets.mosaicdetection_distance import MosaicDetectionWithDistance
from yolox.data import YoloBatchSampler, DataPrefetcher, InfiniteSampler, worker_init_reset_seed


class Exp(BaseExp):
    """
    YOLOX distance estimation experiment for PEGMA datasets.
    Parameters matched to original exp__dist__radar.py implementation.
    """
    
    def __init__(self):
        super(Exp, self).__init__()
        
        # DATASET CONFIGURATION
        # Path to PEGMA distance dataset root
        self.data_dir = "/scratch-shared/tmp.8EGdXT6jjc/yolox_yolov8_distance_shist"
        
        # Source name (folder containing images: davis, prophesee, etc.)
        self.source = "davis"
        
        # Annotation files
        self.train_ann = f"{self.source}.json"
        self.val_ann = f"{self.source}.json"
        self.test_ann = f"{self.source}.json"
        
        # Number of classes
        self.num_classes = 1
        
        # MODEL CONFIGURATION
        # Model settings (matching old custom_tiny_exp.py base)
        self.depth = 0.33
        self.width = 0.375  # OLD: 0.375 (not 0.25 which is nano)
        self.input_size = (416, 416)
        self.test_size = (416, 416)
        self.random_size = (10, 20)
        self.mosaic_scale = (0.5, 1.5)
        
        # TRAINING CONFIGURATION
        # Matching old exp__dist__radar.py exactly
        self.max_epoch = 120  # OLD: 120 (different from base)
        self.warmup_epochs = 2  # OLD: 2
        self.no_aug_epochs = 60  # OLD: max_epoch // 2 = 60
        self.eval_interval = 2  # OLD: 2
        self.save_history_ckpt = True
        
        self.max_labels = 20  # Max number of labels per image
        self.data_num_workers = 8
        
        # Learning rate (from base class default)
        self.basic_lr_per_img = 0.01 / 64.0
        
        # AUGMENTATION CONFIG
        # Matching old exp__dist__radar.py exactly
        self.use_mosaic = True  # OLD: True (different from base detection)
        self.enable_mixup = False
        self.mosaic_prob = 1.0
        self.flip_prob = 1.0  # OLD: 1.0
        self.mixup_prob = 0
        self.hsv_prob = 0  # Disabled for event cameras
        
        # EVALUATION CONFIG
        # For visualization: use higher threshold to reduce false positives
        # For AP calculation: COCO evaluation uses all predictions sorted by confidence
        self.test_conf = 0.25  # Increased from 0.01 - reduces visual false positives
        self.nmsthre = 0.65
        
        # RADAR/DISTANCE SETTINGS
        # Matching old exp__dist__radar.py exactly
        self.use_radar = True  # OLD: True
        self.include_distance = True  # OLD: process_also_distance = True
        self.distance_from_head = True
        
        self.min_distance = 0.0
        self.max_distance = 10.0
        
        # Training settings (matching old exp__dist__radar.py)
        self.use_l1_loss = False  # OLD: False (different from base detection)
        self.distance_loss_multiplier = 3.0  # OLD: 3.0 (different from base)
        
        # Experiment name (used as subdirectory under output_dir)
        self.exp_name = "yolox_distance"
        
        self.output_dir = "runs"
    
    def get_model(self, sublinear=False):
        """
        Build YOLOX model with distance estimation head.
        Matching old custom_tiny_exp.py get_model() for distance mode.
        """
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        
        if getattr(self, "model", None) is None:
            from yolox.models import YOLOPAFPN, YOLOXHead
            
            # Try to import custom distance head
            try:
                from nerve.training.custom_yolo import Customized_YOLOX, YOLOX_custom_distance_head
                
                in_channels = [256, 512, 1024]
                backbone = YOLOPAFPN(
                    self.depth, self.width, in_channels=in_channels,
                    act=self.act
                )
                
                if self.distance_from_head:
                    head = YOLOX_custom_distance_head(
                        self.num_classes, self.width,
                        in_channels=in_channels, act=self.act
                    )
                    head.distance_loss_multiplier = self.distance_loss_multiplier
                else:
                    head = YOLOXHead(
                        self.num_classes, self.width,
                        in_channels=in_channels, act=self.act
                    )
                
                self.model = Customized_YOLOX(backbone, head, self.distance_from_head)
                
            except ImportError as e:
                print(f"Warning: Could not import custom distance modules: {e}")
                print("Falling back to standard YOLOX model (no distance estimation)")
                from yolox.models import YOLOX
                
                in_channels = [256, 512, 1024]
                backbone = YOLOPAFPN(
                    self.depth, self.width, in_channels=in_channels,
                    act=self.act
                )
                head = YOLOXHead(
                    self.num_classes, self.width,
                    in_channels=in_channels, act=self.act
                )
                self.model = YOLOX(backbone, head)
        
        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        self.model.head.use_l1 = self.use_l1_loss
        self.model.train()
        return self.model
    
    def get_dataset(self, cache: bool = False, cache_type: str = "ram"):
        """Get PEGMA training dataset with distance labels."""
        return PEGMADataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            name=self.source,
            split="train",
            img_size=self.input_size,
            preproc=TrainTransformWithDistance(
                max_labels=self.max_labels,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob,
                include_distance=self.include_distance,  # Ensure consistent 6-column output
            ),
            cache=cache,
            cache_type=cache_type,
            use_also_radar=self.use_radar,
            include_distance=self.include_distance,
            min_dist=self.min_distance,
            max_dist=self.max_distance,
        )
    
    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=None):
        """
        Get training dataloader with distance-aware MosaicDetection.
        
        This overrides the base class method to use MosaicDetectionWithDistance
        which preserves distance labels during mosaic/mixup augmentation.
        
        IMPORTANT: Respects self.use_mosaic setting (base class ignores it).
        """
        from yolox.utils import wait_for_the_master
        
        with wait_for_the_master():
            dataset = self.get_dataset(
                cache=cache_img is not None,
                cache_type=cache_img if cache_img is not None else "ram"
            )
        
        # IMPORTANT: Use self.use_mosaic to control whether mosaic is enabled
        # The base class uses `not no_aug` which ignores user's use_mosaic setting
        enable_mosaic = self.use_mosaic and not no_aug
        
        print(f"[get_data_loader] use_mosaic={self.use_mosaic}, no_aug={no_aug}, enable_mosaic={enable_mosaic}")
        print(f"[get_data_loader] mosaic_prob={self.mosaic_prob}, enable_mixup={self.enable_mixup}, mixup_prob={self.mixup_prob}")
        
        # Use distance-aware MosaicDetection
        dataset = MosaicDetectionWithDistance(
            dataset,
            mosaic=enable_mosaic,  # ← NOW respects self.use_mosaic!
            img_size=self.input_size,
            preproc=TrainTransformWithDistance(
                max_labels=self.max_labels,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob,
                include_distance=self.include_distance,  # Ensure consistent 6-column output
            ),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
            include_distance=self.include_distance,
        )
        
        self.dataset = dataset
        
        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
        
        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)
        
        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=enable_mosaic,  # ← Also use enable_mosaic here
        )
        
        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
        }
        dataloader_kwargs["batch_sampler"] = batch_sampler
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed
        
        train_loader = DataLoader(self.dataset, **dataloader_kwargs)
        
        return train_loader
    
    def get_eval_dataset(self, **kwargs):
        """Get PEGMA validation dataset with distance labels."""
        testdev = kwargs.get("testdev", False)
        legacy = kwargs.get("legacy", False)
        
        return PEGMADataset(
            data_dir=self.data_dir,
            json_file=self.test_ann if testdev else self.val_ann,
            name=self.source,
            split="test" if testdev else "val",
            img_size=self.test_size,
            preproc=ValTransformWithDistance(legacy=legacy, include_distance=self.include_distance),
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
        """
        Get evaluator for validation.
        
        Uses Distance_COCO_Evaluator for distance-specific metrics (ADE, DIR)
        if available, otherwise falls back to standard COCOEvaluator.
        """
        if HAS_DISTANCE_EVALUATOR and self.include_distance:
            return Distance_COCO_Evaluator(
                dataloader=self.get_eval_loader(batch_size, is_distributed, testdev, legacy),
                img_size=self.test_size,
                confthre=self.test_conf,
                nmsthre=self.nmsthre,
                num_classes=self.num_classes,
                testdev=testdev,
            )
        else:
            return COCOEvaluator(
                dataloader=self.get_eval_loader(batch_size, is_distributed, testdev, legacy),
                img_size=self.test_size,
                confthre=self.test_conf,
                nmsthre=self.nmsthre,
                num_classes=self.num_classes,
                testdev=testdev,
            )


# For standalone testing
if __name__ == '__main__':
    exp = Exp()
    print(f"Experiment: {exp.exp_name}")
    print(f"Data dir: {exp.data_dir}")
    print(f"Source: {exp.source}")
    print(f"Model: depth={exp.depth}, width={exp.width}")
    print(f"Training: max_epoch={exp.max_epoch}, warmup={exp.warmup_epochs}, no_aug={exp.no_aug_epochs}")
    print(f"Eval: test_conf={exp.test_conf}, nmsthre={exp.nmsthre}")
    print(f"Aug: flip={exp.flip_prob}, hsv={exp.hsv_prob}, mosaic={exp.mosaic_prob}, use_mosaic={exp.use_mosaic}")
    print(f"Use radar: {exp.use_radar}")
    print(f"Include distance: {exp.include_distance}")
    print(f"Distance loss multiplier: {exp.distance_loss_multiplier}")
    print(f"Use L1 loss: {exp.use_l1_loss}")
    
    # Test dataset loading
    try:
        dataset = exp.get_dataset()
        print(f"\n✓ Dataset loaded successfully: {len(dataset)} images")
    except Exception as e:
        print(f"\n✗ Dataset loading failed: {e}")
