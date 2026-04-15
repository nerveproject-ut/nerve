"""
YOLOX Distance Estimation Experiment Template.
Object detection with distance estimation using radar data.

Usage:
    python train.py -f experiments/templates/yolox_distance.py
"""

import os
import sys
from pathlib import Path

# Add parent directories to path

from nerve.training.experiments import YOLOXBase


class Exp(YOLOXBase):
    """
    YOLOX detection + distance estimation experiment configuration.
    Requires radar data fused with DVS/event data.
    """
    
    def __init__(self):
        # Set these to your dataset (must contain a data.yaml describing the
        # train/val/test splits). The dataset must include radar fused with the
        # event/PNG frames - see PEGMA dataset generation docs.
        super().__init__(
            dataset_path="/path/to/your/dataset",
            data_yaml="/path/to/your/data.yaml"
        )
        
        # Experiment Identity
        self.exp_name = "yolox_distance"
        
        # Dataset Configuration (auto-configured from data.yaml)
        # source, train_ann, val_ann, test_ann are automatically extracted from data.yaml
        # Uncomment below to override if needed:
        # self.source = "prophesee"       # Image folder name
        # self.train_ann = "prophesee.json"  # Annotation filename
        
        # Model Configuration
        self.depth = 0.33
        self.width = 0.375
        self.num_classes = 1
        
        # Training
        self.epochs = 10
        self.batch_size = 80
        self.input_size = (384, 288)
        self.workers = 8
        
        # Task: Distance Estimation
        self.include_radar = True           # REQUIRED for distance
        self.process_distance = True        # Enable distance head
        self.distance_from_head = True      # Predict from model vs extract from radar
        
        # Distance Parameters
        self.min_distance = 0.0
        self.max_distance = 10.0            # Match your radar max_dist
        self.distance_loss_multiplier = 1.0
        
        # Augmentation
        self.flip_prob = 0.5
        self.use_mosaic = False
        self.enable_mixup = False


# For standalone testing
if __name__ == '__main__':
    exp = Exp()
    print(exp)


