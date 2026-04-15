"""
ReYOLOv8 Distance Estimation Experiment Template.
Object detection + distance estimation for event camera sequences with radar fusion.

Usage:
    python train.py -f experiments/templates/reyolov8_distance.py
"""

import os
import sys
from pathlib import Path

# Add parent directories to path

from nerve.training.experiments import ReYOLOv8Base


class Exp(ReYOLOv8Base):
    """
    ReYOLOv8 detection + distance estimation experiment configuration.
    For event camera sequences with radar fusion.
    
    Dataset must be created with radar source using:
        templates/reyolov8_distance.template.json
    """
    
    def __init__(self):
        # Initialize with your data.yaml path
        super().__init__(
            dataset_path="/scratch-shared/tmp.8EGdXT6jjc/reyolov8_dataset",
            data_yaml="/scratch-shared/tmp.8EGdXT6jjc/reyolov8_dataset/data.yaml"
        )
        
        # Experiment Identity
        self.exp_name = "reyolov8_distance"
        
        # Model Configuration
        self.model = 'ReYOLOv8n.yaml'
        self.pretrained = None
        self.num_classes = 1
        
        # Sequence Parameters
        # IMPORTANT: channels = event_channels + 1 (for radar)
        # e.g., VTEI bins=10 + 1 radar = 11 channels
        self.channels = 11          # 10 VTEI + 1 radar
        self.clip_length = 11
        self.clip_stride = 11
        
        # Training
        self.epochs = 100
        self.batch_size = 8
        self.imgsz = 416
        self.workers = 4
        self.val_interval = 10      # Validate every N epochs (set to 1 for every epoch)
        
        # Task: Distance Estimation
        self.include_radar = True
        self.process_distance = True
        self.distance_from_head = True
        
        # Distance Parameters
        # Match the max_dist used in dataset generation
        self.min_distance = 0.0
        self.max_distance = 10.0
        self.nbins = 100
        self.distance_loss_multiplier = 1.0
        
        # Event Augmentation
        self.flip = 0.5
        self.invert = 0.0
        self.suppress = 0.0
        
        # Output
        self.project = 'runs/reyolov8_distance'


# For standalone testing
if __name__ == '__main__':
    exp = Exp()
    print(exp)


