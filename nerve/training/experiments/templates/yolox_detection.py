"""
YOLOX Detection Experiment Template.
Standard object detection without distance estimation.

Usage:
    python train.py -f experiments/templates/yolox_detection.py
"""

import os
import sys
from pathlib import Path

# Add parent directories to path

from nerve.training.experiments import YOLOXBase


class Exp(YOLOXBase):
    """
    YOLOX detection experiment configuration.
    Modify parameters as needed for your experiment.
    """
    
    def __init__(self):
        # Initialize with your data.yaml path
        super().__init__(
            dataset_path="/path/to/your/dataset",
            data_yaml="/path/to/your/data.yaml"
        )
        
        # Experiment Identity
        self.exp_name = "yolox_detection"
        
        # Dataset Configuration (auto-configured from data.yaml)
        # source, train_ann, val_ann, test_ann are automatically extracted from data.yaml
        # Uncomment below to override if needed:
        # self.source = "prophesee"       # Image folder name
        # self.train_ann = "prophesee.json"  # Annotation filename
        
        # Model Configuration
        # Tiny model (default)
        self.depth = 0.33
        self.width = 0.375
        self.num_classes = 1
        
        # Training
        self.epochs = 100
        self.batch_size = 16
        self.input_size = (384, 288)
        self.workers = 8
        
        # Task
        self.include_radar = False
        self.process_distance = False
        
        # Augmentation
        self.flip_prob = 0.5
        self.use_mosaic = False
        self.enable_mixup = False


# For standalone testing
if __name__ == '__main__':
    exp = Exp()
    print(exp)
    print("\nConfig dict:")
    for k, v in exp.to_dict().items():
        print(f"  {k}: {v}")


