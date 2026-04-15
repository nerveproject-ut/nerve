"""
YOLOv8 Detection Experiment Template.
Standard object detection without distance estimation.

Usage:
    python train.py -f experiments/templates/yolov8_detection.py
"""

import os
import sys
from pathlib import Path

# Add parent directories to path

from nerve.training.experiments import YOLOv8Base


class Exp(YOLOv8Base):
    """
    YOLOv8 detection experiment configuration.
    """
    
    def __init__(self):
        # Initialize with your data.yaml path
        super().__init__(
            dataset_path="/path/to/your/dataset",
            data_yaml="/path/to/your/data.yaml"
        )
        
        # Experiment Identity
        self.exp_name = "yolov8_detection"
        
        # Model Configuration
        self.model = 'yolov8n.yaml'         # nano model
        self.pretrained = 'yolov8n.pt'      # pretrained weights
        self.num_classes = 1
        
        # Training
        self.epochs = 100
        self.batch_size = 16
        self.imgsz = 416
        self.workers = 8
        
        # Task
        self.include_radar = False
        self.process_distance = False
        
        # Augmentation
        self.flip_prob = 0.5
        self.mosaic = 0.0
        self.mixup = 0.0
        
        # Output
        self.project = 'runs/detect'


# For standalone testing
if __name__ == '__main__':
    exp = Exp()
    print(exp)


