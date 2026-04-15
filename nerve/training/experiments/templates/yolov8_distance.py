"""
YOLOv8 Distance Estimation Experiment Template.
Object detection with distance estimation using radar data.

Usage:
    python train.py -f experiments/templates/yolov8_distance.py
"""

import os
import sys
from pathlib import Path

# Add parent directories to path

from nerve.training.experiments import YOLOv8Base


class Exp(YOLOv8Base):
    """
    YOLOv8 detection + distance estimation experiment configuration.
    Requires radar data fused with DVS/event data.
    """
    
    def __init__(self):
        # Initialize with your data.yaml path
        super().__init__(
            dataset_path="/scratch-shared/tmp.8EGdXT6jjc/yolox_yolov8_dataset",
            data_yaml="/scratch-shared/tmp.8EGdXT6jjc/yolox_yolov8_dataset/data.yaml"
        )
        
        # Experiment Identity
        self.exp_name = "yolov8_distance"
        
        # Model Configuration
        self.model = 'yolov8n.yaml'
        self.pretrained = 'yolov8n.pt'
        self.num_classes = 1
        
        # Training
        self.epochs = 10
        self.batch_size = 80
        self.imgsz = 416
        self.workers = 8
        
        # Task: Distance Estimation
        self.include_radar = True
        self.process_distance = True
        self.distance_from_head = True
        
        # Distance Parameters
        self.min_distance = 0.0
        self.max_distance = 10.0
        self.nbins = 100
        self.distance_loss_multiplier = 1.0
        
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


