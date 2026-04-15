"""
ReYOLOv8 Detection Experiment Template.
Object detection for event camera sequences without distance estimation.

Usage:
    python train.py -f experiments/templates/reyolov8_detection.py
"""

import os
import sys
from pathlib import Path

# Add parent directories to path

from nerve.training.experiments import ReYOLOv8Base


class Exp(ReYOLOv8Base):
    """
    ReYOLOv8 detection experiment configuration.
    For event camera (DVS/DAVIS) HDF5 sequence data.
    """
    
    def __init__(self):
        # Initialize with your data.yaml path
        super().__init__(
            dataset_path="/path/to/your/dataset",
            data_yaml="/path/to/your/data.yaml"
        )
        
        # Experiment Identity
        self.exp_name = "reyolov8_detection"
        
        # Model Configuration
        self.model = 'ReYOLOv8n.yaml'
        self.pretrained = None
        self.num_classes = 1
        
        # Sequence Parameters
        # Update channels to match your event representation:
        # - VTEI/mdes: bins channels (e.g., 10)
        # - voxel_grid/shist: 2*bins channels (e.g., 20)
        self.channels = 10          # e.g., 10 for VTEI with bins=10
        self.clip_length = 11
        self.clip_stride = 11
        
        # Training
        self.epochs = 100
        self.batch_size = 8         # Smaller for sequence data
        self.imgsz = 416
        self.workers = 4
        self.val_interval = 10      # Validate every N epochs (set to 1 for every epoch)
        
        # Task
        self.include_radar = False
        self.process_distance = False
        
        # Event Augmentation
        self.flip = 0.5
        self.invert = 0.0
        self.suppress = 0.0
        
        # Output
        self.project = 'runs/reyolov8'


# For standalone testing
if __name__ == '__main__':
    exp = Exp()
    print(exp)


