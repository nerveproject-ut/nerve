"""
ReYOLOv8 Hyperparameter Tuning Template.

This template is designed for use with the hyperparameter_tuning.py framework.
It contains sensible defaults and comments about which parameters are good
candidates for optimization.

Usage:
    # Standard hyperparameter search (learning rates, loss weights)
    python hyperparameter_tuning.py -f experiments/templates/reyolov8_hyperopt_template.py --n-trials 50

    # Quick exploration with fewer epochs
    python hyperparameter_tuning.py -f experiments/templates/reyolov8_hyperopt_template.py --n-trials 100 --epochs-per-trial 5

    # Comprehensive search including augmentation
    python hyperparameter_tuning.py -f experiments/templates/reyolov8_hyperopt_template.py --search-type comprehensive

    # Focus on event augmentation only
    python hyperparameter_tuning.py -f experiments/templates/reyolov8_hyperopt_template.py --search-type augmentation
"""

import os
import sys
from pathlib import Path


from nerve.training.experiments import ReYOLOv8Base


class Exp(ReYOLOv8Base):
    """
    ReYOLOv8 configuration template for hyperparameter tuning.
    
    Parameters marked with [TUNE] are good candidates for optimization.
    Parameters marked with [FIXED] should typically remain constant.
    """
    
    def __init__(self):
        # DATASET CONFIGURATION [FIXED]
        # Set these to your actual dataset paths
        super().__init__(
            # TODO: Update these paths for your dataset
            dataset_path="/scratch-shared/tmp.8EGdXT6jjc/reyolo_distance_vtei_davis_full_full",
            data_yaml="/scratch-shared/tmp.8EGdXT6jjc/reyolo_distance_vtei_davis_full_full/data.yaml"
        )
        
        # EXPERIMENT IDENTITY [FIXED]
        # Will be overwritten during hyperparameter tuning
        self.exp_name = "reyolov8_hyperopt_medium"
        
        # MODEL CONFIGURATION [FIXED]
        # Model architecture typically fixed during HP tuning
        self.model = 'ReYOLOv8n.yaml'  # Options: ReYOLOv8n, ReYOLOv8s, ReYOLOv8m
        self.pretrained = None
        self.num_classes = 1
        
        # SEQUENCE PARAMETERS [PARTIALLY TUNABLE]
        # clip_length/stride can be tuned in 'comprehensive' mode
        self.channels = 6              # Match your data
        self.clip_length = 5           # [TUNE in comprehensive mode]
        self.clip_stride = 5           # [TUNE in comprehensive mode]
        self.select_channels = None       # Channel selection
        
        # TRAINING CONFIGURATION [FIXED during HP search]
        # epochs_per_trial is set by the tuning framework
        self.epochs = 15              # Will be overwritten by epochs_per_trial
        self.batch_size = 160            # Adjust based on GPU memory
        self.imgsz = 384
        self.workers = 12
        self.val_interval = 6          # Validate every N epochs
        self.save_period = -1
        
        # TASK CONFIGURATION [FIXED]
        self.include_radar = True
        self.process_distance = False
        
        # OPTIMIZATION [TUNE - Primary targets]
        # These are the most important hyperparameters to tune
        self.optimizer = 'SGD'         # [TUNE in comprehensive]: SGD, Adam, AdamW
        self.lr0 = 0.01                # [TUNE]: Initial learning rate (1e-4 to 0.5)
        self.lrf = 0.01                # [TUNE]: Final LR factor (0.001 to 0.5)
        self.momentum = 0.937          # [TUNE]: SGD momentum (0.8 to 0.99)
        self.weight_decay = 0.0005     # [TUNE]: Weight decay (1e-6 to 1e-2)
        self.warmup_epochs = 3         # [TUNE]: Warmup epochs (0 to 5)
        self.warmup_momentum = 0.8     # [TUNE]: Warmup momentum (0.5 to 0.95)
        self.warmup_bias_lr = 0.1      # [TUNE]: Warmup bias LR (0 to 0.2)
        self.nbs = 64                  # Nominal batch size
        
        # EVENT-SPECIFIC AUGMENTATION [TUNE - Important for event cameras]
        # These augmentations are specifically designed for event camera data
        self.flip = 0.5                # [TUNE]: Horizontal flip probability
        self.invert = 0.0              # [TUNE]: Polarity inversion
        self.suppress = 0.0            # [TUNE]: Random polarity suppression
        self.positive = 0.0            # [TUNE]: Positive polarity suppression
        self.zoom_out = 0.0            # [TUNE]: Zoom out probability
        
        # STANDARD AUGMENTATION [TUNE in comprehensive mode]
        # Usually kept minimal for event cameras
        self.hsv_h = 0.0               # HSV augmentation (usually 0 for events)
        self.hsv_s = 0.0
        self.hsv_v = 0.0
        self.degrees = 0.0             # Rotation
        self.translate = 0.1           # [TUNE]: Translation
        self.scale = 0.5               # [TUNE]: Scale
        self.shear = 0.0               # Shear
        self.perspective = 0.0         # Perspective
        self.flipud = 0.0              # Vertical flip
        self.mosaic = 0.0              # [TUNE]: Mosaic augmentation
        self.mixup = 0.0               # [TUNE]: Mixup augmentation
        
        # LOSS CONFIGURATION [TUNE - Important]
        # Loss weights significantly affect training dynamics
        self.box_loss_gain = 7.5       # [TUNE]: Box loss gain (3 to 15)
        self.cls_loss_gain = 0.5       # [TUNE]: Classification loss gain (0.1 to 2)
        self.dfl_loss_gain = 1.5       # [TUNE]: DFL loss gain (0.5 to 3)
        
        # VALIDATION CONFIGURATION [TUNE in comprehensive mode]
        self.conf_threshold = 0.01    # [FIXED]: Confidence threshold
        self.iou_threshold = 0.7       # [FIXED]: IoU threshold for NMS
        self.max_det = 300
        self.single_cls = True
        
        # OUTPUT CONFIGURATION [FIXED]
        self.project = 'runs/hyperopt'
        self.exist_ok = True
        
        # DEVICE CONFIGURATION [FIXED]
        self.device = '0'
        
        # REPRODUCIBILITY [FIXED]
        self.verbose = False  # Less verbose during HP search
        self.seed = 0
        self.deterministic = True
        
        # LOGGING [FIXED during HP search]
        # wandb is managed by the tuning framework
        self.use_wandb = True
        self.wandb_project = 'reyolov8-hyperopt'
        self.wandb_name = None


# For standalone testing
if __name__ == '__main__':
    exp = Exp()
    print(exp)
    print("\nConfig dict:")
    for k, v in exp.to_dict().items():
        print(f"  {k}: {v}")
