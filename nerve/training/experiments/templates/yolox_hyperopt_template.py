"""
YOLOX Hyperparameter Optimization Template.

This template is optimized for hyperparameter tuning with Optuna.
It uses reduced epochs and batch size for faster trials.

Usage:
    python hyperparameter_tuning.py -f experiments/templates/yolox_hyperopt_template.py --model-type yolox --n-trials 50
    
    # Multi-GPU parallel
    python hyperparameter_tuning.py -f experiments/templates/yolox_hyperopt_template.py --model-type yolox --gpus 0,1 --trials-per-gpu 2 --n-jobs 4
"""

import os
import sys
from pathlib import Path

# Add parent directories to path

from nerve.training.experiments import YOLOXBase


class Exp(YOLOXBase):
    """
    YOLOX hyperparameter tuning template.
    
    Parameters marked [TUNE] are optimized by Optuna.
    Parameters marked [FIXED] remain constant during tuning.
    """
    
    def __init__(self):
        # DATASET CONFIGURATION [FIXED]
        # Update these paths for your dataset
        super().__init__(
            dataset_path="/scratch-shared/tmp.8EGdXT6jjc/yolox_yolov8_distance_shist_davis_full_full",
            data_yaml="/scratch-shared/tmp.8EGdXT6jjc/yolox_yolov8_distance_shist_davis_full_full/data.yaml"
        )
        
        # EXPERIMENT IDENTITY [FIXED]
        self.exp_name = "yolox_hyperopt"
        
        # YOLOX experiment file - REQUIRED
        # This defines the YOLOX Exp class architecture
        # Set this to the path of your YOLOX exp file (e.g., 'yoloX/exps/pegma/yolox_medium_pegma.py')
        self.yolox_exp_file = None
        
        # MODEL ARCHITECTURE [FIXED]
        # These define the model size - typically not tuned
        self.depth = 0.67           # Model depth multiplier (0.33=nano)
        self.width = 0.75          # Model width multiplier
        self.act = 'silu'           # Activation function
        self.num_classes = 1        # Number of classes
        
        # TRAINING CONFIGURATION [FIXED FOR HYPEROPT]
        # Reduced for faster hyperparameter search
        self.epochs = 10            # Reduced for hyperopt (full training: 100+)
        self.batch_size = 144        # Adjust based on GPU memory
        self.input_size = (384, 288)  # Training input size (W, H)
        self.test_size = (384, 288)   # Evaluation input size (W, H)
        self.workers = 14            # Number of dataloader workers
        self.eval_interval = 4      # Validate every N epochs
        self.save_history_ckpt = False  # Disable to save disk space during hyperopt
        
        # LEARNING RATE PARAMETERS [TUNE]
        # These are primary tuning targets
        self.warmup_epochs = 2      # [TUNE] Warmup epochs (0-5)
        # Note: YOLOX uses basic_lr_per_img which is scaled by batch size
        # The actual LR = basic_lr_per_img * batch_size
        # basic_lr_per_img tuned range: 1e-5 to 1e-3 (log scale)
        
        # OPTIMIZER PARAMETERS [TUNE]
        # momentum tuned range: 0.85-0.99
        # weight_decay tuned range: 1e-5 to 1e-2 (log scale)
        
        # DATA AUGMENTATION [TUNE]
        # Event camera specific - usually minimal augmentation
        self.flip_prob = 1.0        # [TUNE] Horizontal flip probability (0-1)
        self.hsv_prob = 0           # [TUNE] HSV augmentation (usually 0 for events)
        self.use_mosaic = False     # [TUNE] Enable mosaic augmentation
        self.mosaic_prob = 1.0      # [TUNE] Mosaic probability (when enabled)
        self.mosaic_scale = (0.5, 1.5)  # Mosaic scale range
        self.enable_mixup = False   # [TUNE] Enable mixup augmentation
        self.mixup_prob = 0         # [TUNE] Mixup probability
        self.no_aug_epochs = 3      # [TUNE] Disable augmentation for last N epochs
        self.random_size = (10, 20) # Multi-scale training range
        self.max_labels = 20        # Maximum labels per image
        
        # TASK CONFIGURATION [FIXED]
        self.include_radar = True     # Use radar point cloud data
        self.process_distance = False  # Enable distance estimation
        
        # DISTANCE ESTIMATION [TUNE if process_distance=True]
        self.distance_from_head = True   # Predict distance from model head
        self.min_distance = 0.0          # Minimum distance in meters
        self.max_distance = 10.0         # Maximum distance in meters
        self.nbins = 100                 # [TUNE] Number of distance bins (50-200)
        self.distance_loss_multiplier = 1.0  # [TUNE] Weight for distance loss
        
        # LOSS CONFIGURATION [FIXED]
        self.use_l1_loss = True     # Use L1 loss for bbox regression
        
        # EVALUATION CONFIGURATION [FIXED]
        self.test_conf = 0.01       # [FIXED] Confidence threshold for NMS
        self.nmsthre = 0.7          # [FIXED] NMS IoU threshold
        
        # OUTPUT CONFIGURATION [FIXED]
        self.project = 'runs/yolox_hyperopt'
        self.exist_ok = True        # Allow overwriting for hyperopt trials
        
        # DEVICE CONFIGURATION [MANAGED BY HYPEROPT]
        self.device = ''            # Auto-assigned by hyperparameter tuning
        
        # OTHER SETTINGS [FIXED]
        self.verbose = True
        self.seed = 0               # Random seed for reproducibility
        self.deterministic = True   # Deterministic training
        
        # WEIGHTS & BIASES LOGGING [MANAGED BY HYPEROPT]
        self.use_wandb = True       # Controlled by hyperopt
        self.wandb_project = 'pegma-yolox-hyperopt'
        self.wandb_name = None      # Auto-generated by hyperopt


# For standalone testing
if __name__ == '__main__':
    exp = Exp()
    print(exp)
    print("\nConfig dict:")
    for k, v in exp.to_dict().items():
        print(f"  {k}: {v}")
