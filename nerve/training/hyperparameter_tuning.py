#!/usr/bin/env python3
"""
Hyperparameter Tuning Framework for YOLOX and ReYOLOv8 using Optuna.

This module provides an automated hyperparameter optimization system that integrates
with the existing PEGMA training infrastructure. It uses Optuna for efficient 
Bayesian optimization with support for:
- Automatic search space definition for YOLOX and ReYOLOv8 parameters
- Early stopping (pruning) of poor trials
- Parallel trial execution
- WandB integration for tracking
- SQLite/PostgreSQL storage for resumable studies

Usage:
    # YOLOX hyperparameter search
    python hyperparameter_tuning.py -f experiments/templates/yolox_hyperopt_template.py --model-type yolox --n-trials 50
    
    # ReYOLOv8 hyperparameter search
    python hyperparameter_tuning.py -f experiments/templates/reyolov8_hyperopt_template.py --model-type reyolov8 --n-trials 50
    
    # Auto-detect model type from experiment file
    python hyperparameter_tuning.py -f experiments/templates/my_reyolov8_sequence.py --n-trials 50
    
    # Resume an existing study
    python hyperparameter_tuning.py -f experiments/templates/my_reyolov8_sequence.py --study-name my_study --resume
    
    # Parallel execution with multi-GPU
    python hyperparameter_tuning.py -f experiments/templates/yolox_hyperopt_template.py --gpus 0,1 --trials-per-gpu 2 --n-jobs 4

Author: Generated for PEGMA Hyperparameter Optimization
"""

import argparse
import sys
import os
import subprocess
import json
import tempfile
import shutil
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
import importlib.util

# Add current directory to path


# GPU MANAGEMENT FOR MULTI-GPU PARALLEL TRIALS

class GPUAllocator:
    """
    Thread-safe GPU allocator for distributing trials across multiple GPUs.
    Uses round-robin assignment to balance load across available GPUs.
    """
    
    def __init__(self, gpu_ids: List[int] = None, trials_per_gpu: int = 1):
        """
        Initialize GPU allocator.
        
        Args:
            gpu_ids: List of GPU IDs to use. If None, auto-detect available GPUs.
            trials_per_gpu: Number of trials to run simultaneously per GPU.
        """
        self.gpu_ids = gpu_ids if gpu_ids is not None else self._detect_gpus()
        self.trials_per_gpu = trials_per_gpu
        self._lock = threading.Lock()
        self._gpu_usage = {gpu_id: 0 for gpu_id in self.gpu_ids}
        self._trial_gpu_map = {}  # Maps trial number to GPU ID
    
    def _detect_gpus(self) -> List[int]:
        """Auto-detect available NVIDIA GPUs."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                gpu_ids = [int(x.strip()) for x in result.stdout.strip().split('\n') if x.strip()]
                return gpu_ids if gpu_ids else [0]
        except Exception:
            pass
        
        # Fallback: check CUDA_VISIBLE_DEVICES
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if cuda_visible:
            return [int(x.strip()) for x in cuda_visible.split(',') if x.strip()]
        
        return [0]  # Default to GPU 0
    
    def allocate(self, trial_number: int) -> int:
        """
        Allocate a GPU for a trial using round-robin with load balancing.
        
        Args:
            trial_number: The trial number requesting a GPU.
        
        Returns:
            GPU ID to use for this trial.
        """
        with self._lock:
            # Find GPU with least usage
            min_usage = min(self._gpu_usage.values())
            for gpu_id in self.gpu_ids:
                if self._gpu_usage[gpu_id] == min_usage:
                    self._gpu_usage[gpu_id] += 1
                    self._trial_gpu_map[trial_number] = gpu_id
                    return gpu_id
            
            # Fallback: round-robin
            gpu_id = self.gpu_ids[trial_number % len(self.gpu_ids)]
            self._gpu_usage[gpu_id] += 1
            self._trial_gpu_map[trial_number] = gpu_id
            return gpu_id
    
    def release(self, trial_number: int):
        """
        Release GPU allocation for a completed trial.
        
        Args:
            trial_number: The trial number that completed.
        """
        with self._lock:
            if trial_number in self._trial_gpu_map:
                gpu_id = self._trial_gpu_map[trial_number]
                self._gpu_usage[gpu_id] = max(0, self._gpu_usage[gpu_id] - 1)
                del self._trial_gpu_map[trial_number]
    
    def get_status(self) -> dict:
        """Get current GPU allocation status."""
        with self._lock:
            return {
                'gpu_ids': self.gpu_ids,
                'usage': dict(self._gpu_usage),
                'active_trials': dict(self._trial_gpu_map),
            }
    
    @property
    def num_gpus(self) -> int:
        """Return number of available GPUs."""
        return len(self.gpu_ids)


# Global GPU allocator instance (set in run_optimization)
_gpu_allocator: Optional[GPUAllocator] = None

try:
    import optuna
    from optuna.trial import Trial
    from optuna.pruners import MedianPruner, HyperbandPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not installed. Install with: pip install optuna optuna-dashboard")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# SEARCH SPACE DEFINITIONS

def get_reyolov8_search_space(trial: Trial, config: dict, search_type: str = 'comprehensive') -> dict:
    """
    Define the hyperparameter search space for ReYOLOv8.
    
    Args:
        trial: Optuna trial object
        config: Base configuration from experiment file
        search_type: One of 'minimal', 'standard', 'comprehensive', 'augmentation', 'learning_rate'
    
    Returns:
        Dictionary of suggested hyperparameters
    """
    suggested = {}
    
    # Optimizer Choice (needed first for conditional lr0)
    # Select optimizer early so we can adjust lr0 range accordingly
    if search_type == 'comprehensive':
        suggested['optimizer'] = trial.suggest_categorical('optimizer', ['SGD', 'Adam', 'AdamW'])
    else:
        # Default to base config optimizer or SGD
        suggested['optimizer'] = config.get('optimizer', 'SGD')
    
    optimizer = suggested['optimizer']
    
    if search_type in ['minimal', 'standard', 'comprehensive', 'learning_rate']:
        # Learning Rate Parameters (OPTIMIZER-CONDITIONAL)
        # Different optimizers need different lr ranges:
        # - SGD: Can handle higher learning rates (0.001 - 0.1)
        # - Adam/AdamW: Need much lower learning rates (1e-5 - 0.01)
        if optimizer in ['Adam', 'AdamW']:
            suggested['lr0'] = trial.suggest_float('lr0', 1e-5, 0.08, log=True)
            suggested['lrf'] = trial.suggest_float('lrf', 0.001, 0.008, log=True)
        else:  # SGD
            suggested['lr0'] = trial.suggest_float('lr0', 1e-4, 0.1, log=True)
            suggested['lrf'] = trial.suggest_float('lrf', 0.001, 0.5, log=True)
        
        suggested['warmup_epochs'] = trial.suggest_int('warmup_epochs', 1, 5)
        
    if search_type in ['standard', 'comprehensive', 'learning_rate']:
        # Optimizer Parameters
        suggested['momentum'] = trial.suggest_float('momentum', 0.8, 0.99)
        suggested['weight_decay'] = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
        suggested['warmup_momentum'] = trial.suggest_float('warmup_momentum', 0.5, 0.95)
        suggested['warmup_bias_lr'] = trial.suggest_float('warmup_bias_lr', 0.0, 0.2)
        
    if search_type in ['standard', 'comprehensive']:
        # Loss Weights
        suggested['box'] = trial.suggest_float('box', 3.0, 15.0)
        suggested['cls'] = trial.suggest_float('cls', 0.1, 2.0)
        suggested['dfl'] = trial.suggest_float('dfl', 0.5, 3.0)
        
    if search_type in ['comprehensive', 'augmentation']:
        # Event-Specific Augmentation
        suggested['flip'] = trial.suggest_float('flip', 0.0, 0.5)
        suggested['suppress'] = trial.suggest_float('suppress', 0.0, 0.3)
        suggested['positive'] = trial.suggest_float('positive', 0.0, 0.3)
        suggested['invert'] = trial.suggest_float('invert', 0.0, 0.2)
        suggested['zoom_out'] = trial.suggest_float('zoom_out', 0.0, 0.4)
        
    if search_type == 'comprehensive':
        # Standard Augmentation (usually minimal for event cameras)
        suggested['mosaic'] = trial.suggest_float('mosaic', 0.0, 1.0)
        suggested['mixup'] = trial.suggest_float('mixup', 0.0, 0.3)
        suggested['scale'] = trial.suggest_float('scale', 0.0, 0.9)
        suggested['translate'] = trial.suggest_float('translate', 0.0, 0.3)
        
        # Sequence Parameters
        # Only tune if not fixed in base config
        if not config.get('_fixed_clip_length', False):
            suggested['clip_length'] = trial.suggest_int('clip_length', 5, 15, step=5)
            suggested['clip_stride'] = suggested['clip_length']  # Keep stride = length
    
    return suggested


def get_distance_search_space(trial: Trial, config: dict) -> dict:
    """
    Additional search space for distance estimation parameters.
    
    Args:
        trial: Optuna trial object
        config: Base configuration
    
    Returns:
        Dictionary of distance-related hyperparameters
    """
    suggested = {}
    
    if config.get('process_distance', False):
        suggested['distance_loss_multiplier'] = trial.suggest_float(
            'distance_loss_multiplier', 0.1, 5.0, log=True
        )
        suggested['nbins'] = trial.suggest_int('nbins', 50, 200, step=10)
    
    return suggested


def get_yolox_search_space(trial: Trial, config: dict, search_type: str = 'comprehensive') -> dict:
    """
    Define the hyperparameter search space for YOLOX.
    
    Args:
        trial: Optuna trial object
        config: Base configuration from experiment file
        search_type: One of 'minimal', 'standard', 'comprehensive', 'augmentation', 'learning_rate'
    
    Returns:
        Dictionary of suggested hyperparameters
    """
    suggested = {}
    
    if search_type in ['minimal', 'standard', 'comprehensive', 'learning_rate']:
        # Learning Rate Parameters
        # YOLOX uses basic_lr_per_img which is scaled by batch size
        suggested['basic_lr_per_img'] = trial.suggest_float('basic_lr_per_img', 1e-5, 1e-3, log=True)
        suggested['min_lr_ratio'] = trial.suggest_float('min_lr_ratio', 0.01, 0.2)
        suggested['warmup_epochs'] = trial.suggest_int('warmup_epochs', 0, 5)
        suggested['warmup_lr'] = trial.suggest_float('warmup_lr', 0.0, 0.001)
        
    if search_type in ['standard', 'comprehensive', 'learning_rate']:
        # Optimizer Parameters
        suggested['momentum'] = trial.suggest_float('momentum', 0.85, 0.99)
        suggested['weight_decay'] = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
        
    if search_type in ['standard', 'comprehensive']:
        # Training Configuration
        suggested['no_aug_epochs'] = trial.suggest_int('no_aug_epochs', 0, 10)
        
    if search_type in ['comprehensive', 'augmentation']:
        # Data Augmentation
        suggested['flip_prob'] = trial.suggest_float('flip_prob', 0.0, 1.0)
        suggested['hsv_prob'] = trial.suggest_float('hsv_prob', 0.0, 0.5)  # Usually low for event cameras
        
        # Mosaic and Mixup
        suggested['mosaic_prob'] = trial.suggest_float('mosaic_prob', 0.0, 1.0)
        suggested['mixup_prob'] = trial.suggest_float('mixup_prob', 0.0, 0.5)
        
        # Geometric transforms
        suggested['degrees'] = trial.suggest_float('degrees', 0.0, 15.0)
        suggested['translate'] = trial.suggest_float('translate', 0.0, 0.2)
        suggested['shear'] = trial.suggest_float('shear', 0.0, 5.0)
        
    if search_type == 'comprehensive':
        # Mosaic/Mixup Enable Flags
        suggested['use_mosaic'] = trial.suggest_categorical('use_mosaic', [True, False])
        suggested['enable_mixup'] = trial.suggest_categorical('enable_mixup', [True, False])
        
        # Model Architecture
        # Can optionally tune model scale
        # suggested['depth'] = trial.suggest_categorical('depth', [0.33, 0.67, 1.0])
        # suggested['width'] = trial.suggest_categorical('width', [0.25, 0.375, 0.5, 0.75])
        
        # Note: test_conf and nmsthre are kept fixed as they primarily affect
        # post-processing rather than model learning
    
    return suggested


# OBJECTIVE FUNCTION

class ReYOLOv8Objective:
    """
    Optuna objective function for ReYOLOv8 hyperparameter tuning.
    
    This class wraps the training process and extracts the optimization metric.
    Supports automatic multi-GPU distribution when gpu_allocator is provided.
    """
    
    def __init__(
        self,
        base_config: dict,
        exp_file: str,
        epochs_per_trial: int = 20,
        search_type: str = 'standard',
        metric: str = 'mAP50',
        output_dir: str = 'runs/hyperopt',
        use_wandb: bool = True,
        pruning: bool = True,
        custom_search_space: Optional[Callable] = None,
        gpu_allocator: Optional[GPUAllocator] = None,
    ):
        """
        Initialize the objective function.
        
        Args:
            base_config: Base configuration dictionary from experiment file
            exp_file: Path to experiment file
            epochs_per_trial: Number of epochs per trial (reduced for faster search)
            search_type: Type of search space ('minimal', 'standard', 'comprehensive', etc.)
            metric: Metric to optimize ('mAP50', 'mAP50-95', 'precision', 'recall')
            output_dir: Directory to save trial outputs
            use_wandb: Whether to log to wandb
            pruning: Whether to enable trial pruning
            custom_search_space: Optional custom search space function
            gpu_allocator: Optional GPUAllocator for multi-GPU parallel trials
        """
        self.base_config = base_config
        self.exp_file = exp_file
        self.epochs_per_trial = epochs_per_trial
        self.search_type = search_type
        self.metric = metric
        self.output_dir = Path(output_dir)
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.pruning = pruning
        self.custom_search_space = custom_search_space
        self.gpu_allocator = gpu_allocator
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Metric mapping for final extraction
        # Note: Column names in results.csv don't have the (B) suffix
        self.metric_keys = {
            'mAP50': 'metrics/mAP50',
            'mAP50-95': 'metrics/mAP50-95',
            'precision': 'metrics/precision',
            'recall': 'metrics/recall',
        }
    
    def __call__(self, trial: Trial) -> float:
        """
        Execute one trial of hyperparameter optimization.
        
        Args:
            trial: Optuna trial object
        
        Returns:
            Optimization metric value (to be maximized)
        """
        # Allocate GPU for this trial (if multi-GPU enabled)
        allocated_gpu = None
        if self.gpu_allocator is not None:
            allocated_gpu = self.gpu_allocator.allocate(trial.number)
        
        try:
            # Get suggested hyperparameters
            if self.custom_search_space:
                suggested = self.custom_search_space(trial, self.base_config)
            else:
                suggested = get_reyolov8_search_space(trial, self.base_config, self.search_type)
                
                # Add distance search space if enabled
                distance_params = get_distance_search_space(trial, self.base_config)
                suggested.update(distance_params)
            
            # Create trial-specific config
            trial_config = self.base_config.copy()
            trial_config.update(suggested)
            trial_config['epochs'] = self.epochs_per_trial
            trial_config['name'] = f"trial_{trial.number}"
            trial_config['project'] = str(self.output_dir)
            
            # Assign GPU if allocated
            if allocated_gpu is not None:
                trial_config['device'] = str(allocated_gpu)
            
            # Set wandb config for this trial
            if self.use_wandb:
                trial_config['use_wandb'] = True
                trial_config['wandb_project'] = f"hyperopt-{Path(self.exp_file).stem}"
                trial_config['wandb_name'] = f"trial_{trial.number}"
            else:
                trial_config['use_wandb'] = False
            
            # Log trial parameters
            gpu_info = f" [GPU {allocated_gpu}]" if allocated_gpu is not None else ""
            print(f"\n{'='*60}")
            print(f"Trial {trial.number}{gpu_info}: Testing hyperparameters")
            print(f"{'='*60}")
            for key, value in suggested.items():
                print(f"  {key}: {value}")
            print(f"{'='*60}\n")
            
            # Run training
            metric_value = self._run_training(trial, trial_config)
            
            # Report intermediate values for pruning
            if self.pruning and metric_value is not None:
                trial.report(metric_value, self.epochs_per_trial)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            return metric_value if metric_value is not None else 0.0
            
        except optuna.TrialPruned:
            raise
        except Exception as e:
            print(f"Trial {trial.number} failed with error: {e}")
            # Return a bad value instead of failing
            return 0.0
        finally:
            # Always release GPU allocation when trial completes
            if self.gpu_allocator is not None:
                self.gpu_allocator.release(trial.number)
    
    def _run_training(self, trial: Trial, config: dict) -> Optional[float]:
        """
        Run the actual training and extract the metric.
        
        Args:
            trial: Optuna trial object
            config: Trial configuration
        
        Returns:
            Final metric value or None if failed
        """
        REYOLOV8_DIR = Path(__file__).parent / 'reyolov8'
        
        # Build command line arguments
        cmd_args = [
            sys.executable,
            str(REYOLOV8_DIR / 'train.py'),
            '--data', str(config.get('data', config.get('data_yaml', ''))),
            '--model', str(config.get('model', str(REYOLOV8_DIR / 'ultralytics/models/v8/Recurrent/ReYOLOv8n.yaml'))),
            '--epochs', str(config.get('epochs', 20)),
            '--batch', str(config.get('batch', 16)),
            '--imgsz', str(config.get('imgsz', 384)),
            '--channels', str(config.get('channels', 10)),
            '--clip_length', str(config.get('clip_length', 11)),
            '--clip_stride', str(config.get('clip_stride', 11)),
            '--name', str(config.get('name', f'trial_{trial.number}')),
            '--project', str(config.get('project', str(self.output_dir))),
        ]
        
        # Add optional parameters
        # Note: We use CUDA_VISIBLE_DEVICES for GPU isolation, so always pass --device 0
        # The subprocess only sees its assigned GPU as device 0
        if config.get('device') is not None:
            cmd_args.extend(['--device', '0'])  # Always 0 since CUDA_VISIBLE_DEVICES handles mapping
        if config.get('workers'):
            cmd_args.extend(['--workers', str(config['workers'])])
        if config.get('select_channels') is not None:
            cmd_args.extend(['--select_channels', str(config['select_channels'])])
        if config.get('val_epoch'):
            cmd_args.extend(['--val_epoch', str(config['val_epoch'])])
        if config.get('seed') is not None:
            cmd_args.extend(['--seed', str(config['seed'])])
        if config.get('nbs'):
            cmd_args.extend(['--nbs', str(config['nbs'])])
        
        # Distance estimation
        if config.get('process_distance', False):
            cmd_args.extend([
                '--distance',
                '--nbins', str(config.get('nbins', 100)),
                '--min_dist', str(config.get('min_dist', 0.0)),
                '--max_dist', str(config.get('max_dist', 10.0)),
                '--dist_loss_mult', str(config.get('distance_loss_multiplier', 1.0)),
            ])
        
        # Learning rate parameters
        for param in ['lr0', 'lrf', 'momentum', 'weight_decay', 'warmup_epochs', 
                      'warmup_momentum', 'warmup_bias_lr']:
            if param in config and config[param] is not None:
                cmd_args.extend([f'--{param}', str(config[param])])
        
        # Loss parameters
        for param in ['box', 'cls', 'dfl']:
            if param in config and config[param] is not None:
                cmd_args.extend([f'--{param}', str(config[param])])
        
        # Validation parameters
        for param in ['conf', 'iou', 'max_det']:
            if param in config and config[param] is not None:
                cmd_args.extend([f'--{param}', str(config[param])])
        
        # Augmentation parameters
        for param in ['flip', 'suppress', 'invert', 'positive', 'zoom_out']:
            if param in config:
                cmd_args.extend([f'--{param}', str(config[param])])
        
        # Optimizer
        if config.get('optimizer'):
            cmd_args.extend(['--optimizer', str(config['optimizer'])])
        
        # Force exist_ok=True to prevent directory name auto-increment
        cmd_args.append('--exist_ok')
        
        # Setup environment with GPU isolation via CUDA_VISIBLE_DEVICES
        env = os.environ.copy()
        if config.get('device') is not None:
            # Set CUDA_VISIBLE_DEVICES to isolate this process to its assigned GPU
            env['CUDA_VISIBLE_DEVICES'] = str(config['device'])
        
        # Run training
        result = subprocess.run(
            cmd_args,
            cwd=str(REYOLOV8_DIR),
            env=env,  # Use modified environment with GPU isolation
            capture_output=False,  # Show output in real-time
        )
        
        if result.returncode != 0:
            print(f"Training failed with return code {result.returncode}")
            return None
        
        # Extract final metric from results CSV
        return self._extract_metric(config)
    
    def _extract_metric(self, config: dict) -> Optional[float]:
        """
        Extract the optimization metric from training results.
        
        Args:
            config: Trial configuration
        
        Returns:
            Final metric value or None
        """
        # Training runs in REYOLOV8_DIR, so results are relative to that directory
        # Use resolve() to get absolute path for reliability
        REYOLOV8_DIR = (Path(__file__).parent / 'reyolov8').resolve()
        results_dir = (REYOLOV8_DIR / config['project'] / config['name']).resolve()
        results_csv = results_dir / 'results.csv'
        
        print(f"Looking for results at: {results_csv}")
        
        if not results_csv.exists():
            print(f"Results file not found: {results_csv}")
            return None
        
        try:
            import pandas as pd
            df = pd.read_csv(results_csv, skipinitialspace=True)
            
            # Get the metric column name
            metric_col = self.metric_keys.get(self.metric, self.metric)
            
            # Find the column (handle whitespace in column names)
            for col in df.columns:
                if metric_col.strip() in col.strip():
                    # Return the best value (max for mAP metrics)
                    best_val = float(df[col].max())
                    print(f"Trial metric extraction: {self.metric} = {best_val:.4f}")
                    return best_val
            
            # Fallback: try mAP50
            for col in df.columns:
                if 'mAP50' in col and '95' not in col:
                    best_val = float(df[col].max())
                    print(f"Trial metric extraction (fallback mAP50): {best_val:.4f}")
                    return best_val
            
            print(f"Metric {self.metric} not found in results. Available columns: {list(df.columns)}")
            return None
            
        except Exception as e:
            print(f"Error extracting metric: {e}")
            return None


# YOLOX OBJECTIVE

class YOLOXObjective:
    """
    Optuna objective function for YOLOX hyperparameter tuning.
    
    This class wraps the YOLOX training process and extracts the optimization metric.
    Supports automatic multi-GPU distribution when gpu_allocator is provided.
    """
    
    def __init__(
        self,
        base_config: dict,
        exp_file: str,
        epochs_per_trial: int = 20,
        search_type: str = 'standard',
        metric: str = 'mAP50',
        output_dir: str = 'runs/hyperopt',
        use_wandb: bool = True,
        pruning: bool = True,
        custom_search_space: Optional[Callable] = None,
        gpu_allocator: Optional[GPUAllocator] = None,
    ):
        """
        Initialize the YOLOX objective function.
        
        Args:
            base_config: Base configuration dictionary from experiment file
            exp_file: Path to experiment file
            epochs_per_trial: Number of epochs per trial (reduced for faster search)
            search_type: Type of search space ('minimal', 'standard', 'comprehensive', etc.)
            metric: Metric to optimize ('mAP50', 'mAP50-95', 'AP50', 'AP50_95')
            output_dir: Directory to save trial outputs
            use_wandb: Whether to log to wandb
            pruning: Whether to enable trial pruning
            custom_search_space: Optional custom search space function
            gpu_allocator: Optional GPUAllocator for multi-GPU parallel trials
        """
        self.base_config = base_config
        self.exp_file = exp_file
        self.epochs_per_trial = epochs_per_trial
        self.search_type = search_type
        self.metric = metric
        self.output_dir = Path(output_dir)
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.pruning = pruning
        self.custom_search_space = custom_search_space
        self.gpu_allocator = gpu_allocator
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Metric mapping for YOLOX (uses AP50/AP50_95 naming)
        self.metric_keys = {
            'mAP50': 'AP50',
            'mAP50-95': 'AP50_95',
            'AP50': 'AP50',
            'AP50_95': 'AP50_95',
        }
    
    def __call__(self, trial: Trial) -> float:
        """
        Execute one trial of hyperparameter optimization.
        
        Args:
            trial: Optuna trial object
        
        Returns:
            Optimization metric value (to be maximized)
        """
        # Allocate GPU for this trial (if multi-GPU enabled)
        allocated_gpu = None
        if self.gpu_allocator is not None:
            allocated_gpu = self.gpu_allocator.allocate(trial.number)
        
        try:
            # Get suggested hyperparameters
            if self.custom_search_space:
                suggested = self.custom_search_space(trial, self.base_config)
            else:
                suggested = get_yolox_search_space(trial, self.base_config, self.search_type)
                
                # Add distance search space if enabled
                distance_params = get_distance_search_space(trial, self.base_config)
                suggested.update(distance_params)
            
            # Create trial-specific config
            trial_config = self.base_config.copy()
            trial_config.update(suggested)
            trial_config['epochs'] = self.epochs_per_trial
            trial_config['exp_name'] = f"trial_{trial.number}"
            trial_config['name'] = f"trial_{trial.number}"
            
            # Assign GPU if allocated
            if allocated_gpu is not None:
                trial_config['device'] = str(allocated_gpu)
            
            # Set wandb config for this trial
            if self.use_wandb:
                trial_config['use_wandb'] = True
                trial_config['wandb_project'] = f"hyperopt-yolox-{Path(self.exp_file).stem}"
                trial_config['wandb_name'] = f"trial_{trial.number}"
            else:
                trial_config['use_wandb'] = False
            
            # Log trial parameters
            gpu_info = f" [GPU {allocated_gpu}]" if allocated_gpu is not None else ""
            print(f"\n{'='*60}")
            print(f"Trial {trial.number}{gpu_info}: Testing YOLOX hyperparameters")
            print(f"{'='*60}")
            for key, value in suggested.items():
                print(f"  {key}: {value}")
            print(f"{'='*60}\n")
            
            # Run training
            metric_value = self._run_training(trial, trial_config)
            
            # Report intermediate values for pruning
            if self.pruning and metric_value is not None:
                trial.report(metric_value, self.epochs_per_trial)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            return metric_value if metric_value is not None else 0.0
            
        except optuna.TrialPruned:
            raise
        except Exception as e:
            print(f"Trial {trial.number} failed with error: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
        finally:
            # Always release GPU allocation when trial completes
            if self.gpu_allocator is not None:
                self.gpu_allocator.release(trial.number)
    
    def _run_training(self, trial: Trial, config: dict) -> Optional[float]:
        """
        Run YOLOX training and extract the metric.
        
        Args:
            trial: Optuna trial object
            config: Trial configuration
        
        Returns:
            Final metric value or None if failed
        """
        YOLOX_DIR = Path(__file__).parent / 'yoloX'
        
        # Get YOLOX exp file from config
        yolox_exp_file = config.get('yolox_exp_file')
        if not yolox_exp_file:
            print("Error: YOLOX requires yolox_exp_file in config")
            return None
        
        # Build command line arguments for YOLOX training
        train_args = [
            sys.executable,
            str(YOLOX_DIR / 'tools' / 'train.py'),
            '-f', str(yolox_exp_file),
            '-b', str(config.get('batch_size', config.get('batch', 16))),
            '-expn', config.get('exp_name', f'trial_{trial.number}'),
        ]
        
        # Device handled via CUDA_VISIBLE_DEVICES, pass -d 0
        if config.get('device') is not None:
            train_args.extend(['-d', '0'])
        
        # Build opts for YOLOX (key value pairs)
        opts = []
        
        # Training parameters
        if config.get('epochs'):
            opts.extend(['max_epoch', str(config['epochs'])])
        if config.get('eval_interval'):
            opts.extend(['eval_interval', str(config['eval_interval'])])
        if config.get('input_size'):
            input_size = config['input_size']
            if isinstance(input_size, (list, tuple)):
                opts.extend(['input_size', str(tuple(input_size))])
            else:
                opts.extend(['input_size', f"({input_size},{input_size})"])
        
        # Dataset configuration
        if config.get('dataset_path'):
            opts.extend(['data_dir', str(config['dataset_path'])])
        if config.get('train_ann'):
            opts.extend(['train_ann', str(config['train_ann'])])
        if config.get('val_ann'):
            opts.extend(['val_ann', str(config['val_ann'])])
        if config.get('source'):
            opts.extend(['source', str(config['source'])])
        
        # Learning rate parameters
        if 'basic_lr_per_img' in config:
            opts.extend(['basic_lr_per_img', str(config['basic_lr_per_img'])])
        if 'min_lr_ratio' in config:
            opts.extend(['min_lr_ratio', str(config['min_lr_ratio'])])
        if 'warmup_epochs' in config:
            opts.extend(['warmup_epochs', str(config['warmup_epochs'])])
        if 'warmup_lr' in config:
            opts.extend(['warmup_lr', str(config['warmup_lr'])])
        
        # Optimizer parameters
        if 'momentum' in config:
            opts.extend(['momentum', str(config['momentum'])])
        if 'weight_decay' in config:
            opts.extend(['weight_decay', str(config['weight_decay'])])
        
        # Augmentation parameters
        if 'flip_prob' in config:
            opts.extend(['flip_prob', str(config['flip_prob'])])
        if 'hsv_prob' in config:
            opts.extend(['hsv_prob', str(config['hsv_prob'])])
        if 'mosaic_prob' in config:
            opts.extend(['mosaic_prob', str(config['mosaic_prob'])])
        if 'mixup_prob' in config:
            opts.extend(['mixup_prob', str(config['mixup_prob'])])
        if 'use_mosaic' in config:
            opts.extend(['use_mosaic', str(config['use_mosaic'])])
        if 'enable_mixup' in config:
            opts.extend(['enable_mixup', str(config['enable_mixup'])])
        if 'degrees' in config:
            opts.extend(['degrees', str(config['degrees'])])
        if 'translate' in config:
            opts.extend(['translate', str(config['translate'])])
        if 'shear' in config:
            opts.extend(['shear', str(config['shear'])])
        if 'no_aug_epochs' in config:
            opts.extend(['no_aug_epochs', str(config['no_aug_epochs'])])
        
        # Evaluation parameters
        if 'test_conf' in config:
            opts.extend(['test_conf', str(config['test_conf'])])
        if 'nmsthre' in config:
            opts.extend(['nmsthre', str(config['nmsthre'])])
        if config.get('test_size'):
            test_size = config['test_size']
            if isinstance(test_size, (list, tuple)):
                opts.extend(['test_size', str(tuple(test_size))])
            else:
                opts.extend(['test_size', f"({test_size},{test_size})"])
        
        # Model parameters
        if 'depth' in config:
            opts.extend(['depth', str(config['depth'])])
        if 'width' in config:
            opts.extend(['width', str(config['width'])])
        if 'num_classes' in config:
            opts.extend(['num_classes', str(config['num_classes'])])
        
        # Distance estimation (if enabled)
        if config.get('process_distance', False):
            opts.extend(['include_distance', 'True'])
            opts.extend(['distance_from_head', str(config.get('distance_from_head', True))])
            if 'distance_loss_multiplier' in config:
                opts.extend(['distance_loss_multiplier', str(config['distance_loss_multiplier'])])
            if 'min_distance' in config:
                opts.extend(['min_distance', str(config['min_distance'])])
            if 'max_distance' in config:
                opts.extend(['max_distance', str(config['max_distance'])])
            if 'nbins' in config:
                opts.extend(['nbins', str(config['nbins'])])
        
        # Append opts to train_args
        if opts:
            train_args.extend(opts)
        
        # Print command summary
        print(f"\nExecuting YOLOX training with {len(opts)//2} config overrides:")
        for i in range(0, min(len(opts), 20), 2):  # Show first 10 overrides
            print(f"  {opts[i]} = {opts[i+1]}")
        if len(opts) > 20:
            print(f"  ... and {(len(opts)-20)//2} more")
        
        # Setup environment with GPU isolation
        env = os.environ.copy()
        pythonpath = env.get('PYTHONPATH', '')
        env['PYTHONPATH'] = f"{YOLOX_DIR}:{pythonpath}"
        
        if config.get('device') is not None:
            env['CUDA_VISIBLE_DEVICES'] = str(config['device'])
        
        # Run training
        result = subprocess.run(
            train_args,
            cwd=str(YOLOX_DIR),
            env=env,
            capture_output=False,
        )
        
        if result.returncode != 0:
            print(f"YOLOX training failed with return code {result.returncode}")
            return None
        
        # Extract final metric from training logs
        return self._extract_metric(config, yolox_exp_file)
    
    def _extract_metric(self, config: dict, yolox_exp_file: str) -> Optional[float]:
        """
        Extract the optimization metric from YOLOX training results.
        
        YOLOX stores results in checkpoints and logs. We extract from train_log.txt.
        
        Args:
            config: Trial configuration
            yolox_exp_file: Path to YOLOX exp file
        
        Returns:
            Final metric value or None
        """
        try:
            # Import the exp module to get output directory
            spec = importlib.util.spec_from_file_location("exp_module", yolox_exp_file)
            exp_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(exp_module)
            exp = exp_module.Exp()
            
            exp_name = config.get('exp_name', config.get('name', 'trial'))
            output_dir = Path(exp.output_dir) / exp_name
            
            train_log = output_dir / 'train_log.txt'
            
            print(f"Looking for YOLOX results at: {train_log}")
            
            if not train_log.exists():
                print(f"Train log not found: {train_log}")
                return None
            
            # Parse train_log.txt for AP values
            # YOLOX logs: val/COCOAP50 and val/COCOAP50_95
            best_ap50 = 0.0
            best_ap50_95 = 0.0
            
            with open(train_log, 'r') as f:
                content = f.read()
            
            # Extract AP values from log
            import re
            
            # Look for "best AP is X.XX" pattern (most common in YOLOX logs)
            # This appears at end of training: "Training of experiment is done and the best AP is 0.04"
            best_ap_is_pattern = r'best\s+AP\s+is\s+(\d+\.?\d*)'
            matches = re.findall(best_ap_is_pattern, content, re.IGNORECASE)
            if matches:
                best_ap50_95 = max(float(m) for m in matches if float(m) <= 100)
            
            # Also look for AP values in checkpoint saves with various formats
            ap_pattern = r'(?:curr_ap|best_ap|AP50_95|COCOAP50_95)["\s:=]+(\d+\.?\d*)'
            matches = re.findall(ap_pattern, content, re.IGNORECASE)
            if matches:
                potential = max(float(m) for m in matches if float(m) <= 100)
                if potential > best_ap50_95:
                    best_ap50_95 = potential
            
            # Look for AP50 values
            ap50_pattern = r'(?:AP50|COCOAP50)["\s:=]+(\d+\.?\d*)'
            matches = re.findall(ap50_pattern, content, re.IGNORECASE)
            if matches:
                # Filter for reasonable values (AP is 0-100 scale usually, but sometimes 0-1)
                vals = [float(m) for m in matches]
                # YOLOX uses 0-100 scale
                best_ap50 = max(v for v in vals if v <= 100) if vals else 0.0
            
            # Also try to read from results.csv which has exact metrics
            results_csv = output_dir / 'results.csv'
            if results_csv.exists():
                try:
                    import csv
                    with open(results_csv, 'r') as f:
                        reader = csv.DictReader(f)
                        rows = list(reader)
                        if rows:
                            # Get the last row (final epoch metrics)
                            last_row = rows[-1]
                            # Extract mAP50-95 and mAP50
                            if 'metrics/mAP50-95' in last_row:
                                val = last_row['metrics/mAP50-95']
                                if val:
                                    csv_ap50_95 = float(val)
                                    if csv_ap50_95 > best_ap50_95:
                                        best_ap50_95 = csv_ap50_95
                            if 'metrics/mAP50' in last_row:
                                val = last_row['metrics/mAP50']
                                if val:
                                    csv_ap50 = float(val)
                                    if csv_ap50 > best_ap50:
                                        best_ap50 = csv_ap50
                    print(f"Read metrics from results.csv: mAP50-95={best_ap50_95}, mAP50={best_ap50}")
                except Exception as csv_e:
                    print(f"Warning: Could not read results.csv: {csv_e}")
            
            # Determine which metric to return
            metric_key = self.metric_keys.get(self.metric, 'AP50_95')
            
            if 'AP50_95' in metric_key or 'mAP50-95' in self.metric:
                result = best_ap50_95
            else:
                result = best_ap50
            
            # YOLOX uses 0-100 scale sometimes, normalize to 0-1 for consistency
            if result > 1.0:
                result = result / 100.0
            
            print(f"Trial metric extraction: {self.metric} = {result:.4f}")
            return result
            
        except Exception as e:
            print(f"Error extracting YOLOX metric: {e}")
            import traceback
            traceback.print_exc()
            return None


# STUDY MANAGEMENT

def create_study(
    study_name: str,
    storage: Optional[str] = None,
    direction: str = 'maximize',
    pruner: Optional[str] = 'median',
    sampler: str = 'tpe',
    load_if_exists: bool = False,
) -> 'optuna.Study':
    """
    Create or load an Optuna study.
    
    Args:
        study_name: Name of the study
        storage: Database URL for persistent storage (e.g., 'sqlite:///hyperopt.db')
        direction: 'maximize' or 'minimize'
        pruner: Pruner type ('median', 'hyperband', None)
        sampler: Sampler type ('tpe', 'random', 'cmaes')
        load_if_exists: Whether to load existing study
    
    Returns:
        Optuna study object
    """
    # Configure pruner
    if pruner == 'median':
        pruner_obj = MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    elif pruner == 'hyperband':
        pruner_obj = HyperbandPruner()
    else:
        pruner_obj = None
    
    # Configure sampler
    if sampler == 'tpe':
        sampler_obj = TPESampler(seed=42)
    elif sampler == 'random':
        sampler_obj = optuna.samplers.RandomSampler(seed=42)
    elif sampler == 'cmaes':
        sampler_obj = optuna.samplers.CmaEsSampler(seed=42)
    else:
        sampler_obj = TPESampler(seed=42)
    
    # Create study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=direction,
        pruner=pruner_obj,
        sampler=sampler_obj,
        load_if_exists=load_if_exists,
    )
    
    return study


def run_optimization(
    exp_file: str,
    n_trials: int = 50,
    epochs_per_trial: int = 20,
    search_type: str = 'standard',
    metric: str = 'mAP50',
    study_name: Optional[str] = None,
    storage: Optional[str] = None,
    output_dir: str = 'runs/hyperopt',
    use_wandb: bool = True,
    pruning: bool = True,
    n_jobs: int = 1,
    resume: bool = False,
    timeout: Optional[int] = None,
    gpus: Optional[List[int]] = None,
    trials_per_gpu: int = 1,
    model_type: Optional[str] = None,
) -> 'optuna.Study':
    """
    Run hyperparameter optimization.
    
    Args:
        exp_file: Path to experiment configuration file
        n_trials: Number of trials to run
        epochs_per_trial: Epochs per trial (reduced for faster exploration)
        search_type: Search space type ('minimal', 'standard', 'comprehensive', etc.)
        metric: Metric to optimize ('mAP50', 'mAP50-95', etc.)
        study_name: Name of the study (for resuming)
        storage: Database URL for persistent storage
        output_dir: Directory to save outputs
        use_wandb: Whether to use wandb logging
        pruning: Whether to enable trial pruning
        n_jobs: Number of parallel jobs
        resume: Whether to resume existing study
        timeout: Total timeout in seconds
        gpus: List of GPU IDs to use (None for auto-detect)
        trials_per_gpu: Number of trials to run per GPU simultaneously
        model_type: Model type ('yolox', 'reyolov8', or None for auto-detect)
    
    Returns:
        Completed Optuna study
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is required. Install with: pip install optuna")
    
    # Import experiment configuration
    exp_file = Path(exp_file)
    if not exp_file.exists():
        raise FileNotFoundError(f"Experiment file not found: {exp_file}")
    
    spec = importlib.util.spec_from_file_location("exp_module", exp_file)
    exp_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exp_module)
    
    if not hasattr(exp_module, 'Exp'):
        raise AttributeError(f"Experiment file must contain 'Exp' class: {exp_file}")
    
    exp = exp_module.Exp()
    base_config = exp.to_dict()
    
    # Auto-detect model type if not specified
    if model_type is None:
        model_type = base_config.get('model_type', 'reyolov8').lower()
    model_type = model_type.lower()
    
    # Validate model type
    if model_type not in ['yolox', 'reyolov8']:
        raise ValueError(f"Unsupported model type: {model_type}. Supported: yolox, reyolov8")
    
    # Generate study name if not provided
    if study_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        study_name = f"{model_type}_hyperopt_{timestamp}"
    
    # Setup storage
    if storage is None:
        storage = f"sqlite:///{output_dir}/hyperopt_{study_name}.db"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Setup GPU allocator for multi-GPU parallel trials
    gpu_allocator = None
    if n_jobs > 1:
        gpu_allocator = GPUAllocator(gpu_ids=gpus, trials_per_gpu=trials_per_gpu)
        
        # Auto-adjust n_jobs if it exceeds what GPUs can handle
        max_parallel = gpu_allocator.num_gpus * trials_per_gpu
        if n_jobs > max_parallel:
            print(f"Note: Limiting n_jobs from {n_jobs} to {max_parallel} "
                  f"({gpu_allocator.num_gpus} GPUs x {trials_per_gpu} trials/GPU)")
            n_jobs = max_parallel
    
    print(f"\n{'='*70}")
    print(f"{model_type.upper()} Hyperparameter Optimization")
    print(f"{'='*70}")
    print(f"Model Type: {model_type}")
    print(f"Study Name: {study_name}")
    print(f"Experiment: {exp_file}")
    print(f"Search Type: {search_type}")
    print(f"Metric: {metric}")
    print(f"Trials: {n_trials}")
    print(f"Epochs per Trial: {epochs_per_trial}")
    print(f"Parallel Jobs: {n_jobs}")
    if gpu_allocator:
        print(f"GPUs: {gpu_allocator.gpu_ids} ({trials_per_gpu} trials/GPU)")
    print(f"Pruning: {pruning}")
    print(f"Storage: {storage}")
    print(f"{'='*70}\n")
    
    # Create appropriate objective function based on model type
    if model_type == 'yolox':
        objective = YOLOXObjective(
            base_config=base_config,
            exp_file=str(exp_file),
            epochs_per_trial=epochs_per_trial,
            search_type=search_type,
            metric=metric,
            output_dir=output_dir,
            use_wandb=use_wandb,
            pruning=pruning,
            gpu_allocator=gpu_allocator,
        )
    else:  # reyolov8
        objective = ReYOLOv8Objective(
            base_config=base_config,
            exp_file=str(exp_file),
            epochs_per_trial=epochs_per_trial,
            search_type=search_type,
            metric=metric,
            output_dir=output_dir,
            use_wandb=use_wandb,
            pruning=pruning,
            gpu_allocator=gpu_allocator,
        )
    
    # Create or load study
    study = create_study(
        study_name=study_name,
        storage=storage,
        direction='maximize',
        pruner='median' if pruning else None,
        load_if_exists=resume,
    )
    
    # Run optimization
    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=n_jobs,
        timeout=timeout,
        catch=(Exception,),  # Catch all exceptions to continue with other trials
        show_progress_bar=True,
    )
    
    # Print results
    print_optimization_results(study, output_dir)
    
    return study


def print_optimization_results(study: 'optuna.Study', output_dir: str):
    """Print and save optimization results."""
    print(f"\n{'='*70}")
    print("Optimization Results")
    print(f"{'='*70}")
    
    print(f"\nBest Trial: {study.best_trial.number}")
    print(f"Best Value ({study.direction.name}): {study.best_value:.4f}")
    
    print(f"\nBest Hyperparameters:")
    for key, value in study.best_params.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    
    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save best params as JSON
    best_params_file = output_dir / 'best_params.json'
    with open(best_params_file, 'w') as f:
        json.dump({
            'best_trial': study.best_trial.number,
            'best_value': study.best_value,
            'best_params': study.best_params,
        }, f, indent=2)
    print(f"\nBest parameters saved to: {best_params_file}")
    
    # Save trials dataframe
    try:
        trials_df = study.trials_dataframe()
        trials_csv = output_dir / 'trials_history.csv'
        trials_df.to_csv(trials_csv, index=False)
        print(f"Trial history saved to: {trials_csv}")
    except Exception as e:
        print(f"Could not save trials dataframe: {e}")
    
    # Generate visualization
    try:
        import optuna.visualization as vis
        
        # Parameter importance
        fig = vis.plot_param_importances(study)
        fig.write_html(str(output_dir / 'param_importance.html'))
        
        # Optimization history
        fig = vis.plot_optimization_history(study)
        fig.write_html(str(output_dir / 'optimization_history.html'))
        
        # Parameter relationships
        fig = vis.plot_parallel_coordinate(study)
        fig.write_html(str(output_dir / 'parallel_coordinate.html'))
        
        # Slice plot
        fig = vis.plot_slice(study)
        fig.write_html(str(output_dir / 'slice_plot.html'))
        
        print(f"Visualizations saved to: {output_dir}")
    except Exception as e:
        print(f"Could not generate visualizations: {e}")
        print("Install plotly for visualizations: pip install plotly")
    
    print(f"{'='*70}\n")


def generate_best_config(study: 'optuna.Study', base_exp_file: str, output_file: str):
    """
    Generate a new experiment file with the best hyperparameters.
    
    Args:
        study: Completed Optuna study
        base_exp_file: Path to base experiment file
        output_file: Path to save the optimized experiment file
    """
    # Read base experiment file
    with open(base_exp_file, 'r') as f:
        content = f.read()
    
    # Add best parameters as comments and modifications
    best_params = study.best_params
    
    param_lines = []
    for key, value in best_params.items():
        if isinstance(value, float):
            param_lines.append(f"        self.{key} = {value:.6f}  # Optimized")
        else:
            param_lines.append(f"        self.{key} = {repr(value)}  # Optimized")
    
    # Create header comment
    header = f'''"""
Optimized ReYOLOv8 Configuration
Generated by hyperparameter_tuning.py

Best Trial: {study.best_trial.number}
Best {study.direction.name}: {study.best_value:.4f}
"""

'''
    
    # Insert best params into the config
    # Find the __init__ method and add params after super().__init__
    optimized_content = header + content
    
    # Write optimized config
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(optimized_content)
        f.write(f"\n\n# ===== OPTIMIZED HYPERPARAMETERS =====\n")
        f.write(f"# Add these to your Exp.__init__() method:\n#\n")
        for line in param_lines:
            f.write(f"# {line}\n")
    
    print(f"Optimized config saved to: {output_path}")


# CLI INTERFACE

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Hyperparameter Tuning for YOLOX and ReYOLOv8',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # YOLOX hyperparameter search
    python hyperparameter_tuning.py -f experiments/templates/yolox_hyperopt_template.py --model-type yolox --n-trials 50
    
    # ReYOLOv8 hyperparameter search
    python hyperparameter_tuning.py -f experiments/templates/reyolov8_hyperopt_template.py --n-trials 50
    
    # Quick exploration with fewer epochs
    python hyperparameter_tuning.py -f experiments/templates/my_yolox_detection.py --model-type yolox --epochs-per-trial 10
    
    # Multi-GPU parallel search
    python hyperparameter_tuning.py -f experiments/templates/yolox_hyperopt_template.py --gpus 0,1,2,3 --trials-per-gpu 2 --n-jobs 8
    
    # Comprehensive search
    python hyperparameter_tuning.py -f experiments/templates/my_reyolov8_sequence.py --search-type comprehensive
    
    # Resume existing study
    python hyperparameter_tuning.py -f experiments/templates/my_reyolov8_sequence.py --study-name my_study --resume
    
    # Focus on learning rate optimization
    python hyperparameter_tuning.py -f experiments/templates/my_yolox_detection.py --model-type yolox --search-type learning_rate
        """
    )
    
    # Required arguments
    parser.add_argument(
        '-f', '--exp-file',
        type=str,
        required=True,
        help='Path to base experiment configuration file'
    )
    
    # Model type
    parser.add_argument(
        '--model-type',
        type=str,
        default=None,
        choices=['yolox', 'reyolov8'],
        help='Model type (auto-detected from experiment if not specified)'
    )
    
    # Optimization settings
    parser.add_argument(
        '--n-trials',
        type=int,
        default=50,
        help='Number of optimization trials (default: 50)'
    )
    parser.add_argument(
        '--epochs-per-trial',
        type=int,
        default=20,
        help='Training epochs per trial (default: 20)'
    )
    parser.add_argument(
        '--search-type',
        type=str,
        default='standard',
        choices=['minimal', 'standard', 'comprehensive', 'augmentation', 'learning_rate'],
        help='Search space type (default: standard)'
    )
    parser.add_argument(
        '--metric',
        type=str,
        default='mAP50',
        choices=['mAP50', 'mAP50-95', 'precision', 'recall'],
        help='Metric to optimize (default: mAP50)'
    )
    
    # Study management
    parser.add_argument(
        '--study-name',
        type=str,
        default=None,
        help='Name of the study (for resuming)'
    )
    parser.add_argument(
        '--storage',
        type=str,
        default=None,
        help='Database URL for storage (default: SQLite in output dir)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume existing study'
    )
    
    # Output
    parser.add_argument(
        '--output-dir',
        type=str,
        default='runs/hyperopt',
        help='Output directory (default: runs/hyperopt)'
    )
    
    # Execution
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=1,
        help='Number of parallel jobs (default: 1)'
    )
    parser.add_argument(
        '--gpus',
        type=str,
        default=None,
        help='Comma-separated GPU IDs to use (e.g., "0,1"). Auto-detects if not specified.'
    )
    parser.add_argument(
        '--trials-per-gpu',
        type=int,
        default=1,
        help='Number of trials to run per GPU simultaneously (default: 1)'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=None,
        help='Total timeout in seconds'
    )
    parser.add_argument(
        '--no-pruning',
        action='store_true',
        help='Disable trial pruning'
    )
    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help='Disable wandb logging'
    )
    
    # Generate config
    parser.add_argument(
        '--generate-config',
        type=str,
        default=None,
        help='Generate optimized config file from completed study'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    if not OPTUNA_AVAILABLE:
        print("Error: Optuna is required for hyperparameter tuning.")
        print("Install with: pip install optuna optuna-dashboard plotly")
        sys.exit(1)
    
    # Parse GPU IDs
    gpus = None
    if args.gpus:
        gpus = [int(x.strip()) for x in args.gpus.split(',')]
    
    # Run optimization
    study = run_optimization(
        exp_file=args.exp_file,
        n_trials=args.n_trials,
        epochs_per_trial=args.epochs_per_trial,
        search_type=args.search_type,
        metric=args.metric,
        study_name=args.study_name,
        storage=args.storage,
        output_dir=args.output_dir,
        use_wandb=not args.no_wandb,
        pruning=not args.no_pruning,
        n_jobs=args.n_jobs,
        resume=args.resume,
        timeout=args.timeout,
        gpus=gpus,
        trials_per_gpu=args.trials_per_gpu,
        model_type=args.model_type,
    )
    
    # Generate optimized config if requested
    if args.generate_config:
        generate_best_config(study, args.exp_file, args.generate_config)


if __name__ == '__main__':
    main()
