"""
RVT (Recurrent Vision Transformer) base configuration.
Extends BaseConfig with RVT-specific parameters for event camera data.

RVT uses PyTorch Lightning + Hydra configuration system.
This class bridges PEGMA's experiment system with RVT's config format.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports

from nerve.training.experiments.base import BaseConfig


class RVTBase(BaseConfig):
    """
    Base configuration class for RVT experiments.
    Inherits from BaseConfig and adds RVT-specific parameters.
    
    RVT (Recurrent Vision Transformer) is a state-of-the-art model for
    event-based object detection that uses a MaxVIT backbone with recurrent
    state for temporal processing.
    """
    
    MODEL_TYPE = 'rvt'
    
    def __init__(self, dataset_path: str = None, data_yaml: str = None):
        super().__init__(dataset_path, data_yaml)
        
        # RVT Model Architecture
        self.model_name = 'rnndet'  # RVT model name
        self.backbone = 'maxvit_rnn'  # Backbone type
        
        # Backbone configuration
        self.backbone_stages = [2, 2, 5, 2]  # MaxVIT stage depths
        self.backbone_dim = 64  # Base channel dimension
        self.backbone_in_channels = 20  # Input channels (e.g., 2*bins for stacked_histogram)
        
        # FPN configuration
        self.fpn_in_stages = [1, 2, 3]  # Backbone stages to use for FPN
        self.fpn_out_channels = 64  # FPN output channels
        self.fpn_depth = 0.67  # FPN depth: 0.33=Tiny/Small, 0.67=Base
        
        # Detection head configuration
        self.head_num_classes = 1  # Number of detection classes
        self.head_width = 0.375  # Head width multiplier
        self.head_in_channels = [64, 64, 64]  # Input channels for each FPN level
        self.head_strides = [8, 16, 32]  # Detection strides
        
        # RVT Sequence Parameters
        self.sequence_length = 11  # Number of frames per sequence
        self.ev_repr_name = 'stacked_histogram_dt=50_nbins=10'  # Event representation name
        self.resolution_hw = [240, 304]  # Input resolution [height, width]
        self.downsample_by_factor_2 = False  # Whether to downsample input
        self.only_load_end_labels = False  # Only load labels at sequence end
        
        # Training Parameters
        self.epochs = 100  # Alias for max_epochs
        self.max_epochs = 100
        self.max_steps = 400000
        self.batch_size = 8
        self.precision = 16  # Mixed precision (16 or 32)
        self.gradient_clip_val = 1.0
        
        # Optimization
        self.lr0 = 0.0002  # Learning rate (RVT default)
        self.weight_decay = 0.0
        
        # Learning rate scheduler
        self.use_lr_scheduler = True
        self.lr_scheduler_total_steps = None  # Will be set to max_steps if None
        self.lr_scheduler_pct_start = 0.005
        self.lr_scheduler_div_factor = 25
        self.lr_scheduler_final_div_factor = 10000
        
        # Data Augmentation
        self.prob_hflip = 0.5  # Horizontal flip probability
        self.rotate_prob = 0.0  # Rotation probability
        self.rotate_min_angle = 2
        self.rotate_max_angle = 6
        self.zoom_prob = 0.8  # Zoom probability
        self.zoom_in_weight = 8
        self.zoom_in_factor_min = 1.0
        self.zoom_in_factor_max = 1.5
        self.zoom_out_weight = 2
        self.zoom_out_factor_min = 1.0
        self.zoom_out_factor_max = 1.2
        
        # Distance Estimation
        self.include_radar = False
        self.process_distance = False
        self.distance_from_head = True
        self.min_distance = 0.0
        self.max_distance = 10.0
        self.nbins = 100  # Distance bins
        self.distance_loss_multiplier = 1.0
        
        # Channel Selection
        # Use this to ignore radar channel from a dataset that has it
        # None = use all channels, int = use first N channels
        self.select_channels = None
        
        # Hardware
        self.gpus = 0  # GPU index or list of indices
        self.workers = 6  # Training workers
        self.workers_eval = 2  # Evaluation workers
        self.dist_backend = 'nccl'  # Distributed backend
        
        # Validation
        self.val_check_interval = None  # Optional step-based validation
        self.check_val_every_n_epoch = 1  # Epoch-based validation
        self.limit_train_batches = 1.0  # Fraction of train batches
        self.limit_val_batches = 1.0  # Fraction of val batches
        
        # Postprocessing
        self.confidence_threshold = 0.01
        self.nms_threshold = 0.5
        
        # Logging
        self.ckpt_every_n_epochs = 1
        self.log_model_every_n_steps = 5000
        self.log_every_n_steps = 500
        self.compute_train_metrics = False
        
        # Output
        self.project = 'runs/rvt'
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]
        
        # Pretrained Weights / Checkpoints
        # For fine-tuning from pretrained weights (e.g., rvt-t.ckpt):
        self.pretrained = None  # Path to local checkpoint file (e.g., '/home/user/weights/rvt-t.ckpt')
        self.resume_only_weights = True  # True = load only weights (fine-tuning), False = resume full state
        
        # W&B artifact loading (alternative to local file):
        self.wandb_artifact_name = None  # W&B artifact name (e.g., 'user/project/checkpoint:v0')
        self.wandb_runpath = None  # Resume existing W&B run (e.g., 'user/project/run_id')
        
        # W&B
        self.wandb_group_name = 'pegma-rvt'
        self.wandb_project_name = 'RVT'
    
    def get_input_channels(self):
        """Calculate input channels based on event representation and channel selection."""
        # If select_channels is set, use that directly (overrides automatic calculation)
        if self.select_channels is not None:
            return self.select_channels
        
        # Extract bins from ev_repr_name if available
        if 'nbins=' in self.ev_repr_name:
            bins = int(self.ev_repr_name.split('nbins=')[1].split('_')[0])
        else:
            bins = 10
        
        # stacked_histogram produces 2*bins channels
        if 'histogram' in self.ev_repr_name or 'shist' in self.ev_repr_name:
            channels = 2 * bins
        elif 'voxel' in self.ev_repr_name:
            channels = 2 * bins
        else:
            channels = bins
        
        # Add radar channel if enabled
        if self.include_radar:
            channels += 1
        
        return channels
    
    def to_dict(self) -> dict:
        """Convert config to dictionary for RVT training bridge."""
        return {
            # Model type
            'model_type': self.MODEL_TYPE,
            
            # Data
            'data': self.data_yaml,
            'dataset_path': self.dataset_path,
            
            # Model architecture
            'model_name': self.model_name,
            'backbone': self.backbone,
            'backbone_stages': self.backbone_stages,
            'backbone_dim': self.backbone_dim,
            'backbone_in_channels': self.get_input_channels(),
            'fpn_in_stages': self.fpn_in_stages,
            'fpn_out_channels': self.fpn_out_channels,
            'fpn_depth': self.fpn_depth,
            'head_num_classes': self.head_num_classes,
            'head_width': self.head_width,
            
            # Sequence parameters
            'sequence_length': self.sequence_length,
            'ev_repr_name': self.ev_repr_name,
            'resolution_hw': self.resolution_hw,
            'downsample_by_factor_2': self.downsample_by_factor_2,
            'only_load_end_labels': self.only_load_end_labels,
            
            # Training
            'epochs': self.epochs,
            'max_epochs': self.max_epochs,
            'max_steps': self.max_steps,
            'batch_size': self.batch_size,
            'precision': self.precision,
            'gradient_clip_val': self.gradient_clip_val,
            
            # Optimization
            'lr0': self.lr0,
            'weight_decay': self.weight_decay,
            'use_lr_scheduler': self.use_lr_scheduler,
            'lr_scheduler_total_steps': self.lr_scheduler_total_steps or self.max_steps,
            'lr_scheduler_pct_start': self.lr_scheduler_pct_start,
            'lr_scheduler_div_factor': self.lr_scheduler_div_factor,
            'lr_scheduler_final_div_factor': self.lr_scheduler_final_div_factor,
            
            # Augmentation
            'prob_hflip': self.prob_hflip,
            'rotate_prob': self.rotate_prob,
            'rotate_min_angle': self.rotate_min_angle,
            'rotate_max_angle': self.rotate_max_angle,
            'zoom_prob': self.zoom_prob,
            'zoom_in_weight': self.zoom_in_weight,
            'zoom_in_factor_min': self.zoom_in_factor_min,
            'zoom_in_factor_max': self.zoom_in_factor_max,
            'zoom_out_weight': self.zoom_out_weight,
            'zoom_out_factor_min': self.zoom_out_factor_min,
            'zoom_out_factor_max': self.zoom_out_factor_max,
            
            # Distance estimation
            'include_radar': self.include_radar,
            'process_distance': self.process_distance,
            'distance_from_head': self.distance_from_head,
            'min_dist': self.min_distance,
            'max_dist': self.max_distance,
            'nbins': self.nbins,
            'distance_loss_multiplier': self.distance_loss_multiplier,
            
            # Channel selection
            'select_channels': self.select_channels,
            
            # Hardware
            'gpus': self.gpus,
            'workers': self.workers,
            'workers_eval': self.workers_eval,
            'dist_backend': self.dist_backend,
            
            # Validation
            'val_check_interval': self.val_check_interval,
            'check_val_every_n_epoch': self.check_val_every_n_epoch,
            'limit_train_batches': self.limit_train_batches,
            'limit_val_batches': self.limit_val_batches,
            
            # Postprocessing
            'confidence_threshold': self.confidence_threshold,
            'nms_threshold': self.nms_threshold,
            
            # Logging
            'ckpt_every_n_epochs': self.ckpt_every_n_epochs,
            'log_model_every_n_steps': self.log_model_every_n_steps,
            'log_every_n_steps': self.log_every_n_steps,
            'compute_train_metrics': self.compute_train_metrics,
            
            # Output
            'project': self.project,
            'name': self.exp_name,
            'exist_ok': self.exist_ok,
            
            # Device
            'device': self.device,
            
            # Other
            'verbose': self.verbose,
            'seed': self.seed,
            'deterministic': self.deterministic,
            
            # Pretrained weights
            'pretrained': self.pretrained,
            'resume_only_weights': self.resume_only_weights,
            'wandb_artifact_name': self.wandb_artifact_name,
            'wandb_runpath': self.wandb_runpath,
            
            # W&B
            'use_wandb': self.use_wandb,
            'wandb_project': self.wandb_project_name,
            'wandb_group': self.wandb_group_name,
            'wandb_name': self.wandb_name or self.exp_name,
        }
    
    def to_hydra_config(self) -> dict:
        """
        Convert PEGMA config to RVT's Hydra configuration structure.
        This creates a nested dictionary that can be written as YAML files
        for RVT's Hydra configuration system.
        
        The structure matches RVT's expected config format from:
        - config/general.yaml (reproduce, training, validation, batch_size, hardware, logging, wandb)
        - config/dataset/*.yaml (dataset-specific settings)
        - config/model/maxvit_yolox/default.yaml (model architecture)
        """
        in_channels = self.get_input_channels()
        
        return {
            # Top-level keys that match RVT's general.yaml structure
            'reproduce': {
                'seed_everything': self.seed if self.seed != 0 else None,
                'deterministic_flag': self.deterministic,
                'benchmark': False,
            },
            'training': {
                'precision': self.precision,
                'max_epochs': self.max_epochs,
                'max_steps': self.max_steps,
                'learning_rate': self.lr0,
                'weight_decay': self.weight_decay,
                'gradient_clip_val': self.gradient_clip_val,
                'limit_train_batches': self.limit_train_batches,
                'lr_scheduler': {
                    'use': self.use_lr_scheduler,
                    'total_steps': self.lr_scheduler_total_steps or self.max_steps,
                    'pct_start': self.lr_scheduler_pct_start,
                    'div_factor': self.lr_scheduler_div_factor,
                    'final_div_factor': self.lr_scheduler_final_div_factor,
                },
            },
            'validation': {
                'limit_val_batches': self.limit_val_batches,
                'val_check_interval': self.val_check_interval,
                'check_val_every_n_epoch': self.check_val_every_n_epoch,
            },
            'batch_size': {
                'train': self.batch_size,
                'eval': self.batch_size,
            },
            'hardware': {
                'num_workers': {
                    'train': self.workers,
                    'eval': self.workers_eval,
                },
                'gpus': self.gpus,
                'dist_backend': self.dist_backend,
            },
            'logging': {
                'ckpt_every_n_epochs': self.ckpt_every_n_epochs,
                'train': {
                    'metrics': {
                        'compute': self.compute_train_metrics,
                        'detection_metrics_every_n_steps': None,
                    },
                    'log_model_every_n_steps': self.log_model_every_n_steps,
                    'log_every_n_steps': self.log_every_n_steps,
                    'high_dim': {
                        'enable': True,
                        'every_n_steps': 5000,
                        'n_samples': 4,
                    },
                },
                'validation': {
                    'high_dim': {
                        'enable': True,
                        'every_n_epochs': 1,
                        'n_samples': 8,
                    },
                },
            },
            'wandb': {
                'wandb_runpath': self.wandb_runpath,
                'artifact_name': self.wandb_artifact_name,
                'artifact_local_file': self.pretrained,  # Local checkpoint path for fine-tuning
                'resume_only_weights': self.resume_only_weights,
                'group_name': self.wandb_group_name,
                'project_name': self.wandb_project_name,
            },
            'dataset': {
                'name': 'pegma',
                'path': self.dataset_path,
                'ev_repr_name': self.ev_repr_name,
                'sequence_length': self.sequence_length,
                'resolution_hw': self.resolution_hw,
                'downsample_by_factor_2': self.downsample_by_factor_2,
                'only_load_end_labels': self.only_load_end_labels,
                'has_distance': self.process_distance,  # Enable distance label loading
                'include_radar': self.include_radar,
                'select_channels': self.select_channels,  # Channel selection (None=all, int=first N)
                'train': {
                    'sampling': 'mixed',
                    'random': {
                        'weighted_sampling': False,
                    },
                    'mixed': {
                        'w_stream': 1,
                        'w_random': 1,
                    },
                },
                'eval': {
                    'sampling': 'stream',
                },
                'data_augmentation': {
                    'random': {
                        'prob_hflip': self.prob_hflip,
                        'rotate': {
                            'prob': self.rotate_prob,
                            'min_angle_deg': self.rotate_min_angle,
                            'max_angle_deg': self.rotate_max_angle,
                        },
                        'zoom': {
                            'prob': self.zoom_prob,
                            'zoom_in': {
                                'weight': self.zoom_in_weight,
                                'factor': {
                                    'min': self.zoom_in_factor_min,
                                    'max': self.zoom_in_factor_max,
                                },
                            },
                            'zoom_out': {
                                'weight': self.zoom_out_weight,
                                'factor': {
                                    'min': self.zoom_out_factor_min,
                                    'max': self.zoom_out_factor_max,
                                },
                            },
                        },
                    },
                    'stream': {
                        'prob_hflip': self.prob_hflip,
                        'rotate': {
                            'prob': self.rotate_prob,
                            'min_angle_deg': self.rotate_min_angle,
                            'max_angle_deg': self.rotate_max_angle,
                        },
                        'zoom': {
                            'prob': self.zoom_prob * 0.6,  # Lower zoom prob for streaming
                            'zoom_out': {
                                'factor': {
                                    'min': self.zoom_out_factor_min,
                                    'max': self.zoom_out_factor_max,
                                },
                            },
                        },
                    },
                },
            },
            'model': {
                'name': self.model_name,
                'backbone': {
                    'name': 'MaxViTRNN',  # RVT expects this exact name
                    'compile': {
                        'enable': False,
                        'args': {'mode': 'reduce-overhead'},
                    },
                    'input_channels': in_channels,
                    'enable_masking': False,
                    'partition_split_32': 2,
                    'embed_dim': self.backbone_dim,
                    'dim_multiplier': [1, 2, 4, 8],
                    'num_blocks': [1, 1, 1, 1],
                    'T_max_chrono_init': [4, 8, 16, 32],
                    'in_res_hw': self.resolution_hw,
                    'stem': {
                        'patch_size': 4,
                    },
                    'stage': {
                        'downsample': {
                            'type': 'patch',
                            'overlap': True,
                            'norm_affine': True,
                        },
                        'attention': {
                            'use_torch_mha': False,
                            'partition_size': [1, 1],  # Will be set by modifier
                            'dim_head': 32,
                            'attention_bias': True,
                            'mlp_activation': 'gelu',
                            'mlp_gated': False,
                            'mlp_bias': True,
                            'mlp_ratio': 4,
                            'drop_mlp': 0,
                            'drop_path': 0,
                            'ls_init_value': 1e-5,
                        },
                        'lstm': {
                            'dws_conv': False,
                            'dws_conv_only_hidden': True,
                            'dws_conv_kernel_size': 3,
                            'drop_cell_update': 0,
                        },
                    },
                },
                'fpn': {
                    'name': 'PAFPN',
                    'compile': {
                        'enable': False,
                        'args': {'mode': 'reduce-overhead'},
                    },
                    'depth': 0.67,
                    'in_stages': self.fpn_in_stages,
                    'depthwise': False,
                    'act': 'silu',
                },
                'head': {
                    'name': 'YoloX',
                    'compile': {
                        'enable': False,
                        'args': {'mode': 'reduce-overhead'},
                    },
                    'num_classes': self.head_num_classes,
                    'depthwise': False,
                    'act': 'silu',
                },
                'postprocess': {
                    'confidence_threshold': self.confidence_threshold,
                    'nms_threshold': self.nms_threshold,
                },
                # Distance estimation extension (PEGMA-specific)
                'distance': {
                    'enable': self.process_distance,
                    'nbins': self.nbins,
                    'min_dist': self.min_distance,
                    'max_dist': self.max_distance,
                    'loss_multiplier': self.distance_loss_multiplier,
                },
            },
        }
    
    def __str__(self):
        """String representation with RVT-specific info."""
        base_str = super().__str__()
        lines = base_str.split('\n')
        # Insert RVT-specific info before the closing line
        insert_idx = len(lines) - 1
        lines.insert(insert_idx, f"Model: {self.model_name} ({self.backbone})")
        lines.insert(insert_idx + 1, f"Sequence Length: {self.sequence_length}")
        lines.insert(insert_idx + 2, f"Resolution: {self.resolution_hw}")
        lines.insert(insert_idx + 3, f"Event Repr: {self.ev_repr_name}")
        if self.process_distance:
            lines.insert(insert_idx + 4, f"Distance: {self.nbins} bins [{self.min_distance}, {self.max_distance}]m")
        return '\n'.join(lines)

