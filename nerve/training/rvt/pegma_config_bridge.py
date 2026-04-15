"""
PEGMA to RVT Configuration Bridge.

This module provides functions to translate PEGMA experiment configurations
to RVT's Hydra-based configuration system. It generates the necessary YAML
files for RVT training while maintaining compatibility with PEGMA's dataset
structure and distance estimation features.

Usage:
    from nerve.training.rvt.pegma_config_bridge import generate_rvt_configs
    
    config = exp.to_hydra_config()
    config_dir = generate_rvt_configs(config, output_dir='/path/to/configs')
"""

import os
import yaml
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional


def generate_rvt_configs(pegma_config: Dict[str, Any], 
                         output_dir: Optional[str] = None,
                         config_name: str = 'pegma') -> str:
    """
    Generate RVT Hydra configuration files from PEGMA config.
    
    Args:
        pegma_config: Configuration dictionary from RVTBase.to_hydra_config()
        output_dir: Directory to write config files (creates temp dir if None)
        config_name: Name prefix for configuration files
        
    Returns:
        Path to the config directory containing generated YAML files
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix='rvt_config_')
    
    config_dir = Path(output_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories matching RVT's config structure
    (config_dir / 'dataset').mkdir(exist_ok=True)
    (config_dir / 'model').mkdir(exist_ok=True)
    (config_dir / 'experiment' / config_name).mkdir(parents=True, exist_ok=True)
    
    # Write general.yaml
    general_config = pegma_config.get('general', {})
    with open(config_dir / 'general.yaml', 'w') as f:
        yaml.dump(general_config, f, default_flow_style=False)
    
    # Write dataset configuration
    dataset_config = pegma_config.get('dataset', {})
    
    # Write base dataset config
    dataset_base = {
        'name': '???',
        'path': '???',
        'train': dataset_config.get('train', {}),
        'eval': dataset_config.get('eval', {}),
        'data_augmentation': dataset_config.get('data_augmentation', {}),
    }
    with open(config_dir / 'dataset' / 'base.yaml', 'w') as f:
        yaml.dump(dataset_base, f, default_flow_style=False)
    
    # Write PEGMA-specific dataset config
    pegma_dataset = {
        'defaults': ['base'],
        'name': 'pegma',
        'path': dataset_config.get('path', '???'),
        'ev_repr_name': dataset_config.get('ev_repr_name', 'stacked_histogram_dt=50_nbins=10'),
        'sequence_length': dataset_config.get('sequence_length', 11),
        'resolution_hw': dataset_config.get('resolution_hw', [240, 304]),
        'downsample_by_factor_2': dataset_config.get('downsample_by_factor_2', False),
        'only_load_end_labels': dataset_config.get('only_load_end_labels', False),
    }
    with open(config_dir / 'dataset' / f'{config_name}.yaml', 'w') as f:
        yaml.dump(pegma_dataset, f, default_flow_style=False)
    
    # Write model configuration
    model_config = pegma_config.get('model', {})
    
    # Write base model config
    model_base = {'name': '???'}
    with open(config_dir / 'model' / 'base.yaml', 'w') as f:
        yaml.dump(model_base, f, default_flow_style=False)
    
    # Write rnndet model config (RVT's main model)
    rnndet_config = {
        'defaults': ['base'],
        'name': 'rnndet',
        'backbone': model_config.get('backbone', {}),
        'fpn': model_config.get('fpn', {}),
        'head': model_config.get('head', {}),
        'postprocess': model_config.get('postprocess', {}),
    }
    
    # Add distance configuration if enabled
    if model_config.get('distance', {}).get('enable', False):
        rnndet_config['distance'] = model_config['distance']
    
    with open(config_dir / 'model' / 'rnndet.yaml', 'w') as f:
        yaml.dump(rnndet_config, f, default_flow_style=False)
    
    # Write experiment config
    experiment_config = {
        'defaults': [
            {'dataset': config_name},
            {'model': 'rnndet'},
        ],
    }
    with open(config_dir / 'experiment' / config_name / 'default.yaml', 'w') as f:
        yaml.dump(experiment_config, f, default_flow_style=False)
    
    # Write main train.yaml config
    train_config = {
        'defaults': [
            'general',
            {'dataset': '???'},
            {'model': 'rnndet'},
            {'optional model/dataset': '${model}_${dataset}'},
        ],
    }
    with open(config_dir / 'train.yaml', 'w') as f:
        yaml.dump(train_config, f, default_flow_style=False)
    
    # Write main val.yaml config
    val_config = {
        'defaults': [
            'general',
            {'dataset': '???'},
            {'model': 'rnndet'},
        ],
        'checkpoint': '???',
        'use_test_set': False,
    }
    with open(config_dir / 'val.yaml', 'w') as f:
        yaml.dump(val_config, f, default_flow_style=False)
    
    return str(config_dir)


def generate_hydra_overrides(pegma_config: Dict[str, Any]) -> list:
    """
    Generate Hydra command-line overrides from PEGMA config.
    
    This is useful for passing configuration changes via command line
    when running RVT training with an existing config structure.
    
    Args:
        pegma_config: Configuration dictionary from RVTBase.to_hydra_config()
        
    Returns:
        List of Hydra override strings (e.g., ['training.max_epochs=50', ...])
    """
    overrides = []
    
    def flatten_dict(d: dict, prefix: str = '') -> list:
        items = []
        for key, value in d.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                items.extend(flatten_dict(value, full_key))
            elif value is not None:
                items.append(f"{full_key}={value}")
        return items
    
    # Flatten general config
    general = pegma_config.get('general', {})
    overrides.extend(flatten_dict(general))
    
    # Add dataset path override
    dataset = pegma_config.get('dataset', {})
    if dataset.get('path'):
        overrides.append(f"dataset.path={dataset['path']}")
    
    # Add model overrides for distance estimation
    model = pegma_config.get('model', {})
    if model.get('distance', {}).get('enable'):
        overrides.append('model.distance.enable=true')
        overrides.append(f"model.distance.nbins={model['distance']['nbins']}")
        overrides.append(f"model.distance.min_dist={model['distance']['min_dist']}")
        overrides.append(f"model.distance.max_dist={model['distance']['max_dist']}")
    
    return overrides


def create_pegma_dataset_module_config(dataset_path: str,
                                        ev_repr_name: str = 'stacked_histogram_dt=50_nbins=10',
                                        sequence_length: int = 11,
                                        resolution_hw: list = None,
                                        has_distance: bool = False) -> Dict[str, Any]:
    """
    Create dataset configuration specifically for PEGMA's dataset format.
    
    PEGMA datasets use a slightly different structure than Gen1/Gen4,
    so this function creates the appropriate configuration.
    
    Args:
        dataset_path: Root path to PEGMA dataset
        ev_repr_name: Event representation name
        sequence_length: Sequence length for training
        resolution_hw: Input resolution [height, width]
        has_distance: Whether dataset includes distance labels
        
    Returns:
        Dataset configuration dictionary
    """
    if resolution_hw is None:
        resolution_hw = [240, 304]
    
    return {
        'name': 'pegma',
        'path': dataset_path,
        'ev_repr_name': ev_repr_name,
        'sequence_length': sequence_length,
        'resolution_hw': resolution_hw,
        'downsample_by_factor_2': False,
        'only_load_end_labels': False,
        'has_distance': has_distance,
        'train': {
            'sampling': 'mixed',
            'random': {'weighted_sampling': False},
            'mixed': {'w_stream': 1, 'w_random': 1},
        },
        'eval': {
            'sampling': 'stream',
        },
    }


def write_complete_rvt_config(pegma_experiment, output_dir: str) -> str:
    """
    Write complete RVT configuration from a PEGMA experiment object.
    
    Args:
        pegma_experiment: Instance of RVTBase experiment class
        output_dir: Directory to write configuration files
        
    Returns:
        Path to config directory
    """
    # Get Hydra config from experiment
    hydra_config = pegma_experiment.to_hydra_config()
    
    # Generate config files
    config_dir = generate_rvt_configs(
        hydra_config,
        output_dir=output_dir,
        config_name='pegma'
    )
    
    return config_dir


class RVTConfigWriter:
    """
    Helper class for writing RVT configurations incrementally.
    
    Useful when you need to modify configurations before writing
    or when building configs programmatically.
    """
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = {
            'general': {},
            'dataset': {},
            'model': {},
        }
    
    def set_general(self, **kwargs):
        """Set general training configuration."""
        self.config['general'].update(kwargs)
    
    def set_dataset(self, **kwargs):
        """Set dataset configuration."""
        self.config['dataset'].update(kwargs)
    
    def set_model(self, **kwargs):
        """Set model configuration."""
        self.config['model'].update(kwargs)
    
    def enable_distance(self, nbins: int = 100, 
                        min_dist: float = 0.0,
                        max_dist: float = 10.0,
                        loss_multiplier: float = 1.0):
        """Enable distance estimation in the model."""
        self.config['model']['distance'] = {
            'enable': True,
            'nbins': nbins,
            'min_dist': min_dist,
            'max_dist': max_dist,
            'loss_multiplier': loss_multiplier,
        }
    
    def write(self, config_name: str = 'pegma') -> str:
        """Write all configuration files."""
        return generate_rvt_configs(
            self.config,
            output_dir=str(self.output_dir),
            config_name=config_name
        )


if __name__ == '__main__':
    # Example usage
    config = {
        'general': {
            'training': {
                'max_epochs': 100,
                'learning_rate': 0.0002,
            },
        },
        'dataset': {
            'path': '/path/to/dataset',
            'ev_repr_name': 'stacked_histogram_dt=50_nbins=10',
            'sequence_length': 11,
            'resolution_hw': [240, 304],
        },
        'model': {
            'name': 'rnndet',
            'backbone': {'name': 'maxvit_rnn', 'dim': 64},
            'distance': {
                'enable': True,
                'nbins': 100,
                'min_dist': 0.0,
                'max_dist': 10.0,
            },
        },
    }
    
    config_dir = generate_rvt_configs(config, output_dir='./test_rvt_config')
    print(f"Generated RVT configs in: {config_dir}")

