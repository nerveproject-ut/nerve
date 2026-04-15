from pathlib import Path
from typing import Union
import os
import logging

import wandb
from omegaconf import DictConfig, OmegaConf

from loggers.wandb_logger import WandbLogger
from loggers.local_logger import LocalLogger


def is_wandb_available() -> bool:
    """Check if WandB is available and properly configured."""
    # Check if WANDB_MODE is set to disabled/offline
    wandb_mode = os.environ.get('WANDB_MODE', '').lower()
    if wandb_mode in ('disabled', 'offline', 'dryrun'):
        return False
    
    # Check if WANDB_API_KEY is set or wandb is logged in
    try:
        # Try to check if logged in
        api = wandb.Api()
        # If we get here without error, wandb is available
        return True
    except Exception:
        return False


def get_logger(full_config: DictConfig, force_local: bool = False):
    """
    Get the appropriate logger based on availability.
    
    Args:
        full_config: Full configuration dict
        force_local: If True, always use local logger even if WandB is available
    
    Returns:
        Either WandbLogger or LocalLogger
    """
    use_wandb = not force_local and is_wandb_available()
    
    if use_wandb:
        return get_wandb_logger(full_config)
    else:
        return get_local_logger(full_config)


def get_wandb_logger(full_config: DictConfig) -> WandbLogger:
    wandb_config = full_config.wandb
    wandb_runpath = wandb_config.wandb_runpath

    if wandb_runpath is None:
        wandb_id = wandb.util.generate_id()
        print(f'new run: generating id {wandb_id}')
    else:
        wandb_id = Path(wandb_runpath).name
        print(f'using provided id {wandb_id}')

    full_config_dict = OmegaConf.to_container(full_config, resolve=True, throw_on_missing=True)
    logger = WandbLogger(
        project=wandb_config.project_name,
        group=wandb_config.group_name,
        wandb_id=wandb_id,
        log_model=True,
        save_last_only_final=False,
        save_code=True,
        config_args=full_config_dict,
    )

    return logger


def get_local_logger(full_config: DictConfig) -> LocalLogger:
    """Create a LocalLogger with TensorBoard support."""
    wandb_config = full_config.wandb
    dataset_config = full_config.dataset
    
    # Generate a unique run name
    import wandb
    run_id = wandb.util.generate_id()
    
    # Use group name and project name to create save directory
    project_name = wandb_config.project_name or "PEGMA-RVT"
    group_name = wandb_config.group_name or "default"
    dataset_name = dataset_config.name if hasattr(dataset_config, 'name') else "unknown"
    
    experiment_name = f"{group_name}_{dataset_name}"
    
    logger = LocalLogger(
        save_dir="runs",
        name=experiment_name,
        version=run_id,
        use_tensorboard=True
    )
    
    # Log hyperparams
    full_config_dict = OmegaConf.to_container(full_config, resolve=True, throw_on_missing=True)
    logger.log_hyperparams(full_config_dict)
    
    print(f"Using LocalLogger with TensorBoard (WandB not available or disabled)")
    print(f"Results will be saved to: {logger.log_dir}")
    print(f"View TensorBoard with: tensorboard --logdir {logger.log_dir}")
    
    return logger


def get_ckpt_path(logger, wandb_config: DictConfig) -> Union[Path, None]:
    """
    Get checkpoint path for resuming training.
    
    Args:
        logger: Either WandbLogger or LocalLogger
        wandb_config: WandB configuration dict
    
    Returns:
        Path to checkpoint file or None
    """
    cfg = wandb_config
    artifact_name = cfg.artifact_name
    assert artifact_name is not None, 'Artifact name is required to resume from checkpoint.'
    print(f'resuming checkpoint from artifact {artifact_name}')
    artifact_local_file = cfg.artifact_local_file
    if artifact_local_file is not None:
        artifact_local_file = Path(artifact_local_file)
    
    if isinstance(logger, WandbLogger):
        resume_path = logger.get_checkpoint(
            artifact_name=artifact_name,
            artifact_filepath=artifact_local_file)
    elif isinstance(logger, LocalLogger):
        # For LocalLogger, we can only resume from a local file
        if artifact_local_file is None:
            raise ValueError(
                "LocalLogger cannot download WandB artifacts. "
                "Please provide 'artifact_local_file' with a local path to the checkpoint."
            )
        resume_path = artifact_local_file
    else:
        resume_path = artifact_local_file
    
    if resume_path is None:
        return None
        
    assert resume_path.exists(), f"Checkpoint file not found: {resume_path}"
    assert resume_path.suffix == '.ckpt', f"Expected .ckpt file, got: {resume_path.suffix}"
    return resume_path
