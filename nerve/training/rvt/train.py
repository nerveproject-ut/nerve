import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch

torch.multiprocessing.set_sharing_strategy('file_system')
from torch.backends import cuda, cudnn

cuda.matmul.allow_tf32 = True
cudnn.allow_tf32 = True

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelSummary
from pytorch_lightning.strategies import DDPStrategy

from callbacks.custom import get_ckpt_callback, get_viz_callback
from callbacks.gradflow import GradFlowLogCallback
from config.modifier import dynamically_modify_train_config
from data.utils.types import DatasetSamplingMode
from loggers.utils import get_logger, get_ckpt_path, is_wandb_available
from loggers.wandb_logger import WandbLogger
from loggers.local_logger import LocalLogger
from nerve.training.rvt.modules.utils.fetch import fetch_data_module, fetch_model_module


@hydra.main(config_path='config', config_name='train', version_base='1.2')
def main(config: DictConfig):
    dynamically_modify_train_config(config)
    # Just to check whether config can be resolved
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)

    print('------ Configuration ------')
    print(OmegaConf.to_yaml(config))
    print('---------------------------')

    # ---------------------
    # Reproducibility
    # ---------------------
    dataset_train_sampling = config.dataset.train.sampling
    assert dataset_train_sampling in iter(DatasetSamplingMode)
    disable_seed_everything = dataset_train_sampling in (DatasetSamplingMode.STREAM, DatasetSamplingMode.MIXED)
    if disable_seed_everything:
        print('Disabling PL seed everything because of unresolved issues with shuffling during training on streaming '
              'datasets')
    seed = config.reproduce.seed_everything
    if seed is not None and not disable_seed_everything:
        assert isinstance(seed, int)
        print(f'USING pl.seed_everything WITH {seed=}')
        pl.seed_everything(seed=seed, workers=True)

    # ---------------------
    # DDP
    # ---------------------
    gpu_config = config.hardware.gpus
    gpus = OmegaConf.to_container(gpu_config) if OmegaConf.is_config(gpu_config) else gpu_config
    gpus = gpus if isinstance(gpus, list) else [gpus]
    distributed_backend = config.hardware.dist_backend
    assert distributed_backend in ('nccl', 'gloo'), f'{distributed_backend=}'
    # Note: find_unused_parameters=True is required because:
    # 1. RNN/LSTM states may not contribute to loss on first iteration
    # 2. Some model branches may be conditionally executed
    # 3. Distance estimation head may not be used in all configurations
    # This has a slight performance overhead but prevents DDP synchronization errors
    strategy = DDPStrategy(process_group_backend=distributed_backend,
                           find_unused_parameters=True,
                           gradient_as_bucket_view=True) if len(gpus) > 1 else 'auto'

    # ---------------------
    # Data
    # ---------------------
    data_module = fetch_data_module(config=config)

    # ---------------------
    # Logging and Checkpoints
    # ---------------------
    # Use the unified logger that supports both WandB and local/TensorBoard logging
    logger = get_logger(config)
    using_wandb = isinstance(logger, WandbLogger)
    
    if using_wandb:
        print("Using WandB for logging")
    else:
        print("Using LocalLogger with TensorBoard (WandB disabled or unavailable)")
    
    ckpt_path = None
    # Support loading from W&B artifact or local file
    if config.wandb.artifact_name is not None:
        # Load from W&B artifact (requires WandB for remote artifacts)
        ckpt_path = get_ckpt_path(logger, wandb_config=config.wandb)
    elif config.wandb.artifact_local_file is not None:
        # Load directly from local file (for fine-tuning from pretrained weights)
        from pathlib import Path
        ckpt_path = Path(config.wandb.artifact_local_file)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Pretrained weights not found: {ckpt_path}")
        print(f'Loading pretrained weights from local file: {ckpt_path}')

    # ---------------------
    # Model
    # ---------------------
    module = fetch_model_module(config=config)
    if ckpt_path is not None and config.wandb.resume_only_weights:
        print('Resuming only the weights instead of the full training state')
        module = module.load_from_checkpoint(str(ckpt_path), **{'full_config': config})
        ckpt_path = None

    # ---------------------
    # Callbacks and Misc
    # ---------------------
    callbacks = list()
    callbacks.append(get_ckpt_callback(config))
    callbacks.append(GradFlowLogCallback(config.logging.train.log_model_every_n_steps))
    if config.training.lr_scheduler.use:
        callbacks.append(LearningRateMonitor(logging_interval='step'))
    if config.logging.train.high_dim.enable or config.logging.validation.high_dim.enable:
        viz_callback = get_viz_callback(config=config)
        callbacks.append(viz_callback)
    callbacks.append(ModelSummary(max_depth=2))

    # Watch model (only supported by WandB, LocalLogger will no-op)
    logger.watch(model=module, log='all', log_freq=config.logging.train.log_model_every_n_steps, log_graph=True)
    
    # Save model summary for local logger
    if isinstance(logger, LocalLogger):
        logger.save_model_summary(module)

    # ---------------------
    # Training
    # ---------------------

    val_check_interval = config.validation.val_check_interval
    check_val_every_n_epoch = config.validation.check_val_every_n_epoch
    assert val_check_interval is None or check_val_every_n_epoch is None

    trainer = pl.Trainer(
        accelerator='gpu',
        callbacks=callbacks,
        enable_checkpointing=True,
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=check_val_every_n_epoch,
        default_root_dir=None,
        devices=gpus,
        gradient_clip_val=config.training.gradient_clip_val,
        gradient_clip_algorithm='value',
        limit_train_batches=config.training.limit_train_batches,
        limit_val_batches=config.validation.limit_val_batches,
        logger=logger,
        log_every_n_steps=config.logging.train.log_every_n_steps,
        plugins=None,
        precision=config.training.precision,
        max_epochs=config.training.max_epochs,
        max_steps=config.training.max_steps,
        strategy=strategy,
        sync_batchnorm=True if len(gpus) > 1 else False,
        benchmark=config.reproduce.benchmark,
        deterministic=config.reproduce.deterministic_flag,
    )
    trainer.fit(model=module, ckpt_path=ckpt_path, datamodule=data_module)
    
    # Finalize logger (this will generate results.png for LocalLogger)
    if isinstance(logger, LocalLogger):
        logger.finalize("completed" if trainer.state.status == "finished" else trainer.state.status)
    
    # Print where results are saved
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)
    
    if isinstance(logger, LocalLogger):
        print(f"\nResults saved to: {logger.log_dir}")
        print(f"  - Images: {logger.log_dir / 'images'}")
        print(f"  - Metrics: {logger.log_dir / 'metrics'}")
        print(f"  - Results CSV: {logger.log_dir / 'results_unified.csv'}")
        print(f"  - Results Plot: {logger.log_dir / 'results_unified.png'}")
        print(f"  - Hyperparameters: {logger.log_dir / 'hparams.yaml'}")
        print(f"\nTo view TensorBoard:")
        print(f"  tensorboard --logdir {logger.log_dir}")
    else:
        # WandB logger
        print(f"\nResults logged to WandB project: {config.wandb.project_name}")
        print(f"  - Group: {config.wandb.group_name}")
    
    # Print checkpoint location
    ckpt_callback = None
    for cb in callbacks:
        if hasattr(cb, 'best_model_path'):
            ckpt_callback = cb
            break
    
    if ckpt_callback:
        print(f"\nCheckpoints saved to:")
        print(f"  - Best model: {ckpt_callback.best_model_path}")
        if hasattr(ckpt_callback, 'last_model_path') and ckpt_callback.last_model_path:
            print(f"  - Last model: {ckpt_callback.last_model_path}")
    
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
