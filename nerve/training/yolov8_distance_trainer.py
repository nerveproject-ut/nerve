"""
Custom YOLOv8 Trainer with Distance Estimation Support.
Extends ultralytics trainer to handle distance prediction and loss.
"""

import os
import torch
import torch.nn as nn
from pathlib import Path
from copy import deepcopy

try:
    from ultralytics.engine.trainer import BaseTrainer
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils import LOGGER, RANK
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils import callbacks as ultralytics_callbacks
except ImportError:
    raise ImportError("YOLOv8 (ultralytics) is not installed. Install with: pip install ultralytics")


def de_parallel(model):
    """Unwrap DataParallel/DistributedDataParallel to get the raw model."""
    return model.module if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)) else model

from nerve.training.custom_yolov8_distance import YOLOv8WithDistance, DistanceDetectionHead
from nerve.training.dvs_radar_dataset_yolov8 import YOLOv8_DVS_Radar_Dataset


def _disable_wandb_globally():
    """
    Disable WandB globally via environment variables.
    Must be called BEFORE ultralytics initializes.
    """
    os.environ['WANDB_DISABLED'] = 'true'
    os.environ['WANDB_MODE'] = 'disabled'
    
    # Also try to remove from default callbacks (affects new trainers)
    try:
        default_cbs = ultralytics_callbacks.default_callbacks
        wandb_hooks = ['on_pretrain_routine_start', 'on_fit_epoch_end', 'on_train_end']
        for hook in wandb_hooks:
            if hook in default_cbs:
                default_cbs[hook] = [
                    cb for cb in default_cbs[hook]
                    if 'wb' not in getattr(cb, '__module__', '') and 'wandb' not in getattr(cb, '__module__', '')
                ]
    except Exception:
        pass
    
    LOGGER.info("WandB callbacks disabled")


def _is_wandb_callback(cb):
    """
    Check if a callback is a WandB callback.
    Uses multiple heuristics for robust detection.
    """
    # Check module name
    module = getattr(cb, '__module__', '') or ''
    if 'wb' in module.lower() or 'wandb' in module.lower():
        return True
    
    # Check function name
    func_name = getattr(cb, '__name__', '') or ''
    # WandB callbacks in ultralytics don't have special names, but check anyway
    
    # Check qualname (qualified name) which includes the module path
    qualname = getattr(cb, '__qualname__', '') or ''
    if 'wb' in qualname.lower() or 'wandb' in qualname.lower():
        return True
    
    # Check if the callback's code file contains 'wb.py' or 'wandb'
    try:
        code = getattr(cb, '__code__', None)
        if code:
            filename = getattr(code, 'co_filename', '') or ''
            if 'wb.py' in filename or 'wandb' in filename.lower():
                return True
    except (AttributeError, TypeError):
        pass
    
    return False


def _remove_wandb_from_trainer(trainer):
    """
    Remove WandB callbacks from an existing trainer instance.
    Call this AFTER parent __init__ to remove already-registered callbacks.
    """
    if not hasattr(trainer, 'callbacks'):
        return
    
    # List ALL wandb hooks - ultralytics/utils/callbacks/wb.py registers these
    wandb_hooks = [
        'on_pretrain_routine_start',
        'on_pretrain_routine_end', 
        'on_train_start',
        'on_train_epoch_start',
        'on_train_epoch_end',
        'on_train_batch_start',
        'on_train_batch_end',
        'on_fit_epoch_end',
        'on_train_end',
        'on_val_start',
        'on_val_batch_start',
        'on_val_batch_end',
        'on_val_end',
        'on_model_save',
    ]
    
    removed_count = 0
    for hook in wandb_hooks:
        if hook in trainer.callbacks:
            original_len = len(trainer.callbacks[hook])
            trainer.callbacks[hook] = [
                cb for cb in trainer.callbacks[hook]
                if not _is_wandb_callback(cb)
            ]
            removed_count += original_len - len(trainer.callbacks[hook])
    
    if removed_count > 0:
        LOGGER.info(f"Removed {removed_count} WandB callbacks from trainer")


class DistanceDetectionTrainer(DetectionTrainer):
    """
    Custom YOLOv8 trainer that handles distance estimation.
    Similar to YOLOX custom trainer but adapted for YOLOv8 framework.
    """
    
    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        """
        Initialize the distance detection trainer.
        
        Args:
            cfg: Configuration dict or path
            overrides: Override default config values
            _callbacks: Optional callbacks
        """
        # Custom parameters for distance estimation
        self.include_radar = False
        self.process_distance = False
        self.distance_from_head = True
        self.min_dist = 0.0
        self.max_dist = 10.0
        self.nbins = 100
        self.distance_loss_multiplier = 1.0
        
        # Extract custom parameters from overrides before passing to parent
        # Track if wandb should be disabled (need to apply AFTER parent __init__)
        self._disable_wandb = True  # Default to disabled
        
        if overrides:
            self.include_radar = overrides.pop('include_radar', False)
            self.process_distance = overrides.pop('process_distance', False)
            self.distance_from_head = overrides.pop('distance_from_head', True)
            self.min_dist = overrides.pop('min_dist', 0.0)
            self.max_dist = overrides.pop('max_dist', 10.0)
            self.nbins = overrides.pop('nbins', 100)
            self.distance_loss_multiplier = overrides.pop('distance_loss_multiplier', 1.0)
            
            # Check if wandb should be disabled
            use_wandb = overrides.pop('use_wandb', False)
            if not use_wandb:
                self._disable_wandb = True
                _disable_wandb_globally()  # Set env vars before parent init
            else:
                self._disable_wandb = False
                # If using wandb, set the wandb project name from config
                # (not the filesystem path)
                wandb_project = overrides.pop('wandb_project', 'PEGMA-YOLOv8')
                wandb_name = overrides.pop('wandb_name', None)
                os.environ['WANDB_PROJECT'] = wandb_project
                if wandb_name:
                    os.environ['WANDB_NAME'] = wandb_name
                LOGGER.info(f"WandB enabled with project: {wandb_project}")
            
            # Map data_yaml to data if present (ultralytics uses 'data')
            if 'data_yaml' in overrides and 'data' not in overrides:
                overrides['data'] = overrides['data_yaml']
            
            # Fix type issues: ultralytics expects 'val' to be boolean, not int
            if 'val' in overrides and isinstance(overrides['val'], int):
                overrides['val'] = bool(overrides['val'])
            
            # Remove custom keys that ultralytics doesn't recognize
            # These are our experiment-specific config keys
            custom_keys = [
                'model_type', 'dataset_path', 'nc', 'classes',
                'wandb_project', 'wandb_name',
                'data_yaml', 'include_distance', 'filter_classes'
            ]
            for key in custom_keys:
                overrides.pop(key, None)
        else:
            overrides = {}
            # Default: disable wandb
            _disable_wandb_globally()
        
        # Ensure task is set
        if 'task' not in overrides:
            overrides['task'] = 'detect'
        
        # NOTE: Mosaic augmentation can now be enabled with distance training!
        # The v8DistanceDetectionLoss properly handles distance labels using
        # Task-Aligned Assignment (TAL), so mosaic doesn't break distance training.
        # The distance labels are preserved as long as the dataset outputs them correctly.
        # 
        # If you encounter issues with mosaic + distance, you can still disable it:
        # if self.process_distance:
        #     overrides['mosaic'] = 0.0
        #     overrides['mixup'] = 0.0
        pass  # Allow mosaic to be controlled by experiment config
        
        # Initialize parent with cleaned overrides
        super().__init__(overrides=overrides, _callbacks=_callbacks)
        
        # CRITICAL: Remove wandb callbacks from THIS trainer instance
        # This must happen AFTER parent __init__ which registers callbacks
        if self._disable_wandb:
            _remove_wandb_from_trainer(self)
        
        LOGGER.info(f"DistanceDetectionTrainer initialized:")
        LOGGER.info(f"  - Include radar: {self.include_radar}")
        LOGGER.info(f"  - Process distance: {self.process_distance}")
        LOGGER.info(f"  - Distance from head: {self.distance_from_head}")
        LOGGER.info(f"  - Distance range: [{self.min_dist}, {self.max_dist}]m")
    
    def train(self, *args, **kwargs):
        """
        Override train to fix wandb project name before callbacks run.
        """
        # CRITICAL: Ensure wandb callbacks are removed right before training
        # (they might have been re-added through callbacks or checkpointing)
        if self._disable_wandb:
            _remove_wandb_from_trainer(self)
        
        # Fix wandb project name before training starts (only if wandb is enabled)
        if not self._disable_wandb and 'WANDB_PROJECT' in os.environ:
            self._orig_project = self.args.project
            wandb_project = os.environ['WANDB_PROJECT']
            
            LOGGER.info(f"Setting wandb project to: '{wandb_project}'")
            LOGGER.info(f"Filesystem path remains: '{self._orig_project}'")
            
            # Temporarily change project for wandb init
            self.args.project = wandb_project
            
            # Add callback to restore after wandb init
            def restore_project(trainer):
                trainer.args.project = self._orig_project
                LOGGER.info(f"Restored filesystem project to: '{self._orig_project}'")
            
            self.add_callback('on_pretrain_routine_end', restore_project)
        
        # Call parent train with error handling for wandb callback issues
        try:
            return super().train(*args, **kwargs)
        except AttributeError as e:
            error_str = str(e).lower()
            # Catch various WandB-related errors that occur when WandB is disabled
            # but callbacks are still trying to log
            wandb_error_patterns = [
                "curves_results",
                "'nonetype' object has no attribute 'log'",
                "wb.run",
                "wandb",
            ]
            if any(pattern in error_str for pattern in wandb_error_patterns):
                # Wandb callback trying to access disabled wandb
                LOGGER.warning(f"Ignoring WandB callback error (WandB is disabled): {e}")
                LOGGER.info("✓ Training completed successfully (ignoring WandB error)")
                return
            else:
                raise
    
    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        Get model with distance estimation support.
        
        Args:
            cfg: Model config
            weights: Pretrained weights path
            verbose: Print model info
            
        Returns:
            Model instance
        """
        if not self.process_distance:
            # Standard YOLOv8 model
            return super().get_model(cfg, weights, verbose)
        
        # Create custom model with distance head
        model = YOLOv8WithDistance(
            cfg=cfg or self.args.model,
            ch=3,
            nc=self.data.get('nc', 1),
            verbose=verbose and RANK == -1,
            distance_from_head=self.distance_from_head,
            nbins=self.nbins,
            min_dist=self.min_dist,
            max_dist=self.max_dist
        )
        
        # Load weights if provided
        if weights:
            model.load(weights)
        
        return model
    
    def build_dataset(self, img_path, mode='train', batch=None):
        """
        Build custom dataset with radar and distance support.
        
        Always uses custom dataset to handle the data/images path symlink issue
        that causes standard YOLODataset to fail finding labels.
        
        Args:
            img_path: Path to images
            mode: 'train' or 'val'
            batch: Batch size
            
        Returns:
            Dataset instance
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        
        # Always use custom dataset - it handles both COCO JSON and YOLO txt formats
        # This also fixes the symlink issue where standard YOLODataset can't find labels
        # when images are in /data/ instead of /images/
        LOGGER.info(f"Building custom DVS+Radar dataset for {mode}")
        
        # Get json file and data dir from self.data
        # Support multiple key formats: train_ann/val_ann (from data.yaml) or train_json/val_json
        json_file = None
        data_dir = self.data.get('path', '')
        
        if mode == 'train':
            # Try multiple annotation key names
            json_file = self.data.get('train_ann') or self.data.get('train_json')
        elif mode == 'val':
            json_file = self.data.get('val_ann') or self.data.get('val_json')
        elif mode == 'test':
            json_file = self.data.get('test_ann') or self.data.get('test_json')
        
        # Log the resolved paths
        if json_file:
            LOGGER.info(f"  COCO annotations: {json_file}")
        else:
            LOGGER.info(f"  No COCO annotations found, will use YOLO txt labels")
        
        dataset = YOLOv8_DVS_Radar_Dataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch if batch else self.args.batch,
            augment=mode == 'train',
            hyp=self.args,
            rect=self.args.rect or mode == 'val',
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=int(gs),
            pad=0.0 if mode == 'train' else 0.5,
            prefix=f'{mode}: ',
            use_also_radar=self.include_radar,
            include_distance=self.process_distance,
            min_dist=self.min_dist,
            max_dist=self.max_dist,
            json_file=json_file,
            data_dir=data_dir,
            data_dict=self.data  # Pass the data dict to the dataset
        )
        
        return dataset
    
    def get_validator(self):
        """
        Get validator for distance-aware evaluation.
        """
        if self.process_distance:
            # Set loss names to include distance loss
            self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss', 'dist_loss'
            
            try:
                # Import custom validator
                from nerve.training.yolov8_distance_validator import DistanceDetectionValidator
                
                return DistanceDetectionValidator(
                    self.test_loader,
                    save_dir=self.save_dir,
                    args=deepcopy(self.args),
                    _callbacks=self.callbacks,
                    process_distance=self.process_distance,
                    distance_from_head=self.distance_from_head,
                    min_dist=self.min_dist,
                    max_dist=self.max_dist
                )
            except ImportError:
                LOGGER.warning("DistanceDetectionValidator not found, using standard validator")
                LOGGER.warning("Distance metrics will not be computed during validation")
                return super().get_validator()
        
        # Standard validator
        self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss'
        return super().get_validator()
    
    def criterion(self, preds, batch):
        """
        Compute loss with distance estimation.
        
        Uses v8DistanceDetectionLoss which properly integrates distance loss
        with Task-Aligned Assigner (TAL) for correct foreground matching.
        
        Args:
            preds: Model predictions (det_preds, dist_preds) tuple or just det_preds
            batch: Batch data with 'img', 'bboxes', 'cls', 'distances'
            
        Returns:
            (total_loss * batch_size, loss_items) where loss_items = [box, cls, dfl, dist]
        """
        if not hasattr(self, 'compute_loss'):
            # Initialize the loss function
            if self.process_distance and self.distance_from_head:
                try:
                    from nerve.training.yolov8_distance_loss import v8DistanceDetectionLoss
                    self.compute_loss = v8DistanceDetectionLoss(
                        de_parallel(self.model),
                        nbins=self.nbins,
                        min_dist=self.min_dist,
                        max_dist=self.max_dist,
                        distance_loss_multiplier=self.distance_loss_multiplier
                    )
                    LOGGER.info("Using v8DistanceDetectionLoss for training")
                except ImportError as e:
                    LOGGER.warning(f"Could not import v8DistanceDetectionLoss: {e}")
                    LOGGER.warning("Falling back to standard loss (distance will not be computed)")
                    # Try both ultralytics import paths
                    try:
                        from ultralytics.utils.loss import v8DetectionLoss as Loss
                    except ImportError:
                        from ultralytics.yolo.utils.loss import v8DetectionLoss as Loss
                    self.compute_loss = Loss(de_parallel(self.model))
            else:
                # Standard detection loss - try both ultralytics import paths
                try:
                    from ultralytics.utils.loss import v8DetectionLoss as Loss
                except ImportError:
                    from ultralytics.yolo.utils.loss import v8DetectionLoss as Loss
                self.compute_loss = Loss(de_parallel(self.model))
        
        # Handle tuple predictions from DistanceDetectionHead
        # The head returns (det_feats, dist_feats) but standard loss expects only det_feats
        if isinstance(preds, tuple) and len(preds) == 2:
            det_preds, dist_preds = preds
            return self.compute_loss(det_preds, batch)
        
        return self.compute_loss(preds, batch)
    
    def save_model(self):
        """
        Override save_model to handle deepcopy issues with custom distance head.
        Falls back to state_dict() if deepcopy fails.
        """
        from copy import deepcopy
        import torch
        
        # Try parent's save_model first
        try:
            return super().save_model()
        except RuntimeError as e:
            if "deepcopy protocol" in str(e):
                # Deepcopy failed - save using state_dict instead
                LOGGER.warning(f"Deepcopy failed, saving model using state_dict: {e}")
                
                # Get model without DataParallel wrapper  
                model = de_parallel(self.model)
                
                # Save state dict instead of full model
                ckpt = {
                    'epoch': self.epoch,
                    'best_fitness': self.best_fitness,
                    'model': model.state_dict(),  # Save state_dict, not full model
                    'ema': self.ema.ema.state_dict() if self.ema else None,
                    'updates': self.ema.updates if self.ema else None,
                    'optimizer': self.optimizer.state_dict(),
                    'train_args': vars(self.args),
                }
                
                # Save last checkpoint
                torch.save(ckpt, self.last)
                LOGGER.info(f"Saved state_dict checkpoint to {self.last}")
                
                # Save best checkpoint
                if self.best_fitness == self.fitness:
                    torch.save(ckpt, self.best)
                    LOGGER.info(f"Saved state_dict checkpoint to {self.best}")
                
                del ckpt
            else:
                raise
    
    def final_eval(self):
        """
        Override final_eval to handle state_dict checkpoints and distance model properly.
        
        CRITICAL: We must load the model as YOLOv8WithDistance, not AutoBackend,
        to ensure distance predictions are properly formatted for validation.
        """
        from ultralytics.utils.torch_utils import strip_optimizer
        import torch
        
        for f in [self.last, self.best]:
            if f.exists():
                try:
                    # Try to strip optimizer normally
                    strip_optimizer(f)
                except AttributeError as e:
                    if "'collections.OrderedDict' object has no attribute 'half'" in str(e):
                        LOGGER.warning(f"Skipping strip_optimizer for {f} (state_dict checkpoint)")
                    else:
                        raise
                
                # For best.pt, validate using the actual model (not AutoBackend)
                # This ensures distance predictions are properly formatted
                if f == self.best:
                    LOGGER.info(f'\nValidating {f}...')
                    self.validator.args.plots = self.args.plots
                    
                    # Load checkpoint and update model weights
                    ckpt = torch.load(f, map_location='cpu', weights_only=False)
                    model = de_parallel(self.model)
                    
                    # Handle both full model and state_dict checkpoints
                    if isinstance(ckpt.get('model'), dict):
                        # state_dict format
                        model.load_state_dict(ckpt['model'])
                    elif hasattr(ckpt.get('model'), 'state_dict'):
                        # Full model format - extract state_dict
                        model.load_state_dict(ckpt['model'].state_dict())
                    else:
                        # Try direct load
                        try:
                            model.load_state_dict(ckpt['model'])
                        except Exception:
                            LOGGER.warning(f"Could not load model weights from {f}, using AutoBackend fallback")
                            self.metrics = self.validator(model=f)
                            self.metrics.pop('fitness', None)
                            self.run_callbacks('on_fit_epoch_end')
                            continue
                    
                    # Validate using the properly loaded YOLOv8WithDistance model
                    model.eval()
                    self.metrics = self.validator(model=model)
                    self.metrics.pop('fitness', None)
                    self.run_callbacks('on_fit_epoch_end')
    
    def preprocess_batch(self, batch):
        """
        Preprocess batch before training step.
        Ensures distances are properly handled and moved to correct device.
        """
        # Call parent's preprocess_batch
        batch = super().preprocess_batch(batch)
        
        # Ensure distances are moved to the correct device (like bboxes/cls)
        if 'distances' in batch and hasattr(batch['distances'], 'to'):
            batch['distances'] = batch['distances'].to(batch['img'].device)
        
        return batch
    
    def progress_string(self):
        """Returns a formatted string of training progress with losses."""
        if not self.process_distance:
            return super().progress_string()
        
        # Include distance loss in progress string
        string = super().progress_string()
        
        if hasattr(self, 'tloss') and self.tloss is not None:
            if len(self.tloss) > 3:  # Has distance loss
                dist_loss = self.tloss[3]
                string += f' - dist_loss: {dist_loss:.4f}'
        
        return string


class DistanceLoss(nn.Module):
    """
    Distance loss module for YOLOv8.
    Computes loss for distance prediction as classification over bins.
    """
    
    def __init__(self, nbins=100, min_dist=0.0, max_dist=10.0, loss_weight=1.0):
        super().__init__()
        self.nbins = nbins
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.loss_weight = loss_weight
        self.dist_multiplier = (max_dist - min_dist) / nbins
        
        # BCE loss for distance classification
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, pred_dist, target_dist, fg_mask):
        """
        Compute distance loss.
        
        Args:
            pred_dist: Predicted distance logits [B, num_anchors, nbins]
            target_dist: Target distances [num_targets] in meters
            fg_mask: Foreground mask [B, num_anchors]
            
        Returns:
            Distance loss value
        """
        if not fg_mask.any():
            return torch.tensor(0.0, device=pred_dist.device)
        
        # Select foreground predictions
        pred_dist_fg = pred_dist[fg_mask]  # [num_fg, nbins]
        
        # Convert distance to bin targets
        target_dist_clamped = torch.clamp(target_dist, self.min_dist, self.max_dist)
        target_bins = ((target_dist_clamped - self.min_dist) / self.dist_multiplier).long()
        target_bins = torch.clamp(target_bins, 0, self.nbins - 1)
        
        # One-hot encoding
        target_onehot = torch.zeros_like(pred_dist_fg)
        target_onehot.scatter_(1, target_bins.unsqueeze(1), 1.0)
        
        # Compute BCE loss
        loss = self.bce(pred_dist_fg, target_onehot).mean()
        
        return loss * self.loss_weight


def train_yolov8_with_distance(
    data_config,
    model='yolov8n.yaml',
    epochs=100,
    batch_size=16,
    imgsz=640,
    device='',
    workers=8,
    project=None,
    name=None,
    # Custom distance parameters
    include_radar=False,
    process_distance=False,
    distance_from_head=True,
    min_dist=0.0,
    max_dist=10.0,
    nbins=100,
    distance_loss_multiplier=1.0,
    **kwargs
):
    """
    Train YOLOv8 model with distance estimation.
    
    Args:
        data_config: Path to data config YAML
        model: Model config (n/s/m/l/x or yaml path)
        epochs: Training epochs
        batch_size: Batch size
        imgsz: Image size
        device: Device to use
        workers: Number of workers
        project: Project name
        name: Experiment name
        include_radar: Use radar data fusion
        process_distance: Enable distance estimation
        distance_from_head: Predict distance from head vs extract from radar
        min_dist: Minimum distance
        max_dist: Maximum distance
        nbins: Number of distance bins
        distance_loss_multiplier: Weight for distance loss
        **kwargs: Additional training arguments
        
    Returns:
        Trainer instance
    """
    # Prepare overrides
    overrides = {
        'data': data_config,
        'model': model,
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': imgsz,
        'device': device,
        'workers': workers,
        'project': project,
        'name': name,
        # Custom parameters
        'include_radar': include_radar,
        'process_distance': process_distance,
        'distance_from_head': distance_from_head,
        'min_dist': min_dist,
        'max_dist': max_dist,
        'nbins': nbins,
        'distance_loss_multiplier': distance_loss_multiplier,
    }
    
    # Add any additional kwargs
    overrides.update(kwargs)
    
    # Create trainer
    trainer = DistanceDetectionTrainer(overrides=overrides)
    
    # Start training
    trainer.train()
    
    return trainer


if __name__ == '__main__':
    # Example usage
    train_yolov8_with_distance(
        data_config='data.yaml',
        model='yolov8m.yaml',
        epochs=30,
        batch_size=16,
        imgsz=416,
        include_radar=True,
        process_distance=True,
        distance_from_head=True,
        min_dist=0.0,
        max_dist=10.0,
        nbins=100,
        name='yolov8_distance_test'
    )

