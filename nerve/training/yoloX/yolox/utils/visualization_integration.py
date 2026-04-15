"""
YOLOX Visualization Integration.

Provides utilities for integrating the shared visualization module
with YOLOX training, matching YOLOv8's output visualizations.

Generates:
- results.csv, results.png - Training/validation metrics
- confusion_matrix.png, confusion_matrix_normalized.png - Confusion matrices  
- train_batch*.jpg - Training batch visualizations
- val_batch*_labels.jpg, val_batch*_pred.jpg - Validation batch visualizations
- labels.jpg, labels_correlogram.jpg - Label distribution analysis
- P_curve.png, R_curve.png, F1_curve.png, PR_curve.png - PR curves (when data available)
- args.yaml - Training configuration
"""

import os
import sys
import yaml
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import torch

# Add parent directory to path for imports (deep/ folder contains visualization module)
# The visualization module is in deep/visualization/
_CURRENT_FILE = Path(__file__).resolve()
_DEEP_DIR = _CURRENT_FILE.parent.parent.parent.parent  # deep/ directory
if str(_DEEP_DIR) not in sys.path:
    sys.path.insert(0, str(_DEEP_DIR))

HAS_VIZ = False
_VIZ_IMPORT_ERROR = None

try:
    from visualization import (
        StandardizedCSVLogger,
        ConfusionMatrix,
        BatchVisualizer,
        plot_results,
        plot_all_curves,
    )
    HAS_VIZ = True
except ImportError as e:
    _VIZ_IMPORT_ERROR = str(e)
    print(f"Warning: Could not import visualization module: {e}")
except Exception as e:
    _VIZ_IMPORT_ERROR = str(e)
    print(f"Warning: Error importing visualization module: {e}")


class SimpleCSVLogger:
    """Simple CSV logger fallback when visualization module is not available."""
    
    def __init__(self, save_dir: Path, include_distance: bool = False):
        self.filepath = save_dir / 'results.csv'
        self.include_distance = include_distance
        self._initialized = False
        
        # Standard columns
        self.columns = [
            'epoch', 'train/total_loss', 'train/iou_loss', 'train/conf_loss', 
            'train/cls_loss', 'train/l1_loss', 'metrics/mAP50', 'metrics/mAP50-95', 'lr'
        ]
        if include_distance:
            self.columns.insert(-1, 'train/dist_loss')
    
    def log(self, metrics: Dict[str, Any], epoch: int = 0) -> None:
        import csv
        
        # Initialize with headers on first write
        if not self._initialized:
            with open(self.filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.columns)
                writer.writeheader()
            self._initialized = True
        
        # Prepare row
        row = {col: '' for col in self.columns}
        row['epoch'] = epoch
        
        for key, value in metrics.items():
            if key in row:
                row[key] = f'{value:.5f}' if isinstance(value, float) else str(value)
        
        # Write row
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writerow(row)


class YOLOXVisualization:
    """
    Visualization utilities for YOLOX training.
    """
    
    def __init__(
        self,
        save_dir: str,
        class_names: Optional[List[str]] = None,
        num_classes: int = 80,
        include_distance: bool = False,
    ):
        """
        Initialize YOLOX visualization.
        
        Args:
            save_dir: Directory to save visualizations
            class_names: List of class names
            num_classes: Number of classes
            include_distance: Whether model includes distance estimation
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.class_names = class_names or [f'class_{i}' for i in range(num_classes)]
        self.num_classes = num_classes
        self.include_distance = include_distance
        self.has_viz = HAS_VIZ
        
        # Initialize components if available
        if HAS_VIZ:
            self.csv_logger = StandardizedCSVLogger(
                save_dir=self.save_dir,
                include_distance=include_distance,
            )
            self.confusion_matrix = ConfusionMatrix(
                num_classes=num_classes,
                class_names=self.class_names,
            )
            self.batch_visualizer = BatchVisualizer(class_names=self.class_names)
            print(f"Full visualization module loaded successfully")
        else:
            # Use simple fallback CSV logger
            self.csv_logger = SimpleCSVLogger(
                save_dir=self.save_dir,
                include_distance=include_distance,
            )
            self.confusion_matrix = None
            self.batch_visualizer = None
            print(f"Using simple CSV logger (full visualization module not available: {_VIZ_IMPORT_ERROR})")
        
        # Track batch indices for visualization
        self._train_batch_saved = set()
        self._val_batch_saved = set()
        
        # Metrics accumulator for epoch
        self._epoch_metrics: Dict[str, List[float]] = {}
        
        # Store experiment config for args.yaml
        self._exp_config: Dict[str, Any] = {}
    
    def log_iter_metrics(
        self,
        metrics: Dict[str, float],
        lr: float,
        epoch: int,
        iteration: int,
    ) -> None:
        """
        Log iteration-level metrics.
        
        Args:
            metrics: Dictionary of loss values
            lr: Current learning rate
            epoch: Current epoch
            iteration: Current iteration
        """
        # Accumulate metrics (works for both full and simple logger)
        for key, value in metrics.items():
            if key not in self._epoch_metrics:
                self._epoch_metrics[key] = []
            self._epoch_metrics[key].append(value)
    
    def log_epoch_metrics(
        self,
        epoch: int,
        lr: float,
        ap50: Optional[float] = None,
        ap50_95: Optional[float] = None,
        precision: Optional[float] = None,
        recall: Optional[float] = None,
    ) -> None:
        """
        Log epoch-level metrics to CSV.
        
        Args:
            epoch: Current epoch
            lr: Learning rate
            ap50: AP at IoU 0.5
            ap50_95: AP at IoU 0.5:0.95
            precision: Precision
            recall: Recall
        """
        if self.csv_logger is None:
            return
        
        # Average iteration metrics
        epoch_avg = {}
        for key, values in self._epoch_metrics.items():
            if values:
                avg_key = f'train/{key}' if not key.startswith('train/') else key
                epoch_avg[avg_key] = sum(values) / len(values)
        
        # Clear for next epoch
        self._epoch_metrics = {}
        
        # Add validation metrics
        if ap50 is not None:
            epoch_avg['metrics/mAP50'] = ap50
        if ap50_95 is not None:
            epoch_avg['metrics/mAP50-95'] = ap50_95
        if precision is not None:
            epoch_avg['metrics/precision'] = precision
        if recall is not None:
            epoch_avg['metrics/recall'] = recall
        
        epoch_avg['lr'] = lr
        
        # Log to CSV
        self.csv_logger.log(epoch_avg, epoch=epoch)
    
    def save_training_batch(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        batch_idx: int,
    ) -> None:
        """
        Save visualization of a training batch.
        
        Args:
            images: Batch of images (N, C, H, W)
            targets: Batch of targets (batch_size, max_labels, 5+) with format [cls, cx, cy, w, h]
            batch_idx: Batch index
        """
        if not HAS_VIZ or self.batch_visualizer is None:
            return
        
        # Only save first 3 batches
        if batch_idx >= 3 or batch_idx in self._train_batch_saved:
            return
        
        self._train_batch_saved.add(batch_idx)
        
        save_path = self.save_dir / f'train_batch{batch_idx}.jpg'
        
        # Debug: print image and target info
        if batch_idx == 0:
            print(f"\n[DEBUG] Training batch {batch_idx}:")
            print(f"  Images shape: {images.shape}")
            print(f"  Images dtype: {images.dtype}")
            print(f"  Targets shape: {targets.shape}")
            print(f"  Targets dtype: {targets.dtype}")
            # Print first image's first few targets
            if len(targets) > 0:
                first_img_targets = targets[0]
                valid_count = 0
                for i, t in enumerate(first_img_targets):
                    if t[3] > 0 and t[4] > 0:  # w > 0 and h > 0
                        print(f"  Target {i}: cls={t[0]:.0f}, cx={t[1]:.1f}, cy={t[2]:.1f}, w={t[3]:.1f}, h={t[4]:.1f}")
                        valid_count += 1
                        if valid_count >= 3:
                            break
                print(f"  Total valid targets in first image: {sum(1 for t in first_img_targets if t[3] > 0 and t[4] > 0)}")
        
        # Convert targets to list format
        # YOLOX targets are [cls, cx, cy, w, h] in PIXEL coordinates (not normalized!)
        # The TrainTransform outputs pixel coords: boxes *= r_ (resize ratio)
        targets_list = self._convert_targets(targets, len(images))
        
        # Debug: print converted targets
        if batch_idx == 0:
            _, _, img_h, img_w = images.shape  # (N, C, H, W)
            print(f"  Image dimensions: {img_w}x{img_h}")
            if len(targets_list) > 0 and len(targets_list[0]) > 0:
                print(f"  Converted targets (first image, xyxy format):")
                out_of_bounds = 0
                for i, t in enumerate(targets_list[0]):
                    x1, y1, x2, y2 = t[1], t[2], t[3], t[4]
                    if i < 3:
                        print(f"    {i}: cls={t[0]:.0f}, x1={x1:.1f}, y1={y1:.1f}, x2={x2:.1f}, y2={y2:.1f}")
                    if x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h:
                        out_of_bounds += 1
                if out_of_bounds > 0:
                    print(f"  WARNING: {out_of_bounds} boxes out of bounds!")
        
        self.batch_visualizer.plot_batch(
            images=images,
            targets=targets_list,
            predictions=None,
            save_path=save_path,
            max_samples=16,
            normalized=False,  # YOLOX training targets are in pixel coordinates
        )
    
    def _convert_targets(self, targets: torch.Tensor, batch_size: int) -> List[np.ndarray]:
        """
        Convert YOLOX targets to list format for visualization.
        
        YOLOX targets from dataloader have shape: (batch_size, max_labels, 5+)
        Each label is: [class, cx, cy, w, h, ...] where cx, cy, w, h are in PIXEL coordinates
        (NOT normalized - the TrainTransform scales boxes by resize ratio)
        
        Returns list of arrays with format: [class, x1, y1, x2, y2] (pixel coords, xyxy format)
        """
        targets_np = targets.cpu().numpy() if isinstance(targets, torch.Tensor) else targets
        
        result = []
        
        if len(targets_np.shape) == 3:
            # Format: (batch_size, max_labels, num_cols)
            # Each label: [class, cx, cy, w, h, ...] - pixel coords
            for batch_idx in range(min(batch_size, len(targets_np))):
                batch_targets = targets_np[batch_idx]
                valid_targets = []
                
                for t in batch_targets:
                    # Check if this is a valid label (not padding)
                    # Padding labels typically have all zeros or class=-1
                    if len(t) >= 5:
                        cls, cx, cy, w, h = t[0], t[1], t[2], t[3], t[4]
                        # Valid labels have non-zero width and height
                        if w > 0 and h > 0:
                            # Convert from xywh (center) to xyxy format for BatchVisualizer
                            x1 = cx - w / 2
                            y1 = cy - h / 2
                            x2 = cx + w / 2
                            y2 = cy + h / 2
                            valid_targets.append([cls, x1, y1, x2, y2])
                
                result.append(np.array(valid_targets) if valid_targets else np.array([]).reshape(0, 5))
        
        elif len(targets_np.shape) == 2:
            # Legacy format: (num_labels, num_cols) with batch_idx as first column
            # Format: [batch_idx, class, cx, cy, w, h, ...]
            batch_result = [[] for _ in range(batch_size)]
            for t in targets_np:
                if len(t) >= 6:
                    batch_idx = int(t[0])
                    if 0 <= batch_idx < batch_size:
                        cls = t[1]
                        cx, cy, w, h = t[2], t[3], t[4], t[5]
                        if w > 0 and h > 0:
                            # Convert from xywh (center) to xyxy format
                            x1 = cx - w / 2
                            y1 = cy - h / 2
                            x2 = cx + w / 2
                            y2 = cy + h / 2
                            batch_result[batch_idx].append([cls, x1, y1, x2, y2])
            result = [np.array(r) if r else np.array([]).reshape(0, 5) for r in batch_result]
        
        # Pad if needed
        while len(result) < batch_size:
            result.append(np.array([]).reshape(0, 5))
        
        return result
    
    def update_confusion_matrix(
        self,
        detections: List[np.ndarray],
        labels: List[np.ndarray],
    ) -> None:
        """
        Update confusion matrix with batch results.
        
        Args:
            detections: List of detection arrays per image
            labels: List of label arrays per image
        """
        if not HAS_VIZ or self.confusion_matrix is None:
            return
        
        for det, lab in zip(detections, labels):
            self.confusion_matrix.process_batch(det, lab)
    
    def save_validation_batch(
        self,
        images: torch.Tensor,
        targets: List[np.ndarray],
        predictions: List[np.ndarray],
        batch_idx: int,
    ) -> None:
        """
        Save visualization of a validation batch (labels and predictions).
        
        Args:
            images: Batch of images (N, C, H, W)
            targets: List of ground truth arrays per image [cls, x1, y1, x2, y2]
            predictions: List of prediction arrays per image [x1, y1, x2, y2, conf, cls]
            batch_idx: Batch index
        """
        if not HAS_VIZ or self.batch_visualizer is None:
            return
        
        # Only save first 3 batches
        if batch_idx >= 3 or batch_idx in self._val_batch_saved:
            return
        
        self._val_batch_saved.add(batch_idx)
        
        # Save labels visualization
        labels_path = self.save_dir / f'val_batch{batch_idx}_labels.jpg'
        self.batch_visualizer.plot_batch(
            images=images,
            targets=targets,
            predictions=None,
            save_path=labels_path,
            max_samples=16,
            normalized=False,
        )
        
        # Save predictions visualization
        pred_path = self.save_dir / f'val_batch{batch_idx}_pred.jpg'
        # Convert predictions to targets format for visualization
        # Filter by minimum confidence to reduce visual clutter from low-confidence FPs
        vis_conf_threshold = 0.25  # Higher threshold for visualization than NMS
        pred_as_targets = []
        for pred in predictions:
            if len(pred) > 0:
                # pred format: [x1, y1, x2, y2, conf, cls]
                # Filter by confidence threshold for cleaner visualization
                high_conf_mask = pred[:, 4] >= vis_conf_threshold
                pred_filtered = pred[high_conf_mask]
                if len(pred_filtered) > 0:
                    # Convert to: [cls, x1, y1, x2, y2]
                    pred_converted = np.column_stack([
                        pred_filtered[:, 5],  # cls
                        pred_filtered[:, 0],  # x1
                        pred_filtered[:, 1],  # y1
                        pred_filtered[:, 2],  # x2
                        pred_filtered[:, 3],  # y2
                    ])
                    pred_as_targets.append(pred_converted)
                else:
                    pred_as_targets.append(np.array([]).reshape(0, 5))
            else:
                pred_as_targets.append(np.array([]).reshape(0, 5))
        
        self.batch_visualizer.plot_batch(
            images=images,
            targets=pred_as_targets,
            predictions=None,
            save_path=pred_path,
            max_samples=16,
            normalized=False,
        )
        print(f"Saved validation batch {batch_idx} visualizations")
    
    def set_exp_config(self, exp) -> None:
        """
        Store experiment configuration for args.yaml.
        
        Args:
            exp: Experiment object with configuration attributes
        """
        # Extract relevant config attributes
        config = {}
        config_attrs = [
            'exp_name', 'num_classes', 'depth', 'width', 'act',
            'input_size', 'test_size', 'data_dir', 'train_ann', 'val_ann',
            'max_epoch', 'warmup_epochs', 'no_aug_epochs', 'eval_interval',
            'basic_lr_per_img', 'weight_decay', 'momentum',
            'mosaic_prob', 'mixup_prob', 'hsv_prob', 'flip_prob',
            'degrees', 'translate', 'shear', 'mosaic_scale', 'mixup_scale',
            'enable_mixup', 'use_mosaic', 'test_conf', 'nmsthre',
            'use_radar', 'include_distance', 'min_distance', 'max_distance',
            'distance_loss_multiplier', 'use_l1_loss',
        ]
        
        for attr in config_attrs:
            if hasattr(exp, attr):
                value = getattr(exp, attr)
                # Convert tuples and other non-serializable types
                if isinstance(value, tuple):
                    value = list(value)
                config[attr] = value
        
        self._exp_config = config
    
    def generate_label_distribution(self, dataset) -> None:
        """
        Generate label distribution visualizations (labels.jpg, labels_correlogram.jpg).
        
        Args:
            dataset: Training dataset to analyze
        """
        if not HAS_VIZ:
            return
        
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib import patches
            
            # Collect all labels
            all_boxes = []  # [x_center, y_center, width, height] normalized
            all_classes = []
            
            # Sample dataset for label statistics
            max_samples = min(len(dataset), 5000)
            indices = np.random.choice(len(dataset), max_samples, replace=False)
            
            for idx in indices:
                try:
                    _, target, _, _ = dataset.pull_item(idx)
                    if target is not None and len(target) > 0:
                        # Target format: [cls, x, y, w, h, ...]
                        for t in target:
                            if len(t) >= 5:
                                cls = int(t[0])
                                cx, cy, w, h = t[1:5]
                                # Normalize to 0-1 range if not already
                                if cx > 1 or cy > 1 or w > 1 or h > 1:
                                    img_h, img_w = dataset.input_dim if hasattr(dataset, 'input_dim') else (640, 640)
                                    cx, w = cx / img_w, w / img_w
                                    cy, h = cy / img_h, h / img_h
                                all_boxes.append([cx, cy, w, h])
                                all_classes.append(cls)
                except Exception:
                    continue
            
            if not all_boxes:
                return
            
            boxes = np.array(all_boxes)
            classes = np.array(all_classes)
            
            # Create labels.jpg - class distribution and box statistics
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))
            
            # Class histogram
            ax = axes[0, 0]
            class_counts = np.bincount(classes.astype(int), minlength=self.num_classes)
            ax.bar(range(len(class_counts)), class_counts, color='steelblue')
            ax.set_xlabel('Class')
            ax.set_ylabel('Count')
            ax.set_title('Class Distribution')
            
            # Box size distribution (width vs height)
            ax = axes[0, 1]
            ax.scatter(boxes[:, 2], boxes[:, 3], alpha=0.3, s=1, c='steelblue')
            ax.set_xlabel('Width (normalized)')
            ax.set_ylabel('Height (normalized)')
            ax.set_title('Box Size Distribution')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            
            # Box center distribution
            ax = axes[1, 0]
            ax.hist2d(boxes[:, 0], boxes[:, 1], bins=50, cmap='Blues')
            ax.set_xlabel('X Center (normalized)')
            ax.set_ylabel('Y Center (normalized)')
            ax.set_title('Box Center Distribution')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.invert_yaxis()
            
            # Width/Height histogram
            ax = axes[1, 1]
            ax.hist(boxes[:, 2], bins=50, alpha=0.5, label='Width', color='blue')
            ax.hist(boxes[:, 3], bins=50, alpha=0.5, label='Height', color='orange')
            ax.set_xlabel('Size (normalized)')
            ax.set_ylabel('Count')
            ax.set_title('Width/Height Distribution')
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(self.save_dir / 'labels.jpg', dpi=200, bbox_inches='tight')
            plt.close()
            print(f"Saved label distribution to {self.save_dir / 'labels.jpg'}")
            
            # Create labels_correlogram.jpg
            fig, axes = plt.subplots(4, 4, figsize=(12, 12))
            
            labels = ['x_center', 'y_center', 'width', 'height']
            for i in range(4):
                for j in range(4):
                    ax = axes[i, j]
                    if i == j:
                        # Diagonal - histogram
                        ax.hist(boxes[:, i], bins=50, color='steelblue', alpha=0.7)
                    else:
                        # Off-diagonal - scatter
                        ax.scatter(boxes[:, j], boxes[:, i], alpha=0.1, s=1, c='steelblue')
                    
                    if i == 3:
                        ax.set_xlabel(labels[j])
                    if j == 0:
                        ax.set_ylabel(labels[i])
            
            plt.tight_layout()
            plt.savefig(self.save_dir / 'labels_correlogram.jpg', dpi=200, bbox_inches='tight')
            plt.close()
            print(f"Saved labels correlogram to {self.save_dir / 'labels_correlogram.jpg'}")
            
        except Exception as e:
            print(f"Warning: Could not generate label distribution: {e}")
    
    def finalize(self, distance_metrics: Optional[Dict[str, float]] = None) -> None:
        """
        Generate final visualizations at end of training.
        Generates all visualizations to match YOLOv8 output format.
        
        Args:
            distance_metrics: Optional dict with distance metrics for distance estimation tasks
        """
        csv_path = self.save_dir / 'results.csv'
        
        if HAS_VIZ:
            # Use full visualization module
            if csv_path.exists():
                plot_results(csv_path, self.save_dir)
            
            # Handle confusion matrix vs distance metrics
            if self.include_distance:
                # For distance estimation tasks, print distance metrics instead of confusion matrix
                self._print_distance_metrics_summary(distance_metrics)
            elif self.confusion_matrix is not None:
                # For detection tasks, show confusion matrix
                has_data = hasattr(self.confusion_matrix, 'matrix') and self.confusion_matrix.matrix.sum() > 0
                if has_data:
                    self.confusion_matrix.plot(self.save_dir, self.class_names)
                    self.confusion_matrix.print_summary()
                else:
                    # Generate a placeholder confusion matrix with warning
                    self._generate_placeholder_confusion_matrix()
        else:
            # Generate simple plot without full visualization module
            if csv_path.exists():
                self._simple_plot_results(csv_path)
        
        # Save args.yaml
        self._save_args_yaml()
        
        print(f"Results CSV saved to: {csv_path}")
    
    def _print_distance_metrics_summary(self, metrics: Optional[Dict[str, float]] = None) -> None:
        """
        Print distance estimation metrics summary.
        
        Args:
            metrics: Dict with distance metrics (mae, rmse, median_ae, etc.)
        """
        print("\n" + "=" * 70)
        print("Distance Estimation Metrics")
        print("=" * 70)
        
        if metrics and metrics.get('samples', 0) > 0:
            print(f"  Samples:      {int(metrics.get('samples', 0))}")
            print(f"  MAE:          {metrics.get('mae', 0):.3f} m")
            print(f"  RMSE:         {metrics.get('rmse', 0):.3f} m")
            print(f"  Median AE:    {metrics.get('median_ae', 0):.3f} m")
            print(f"  Max Error:    {metrics.get('max_error', 0):.3f} m")
            print(f"  Min Error:    {metrics.get('min_error', 0):.3f} m")
            print(f"  Acc @ 0.5m:   {metrics.get('acc_05', 0):.1f}%")
            print(f"  Acc @ 1.0m:   {metrics.get('acc_10', 0):.1f}%")
            print(f"  Acc @ 2.0m:   {metrics.get('acc_20', 0):.1f}%")
        else:
            print("  No distance predictions matched with ground truth.")
            print("  This may be due to:")
            print("    - Model undertrained (too few epochs)")
            print("    - No detections produced (high confidence threshold)")
            print("    - Missing ground truth distance labels")
        
        print("=" * 70)
    
    def _save_args_yaml(self) -> None:
        """Save training configuration to args.yaml."""
        if not self._exp_config:
            return
        
        try:
            yaml_path = self.save_dir / 'args.yaml'
            with open(yaml_path, 'w') as f:
                yaml.dump(self._exp_config, f, default_flow_style=False, sort_keys=False)
            print(f"Saved args to {yaml_path}")
        except Exception as e:
            print(f"Warning: Could not save args.yaml: {e}")
    
    def _generate_placeholder_confusion_matrix(self) -> None:
        """Generate a placeholder confusion matrix when no evaluation data is available."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Create a simple placeholder
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # Empty confusion matrix
            matrix = np.zeros((self.num_classes, self.num_classes))
            im = ax.imshow(matrix, cmap='Blues')
            
            ax.set_title('Confusion Matrix (No detections)')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            
            if self.num_classes <= 10:
                ax.set_xticks(range(self.num_classes))
                ax.set_yticks(range(self.num_classes))
                ax.set_xticklabels(self.class_names[:self.num_classes], rotation=45, ha='right')
                ax.set_yticklabels(self.class_names[:self.num_classes])
            
            plt.colorbar(im)
            plt.tight_layout()
            
            # Save both normalized and non-normalized versions
            plt.savefig(self.save_dir / 'confusion_matrix.png', dpi=200, bbox_inches='tight')
            plt.savefig(self.save_dir / 'confusion_matrix_normalized.png', dpi=200, bbox_inches='tight')
            plt.close()
            
            print(f"Saved confusion matrix to {self.save_dir / 'confusion_matrix.png'}")
            print(f"Saved confusion matrix to {self.save_dir / 'confusion_matrix_normalized.png'}")
            
            # Print summary
            print("\n" + "=" * 60)
            print("Confusion Matrix Summary")
            print("=" * 60)
            print(f"{'Class':<20} {'TP':<8} {'FP':<8} {'FN':<8} {'Precision':<10} {'Recall':<10}")
            print("-" * 60)
            for i, name in enumerate(self.class_names[:self.num_classes]):
                print(f"{name:<20} {0:<8} {0:<8} {0:<8} {0.000:<10.3f} {0.000:<10.3f}")
            print("-" * 60)
            print(f"{'Total/Mean':<20} {0:<8} {0:<8} {0:<8} {0.000:<10.3f} {0.000:<10.3f}")
            print("=" * 60)
            print("\nNote: No detections matched ground truth. This may be due to:")
            print("  - Model undertrained (only few epochs)")
            print("  - High confidence threshold")
            print("  - Misaligned predictions")
            
        except Exception as e:
            print(f"Warning: Could not generate placeholder confusion matrix: {e}")

    def _simple_plot_results(self, csv_path: Path) -> None:
        """Generate a simple results plot using matplotlib directly."""
        try:
            import pandas as pd
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            
            df = pd.read_csv(csv_path)
            
            # Create figure
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            # Plot loss columns if they exist
            loss_cols = [col for col in df.columns if 'loss' in col.lower()]
            for i, col in enumerate(loss_cols[:4]):
                if i < len(axes):
                    axes[i].plot(df['epoch'] if 'epoch' in df.columns else range(len(df)), 
                                df[col], 'b-', linewidth=2)
                    axes[i].set_title(col.replace('train/', '').replace('_', ' ').title())
                    axes[i].set_xlabel('Epoch')
                    axes[i].set_ylabel('Loss')
                    axes[i].grid(True, alpha=0.3)
            
            # Plot mAP if exists
            map_cols = [col for col in df.columns if 'mAP' in col or 'map' in col.lower()]
            for j, col in enumerate(map_cols[:2]):
                idx = 4 + j
                if idx < len(axes):
                    axes[idx].plot(df['epoch'] if 'epoch' in df.columns else range(len(df)), 
                                  df[col], 'g-', linewidth=2)
                    axes[idx].set_title(col)
                    axes[idx].set_xlabel('Epoch')
                    axes[idx].set_ylabel('mAP')
                    axes[idx].set_ylim(0, 1)
                    axes[idx].grid(True, alpha=0.3)
            
            # Hide unused axes
            for i in range(len(loss_cols) + len(map_cols), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            save_path = self.save_dir / 'results.png'
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close()
            print(f"Results plot saved to: {save_path}")
            
        except Exception as e:
            print(f"Warning: Could not generate results plot: {e}")
    
    def reset_epoch(self) -> None:
        """Reset per-epoch state."""
        self._epoch_metrics = {}


def integrate_with_trainer(trainer, class_names: Optional[List[str]] = None):
    """
    Integrate visualization with a YOLOX trainer instance.
    
    Args:
        trainer: YOLOX Trainer instance
        class_names: Optional list of class names
        
    Returns:
        YOLOXVisualization instance
    """
    save_dir = trainer.file_name
    num_classes = trainer.exp.num_classes if hasattr(trainer.exp, 'num_classes') else 80
    include_distance = hasattr(trainer.exp, 'use_distance_loss') and trainer.exp.use_distance_loss
    
    viz = YOLOXVisualization(
        save_dir=save_dir,
        class_names=class_names,
        num_classes=num_classes,
        include_distance=include_distance,
    )
    
    return viz












