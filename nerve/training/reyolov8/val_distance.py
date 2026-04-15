"""
EventVideoDistanceValidator: Complete validator for REYOLOv8 with distance estimation.

Extends EventVideoDetectionValidator to add distance metrics:
- Mean Absolute Error (MAE) / ADE (Average Distance Error)
- Root Mean Square Error (RMSE)
- Median Absolute Error
- Distance accuracy at thresholds (0.5m, 1.0m, 2.0m)

Uses IoU-based matching (consistent with YOLOX/YOLOv8 evaluation methodology)
for reliable and comparable distance metrics.

Author: REYOLOv8 Distance Integration
"""

import numpy as np
import torch
from pathlib import Path
import json
from ultralytics.yolo.utils import LOGGER
import val  # Import base validator


def box_iou(box1, box2, eps=1e-7):
    """
    Compute IoU between two sets of boxes.
    
    Args:
        box1: Tensor of shape (N, 4) in xyxy format
        box2: Tensor of shape (M, 4) in xyxy format
        eps: Small value to avoid division by zero
        
    Returns:
        IoU matrix of shape (N, M)
    """
    # Get areas
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    
    # Get intersection coordinates
    inter_x1 = torch.max(box1[:, None, 0], box2[:, 0])
    inter_y1 = torch.max(box1[:, None, 1], box2[:, 1])
    inter_x2 = torch.min(box1[:, None, 2], box2[:, 2])
    inter_y2 = torch.min(box1[:, None, 3], box2[:, 3])
    
    # Get intersection area
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h
    
    # IoU = intersection / union
    union = area1[:, None] + area2 - inter_area
    iou = inter_area / (union + eps)
    
    return iou


def xywh2xyxy(boxes):
    """Convert boxes from xywh (center) format to xyxy format."""
    if len(boxes) == 0:
        return boxes
    
    x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    
    return torch.stack([x1, y1, x2, y2], dim=1)


class EventVideoDistanceValidator(val.EventVideoDetectionValidator):
    """
    Extended validator for REYOLOv8 with distance estimation.
    
    Inherits from EventVideoDetectionValidator and adds distance-specific metrics.
    Compatible with REYOLOv8's recurrent architecture and event camera data.
    """
    
    def __init__(self, video_config, dataloader, save_dir, logger, args):
        """
        Initialize distance validator.
        
        Args:
            video_config: Video configuration dict (clip_length, clip_stride, channels)
            dataloader: Validation dataloader
            save_dir: Directory to save results
            logger: Logger instance
            args: Training arguments
        """
        # Use keyword arguments to match EventVideoDetectionValidator signature
        # which has: (video_config, dataloader, save_dir, pbar, logger, args)
        super().__init__(video_config=video_config, dataloader=dataloader, 
                         save_dir=save_dir, pbar=None, logger=logger, args=args)
        
        # Distance metrics storage
        self.distance_errors = []
        self.distance_predictions = []
        self.distance_targets = []
        self.distance_valid_count = 0
        
        LOGGER.info("EventVideoDistanceValidator initialized with distance metrics")
    
    def __call__(self, trainer=None, model=None):
        """
        Run validation with distance metrics.
        
        Args:
            trainer: Optional trainer instance (if called during training)
            model: Optional model instance (if called standalone)
            
        Returns:
            metrics: Dictionary with detection and distance metrics
        """
        # Reset distance metrics for this validation run
        self.distance_errors = []
        self.distance_predictions = []
        self.distance_targets = []
        self.distance_valid_count = 0
        
        # Run base detection validation
        LOGGER.info("Running detection validation...")
        metrics = super().__call__(trainer, model)
        
        # Compute and add distance metrics if we collected any
        if len(self.distance_errors) > 0:
            LOGGER.info(f"Computing distance metrics from {len(self.distance_errors)} predictions...")
            distance_metrics = self.compute_distance_metrics()
            metrics.update(distance_metrics)
            
            # Log distance metrics to console and file
            self.log_distance_metrics(distance_metrics)
        else:
            LOGGER.warning("No distance predictions collected during validation")
        
        return metrics
    
    def update_metrics(self, preds, batch_, batch, sequence_mask, T):
        """
        Update detection and distance metrics.
        
        Args:
            preds: Model predictions
            batch_: Preprocessed batch data
            batch: Full batch data with ground truth
            sequence_mask: Mask for valid sequence indices
            T: Current timestep
        """
        # Update detection metrics (base class)
        super().update_metrics(preds, batch_, batch, sequence_mask, T)
        
        # Update distance metrics if available
        if 'distances' in batch:
            self.update_distance_metrics(preds, batch, sequence_mask)
    
    def update_distance_metrics(self, preds, batch, sequence_mask):
        """
        Extract and store distance predictions and targets using IoU-based matching.
        
        This method uses proper IoU-based matching (consistent with YOLOX/YOLOv8
        evaluation methodology) to ensure reliable and comparable distance metrics.
        
        Args:
            preds: Model predictions - can be tuple (y, x) or just predictions
            batch: Batch data containing ground truth distances and boxes
            sequence_mask: Mask for valid sequence indices
        """
        # Handle different prediction formats
        if isinstance(preds, tuple):
            # Format: (predictions, features) or (det_preds, dist_preds)
            preds = preds[0]  # Take predictions
        
        # Check if predictions include distance channel
        # Expected format after NMS: [x1, y1, x2, y2, conf, cls, distance] or
        #                            [cx, cy, w, h, conf, cls, distance]
        if isinstance(preds, (list, tuple)):
            for si, pred in enumerate(preds):
                if pred is None or len(pred) == 0:
                    continue
                
                # Check if distance channel exists (7th column)
                if pred.shape[-1] < 7:
                    continue
                
                device = pred.device
                
                # Get prediction boxes and distances
                pred_boxes = pred[:, :4]  # First 4 columns are box coordinates
                pred_distances = pred[:, 6]  # Distance is 7th column
                
                # Get ground truth for this batch item using sequence_mask
                if 'distances' not in batch or 'bboxes' not in batch:
                    continue
                
                # Find which ground truth items belong to this batch item
                batch_idx = batch['batch_idx']
                if sequence_mask is not None:
                    batch_idx = batch_idx[sequence_mask]
                    gt_distances = batch['distances'][sequence_mask]
                    gt_boxes = batch['bboxes'][sequence_mask]
                else:
                    gt_distances = batch['distances']
                    gt_boxes = batch['bboxes']
                
                idx = batch_idx == si
                
                if not idx.any():
                    continue
                
                gt_distances_i = gt_distances[idx].to(device)
                gt_boxes_i = gt_boxes[idx].to(device)
                
                if len(gt_boxes_i) == 0:
                    continue
                
                # Convert GT boxes to xyxy if they are in xywh format (normalized)
                # GT boxes are typically normalized xywh format
                # Scale to match prediction format
                if hasattr(self, 'args') and hasattr(self.args, 'imgsz'):
                    imgsz = self.args.imgsz
                    if isinstance(imgsz, (list, tuple)):
                        height, width = imgsz[0], imgsz[1] if len(imgsz) > 1 else imgsz[0]
                    else:
                        height = width = imgsz
                else:
                    height = width = 640  # Default
                
                # Convert normalized xywh to pixel xyxy
                gt_boxes_xyxy = xywh2xyxy(gt_boxes_i.clone())
                gt_boxes_xyxy[:, [0, 2]] *= width
                gt_boxes_xyxy[:, [1, 3]] *= height
                
                # Check if predictions are in xywh or xyxy format
                # If max value of boxes is small (< 2), likely normalized - need to convert
                pred_boxes_xyxy = pred_boxes
                if pred_boxes[:, 2:4].max() < 2:  # Likely xywh normalized
                    pred_boxes_xyxy = xywh2xyxy(pred_boxes.clone())
                    pred_boxes_xyxy[:, [0, 2]] *= width
                    pred_boxes_xyxy[:, [1, 3]] *= height
                
                # Compute IoU between predictions and ground truth
                iou = box_iou(pred_boxes_xyxy, gt_boxes_xyxy)
                
                if iou.numel() == 0:
                    continue
                
                # Match using IoU threshold (0.5, consistent with COCO evaluation)
                iou_threshold = 0.5
                
                # For each prediction, find best matching GT
                max_iou, matched_gt_idx = iou.max(dim=1)
                valid_matches = max_iou > iou_threshold
                
                if not valid_matches.any():
                    continue
                
                # Get matched predictions and GT distances
                matched_pred_dist = pred_distances[valid_matches]
                matched_gt_idx_valid = matched_gt_idx[valid_matches]
                matched_gt_dist = gt_distances_i[matched_gt_idx_valid]
                
                # Filter out invalid GT distances (-1.0 = no radar data)
                valid_gt_mask = matched_gt_dist >= 0.0
                
                if not valid_gt_mask.any():
                    continue
                
                matched_pred_dist = matched_pred_dist[valid_gt_mask]
                matched_gt_dist = matched_gt_dist[valid_gt_mask]
                
                # Store errors for metric computation
                for pred_d, gt_d in zip(matched_pred_dist.cpu().numpy(), 
                                        matched_gt_dist.cpu().numpy()):
                    self.distance_predictions.append(float(pred_d))
                    self.distance_targets.append(float(gt_d))
                    self.distance_errors.append(abs(float(pred_d) - float(gt_d)))
                    self.distance_valid_count += 1
    
    def compute_distance_metrics(self):
        """
        Compute comprehensive distance metrics.
        
        Computes metrics consistent with YOLOX evaluation:
        - ADE (Average Distance Error) = MAE = Mean Absolute Error
        - RMSE (Root Mean Square Error)
        - Accuracy at various thresholds
        
        Returns:
            metrics: Dictionary with distance metrics
        """
        if len(self.distance_errors) == 0:
            return {}
        
        errors = np.array(self.distance_errors)
        preds = np.array(self.distance_predictions)
        targets = np.array(self.distance_targets)
        
        # Compute signed errors for RMSE
        signed_errors = preds - targets
        
        mae = float(np.mean(errors))
        rmse = float(np.sqrt(np.mean(signed_errors ** 2)))
        
        metrics = {
            # Primary metrics (consistent with YOLOX ADE)
            'distance/ADE': mae,  # Average Distance Error = MAE (YOLOX naming)
            'distance/MAE': mae,  # Also report as MAE for clarity
            'distance/RMSE': rmse,
            
            # Additional statistics
            'distance/MedianAE': float(np.median(errors)),
            'distance/MaxError': float(np.max(errors)),
            'distance/MinError': float(np.min(errors)),
            'distance/StdError': float(np.std(errors)),
            
            # Mean Relative Error
            'distance/MRE': float(np.mean(errors / (targets + 1e-6))),
            
            # Accuracy at thresholds
            'distance/Acc@0.5m': float(np.mean(errors <= 0.5)),
            'distance/Acc@1.0m': float(np.mean(errors <= 1.0)),
            'distance/Acc@2.0m': float(np.mean(errors <= 2.0)),
            
            # Sample information
            'distance/NumSamples': len(errors),
            'distance/MeanPrediction': float(np.mean(preds)),
            'distance/MeanTarget': float(np.mean(targets)),
        }
        
        return metrics
    
    def log_distance_metrics(self, metrics):
        """
        Log distance metrics to console and save to file.
        
        Args:
            metrics: Dictionary with distance metrics
        """
        LOGGER.info("\n" + "="*70)
        LOGGER.info("Distance Estimation Metrics (IoU@0.5 matched pairs)")
        LOGGER.info("="*70)
        LOGGER.info(f"  Matched Samples:   {metrics.get('distance/NumSamples', 0)}")
        LOGGER.info("")
        LOGGER.info("  [Primary Metrics - Use for model comparison]")
        LOGGER.info(f"  ADE (Avg Dist Err): {metrics.get('distance/ADE', 0):.3f} m  <- Compare with YOLOX")
        LOGGER.info(f"  RMSE:               {metrics.get('distance/RMSE', 0):.3f} m")
        LOGGER.info(f"  MRE:                {metrics.get('distance/MRE', 0):.3f}")
        LOGGER.info("")
        LOGGER.info("  [Additional Statistics]")
        LOGGER.info(f"  Median AE:          {metrics.get('distance/MedianAE', 0):.3f} m")
        LOGGER.info(f"  Std Error:          {metrics.get('distance/StdError', 0):.3f} m")
        LOGGER.info(f"  Min Error:          {metrics.get('distance/MinError', 0):.3f} m")
        LOGGER.info(f"  Max Error:          {metrics.get('distance/MaxError', 0):.3f} m")
        LOGGER.info("")
        LOGGER.info("  [Accuracy at Thresholds]")
        LOGGER.info(f"  Acc @ 0.5m:         {metrics.get('distance/Acc@0.5m', 0)*100:.1f}%")
        LOGGER.info(f"  Acc @ 1.0m:         {metrics.get('distance/Acc@1.0m', 0)*100:.1f}%")
        LOGGER.info(f"  Acc @ 2.0m:         {metrics.get('distance/Acc@2.0m', 0)*100:.1f}%")
        LOGGER.info("="*70 + "\n")
        
        # Save distance metrics to JSON file
        save_path = Path(self.save_dir) / 'distance_metrics.json'
        try:
            with open(save_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            LOGGER.info(f"Distance metrics saved to: {save_path}")
        except Exception as e:
            LOGGER.warning(f"Failed to save distance metrics: {e}")
        
        # Also save detailed predictions for analysis
        if len(self.distance_predictions) > 0:
            detailed_path = Path(self.save_dir) / 'distance_predictions.npz'
            try:
                np.savez(
                    detailed_path,
                    predictions=np.array(self.distance_predictions),
                    targets=np.array(self.distance_targets),
                    errors=np.array(self.distance_errors)
                )
                LOGGER.info(f"Detailed distance data saved to: {detailed_path}")
            except Exception as e:
                LOGGER.warning(f"Failed to save detailed distance data: {e}")
    
    def get_desc(self):
        """Return formatted string describing validation metrics."""
        desc = super().get_desc()
        
        # Add distance metrics to description if available
        if len(self.distance_errors) > 0:
            mae = np.mean(self.distance_errors)
            desc += f", MAE: {mae:.3f}m"
        
        return desc


# Convenience function for standalone validation
def validate_with_distance(model_path, data_yaml, video_config, **kwargs):
    """
    Standalone validation function for REYOLOv8 with distance estimation.
    
    Args:
        model_path: Path to trained model checkpoint
        data_yaml: Path to dataset YAML configuration
        video_config: Video configuration dict (clip_length, clip_stride, channels)
        **kwargs: Additional arguments (batch_size, device, etc.)
        
    Returns:
        metrics: Validation metrics including distance
    """
    from ultralytics.yolo.cfg import get_cfg
    from ultralytics.nn.tasks import attempt_load_weights
    
    # Load model
    model = attempt_load_weights(model_path)
    model.eval()
    
    # Setup args
    args = get_cfg(DEFAULT_CFG)
    args.data = data_yaml
    args.batch = kwargs.get('batch_size', 16)
    args.device = kwargs.get('device', '0')
    args.half = kwargs.get('half', True)
    args.plots = kwargs.get('plots', True)
    
    # Create validator
    validator = EventVideoDistanceValidator(
        video_config=video_config,
        dataloader=None,
        save_dir=kwargs.get('save_dir', Path('runs/val')),
        logger=LOGGER,
        args=args
    )
    
    # Run validation
    metrics = validator(model=model)
    
    return metrics




