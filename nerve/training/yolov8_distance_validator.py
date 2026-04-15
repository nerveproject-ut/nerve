"""
Custom YOLOv8 Validator with Distance Estimation Metrics.
Evaluates detection performance and distance estimation accuracy.

Uses COCO-style evaluation for distance metrics (ADE) to ensure
consistency with YOLOX evaluation methodology.
"""

import torch
import numpy as np
from pathlib import Path
import json
import tempfile

try:
    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.utils import LOGGER, ops
    from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, box_iou
except ImportError:
    raise ImportError("YOLOv8 (ultralytics) is not installed. Install with: pip install ultralytics")

# Import Custom_COCOeval for standardized distance evaluation
try:
    from nerve.training.custom_cocoeval import Custom_COCOeval
    COCO_EVAL_AVAILABLE = True
except ImportError:
    COCO_EVAL_AVAILABLE = False
    LOGGER.warning("Custom_COCOeval not available. Using simple IoU-based distance matching.")


class DistanceDetectionValidator(DetectionValidator):
    """
    Custom validator for YOLOv8 with distance estimation.
    Extends standard detection validation with distance metrics.
    
    Uses COCO-style evaluation (Custom_COCOeval) for ADE metric to ensure
    consistency with YOLOX evaluation methodology.
    """
    
    def __init__(
        self,
        dataloader=None,
        save_dir=None,
        args=None,
        _callbacks=None,
        process_distance=False,
        distance_from_head=True,
        min_dist=0.0,
        max_dist=10.0
    ):
        """
        Initialize distance validator.
        
        Args:
            dataloader: Validation dataloader
            save_dir: Directory to save results
            args: Arguments
            _callbacks: Callbacks
            process_distance: Enable distance evaluation
            distance_from_head: Predict distance from head vs radar
            min_dist: Minimum distance
            max_dist: Maximum distance
        """
        self.process_distance = process_distance
        self.distance_from_head = distance_from_head
        self.min_dist = min_dist
        self.max_dist = max_dist
        
        # Initialize distance metrics storage (simple IoU-based matching)
        self.distance_errors = []
        self.distance_abs_errors = []
        self.distance_rel_errors = []
        
        # COCO-style evaluation data (for standardized ADE)
        self.coco_predictions = []  # List of predictions in COCO format
        self.coco_gt_annotations = None  # Will be set from dataloader
        self._coco_ade = None  # ADE from COCO-style evaluation
        self._coco_dir = None  # Distance invalidity ratio
        
        super().__init__(dataloader, save_dir, args, _callbacks)
        
        if process_distance:
            LOGGER.info(f"Distance validation enabled: range [{min_dist}, {max_dist}]m")
            if COCO_EVAL_AVAILABLE:
                LOGGER.info("Using COCO-style evaluation for standardized ADE metric")
    
    def init_metrics(self, model):
        """Initialize metrics including distance metrics."""
        super().init_metrics(model)
        
        if self.process_distance:
            # Simple IoU-based matching metrics
            self.distance_errors = []
            self.distance_abs_errors = []
            self.distance_rel_errors = []
            
            # COCO-style evaluation data
            self.coco_predictions = []
            self._coco_ade = None
            self._coco_dir = None
    
    def postprocess(self, preds):
        """Post-process predictions with NMS, delegating to the parent implementation."""
        return super().postprocess(preds)
    
    def update_metrics(self, preds, batch):
        """
        Update metrics with predictions and batch.
        Includes distance metrics if enabled.
        """
        # Standard detection metrics
        super().update_metrics(preds, batch)
        
        # Distance metrics
        if self.process_distance and 'distances' in batch:
            self._update_distance_metrics(preds, batch)
    
    def _update_distance_metrics(self, preds, batch):
        """
        Update distance-specific metrics.
        
        Collects data for both:
        1. Simple IoU-based matching (for backward compatibility)
        2. COCO-style evaluation (for standardized ADE metric)
        
        Args:
            preds: Predictions with distance - list of tensors (one per image)
                   Each tensor shape: [num_preds, 6] = [x1, y1, x2, y2, conf, cls] + optional distance
                   or [num_preds, 7] = [x1, y1, x2, y2, conf, cls, distance]
            batch: Batch data with ground truth distances
        """
        if 'distances' not in batch:
            return  # No ground truth distances to compare against
        
        gt_distances = batch['distances']
        gt_boxes = batch['bboxes']
        batch_idx = batch['batch_idx']
        
        # Get image IDs if available (for COCO evaluation)
        img_ids = batch.get('im_file', None)  # List of image paths
        
        # Handle list of predictions (one per image in batch)
        if not isinstance(preds, list):
            return
        
        for si, pred in enumerate(preds):
            if pred is None:
                continue
            
            # Handle both dict-format (ultralytics 8.4+) and tensor-format preds
            if isinstance(pred, dict):
                pred_boxes = pred["bboxes"]
                pred_confs = pred["conf"]
                pred_cls = pred["cls"]
                extra = pred.get("extra", None)
                if extra is None or extra.shape[-1] < 1:
                    continue
                pred_distances = extra[:, 0]
            elif hasattr(pred, 'shape'):
                if len(pred) == 0 or pred.shape[-1] < 7:
                    continue
                pred_boxes = pred[:, :4]
                pred_confs = pred[:, 4]
                pred_cls = pred[:, 5]
                pred_distances = pred[:, 6]
            else:
                continue
            
            if len(pred_boxes) == 0:
                continue
            
            device = pred_boxes.device
            gt_boxes_dev = gt_boxes.to(device) if gt_boxes.device != device else gt_boxes
            gt_distances_dev = gt_distances.to(device) if gt_distances.device != device else gt_distances
            batch_idx_dev = batch_idx.to(device) if batch_idx.device != device else batch_idx
            
            idx = batch_idx_dev == si
            gt_boxes_i = gt_boxes_dev[idx]
            gt_distances_i = gt_distances_dev[idx]
            
            if len(gt_boxes_i) == 0:
                continue
            
            self._compute_distance_errors(
                pred_boxes, pred_distances, 
                gt_boxes_i, gt_distances_i
            )
            
            if COCO_EVAL_AVAILABLE:
                img_id = si
                if img_ids is not None and si < len(img_ids):
                    img_path = img_ids[si] if isinstance(img_ids, list) else str(img_ids)
                    try:
                        img_id = int(Path(img_path).stem.split('_')[-1])
                    except (ValueError, IndexError):
                        img_id = hash(img_path) % (10**9)
                
                self._collect_coco_predictions(
                    pred_boxes, pred_confs, pred_cls, pred_distances,
                    img_id, gt_boxes_i, gt_distances_i
                )
    
    def _collect_coco_predictions(self, pred_boxes, pred_confs, pred_cls, pred_distances,
                                   img_id, gt_boxes, gt_distances):
        """
        Collect predictions in COCO format for standardized evaluation.
        
        Args:
            pred_boxes: Predicted boxes [N, 4] in xyxy format
            pred_confs: Prediction confidences [N]
            pred_cls: Predicted classes [N]
            pred_distances: Predicted distances [N]
            img_id: Image ID
            gt_boxes: Ground truth boxes (for storing GT)
            gt_distances: Ground truth distances
        """
        height, width = self.args.imgsz, self.args.imgsz
        
        for i in range(len(pred_boxes)):
            box = pred_boxes[i].cpu().numpy()
            # Convert xyxy to xywh for COCO format
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            
            pred_data = {
                "image_id": int(img_id),
                "category_id": int(pred_cls[i].cpu().item()) + 1,  # Convert 0-indexed YOLO to 1-indexed COCO
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "score": float(pred_confs[i].cpu().item()),
                "distance": float(pred_distances[i].cpu().item()),
            }
            self.coco_predictions.append(pred_data)
    
    def _compute_distance_errors(self, pred_boxes, pred_distances, gt_boxes, gt_distances):
        """
        Compute distance errors between predictions and ground truth.
        
        Args:
            pred_boxes: Predicted boxes [N, 4]
            pred_distances: Predicted distances [N]
            gt_boxes: Ground truth boxes [M, 4]
            gt_distances: Ground truth distances [M]
        """
        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            return
        
        # Convert GT boxes to xyxy if needed (they might be in xywh format)
        # The GT boxes from batch are typically in xywh normalized format
        # Scale them to match prediction format
        height, width = self.args.imgsz, self.args.imgsz
        if hasattr(gt_boxes, 'shape') and gt_boxes.shape[-1] == 4:
            # Convert normalized xywh to xyxy pixels
            gt_boxes_xyxy = ops.xywh2xyxy(gt_boxes.clone())
            gt_boxes_xyxy[:, [0, 2]] *= width
            gt_boxes_xyxy[:, [1, 3]] *= height
        else:
            gt_boxes_xyxy = gt_boxes
        
        # Compute IoU
        iou = box_iou(pred_boxes, gt_boxes_xyxy)
        
        # Match using IoU threshold
        iou_threshold = 0.5
        if iou.numel() == 0:
            return
            
        matched_gt_idx = torch.argmax(iou, dim=1)
        max_ious = torch.max(iou, dim=1)[0]
        
        # Only compute errors for matched predictions
        valid_matches = max_ious > iou_threshold
        
        if not valid_matches.any():
            return
        
        matched_pred_dist = pred_distances[valid_matches]
        matched_gt_dist = gt_distances[matched_gt_idx[valid_matches]]
        
        # CRITICAL: Filter out invalid ground truth distances (-1.0 = no radar data)
        valid_gt_mask = matched_gt_dist >= 0.0
        
        if not valid_gt_mask.any():
            return
        
        matched_pred_dist = matched_pred_dist[valid_gt_mask]
        matched_gt_dist = matched_gt_dist[valid_gt_mask]
        
        # Compute distance errors
        errors = matched_pred_dist - matched_gt_dist
        abs_errors = torch.abs(errors)
        
        # Relative errors (avoid division by zero)
        rel_errors = abs_errors / (matched_gt_dist + 1e-6)
        
        # Store errors
        self.distance_errors.extend(errors.cpu().numpy().tolist())
        self.distance_abs_errors.extend(abs_errors.cpu().numpy().tolist())
        self.distance_rel_errors.extend(rel_errors.cpu().numpy().tolist())
    
    def _run_coco_distance_evaluation(self):
        """
        Run COCO-style distance evaluation using Custom_COCOeval.
        
        This provides the standardized ADE (Average Distance Error) metric
        that is comparable with YOLOX evaluation.
        
        Returns:
            tuple: (ade, dir) - Average Distance Error and Distance Invalidity Ratio
        """
        if not COCO_EVAL_AVAILABLE or len(self.coco_predictions) == 0:
            return None, None
        
        try:
            from pycocotools.coco import COCO
            
            # Check if dataloader has COCO annotations
            if not hasattr(self, 'dataloader') or self.dataloader is None:
                LOGGER.warning("No dataloader available for COCO evaluation")
                return None, None
            
            dataset = self.dataloader.dataset
            
            # Try to get COCO GT from dataset
            coco_gt = None
            if hasattr(dataset, 'coco'):
                coco_gt = dataset.coco
            elif hasattr(dataset, 'data') and hasattr(dataset.data, 'get'):
                # Try loading from data.yaml path
                pass
            
            if coco_gt is None:
                LOGGER.warning("COCO ground truth not available in dataset")
                return None, None
            
            # Create COCO detections object
            _, tmp = tempfile.mkstemp(suffix='.json')
            with open(tmp, 'w') as f:
                json.dump(self.coco_predictions, f)
            
            coco_dt = coco_gt.loadRes(tmp)
            
            # Run Custom_COCOeval
            coco_eval = Custom_COCOeval(coco_gt, coco_dt, 'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            
            # Extract distance metrics
            if hasattr(coco_eval, 'eval') and 'dist_error' in coco_eval.eval:
                dist_error = coco_eval.eval['dist_error']
                dist_invalid = coco_eval.eval['dist_invalid_ratio']
                
                valid_errors = dist_error[dist_error > -1]
                valid_invalid = dist_invalid[dist_invalid > -1]
                
                ade = np.mean(valid_errors) if len(valid_errors) > 0 else None
                dir_ratio = np.mean(valid_invalid) if len(valid_invalid) > 0 else None
                
                return ade, dir_ratio
            
            return None, None
            
        except Exception as e:
            LOGGER.warning(f"COCO distance evaluation failed: {e}")
            return None, None
    
    def finalize_metrics(self, *args, **kwargs):
        """Finalize all metrics including distance metrics."""
        # Standard metrics
        metrics = super().finalize_metrics(*args, **kwargs)
        
        # Run COCO-style distance evaluation
        if self.process_distance and COCO_EVAL_AVAILABLE:
            self._coco_ade, self._coco_dir = self._run_coco_distance_evaluation()
        
        # Add distance metrics
        if self.process_distance and len(self.distance_abs_errors) > 0:
            mae = np.mean(self.distance_abs_errors)
            rmse = np.sqrt(np.mean(np.square(self.distance_errors)))
            mre = np.mean(self.distance_rel_errors)
            
            LOGGER.info(f"\nDistance Estimation Metrics:")
            
            # COCO-style ADE (standardized, comparable with YOLOX)
            if self._coco_ade is not None:
                LOGGER.info(f"  ADE (COCO-style):          {self._coco_ade:.3f} m  <- Use this for comparison with YOLOX")
                if self._coco_dir is not None:
                    LOGGER.info(f"  Distance Invalid Ratio:    {self._coco_dir*100:.1f}%")
            
            # Simple IoU-based metrics (for reference)
            LOGGER.info(f"  MAE (Simple IoU@0.5):      {mae:.3f} m")
            LOGGER.info(f"  RMSE:                      {rmse:.3f} m")
            LOGGER.info(f"  MRE:                       {mre:.3f}")
            
            # Add to metrics dict
            if hasattr(metrics, 'results_dict'):
                metrics.results_dict['distance_mae'] = mae
                metrics.results_dict['distance_rmse'] = rmse
                metrics.results_dict['distance_mre'] = mre
                if self._coco_ade is not None:
                    metrics.results_dict['distance_ade_coco'] = self._coco_ade
        
        return metrics
    
    def get_stats(self):
        """Return statistics including distance metrics."""
        stats = super().get_stats()
        
        # Add distance metrics as individual float values (not nested dict)
        # to avoid TypeError when validator converts results to floats
        if self.process_distance and len(self.distance_abs_errors) > 0:
            stats['distance_mae'] = float(np.mean(self.distance_abs_errors))
            stats['distance_rmse'] = float(np.sqrt(np.mean(np.square(self.distance_errors))))
            stats['distance_mre'] = float(np.mean(self.distance_rel_errors))
            stats['distance_samples'] = float(len(self.distance_abs_errors))
            
            # Add COCO-style ADE if available
            if self._coco_ade is not None:
                stats['distance_ade_coco'] = float(self._coco_ade)
        
        return stats
    
    def print_results(self):
        """Print results including distance metrics."""
        super().print_results()
        
        if self.process_distance:
            print("\n" + "="*70)
            print("Distance Estimation Results")
            print("="*70)
            
            # COCO-style ADE (standardized metric for cross-model comparison)
            if self._coco_ade is not None:
                print("\n[COCO-style Evaluation - Use for YOLOX comparison]")
                print(f"  ADE (Average Distance Error):  {self._coco_ade:.4f} m")
                if self._coco_dir is not None:
                    print(f"  Distance Invalid Ratio:        {self._coco_dir*100:.1f}%")
            
            # Simple IoU-based metrics
            print("\n[Simple IoU@0.5 Matching]")
            if len(self.distance_abs_errors) > 0:
                mae = np.mean(self.distance_abs_errors)
                rmse = np.sqrt(np.mean(np.square(self.distance_errors)))
                mre = np.mean(self.distance_rel_errors)
                
                print(f"  Valid samples:   {len(self.distance_abs_errors)}")
                print(f"  MAE:             {mae:.4f} m")
                print(f"  RMSE:            {rmse:.4f} m")
                print(f"  MRE:             {mre:.4f}")
                print(f"  Min error:       {np.min(self.distance_abs_errors):.4f} m")
                print(f"  Max error:       {np.max(self.distance_abs_errors):.4f} m")
                
                # Accuracy at thresholds
                abs_errors = np.array(self.distance_abs_errors)
                acc_05 = np.mean(abs_errors <= 0.5) * 100
                acc_10 = np.mean(abs_errors <= 1.0) * 100
                acc_20 = np.mean(abs_errors <= 2.0) * 100
                print(f"  Acc @ 0.5m:      {acc_05:.1f}%")
                print(f"  Acc @ 1.0m:      {acc_10:.1f}%")
                print(f"  Acc @ 2.0m:      {acc_20:.1f}%")
            else:
                print("  ⚠️  No valid distance samples found (all distances are -1.0)")
                print("      This is expected with sparse radar data (radar_dilation=0)")
            
            print("="*70)


def validate_yolov8_with_distance(
    model_path,
    data_config,
    batch_size=16,
    imgsz=640,
    conf=0.001,
    iou=0.6,
    device='',
    workers=8,
    split='val',
    # Custom distance parameters
    include_radar=False,
    process_distance=False,
    distance_from_head=True,
    min_dist=0.0,
    max_dist=10.0,
    **kwargs
):
    """
    Validate YOLOv8 model with distance estimation.
    
    Args:
        model_path: Path to trained model
        data_config: Path to data config YAML
        batch_size: Batch size
        imgsz: Image size
        conf: Confidence threshold
        iou: IoU threshold for NMS
        device: Device to use
        workers: Number of workers
        split: Dataset split ('val' or 'test')
        include_radar: Use radar data fusion
        process_distance: Enable distance evaluation
        distance_from_head: Predict distance from head vs radar
        min_dist: Minimum distance
        max_dist: Maximum distance
        **kwargs: Additional arguments
        
    Returns:
        Validation results
    """
    from nerve.training.yolov8_distance_trainer import DistanceDetectionTrainer
    
    # Prepare overrides
    overrides = {
        'model': model_path,
        'data': data_config,
        'batch': batch_size,
        'imgsz': imgsz,
        'conf': conf,
        'iou': iou,
        'device': device,
        'workers': workers,
        'split': split,
        # Custom parameters
        'include_radar': include_radar,
        'process_distance': process_distance,
        'distance_from_head': distance_from_head,
        'min_dist': min_dist,
        'max_dist': max_dist,
    }
    
    # Add any additional kwargs
    overrides.update(kwargs)
    
    # Create trainer and validate
    trainer = DistanceDetectionTrainer(overrides=overrides)
    results = trainer.val()
    
    return results


if __name__ == '__main__':
    # Example usage
    results = validate_yolov8_with_distance(
        model_path='runs/detect/train/weights/best.pt',
        data_config='data.yaml',
        batch_size=16,
        imgsz=416,
        split='test',
        include_radar=True,
        process_distance=True,
        distance_from_head=True,
        min_dist=0.0,
        max_dist=10.0
    )
    
    print("Validation complete!")

