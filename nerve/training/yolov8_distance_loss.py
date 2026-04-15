"""
Custom YOLOv8 Loss with Distance Estimation.

This module extends YOLOv8's detection loss to include distance prediction loss.
The key insight is that distance targets must be matched to predictions using the
same Task-Aligned Assigner (TAL) that matches detection targets.

This implementation mirrors how YOLOX handles distance loss in custom_yolo.py
but adapted for YOLOv8's architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Handle different ultralytics versions - try both import paths
try:
    # Newer ultralytics (8.0+)
    from ultralytics.utils.loss import BboxLoss
    from ultralytics.utils.ops import xywh2xyxy
    from ultralytics.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors
    from ultralytics.utils import LOGGER
except ImportError:
    # Older ultralytics structure
    from ultralytics.yolo.utils.loss import BboxLoss
    from ultralytics.yolo.utils.ops import xywh2xyxy
    from ultralytics.yolo.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors
    from ultralytics.utils import LOGGER


class DistanceBCELoss(nn.Module):
    """
    Distance loss using Binary Cross Entropy over distance bins.
    Similar to YOLOX's distance classification approach.
    """
    
    def __init__(self, nbins=100, min_dist=0.0, max_dist=10.0):
        super().__init__()
        self.nbins = nbins
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.dist_multiplier = (max_dist - min_dist) / nbins
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, pred_dist, target_dist, fg_mask, target_scores_sum):
        """
        Compute distance loss for matched foreground predictions.
        
        Args:
            pred_dist: Distance predictions [B, num_anchors, nbins]
            target_dist: Target distances [B, max_gt] in meters
            fg_mask: Foreground mask [B, num_anchors] from TAL assigner
            target_scores_sum: Sum of target scores for normalization
            
        Returns:
            Distance loss value
        """
        if not fg_mask.any():
            return torch.tensor(0.0, device=pred_dist.device, dtype=pred_dist.dtype)
        
        # Select foreground predictions
        pred_dist_fg = pred_dist[fg_mask]  # [num_fg, nbins]
        
        # Filter out invalid distances (marked as -1.0 or negative)
        valid_mask = target_dist >= 0.0
        if not valid_mask.any():
            return torch.tensor(0.0, device=pred_dist.device, dtype=pred_dist.dtype)
        
        # Get valid targets
        target_dist_valid = target_dist[valid_mask]
        pred_dist_valid = pred_dist_fg[:len(target_dist_valid)]  # Match sizes
        
        # Clamp to valid range
        target_dist_clamped = torch.clamp(target_dist_valid, self.min_dist, self.max_dist)
        
        # Convert to bin indices
        target_bins = ((target_dist_clamped - self.min_dist) / self.dist_multiplier).long()
        target_bins = torch.clamp(target_bins, 0, self.nbins - 1)
        
        # One-hot encoding
        target_onehot = F.one_hot(target_bins, num_classes=self.nbins).float()
        
        # BCE loss
        loss = self.bce(pred_dist_valid, target_onehot).sum() / max(target_scores_sum, 1)
        
        return loss


class v8DistanceDetectionLoss:
    """
    YOLOv8 Detection Loss with Distance Estimation.
    
    Extends the standard v8DetectionLoss to compute distance loss using
    the same target assignment from Task-Aligned Assigner.
    
    This is the key fix: distance loss must be computed on the SAME foreground
    mask that detection uses, not a placeholder mask.
    """
    
    def __init__(self, model, nbins=100, min_dist=0.0, max_dist=10.0, 
                 distance_loss_multiplier=1.0):
        """
        Initialize the distance-aware loss.
        
        Args:
            model: The detection model (de-paralleled)
            nbins: Number of distance bins
            min_dist: Minimum distance in meters
            max_dist: Maximum distance in meters
            distance_loss_multiplier: Weight for distance loss
        """
        device = next(model.parameters()).device
        h = model.args  # hyperparameters
        
        m = model.model[-1]  # Detect() module (or DistanceDetectionHead)
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.hyp = h
        self.stride = m.stride
        self.nc = m.nc
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device
        
        self.use_dfl = m.reg_max > 1
        
        self.assigner = TaskAlignedAssigner(
            topk=10,
            num_classes=self.nc,
            alpha=0.5,
            beta=6.0,
            stride=m.stride.tolist(),
        )
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)
        
        # Distance loss parameters
        self.nbins = nbins
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.dist_multiplier = (max_dist - min_dist) / nbins
        self.distance_loss_multiplier = distance_loss_multiplier
        
        # Check if model has distance head
        self.has_distance_head = hasattr(m, 'cv4') and m.cv4 is not None
        
        LOGGER.info(f"v8DistanceDetectionLoss initialized:")
        LOGGER.info(f"  - Has distance head: {self.has_distance_head}")
        LOGGER.info(f"  - Distance bins: {nbins}, range [{min_dist}, {max_dist}]m")
        LOGGER.info(f"  - Distance loss multiplier: {distance_loss_multiplier}")
    
    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocess detection targets."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out
    
    def preprocess_distances(self, batch, batch_size, max_gt):
        """
        Preprocess distance targets to match GT structure.
        
        Args:
            batch: Batch dictionary containing 'distances' and 'batch_idx'
            batch_size: Batch size
            max_gt: Maximum number of GT boxes per image
            
        Returns:
            Distance targets [B, max_gt] with -1.0 for invalid/padding
        """
        if 'distances' not in batch:
            return torch.full((batch_size, max_gt), -1.0, device=self.device)
        
        distances = batch['distances'].to(self.device)
        batch_idx = batch['batch_idx'].to(self.device)
        
        # Initialize with -1.0 (invalid marker)
        out = torch.full((batch_size, max_gt), -1.0, device=self.device)
        
        for j in range(batch_size):
            matches = batch_idx == j
            n = matches.sum()
            if n:
                out[j, :n] = distances[matches]
        
        return out
    
    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted bboxes from anchor points."""
        if self.use_dfl:
            b, a, c = pred_dist.shape
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(
                self.proj.type(pred_dist.dtype)
            )
        return dist2bbox(pred_dist, anchor_points, xywh=False)
    
    def compute_distance_loss(self, dist_preds, target_distances, fg_mask, 
                               matched_gt_inds, target_scores_sum):
        """
        Compute distance loss for matched foreground predictions.
        
        This is the CRITICAL function that properly matches distance predictions
        to ground truth using the TAL assignment.
        
        Args:
            dist_preds: Distance predictions [B, num_anchors, nbins]
            target_distances: GT distances [B, max_gt]
            fg_mask: Foreground mask [B, num_anchors] from TAL
            matched_gt_inds: GT indices for each anchor [B, num_anchors] from TAL
            target_scores_sum: Normalization factor
            
        Returns:
            Distance loss value
        """
        if not fg_mask.any():
            return torch.tensor(0.0, device=dist_preds.device, dtype=dist_preds.dtype)
        
        batch_size = dist_preds.shape[0]
        
        # Select foreground distance predictions using boolean mask
        # fg_mask: [B, num_anchors], dist_preds: [B, num_anchors, nbins]
        dist_pred_fg = dist_preds[fg_mask]  # [num_fg, nbins]
        
        # matched_gt_inds has shape [B, num_anchors] - it contains GT index for each anchor
        # We need to get the GT indices only for foreground anchors
        # fg_mask selects which anchors are foreground
        matched_gt_inds_fg = matched_gt_inds[fg_mask]  # [num_fg]
        
        # Build the matched distance targets by gathering from target_distances
        # We need to handle the batch dimension properly
        matched_distances = []
        fg_per_image = fg_mask.sum(dim=1)  # [B] - number of foreground per image
        
        offset = 0
        for b in range(batch_size):
            num_fg = int(fg_per_image[b])
            if num_fg > 0:
                # Get the GT indices for this image's foreground anchors
                gt_inds = matched_gt_inds_fg[offset:offset + num_fg]
                # Clamp indices to valid range (in case of padding)
                max_gt = target_distances.shape[1]
                gt_inds = torch.clamp(gt_inds, 0, max_gt - 1)
                # Get the corresponding distances
                img_distances = target_distances[b][gt_inds]
                matched_distances.append(img_distances)
                offset += num_fg
        
        if len(matched_distances) == 0:
            return torch.tensor(0.0, device=dist_preds.device, dtype=dist_preds.dtype)
        
        matched_distances = torch.cat(matched_distances, dim=0)  # [num_fg]
        
        # Filter out invalid distances (marked as -1.0 or negative)
        valid_mask = matched_distances >= 0.0
        
        if not valid_mask.any():
            return torch.tensor(0.0, device=dist_preds.device, dtype=dist_preds.dtype)
        
        # Select valid predictions and targets
        dist_pred_valid = dist_pred_fg[valid_mask]
        dist_target_valid = matched_distances[valid_mask]
        
        # Clamp to valid range
        dist_target_clamped = torch.clamp(
            dist_target_valid, self.min_dist, self.max_dist
        )
        
        # Convert to bin indices
        dist_bins = ((dist_target_clamped - self.min_dist) / self.dist_multiplier).long()
        dist_bins = torch.clamp(dist_bins, 0, self.nbins - 1)
        
        # One-hot encoding
        dist_target_onehot = F.one_hot(dist_bins, num_classes=self.nbins).float()
        
        # BCE loss (same as YOLOX)
        loss = F.binary_cross_entropy_with_logits(
            dist_pred_valid,
            dist_target_onehot,
            reduction='sum'
        ) / max(valid_mask.sum(), 1)
        
        return loss * self.distance_loss_multiplier
    
    def __call__(self, preds, batch):
        """
        Compute total loss including distance.
        
        Args:
            preds: Predictions - either:
                   - Tuple (det_feats, dist_preds) from DistanceDetectionHead
                   - List of feature maps for standard detection
            batch: Batch dictionary with 'img', 'bboxes', 'cls', 'distances'
            
        Returns:
            (total_loss * batch_size, loss_items) where loss_items = [box, cls, dfl, dist]
        """
        # Initialize loss with 4 components: box, cls, dfl, dist
        loss = torch.zeros(4, device=self.device)
        
        # Handle predictions format
        dist_preds = None
        if isinstance(preds, tuple) and len(preds) == 2:
            feats, dist_preds = preds
            # dist_preds is a list of [B, nbins, H, W] tensors from each scale
        else:
            feats = preds[1] if isinstance(preds, tuple) else preds
        
        # Standard detection loss computation
        # Concatenate feature maps and split into distribution and scores
        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)
        
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(
            feats[0].shape[2:], device=self.device, dtype=dtype
        ) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
        
        # Prepare detection targets
        targets = torch.cat(
            (batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1
        )
        targets = self.preprocess(
            targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]]
        )
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
        
        # Decode predicted bboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        
        # Task-Aligned Assignment - this gives us the crucial fg_mask!
        _, target_bboxes, target_scores, fg_mask, matched_gt_inds = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt
        )
        
        target_bboxes /= stride_tensor
        target_scores_sum = max(target_scores.sum(), 1)
        
        # Classification loss (BCE)
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum
        
        # Bbox loss (IoU + DFL)
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, 
                target_bboxes, target_scores, target_scores_sum, fg_mask,
                imgsz, stride_tensor,
            )
        
        # Distance loss - using the SAME fg_mask from TAL!
        if dist_preds is not None and self.has_distance_head:
            # Flatten distance predictions: [B, nbins, H, W] -> [B, num_anchors, nbins]
            dist_pred_flat = []
            for dp in dist_preds:
                b, nbins, h, w = dp.shape
                dp_flat = dp.view(b, nbins, -1).permute(0, 2, 1)  # [B, H*W, nbins]
                dist_pred_flat.append(dp_flat)
            dist_pred_flat = torch.cat(dist_pred_flat, dim=1)  # [B, total_anchors, nbins]
            
            # Prepare distance targets
            max_gt = targets.shape[1]
            target_distances = self.preprocess_distances(batch, batch_size, max_gt)
            
            # Compute distance loss
            loss[3] = self.compute_distance_loss(
                dist_pred_flat,
                target_distances,
                fg_mask,
                matched_gt_inds,
                target_scores_sum
            )
        
        # Apply loss gains
        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        # loss[3] already has distance_loss_multiplier applied
        
        return loss.sum() * batch_size, loss.detach()


def create_distance_loss(model, nbins=100, min_dist=0.0, max_dist=10.0, 
                         distance_loss_multiplier=1.0):
    """
    Factory function to create distance-aware loss.
    
    Args:
        model: Detection model (will be de-paralleled if needed)
        nbins: Number of distance bins
        min_dist: Minimum distance
        max_dist: Maximum distance
        distance_loss_multiplier: Weight for distance loss
        
    Returns:
        v8DistanceDetectionLoss instance
    """
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        model = model.module
    
    return v8DistanceDetectionLoss(
        model,
        nbins=nbins,
        min_dist=min_dist,
        max_dist=max_dist,
        distance_loss_multiplier=distance_loss_multiplier
    )
