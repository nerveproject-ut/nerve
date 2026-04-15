"""
Custom YOLOv8 Model with Distance Estimation Head.
Similar to custom_yolo.py but adapted for YOLOv8 architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path

try:
    from ultralytics.nn.modules import Detect, Conv, C2f
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils import LOGGER
    from ultralytics.utils.tal import make_anchors
except ImportError:
    raise ImportError("YOLOv8 (ultralytics) is not installed. Install with: pip install ultralytics")


class DistanceDetectionHead(Detect):
    """
    Extended YOLOv8 detection head with distance estimation.
    Similar to YOLOX_custom_distance_head but for YOLOv8 architecture.
    """
    
    def __init__(self, nc=80, nbins=100, min_dist=0.0, max_dist=10.0, ch=()):
        """
        Args:
            nc: Number of classes
            nbins: Number of distance bins for classification-style distance prediction
            min_dist: Minimum distance (meters)
            max_dist: Maximum distance (meters)
            ch: Channels from backbone
        """
        super().__init__(nc=nc, ch=ch)
        
        self.nbins = nbins
        self.min_distance = min_dist
        self.max_distance = max_dist
        self.dist_multiplier = (max_dist - min_dist) / nbins
        self.distance_loss_multiplier = 1.0
        
        # Add distance prediction layers (one per detection layer)
        self.cv4 = nn.ModuleList()  # Distance prediction convs
        
        c2 = max(ch[0] // 4, nbins)  # channels for distance head
        c3 = max(ch[0], min(self.nc, 100))  # channels
        
        for i, x in enumerate(ch):
            # Distance branch: conv layers + prediction layer
            self.cv4.append(
                nn.Sequential(
                    Conv(x, c2, 3),
                    Conv(c2, c2, 3),
                    nn.Conv2d(c2, nbins, 1)  # Output: nbins for distance classification
                )
            )
        
        LOGGER.info(f"DistanceDetectionHead initialized: {nbins} bins, range [{min_dist}, {max_dist}]m")
    
    def forward(self, x):
        """
        Forward pass with distance prediction.
        
        Args:
            x: List of feature maps from backbone
            
        Returns:
            If training: (bbox_pred, cls_pred, dist_pred) for each scale
            If inference: predictions with distance appended
        """
        shape = x[0].shape  # BCHW
        
        # Save raw features BEFORE processing for distance head
        # Must clone to avoid modifying when we process x
        raw_features = [feat.clone() for feat in x]
        
        # Standard YOLOv8 detection outputs
        for i in range(self.nl):  # number of detection layers
            # Original YOLOv8 predictions
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        
        if self.training:
            # During training, use RAW features for distance prediction
            dist_outputs = []
            for i in range(self.nl):
                # Use raw features (192/384/576 ch) not processed output (65 ch)
                dist_out = self.cv4[i](raw_features[i])
                dist_outputs.append(dist_out)
            
            return x, dist_outputs
        
        # Inference mode - pass raw features
        return self.inference(x, raw_features)
    
    def inference(self, x, raw_features):
        """
        Inference with distance prediction.
        Similar to standard YOLOv8 but adds distance channel.
        
        Args:
            x: Processed detection outputs (bbox + class)
            raw_features: Raw backbone features for distance prediction
            
        Returns:
            (y, x) tuple to match standard Detect.forward() return format:
            - y: Decoded predictions [B, 4+nc+1, N] (boxes, classes, distance)
            - x: Raw feature outputs for potential further processing
        """
        # Get distance predictions from RAW features
        dist_preds = []
        for i in range(self.nl):
            dist_out = self.cv4[i](raw_features[i])
            dist_preds.append(dist_out)
        
        # Convert distance logits to distance values
        distances = []
        for dist_out in dist_preds:
            # Apply sigmoid and get argmax for most likely bin
            dist_sigmoid = dist_out.sigmoid()
            dist_bins = torch.argmax(dist_sigmoid, dim=1, keepdim=True)
            dist_values = dist_bins.float() * self.dist_multiplier + self.min_distance
            distances.append(dist_values)
        
        # Standard YOLOv8 inference
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape
        
        # Split predictions
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        
        # Decode boxes using parent's method (Detect.decode_bboxes)
        # Note: dist2bbox is the standard YOLOv8 decoding function
        from ultralytics.utils.tal import dist2bbox
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        
        # Concatenate distances from all scales
        dist_cat = torch.cat([d.view(shape[0], 1, -1) for d in distances], 2)
        
        # Include distance in output (like YOLOX does)
        # Output format: [boxes (4), class scores (nc), distance (1)]
        # This allows the validator to access distances from predictions
        y = torch.cat((dbox, cls.sigmoid(), dist_cat), 1)
        
        # Also store separately for backwards compatibility
        self._inference_distances = dist_cat
        
        # Return tuple format like standard Detect.forward(): (y, x)
        # This is expected by the validator and post-processing code
        return (y, x)
    
    def compute_distance_loss(self, dist_preds, targets, batch_idx, fg_mask):
        """
        Compute distance loss using BCE with logits.
        Similar to YOLOX implementation.
        
        Args:
            dist_preds: Distance predictions (logits) [B, nbins, H, W]
            targets: Ground truth distances
            batch_idx: Batch indices for matching
            fg_mask: Foreground mask
            
        Returns:
            Distance loss value
        """
        if not fg_mask.any():
            return torch.tensor(0.0, device=dist_preds[0].device)
        
        # Flatten distance predictions
        dist_pred_flat = []
        for i, dist_pred in enumerate(dist_preds):
            b, nbins, h, w = dist_pred.shape
            dist_pred = dist_pred.view(b, nbins, -1).permute(0, 2, 1)  # [B, H*W, nbins]
            dist_pred_flat.append(dist_pred)
        
        dist_pred_flat = torch.cat(dist_pred_flat, 1)  # [B, total_anchors, nbins]
        
        # Select foreground predictions
        dist_pred_fg = dist_pred_flat[fg_mask]  # [num_fg, nbins]
        
        # Convert distance targets to bin classification targets
        if 'distances' in targets:
            dist_targets = targets['distances']
            
            # CRITICAL: Filter out invalid distances (marked as -1.0 from dataset)
            # Only compute loss on boxes with valid radar data
            valid_mask = dist_targets >= 0.0
            
            if not valid_mask.any():
                # No valid distance targets in this batch
                return torch.tensor(0.0, device=dist_pred_fg.device)
            
            # Select only valid distances
            dist_pred_fg_valid = dist_pred_fg[valid_mask]
            dist_targets_valid = dist_targets[valid_mask]
            
            # Clamp distances to valid range
            dist_targets_clamped = torch.clamp(dist_targets_valid, self.min_distance, self.max_distance)
            
            # Convert to bin indices
            dist_bins = ((dist_targets_clamped - self.min_distance) / self.dist_multiplier).long()
            dist_bins = torch.clamp(dist_bins, 0, self.nbins - 1)
            
            # Create one-hot targets
            dist_target_onehot = F.one_hot(dist_bins, num_classes=self.nbins).float()
            
            # BCE loss (only on valid samples)
            loss = F.binary_cross_entropy_with_logits(
                dist_pred_fg_valid,
                dist_target_onehot,
                reduction='sum'
            ) / max(valid_mask.sum(), 1)  # Normalize by number of valid samples
            
            return loss * self.distance_loss_multiplier
        
        return torch.tensor(0.0, device=dist_pred_flat.device)
    
    def __getstate__(self):
        """
        Custom serialization to exclude intermediate tensors from deepcopy/pickle.
        This prevents 'Only Tensors created explicitly by the user' error during model saving.
        """
        state = self.__dict__.copy()
        # Remove intermediate tensor attributes that shouldn't be serialized
        state.pop('_dist_preds', None)
        return state
    
    def __setstate__(self, state):
        """
        Custom deserialization to restore model state.
        """
        self.__dict__.update(state)
        # Initialize any attributes that were excluded from serialization
        self._dist_preds = None


class YOLOv8WithDistance(DetectionModel):
    """
    YOLOv8 model with distance estimation capability.
    Similar to Customized_YOLOX but for YOLOv8.
    """
    
    def __init__(
        self,
        cfg='yolov8n.yaml',
        ch=3,
        nc=None,
        verbose=True,
        distance_from_head=True,
        max_radar_distance=10.0,
        original_img_size=(640, 360),
        nbins=100,
        min_dist=0.0,
        max_dist=10.0
    ):
        """
        Args:
            cfg: Model configuration file or dict
            ch: Input channels
            nc: Number of classes
            verbose: Print model info
            distance_from_head: If True, predict distance from head; if False, extract from radar
            max_radar_distance: Maximum radar distance for post-processing
            original_img_size: Original image size (w, h)
            nbins: Number of distance bins
            min_dist: Minimum distance
            max_dist: Maximum distance
        """
        self.distance_from_head = distance_from_head
        self.max_radar_dist = max_radar_distance
        self.original_img_size = original_img_size
        self.nbins = nbins
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.nc = nc  # Store nc before parent init
        
        # Initialize parent (this will build the model)
        super().__init__(cfg, ch, nc, verbose)
        
        # Replace detection head with distance-aware head if needed
        if distance_from_head:
            self._replace_head_with_distance_head()
    
    def _replace_head_with_distance_head(self):
        """Replace standard Detect head with DistanceDetectionHead."""
        # Find the detection head in the model
        for i, m in enumerate(self.model):
            if isinstance(m, Detect):
                # Extract channels from the existing Detect head
                # The Detect head has cv2/cv3 ModuleLists with one element per detection scale
                # We can get the input channels from cv2's in_channels
                ch = []
                if hasattr(m, 'cv2') and len(m.cv2) > 0:
                    for cv in m.cv2:
                        # cv2 is a Sequential, get the first Conv layer's in_channels
                        if hasattr(cv[0], 'conv'):
                            ch.append(cv[0].conv.in_channels)
                        elif hasattr(cv[0], 'in_channels'):
                            ch.append(cv[0].in_channels)
                
                # Create new distance head
                new_head = DistanceDetectionHead(
                    nc=self.nc,
                    nbins=self.nbins,
                    min_dist=self.min_dist,
                    max_dist=self.max_dist,
                    ch=tuple(ch)  # Pass as tuple like original Detect
                )
                
                # CRITICAL: Copy cv2, cv3 weights from original pretrained head
                # Without this, the detection head has random weights and won't work!
                new_head.cv2.load_state_dict(m.cv2.state_dict())
                new_head.cv3.load_state_dict(m.cv3.state_dict())
                LOGGER.info("Copied cv2/cv3 weights from pretrained Detect head")
                
                # Copy DFL weights if available
                if hasattr(m, 'dfl') and hasattr(new_head, 'dfl'):
                    new_head.dfl.load_state_dict(m.dfl.state_dict())
                
                # Copy other Detect attributes
                new_head.stride = m.stride
                new_head.reg_max = m.reg_max
                
                # Copy metadata attributes from original head
                # These are used by YOLOv8's model structure
                if hasattr(m, 'f'):
                    new_head.f = m.f  # Input layer indices
                if hasattr(m, 'i'):
                    new_head.i = m.i  # Module index
                if hasattr(m, 'type'):
                    new_head.type = m.type  # Module type string
                if hasattr(m, 'np'):
                    new_head.np = m.np  # Number of parameters
                
                # Replace with distance head
                self.model[i] = new_head
                LOGGER.info(f"Replaced Detect head with DistanceDetectionHead (ch={ch})")
                break
    
    def forward(self, x, *args, **kwargs):
        """
        Forward pass with optional distance extraction from radar.
        
        Args:
            x: Input tensor [B, C, H, W] or dict (during training)
            *args: Additional positional arguments (e.g., targets)
            **kwargs: Additional keyword arguments (e.g., augment, profile, visualize)
            
        Returns:
            Model predictions with distance
        """
        if self.training:
            return super().forward(x, *args, **kwargs)
        
        # Inference mode
        outputs = super().forward(x, *args, **kwargs)
        
        # If not using distance from head, extract from radar map
        if not self.distance_from_head:
            outputs = self._extract_distance_from_radar(outputs, x)
        
        return outputs
    
    def init_criterion(self):
        """
        Initialize the loss criterion.
        Uses v8DistanceDetectionLoss for proper distance loss integration.
        """
        if self.distance_from_head:
            try:
                from nerve.training.yolov8_distance_loss import v8DistanceDetectionLoss
                return v8DistanceDetectionLoss(
                    self,
                    nbins=self.nbins,
                    min_dist=self.min_dist,
                    max_dist=self.max_dist,
                    distance_loss_multiplier=getattr(self, 'distance_loss_multiplier', 1.0)
                )
            except Exception as e:
                LOGGER.warning(f"Could not import v8DistanceDetectionLoss: {e}, using standard loss")
                return super().init_criterion()
        return super().init_criterion()
    
    def loss(self, batch, preds=None):
        """
        Compute loss with distance prediction handling.
        
        Uses v8DistanceDetectionLoss which properly integrates distance loss
        with the Task-Aligned Assigner (TAL) for correct foreground matching.
        
        Args:
            batch: Batch dictionary with 'img', 'bboxes', 'cls', 'distances'
            preds: Optional pre-computed predictions
            
        Returns:
            Tuple (total_loss * batch_size, loss_items) where loss_items = [box, cls, dfl, dist]
        """
        if not hasattr(self, 'criterion'):
            self.criterion = self.init_criterion()
        
        # Determine if we need to re-forward to get training-mode predictions
        need_reforward = False
        
        if preds is None:
            need_reforward = True
        elif isinstance(preds, tuple) and len(preds) == 2:
            first_elem = preds[0]
            if isinstance(first_elem, list):
                need_reforward = False
            elif isinstance(first_elem, torch.Tensor):
                if first_elem.dim() == 4:
                    need_reforward = False
                else:
                    need_reforward = True
            else:
                need_reforward = True
        elif not isinstance(preds, (list, tuple)):
            need_reforward = True
        
        if need_reforward:
            was_training = self.training
            self.train()
            preds = self.forward(batch['img'])
            if not was_training:
                self.eval()
        
        criterion_name = type(self.criterion).__name__
        
        # If preds is tuple and criterion is standard loss, pass only det_preds
        if isinstance(preds, tuple) and len(preds) == 2 and criterion_name != 'v8DistanceDetectionLoss':
            det_preds, dist_preds = preds
            return self.criterion(det_preds, batch)
        
        # v8DistanceDetectionLoss handles the tuple (det_preds, dist_preds) directly
        return self.criterion(preds, batch)
    
    def _extract_distance_from_radar(self, outputs, x):
        """
        Extract distance from radar point cloud in image.
        Similar to Customized_YOLOX implementation.
        
        Args:
            outputs: Model predictions [B, num_detections, 6] (x, y, w, h, conf, cls)
            x: Input images [B, C, H, W] where channel 2 contains radar data
            
        Returns:
            Predictions with distance appended [B, num_detections, 7]
        """
        batch_size, _, h, w = x.shape
        
        # Calculate resize ratio
        ratio = max(self.original_img_size[0] / w, self.original_img_size[1] / h)
        resized_h = round(self.original_img_size[1] / ratio)
        resized_w = round(self.original_img_size[0] / ratio)
        
        # Radar maps are in channel 2 (R channel in BGR)
        dist_maps_from_radar = x[:, 2].to(torch.uint8)
        
        # Convert predictions to bbox coordinates
        # Assuming outputs format: [x, y, w, h, conf, cls]
        xmin = torch.round(outputs[:, :, 0] - outputs[:, :, 2] / 2).clamp(0, resized_w)
        ymin = torch.round(outputs[:, :, 1] - outputs[:, :, 3] / 2).clamp(0, resized_h)
        xmax = torch.round(outputs[:, :, 0] + outputs[:, :, 2] / 2).clamp(0, resized_w)
        ymax = torch.round(outputs[:, :, 1] + outputs[:, :, 3] / 2).clamp(0, resized_h)
        
        num_bbs = xmin.shape[1]
        dist = torch.empty((batch_size, num_bbs, 1), device=outputs.device)
        
        # Process each batch
        for b in range(batch_size):
            # Extract average distance from radar points in each bbox
            for i in range(num_bbs):
                x1, y1, x2, y2 = int(xmin[b, i]), int(ymin[b, i]), int(xmax[b, i]), int(ymax[b, i])
                
                if x2 > x1 and y2 > y1:
                    # Crop radar map to bbox
                    radar_crop = dist_maps_from_radar[b, y1:y2, x1:x2]
                    
                    # Calculate average distance from non-zero radar pixels
                    nonzero_pixels = radar_crop[radar_crop > 0]
                    
                    if len(nonzero_pixels) > 0:
                        avg_dist = nonzero_pixels.float().mean() * (self.max_radar_dist / 255.0)
                        dist[b, i, 0] = avg_dist
                    else:
                        dist[b, i, 0] = -1.0  # No radar data
                else:
                    dist[b, i, 0] = -1.0
        
        # Append distance to outputs
        outputs = torch.cat((outputs, dist), -1)
        
        return outputs


def mask_image(img: torch.Tensor, xmin: torch.Tensor, xmax: torch.Tensor, 
               ymin: torch.Tensor, ymax: torch.Tensor):
    """
    Mask image to extract regions within bounding boxes.
    Used for extracting distance from radar point clouds.
    """
    h, w = img.shape
    d = len(xmin)
    
    images = img.unsqueeze(1).repeat(1, d, 1)
    xmin = xmin.unsqueeze(-1)
    xmax = xmax.unsqueeze(-1)
    ymin = ymin.unsqueeze(0)
    ymax = ymax.unsqueeze(0)
    
    w_range = torch.arange(w, device=img.device).unsqueeze(0)
    h_range = torch.arange(h, device=img.device).unsqueeze(-1)
    
    images[h_range < ymin, :] = 0
    images[h_range >= ymax, :] = 0
    images[:, w_range < xmin] = 0
    images[:, w_range >= xmax] = 0
    
    images = torch.moveaxis(images, 1, 0)
    
    return images


def create_yolov8_distance_model(
    model_cfg='yolov8n.yaml',
    nc=1,
    distance_from_head=True,
    nbins=100,
    min_dist=0.0,
    max_dist=10.0,
    pretrained=False
):
    """
    Factory function to create YOLOv8 model with distance estimation.
    
    Args:
        model_cfg: Model configuration (n, s, m, l, x or yaml file)
        nc: Number of classes
        distance_from_head: If True, predict distance from neural network head
        nbins: Number of distance bins
        min_dist: Minimum distance
        max_dist: Maximum distance
        pretrained: Load pretrained weights
        
    Returns:
        YOLOv8WithDistance model
    """
    model = YOLOv8WithDistance(
        cfg=model_cfg,
        nc=nc,
        distance_from_head=distance_from_head,
        nbins=nbins,
        min_dist=min_dist,
        max_dist=max_dist
    )
    
    LOGGER.info(f"Created YOLOv8 model with distance estimation: "
                f"{'from head' if distance_from_head else 'from radar map'}")
    
    return model

