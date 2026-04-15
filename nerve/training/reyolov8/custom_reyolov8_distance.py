"""
Custom REYOLOv8 Model with Distance Estimation Head.
Adapted from custom_yolov8_distance.py for REYOLOv8's recurrent architecture.

This module extends REYOLOv8 to support distance estimation alongside object detection.
The distance head operates on features processed through Conv_LSTM modules, providing
temporal context for distance prediction on event camera data.
"""

import sys
import math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add REYOLOv8 to path
REYOLOV8_DIR = Path(__file__).parent

try:
    from ultralytics.nn.modules import Detect, Conv, C2f, DFL
    from ultralytics.yolo.utils.tal import dist2bbox, make_anchors
except ImportError:
    raise ImportError("REYOLOv8 ultralytics modules not found. Check REYOLOv8 installation.")


class ReYOLOv8_DistanceDetectionHead(Detect):
    """
    Extended REYOLOv8 detection head with distance estimation.
    Similar to YOLOv8's DistanceDetectionHead but compatible with recurrent features.
    
    The distance head adds an additional prediction branch (cv4) that predicts
    distance using classification over nbins distance bins.
    """
    
    def __init__(self, nc=80, nbins=100, min_dist=0.0, max_dist=10.0, ch=()):
        """
        Initialize distance-aware detection head.
        
        Args:
            nc: Number of classes
            nbins: Number of distance bins for classification-style distance prediction
            min_dist: Minimum distance (meters)
            max_dist: Maximum distance (meters)
            ch: Channels from backbone (list of channel counts for each detection layer)
        """
        super().__init__(nc, ch)
        
        # Distance parameters
        self.nbins = nbins
        self.min_distance = min_dist
        self.max_distance = max_dist
        self.dist_multiplier = (max_dist - min_dist) / nbins
        self.distance_loss_multiplier = 1.0
        
        # Add distance prediction layers (one per detection layer)
        self.cv4 = nn.ModuleList()  # Distance prediction convs
        
        c2 = max(ch[0] // 4, nbins)  # channels for distance head
        
        for i, x in enumerate(ch):
            # Distance branch: conv layers + prediction layer
            self.cv4.append(
                nn.Sequential(
                    Conv(x, c2, 3),
                    Conv(c2, c2, 3),
                    nn.Conv2d(c2, nbins, 1)  # Output: nbins for distance classification
                )
            )
        
        print(f"ReYOLOv8_DistanceDetectionHead initialized: {nbins} bins, range [{min_dist}, {max_dist}]m")
    
    def forward(self, x):
        """
        Forward pass with distance prediction.
        
        Args:
            x: List of feature maps from backbone (after Conv_LSTM processing)
            
        Returns:
            If training: (detection_outputs, dist_outputs)
            If inference: predictions with distance appended
        """
        shape = x[0].shape  # BCHW
        
        # Get distance predictions before concatenation
        dist_outputs = []
        for i in range(self.nl):
            dist_out = self.cv4[i](x[i])
            dist_outputs.append(dist_out)
        
        # Standard REYOLOv8 detection outputs
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        
        if self.training:
            # During training, return both detection and distance outputs
            return x, dist_outputs
        
        # Inference mode
        return self.inference(x, dist_outputs)
    
    def inference(self, x, dist_preds):
        """
        Inference with distance prediction.
        Combines standard detection with distance estimates.
        
        Args:
            x: Detection feature maps (after cv2/cv3 processing)
            dist_preds: Distance prediction logits from cv4
            
        Returns:
            Predictions with distance channel appended
        """
        shape = x[0].shape  # BCHW
        
        # Convert distance logits to distance values
        distances = []
        for dist_out in dist_preds:
            # Apply sigmoid and get argmax for most likely bin
            dist_sigmoid = dist_out.sigmoid()
            dist_bins = torch.argmax(dist_sigmoid, dim=1, keepdim=True)
            dist_values = dist_bins.float() * self.dist_multiplier + self.min_distance
            distances.append(dist_values)
        
        # Standard REYOLOv8 inference (similar to Detect.forward)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (t.transpose(0, 1) for t in make_anchors(x, self.stride, 0.5))
            self.shape = shape
        
        # Concatenate predictions from all detection layers
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        
        # Split into boxes and classes
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        
        # Decode boxes using DFL
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        
        # Concatenate distances from all layers
        dist_cat = torch.cat([d.view(shape[0], 1, -1) for d in distances], 2)
        
        # Combine: [boxes, scores, classes, distance]
        y = torch.cat((dbox, cls.sigmoid(), dist_cat), 1)
        
        return y if self.export else (y, x)
    
    def bias_init(self):
        """Initialize biases for detection and distance heads."""
        super().bias_init()
        
        # Initialize distance prediction biases
        for conv in self.cv4:
            if hasattr(conv[-1], 'bias') and conv[-1].bias is not None:
                b = conv[-1].bias.view(1, -1)
                b.data.fill_(-math.log((1 - 0.01) / 0.01))  # Initialize with low probability
                conv[-1].bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
    
    def __getstate__(self):
        """
        Custom pickling to exclude non-serializable intermediate tensors.
        This is needed for deepcopy during EMA model creation and checkpoint saving.
        """
        state = self.__dict__.copy()
        # Remove any cached tensors that might cause deepcopy issues
        keys_to_remove = [k for k in state.keys() if k.startswith('_') and isinstance(state.get(k), torch.Tensor)]
        for key in keys_to_remove:
            state.pop(key, None)
        return state
    
    def __setstate__(self, state):
        """Restore from pickled state."""
        self.__dict__.update(state)
    
    def compute_distance_loss(self, dist_preds, target_distances, batch_idx, valid_mask):
        """
        Compute distance loss for matched targets.
        
        Args:
            dist_preds: List of distance predictions [B, nbins, H, W]
            target_distances: Pre-matched distance targets (1D tensor of valid distances)
            batch_idx: Not used (kept for API compatibility)
            valid_mask: Boolean mask indicating which targets are valid
            
        Returns:
            Distance loss (scalar)
        """
        # Flatten distance predictions
        dist_pred_flat = []
        for dist_pred in dist_preds:
            b, nbins, h, w = dist_pred.shape
            dist_pred = dist_pred.view(b, nbins, -1).permute(0, 2, 1)
            dist_pred_flat.append(dist_pred)
        
        dist_pred_flat = torch.cat(dist_pred_flat, 1)  # [B, total_anchors, nbins]
        dist_pred_flat = dist_pred_flat.view(-1, self.nbins)  # [B*total_anchors, nbins]
        
        # Select predictions corresponding to valid targets
        # valid_mask tells us which entries to use
        dist_pred_valid = dist_pred_flat[valid_mask]
        
        # Clamp and convert to bins
        dist_targets_clamped = torch.clamp(target_distances, self.min_distance, self.max_distance)
        dist_bins = ((dist_targets_clamped - self.min_distance) / self.dist_multiplier).long()
        dist_bins = torch.clamp(dist_bins, 0, self.nbins - 1)
        
        # One-hot encode
        dist_target_onehot = F.one_hot(dist_bins, num_classes=self.nbins).float()
        
        # BCE loss
        loss = F.binary_cross_entropy_with_logits(
            dist_pred_valid,
            dist_target_onehot,
            reduction='sum'
        ) / max(valid_mask.sum(), 1)
        
        return loss * self.distance_loss_multiplier


class ReYOLOv8_WithDistance:
    """
    REYOLOv8 model wrapper with distance estimation capability.
    
    This class provides utilities to modify REYOLOv8 models to include
    distance estimation. It's designed to work with the recurrent architecture
    that includes Conv_LSTM modules.
    """
    
    @staticmethod
    def create_model(
        model_cfg='ReYOLOv8n.yaml',
        nc=1,
        nbins=100,
        min_dist=0.0,
        max_dist=10.0,
        pretrained=None,
        channels=5
    ):
        """
        Create a REYOLOv8 model with distance estimation.
        
        Args:
            model_cfg: Model configuration file (e.g., 'ReYOLOv8n.yaml')
            nc: Number of classes
            nbins: Number of distance bins
            min_dist: Minimum distance (meters)
            max_dist: Maximum distance (meters)
            pretrained: Path to pretrained weights (optional)
            channels: Number of input channels (default: 5 for VTEI encoding)
            
        Returns:
            Model with distance-aware detection head
        """
        from ultralytics.nn.tasks import DetectionModel2
        
        # Load base model configuration
        if isinstance(model_cfg, str):
            # Check if it's a path to YAML file
            model_path = Path(model_cfg)
            if not model_path.is_absolute():
                # Try to find in REYOLOv8 model directory
                model_path = REYOLOV8_DIR / 'ultralytics' / 'models' / 'v8' / 'Recurrent' / model_cfg
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model config not found: {model_path}")
            
            model_cfg = str(model_path)
        
        # Create base model
        # Note: REYOLOv8 uses DetectionModel2 which has proper recurrent/video forward logic
        # We'll need to build the model first, then replace the head
        model = DetectionModel2(model_cfg, imgsz=640, ch=channels, nc=nc)
        
        # Replace detection head with distance-aware head
        for i, m in enumerate(model.model):
            if isinstance(m, Detect):
                # Get channels from the head
                if hasattr(m, 'ch') and len(m.ch) > 0:
                    ch = m.ch
                else:
                    # Fallback: extract from cv2 or cv3 modules
                    # Typical values for YOLOv8n are [64, 128, 256]
                    ch = (64, 128, 256)
                    print(f"Warning: Could not get channels from Detect head, using default: {ch}")
                
                # Create new distance head
                new_head = ReYOLOv8_DistanceDetectionHead(
                    nc=nc,
                    nbins=nbins,
                    min_dist=min_dist,
                    max_dist=max_dist,
                    ch=ch
                )
                
                # Copy essential attributes from old head
                # These are required by the model's forward pass logic
                if hasattr(m, 'stride'):
                    new_head.stride = m.stride
                if hasattr(m, 'i'):
                    new_head.i = m.i  # Module index
                if hasattr(m, 'f'):
                    new_head.f = m.f  # 'from' index (which layers feed into this)
                if hasattr(m, 'type'):
                    new_head.type = m.type  # Module type string
                
                # Replace
                model.model[i] = new_head
                
                print(f"Replaced Detect head at index {i} with ReYOLOv8_DistanceDetectionHead")
                break
        
        # Load pretrained weights if provided
        if pretrained:
            print(f"Loading pretrained weights from: {pretrained}")
            checkpoint = torch.load(pretrained, map_location='cpu', weights_only=False)
            
            # Extract state dict
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Load weights (strict=False to allow new distance head parameters)
            model.load_state_dict(state_dict, strict=False)
            print("Pretrained weights loaded (distance head initialized randomly)")
        
        # Initialize head biases
        for m in model.model:
            if isinstance(m, ReYOLOv8_DistanceDetectionHead):
                m.bias_init()
        
        return model
    
    @staticmethod
    def replace_head_in_model(model, nbins=100, min_dist=0.0, max_dist=10.0):
        """
        Replace the detection head in an existing model with distance-aware head.
        
        Args:
            model: Existing REYOLOv8 model
            nbins: Number of distance bins
            min_dist: Minimum distance
            max_dist: Maximum distance
            
        Returns:
            Modified model with distance head
        """
        for i, m in enumerate(model.model):
            if isinstance(m, Detect) and not isinstance(m, ReYOLOv8_DistanceDetectionHead):
                # Get necessary information from old head
                nc = m.nc
                ch = m.ch if hasattr(m, 'ch') else []
                
                # Create new distance head
                new_head = ReYOLOv8_DistanceDetectionHead(
                    nc=nc,
                    nbins=nbins,
                    min_dist=min_dist,
                    max_dist=max_dist,
                    ch=ch
                )
                
                # Copy essential attributes from old head
                if hasattr(m, 'stride'):
                    new_head.stride = m.stride
                if hasattr(m, 'i'):
                    new_head.i = m.i
                if hasattr(m, 'f'):
                    new_head.f = m.f
                if hasattr(m, 'type'):
                    new_head.type = m.type
                
                # Replace
                model.model[i] = new_head
                
                # Initialize biases
                new_head.bias_init()
                
                print(f"Replaced head at index {i} with distance-aware head")
                break
        
        return model


# Convenience function for creating distance-enabled REYOLOv8 models
def create_reyolov8_distance_model(
    model_size='n',
    nc=1,
    nbins=100,
    min_dist=0.0,
    max_dist=10.0,
    pretrained=None,
    channels=5
):
    """
    Factory function to create REYOLOv8 model with distance estimation.
    
    Args:
        model_size: Model size ('n', 's', 'm', 'l', 'x')
        nc: Number of classes
        nbins: Number of distance bins
        min_dist: Minimum distance (meters)
        max_dist: Maximum distance (meters)
        pretrained: Path to pretrained weights
        channels: Number of input channels
        
    Returns:
        REYOLOv8 model with distance estimation capability
    """
    model_cfg = f'ReYOLOv8{model_size}.yaml'
    
    return ReYOLOv8_WithDistance.create_model(
        model_cfg=model_cfg,
        nc=nc,
        nbins=nbins,
        min_dist=min_dist,
        max_dist=max_dist,
        pretrained=pretrained,
        channels=channels
    )


if __name__ == '__main__':
    # Test distance head creation
    print("Testing ReYOLOv8 Distance Head Creation...")
    
    # Create a simple test
    ch = [256, 512, 1024]  # Example channel sizes
    head = ReYOLOv8_DistanceDetectionHead(nc=1, nbins=100, min_dist=0.0, max_dist=10.0, ch=ch)
    
    print(f"\nHead created successfully!")
    print(f"Number of detection layers: {head.nl}")
    print(f"Number of distance branches: {len(head.cv4)}")
    print(f"Distance bins: {head.nbins}")
    print(f"Distance range: [{head.min_distance}, {head.max_distance}]m")
    
    # Test with dummy input
    batch_size = 2
    dummy_inputs = [
        torch.randn(batch_size, ch[0], 52, 52),
        torch.randn(batch_size, ch[1], 26, 26),
        torch.randn(batch_size, ch[2], 13, 13)
    ]
    
    print("\nTesting forward pass (training mode)...")
    head.training = True
    det_out, dist_out = head(dummy_inputs)
    print(f"Detection outputs: {len(det_out)} layers")
    print(f"Distance outputs: {len(dist_out)} layers")
    print(f"Distance output shapes: {[d.shape for d in dist_out]}")
    
    print("\nREYOLOv8 Distance Head test passed!")

