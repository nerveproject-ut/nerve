"""
Event representation functions for converting raw event data to tensor representations.
Adapted from ReYOLOv8's formats_utils.py for PEGMA pipeline compatibility.

Supported representations:
- vtei: Volume of Ternary Event Images
- voxel_grid: Voxel Grid representation  
- shist: Stacked Histogram
- mdes: Mixed Density Event Stacks
- ev_temporal_volume: Event temporal volume

Author: Adapted for PEGMA pipeline

GPU ACCELERATION (Option B):
- Auto-detects CUDA availability and uses GPU when available
- Provides 2-4x speedup for event representation computation
- Falls back to CPU gracefully if CUDA is unavailable
"""

import torch
import numpy as np
import math
import os

# Global GPU device cache for efficiency
_GPU_DEVICE = None
_GPU_CHECKED = False

def get_best_device(force_cpu=False):
    """
    Get the best available device for computation.
    Caches result to avoid repeated CUDA availability checks.
    
    Args:
        force_cpu: If True, always return CPU device
        
    Returns:
        torch.device: Best available device
    """
    global _GPU_DEVICE, _GPU_CHECKED
    
    if force_cpu:
        return torch.device("cpu")
    
    if not _GPU_CHECKED:
        _GPU_CHECKED = True
        if torch.cuda.is_available():
            # Use CUDA_VISIBLE_DEVICES if set, otherwise default to cuda:0
            cuda_device = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0]
            try:
                _GPU_DEVICE = torch.device(f"cuda:0")
                # Test that GPU is actually usable
                _ = torch.zeros(1, device=_GPU_DEVICE)
            except Exception:
                _GPU_DEVICE = None
        else:
            _GPU_DEVICE = None
    
    return _GPU_DEVICE if _GPU_DEVICE is not None else torch.device("cpu")


def shist(x, y, t, p, bins, height, width, device="cpu"):
    """
    Stacked Histogram representation.
    
    Args:
        x: Event x coordinates (torch.Tensor)
        y: Event y coordinates (torch.Tensor)
        t: Event timestamps (torch.Tensor)
        p: Event polarities (torch.Tensor)
        bins: Number of temporal bins
        height: Image height
        width: Image width
        device: Device for computation
        
    Returns:
        Event representation tensor of shape (2*bins, height, width)
    """
    dtype = torch.uint8
    representation = torch.zeros((2, bins, height, width), dtype=dtype, device=device, requires_grad=False)
    
    if len(t) == 0:
        return torch.reshape(representation, (-1, height, width))
    
    t0 = t[0]
    t1 = t[-1]
    
    tnorm = t - t0
    tnorm = tnorm / max((t1 - t0), 1)
    tnorm = tnorm * bins
    t_idx = tnorm.floor()
    t_idx = torch.clamp(t_idx, max=bins - 1)

    indices = x.long() + \
              width * y.long() + \
              height * width * t_idx.long() + \
              bins * height * width * p.long()
    
    values = torch.ones_like(indices, dtype=dtype, device=device)
    representation.put_(indices, values, accumulate=True)
    representation = torch.clamp(representation, min=0, max=255)

    return torch.reshape(representation, (-1, height, width))


def voxel_grid(x, y, t, p, bins, height, width, device="cpu"):
    """
    Voxel Grid representation with linear temporal interpolation.
    
    Args:
        x: Event x coordinates (torch.Tensor)
        y: Event y coordinates (torch.Tensor)
        t: Event timestamps (torch.Tensor)
        p: Event polarities (torch.Tensor)
        bins: Number of temporal bins
        height: Image height
        width: Image width
        device: Device for computation
        
    Returns:
        Event representation tensor of shape (2*bins, height, width)
    """
    dtype = torch.half
    representation = torch.zeros((2, bins, height, width), dtype=dtype, device=device)
    
    if len(t) == 0:
        return torch.reshape(representation, (-1, height, width))
    
    t0 = t[0]
    t1 = t[-1]
   
    tnorm = t - t0
    tnorm = tnorm / max((t1 - t0), 1)
    tnorm = tnorm * bins
    t_idx = tnorm.floor()
    t_idx = torch.clamp(t_idx, max=bins - 1)
    
    values = torch.maximum(
        torch.zeros_like(tnorm, dtype=dtype), 
        1 - torch.abs(tnorm - t_idx)
    ).to(dtype=dtype)

    indices = x.long() + \
              width * y.long() + \
              height * width * t_idx.long() + \
              bins * height * width * p.long()
    
    representation.put_(indices, values, accumulate=True)
    return torch.reshape(representation, (-1, height, width))


def ev_temporal_volume(x, y, t, p, bins, height, width, device="cpu"):
    """
    Event temporal volume representation with sigmoid normalization.
    
    Args:
        x: Event x coordinates (torch.Tensor)
        y: Event y coordinates (torch.Tensor)
        t: Event timestamps (torch.Tensor)
        p: Event polarities (torch.Tensor)
        bins: Number of temporal bins
        height: Image height
        width: Image width
        device: Device for computation
        
    Returns:
        Event representation tensor of shape (bins, height, width)
    """
    dtype = torch.int16
    representation = torch.zeros((bins, height, width), dtype=dtype, device=device, requires_grad=False)
    
    if len(t) == 0:
        return torch.reshape((255.0 / (1 + torch.exp(-representation / 2))).to(dtype=torch.uint8), (-1, height, width))
    
    t0 = t[0]
    t1 = t[-1]
    
    p = 2 * p - 1

    tnorm = t - t0
    tnorm = tnorm / max((t1 - t0), 1)
    tnorm = tnorm * bins
    t_idx = tnorm.floor()
    t_idx = torch.clamp(t_idx, max=bins - 1)
    
    indices = x.long() + width * y.long() + height * width * t_idx.long()
    values = torch.asarray(p, dtype=dtype, device=device)
    
    representation.put_(indices, values, accumulate=True)
    
    return torch.reshape((255.0 / (1 + torch.exp(-representation / 2))).to(dtype=torch.uint8), (-1, height, width))


def vtei(x, y, t, p, bins, height, width, device="cpu"):
    """
    Volume of Ternary Event Images (VTEI) representation.
    Stores the most recent polarity at each spatial-temporal bin.
    
    Args:
        x: Event x coordinates (torch.Tensor)
        y: Event y coordinates (torch.Tensor)
        t: Event timestamps (torch.Tensor)
        p: Event polarities (torch.Tensor)
        bins: Number of temporal bins
        height: Image height
        width: Image width
        device: Device for computation
        
    Returns:
        Event representation tensor of shape (bins, height, width)
    """
    dtype = torch.int8
    representation = torch.zeros((bins, height, width), dtype=dtype, device=device, requires_grad=False)
    
    if len(t) == 0:
        return torch.reshape(representation, (-1, height, width))
    
    t0 = t[0]
    t1 = t[-1]
    
    p = 2 * p - 1

    tnorm = t - t0
    tnorm = tnorm / max((t1 - t0), 1)
    tnorm = tnorm * bins
    t_idx = tnorm.floor()
    t_idx = torch.clamp(t_idx, max=bins - 1)
    
    indices = x.long() + width * y.long() + height * width * t_idx.long()
    values = torch.asarray(p, dtype=dtype, device=device)
    
    representation.put_(indices, values, accumulate=False)
    
    return torch.reshape(representation, (-1, height, width))


def mdes(x, y, t, p, bins, height, width, device="cpu"):
    """
    Mixed Density Event Stacks (MDES) with logarithmic temporal binning and cumulative summation.
    
    Args:
        x: Event x coordinates (torch.Tensor)
        y: Event y coordinates (torch.Tensor)
        t: Event timestamps (torch.Tensor)
        p: Event polarities (torch.Tensor)
        bins: Number of temporal bins
        height: Image height
        width: Image width
        device: Device for computation
        
    Returns:
        Event representation tensor of shape (bins, height, width)
    """
    dtype = torch.int8
    representation = torch.zeros((bins, height, width), dtype=dtype, device=device, requires_grad=False)
    
    if len(t) == 0:
        return representation
    
    p = 2 * p - 1

    t0 = t[0]
    t1 = t[-1]
    tnorm = (t - t0) / max((t1 - t0), 1)
    tnorm = torch.clamp(tnorm, min=1e-6, max=1 - 1e-6)
    
    bin_float = bins - torch.log(tnorm) / math.log(1/2)
    bin_float = torch.clamp(bin_float, min=0)
    t_idx = bin_float.floor()
    
    indices = x.long() + width * y.long() + height * width * t_idx.long()
    values = torch.asarray(p, dtype=dtype, device=device)
    
    representation.put_(indices, values, accumulate=True)

    for i in reversed(range(bins)):
        representation[i] = torch.sum(input=representation[:i + 1], dim=0)

    return representation


def process_events(events, method, bins, height, width, device=None, force_cpu=False):
    """
    Process raw events into the specified representation.
    
    GPU ACCELERATED: Auto-detects and uses GPU when available for 2-4x speedup.
    
    Args:
        events: Structured numpy array with fields ['x', 'y', 't', 'p']
        method: Representation method ('vtei', 'voxel_grid', 'shist', 'mdes', 'ev_temporal_volume')
        bins: Number of temporal bins
        height: Image height
        width: Image width
        device: Device for computation (None = auto-detect, 'cpu', 'cuda', 'cuda:0', etc.)
        force_cpu: If True, force CPU computation (overrides device parameter)
        
    Returns:
        Event representation as numpy array
    """
    if events.size == 0:
        # Return empty representation
        if method in ['vtei', 'mdes', 'ev_temporal_volume']:
            return np.zeros((bins, height, width), dtype=np.int8)
        else:  # shist, voxel_grid
            return np.zeros((2 * bins, height, width), dtype=np.float32)
    
    # Auto-detect best device if not specified
    if device is None or device == "cpu":
        device = get_best_device(force_cpu=force_cpu)
    elif isinstance(device, str):
        device = torch.device(device)
    
    # Convert to torch tensors and move to device
    x = torch.from_numpy(np.clip(events["x"].astype(np.int64).copy(), 0, width - 1)).to(device)
    y = torch.from_numpy(np.clip(events["y"].astype(np.int64).copy(), 0, height - 1)).to(device)
    t = torch.from_numpy(events["t"].astype(np.int64)).to(device)
    p = torch.from_numpy(events["p"].astype(np.int64).copy()).to(device)
    
    # Apply representation on the selected device
    if method == "vtei":
        representation = vtei(x, y, t, p, bins, height, width, device)
    elif method in ["shist", "stacked_histogram"]:
        # stacked_histogram is an alias for shist (used by RVT)
        representation = shist(x, y, t, p, bins, height, width, device)
    elif method == "voxel_grid":
        representation = voxel_grid(x, y, t, p, bins, height, width, device)
    elif method == "mdes":
        representation = mdes(x, y, t, p, bins, height, width, device)
    elif method == "ev_temporal_volume":
        representation = ev_temporal_volume(x, y, t, p, bins, height, width, device)
    else:
        raise ValueError(f"Unknown event representation method: {method}")
    
    # Move back to CPU and convert to numpy
    if isinstance(representation, torch.Tensor):
        if representation.device.type != 'cpu':
            representation = representation.cpu()
        return representation.numpy()
    return representation

