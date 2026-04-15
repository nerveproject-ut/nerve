"""
Batch Visualizer.

Provides functions to visualize training and validation batches with
bounding boxes. Supports different input formats including:
- Standard RGB images
- Event representations (multi-channel)
- Grayscale images
"""

import numpy as np
import cv2
from pathlib import Path
from typing import List, Optional, Union, Tuple, Any
import warnings

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

warnings.filterwarnings('ignore')


def plot_training_batch(
    images: Any,
    targets: Any,
    save_path: Union[str, Path],
    class_names: Optional[List[str]] = None,
    max_samples: int = 16,
    normalized: bool = True,
    image_size: Optional[Tuple[int, int]] = None,
) -> None:
    """
    Plot a training batch with ground truth boxes.
    
    Args:
        images: Batch of images, shape (N, C, H, W) or (N, H, W, C)
        targets: Ground truth labels, list of arrays or tensor
                Each target: [class_id, cx, cy, w, h] (normalized) or [class_id, x1, y1, x2, y2]
        save_path: Path to save the visualization
        class_names: List of class names
        max_samples: Maximum number of samples to visualize
        normalized: Whether target coordinates are normalized (0-1)
        image_size: Original image size (H, W) for denormalization
    """
    visualizer = BatchVisualizer(class_names=class_names)
    visualizer.plot_batch(
        images=images,
        targets=targets,
        predictions=None,
        save_path=save_path,
        max_samples=max_samples,
        normalized=normalized,
        image_size=image_size,
        title='Training Batch - Ground Truth',
    )


def plot_validation_batch(
    images: Any,
    targets: Any,
    predictions: Any,
    save_path: Union[str, Path],
    class_names: Optional[List[str]] = None,
    max_samples: int = 16,
    normalized: bool = True,
    image_size: Optional[Tuple[int, int]] = None,
) -> None:
    """
    Plot a validation batch with both ground truth and predictions.
    
    Args:
        images: Batch of images, shape (N, C, H, W) or (N, H, W, C)
        targets: Ground truth labels
        predictions: Model predictions, each: [x1, y1, x2, y2, conf, class_id]
        save_path: Path to save the visualization
        class_names: List of class names
        max_samples: Maximum number of samples to visualize
        normalized: Whether target coordinates are normalized (0-1)
        image_size: Original image size (H, W) for denormalization
    """
    visualizer = BatchVisualizer(class_names=class_names)
    
    save_path = Path(save_path)
    
    # Save ground truth visualization
    visualizer.plot_batch(
        images=images,
        targets=targets,
        predictions=None,
        save_path=save_path.parent / f'{save_path.stem}_labels{save_path.suffix}',
        max_samples=max_samples,
        normalized=normalized,
        image_size=image_size,
        title='Validation Batch - Ground Truth',
    )
    
    # Save predictions visualization
    visualizer.plot_batch(
        images=images,
        targets=None,
        predictions=predictions,
        save_path=save_path.parent / f'{save_path.stem}_pred{save_path.suffix}',
        max_samples=max_samples,
        normalized=normalized,
        image_size=image_size,
        title='Validation Batch - Predictions',
    )


class BatchVisualizer:
    """
    Batch visualizer for object detection training.
    
    Handles various input formats and creates grid visualizations
    of images with bounding boxes.
    """
    
    # Color palette for different classes (BGR format for OpenCV)
    COLORS = [
        (255, 0, 0),      # Blue
        (0, 255, 0),      # Green
        (0, 0, 255),      # Red
        (255, 255, 0),    # Cyan
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Yellow
        (128, 0, 255),    # Purple
        (255, 128, 0),    # Orange
        (0, 128, 255),    # Light blue
        (128, 255, 0),    # Light green
    ]
    
    def __init__(
        self,
        class_names: Optional[List[str]] = None,
        line_thickness: int = 2,
        font_scale: float = 0.5,
    ):
        """
        Initialize the visualizer.
        
        Args:
            class_names: List of class names
            line_thickness: Thickness of bounding box lines
            font_scale: Font scale for labels
        """
        self.class_names = class_names
        self.line_thickness = line_thickness
        self.font_scale = font_scale
    
    def plot_batch(
        self,
        images: Any,
        targets: Optional[Any] = None,
        predictions: Optional[Any] = None,
        save_path: Union[str, Path] = 'batch.jpg',
        max_samples: int = 16,
        normalized: bool = True,
        image_size: Optional[Tuple[int, int]] = None,
        title: str = 'Batch',
    ) -> None:
        """
        Plot a batch of images with boxes.
        
        Args:
            images: Batch of images
            targets: Ground truth boxes (green)
            predictions: Predicted boxes (red)
            save_path: Path to save the visualization
            max_samples: Maximum samples to show
            normalized: Whether coordinates are normalized
            image_size: Image size for denormalization
            title: Title for the plot
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert images to numpy
        images_np = self._to_numpy(images)
        
        # Limit samples
        n = min(len(images_np), max_samples)
        images_np = images_np[:n]
        
        # Determine grid size
        cols = min(4, n)
        rows = (n + cols - 1) // cols
        
        # Get single image dimensions
        img_h, img_w = images_np.shape[1:3]
        
        # Create output canvas
        canvas_h = rows * img_h
        canvas_w = cols * img_w
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        
        for i in range(n):
            # Get image
            img = self._prepare_image(images_np[i])
            
            # Get targets and predictions for this image
            img_targets = self._get_boxes_for_image(targets, i) if targets is not None else None
            img_preds = self._get_boxes_for_image(predictions, i) if predictions is not None else None
            
            # Draw boxes on image
            if img_targets is not None:
                img = self._draw_boxes(
                    img, img_targets, 
                    color=(0, 255, 0),  # Green for GT
                    normalized=normalized,
                    format='target',
                )
            
            if img_preds is not None:
                img = self._draw_boxes(
                    img, img_preds,
                    color=(0, 0, 255),  # Red for predictions
                    normalized=normalized,
                    format='prediction',
                )
            
            # Place in canvas
            row = i // cols
            col = i % cols
            y1 = row * img_h
            y2 = y1 + img_h
            x1 = col * img_w
            x2 = x1 + img_w
            canvas[y1:y2, x1:x2] = img
        
        # Save
        cv2.imwrite(str(save_path), canvas)
        print(f'Saved batch visualization to {save_path}')
    
    def _to_numpy(self, images: Any) -> np.ndarray:
        """Convert images to numpy array."""
        if HAS_TORCH and isinstance(images, torch.Tensor):
            images = images.detach().cpu().numpy()
        
        images = np.array(images)
        
        # Handle different shapes
        if images.ndim == 3:
            # Single image: add batch dimension
            images = images[np.newaxis, ...]
        
        # Convert from (N, C, H, W) to (N, H, W, C) if needed
        if images.shape[1] in [1, 3, 20, 21]:  # Likely channel-first
            if images.ndim == 4 and images.shape[1] < images.shape[2]:
                images = np.transpose(images, (0, 2, 3, 1))
        
        return images
    
    def _prepare_image(self, img: np.ndarray) -> np.ndarray:
        """Prepare a single image for visualization."""
        # Handle different channel counts
        if img.ndim == 2:
            # Grayscale
            img = np.stack([img, img, img], axis=-1)
        elif img.shape[-1] == 1:
            # Single channel
            img = np.repeat(img, 3, axis=-1)
        elif img.shape[-1] > 3:
            # Multi-channel (event representation)
            img = self._event_repr_to_image(img)
        
        # Normalize to 0-255
        if img.dtype == np.float32 or img.dtype == np.float64:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = np.clip(img, 0, 255).astype(np.uint8)
        elif img.dtype != np.uint8:
            img = img.astype(np.uint8)
        
        # Ensure 3 channels
        if img.shape[-1] != 3:
            img = img[:, :, :3]
        
        return img
    
    def _event_repr_to_image(self, ev_repr: np.ndarray) -> np.ndarray:
        """
        Convert event representation to RGB image.
        
        Handles stacked histograms and other event encodings by
        visualizing positive/negative events.
        """
        ch = ev_repr.shape[-1]
        ht, wd = ev_repr.shape[:2]
        
        # Handle odd channel counts (e.g., with radar channel)
        if ch > 1 and ch % 2 != 0:
            ev_repr = ev_repr[..., :-1]
            ch = ch - 1
        
        if ch < 2:
            # Single channel - grayscale
            img = ev_repr[..., 0]
            img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
            return np.stack([img, img, img], axis=-1)
        
        # Split into positive and negative
        half_ch = ch // 2
        neg_events = ev_repr[..., :half_ch].sum(axis=-1)
        pos_events = ev_repr[..., half_ch:].sum(axis=-1)
        
        # Create visualization
        # Red for negative, blue for positive, gray for neutral
        img = np.ones((ht, wd, 3), dtype=np.uint8) * 127
        
        # Normalize
        max_val = max(abs(neg_events).max(), abs(pos_events).max(), 1e-8)
        
        # Positive events -> more blue
        pos_mask = pos_events > neg_events
        pos_intensity = np.clip(pos_events / max_val * 255, 0, 255).astype(np.uint8)
        img[pos_mask, 0] = pos_intensity[pos_mask]  # Blue
        img[pos_mask, 1] = 127
        img[pos_mask, 2] = 127
        
        # Negative events -> more red
        neg_mask = neg_events > pos_events
        neg_intensity = np.clip(neg_events / max_val * 255, 0, 255).astype(np.uint8)
        img[neg_mask, 0] = 127
        img[neg_mask, 1] = 127
        img[neg_mask, 2] = neg_intensity[neg_mask]  # Red
        
        return img
    
    def _get_boxes_for_image(self, boxes: Any, idx: int) -> Optional[np.ndarray]:
        """Get boxes for a specific image index."""
        if boxes is None:
            return None
        
        # Handle different formats
        if HAS_TORCH and isinstance(boxes, torch.Tensor):
            boxes = boxes.detach().cpu().numpy()
        
        if isinstance(boxes, list):
            if idx < len(boxes):
                box = boxes[idx]
                if HAS_TORCH and isinstance(box, torch.Tensor):
                    return box.detach().cpu().numpy()
                return np.array(box) if len(box) > 0 else None
            return None
        
        # Assume batch format with image index column
        boxes = np.array(boxes)
        if boxes.ndim == 2 and boxes.shape[1] >= 6:
            # Format: [batch_idx, class, x, y, w, h, ...]
            mask = boxes[:, 0] == idx
            if mask.any():
                return boxes[mask, 1:]  # Remove batch index
        
        return None
    
    def _draw_boxes(
        self,
        img: np.ndarray,
        boxes: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0),
        normalized: bool = True,
        format: str = 'target',
    ) -> np.ndarray:
        """
        Draw bounding boxes on an image.
        
        Args:
            img: Image to draw on
            boxes: Boxes to draw
            color: Box color (BGR)
            normalized: Whether coordinates are normalized
            format: 'target' for [class, cx, cy, w, h] or 'prediction' for [x1, y1, x2, y2, conf, class]
        """
        if len(boxes) == 0:
            return img
        
        img = img.copy()
        h, w = img.shape[:2]
        
        for box in boxes:
            if format == 'target':
                # [class_id, cx, cy, w, h] or [class_id, x1, y1, x2, y2]
                class_id = int(box[0])
                
                # Detect format: normalized xywh vs xyxy
                if normalized and all(0 <= box[i] <= 1 for i in range(1, 5)):
                    # Normalized xywh format
                    cx, cy, bw, bh = box[1:5]
                    x1 = int((cx - bw / 2) * w)
                    y1 = int((cy - bh / 2) * h)
                    x2 = int((cx + bw / 2) * w)
                    y2 = int((cy + bh / 2) * h)
                else:
                    # xyxy format
                    x1, y1, x2, y2 = int(box[1]), int(box[2]), int(box[3]), int(box[4])
                
                conf = None
                
            else:  # prediction format
                # [x1, y1, x2, y2, conf, class_id]
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                conf = float(box[4]) if len(box) > 4 else None
                class_id = int(box[5]) if len(box) > 5 else 0
            
            # Clamp coordinates
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))
            
            # Get color for class
            box_color = self.COLORS[class_id % len(self.COLORS)] if format == 'target' else color
            
            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), box_color, self.line_thickness)
            
            # Draw label
            class_name = self.class_names[class_id] if self.class_names and class_id < len(self.class_names) else f'{class_id}'
            label = f'{class_name}'
            if conf is not None:
                label += f' {conf:.2f}'
            
            # Get text size
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1
            )
            
            # Draw label background
            cv2.rectangle(
                img,
                (x1, y1 - text_h - baseline - 2),
                (x1 + text_w, y1),
                box_color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                img, label,
                (x1, y1 - baseline - 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                (255, 255, 255),
                1
            )
        
        return img
    
    def save_single_image(
        self,
        img: Any,
        boxes: Optional[Any] = None,
        save_path: Union[str, Path] = 'image.jpg',
        normalized: bool = True,
        format: str = 'target',
    ) -> None:
        """
        Save a single image with boxes.
        
        Args:
            img: Single image
            boxes: Boxes to draw
            save_path: Path to save
            normalized: Whether coordinates are normalized
            format: Box format ('target' or 'prediction')
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to numpy
        img_np = self._to_numpy(img[np.newaxis, ...])[0]
        img_np = self._prepare_image(img_np)
        
        # Draw boxes if provided
        if boxes is not None:
            if HAS_TORCH and isinstance(boxes, torch.Tensor):
                boxes = boxes.detach().cpu().numpy()
            boxes = np.array(boxes)
            
            if len(boxes) > 0:
                img_np = self._draw_boxes(
                    img_np, boxes,
                    normalized=normalized,
                    format=format,
                )
        
        cv2.imwrite(str(save_path), img_np)


















