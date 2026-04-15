"""
Confusion Matrix Visualization.

Provides a ConfusionMatrix class for accumulating predictions during
evaluation and generating confusion matrix visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Union, Tuple
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


class ConfusionMatrix:
    """
    Confusion matrix for object detection evaluation.
    
    Accumulates detection results across batches and generates
    visualization of true positives, false positives, and false negatives.
    """
    
    def __init__(
        self,
        num_classes: int,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        class_names: Optional[List[str]] = None,
    ):
        """
        Initialize confusion matrix.
        
        Args:
            num_classes: Number of object classes
            conf_threshold: Confidence threshold for predictions
            iou_threshold: IoU threshold for matching predictions to ground truth
            class_names: Optional list of class names
        """
        self.num_classes = num_classes
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_names = class_names or [f'Class {i}' for i in range(num_classes)]
        
        # Matrix shape: (nc + 1) x (nc + 1) where last row/col is background
        # Rows: predicted classes, Columns: true classes
        self._matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)
    
    def reset(self) -> None:
        """Reset the confusion matrix."""
        self._matrix.fill(0)
    
    @property
    def matrix(self) -> np.ndarray:
        """Return the confusion matrix."""
        return self._matrix
    
    def process_batch(
        self,
        detections: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        """
        Process a batch of detections and labels.
        
        Args:
            detections: Array of detections, shape (N, 6+) with columns
                       [x1, y1, x2, y2, confidence, class_id, ...]
            labels: Array of ground truth, shape (M, 5+) with columns
                   [class_id, x1, y1, x2, y2, ...] or [class_id, cx, cy, w, h, ...]
        """
        if len(labels) == 0:
            # No ground truth - all detections are false positives
            if len(detections) > 0:
                for det in detections:
                    if det[4] >= self.conf_threshold:
                        pred_class = int(det[5])
                        if pred_class < self.num_classes:
                            self._matrix[pred_class, self.num_classes] += 1  # FP (predicted, no GT)
            return
        
        if len(detections) == 0:
            # No detections - all labels are false negatives
            for label in labels:
                true_class = int(label[0])
                if true_class < self.num_classes:
                    self._matrix[self.num_classes, true_class] += 1  # FN (no pred, GT exists)
            return
        
        # Filter detections by confidence
        mask = detections[:, 4] >= self.conf_threshold
        detections = detections[mask]
        
        if len(detections) == 0:
            # No detections after filtering - all labels are FN
            for label in labels:
                true_class = int(label[0])
                if true_class < self.num_classes:
                    self._matrix[self.num_classes, true_class] += 1
            return
        
        # Convert labels to xyxy format if needed
        labels_xyxy = self._convert_labels_to_xyxy(labels)
        
        # Compute IoU matrix
        iou_matrix = self._box_iou(detections[:, :4], labels_xyxy[:, 1:5])
        
        # Track which labels have been matched
        matched_labels = set()
        
        # Sort detections by confidence (descending)
        sorted_indices = np.argsort(-detections[:, 4])
        
        for det_idx in sorted_indices:
            det = detections[det_idx]
            pred_class = int(det[5])
            
            if pred_class >= self.num_classes:
                continue
            
            # Find best matching label
            best_iou = 0
            best_label_idx = -1
            
            for label_idx in range(len(labels_xyxy)):
                if label_idx in matched_labels:
                    continue
                
                true_class = int(labels_xyxy[label_idx, 0])
                iou = iou_matrix[det_idx, label_idx]
                
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_label_idx = label_idx
            
            if best_label_idx >= 0:
                # Match found
                true_class = int(labels_xyxy[best_label_idx, 0])
                if true_class < self.num_classes:
                    self._matrix[pred_class, true_class] += 1  # TP or class confusion
                    matched_labels.add(best_label_idx)
            else:
                # No match - false positive
                self._matrix[pred_class, self.num_classes] += 1
        
        # Count unmatched labels as false negatives
        for label_idx in range(len(labels_xyxy)):
            if label_idx not in matched_labels:
                true_class = int(labels_xyxy[label_idx, 0])
                if true_class < self.num_classes:
                    self._matrix[self.num_classes, true_class] += 1
    
    def _convert_labels_to_xyxy(self, labels: np.ndarray) -> np.ndarray:
        """
        Convert labels to [class_id, x1, y1, x2, y2] format.
        
        Args:
            labels: Labels in either [class_id, x1, y1, x2, y2] or [class_id, cx, cy, w, h] format
            
        Returns:
            Labels in [class_id, x1, y1, x2, y2] format
        """
        labels = np.array(labels)
        if len(labels) == 0:
            return labels
        
        # Check if format is xywh (normalized) by checking if values are small
        # Typically xywh format has values between 0 and 1
        if labels.shape[1] >= 5:
            coords = labels[:, 1:5]
            if np.all(coords >= 0) and np.all(coords <= 1.5):
                # Likely xywh format - convert to xyxy
                cx, cy, w, h = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2
                labels_xyxy = np.column_stack([labels[:, 0], x1, y1, x2, y2])
                if labels.shape[1] > 5:
                    labels_xyxy = np.column_stack([labels_xyxy, labels[:, 5:]])
                return labels_xyxy
        
        return labels
    
    def _box_iou(self, boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """
        Compute IoU between two sets of boxes.
        
        Args:
            boxes1: First set of boxes, shape (N, 4) in xyxy format
            boxes2: Second set of boxes, shape (M, 4) in xyxy format
            
        Returns:
            IoU matrix, shape (N, M)
        """
        n = len(boxes1)
        m = len(boxes2)
        
        if n == 0 or m == 0:
            return np.zeros((n, m))
        
        # Compute intersection
        x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
        y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
        x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
        y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
        
        inter_w = np.maximum(0, x2 - x1)
        inter_h = np.maximum(0, y2 - y1)
        intersection = inter_w * inter_h
        
        # Compute areas
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        # Compute union
        union = area1[:, None] + area2[None, :] - intersection
        
        # Compute IoU
        iou = intersection / (union + 1e-16)
        return iou
    
    def tp_fp(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return true positive and false positive counts per class.
        
        Returns:
            Tuple of (tp, fp) arrays, each of shape (num_classes,)
        """
        # True positives are diagonal elements (correct predictions)
        tp = np.diag(self._matrix)[:self.num_classes]
        
        # False positives are sum of row minus diagonal (wrong predictions)
        fp = self._matrix[:self.num_classes, :].sum(axis=1) - tp
        
        return tp.astype(np.int64), fp.astype(np.int64)
    
    def fn(self) -> np.ndarray:
        """
        Return false negative counts per class.
        
        Returns:
            Array of false negative counts, shape (num_classes,)
        """
        # False negatives are sum of column minus diagonal (missed detections)
        fn = self._matrix[:, :self.num_classes].sum(axis=0) - np.diag(self._matrix)[:self.num_classes]
        return fn.astype(np.int64)
    
    def precision(self) -> np.ndarray:
        """Calculate precision per class."""
        tp, fp = self.tp_fp()
        return tp / (tp + fp + 1e-16)
    
    def recall(self) -> np.ndarray:
        """Calculate recall per class."""
        tp, _ = self.tp_fp()
        fn = self.fn()
        return tp / (tp + fn + 1e-16)
    
    def plot(
        self,
        save_dir: Union[str, Path],
        names: Optional[List[str]] = None,
        normalize: bool = True,
        figsize: Tuple[int, int] = (12, 10),
    ) -> None:
        """
        Generate and save confusion matrix visualizations.
        
        Args:
            save_dir: Directory to save plots
            names: Class names (overrides self.class_names)
            normalize: Whether to also save normalized version
            figsize: Figure size (width, height)
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        names = names or self.class_names
        
        # Add 'background' for the extra row/col
        names_with_bg = list(names) + ['background']
        
        # Plot raw counts
        self._plot_matrix(
            matrix=self._matrix,
            save_path=save_dir / 'confusion_matrix.png',
            names=names_with_bg,
            title='Confusion Matrix',
            figsize=figsize,
            normalize=False,
        )
        
        # Plot normalized
        if normalize:
            # Normalize by column (true class)
            col_sums = self._matrix.sum(axis=0, keepdims=True)
            normalized = self._matrix.astype(np.float32) / (col_sums + 1e-16)
            
            self._plot_matrix(
                matrix=normalized,
                save_path=save_dir / 'confusion_matrix_normalized.png',
                names=names_with_bg,
                title='Confusion Matrix (Normalized)',
                figsize=figsize,
                normalize=True,
            )
    
    def _plot_matrix(
        self,
        matrix: np.ndarray,
        save_path: Path,
        names: List[str],
        title: str,
        figsize: Tuple[int, int],
        normalize: bool,
    ) -> None:
        """Internal method to plot a single confusion matrix."""
        fig, ax = plt.subplots(1, 1, figsize=figsize, tight_layout=True)
        
        n = len(names)
        
        # Plot heatmap
        im = ax.imshow(matrix, interpolation='nearest', cmap='Blues')
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Counts' if not normalize else 'Proportion', rotation=-90, va='bottom')
        
        # Set ticks
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(names, fontsize=8)
        
        # Add text annotations
        fmt = '.2f' if normalize else 'd'
        thresh = matrix.max() / 2
        for i in range(n):
            for j in range(n):
                val = matrix[i, j]
                if normalize:
                    text = f'{val:.2f}' if val > 0.01 else ''
                else:
                    text = f'{int(val)}' if val > 0 else ''
                
                ax.text(j, i, text, ha='center', va='center',
                       color='white' if matrix[i, j] > thresh else 'black',
                       fontsize=7)
        
        ax.set_xlabel('True Class', fontsize=12)
        ax.set_ylabel('Predicted Class', fontsize=12)
        ax.set_title(title, fontsize=14)
        
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
        print(f'Saved confusion matrix to {save_path}')
    
    def print_summary(self) -> None:
        """Print a summary of the confusion matrix."""
        tp, fp = self.tp_fp()
        fn = self.fn()
        precision = self.precision()
        recall = self.recall()
        
        print('\n' + '=' * 60)
        print('Confusion Matrix Summary')
        print('=' * 60)
        print(f'{"Class":<20} {"TP":<8} {"FP":<8} {"FN":<8} {"Precision":<10} {"Recall":<10}')
        print('-' * 60)
        
        for i, name in enumerate(self.class_names):
            print(f'{name:<20} {tp[i]:<8} {fp[i]:<8} {fn[i]:<8} {precision[i]:<10.3f} {recall[i]:<10.3f}')
        
        print('-' * 60)
        print(f'{"Total/Mean":<20} {tp.sum():<8} {fp.sum():<8} {fn.sum():<8} {precision.mean():<10.3f} {recall.mean():<10.3f}')
        print('=' * 60 + '\n')


















