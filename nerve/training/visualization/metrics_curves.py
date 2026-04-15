"""
Metrics Curves Visualization.

Provides functions to plot:
- Precision-Recall (PR) curves
- F1 vs confidence curves
- Precision vs confidence curves
- Recall vs confidence curves

These curves help visualize model performance across different confidence thresholds.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Union, Tuple
import warnings

# Suppress matplotlib warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


def smooth(y: np.ndarray, f: float = 0.05) -> np.ndarray:
    """Smooth a curve using box filter."""
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode='valid')  # y-smoothed


def plot_pr_curve(
    precision: np.ndarray,
    recall: np.ndarray,
    ap: Union[np.ndarray, float],
    save_path: Union[str, Path],
    class_names: Optional[List[str]] = None,
    title: str = 'Precision-Recall Curve',
    figsize: Tuple[int, int] = (10, 8),
) -> None:
    """
    Plot Precision-Recall curve.

    Args:
        precision: Precision values, shape (num_classes, num_thresholds) or (num_thresholds,)
        recall: Recall values (x-axis), shape (num_thresholds,)
        ap: Average Precision per class or single AP value
        save_path: Path to save the plot
        class_names: List of class names
        title: Plot title
        figsize: Figure size (width, height)
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(1, 1, figsize=figsize, tight_layout=True)
    
    # Handle different input shapes
    if precision.ndim == 1:
        precision = precision.reshape(1, -1)
        ap = np.array([ap]) if isinstance(ap, (int, float)) else ap
    
    num_classes = precision.shape[0]
    
    # Define colors
    colors = plt.cm.tab10(np.linspace(0, 1, min(num_classes, 10)))
    
    if num_classes <= 10:
        # Plot each class
        for i in range(num_classes):
            class_name = class_names[i] if class_names else f'Class {i}'
            ap_val = ap[i] if isinstance(ap, np.ndarray) and len(ap) > i else ap
            ax.plot(recall, precision[i], color=colors[i % len(colors)], 
                   linewidth=2, label=f'{class_name} (AP={ap_val:.3f})')
    
    # Plot mean PR curve
    if num_classes > 1:
        mean_precision = precision.mean(axis=0)
        mean_ap = ap.mean() if isinstance(ap, np.ndarray) else ap
        ax.plot(recall, mean_precision, color='blue', linewidth=3, linestyle='--',
               label=f'All classes (mAP={mean_ap:.3f})')
    else:
        mean_ap = ap[0] if isinstance(ap, np.ndarray) else ap
        ax.set_title(f'{title} (AP={mean_ap:.3f})', fontsize=14)
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f'Saved PR curve to {save_path}')


def plot_f1_curve(
    precision: np.ndarray,
    recall: np.ndarray,
    save_path: Union[str, Path],
    class_names: Optional[List[str]] = None,
    confidence: Optional[np.ndarray] = None,
    title: str = 'F1-Confidence Curve',
    figsize: Tuple[int, int] = (10, 8),
) -> None:
    """
    Plot F1 score vs confidence threshold curve.

    Args:
        precision: Precision values per confidence threshold, shape (num_classes, num_thresholds)
        recall: Recall values per confidence threshold, shape (num_classes, num_thresholds)
        save_path: Path to save the plot
        class_names: List of class names
        confidence: Confidence thresholds (x-axis)
        title: Plot title
        figsize: Figure size (width, height)
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(1, 1, figsize=figsize, tight_layout=True)
    
    # Handle different input shapes
    if precision.ndim == 1:
        precision = precision.reshape(1, -1)
        recall = recall.reshape(1, -1) if recall.ndim == 1 else recall
    
    num_classes = precision.shape[0]
    num_thresholds = precision.shape[1]
    
    # Generate confidence values if not provided
    if confidence is None:
        confidence = np.linspace(0, 1, num_thresholds)
    
    # Calculate F1 scores
    # F1 = 2 * (precision * recall) / (precision + recall + eps)
    eps = 1e-16
    f1 = 2 * precision * recall / (precision + recall + eps)
    
    # Define colors
    colors = plt.cm.tab10(np.linspace(0, 1, min(num_classes, 10)))
    
    if num_classes <= 10:
        for i in range(num_classes):
            class_name = class_names[i] if class_names else f'Class {i}'
            f1_smooth = smooth(f1[i])
            best_idx = f1_smooth.argmax()
            best_f1 = f1_smooth[best_idx]
            best_conf = confidence[best_idx]
            ax.plot(confidence, f1_smooth, color=colors[i % len(colors)],
                   linewidth=2, label=f'{class_name} (F1={best_f1:.3f} @ {best_conf:.3f})')
    
    # Plot mean F1 curve
    mean_f1 = f1.mean(axis=0)
    mean_f1_smooth = smooth(mean_f1)
    best_idx = mean_f1_smooth.argmax()
    best_f1 = mean_f1_smooth[best_idx]
    best_conf = confidence[best_idx]
    
    ax.plot(confidence, mean_f1_smooth, color='blue', linewidth=3, linestyle='--',
           label=f'All classes (F1={best_f1:.3f} @ {best_conf:.3f})')
    
    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f'Saved F1 curve to {save_path}')


def plot_precision_curve(
    confidence: np.ndarray,
    precision: np.ndarray,
    save_path: Union[str, Path],
    class_names: Optional[List[str]] = None,
    title: str = 'Precision-Confidence Curve',
    figsize: Tuple[int, int] = (10, 8),
) -> None:
    """
    Plot Precision vs confidence threshold curve.

    Args:
        confidence: Confidence thresholds (x-axis)
        precision: Precision values per threshold, shape (num_classes, num_thresholds)
        save_path: Path to save the plot
        class_names: List of class names
        title: Plot title
        figsize: Figure size (width, height)
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(1, 1, figsize=figsize, tight_layout=True)
    
    # Handle different input shapes
    if precision.ndim == 1:
        precision = precision.reshape(1, -1)
    
    num_classes = precision.shape[0]
    
    # Define colors
    colors = plt.cm.tab10(np.linspace(0, 1, min(num_classes, 10)))
    
    if num_classes <= 10:
        for i in range(num_classes):
            class_name = class_names[i] if class_names else f'Class {i}'
            p_smooth = smooth(precision[i])
            ax.plot(confidence, p_smooth, color=colors[i % len(colors)],
                   linewidth=2, label=f'{class_name}')
    
    # Plot mean precision curve
    mean_precision = precision.mean(axis=0)
    mean_precision_smooth = smooth(mean_precision)
    ax.plot(confidence, mean_precision_smooth, color='blue', linewidth=3, linestyle='--',
           label='All classes')
    
    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f'Saved Precision curve to {save_path}')


def plot_recall_curve(
    confidence: np.ndarray,
    recall: np.ndarray,
    save_path: Union[str, Path],
    class_names: Optional[List[str]] = None,
    title: str = 'Recall-Confidence Curve',
    figsize: Tuple[int, int] = (10, 8),
) -> None:
    """
    Plot Recall vs confidence threshold curve.

    Args:
        confidence: Confidence thresholds (x-axis)
        recall: Recall values per threshold, shape (num_classes, num_thresholds)
        save_path: Path to save the plot
        class_names: List of class names
        title: Plot title
        figsize: Figure size (width, height)
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(1, 1, figsize=figsize, tight_layout=True)
    
    # Handle different input shapes
    if recall.ndim == 1:
        recall = recall.reshape(1, -1)
    
    num_classes = recall.shape[0]
    
    # Define colors
    colors = plt.cm.tab10(np.linspace(0, 1, min(num_classes, 10)))
    
    if num_classes <= 10:
        for i in range(num_classes):
            class_name = class_names[i] if class_names else f'Class {i}'
            r_smooth = smooth(recall[i])
            ax.plot(confidence, r_smooth, color=colors[i % len(colors)],
                   linewidth=2, label=f'{class_name}')
    
    # Plot mean recall curve
    mean_recall = recall.mean(axis=0)
    mean_recall_smooth = smooth(mean_recall)
    ax.plot(confidence, mean_recall_smooth, color='blue', linewidth=3, linestyle='--',
           label='All classes')
    
    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Recall', fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f'Saved Recall curve to {save_path}')


def plot_all_curves(
    precision: np.ndarray,
    recall: np.ndarray,
    ap: Union[np.ndarray, float],
    save_dir: Union[str, Path],
    class_names: Optional[List[str]] = None,
    confidence: Optional[np.ndarray] = None,
) -> None:
    """
    Plot all metric curves (PR, F1, Precision, Recall).

    Args:
        precision: Precision values, shape (num_classes, num_thresholds)
        recall: Recall values, shape (num_classes, num_thresholds) or (num_thresholds,) for PR x-axis
        ap: Average Precision per class
        save_dir: Directory to save all plots
        class_names: List of class names
        confidence: Confidence thresholds (for F1, P, R curves)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle recall shape for PR curve
    recall_for_pr = recall if recall.ndim == 1 else recall.mean(axis=0)
    
    # Generate confidence if not provided
    if confidence is None:
        num_thresholds = precision.shape[-1] if precision.ndim > 1 else len(precision)
        confidence = np.linspace(0, 1, num_thresholds)
    
    # Ensure 2D arrays for other curves
    if precision.ndim == 1:
        precision_2d = precision.reshape(1, -1)
    else:
        precision_2d = precision
    
    if recall.ndim == 1:
        recall_2d = recall.reshape(1, -1)
    else:
        recall_2d = recall
    
    # Plot PR curve
    plot_pr_curve(
        precision=precision_2d,
        recall=recall_for_pr,
        ap=ap,
        save_path=save_dir / 'PR_curve.png',
        class_names=class_names,
    )
    
    # Plot F1 curve
    plot_f1_curve(
        precision=precision_2d,
        recall=recall_2d,
        save_path=save_dir / 'F1_curve.png',
        class_names=class_names,
        confidence=confidence,
    )
    
    # Plot Precision curve
    plot_precision_curve(
        confidence=confidence,
        precision=precision_2d,
        save_path=save_dir / 'P_curve.png',
        class_names=class_names,
    )
    
    # Plot Recall curve
    plot_recall_curve(
        confidence=confidence,
        recall=recall_2d,
        save_path=save_dir / 'R_curve.png',
        class_names=class_names,
    )
    
    print(f'All metric curves saved to {save_dir}')


















