"""
Unified Visualization Module for PEGMA Training Pipeline.

This module provides consistent visualization utilities for all models:
- YOLOX
- YOLOv8
- ReYOLOv8
- RVT

All models use these shared utilities to generate standardized outputs.
"""

from .metrics_curves import (
    plot_pr_curve,
    plot_f1_curve,
    plot_precision_curve,
    plot_recall_curve,
    plot_all_curves,
)
from .confusion_matrix import ConfusionMatrix
from .results_plotter import plot_results, ResultsPlotter
from .batch_visualizer import (
    plot_training_batch,
    plot_validation_batch,
    BatchVisualizer,
)
from .csv_logger import StandardizedCSVLogger

__all__ = [
    # Metric curves
    'plot_pr_curve',
    'plot_f1_curve',
    'plot_precision_curve',
    'plot_recall_curve',
    'plot_all_curves',
    # Confusion matrix
    'ConfusionMatrix',
    # Results plotting
    'plot_results',
    'ResultsPlotter',
    # Batch visualization
    'plot_training_batch',
    'plot_validation_batch',
    'BatchVisualizer',
    # CSV logging
    'StandardizedCSVLogger',
]


















