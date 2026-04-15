"""
Results Plotter.

Generates training progress visualizations from CSV logs.
Creates results.png with learning curves for losses and metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


def plot_results(
    csv_file: Union[str, Path],
    save_dir: Optional[Union[str, Path]] = None,
    figsize: tuple = (16, 12),
) -> None:
    """
    Generate results.png from a CSV file with training metrics.
    
    Args:
        csv_file: Path to the CSV file containing training metrics
        save_dir: Directory to save results.png (defaults to same dir as csv)
        figsize: Figure size (width, height)
    """
    csv_file = Path(csv_file)
    if not csv_file.exists():
        print(f'Warning: CSV file not found: {csv_file}')
        return
    
    if save_dir is None:
        save_dir = csv_file.parent
    else:
        save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Read CSV
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f'Error reading CSV file: {e}')
        return
    
    # Clean column names (strip whitespace)
    df.columns = [col.strip() for col in df.columns]
    
    # Create plotter and generate figure
    plotter = ResultsPlotter(df)
    plotter.plot(save_path=save_dir / 'results.png', figsize=figsize)


class ResultsPlotter:
    """
    Results plotter for training metrics.
    
    Handles different column naming conventions and creates
    a comprehensive visualization of training progress.
    """
    
    # Column name mappings for different models
    COLUMN_MAPPINGS = {
        # Loss columns
        'train_box_loss': ['train/box_loss', 'train_box_loss', 'box_loss', 'loss_iou'],
        'train_cls_loss': ['train/cls_loss', 'train_cls_loss', 'cls_loss', 'loss_cls'],
        'train_dfl_loss': ['train/dfl_loss', 'train_dfl_loss', 'dfl_loss', 'loss_dfl'],
        'train_obj_loss': ['train/obj_loss', 'train_obj_loss', 'obj_loss', 'loss_obj'],
        'train_total_loss': ['train/total_loss', 'train_total_loss', 'total_loss', 'loss'],
        
        # Validation loss columns
        'val_box_loss': ['val/box_loss', 'val_box_loss'],
        'val_cls_loss': ['val/cls_loss', 'val_cls_loss'],
        'val_dfl_loss': ['val/dfl_loss', 'val_dfl_loss'],
        'val_obj_loss': ['val/obj_loss', 'val_obj_loss'],
        'val_total_loss': ['val/total_loss', 'val_total_loss', 'val_loss'],
        
        # Metric columns
        'precision': ['metrics/precision(B)', 'metrics/precision', 'precision', 'P'],
        'recall': ['metrics/recall(B)', 'metrics/recall', 'recall', 'R'],
        'mAP50': ['metrics/mAP50(B)', 'metrics/mAP50', 'mAP_0.5', 'mAP50', 'AP50'],
        'mAP50_95': ['metrics/mAP50-95(B)', 'metrics/mAP50-95', 'mAP_0.5:0.95', 'mAP50-95', 'mAP'],
        
        # Learning rate
        'lr': ['lr/pg0', 'lr0', 'lr', 'learning_rate'],
        
        # Distance metrics (if available)
        'distance_loss': ['train/distance_loss', 'distance_loss'],
        'distance_mae': ['metrics/distance_mae', 'distance_mae'],
    }
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with a DataFrame of training results.
        
        Args:
            df: DataFrame with training metrics
        """
        self.df = df
        self.mapped_columns = self._map_columns()
    
    def _map_columns(self) -> Dict[str, str]:
        """Map standardized column names to actual DataFrame columns."""
        mapped = {}
        available_cols = set(self.df.columns)
        
        for standard_name, alternatives in self.COLUMN_MAPPINGS.items():
            for alt in alternatives:
                if alt in available_cols:
                    mapped[standard_name] = alt
                    break
        
        return mapped
    
    def _get_column(self, name: str) -> Optional[np.ndarray]:
        """Get column data by standardized name."""
        if name in self.mapped_columns:
            col_name = self.mapped_columns[name]
            return self.df[col_name].values
        return None
    
    def _has_column(self, name: str) -> bool:
        """Check if a column exists."""
        return name in self.mapped_columns
    
    def plot(
        self,
        save_path: Union[str, Path],
        figsize: tuple = (16, 12),
    ) -> None:
        """
        Generate and save the results plot.
        
        Args:
            save_path: Path to save the plot
            figsize: Figure size (width, height)
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine layout based on available columns
        has_distance = self._has_column('distance_loss') or self._has_column('distance_mae')
        
        if has_distance:
            fig, axes = plt.subplots(3, 3, figsize=figsize, tight_layout=True)
            axes = axes.flatten()
        else:
            fig, axes = plt.subplots(2, 4, figsize=figsize, tight_layout=True)
            axes = axes.flatten()
        
        x = np.arange(len(self.df))
        
        plot_idx = 0
        
        # Plot training losses
        for loss_name in ['train_box_loss', 'train_cls_loss', 'train_dfl_loss', 'train_obj_loss']:
            data = self._get_column(loss_name)
            if data is not None and plot_idx < len(axes):
                ax = axes[plot_idx]
                ax.plot(x, data, color='blue', linewidth=2)
                ax.set_title(loss_name.replace('train_', '').replace('_', ' ').title(), fontsize=12)
                ax.set_xlabel('Epoch', fontsize=10)
                ax.set_ylabel('Loss', fontsize=10)
                ax.grid(True, alpha=0.3)
                plot_idx += 1
        
        # Plot validation losses if available
        val_box = self._get_column('val_box_loss')
        if val_box is not None and plot_idx < len(axes):
            ax = axes[plot_idx]
            ax.plot(x, val_box, color='orange', linewidth=2)
            ax.set_title('Val Box Loss', fontsize=12)
            ax.set_xlabel('Epoch', fontsize=10)
            ax.set_ylabel('Loss', fontsize=10)
            ax.grid(True, alpha=0.3)
            plot_idx += 1
        
        # Plot metrics
        for metric_name in ['precision', 'recall', 'mAP50', 'mAP50_95']:
            data = self._get_column(metric_name)
            if data is not None and plot_idx < len(axes):
                ax = axes[plot_idx]
                ax.plot(x, data, color='green', linewidth=2)
                title = metric_name.replace('_', '-')
                ax.set_title(title, fontsize=12)
                ax.set_xlabel('Epoch', fontsize=10)
                ax.set_ylabel('Value', fontsize=10)
                ax.set_ylim(0, 1)
                ax.grid(True, alpha=0.3)
                plot_idx += 1
        
        # Plot learning rate
        lr = self._get_column('lr')
        if lr is not None and plot_idx < len(axes):
            ax = axes[plot_idx]
            ax.plot(x, lr, color='purple', linewidth=2)
            ax.set_title('Learning Rate', fontsize=12)
            ax.set_xlabel('Epoch', fontsize=10)
            ax.set_ylabel('LR', fontsize=10)
            ax.grid(True, alpha=0.3)
            plot_idx += 1
        
        # Plot distance metrics if available
        if has_distance:
            dist_loss = self._get_column('distance_loss')
            if dist_loss is not None and plot_idx < len(axes):
                ax = axes[plot_idx]
                ax.plot(x, dist_loss, color='red', linewidth=2)
                ax.set_title('Distance Loss', fontsize=12)
                ax.set_xlabel('Epoch', fontsize=10)
                ax.set_ylabel('Loss', fontsize=10)
                ax.grid(True, alpha=0.3)
                plot_idx += 1
            
            dist_mae = self._get_column('distance_mae')
            if dist_mae is not None and plot_idx < len(axes):
                ax = axes[plot_idx]
                ax.plot(x, dist_mae, color='brown', linewidth=2)
                ax.set_title('Distance MAE', fontsize=12)
                ax.set_xlabel('Epoch', fontsize=10)
                ax.set_ylabel('MAE (m)', fontsize=10)
                ax.grid(True, alpha=0.3)
                plot_idx += 1
        
        # Hide unused axes
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)
        
        fig.suptitle('Training Results', fontsize=16, y=1.02)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved results plot to {save_path}')
    
    def get_best_epoch(self, metric: str = 'mAP50_95') -> Dict[str, Any]:
        """
        Get the best epoch based on a metric.
        
        Args:
            metric: Metric name to use for selection
            
        Returns:
            Dictionary with best epoch info
        """
        data = self._get_column(metric)
        if data is None:
            return {'epoch': -1, 'value': 0}
        
        best_idx = np.argmax(data)
        return {
            'epoch': best_idx,
            'value': data[best_idx],
        }
    
    def get_final_metrics(self) -> Dict[str, float]:
        """Get final epoch metrics."""
        metrics = {}
        
        for name in ['precision', 'recall', 'mAP50', 'mAP50_95']:
            data = self._get_column(name)
            if data is not None and len(data) > 0:
                metrics[name] = float(data[-1])
        
        return metrics
    
    def print_summary(self) -> None:
        """Print a summary of training results."""
        print('\n' + '=' * 60)
        print('Training Results Summary')
        print('=' * 60)
        
        final_metrics = self.get_final_metrics()
        best_map = self.get_best_epoch('mAP50_95')
        
        print(f'\nFinal Epoch Metrics:')
        for name, value in final_metrics.items():
            print(f'  {name}: {value:.4f}')
        
        if best_map['epoch'] >= 0:
            print(f'\nBest mAP50-95: {best_map["value"]:.4f} (epoch {best_map["epoch"]})')
        
        print('=' * 60 + '\n')


















