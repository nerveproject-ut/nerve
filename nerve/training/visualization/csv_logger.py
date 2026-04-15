"""
Standardized CSV Logger.

Provides consistent CSV logging format across all models in the PEGMA pipeline.
"""

import csv
from pathlib import Path
from typing import Dict, Optional, Union, List, Any
from datetime import datetime


class StandardizedCSVLogger:
    """
    Standardized CSV logger for training metrics.
    
    Ensures consistent column format across YOLOX, YOLOv8, ReYOLOv8, and RVT.
    """
    
    # Standard columns for detection models
    STANDARD_COLUMNS = [
        'epoch',
        'train/box_loss',
        'train/cls_loss',
        'train/dfl_loss',
        'train/obj_loss',
        'train/total_loss',
        'val/box_loss',
        'val/cls_loss',
        'val/dfl_loss',
        'val/total_loss',
        'metrics/precision',
        'metrics/recall',
        'metrics/mAP50',
        'metrics/mAP50-95',
        'lr',
    ]
    
    # Additional columns for distance estimation
    DISTANCE_COLUMNS = [
        'train/distance_loss',
        'metrics/distance_mae',
        'metrics/distance_rmse',
    ]
    
    def __init__(
        self,
        save_dir: Union[str, Path],
        filename: str = 'results.csv',
        include_distance: bool = False,
        extra_columns: Optional[List[str]] = None,
    ):
        """
        Initialize the CSV logger.
        
        Args:
            save_dir: Directory to save the CSV file
            filename: Name of the CSV file
            include_distance: Whether to include distance estimation columns
            extra_columns: Additional custom columns to include
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.filepath = self.save_dir / filename
        self.include_distance = include_distance
        
        # Build column list
        self.columns = list(self.STANDARD_COLUMNS)
        if include_distance:
            self.columns.extend(self.DISTANCE_COLUMNS)
        if extra_columns:
            self.columns.extend(extra_columns)
        
        # Current epoch data buffer
        self._current_data: Dict[str, Any] = {}
        
        # Initialize file with header
        self._initialized = False
    
    def _ensure_initialized(self) -> None:
        """Ensure the CSV file is initialized with headers."""
        if not self._initialized:
            with open(self.filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.columns)
                writer.writeheader()
            self._initialized = True
    
    def log(self, metrics: Dict[str, float], epoch: Optional[int] = None) -> None:
        """
        Log metrics for an epoch.
        
        Args:
            metrics: Dictionary of metric name -> value
            epoch: Epoch number (optional if 'epoch' is in metrics)
        """
        self._ensure_initialized()
        
        # Prepare row data
        row = {col: '' for col in self.columns}
        
        # Set epoch
        if epoch is not None:
            row['epoch'] = epoch
        elif 'epoch' in metrics:
            row['epoch'] = metrics['epoch']
        
        # Map input metrics to standard columns
        for key, value in metrics.items():
            # Direct match
            if key in row:
                row[key] = self._format_value(value)
                continue
            
            # Try common variations
            normalized_key = self._normalize_key(key)
            if normalized_key in row:
                row[normalized_key] = self._format_value(value)
        
        # Write row
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writerow(row)
    
    def log_batch(self, key: str, value: float) -> None:
        """
        Log a single metric value (accumulates until flush).
        
        Args:
            key: Metric name
            value: Metric value
        """
        normalized_key = self._normalize_key(key)
        self._current_data[normalized_key] = value
    
    def flush(self, epoch: int) -> None:
        """
        Write accumulated batch metrics as a single row.
        
        Args:
            epoch: Epoch number
        """
        if self._current_data:
            self._current_data['epoch'] = epoch
            self.log(self._current_data, epoch)
            self._current_data = {}
    
    def _normalize_key(self, key: str) -> str:
        """Normalize a metric key to standard format."""
        # Common transformations
        key = key.strip()
        
        # Map common variations
        mappings = {
            'box_loss': 'train/box_loss',
            'cls_loss': 'train/cls_loss',
            'dfl_loss': 'train/dfl_loss',
            'obj_loss': 'train/obj_loss',
            'total_loss': 'train/total_loss',
            'loss': 'train/total_loss',
            
            'precision': 'metrics/precision',
            'recall': 'metrics/recall',
            'map50': 'metrics/mAP50',
            'mAP_0.5': 'metrics/mAP50',
            'map50-95': 'metrics/mAP50-95',
            'mAP_0.5:0.95': 'metrics/mAP50-95',
            'map': 'metrics/mAP50-95',
            
            'learning_rate': 'lr',
            'lr0': 'lr',
            'lr/pg0': 'lr',
            
            'distance_loss': 'train/distance_loss',
            'distance_mae': 'metrics/distance_mae',
            'distance_rmse': 'metrics/distance_rmse',
        }
        
        key_lower = key.lower()
        if key_lower in mappings:
            return mappings[key_lower]
        
        return key
    
    def _format_value(self, value: Any) -> str:
        """Format a value for CSV output."""
        if value is None:
            return ''
        if isinstance(value, float):
            if abs(value) < 0.0001:
                return f'{value:.6f}'
            return f'{value:.5f}'
        return str(value)
    
    def get_filepath(self) -> Path:
        """Get the path to the CSV file."""
        return self.filepath
    
    def close(self) -> None:
        """
        Close the logger and finalize the CSV file.
        
        This is called at the end of training to ensure all data is written.
        """
        # Flush any remaining data
        if self._current_data:
            # Use -1 as epoch if not specified (indicates incomplete epoch)
            epoch = self._current_data.get('epoch', -1)
            self.flush(epoch)
        
        print(f"CSV log saved to: {self.filepath}")
    
    @classmethod
    def from_existing(
        cls,
        csv_file: Union[str, Path],
    ) -> 'StandardizedCSVLogger':
        """
        Create a logger from an existing CSV file.
        
        Args:
            csv_file: Path to existing CSV file
            
        Returns:
            StandardizedCSVLogger instance
        """
        csv_file = Path(csv_file)
        logger = cls(
            save_dir=csv_file.parent,
            filename=csv_file.name,
        )
        logger._initialized = True
        return logger


class MetricsAccumulator:
    """
    Accumulator for computing epoch-level metrics from batch-level values.
    """
    
    def __init__(self):
        """Initialize the accumulator."""
        self._values: Dict[str, List[float]] = {}
        self._counts: Dict[str, int] = {}
    
    def update(self, key: str, value: float, count: int = 1) -> None:
        """
        Update accumulator with a new value.
        
        Args:
            key: Metric name
            value: Metric value (or sum of values)
            count: Number of samples this value represents
        """
        if key not in self._values:
            self._values[key] = []
            self._counts[key] = 0
        
        self._values[key].append(value * count)  # Store weighted sum
        self._counts[key] += count
    
    def get_mean(self, key: str) -> Optional[float]:
        """Get the mean value for a metric."""
        if key not in self._values or self._counts[key] == 0:
            return None
        
        total = sum(self._values[key])
        return total / self._counts[key]
    
    def get_all_means(self) -> Dict[str, float]:
        """Get mean values for all metrics."""
        return {key: self.get_mean(key) for key in self._values if self.get_mean(key) is not None}
    
    def reset(self) -> None:
        """Reset the accumulator."""
        self._values.clear()
        self._counts.clear()

