"""
Local logger with TensorBoard support and local file saving.
Used as a fallback when WandB is disabled.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from argparse import Namespace
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import csv

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.utilities.rank_zero import rank_zero_only

# Try to import shared visualization module
try:
    # Add parent directory to path to import from deep/visualization
    _viz_path = Path(__file__).parent.parent.parent / 'visualization'
    if _viz_path.exists():
        from visualization import (
            StandardizedCSVLogger,
            plot_results as plot_unified_results,
            plot_pr_curve,
            plot_f1_curve
        )
        UNIFIED_VIZ_AVAILABLE = True
    else:
        UNIFIED_VIZ_AVAILABLE = False
except ImportError:
    UNIFIED_VIZ_AVAILABLE = False


class LocalLogger(Logger):
    """
    A logger that saves outputs locally and optionally logs to TensorBoard.
    This serves as a fallback when WandB is unavailable or disabled.
    """
    
    LOGGER_JOIN_CHAR = "-"
    
    def __init__(
        self,
        save_dir: str = "runs",
        name: str = "rvt_experiment",
        version: Optional[str] = None,
        use_tensorboard: bool = True,
        **kwargs
    ):
        super().__init__()
        
        self._save_dir = Path(save_dir)
        self._name = name
        self._version = version or datetime.now().strftime("%Y%m%d_%H%M%S")
        self._use_tensorboard = use_tensorboard
        
        # Create directories
        self._log_dir = self._save_dir / self._name / self._version
        self._images_dir = self._log_dir / "images"
        self._metrics_dir = self._log_dir / "metrics"
        
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._images_dir.mkdir(parents=True, exist_ok=True)
        self._metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize TensorBoard logger if enabled
        self._tb_logger = None
        if use_tensorboard:
            self._tb_logger = TensorBoardLogger(
                save_dir=str(self._save_dir),
                name=self._name,
                version=self._version
            )
        
        # CSV file for metrics
        self._metrics_file = self._metrics_dir / "metrics.csv"
        self._metrics_header_written = False
        self._metrics_keys = set()
        
        # Initialize unified CSV logger if available
        self._unified_csv_logger = None
        if UNIFIED_VIZ_AVAILABLE:
            try:
                self._unified_csv_logger = StandardizedCSVLogger(self._log_dir)
                print(f"Unified CSV logger initialized")
            except Exception as e:
                print(f"Warning: Could not initialize unified CSV logger: {e}")
        
        print(f"LocalLogger initialized. Saving to: {self._log_dir}")
        if use_tensorboard:
            print(f"TensorBoard logs at: {self._tb_logger.log_dir}")
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def version(self) -> str:
        return self._version
    
    @property
    def log_dir(self) -> Path:
        return self._log_dir
    
    @property
    def experiment(self):
        """Return TensorBoard SummaryWriter if available."""
        if self._tb_logger:
            return self._tb_logger.experiment
        return None
    
    def watch(self, model: nn.Module, log: str = 'all', log_freq: int = 100, log_graph: bool = True):
        """Watch model (no-op for local logger, but don't fail)."""
        pass
    
    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        """Log hyperparameters to a YAML file."""
        import yaml
        
        if isinstance(params, Namespace):
            params = vars(params)
        
        hparams_file = self._log_dir / "hparams.yaml"
        with open(hparams_file, 'w') as f:
            yaml.dump(params, f, default_flow_style=False)
        
        if self._tb_logger:
            self._tb_logger.log_hyperparams(params)
    
    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to CSV and TensorBoard."""
        # Filter out non-numeric values for TensorBoard
        numeric_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                numeric_metrics[k] = v
            elif isinstance(v, torch.Tensor):
                if v.numel() == 1:
                    numeric_metrics[k] = v.item()
        
        # Log to TensorBoard
        if self._tb_logger and numeric_metrics:
            for k, v in numeric_metrics.items():
                self._tb_logger.experiment.add_scalar(k, v, global_step=step)
        
        # Log to CSV
        if numeric_metrics:
            self._log_to_csv(numeric_metrics, step)
        
        # Also log to unified CSV logger for standardized format
        if self._unified_csv_logger is not None and numeric_metrics:
            try:
                # Map RVT metrics to standardized format
                unified_metrics = {
                    'epoch': numeric_metrics.get('epoch', step // 1000 if step else 0),
                    'train/box_loss': numeric_metrics.get('train/box_loss', numeric_metrics.get('train_loss', 0)),
                    'train/cls_loss': numeric_metrics.get('train/cls_loss', 0),
                    'train/dfl_loss': numeric_metrics.get('train/dfl_loss', 0),
                    'metrics/precision': numeric_metrics.get('val/precision', numeric_metrics.get('val/AP50', 0)),
                    'metrics/recall': numeric_metrics.get('val/recall', 0),
                    'metrics/mAP50': numeric_metrics.get('val/AP50', numeric_metrics.get('val/mAP', 0)),
                    'metrics/mAP50-95': numeric_metrics.get('val/AP', 0),
                    'val/box_loss': numeric_metrics.get('val/box_loss', numeric_metrics.get('val_loss', 0)),
                    'val/cls_loss': numeric_metrics.get('val/cls_loss', 0),
                    'val/dfl_loss': numeric_metrics.get('val/dfl_loss', 0),
                    'lr/pg0': numeric_metrics.get('lr', numeric_metrics.get('lr/pg0', 0)),
                }
                self._unified_csv_logger.log(unified_metrics)
            except Exception as e:
                print(f"Warning: Failed to log to unified CSV: {e}")
    
    def _log_to_csv(self, metrics: Dict[str, Any], step: Optional[int]):
        """Append metrics to CSV file."""
        metrics_with_step = {'step': step, **metrics}
        
        # Check if we need to update header
        new_keys = set(metrics_with_step.keys()) - self._metrics_keys
        
        if new_keys or not self._metrics_header_written:
            self._metrics_keys.update(metrics_with_step.keys())
            
            # Rewrite CSV with new header if keys changed
            if self._metrics_file.exists() and new_keys and self._metrics_header_written:
                # Read existing data
                existing_data = []
                with open(self._metrics_file, 'r') as f:
                    reader = csv.DictReader(f)
                    existing_data = list(reader)
                
                # Rewrite with new header
                with open(self._metrics_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=sorted(self._metrics_keys))
                    writer.writeheader()
                    for row in existing_data:
                        writer.writerow(row)
            
            self._metrics_header_written = True
        
        # Append new row
        with open(self._metrics_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=sorted(self._metrics_keys))
            if os.path.getsize(self._metrics_file) == 0:
                writer.writeheader()
            writer.writerow({k: metrics_with_step.get(k, '') for k in sorted(self._metrics_keys)})
    
    @rank_zero_only
    def log_images(
        self,
        key: str,
        images: List[Any],
        step: Optional[int] = None,
        caption: Optional[List[str]] = None,
        **kwargs
    ) -> None:
        """
        Save images locally and log to TensorBoard.
        
        Args:
            key: Identifier for the images (e.g., 'train/predictions', 'val/predictions')
            images: List of images (numpy arrays or tensors)
            step: Global step
            caption: Optional captions for each image
        """
        if not isinstance(images, list):
            images = [images]
        
        # Create subdirectory for this key
        key_clean = key.replace('/', '_')
        step_str = f"step_{step:06d}" if step is not None else "step_none"
        save_subdir = self._images_dir / key_clean / step_str
        save_subdir.mkdir(parents=True, exist_ok=True)
        
        for idx, img in enumerate(images):
            # Convert to numpy if tensor
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()
            
            # Handle different image formats
            if isinstance(img, np.ndarray):
                # Ensure proper shape and type for PIL
                if img.ndim == 4:  # Batch dimension
                    img = img[0]
                if img.ndim == 3:
                    if img.shape[0] in [1, 3, 4]:  # CHW format
                        img = np.transpose(img, (1, 2, 0))
                    if img.shape[2] == 1:  # Grayscale
                        img = img.squeeze(2)
                
                # Normalize to 0-255 if needed
                if img.dtype == np.float32 or img.dtype == np.float64:
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)
                
                # Save image
                caption_str = caption[idx] if caption and idx < len(caption) else f"img_{idx:03d}"
                caption_clean = caption_str.replace('/', '_').replace(' ', '_')
                img_path = save_subdir / f"{caption_clean}.png"
                
                pil_img = Image.fromarray(img)
                pil_img.save(img_path)
                
                # Also log to TensorBoard if available
                if self._tb_logger and step is not None:
                    # Convert back to CHW for TensorBoard
                    if img.ndim == 2:
                        tb_img = np.expand_dims(img, 0)
                    elif img.ndim == 3 and img.shape[2] in [1, 3, 4]:
                        tb_img = np.transpose(img, (2, 0, 1))
                    else:
                        tb_img = img
                    
                    self._tb_logger.experiment.add_image(
                        f"{key}/{caption_clean}",
                        tb_img,
                        global_step=step,
                        dataformats='CHW' if tb_img.ndim == 3 else 'HW'
                    )
        
        print(f"Saved {len(images)} images to: {save_subdir}")
    
    @rank_zero_only
    def log_text(self, key: str, text: str, step: Optional[int] = None) -> None:
        """Log text to a file."""
        text_file = self._log_dir / f"{key.replace('/', '_')}.txt"
        with open(text_file, 'a') as f:
            f.write(f"[step={step}] {text}\n")
        
        if self._tb_logger:
            self._tb_logger.experiment.add_text(key, text, global_step=step)
    
    @rank_zero_only
    def log_videos(
        self,
        key: str,
        videos: List[Union[np.ndarray, str]],
        step: Optional[int] = None,
        captions: Optional[List[str]] = None,
        fps: int = 4,
        format_: Optional[str] = None
    ) -> None:
        """
        Save videos locally.
        
        Args:
            key: Identifier for the videos
            videos: List of video arrays (T,C,H,W) or file paths
            step: Global step
            captions: Optional captions
            fps: Frames per second
            format_: Video format (ignored for local saving, uses mp4)
        """
        try:
            import cv2
            
            key_clean = key.replace('/', '_')
            step_str = f"step_{step:06d}" if step is not None else "step_none"
            save_subdir = self._log_dir / "videos" / key_clean / step_str
            save_subdir.mkdir(parents=True, exist_ok=True)
            
            for idx, video in enumerate(videos):
                caption = captions[idx] if captions and idx < len(captions) else f"video_{idx:03d}"
                caption_clean = caption.replace('/', '_').replace(' ', '_')
                video_path = save_subdir / f"{caption_clean}.mp4"
                
                if isinstance(video, str):
                    # Copy existing video file
                    import shutil
                    shutil.copy(video, video_path)
                elif isinstance(video, np.ndarray):
                    # Write video from array
                    if video.ndim == 4:  # T, C, H, W
                        t, c, h, w = video.shape
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        out = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
                        
                        for frame_idx in range(t):
                            frame = video[frame_idx]
                            if c == 3:
                                frame = np.transpose(frame, (1, 2, 0))
                                frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
                            elif c == 1:
                                frame = frame[0]
                                frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_GRAY2BGR)
                            out.write(frame)
                        
                        out.release()
            
            print(f"Saved {len(videos)} videos to: {save_subdir}")
            
            # Also log to TensorBoard if available
            if self._tb_logger and step is not None:
                for idx, video in enumerate(videos):
                    if isinstance(video, np.ndarray) and video.ndim == 4:
                        # TensorBoard expects (N, T, C, H, W)
                        video_tb = np.expand_dims(video, 0)
                        self._tb_logger.experiment.add_video(
                            f"{key}/{idx}",
                            video_tb,
                            global_step=step,
                            fps=fps
                        )
        except ImportError:
            print(f"Warning: cv2 not available, skipping video save for {key}")
        except Exception as e:
            print(f"Warning: Could not save video for {key}: {e}")
    
    def save_model_summary(self, model: nn.Module):
        """Save model summary to a text file."""
        summary_file = self._log_dir / "model_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(str(model))
            f.write("\n\nParameter count:\n")
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            f.write(f"Total: {total_params:,}\n")
            f.write(f"Trainable: {trainable_params:,}\n")
    
    def finalize(self, status: str) -> None:
        """Finalize logging."""
        # Save final status
        status_file = self._log_dir / "status.txt"
        with open(status_file, 'w') as f:
            f.write(f"Final status: {status}\n")
            f.write(f"Completed at: {datetime.now().isoformat()}\n")
        
        # Close unified CSV logger and generate results.png
        if self._unified_csv_logger is not None:
            try:
                self._unified_csv_logger.close()
                # Generate unified results.png
                if UNIFIED_VIZ_AVAILABLE:
                    unified_csv = self._log_dir / 'results_unified.csv'
                    if unified_csv.exists():
                        plot_unified_results(str(unified_csv), str(self._log_dir))
                        print(f"Results plot saved to: {self._log_dir / 'results_unified.png'}")
            except Exception as e:
                print(f"Warning: Failed to finalize unified CSV: {e}")
        
        if self._tb_logger:
            self._tb_logger.finalize(status)
        
        print(f"Training {status}. Results saved to: {self._log_dir}")

