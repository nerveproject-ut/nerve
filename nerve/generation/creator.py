"""
@Author  :   Pietro Martinello
@Contact :   martin66@imec.be / pietromartinello.dev@gmail.com
"""

import os
import shutil
from tqdm import tqdm
import gc
import json
import argparse
import warnings

from nerve.generation.sources import *
from nerve.generation.label_writer import LabelWriter

import sys
from nerve.extraction.utils.dataset_utils import ResolveClassNamesToIds
from nerve.extraction.mapping.mapping_utils import MapLabels


def load_rgb_frame(session_path: str, frame_idx: int):
    """
    Load an RGB frame from the raw data session.
    
    The RGB data can be stored either as:
    1. Individual images in rgb/images/ folder (rare)
    2. Video file L515_rgb.mp4 (common) - needs frame extraction
    
    Args:
        session_path: Path to the session folder (e.g., .../2023_10_26/2023-10-26_15-34-07)
        frame_idx: Frame index
    
    Returns:
        numpy array of the RGB image (H, W, 3), or None if not found
    """
    from PIL import Image
    
    if not session_path:
        return None
    
    # First, try loading from individual image files
    possible_paths = [
        os.path.join(session_path, 'rgb', 'images', f'{frame_idx}.jpg'),
        os.path.join(session_path, 'rgb', 'images', f'{frame_idx}.png'),
        os.path.join(session_path, 'rgb', 'images', f'{frame_idx:06d}.jpg'),
        os.path.join(session_path, 'rgb', 'images', f'{frame_idx:06d}.png'),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                img = Image.open(path)
                return np.array(img)
            except Exception:
                continue
    
    # If individual images don't exist, try extracting from video file
    video_paths = [
        os.path.join(session_path, 'L515_rgb.mp4'),
        os.path.join(session_path, 'rgb.mp4'),
    ]
    
    for video_path in video_paths:
        if os.path.exists(video_path):
            try:
                import cv2
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    continue
                
                # Set frame position
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                cap.release()
                
                if ret and frame is not None:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    return frame_rgb
            except ImportError:
                # cv2 not available, try with imageio
                try:
                    import imageio
                    reader = imageio.get_reader(video_path)
                    try:
                        frame = reader.get_data(frame_idx)
                        reader.close()
                        return np.array(frame)
                    except:
                        reader.close()
                except:
                    pass
            except Exception:
                continue
    
    return None


def create_view_samples(result_dir: str, samples_data: list, output_format: str, verbose: bool = False):
    """
    Create a view_samples folder with visualizations of sample frames with labels.
    Now includes RGB frame counterparts from raw data when available.
    
    This function generates visualizations regardless of the output format used,
    allowing quick visual verification of the generated dataset.
    
    Args:
        result_dir: Output directory (where to create view_samples/)
        samples_data: List of tuples collected during processing:
                      (frame_data, labels, frame_info, session_path, rgb_frame_idx, raw_rgb_labels)
        output_format: Dataset format ('rvt', 'reyolov8', 'png', etc.)
        verbose: Print progress
    """
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from PIL import Image
    
    if not samples_data:
        if verbose:
            print("No samples collected for visualization")
        return
    
    view_samples_dir = os.path.join(result_dir, 'view_samples')
    os.makedirs(view_samples_dir, exist_ok=True)
    
    if verbose:
        print(f"\nCreating sample visualizations in: {view_samples_dir}")
    
    # Class colors for visualization
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for idx, sample in enumerate(samples_data[:3]):
        try:
            # Handle both old format (3 elements) and new format (6 elements)
            if len(sample) >= 6:
                frame_data, labels, frame_info, session_path, rgb_frame_idx, raw_rgb_labels = sample
            else:
                frame_data, labels, frame_info = sample[:3]
                session_path, rgb_frame_idx, raw_rgb_labels = None, None, None
            
            # Try to load RGB frame
            rgb_frame = load_rgb_frame(session_path, rgb_frame_idx) if session_path else None
            
            # Create figure with 1 or 2 subplots depending on RGB availability
            if rgb_frame is not None:
                fig, (ax_rgb, ax_event) = plt.subplots(1, 2, figsize=(20, 9))
            else:
                fig, ax_event = plt.subplots(1, 1, figsize=(12, 9))
                ax_rgb = None
            
            ax = ax_event  # For backward compatibility with the rest of the code
            
            # Handle different frame data formats
            if isinstance(frame_data, np.ndarray):
                if frame_data.ndim == 3:
                    # Detect format: (C, H, W) vs (H, W, C)
                    # If first dim is small (<=20 for event channels) and last dim is large, it's (C, H, W)
                    # If last dim is small (3 or 4), it's (H, W, C) - standard image format
                    if frame_data.shape[2] <= 4 and frame_data.shape[0] > 20:
                        # (H, W, C) format - standard RGB/RGBA image
                        height, width = frame_data.shape[0], frame_data.shape[1]
                        # Normalize to [0, 1] for display
                        if frame_data.max() > 1:
                            img = frame_data.astype(np.float32) / 255.0
                        else:
                            img = frame_data.astype(np.float32)
                        # Handle grayscale, RGB, RGBA
                        if img.shape[2] == 1:
                            ax.imshow(img[:, :, 0], cmap='gray')
                        elif img.shape[2] == 3:
                            ax.imshow(img)
                        elif img.shape[2] == 4:
                            ax.imshow(img[:, :, :3])  # Drop alpha
                        else:
                            ax.imshow(img[:, :, 0], cmap='gray')
                    else:
                        # (C, H, W) format - Multi-channel event representation
                        channels, height, width = frame_data.shape
                        
                        if channels >= 2:
                            # Create RGB visualization from event channels
                            # Positive polarity -> Red, Negative -> Blue
                            img = np.zeros((height, width, 3), dtype=np.float32)
                            
                            # Sum first half of channels for positive, second half for negative
                            half_c = channels // 2
                            if half_c > 0:
                                pos = frame_data[:half_c].sum(axis=0)
                                neg = frame_data[half_c:2*half_c].sum(axis=0) if channels > half_c else np.zeros_like(pos)
                            else:
                                pos = frame_data[0]
                                neg = frame_data[1] if channels > 1 else np.zeros_like(pos)
                            
                            # Normalize
                            if pos.max() > 0:
                                img[:, :, 0] = pos / pos.max()  # Red
                            if neg.max() > 0:
                                img[:, :, 2] = neg / neg.max()  # Blue
                            
                            ax.imshow(img)
                        else:
                            # Single channel - grayscale
                            ax.imshow(frame_data[0], cmap='gray')
                elif frame_data.ndim == 2:
                    # 2D grayscale image
                    ax.imshow(frame_data, cmap='gray')
                    height, width = frame_data.shape
                else:
                    ax.text(0.5, 0.5, f"Unsupported frame format: {frame_data.shape}", 
                           ha='center', va='center', transform=ax.transAxes)
                    height, width = 288, 384  # Default
            else:
                ax.text(0.5, 0.5, "Frame data not available", 
                       ha='center', va='center', transform=ax.transAxes)
                height, width = 288, 384  # Default
            
            # Draw bounding boxes
            num_boxes = 0
            if labels is not None and len(labels) > 0:
                for label in labels:
                    try:
                        # Handle different label formats
                        if isinstance(label, dict) and 'bbox' in label:
                            # COCO format: [x, y, w, h]
                            bbox = label['bbox']
                            class_id = label.get('category_id', 0) - 1
                            x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
                        elif isinstance(label, (list, np.ndarray)) and len(label) >= 5:
                            # YOLO format: [class, cx, cy, w, h, ...] (normalized)
                            class_id = int(label[0])
                            cx, cy, bw, bh = label[1], label[2], label[3], label[4]
                            # Denormalize
                            x = (cx - bw/2) * width
                            y = (cy - bh/2) * height
                            w = bw * width
                            h = bh * height
                        elif isinstance(label, (tuple, np.void)) and hasattr(label, '__len__'):
                            # RVT structured format: (t, x, y, w, h, class_id, ...)
                            if len(label) >= 6:
                                x, y, w, h = float(label[1]), float(label[2]), float(label[3]), float(label[4])
                                class_id = int(label[5])
                            else:
                                continue
                        else:
                            continue
                        
                        # Skip invalid boxes
                        if w <= 0 or h <= 0:
                            continue
                        
                        color = colors[class_id % len(colors)]
                        rect = patches.Rectangle(
                            (x, y), w, h,
                            linewidth=2, edgecolor=color, facecolor='none'
                        )
                        ax.add_patch(rect)
                        ax.text(x, y - 2, f'cls:{class_id}', color=color, fontsize=8,
                               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
                        num_boxes += 1
                    except Exception as e:
                        continue
            
            # Add title for event frame
            title = f"Event/Sensor Frame"
            if frame_info:
                title += f" | {frame_info}"
            title += f" | {num_boxes} boxes"
            ax.set_title(title, fontsize=10)
            ax.set_xlabel(f'Width: {width}')
            ax.set_ylabel(f'Height: {height}')
            
            # Draw RGB frame with labels if available
            if ax_rgb is not None and rgb_frame is not None:
                ax_rgb.imshow(rgb_frame)
                rgb_h, rgb_w = rgb_frame.shape[:2]
                
                # Draw RGB labels (in RGB camera space)
                rgb_num_boxes = 0
                if raw_rgb_labels is not None and len(raw_rgb_labels) > 0:
                    for label in raw_rgb_labels:
                        try:
                            if isinstance(label, dict) and 'bbox' in label:
                                bbox = label['bbox']
                                class_id = label.get('category_id', 0) - 1
                                x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
                                
                                if w > 0 and h > 0:
                                    color = colors[class_id % len(colors)]
                                    rect = patches.Rectangle(
                                        (x, y), w, h,
                                        linewidth=2, edgecolor=color, facecolor='none'
                                    )
                                    ax_rgb.add_patch(rect)
                                    ax_rgb.text(x, y - 2, f'cls:{class_id}', color=color, fontsize=8,
                                               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
                                    rgb_num_boxes += 1
                        except Exception:
                            continue
                
                ax_rgb.set_title(f"RGB Source Frame | {rgb_num_boxes} boxes ({rgb_w}x{rgb_h})", fontsize=10)
                ax_rgb.set_xlabel(f'Width: {rgb_w}')
                ax_rgb.set_ylabel(f'Height: {rgb_h}')
            
            # Add overall title
            fig.suptitle(f'Sample {idx + 1}: Event vs RGB Comparison', fontsize=12, fontweight='bold')
            
            # Save figure
            sample_path = os.path.join(view_samples_dir, f'sample_{idx + 1:02d}.png')
            plt.tight_layout()
            plt.savefig(sample_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            if verbose:
                rgb_info = f" + RGB" if rgb_frame is not None else ""
                print(f"  Saved: sample_{idx + 1:02d}.png ({num_boxes} boxes{rgb_info})")
                
        except Exception as e:
            if verbose:
                print(f"  Warning: Failed to create sample {idx + 1}: {e}")
            plt.close('all')
    
    # Create summary image with all samples
    try:
        n_samples = min(3, len(samples_data))
        if n_samples > 0:
            fig, axes = plt.subplots(1, n_samples, figsize=(6 * n_samples, 5))
            if n_samples == 1:
                axes = [axes]
            
            for idx, sample in enumerate(samples_data[:n_samples]):
                # Handle both old format (3 elements) and new format (6 elements)
                if len(sample) >= 6:
                    frame_data, labels, frame_info, _, _, _ = sample
                else:
                    frame_data, labels, frame_info = sample[:3]
                    
                ax = axes[idx]
                height, width = 288, 384  # Default
                
                if isinstance(frame_data, np.ndarray) and frame_data.ndim == 3:
                    # Detect format: (C, H, W) vs (H, W, C)
                    if frame_data.shape[2] <= 4 and frame_data.shape[0] > 20:
                        # (H, W, C) format - standard image
                        height, width = frame_data.shape[0], frame_data.shape[1]
                        if frame_data.max() > 1:
                            img = frame_data.astype(np.float32) / 255.0
                        else:
                            img = frame_data.astype(np.float32)
                        if img.shape[2] >= 3:
                            ax.imshow(img[:, :, :3])
                        else:
                            ax.imshow(img[:, :, 0], cmap='gray')
                    else:
                        # (C, H, W) format - event representation
                        channels, height, width = frame_data.shape
                        if channels >= 2:
                            img = np.zeros((height, width, 3), dtype=np.float32)
                            half_c = max(1, channels // 2)
                            pos = frame_data[:half_c].sum(axis=0)
                            neg = frame_data[half_c:2*half_c].sum(axis=0) if channels > half_c else np.zeros_like(pos)
                            if pos.max() > 0:
                                img[:, :, 0] = pos / pos.max()
                            if neg.max() > 0:
                                img[:, :, 2] = neg / neg.max()
                            ax.imshow(img)
                        else:
                            ax.imshow(frame_data[0], cmap='gray')
                elif isinstance(frame_data, np.ndarray) and frame_data.ndim == 2:
                    ax.imshow(frame_data, cmap='gray')
                    height, width = frame_data.shape
                
                # Draw boxes
                num_boxes = 0
                if labels is not None:
                    for label in labels:
                        try:
                            if isinstance(label, dict) and 'bbox' in label:
                                bbox = label['bbox']
                                x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
                            elif isinstance(label, (list, np.ndarray)) and len(label) >= 5:
                                cx, cy, bw, bh = label[1], label[2], label[3], label[4]
                                x = (cx - bw/2) * width
                                y = (cy - bh/2) * height
                                w, h = bw * width, bh * height
                            elif hasattr(label, '__len__') and len(label) >= 6:
                                x, y, w, h = float(label[1]), float(label[2]), float(label[3]), float(label[4])
                            else:
                                continue
                            if w > 0 and h > 0:
                                rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                                        edgecolor='lime', facecolor='none')
                                ax.add_patch(rect)
                                num_boxes += 1
                        except:
                            pass
                
                ax.set_title(f'Sample {idx + 1}: {num_boxes} boxes', fontsize=10)
                ax.axis('off')
            
            plt.suptitle(f'Dataset Samples Overview ({output_format} format)', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(view_samples_dir, 'samples_overview.png'), dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            if verbose:
                print(f"  Saved: samples_overview.png")
    except Exception as e:
        if verbose:
            print(f"  Warning: Failed to create overview: {e}")
        plt.close('all')


# Global list to collect samples during processing
_collected_samples = []

def collect_sample_for_visualization(frame_data, labels, frame_info: str = "", 
                                      session_path: str = None, rgb_frame_idx: int = None,
                                      raw_rgb_labels: list = None):
    """
    Collect a sample for later visualization.
    Only collects up to 3 samples.
    
    Args:
        frame_data: Frame data (numpy array or tensor)
        labels: Labels for this frame (various formats supported) - in sensor space
        frame_info: Optional string with frame information
        session_path: Path to raw session folder for loading RGB frames
        rgb_frame_idx: Index of the corresponding RGB frame
        raw_rgb_labels: Original RGB labels in RGB camera space (COCO format)
    """
    global _collected_samples
    if len(_collected_samples) < 3:
        # Make a copy to avoid reference issues
        if hasattr(frame_data, 'copy'):
            frame_copy = frame_data.copy()
        elif hasattr(frame_data, 'cpu'):
            frame_copy = frame_data.cpu().numpy().copy()
        else:
            frame_copy = frame_data
        
        # Copy labels
        if labels is not None:
            if isinstance(labels, list):
                labels_copy = [l.copy() if hasattr(l, 'copy') else l for l in labels]
            elif hasattr(labels, 'copy'):
                labels_copy = labels.copy()
            else:
                labels_copy = labels
        else:
            labels_copy = None
        
        # Copy raw RGB labels
        if raw_rgb_labels is not None:
            if isinstance(raw_rgb_labels, list):
                raw_labels_copy = [l.copy() if hasattr(l, 'copy') else dict(l) for l in raw_rgb_labels]
            else:
                raw_labels_copy = raw_rgb_labels
        else:
            raw_labels_copy = None
        
        _collected_samples.append((frame_copy, labels_copy, frame_info, session_path, rgb_frame_idx, raw_labels_copy))

def reset_collected_samples():
    """Reset the collected samples list."""
    global _collected_samples
    _collected_samples = []

def get_collected_samples():
    """Get the collected samples."""
    global _collected_samples
    return _collected_samples

# Suppress expected radar processing warnings (divide by zero in AoA computation)
# These occur when radar has no valid detections in a frame and are handled gracefully
warnings.filterwarnings('ignore', message='divide by zero encountered in divide')
warnings.filterwarnings('ignore', message='invalid value encountered in divide')



def dilate_sparse_radar(radar_frame, kernel_size=15):
    """
    Dilate sparse radar data to improve bbox/radar overlap.
    Uses distance-weighted dilation to preserve accurate distance values.
    
    Args:
        radar_frame: numpy array, shape (H, W), values 0-255 (normalized distances)
        kernel_size: int, dilation kernel size (larger = more spread)
        
    Returns:
        dilated_frame: numpy array, same shape as input, with dilated radar values
    """
    import cv2
    
    if np.count_nonzero(radar_frame) == 0:
        return radar_frame
    
    # Create binary mask of radar detections
    mask = (radar_frame > 0).astype(np.uint8)
    
    # Dilate the mask to expand coverage area
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    
    # Distance transform to find closest radar pixel for each dilated pixel
    dist_transform = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 5)
    
    # For each dilated pixel, find nearest radar value using interpolation
    # This preserves distance accuracy better than naive dilation
    dilated_frame = np.zeros_like(radar_frame, dtype=np.float32)
    
    # Use inpainting to fill dilated regions with nearby radar values
    # INPAINT_TELEA is good for sparse data
    radar_float = radar_frame.astype(np.float32)
    dilated_frame = cv2.inpaint(radar_float, (1 - mask), kernel_size // 2, cv2.INPAINT_TELEA)
    
    # Only keep values within the dilated mask
    dilated_frame = dilated_frame * dilated_mask
    
    return dilated_frame.astype(np.uint8)


def fuse_radar_with_representation(dvs_representation, radar_frame):
    """
    Fuse radar point cloud as additional channel to DVS event representation.
    Works with ANY event representation (VTEI, voxel_grid, shist, mdes) and ANY bin count.
    
    If the radar frame has different (H, W) from the DVS representation, it is
    resized with nearest-neighbor interpolation (preserving the sparse distance
    values) to match.
    
    Args:
        dvs_representation: numpy array, shape (C, H, W) where C depends on representation:
                           - VTEI/mdes: bins channels (e.g., 5, 10)
                           - voxel_grid/shist: 2*bins channels (e.g., 10, 20)
        radar_frame: numpy array, shape (H, W), values 0-255 (distance encoded as grayscale)
        
    Returns:
        fused_representation: numpy array, shape (C+1, H, W) with radar as last channel
    """
    import cv2
    
    radar_normalized = radar_frame.astype(np.float32) / 255.0
    
    if radar_normalized.ndim == 2:
        radar_channel = radar_normalized[np.newaxis, :, :]
    else:
        radar_channel = radar_normalized[0:1, :, :]
    
    target_h, target_w = dvs_representation.shape[1], dvs_representation.shape[2]
    radar_h, radar_w = radar_channel.shape[1], radar_channel.shape[2]
    
    if (radar_h, radar_w) != (target_h, target_w):
        resized = cv2.resize(
            radar_channel[0], (target_w, target_h),
            interpolation=cv2.INTER_NEAREST
        )
        radar_channel = resized[np.newaxis, :, :]
    
    fused = np.concatenate([dvs_representation, radar_channel], axis=0)
    
    return fused


def extract_bbox_distance_from_radar(bbox, radar_frame, width, height, max_dist):
    """
    Extract average distance from radar point cloud within a bounding box.
    Follows the same approach as YOLOv8/YOLOX for distance annotation generation.
    
    Args:
        bbox: list [class_id, center_x, center_y, box_w, box_h] (normalized [0, 1])
        radar_frame: numpy array, shape (H, W), values 0-255 (distance encoded as grayscale)
        width: int, image width in pixels
        height: int, image height in pixels
        max_dist: float, maximum radar distance in meters (for denormalization)
        
    Returns:
        average_distance: float, average distance in meters, or -1.0 if no radar points
    """
    # Convert normalized bbox to pixel coordinates
    center_x = bbox[1] * width
    center_y = bbox[2] * height
    box_w = bbox[3] * width
    box_h = bbox[4] * height
    
    x1 = int(max(0, center_x - box_w / 2))
    y1 = int(max(0, center_y - box_h / 2))
    x2 = int(min(width, center_x + box_w / 2))
    y2 = int(min(height, center_y + box_h / 2))
    
    # Ensure valid bbox
    if x2 <= x1 or y2 <= y1:
        return -1.0
    
    # Extract radar values in bbox region
    radar_crop = radar_frame[y1:y2, x1:x2]
    
    # Filter out zero values (no radar detection)
    valid_pixels = radar_crop[radar_crop > 0]
    
    if len(valid_pixels) > 0:
        # Convert pixel values (0-255) back to meters
        # Radar frame is normalized: pixel_value = distance_meters * 255 / max_dist
        distances_meters = valid_pixels.astype(np.float32) * max_dist / 255.0
        return float(np.mean(distances_meters))
    else:
        return -1.0  # No radar data available in this bbox



# This code is supposed to be used by a general user to generate custom, trainable datasets
# starting from the NERVE data sessions collected in winter 2023/24.
# Concrete code about how to handle supported types of data can be found in data_source.py

# It tries to be customizable and extinsible enough, with some trade-off. You are supposed to point which sessions to take data from,
# as well as a JSON setting file which contains indications about which type of data include, and some basic pre-processing to be performed over it.

# example of usage:
# python dataset_creator.py --list test_list.txt --settings example_source_settings_05.json -d /home/neuro-gpu/davis_dataset/test
# Using dataset settings in 'example_source_settings_05.json', generate a customized (trainable) dataset starting from the sessions archive. This dataset will contains only data from sessions listed in 'example_source_settings_05.json'


def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="PEGMA Dataset Generator - Creates datasets for YOLOX, YOLOv8, and ReYOLOv8 training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create train split for ReYOLOv8
  python dataset_creator.py -l train_sessions.txt -s templates/reyolov8_sequence.template.json -d /path/to/output --split train --clean

  # Add val split to existing dataset
  python dataset_creator.py -l val_sessions.txt -s templates/reyolov8_sequence.template.json -d /path/to/output --split val --add

  # Create test split and generate data.yaml
  python dataset_creator.py -l test_sessions.txt -s templates/reyolov8_sequence.template.json -d /path/to/output --split test --add
        """
    )
    
    parser.add_argument("--list", "-l", type=str, default="", help="Path of a .txt file where sessions to be included are listed. There must be one session path for each line of the file.")
    parser.add_argument("--single-session", "-ss", type=str, default="", help="Path of one single session.")

    parser.add_argument("--dataset", "-d", type=str, required=True, help="Path of the output dataset root directory.")
    parser.add_argument("--settings", "-s", type=str, required=True, help="Path of a .json file containing settings for the data sources to include.")
    parser.add_argument("--split", type=str, default="", choices=["train", "val", "test", ""], 
                       help="Dataset split name (train/val/test). If specified, creates {dataset}/{split}/ structure.")

    parser.add_argument("--add", "-a", action='store_true', help="Add new sessions to a previously created dataset.")
    parser.add_argument("--clean", action='store_true', help="Override existing data if the output path already exists.")
    parser.add_argument("--verbose", "-v", action='store_true', help="Enable verbose output with detailed progress information.")
    parser.add_argument("--no-yaml", action='store_true', help="Skip automatic data.yaml generation.")
    return parser.parse_args()


def calculate_initial_time_shifts(sources:dict):
    start_time_ms = {"rgb":0}
    to_be_processed = dict(sources)

    n_sources = len(to_be_processed)
    while n_sources > 0:
        for k in list(to_be_processed.keys()):
            mapping =  to_be_processed[k].mapping
            src, dst = mapping.src.name, mapping.dst.name
            if src in start_time_ms:
               assert not dst in start_time_ms
               start_time_ms[dst] = -start_time_ms[src] - to_be_processed[k].GetInitialDelay_ms()
               del to_be_processed[k]
            elif dst in start_time_ms:
               start_time_ms[src] = -start_time_ms[dst] - to_be_processed[k].GetInitialDelay_ms()
               del to_be_processed[k]
        
        assert len(to_be_processed) < n_sources, "Impossible to link together all time shifts.. Apparently, not all sources are connected together."
        n_sources = len(to_be_processed)

    start_time_ms = sorted(start_time_ms.items(), key=lambda x:x[1])
    max_delay_ms = start_time_ms[-1][1]
    return start_time_ms, max_delay_ms


def generate_yolo_txt_labels(coco_json_path, output_labels_dir, verbose=False):
    """
    Convert COCO JSON annotations to YOLO txt format labels.
    
    Adapted from original coco2yolo.py for PEGMA compatibility.
    
    YOLOv8/YOLOX require labels as .txt files with format:
    class_id x_center y_center width height (all normalized 0-1)
    
    Key behaviors (matching original coco2yolo.py):
    - Skips 'iscrowd' annotations
    - Skips invalid boxes (width <= 0 or height <= 0)
    - No clamping - preserves data integrity, errors surface during training
    - Uses %g format for compact output
    - Deletes and recreates output directory for clean state
    
    Args:
        coco_json_path: Path to COCO format JSON file
        output_labels_dir: Directory to write .txt label files
        verbose: Print progress
        
    Returns:
        Number of label files created
    """
    from pathlib import Path
    from collections import defaultdict
    from tqdm import tqdm
    import numpy as np
    
    coco_path = Path(coco_json_path)
    labels_dir = Path(output_labels_dir)
    
    if not coco_path.exists():
        if verbose:
            print(f"  ⚠️  COCO file not found: {coco_path}")
        return 0
    
    # Delete and recreate output directory for clean state (matching original behavior)
    if labels_dir.exists():
        shutil.rmtree(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Load COCO annotations with error handling for corrupted/empty files
    # (can happen if previous job crashed with OOM mid-write)
    try:
        with open(coco_path, 'r') as f:
            content = f.read()
            if not content.strip():
                if verbose:
                    print(f"  ⚠️  COCO JSON file is empty: {coco_path}")
                    print(f"     This usually means the previous job crashed (OOM) before completion.")
                    print(f"     Re-run the data generation for this split.")
                return 0
            coco = json.loads(content)
    except json.JSONDecodeError as e:
        if verbose:
            print(f"  ⚠️  Failed to parse COCO JSON: {coco_path}")
            print(f"     Error: {e}")
            print(f"     This usually means the previous job crashed (OOM) mid-write.")
            print(f"     Re-run the data generation for this split.")
        return 0
    
    # Build class ID mapping
    # PEGMA uses filtered classes with original COCO IDs preserved
    # Map sorted COCO IDs to 0-indexed YOLO class indices
    # E.g., if filter_classes=['person', 'car'] → COCO IDs [1, 3] → YOLO indices {1:0, 3:1}
    categories = {cat['id']: cat for cat in coco.get('categories', [])}
    sorted_cat_ids = sorted(categories.keys())
    coco_id_to_yolo_idx = {coco_id: idx for idx, coco_id in enumerate(sorted_cat_ids)}
    
    # Create image dict (matching original: uses string keys)
    images = {"%g" % x["id"]: x for x in coco.get("images", [])}
    
    # Create image-annotations dict (matching original)
    imgToAnns = defaultdict(list)
    for ann in coco.get("annotations", []):
        imgToAnns[ann["image_id"]].append(ann)
    
    # Track statistics
    created = 0
    skipped_crowd = 0
    skipped_invalid = 0
    
    # Write labels file (matching original loop structure)
    for img_id, anns in tqdm(imgToAnns.items(), desc=f"  Converting {coco_path.name}", disable=not verbose):
        img = images.get("%g" % img_id)
        if img is None:
            continue
            
        h, w, f = img["height"], img["width"], img["file_name"]
        
        bboxes = []
        for ann in anns:
            # Skip crowd annotations (matching original)
            if ann.get("iscrowd", False):
                skipped_crowd += 1
                continue
            
            # Skip categories not in our mapping
            cat_id = ann.get("category_id")
            if cat_id not in coco_id_to_yolo_idx:
                continue
            
            # The COCO box format is [top left x, top left y, width, height]
            box = np.array(ann["bbox"], dtype=np.float64)
            box[:2] += box[2:] / 2  # xy top-left corner to center
            box[[0, 2]] /= w  # normalize x
            box[[1, 3]] /= h  # normalize y
            
            # Skip invalid boxes (matching original - no clamping!)
            if box[2] <= 0 or box[3] <= 0:
                skipped_invalid += 1
                continue
            
            # Map COCO category ID to YOLO class index
            cls = coco_id_to_yolo_idx[cat_id]
            box = [cls] + box.tolist()
            
            # Avoid duplicates (matching original)
            if box not in bboxes:
                bboxes.append(box)
        
        # Write label file (matching original format with %g)
        f = os.path.basename(f)
        txt_path = (labels_dir / f).with_suffix(".txt")
        
        with open(txt_path, "a") as file:
            for i in range(len(bboxes)):
                line = (*(bboxes[i]),)  # cls, box
                file.write(("%g " * len(line)).rstrip() % line + "\n")
        
        if bboxes:
            created += 1
    
    if verbose:
        print(f"  Created {created} YOLO txt label files in {labels_dir}")
        if skipped_crowd > 0:
            print(f"  Skipped {skipped_crowd} crowd annotations")
        if skipped_invalid > 0:
            print(f"  Skipped {skipped_invalid} invalid boxes (w<=0 or h<=0)")
    
    return created


def generate_data_yaml(dataset_root, settings, verbose=False):
    """
    Auto-generate data.yaml for training after dataset creation.
    
    Works for all output formats:
    - YOLOX/YOLOv8: PNG images with COCO JSON annotations
    - ReYOLOv8: HDF5 sequences with .npy labels
    - RVT: HDF5 sequences with structured array labels (labels_v2/)
    
    Args:
        dataset_root: Root directory of the dataset (contains train/val/test subdirs)
        settings: Dataset generation settings (list of source configs)
        verbose: Print progress information
    """
    from pathlib import Path
    
    dataset_path = Path(dataset_root).resolve()
    
    # Determine the primary sensor source (not ti_radar)
    source = None
    for s in settings:
        if s['data'] != 'ti_radar':
            source = s['data']
            break
    
    if source is None:
        if verbose:
            print("⚠️  Could not determine sensor source, skipping data.yaml generation")
        return None
    
    # Determine output format
    output_format = settings[0].get('output_format', 'reyolov8') if settings else 'reyolov8'
    is_rvt = output_format == 'rvt'
    
    # Detect which splits exist
    splits = []
    for split_name in ['train', 'val', 'test']:
        if is_rvt:
            # RVT format: sequences directly under split dir (train/sequence_NNNNNN/)
            split_dir = dataset_path / split_name
            if split_dir.exists():
                # Check for sequence directories
                seqs = [d for d in split_dir.iterdir() if d.is_dir() and d.name.startswith('sequence_')]
                if seqs:
                    splits.append(split_name)
        else:
            # Standard format: split/data/source/
            split_data_dir = dataset_path / split_name / 'data' / source
            if split_data_dir.exists() and any(split_data_dir.iterdir()):
                splits.append(split_name)
    
    if not splits and not is_rvt:
        # Try direct structure (no split subdirs)
        data_dir = dataset_path / 'data' / source
        if data_dir.exists():
            splits = ['train']  # Treat as single split
    
    if not splits:
        if verbose:
            print("⚠️  No data found, skipping data.yaml generation")
        return None
    
    # Determine output format and channels
    clip_mode = settings[0].get('clip_mode', 'single_frame')
    is_hdf5 = settings[0].get('store_as_hdf5', False) or is_rvt
    bins = settings[0].get('bins', 10)
    event_rep = settings[0].get('event_representation', None)
    
    # Check for radar (adds +1 channel)
    has_radar = any(s['data'] == 'ti_radar' for s in settings)
    
    # Calculate channels
    if event_rep in ['vtei', 'mdes', 'ev_temporal_volume']:
        channels = bins
    elif event_rep in ['voxel_grid', 'shist']:
        channels = 2 * bins
    else:
        channels = 3  # Default PNG
    
    if has_radar and is_hdf5:
        channels += 1  # Radar fusion adds 1 channel
    
    # Get distance settings if radar is present
    max_dist = 10.0
    min_dist = 0.0
    for s in settings:
        if s['data'] == 'ti_radar':
            max_dist = s.get('max_dist', 10.0)
            break
    
    # Get class names from config or auto-detect from COCO annotations
    # COCO annotations keep original COCO IDs (person=1, car=3, etc.)
    # YOLO uses 0-indexed class indices based on position in filtered category list
    explicit_classes = settings[0].get('filter_classes', None)  # New: class names to filter
    
    if explicit_classes:
        # Use explicit class list from config
        nc = len(explicit_classes)
        names = explicit_classes
        if verbose:
            print(f"  Using classes from filter_classes config: {names}")
    else:
        # Auto-detect from COCO annotations
        # Categories keep original COCO IDs - we just need the names in sorted order by ID
        nc = 1
        names = ['person']
        
        for split in splits:
            ann_path = dataset_path / split / 'annotations' / f'{source}.json'
            if ann_path.exists():
                try:
                    with open(ann_path, 'r') as f:
                        coco = json.load(f)
                    if 'categories' in coco:
                        categories = coco['categories']
                        nc = len(categories)
                        # Sort by original COCO ID to get consistent ordering
                        # YOLO index = position in this sorted list (0-indexed)
                        sorted_cats = sorted(categories, key=lambda x: x['id'])
                        names = [cat['name'] for cat in sorted_cats]
                        if verbose:
                            print(f"  Auto-detected {nc} class(es) from COCO annotations: {names}")
                            print(f"  (Original COCO IDs: {[cat['id'] for cat in sorted_cats]})")
                        break
                except Exception:
                    pass
    
    # For PNG datasets (YOLOX/YOLOv8), generate YOLO txt labels from COCO annotations
    # Also create 'images' symlink for YOLOv8 compatibility (it looks for 'images' -> 'labels' mapping)
    if not is_hdf5:
        if verbose:
            print("\n📝 Generating YOLO txt labels for YOLOv8/YOLOX training...")
        
        for split in splits:
            coco_path = dataset_path / split / 'annotations' / f'{source}.json'
            labels_dir = dataset_path / split / 'labels' / source
            
            if coco_path.exists():
                generate_yolo_txt_labels(coco_path, labels_dir, verbose=verbose)
            
            # Create 'images' symlink pointing to 'data' for YOLOv8 compatibility
            # YOLOv8 replaces 'images' with 'labels' in paths, so this makes it work
            images_link = dataset_path / split / 'images'
            data_dir = dataset_path / split / 'data'
            if data_dir.exists() and not images_link.exists():
                try:
                    images_link.symlink_to('data')
                    if verbose:
                        print(f"  Created symlink: {split}/images -> data")
                except OSError:
                    pass  # Symlink creation may fail on some filesystems
    
    # Generate YAML content
    yaml_lines = [
        "# PEGMA Dataset Configuration",
        f"# Auto-generated for {source.upper()} sensor",
        f"# Output format: {'HDF5 sequences' if is_hdf5 else 'PNG images with YOLO txt labels'}",
        "",
        f"path: {dataset_path}",
    ]
    
    # Add split paths
    # For YOLOv8: use 'images' (symlink to 'data') so label auto-discovery works
    # For ReYOLOv8/HDF5: use 'data' directly
    for split in splits:
        if is_hdf5:
            yaml_lines.append(f"{split}: {split}/data/{source}")
        else:
            yaml_lines.append(f"{split}: {split}/images/{source}")
    
    yaml_lines.append("")
    
    # Add label paths
    for split in splits:
        if is_hdf5:
            yaml_lines.append(f"{split}_labels: {split}/labels/{source}")
        else:
            # YOLOv8 auto-discovers labels by replacing 'data' with 'labels' in path
            # But we also keep the COCO annotation reference for compatibility
            yaml_lines.append(f"{split}_ann: {split}/annotations/{source}.json")
    
    yaml_lines.extend([
        "",
        "# Dataset classes",
        f"nc: {nc}",
        "names:",
    ])
    
    for i, name in enumerate(names):
        yaml_lines.append(f"  {i}: {name}")
    
    # Add format-specific parameters
    if is_hdf5:
        clip_length = settings[0].get('clip_length', 11)
        clip_stride = settings[0].get('clip_stride', 11)
        sequence_length = settings[0].get('sequence_length', 11)
        frame_period_ms = settings[0].get('frame_period_ms', 50.0)
        
        if is_rvt:
            # RVT-specific parameters
            ev_repr_name = f"{event_rep}_dt={int(frame_period_ms)}_nbins={bins}"
            # output_shape is [W, H], but resolution_hw expects [H, W]
            output_shape = settings[0].get('output_shape', [304, 240])
            resolution_hw = [output_shape[1], output_shape[0]]
            yaml_lines.extend([
                "",
                "# RVT model parameters",
                f"channels: {channels}",
                f"sequence_length: {sequence_length}",
                f"ev_repr_name: '{ev_repr_name}'",
                f"resolution_hw: {resolution_hw}",
                "downsample_by_factor_2: false",
                "only_load_end_labels: false",
            ])
        else:
            # ReYOLOv8-specific parameters
            yaml_lines.extend([
                "",
                "# ReYOLOv8 sequence parameters",
                f"channels: {channels}",
                f"clip_length: {clip_length}",
                f"clip_stride: {clip_stride}",
            ])
        
        if has_radar:
            yaml_lines.extend([
                "",
                "# Distance estimation",
                "has_distance: true",
                f"min_dist: {min_dist}",
                f"max_dist: {max_dist}",
            ])
    
    # Write to file
    yaml_path = dataset_path / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write('\n'.join(yaml_lines) + '\n')
    
    if verbose:
        print(f"\n✓ Generated data.yaml: {yaml_path}")
        print(f"  Sensor: {source}")
        print(f"  Splits: {splits}")
        print(f"  Classes: {nc} ({', '.join(names)})")
        if is_hdf5:
            print(f"  Channels: {channels}")
            if has_radar:
                print(f"  Distance: enabled (range [{min_dist}, {max_dist}]m)")
    
    return yaml_path


def transform_annotation_common(annotation:dict) -> dict:
    # removing things in which we are not interested
    if 'segmentation' in annotation:
        del annotation['segmentation']
    if 'keypoints' in annotation:
        del annotation['keypoints']
    if 'parts' in annotation:
        del annotation['parts']
    if 'distance_points' in annotation:
        del annotation['distance_points']

    bbox = annotation['bbox']
    annotation['area'] = bbox[2] * bbox[3]
    return annotation

def transform_annotation_dvs(annotation:dict, source:DVS_Source) -> dict:
    """
    Transform DVS annotations to match stored image dimensions.
    
    MapLabels() in label_writer.py already transforms annotations from RGB camera 
    space to DAVIS/DVS camera space (mapping.dst) using proper perspective projection.
    The output coordinates are in the native sensor resolution.
    
    This function only needs to handle:
    - avg_pool_kernel: divides coordinates if pooling is applied
    - padding: NO transformation needed (padding adds to edges, doesn't scale)
    
    This matches the original PURE implementation behavior.
    """
    bbox = annotation['bbox']
    
    # Only apply pooling downscale factor
    if source.avg_pool_kernel != 1:
        bbox = [v / source.avg_pool_kernel for v in bbox]
        annotation['bbox'] = bbox
    
    # Note: Padding does NOT require coordinate transformation!
    # Padding adds zeros to the right/bottom edges, so coordinates remain valid.
    
    return transform_annotation_common(annotation)

def transform_annotation_radar(annotation:dict, source:Radar_source) -> dict:
    bbox = annotation['bbox']
    bbox = [bbox[0]*source.resize_width_ratio, bbox[1]*source.resize_height_ratio, bbox[2]*source.resize_width_ratio, bbox[3]*source.resize_height_ratio]
    annotation['bbox'] = bbox
    return transform_annotation_common(annotation)

def extract_from_single_session(session_path:str, settings:list, label_writers:dict, result_dir:str, current_index:int, end_time_to_be_skipped_ms=5000, verbose=False, clip_mode='sequence', clip_length=1, clip_stride=1, filter_class_ids=None):
    """
    Extract data from a single session.
    
    SUPPORTS THREE MODES:
    - 'sequence': Store ENTIRE session as one sequence (like original ReYOLOv8 scripts)
                  EventVideoDataset creates clips at training time. RECOMMENDED!
    - 'clip_based': Pre-create fixed-length clips (less flexible)
    - 'single_frame': Store individual frames (for non-ReYOLOv8 uses)
    
    Args:
        session_path: Path to session directory
        settings: List of source settings
        label_writers: Dictionary of label writers per source
        result_dir: Output directory
        current_index: Starting index for sequences/clips
        end_time_to_be_skipped_ms: Time to skip at end of recording
        verbose: Verbose output
        clip_mode: 'sequence' (recommended), 'clip_based', or 'single_frame'
        clip_length: Frames per clip (only for 'clip_based' mode)
        clip_stride: Stride between clips (only for 'clip_based' mode)
        filter_class_ids: List of COCO category IDs to include (None = all classes)
    """
    assert os.path.isdir(result_dir)
    
    if verbose:
        print("Starting to analyze session " + session_path)
        print(f"Mode: {clip_mode}")
        if clip_mode == 'clip_based':
            print(f"  Clip length: {clip_length}, stride: {clip_stride}")
        elif clip_mode == 'sequence':
            print(f"  Storing entire session as one sequence (like original ReYOLOv8 scripts)")

    timing_file = os.path.join(session_path, 'timings.json')

    sources = {}
    for el in settings:
        el['timings'] = timing_file

        if el['data'] == 'prophesee':
            el['data_path'] = os.path.join(session_path, "prophesee", "events.hdf5")
            sources['prophesee'] = Prophesee_source(el, transform_annotation_dvs, verbose)
        elif el['data'] == 'davis':
            el['data_path'] = os.path.join(session_path, "davis", "events.hdf5")
            sources['davis'] = DAVIS_source(el, transform_annotation_dvs, verbose)
        elif el['data'] == 'ti_radar':
            el['data_path'] = os.path.join(session_path, "ti_radar")
            sources['ti_radar'] = Radar_source(el, transform_annotation_radar)
        elif el['data'] == 'infineon_radar':
            el['data_path'] = os.path.join(session_path, "infineon_radar")
            sources['infineon_radar'] = Radar_source(el, transform_annotation_radar)
        else:
            print("I don't know how to handle setting {} ..".format(el))
            raise Exception
        
        # Determine output format - for RVT, don't create source subdirectories
        output_format = settings[0].get('output_format', 'reyolov8') if settings else 'reyolov8'
        is_rvt_format = output_format == 'rvt'
        
        # Only create source subdirectories for non-RVT formats
        # RVT expects sequences directly under split dir (train/sequence_NNNNNN/)
        if not is_rvt_format:
            data_path = os.path.join(result_dir, el['data'])
            if not os.path.isdir(data_path):
                os.mkdir(data_path)

    # Let's store all output spaces we want to have --> one for each different output stream
    labels_to_output_mappings = {}
    for s in sources:
        mapping= sources[s].mapping
        if mapping.src.name == 'rgb':
            labels_to_output_mappings[mapping.dst.name] = mapping
        elif mapping.dst.name == 'rgb':
            labels_to_output_mappings['rgb'] = None

    if verbose:
        for s in sources:
            print("Stream {} has {} mS of delay respect stream {}..".format(sources[s].mapping.dst.name, sources[s].GetInitialDelay_ms(), sources[s].mapping.src.name))

    # Align streams
    time_shifts, max_delay = calculate_initial_time_shifts(sources)

    # Add RGB annotations as source
    rgb_settings = {
        'data':'rgb',
        'data_path':os.path.join(session_path, 'rgb', 'annotations','annotations.json'),
    }
    # Apply class filtering if specified
    if filter_class_ids is not None:
        rgb_settings['only_classes'] = filter_class_ids
    sources['rgb'] = RGB_annotations_source(rgb_settings, verbose=verbose)

    minimum_duration_ms = -1
    for (s,t) in time_shifts:
        time_gap_ms = max_delay - t
        sources[s].Time_forward_ms(time_gap_ms, return_frame=False)
        stream_duration = (sources[s].TotalNumberOfFrames() * sources[s].GetFramePeriod_ms()) - time_gap_ms
        if minimum_duration_ms < 0 or stream_duration < minimum_duration_ms:
            minimum_duration_ms = stream_duration

    maximum_frame_period_ms = 0
    for s in sources:
        frame_period_ms = sources[s].GetFramePeriod_ms()
        if frame_period_ms > maximum_frame_period_ms:
            maximum_frame_period_ms = frame_period_ms
    
    step_time_ms = maximum_frame_period_ms
    total_steps = int((minimum_duration_ms - end_time_to_be_skipped_ms) // step_time_ms)
    
    if verbose:
        print("extracting {} frames @ {:.2f} Hz. Total time coverage: {:.2f} seconds".format(
            total_steps, 1e3/step_time_ms, (minimum_duration_ms - end_time_to_be_skipped_ms)/1e3))


    if clip_mode == 'sequence':
        # Detect if radar is present for fusion and distance extraction
        has_radar = 'ti_radar' in sources
        radar_max_dist = 10.0  # Default
        radar_dilation = 15  # Default kernel size
        
        # Detect output format (default: reyolov8, alternative: rvt)
        output_format = settings[0].get('output_format', 'reyolov8') if settings else 'reyolov8'
        ev_repr_name = settings[0].get('event_representation', 'stacked_histogram')
        bins = settings[0].get('bins', 10)
        frame_period_ms = settings[0].get('frame_period_ms', 50.0)
        fuse_radar_channel = True
        
        # Build RVT-style representation name
        if output_format == 'rvt':
            ev_repr_name = f"{ev_repr_name}_dt={int(frame_period_ms)}_nbins={bins}"
        
        if has_radar:
            # Extract radar settings
            for el in settings:
                if el['data'] == 'ti_radar':
                    radar_max_dist = el.get('max_dist', 10.0)
                    radar_dilation = el.get('radar_dilation', 15)  # 0 = disabled
                    fuse_radar_channel = el.get('fuse_radar_channel', True)
                    break
            if verbose:
                print(f"  Radar fusion enabled: max_dist = {radar_max_dist}m")
                if radar_dilation > 0:
                    print(f"  Radar dilation: kernel_size = {radar_dilation} (increases bbox/radar overlap)")
                else:
                    print(f"  Radar dilation: DISABLED (sparse radar, expect low distance coverage)")
        
        if verbose and output_format == 'rvt':
            print(f"  Output format: RVT (event_representations_v2/{ev_repr_name})")
        
        # Accumulate entire session in memory
        session_buffers = {s: [] for s in sources if s != 'rgb' and s != 'ti_radar'}
        label_buffers = {s: [] for s in sources if s != 'rgb' and s != 'ti_radar'}
        radar_buffers = {s: [] for s in session_buffers} if has_radar else {}
        
        session_name = os.path.basename(session_path)
        pbar = tqdm(total=total_steps, desc=f'  Processing ({session_name})', 
                   leave=True, position=1, disable=verbose)
        
        for i in range(total_steps):
            generated_data = {}
            for s in sources:
                move_forward(s, sources, generated_data, step_time_ms)

            if len(generated_data) < len(sources):
                if verbose:
                    print(f"  Session ended early at frame {i}/{total_steps}")
                break

            # Accumulate frames (DVS + labels + radar)
            for s in generated_data:
                if s != 'rgb' and s != 'ti_radar':
                    session_buffers[s].append(generated_data[s])
                    label_buffers[s].append(generated_data['rgb'])
                    
                    # Store corresponding radar frame for fusion
                    if has_radar:
                        if 'ti_radar' in generated_data:
                            radar_data = generated_data['ti_radar']
                            radar_pc_frame = sources['ti_radar'].create_frame_from_points(radar_data[2])
                            radar_pc_frame = (radar_pc_frame * 255.0 / radar_max_dist).clip(0, 255).astype(np.uint8)
                            
                            if radar_dilation > 0:
                                radar_pc_frame = dilate_sparse_radar(radar_pc_frame, kernel_size=radar_dilation)
                            
                            radar_buffers[s].append(radar_pc_frame)
                        else:
                            radar_buffers[s].append(None)
            
            pbar.update(1)
            del generated_data
        
        pbar.close()
        
        # Store the complete session as one sequence
        for s in session_buffers:
            if len(session_buffers[s]) > 0:
                mapping = labels_to_output_mappings[sources[s].mapping.dst.name]
                radar_frames_for_source = radar_buffers.get(s, None) if has_radar else None
                
                if output_format == 'rvt':
                    store_rvt_sequence_data(
                        s, sources[s], session_buffers[s], label_buffers[s],
                        result_dir, current_index, mapping, label_writers[s], verbose,
                        radar_frames=radar_frames_for_source, radar_max_dist=radar_max_dist,
                        radar_dilation=radar_dilation, filter_class_ids=filter_class_ids,
                        ev_repr_name=ev_repr_name, frame_period_ms=frame_period_ms,
                        fuse_radar_channel=fuse_radar_channel
                    )
                else:
                    store_sequence_data(
                        s, sources[s], session_buffers[s], label_buffers[s],
                        result_dir, current_index, mapping, label_writers[s], verbose,
                        radar_frames=radar_frames_for_source, radar_max_dist=radar_max_dist,
                        radar_dilation=radar_dilation, filter_class_ids=filter_class_ids
                    )
        
        current_index += 1
        
        del session_buffers
        del label_buffers
        if has_radar:
            del radar_buffers
    

    elif clip_mode == 'clip_based':
        clip_buffers = {s: [] for s in sources if s != 'rgb'}
        label_buffers = {s: [] for s in sources if s != 'rgb'}
        frame_count = 0

        session_name = os.path.basename(session_path)
        pbar = tqdm(total=total_steps, desc=f'  Processing clips ({session_name})', 
                   leave=True, position=1, disable=verbose)
        
        for i in range(total_steps):
            generated_data = {}
            for s in sources:
                move_forward(s, sources, generated_data, step_time_ms)

            if len(generated_data) < len(sources):
                break

            for s in generated_data:
                if s != 'rgb':
                    clip_buffers[s].append(generated_data[s])
                    label_buffers[s].append(generated_data['rgb'])
            
            frame_count += 1
            
            if frame_count >= clip_length and (frame_count - clip_length) % clip_stride == 0:
                for s in clip_buffers:
                    if len(clip_buffers[s]) >= clip_length:
                        clip_data = clip_buffers[s][-clip_length:]
                        clip_labels = label_buffers[s][-clip_length:]
                        
                        mapping = labels_to_output_mappings[sources[s].mapping.dst.name]
                        store_sequence_data(s, sources[s], clip_data, clip_labels, result_dir,
                                           current_index, mapping, label_writers[s], verbose,
                                           filter_class_ids=filter_class_ids)
                        
                        keep_size = clip_length + min(clip_stride, clip_length // 2)
                        if len(clip_buffers[s]) > keep_size:
                            clip_buffers[s] = clip_buffers[s][-keep_size:]
                            label_buffers[s] = label_buffers[s][-keep_size:]
                        
                        current_index += 1
            
            if frame_count % 100 == 0:
                gc.collect()
            
            pbar.update(1)
            del generated_data
        
        pbar.close()
        del clip_buffers
        del label_buffers
    

    else:
        # Check for DVS-Radar fusion settings
        fuse_dvs_radar = False
        fused_modality_name = None
        fusion_dilation = 5
        dvs_source_key = None
        
        for s_settings in settings:
            if s_settings.get('fuse_dvs_radar_png', False):
                fuse_dvs_radar = True
                fused_modality_name = s_settings.get('fused_modality_name', None)
                fusion_dilation = s_settings.get('fusion_dilation', 5)
                dvs_source_key = s_settings['data']  # This is the DVS source to fuse
                break
        
        # Determine actual DVS key and set default fused name
        if fuse_dvs_radar and dvs_source_key and 'ti_radar' in sources:
            if fused_modality_name is None:
                fused_modality_name = f"{dvs_source_key}_radar"
            
            # Create fused modality directory and label writer
            fused_data_path = os.path.join(result_dir, fused_modality_name)
            os.makedirs(fused_data_path, exist_ok=True)
            
            if verbose:
                print(f"  DVS-Radar PNG fusion enabled: {dvs_source_key} + ti_radar → {fused_modality_name}")
                print(f"  Fusion dilation kernel: {fusion_dilation}")
        else:
            fuse_dvs_radar = False  # Disable if ti_radar not present
        
        session_name = os.path.basename(session_path)
        pbar = tqdm(total=total_steps, desc=f'  Processing frames ({session_name})', 
                   leave=True, position=1, disable=verbose)
        
        for i in range(total_steps):
            generated_data = {}
            for s in sources:
                move_forward(s, sources, generated_data, step_time_ms)

            if len(generated_data) < len(sources):
                break

            # Store individual source data
            for s in generated_data:
                if s != 'rgb':
                    mapping = labels_to_output_mappings[sources[s].mapping.dst.name]
                    store_data(s, sources[s], generated_data, result_dir, 
                              current_index, mapping, label_writers[s],
                              session_path=session_path, frame_counter=i)
            
            # Create fused DVS+Radar image if enabled
            if fuse_dvs_radar and dvs_source_key in generated_data and 'ti_radar' in generated_data:
                try:
                    fused_path, fused_h, fused_w, fused_rel_path = store_fused_dvs_radar_data(
                        dvs_key=dvs_source_key,
                        dvs_source=sources[dvs_source_key],
                        radar_source=sources['ti_radar'],
                        dvs_data=generated_data[dvs_source_key],
                        radar_data=generated_data['ti_radar'],
                        result_dir=result_dir,
                        current_idx=current_index,
                        mapping=labels_to_output_mappings[sources[dvs_source_key].mapping.dst.name],
                        label_writer=label_writers[fused_modality_name] if fused_modality_name in label_writers else None,
                        fused_modality_name=fused_modality_name,
                        fusion_dilation=fusion_dilation
                    )
                    
                    # Add labels for the fused modality (same as DVS since same viewpoint)
                    if fused_modality_name in label_writers:
                        dvs_mapping = labels_to_output_mappings[sources[dvs_source_key].mapping.dst.name]
                        label_writers[fused_modality_name].add_data(
                            generated_data['rgb'], dvs_mapping, fused_rel_path,
                            (fused_h, fused_w), current_index, sources[dvs_source_key].Transform_annotation
                        )
                except Exception as e:
                    if verbose:
                        print(f"  Warning: DVS-Radar fusion failed for frame {current_index}: {e}")

            current_index += 1
            pbar.update(1)
            del generated_data
        
        pbar.close()

    for s in sources:
        sources[s].Close()
    
    del sources
    gc.collect()

    return current_index


def move_forward(key:str, sources:dict, results:dict, time:int):
    try:
        results[key] = sources[key].Time_forward_ms(time)
    except StopIteration:
        return

def store_data(key:str, source:DataSource, generated_data:dict, result_dir:str, current_idx:int, mapping:Mapping, label_writer:LabelWriter, session_path:str=None, frame_counter:int=None):
    base_path = os.path.join(result_dir, key)
    full_data_path, height, width = source.StoreData(generated_data[key], base_path, current_idx)
    data_path = full_data_path.replace(base_path, key)  # Relative path for annotations
    label_writer.add_data(generated_data['rgb'], mapping, data_path, (height, width), current_idx, source.Transform_annotation)
    
    # Collect sample for visualization (PNG format)
    # CRITICAL: Labels must be mapped from RGB space to sensor space before visualization
    if len(get_collected_samples()) < 3:
        raw_labels = generated_data.get('rgb')
        if raw_labels is not None and len(raw_labels) > 0:
            import torch as th
            from PIL import Image
            device = th.device("cpu")
            
            try:
                # Ensure all annotations have avg_distance (required by MapLabels)
                labels_with_distance = []
                for ann in raw_labels:
                    ann_copy = ann.copy() if hasattr(ann, 'copy') else dict(ann)
                    if 'avg_distance' not in ann_copy:
                        ann_copy['avg_distance'] = 5.0  # Default distance for visualization
                    labels_with_distance.append(ann_copy)
                
                # Map labels from RGB camera space to sensor space (like label_writer.add_data does)
                mapped_labels = MapLabels(labels_with_distance, mapping, None, device)
                
                # Apply transform_annotation for pooling/padding adjustments
                transformed_labels = []
                for ann in mapped_labels:
                    if 'bbox' not in ann:
                        continue
                    transformed_ann = source.Transform_annotation(ann)
                    bbox = transformed_ann.get('bbox', [0, 0, 0, 0])
                    if bbox[2] > 0 and bbox[3] > 0:  # Valid width and height
                        transformed_labels.append(transformed_ann)
                
                if transformed_labels:
                    # Load the saved PNG file for visualization (already properly processed)
                    if os.path.exists(full_data_path) and full_data_path.endswith('.png'):
                        img = Image.open(full_data_path)
                        frame_data = np.array(img)  # (H, W, C) format
                    else:
                        # Fallback: try to get raw frame data
                        frame_data = generated_data[key]
                        if isinstance(frame_data, tuple):
                            frame_data = frame_data[0]
                        if hasattr(frame_data, 'cpu'):
                            frame_data = frame_data.cpu().numpy()
                        elif hasattr(frame_data, 'numpy'):
                            frame_data = frame_data.numpy()
                        if isinstance(frame_data, np.ndarray):
                            frame_data = frame_data.copy()
                    
                    # Get the actual RGB frame index from the annotation's image_id
                    # This is more reliable than using frame_counter which may not match
                    actual_rgb_frame_idx = None
                    if raw_labels and len(raw_labels) > 0:
                        first_ann = raw_labels[0]
                        if isinstance(first_ann, dict) and 'image_id' in first_ann:
                            actual_rgb_frame_idx = first_ann['image_id']
                    
                    # Fall back to frame_counter if image_id not available
                    if actual_rgb_frame_idx is None:
                        actual_rgb_frame_idx = frame_counter
                    
                    collect_sample_for_visualization(
                        frame_data,
                        transformed_labels,
                        f"PNG {key} frame {current_idx} ({height}x{width})",
                        session_path=session_path,
                        rgb_frame_idx=actual_rgb_frame_idx,
                        raw_rgb_labels=raw_labels
                    )
            except Exception as e:
                # Don't let visualization errors break dataset generation
                pass



def merge_dvs_radar_images(dvs_image, radar_image, fusion_dilation=5):
    """
    Merge DVS histogram and radar point cloud into a single 3-channel image.
    
    Takes 2 channels from DVS (polarities) and replaces the 3rd channel with radar.
    This matches the original dataset_finalizer.py merge behavior.
    
    Args:
        dvs_image: numpy array (H, W, 3) - DVS histogram image (BGR from cv2)
        radar_image: numpy array (H, W, 3) or (H, W) - Radar point cloud image
        fusion_dilation: int - Kernel size for dilating radar before fusion (default 5)
        
    Returns:
        fused_image: numpy array (H, W, 3) - Fused image with radar in 3rd channel
    """
    import cv2
    
    # Ensure dvs_image is 3-channel
    if dvs_image.ndim == 2:
        dvs_image = cv2.cvtColor(dvs_image, cv2.COLOR_GRAY2BGR)
    
    # Ensure radar_image is properly formatted
    if radar_image.ndim == 3:
        radar_channel = radar_image[:, :, 0]  # Take first channel
    else:
        radar_channel = radar_image
    
    # Resize radar to match DVS if needed
    if radar_channel.shape[:2] != dvs_image.shape[:2]:
        radar_channel = cv2.resize(radar_channel, (dvs_image.shape[1], dvs_image.shape[0]))
    
    # Apply dilation to sparse radar data (helps preserve during any downscaling)
    if fusion_dilation > 0:
        kernel = np.ones((fusion_dilation, fusion_dilation), np.uint8)
        radar_channel = cv2.dilate(radar_channel, kernel)
    
    # Create fused image: replace 3rd channel (blue in BGR) with radar
    fused_image = dvs_image.copy()
    fused_image[:, :, -1] = radar_channel  # Replace last channel with radar
    
    return fused_image


def store_fused_dvs_radar_data(dvs_key: str, dvs_source: DataSource, radar_source: DataSource,
                               dvs_data, radar_data, result_dir: str, current_idx: int,
                               mapping: Mapping, label_writer: LabelWriter,
                               fused_modality_name: str = None, fusion_dilation: int = 5):
    """
    Store fused DVS+Radar PNG image as a new modality.
    
    This creates a 3-channel image combining:
    - 2 channels from DVS histogram (polarities)
    - 1 channel from radar point cloud
    
    Args:
        dvs_key: Source key for DVS (e.g., 'davis', 'prophesee')
        dvs_source: DVS DataSource instance
        radar_source: Radar DataSource instance
        dvs_data: DVS frame data tuple
        radar_data: Radar frame data tuple
        result_dir: Output directory
        current_idx: Current frame index
        mapping: Coordinate mapping for labels
        label_writer: LabelWriter for the fused modality
        fused_modality_name: Name for the fused modality (default: '{dvs_key}_radar')
        fusion_dilation: Kernel size for radar dilation before fusion
        
    Returns:
        tuple: (data_path, height, width)
    """
    import cv2
    
    # Determine fused modality name
    if fused_modality_name is None:
        fused_modality_name = f"{dvs_key}_radar"
    
    # Create output directory
    fused_path = os.path.join(result_dir, fused_modality_name)
    if not os.path.isdir(fused_path):
        os.makedirs(fused_path, exist_ok=True)
    
    # First, store DVS and radar individually to get their PNG paths
    # We'll read them back and merge (matches original workflow)
    dvs_base_path = os.path.join(result_dir, dvs_key)
    dvs_data_path, height, width = dvs_source.StoreData(dvs_data, dvs_base_path, current_idx)
    
    radar_base_path = os.path.join(result_dir, 'ti_radar')
    radar_data_path, _, _ = radar_source.StoreData(radar_data, radar_base_path, current_idx)
    
    # Read the saved images
    dvs_img = cv2.imread(dvs_data_path)
    radar_img = cv2.imread(radar_data_path)
    
    if dvs_img is None:
        raise ValueError(f"Could not read DVS image: {dvs_data_path}")
    if radar_img is None:
        raise ValueError(f"Could not read radar image: {radar_data_path}")
    
    # Merge images
    fused_img = merge_dvs_radar_images(dvs_img, radar_img, fusion_dilation)
    
    # Save fused image with consistent naming
    # Extract the filename pattern from DVS
    dvs_filename = os.path.basename(dvs_data_path)
    # Replace any DVS-specific prefix with fused prefix
    if 'histogram_img__' in dvs_filename:
        fused_filename = dvs_filename  # Keep same naming for compatibility
    else:
        fused_filename = f"fused_img__{str(current_idx).zfill(10)}.png"
    
    fused_data_path = os.path.join(fused_path, fused_filename)
    cv2.imwrite(fused_data_path, fused_img)
    
    # Add to label writer (uses DVS labels since same viewpoint)
    fused_rel_path = fused_data_path.replace(fused_path, fused_modality_name)
    
    return fused_data_path, height, width, fused_rel_path

def store_sequence_data(key:str, source:DataSource, frames_data:list, frames_labels:list, 
                        result_dir:str, current_idx:int, mapping:Mapping, label_writer:LabelWriter,
                        verbose=False, radar_frames:list=None, radar_max_dist:float=10.0, 
                        radar_dilation:int=15, filter_class_ids:list=None):
    """
    Store a sequence of frames for ReYOLOv8 compatibility with optional radar fusion.
    
    EXTENDED from original ReYOLOv8 scripts to support:
    - Radar fusion: Add radar as additional channel (DVS_channels + 1)
    - Distance labels: Extract distance from radar, store as 6th column
    
    Original compatibility maintained:
    - HDF5 dataset key is '1mp' (matching original)
    - Labels stored as np.array(labels, dtype=object) with per-frame bboxes
    - Naming: sequence_00_subseq_XXXXXXX.h5 and .npy
    - Compression: Blosc zstd (matching original)
    
    Args:
        key: Source key (e.g., 'prophesee', 'davis')
        source: DataSource instance
        frames_data: List of frame data (entire session or clip)
        frames_labels: List of label data for each frame
        result_dir: Output directory
        current_idx: Sequence index
        mapping: Mapping object for label transformation
        label_writer: LabelWriter instance
        verbose: Print progress
        radar_frames: Optional list of radar frames (H, W) for fusion and distance extraction
        radar_max_dist: Maximum radar distance in meters (for distance denormalization)
        radar_dilation: Dilation kernel size for sparse radar (0 = disabled, 15 = default)
        filter_class_ids: List of COCO category IDs being used (for COCO ID -> YOLO index mapping)
    """
    import h5py
    import hdf5plugin
    import torch
    
    base_path = os.path.join(result_dir, key)
    
    # Naming convention EXACTLY matching original scripts
    # Original (GEN1): sequence_XX_subseq_NNNNNN.h5
    seq_name = 'sequence_00_subseq_' + str(current_idx).zfill(7) + '.h5'
    seq_path = os.path.join(base_path, seq_name)
    
    if verbose:
        print(f"  Storing {len(frames_data)} frames to {seq_name}")
    
    # Process all frames into event representations
    # This matches lines 160-171 in singleShot_eventDataHandler_GEN1.py
    imgs = []
    labels = []
    
    height = None
    width = None
    
    # Create COCO ID -> YOLO index mapping
    # YOLO uses 0-indexed sequential class indices based on sorted COCO IDs
    # E.g., if filter_class_ids = [1, 3] (person, car), then: {1: 0, 3: 1}
    if filter_class_ids is not None and len(filter_class_ids) > 0:
        sorted_class_ids = sorted(filter_class_ids)
        coco_to_yolo_idx = {coco_id: yolo_idx for yolo_idx, coco_id in enumerate(sorted_class_ids)}
    else:
        # Fallback: assume all COCO classes, YOLO index = COCO ID - 1
        # This maintains backwards compatibility but may not work for all 80 classes
        coco_to_yolo_idx = None
    
    # Process frames with optional progress bar (only shown when not verbose and processing many frames)
    frame_iterator = tqdm(enumerate(frames_data), total=len(frames_data),
                          desc=f'    Converting to {source.event_representation if hasattr(source, "event_representation") else "format"}',
                          leave=False, position=2, disable=verbose or len(frames_data) < 100)
    
    for frame_idx, frame_data in frame_iterator:
        # Apply event representation if specified (like original scripts)
        if hasattr(source, 'event_representation') and source.event_representation:
            from nerve.processing.event_representations import process_events
            if isinstance(frame_data[0], np.ndarray) and frame_data[0].dtype.names:
                # IMPORTANT: Process at NATIVE resolution first, then apply pooling/padding
                # This ensures events are correctly placed before any downsampling
                native_h = getattr(source, 'native_height', source.height)
                native_w = getattr(source, 'native_width', source.width)
                
                representation = process_events(
                    frame_data[0],
                    source.event_representation,
                    source.representation_bins,
                    native_h,
                    native_w
                )
                
                # Apply average pooling if specified (kernel > 1)
                pool_kernel = getattr(source, 'avg_pool_kernel', 1)
                if pool_kernel > 1:
                    # representation is (C, H, W) - apply mean pooling
                    c, h, w = representation.shape
                    new_h = h // pool_kernel
                    new_w = w // pool_kernel
                    # Crop to exact multiple of pool_kernel
                    representation = representation[:, :new_h * pool_kernel, :new_w * pool_kernel]
                    # Reshape and average
                    representation = representation.reshape(c, new_h, pool_kernel, new_w, pool_kernel)
                    representation = representation.mean(axis=(2, 4))
                
                # Apply padding if specified
                pad_size = getattr(source, 'pad_size', [0, 0])
                if pad_size[0] > 0 or pad_size[1] > 0:
                    # Pad format for numpy: ((before_1, after_1), (before_2, after_2), ...)
                    # representation is (C, H, W), we pad H and W
                    pad_width = ((0, 0), (0, pad_size[1]), (0, pad_size[0]))
                    representation = np.pad(representation, pad_width, mode='constant', constant_values=0)
            else:
                representation = frame_data[0]
        else:
            representation = frame_data[0]
        
        if isinstance(representation, torch.Tensor):
            representation = representation.cpu().numpy()
        
        if height is None:
            _, height, width = representation.shape
        
        # RADAR FUSION: Add radar as additional channel if available
        radar_frame_data = None
        if radar_frames is not None and frame_idx < len(radar_frames):
            radar_frame_data = radar_frames[frame_idx]
            
            if radar_frame_data is not None:
                # Resize radar to match DVS if needed (radar may be projected at
                # template output_shape which differs from native DVS resolution)
                if radar_frame_data.shape != (height, width):
                    import cv2 as _cv2_r
                    radar_frame_data = _cv2_r.resize(
                        radar_frame_data, (width, height),
                        interpolation=_cv2_r.INTER_NEAREST
                    )
                try:
                    representation = fuse_radar_with_representation(representation, radar_frame_data)
                    if verbose and frame_idx == 0:
                        print(f"  Radar fusion: {representation.shape[0]-1} DVS channels + 1 radar channel = {representation.shape[0]} total")
                except Exception as e:
                    if verbose:
                        print(f"  Warning: Radar fusion failed for frame {frame_idx}: {e}")
                    radar_frame_data = None  # Mark as unavailable
        
        imgs.append(representation)
        
        # Convert labels to YOLO format for this frame
        # NOW WITH DISTANCE AS 6TH COLUMN (for distance estimation support)
        # CRITICAL: Labels come from RGB camera space and need to be mapped to DVS space
        # This mirrors the PNG/YOLO pipeline which uses:
        # 1. MapLabels() - transforms from RGB to DVS camera space via perspective projection
        # 2. Transform_annotation() - applies pooling/padding adjustments
        frame_labels = frames_labels[frame_idx]
        if frame_labels is None or len(frame_labels) == 0:
            # Empty frame: 6 columns for distance support
            frame_bboxes = np.zeros((0, 6), dtype=np.float64)
        else:
            # First apply MapLabels to transform from RGB to DVS camera space
            # This is what LabelWriter.add_data() does for PNG output
            import torch
            device = torch.device("cpu")
            mapped_labels = MapLabels(frame_labels, mapping, None, device)
            
            frame_bboxes = []
            for ann in mapped_labels:
                if 'bbox' not in ann:
                    continue
                
                # Now apply transform_annotation_dvs for pooling/padding adjustments
                transformed_ann = source.Transform_annotation(ann)
                bbox = transformed_ann['bbox']  # COCO format: [x, y, w, h], now in sensor space
                if bbox[2] <= 0 or bbox[3] <= 0:
                    continue
                
                # Convert to YOLO format: [class, center_x, center_y, width, height] normalized
                # Matches to_bbox_yolo_format() in original utils.py
                # Map COCO category ID to YOLO class index (0-indexed, based on sorted filter list)
                coco_cat_id = ann.get('category_id', 1)
                if coco_to_yolo_idx is not None:
                    if coco_cat_id not in coco_to_yolo_idx:
                        continue  # Skip categories not in filter
                    class_id = coco_to_yolo_idx[coco_cat_id]
                else:
                    # Fallback for backwards compatibility (assumes sequential from 1)
                    class_id = coco_cat_id - 1
                center_x = (bbox[0] + bbox[2] / 2) / width
                center_y = (bbox[1] + bbox[3] / 2) / height
                box_w = bbox[2] / width
                box_h = bbox[3] / height
                
                # DISTANCE EXTRACTION: Extract from radar if available
                distance = -1.0  # Default: no distance available
                if radar_frame_data is not None:
                    yolo_bbox = [class_id, center_x, center_y, box_w, box_h]
                    distance = extract_bbox_distance_from_radar(
                        yolo_bbox, radar_frame_data, width, height, radar_max_dist
                    )
                
                # Store with 6 columns: [class, cx, cy, w, h, distance]
                frame_bboxes.append([class_id, center_x, center_y, box_w, box_h, distance])
            
            if len(frame_bboxes) > 0:
                frame_bboxes = np.array(frame_bboxes, dtype=np.float64)
            else:
                # Empty frame: 6 columns for distance support
                frame_bboxes = np.zeros((0, 6), dtype=np.float64)
        
        labels.append(frame_bboxes)
        
        # Collect sample for visualization (first few frames with labels)
        if len(frame_bboxes) > 0 and len(get_collected_samples()) < 3:
            collect_sample_for_visualization(
                representation, 
                frame_bboxes,
                f"ReYOLOv8 {key} frame {frame_idx}"
            )
    
    # Stack frames: shape (num_frames, channels, height, width)
    # Matches line 193 in singleShot_eventDataHandler_GEN1.py
    imgs_array = np.stack(imgs, axis=0)
    del imgs
    
    # Save HDF5 with key '1mp' and Blosc compression
    # EXACTLY matches save_compressed_clip() in utils.py (lines 113-122)
    with h5py.File(seq_path, 'w') as hf:
        hf.create_dataset(
            '1mp',  # MUST be '1mp' to match original scripts
            data=imgs_array,
            **hdf5plugin.Blosc(cname='zstd')  # Match original compression
        )
    
    del imgs_array
    
    # Save labels as np.array(labels, dtype=object)
    # EXACTLY matches line 125 in utils.py
    labels_array = np.array(labels, dtype=object)
    
    # Labels go in labels/ directory with same naming
    label_name = 'sequence_00_subseq_' + str(current_idx).zfill(7) + '.npy'
    
    # Create labels directory structure matching images
    labels_base = base_path.replace('/data/', '/labels/').replace('\\data\\', '\\labels\\')
    os.makedirs(labels_base, exist_ok=True)
    label_path = os.path.join(labels_base, label_name)
    
    np.save(label_path, labels_array)
    
    if verbose:
        total_boxes = sum(len(l) for l in labels)
        boxes_with_dist = sum(np.sum(l[:, 5] > 0) if len(l) > 0 else 0 for l in labels)
        print(f"  Saved {len(labels)} frames with {total_boxes} total boxes")
        if radar_frames is not None:
            print(f"  Distance annotations: {boxes_with_dist}/{total_boxes} boxes have valid radar distance")
            if boxes_with_dist == 0:
                # Count how many frames had radar data
                frames_with_radar = sum(1 for rf in radar_frames if rf is not None and np.count_nonzero(rf) > 0)
                total_radar_pixels = sum(np.count_nonzero(rf) if rf is not None else 0 for rf in radar_frames)
                print(f"  ⚠️  SPARSE RADAR: {frames_with_radar}/{len(radar_frames)} frames had radar data, "
                      f"{total_radar_pixels} total non-zero pixels across all frames")
                print(f"  ℹ️  TI radar is very sparse - consider using dilation or accepting limited distance coverage")
    
    # Also add to COCO annotations for compatibility with other tools
    seq_path_rel = seq_path.replace(base_path, key)
    mid_frame_idx = len(frames_labels) // 2
    label_writer.add_data(frames_labels[mid_frame_idx], mapping, seq_path_rel, 
                         (height, width), current_idx, source.Transform_annotation)



def store_rvt_sequence_data(key: str, source, frames_data: list, frames_labels: list,
                            result_dir: str, sequence_idx: int, mapping, label_writer,
                            verbose=False, radar_frames: list = None, radar_max_dist: float = 10.0,
                            radar_dilation: int = 15, filter_class_ids: list = None,
                            ev_repr_name: str = 'stacked_histogram_dt=50_nbins=10',
                            frame_period_ms: float = 50.0, fuse_radar_channel: bool = True):
    """
    Store a sequence in RVT-compatible format.
    
    RVT expects a specific directory structure:
    - event_representations_v2/{ev_repr_name}/event_representations.h5 (key: 'data')
    - event_representations_v2/{ev_repr_name}/objframe_idx_2_repr_idx.npy
    - event_representations_v2/{ev_repr_name}/timestamps_us.npy
    - labels_v2/labels.npz (structured array with t, x, y, w, h, class_id, class_confidence, distance)
    - labels_v2/timestamps_us.npy
    
    Args:
        key: Source key (e.g., 'davis', 'prophesee')
        source: DataSource instance
        frames_data: List of frame data (event representations)
        frames_labels: List of annotations per frame
        result_dir: Output directory (e.g., train/data)
        sequence_idx: Index for this sequence
        mapping: Coordinate mapping object
        label_writer: LabelWriter for COCO annotations
        verbose: Print progress info
        radar_frames: Optional radar frames for fusion and distance
        radar_max_dist: Maximum radar distance in meters
        radar_dilation: Radar dilation kernel size
        filter_class_ids: COCO category IDs to include
        ev_repr_name: Event representation name (for directory naming)
        frame_period_ms: Frame period in milliseconds
        fuse_radar_channel: Whether to fuse radar as additional channel
    """
    import h5py
    import hdf5plugin
    import torch
    
    # RVT expects sequences directly under the split directory (train/sequence_NNNNNN/)
    # NOT under a source key subdirectory (train/davis/sequence_NNNNNN/)
    # This is different from other formats that organize by source key
    
    # Create RVT directory structure for this sequence
    # Format: sequence_NNNNNN/
    seq_name = f'sequence_{str(sequence_idx).zfill(6)}'
    seq_dir = os.path.join(result_dir, seq_name)
    
    ev_repr_dir = os.path.join(seq_dir, 'event_representations_v2', ev_repr_name)
    labels_dir = os.path.join(seq_dir, 'labels_v2')
    
    os.makedirs(ev_repr_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    if verbose:
        print(f"  Storing RVT sequence: {seq_name} ({len(frames_data)} frames)")
    
    # Detect if radar is available
    has_radar = radar_frames is not None and len(radar_frames) > 0
    
    # Create COCO ID -> YOLO index mapping
    if filter_class_ids is not None and len(filter_class_ids) > 0:
        sorted_class_ids = sorted(filter_class_ids)
        coco_to_yolo_idx = {coco_id: yolo_idx for yolo_idx, coco_id in enumerate(sorted_class_ids)}
    else:
        coco_to_yolo_idx = None
    
    # Process frames
    imgs = []
    all_labels = []
    timestamps_us = []
    objframe_idx_2_label_idx = []
    label_timestamps_us = []
    current_label_idx = 0
    
    height = None
    width = None
    
    for frame_idx, frame_data in enumerate(frames_data):
        # Compute timestamp in microseconds
        timestamp_us = int(frame_idx * frame_period_ms * 1000)
        timestamps_us.append(timestamp_us)
        
        # Apply event representation if needed
        if hasattr(source, 'event_representation') and source.event_representation:
            from nerve.processing.event_representations import process_events
            if isinstance(frame_data[0], np.ndarray) and frame_data[0].dtype.names:
                # IMPORTANT: Process at NATIVE resolution first, then apply pooling/padding
                # This ensures events are correctly placed before any downsampling
                native_h = getattr(source, 'native_height', source.height)
                native_w = getattr(source, 'native_width', source.width)
                
                representation = process_events(
                    frame_data[0],
                    source.event_representation,
                    source.representation_bins,
                    native_h,
                    native_w
                )
                
                # Apply average pooling if specified (kernel > 1)
                pool_kernel = getattr(source, 'avg_pool_kernel', 1)
                if pool_kernel > 1:
                    # representation is (C, H, W) - apply mean pooling
                    c, h, w = representation.shape
                    new_h = h // pool_kernel
                    new_w = w // pool_kernel
                    # Crop to exact multiple of pool_kernel
                    representation = representation[:, :new_h * pool_kernel, :new_w * pool_kernel]
                    # Reshape and average
                    representation = representation.reshape(c, new_h, pool_kernel, new_w, pool_kernel)
                    representation = representation.mean(axis=(2, 4))
                
                # Apply padding if specified
                pad_size = getattr(source, 'pad_size', [0, 0])
                if pad_size[0] > 0 or pad_size[1] > 0:
                    # Pad format for numpy: ((before_1, after_1), (before_2, after_2), ...)
                    # representation is (C, H, W), we pad H and W
                    pad_width = ((0, 0), (0, pad_size[1]), (0, pad_size[0]))
                    representation = np.pad(representation, pad_width, mode='constant', constant_values=0)
            else:
                representation = frame_data[0]
        else:
            representation = frame_data[0]
        
        if isinstance(representation, torch.Tensor):
            representation = representation.cpu().numpy()
        
        if height is None:
            _, height, width = representation.shape
        
        # Radar fusion (if enabled and available)
        radar_frame_data = None
        if has_radar and frame_idx < len(radar_frames) and fuse_radar_channel:
            radar_frame_data = radar_frames[frame_idx]
            if radar_frame_data is not None:
                if radar_frame_data.shape != (height, width):
                    import cv2 as _cv2_r2
                    radar_frame_data = _cv2_r2.resize(
                        radar_frame_data, (width, height),
                        interpolation=_cv2_r2.INTER_NEAREST
                    )
                try:
                    representation = fuse_radar_with_representation(representation, radar_frame_data)
                except Exception as e:
                    if verbose:
                        print(f"  Warning: Radar fusion failed for frame {frame_idx}: {e}")
                    radar_frame_data = None
        
        imgs.append(representation)
        

        frame_labels = frames_labels[frame_idx]
        frame_has_labels = frame_labels is not None and len(frame_labels) > 0
        
        if frame_has_labels:
            objframe_idx_2_label_idx.append(current_label_idx)
            label_timestamps_us.append(timestamp_us)
            
            # First apply MapLabels to transform from RGB to DVS camera space
            device = torch.device("cpu")
            mapped_labels = MapLabels(frame_labels, mapping, None, device)
            
            for ann in mapped_labels:
                if 'bbox' not in ann:
                    continue
                
                # Now apply transform_annotation_dvs for pooling/padding adjustments
                transformed_ann = source.Transform_annotation(ann)
                bbox = transformed_ann['bbox']  # COCO format: [x, y, w, h]
                
                if bbox[2] <= 0 or bbox[3] <= 0:
                    continue
                
                # Map COCO category ID to class index
                coco_cat_id = ann.get('category_id', 1)
                if coco_to_yolo_idx is not None:
                    if coco_cat_id not in coco_to_yolo_idx:
                        continue
                    class_id = coco_to_yolo_idx[coco_cat_id]
                else:
                    class_id = coco_cat_id - 1
                
                # Extract distance from radar if available
                distance = -1.0
                if has_radar and frame_idx < len(radar_frames):
                    radar_for_dist = radar_frames[frame_idx]
                    if radar_for_dist is not None:
                        if radar_for_dist.shape != (height, width):
                            import cv2 as _cv2_r3
                            radar_for_dist = _cv2_r3.resize(
                                radar_for_dist, (width, height),
                                interpolation=_cv2_r3.INTER_NEAREST
                            )
                        center_x = (bbox[0] + bbox[2] / 2) / width
                        center_y = (bbox[1] + bbox[3] / 2) / height
                        box_w = bbox[2] / width
                        box_h = bbox[3] / height
                        yolo_bbox = [class_id, center_x, center_y, box_w, box_h]
                        distance = extract_bbox_distance_from_radar(
                            yolo_bbox, radar_for_dist, width, height, radar_max_dist
                        )
                
                # RVT label format: (t, x, y, w, h, class_id, class_confidence, distance)
                label_entry = (
                    timestamp_us,  # t
                    bbox[0],       # x (top-left)
                    bbox[1],       # y (top-left)
                    bbox[2],       # w
                    bbox[3],       # h
                    class_id,      # class_id
                    1.0,           # class_confidence (ground truth = 1.0)
                    distance       # distance in meters (-1 if unavailable)
                )
                all_labels.append(label_entry)
                current_label_idx += 1
            
            # Collect sample for visualization (first few frames with labels)
            if len(get_collected_samples()) < 3:
                # Get labels for this frame in RVT format
                frame_labels_rvt = [(timestamp_us, bbox[0], bbox[1], bbox[2], bbox[3], class_id, 1.0, distance)
                                    for ann in mapped_labels if 'bbox' in ann 
                                    for bbox, class_id in [(source.Transform_annotation(ann)['bbox'], 
                                                           coco_to_yolo_idx.get(ann.get('category_id', 1), ann.get('category_id', 1) - 1) 
                                                           if coco_to_yolo_idx else ann.get('category_id', 1) - 1)]
                                    if bbox[2] > 0 and bbox[3] > 0][:5]  # Limit to 5 labels per sample
                if frame_labels_rvt:
                    collect_sample_for_visualization(
                        representation.copy() if hasattr(representation, 'copy') else representation,
                        frame_labels_rvt,
                        f"RVT {key} frame {frame_idx}"
                    )
    
    # Stack all frames: (num_frames, channels, height, width)
    imgs_array = np.stack(imgs, axis=0)
    
    # Save event representations HDF5
    ev_repr_path = os.path.join(ev_repr_dir, 'event_representations.h5')
    with h5py.File(ev_repr_path, 'w') as hf:
        hf.create_dataset(
            'data',  # RVT uses 'data' key
            data=imgs_array,
            **hdf5plugin.Blosc(cname='zstd')
        )
    
    # Save timestamps
    np.save(os.path.join(ev_repr_dir, 'timestamps_us.npy'), np.array(timestamps_us, dtype=np.int64))
    
    # Save labels as structured array (RVT format)
    # Define structured dtype with distance field
    label_dtype = np.dtype([
        ('t', np.int64),
        ('x', np.float64),
        ('y', np.float64),
        ('w', np.float64),
        ('h', np.float64),
        ('class_id', np.int64),
        ('class_confidence', np.float64),
        ('distance', np.float64)  # Distance extension for PEGMA
    ])
    
    if len(all_labels) > 0:
        # CRITICAL FIX: Sort labels by timestamp before creating array
        # This ensures labels are contiguous per timestamp (required by RVT)
        all_labels_sorted = sorted(all_labels, key=lambda x: x[0])  # Sort by timestamp (index 0)
        labels_array = np.array(all_labels_sorted, dtype=label_dtype)
        
        # Rebuild objframe_idx_2_label_idx based on sorted labels
        # Get unique timestamps in sorted order (preserving order from sorted labels)
        unique_timestamps = []
        seen = set()
        for lbl in all_labels_sorted:
            ts = lbl[0]
            if ts not in seen:
                unique_timestamps.append(ts)
                seen.add(ts)
        
        # Build mapping: each timestamp -> first label index in sorted array
        objframe_idx_2_label_idx_new = []
        for ts in unique_timestamps:
            for idx, lbl in enumerate(all_labels_sorted):
                if lbl[0] == ts:
                    objframe_idx_2_label_idx_new.append(idx)
                    break
        
        objframe_idx_2_label_idx_array = np.array(objframe_idx_2_label_idx_new, dtype=np.int64)
        
        # Update label_timestamps_us to match sorted unique timestamps
        label_timestamps_us = unique_timestamps
        
        # Now compute objframe_idx_2_repr_idx using the updated label_timestamps_us
        # Create mapping from timestamp to FIRST occurrence index in representation timestamps
        label_ts_to_repr_idx = {}
        for idx, ts in enumerate(timestamps_us):
            if ts not in label_ts_to_repr_idx:
                label_ts_to_repr_idx[ts] = idx
        
        repr_indices = [label_ts_to_repr_idx.get(ts, 0) for ts in label_timestamps_us]
        
        # Validation: ensure monotonicity of objframe_idx_2_repr_idx
        if len(repr_indices) > 1:
            is_monotonic = all(repr_indices[i] <= repr_indices[i+1] for i in range(len(repr_indices)-1))
            if not is_monotonic:
                print(f"  WARNING: objframe_idx_2_repr_idx is not monotonic after sorting, this may indicate data issues")
        
        np.save(os.path.join(ev_repr_dir, 'objframe_idx_2_repr_idx.npy'), np.array(repr_indices, dtype=np.int64))
    else:
        labels_array = np.array([], dtype=label_dtype)
        objframe_idx_2_label_idx_array = np.array([], dtype=np.int64)
        np.save(os.path.join(ev_repr_dir, 'objframe_idx_2_repr_idx.npy'), np.array([], dtype=np.int64))
    
    # Save labels.npz
    np.savez(
        os.path.join(labels_dir, 'labels.npz'),
        labels=labels_array,
        objframe_idx_2_label_idx=objframe_idx_2_label_idx_array
    )
    
    # Save label timestamps
    np.save(os.path.join(labels_dir, 'timestamps_us.npy'), np.array(label_timestamps_us, dtype=np.int64))
    
    if verbose:
        total_boxes = len(all_labels)
        boxes_with_dist = sum(1 for l in all_labels if l[7] > 0)  # distance is index 7
        print(f"  Saved RVT sequence: {len(frames_data)} frames, {total_boxes} boxes")
        if has_radar:
            print(f"  Distance annotations: {boxes_with_dist}/{total_boxes} boxes have valid radar distance")
    
    # Also add to COCO annotations for compatibility
    mid_frame_idx = len(frames_labels) // 2
    seq_path_rel = os.path.join(key, seq_name)
    label_writer.add_data(frames_labels[mid_frame_idx], mapping, seq_path_rel,
                         (height, width), sequence_idx, source.Transform_annotation)
    
    del imgs_array
    del imgs


def main():
    args = get_arguments()
    dataset_root = str(args.dataset)
    is_adding = bool(args.add)
    clean = bool(args.clean)
    settings_file = str(args.settings)
    verbose = bool(args.verbose)
    split_name = str(args.split) if args.split else ""
    no_yaml = bool(args.no_yaml)
    
    # Reset sample collection for visualization
    reset_collected_samples()
    
    assert settings_file.endswith('.json') and os.path.isfile(settings_file), f"Settings file must be a .json file: {settings_file}"

    sessions_list = str(args.list)
    single_session = str(args.single_session)

    assert single_session != "" or sessions_list != "", "At least one between --single-session and --list must be used."
    assert not (single_session != "" and sessions_list != ""), "You can use just one between --single-session and --list"

    if sessions_list != "":
        assert sessions_list.endswith('.txt') and os.path.isfile(sessions_list), f"Session list must be a .txt file: {sessions_list}"
        with open(sessions_list, 'r') as file:
            sessions = [line.strip() for line in file.readlines() if line.strip()]
    else:
        sessions = [single_session]

    assert not (clean and is_adding), "Cannot use both --clean and --add flags together."

    with open(settings_file, 'r') as openfile:
        settings = json.load(openfile)

    # Determine output directory structure
    # If --split is specified, create {dataset_root}/{split}/ structure
    if split_name:
        result_dir = os.path.join(dataset_root, split_name)
    else:
        result_dir = dataset_root

    # Handle clean flag - only clean the split directory, not the entire dataset root
    if clean and os.path.isdir(result_dir):
        shutil.rmtree(result_dir)

    # Check if we can proceed
    if os.path.isdir(result_dir) and not is_adding:
        raise ValueError(f"Output directory exists: {result_dir}. Use --clean to override or --add to append.")

    label_writers = {}
    
    # Detect output format to handle directory structure
    output_format = settings[0].get('output_format', 'reyolov8') if settings else 'reyolov8'
    is_rvt_format = output_format == 'rvt'

    # Extract class filtering from settings
    # Support both "filter_classes" (class names) and legacy "only_classes" (class IDs)
    filter_class_ids = None
    for s in settings:
        if 'filter_classes' in s:
            # Class names - resolve to IDs
            filter_class_ids = ResolveClassNamesToIds(s['filter_classes'])
            if verbose:
                print(f"Filtering classes: {s['filter_classes']} -> IDs: {filter_class_ids}")
            break
        elif 'only_classes' in s:
            # Legacy: direct IDs
            filter_class_ids = s['only_classes']
            if verbose:
                print(f"Filtering classes by ID: {filter_class_ids}")
            break

    if is_rvt_format:
        # RVT format: sequences go directly under split dir (train/sequence_NNNNNN/)
        # No annotations/ or data/ subdirectories - RVT iterates over split dir
        data_path = result_dir  # Pass result_dir directly for RVT
        
        # For RVT, label_writers are still needed but annotations are stored differently
        # Create a dummy writer path that won't be used (RVT stores labels in labels_v2/)
        annotations_dir = os.path.join(result_dir, '_rvt_annotations_tmp')
        for s in settings:
            name = s['data']
            out_path = os.path.join(annotations_dir, name + '.json')
            label_writers[name] = LabelWriter(out_path, filter_class_ids=filter_class_ids)
        
        if not is_adding:
            os.makedirs(result_dir, exist_ok=True)
            current_index = 0
        else:
            # For RVT, count existing sequences
            existing_seqs = [d for d in os.listdir(result_dir) if d.startswith('sequence_')]
            current_index = len(existing_seqs)
    else:
        # Standard format: use annotations/ and data/ subdirectories
        annotations_dir = os.path.join(result_dir, 'annotations')
        data_path = os.path.join(result_dir, 'data')

        for s in settings:
            name = s['data']
            out_path = os.path.join(annotations_dir, name + '.json')
            label_writers[name] = LabelWriter(out_path, filter_class_ids=filter_class_ids)
        
        # Check for DVS-Radar PNG fusion and create label writer for fused modality
        fused_modality_name = None
        for s in settings:
            if s.get('fuse_dvs_radar_png', False):
                dvs_source_name = s['data']
                fused_modality_name = s.get('fused_modality_name', f"{dvs_source_name}_radar")
                # Create label writer for the fused modality
                fused_out_path = os.path.join(annotations_dir, fused_modality_name + '.json')
                label_writers[fused_modality_name] = LabelWriter(fused_out_path, filter_class_ids=filter_class_ids)
                if verbose:
                    print(f"DVS-Radar PNG fusion enabled: will create '{fused_modality_name}' modality")
                break

        if not is_adding:
            os.makedirs(result_dir, exist_ok=True)
            os.makedirs(annotations_dir, exist_ok=True)
            os.makedirs(data_path, exist_ok=True)
            current_index = 0
        else:
            current_index = list(label_writers.values())[0].get_last_image_index() + 1 

    # Extract clip mode settings from first source (if specified)
    clip_mode = settings[0].get('clip_mode', 'single_frame') if settings else 'single_frame'
    clip_length = settings[0].get('clip_length', 1) if settings else 1
    clip_stride = settings[0].get('clip_stride', 1) if settings else 1
    
    if verbose:
        print("="*70)
        print("PEGMA Dataset Generation")
        print("="*70)
        print(f"Mode: {clip_mode}")
        if split_name:
            print(f"Split: {split_name}")
        print(f"Output: {result_dir}")
        print(f"Sessions to process: {len(sessions)}")
        print("="*70)
    
    for session_idx, s in enumerate(tqdm(sessions, desc="Sessions", position=0, leave=True)):
        current_index = extract_from_single_session(s, settings, label_writers, data_path, current_index,
                                                    clip_mode=clip_mode, clip_length=clip_length, 
                                                    clip_stride=clip_stride, verbose=verbose,
                                                    filter_class_ids=filter_class_ids)
        
        # IMPORTANT: Write labels after each session to avoid data loss (not for RVT)
        if not is_rvt_format:
            for w in label_writers:
                label_writers[w].write_file()
        
        # Memory management: Force garbage collection between sessions
        # This helps prevent memory accumulation across many sessions
        gc.collect()

    # Final write to ensure everything is saved (not for RVT - uses labels_v2/ format)
    if not is_rvt_format:
        for w in label_writers:
            label_writers[w].write_file()
    else:
        # Clean up temp annotations dir for RVT
        if os.path.exists(annotations_dir):
            shutil.rmtree(annotations_dir, ignore_errors=True)
    
    if verbose:
        print("\n" + "="*70)
        print("Dataset generation complete!")
        print("="*70)
    
    # Auto-generate data.yaml if --split is used (dataset has proper structure)
    if not no_yaml and split_name:
        generate_data_yaml(dataset_root, settings, verbose=verbose)
    elif not no_yaml and verbose:
        print("\nℹ️  Use --split to enable automatic data.yaml generation")
        print("   Or run create_data_yaml.py manually after creating all splits")
    
    # Create view_samples folder with sample visualizations
    # This helps verify dataset generation regardless of output format
    collected_samples = get_collected_samples()
    if collected_samples:
        create_view_samples(result_dir, collected_samples, output_format, verbose=verbose)
    elif verbose:
        print("\nNo samples collected for visualization (no labels found)")
    
    return



if __name__ == '__main__':
    main()

