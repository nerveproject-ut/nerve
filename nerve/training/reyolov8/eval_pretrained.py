#!/usr/bin/env python3
"""
Zero-shot and few-shot evaluation script for ReYOLOv8 pretrained weights.

Usage:
    # Zero-shot evaluation (no training, just test pretrained weights)
    python eval_pretrained.py --weights /path/to/reyolov8n_pedro.pt --data /path/to/data.yaml --mode zero_shot
    
    # Few-shot evaluation info
    python eval_pretrained.py --mode info
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
from ultralytics.yolo.utils import LOGGER, colorstr
from ultralytics.yolo.data.utils import check_det_dataset


def inspect_weights(weights_path):
    """Inspect pretrained weights to show model info."""
    LOGGER.info(f"\n{'='*60}")
    LOGGER.info(f"Inspecting weights: {weights_path}")
    LOGGER.info(f"{'='*60}")
    
    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
    
    if isinstance(checkpoint, dict):
        LOGGER.info(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        if 'model' in checkpoint:
            model = checkpoint['model']
            if hasattr(model, 'names'):
                LOGGER.info(f"\nClasses in pretrained model:")
                for i, name in enumerate(model.names):
                    LOGGER.info(f"  {i}: {name}")
                LOGGER.info(f"\nTotal classes: {len(model.names)}")
            
            if hasattr(model, 'yaml'):
                LOGGER.info(f"\nModel config: {model.yaml.get('yaml_file', 'N/A')}")
                LOGGER.info(f"Number of classes (nc): {model.yaml.get('nc', 'N/A')}")
        
        if 'epoch' in checkpoint:
            LOGGER.info(f"\nTrained for {checkpoint['epoch']} epochs")
        
        if 'best_fitness' in checkpoint:
            LOGGER.info(f"Best fitness: {checkpoint['best_fitness']:.4f}")
    else:
        LOGGER.info("Checkpoint is a raw model (not a dict)")
        if hasattr(checkpoint, 'names'):
            LOGGER.info(f"Classes: {checkpoint.names}")


def run_zero_shot_eval(args):
    """Run zero-shot evaluation using pretrained weights."""
    from val import EventVideoDetectionValidator, parse_opt
    from ultralytics.nn.autobackend import AutoBackendMemory
    from ultralytics.yolo.utils.torch_utils import select_device
    from ultralytics.yolo.cfg import get_cfg
    
    LOGGER.info(f"\n{'='*60}")
    LOGGER.info(f"ZERO-SHOT EVALUATION")
    LOGGER.info(f"{'='*60}")
    LOGGER.info(f"Weights: {args.weights}")
    LOGGER.info(f"Data: {args.data}")
    LOGGER.info(f"Image size: {args.imgsz}")
    LOGGER.info(f"Channels: {args.channels}")
    LOGGER.info(f"Clip: length={args.clip_length}, stride={args.clip_stride}")
    LOGGER.info(f"{'='*60}\n")
    
    # Build video config
    video_config = {
        "clip_length": args.clip_length,
        "clip_stride": args.clip_stride,
        "channels": args.channels,
    }
    
    # Create validator with args
    class Args:
        pass
    
    val_args = Args()
    val_args.data = args.data
    val_args.imgsz = args.imgsz
    val_args.batch = args.batch
    val_args.device = args.device
    val_args.workers = args.workers
    val_args.conf = args.conf
    val_args.iou = args.iou
    val_args.max_det = args.max_det
    val_args.half = args.half
    val_args.dnn = False
    val_args.plots = args.plots
    val_args.save_json = False
    val_args.save_hybrid = False
    val_args.verbose = True
    val_args.split = args.split
    val_args.project = Path(args.project)
    val_args.name = args.name
    val_args.exist_ok = True
    val_args.task = 'detect'
    val_args.show_sequences = 3
    
    validator = EventVideoDetectionValidator(
        video_config=video_config,
        args=val_args
    )
    
    # Run validation
    results = validator(model=args.weights)
    
    LOGGER.info(f"\n{'='*60}")
    LOGGER.info("ZERO-SHOT RESULTS")
    LOGGER.info(f"{'='*60}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='ReYOLOv8 Zero-shot/Few-shot Evaluation')
    parser.add_argument('--weights', type=str, default=None, help='Path to pretrained weights (.pt)')
    parser.add_argument('--data', type=str, default=None, help='Path to data.yaml')
    parser.add_argument('--imgsz', type=int, default=346, help='Image size')
    parser.add_argument('--channels', type=int, default=5, help='Number of input channels')
    parser.add_argument('--clip_length', type=int, default=5, help='Clip length')
    parser.add_argument('--clip_stride', type=int, default=5, help='Clip stride')
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    parser.add_argument('--device', default='0', help='CUDA device')
    parser.add_argument('--workers', type=int, default=8, help='Dataloader workers')
    parser.add_argument('--conf', type=float, default=0.001, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.6, help='IoU threshold')
    parser.add_argument('--max_det', type=int, default=300, help='Max detections')
    parser.add_argument('--half', action='store_true', help='Use FP16')
    parser.add_argument('--plots', action='store_true', help='Save plots')
    parser.add_argument('--split', default='val', help='Dataset split (train/val/test)')
    parser.add_argument('--project', default='runs/zero_shot', help='Project directory')
    parser.add_argument('--name', default='eval', help='Experiment name')
    parser.add_argument('--mode', choices=['zero_shot', 'inspect', 'info'], default='info',
                        help='Mode: zero_shot (evaluate), inspect (show weight info), info (show usage)')
    parser.add_argument('--filter_class', type=int, default=None,
                        help='Only evaluate predictions for this class (e.g., 0 for pedestrian only)')
    
    args = parser.parse_args()
    
    if args.mode == 'info':
        print("""
╔══════════════════════════════════════════════════════════════╗
║           ReYOLOv8 Zero-Shot & Few-Shot Evaluation           ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  ZERO-SHOT (No training, test pretrained weights directly):  ║
║  ─────────────────────────────────────────────────────────── ║
║  python eval_pretrained.py --mode zero_shot \\                ║
║      --weights /home/omansour/weights/pedro/reyolov8n_pedro.pt \\
║      --data /path/to/your/data.yaml \\                        ║
║      --imgsz 346 --channels 5 \\                              ║
║      --clip_length 5 --clip_stride 5 \\                       ║
║      --filter_class 0                                        ║
║                                                              ║
║  --filter_class 0 = Only evaluate pedestrian predictions     ║
║                     (ignores two-wheeler predictions)        ║
║                                                              ║
║  INSPECT WEIGHTS (See what classes the model was trained on):║
║  ─────────────────────────────────────────────────────────── ║
║  python eval_pretrained.py --mode inspect \\                  ║
║      --weights /home/omansour/weights/pedro/reyolov8n_pedro.pt║
║                                                              ║
║  FEW-SHOT (Train with pretrained weights):                   ║
║  ─────────────────────────────────────────────────────────── ║
║  Set in your experiment file (my_reyolov8_sequence.py):      ║
║      self.pretrained = '/path/to/reyolov8n_pedro.pt'         ║
║  Then run normal training with fewer epochs.                 ║
║                                                              ║
║  CLASS MAPPING:                                              ║
║  ─────────────────────────────────────────────────────────── ║
║  Pretrained: 0=pedestrian, 1=two-wheeler                     ║
║  Your data:  0=person (maps to pedestrian)                   ║
║                                                              ║
║  Use --filter_class 0 if your data only has pedestrians!     ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
        """)
        return
    
    if args.mode == 'inspect':
        if not args.weights:
            print("Error: --weights required for inspect mode")
            return
        inspect_weights(args.weights)
        return
    
    if args.mode == 'zero_shot':
        if not args.weights or not args.data:
            print("Error: --weights and --data required for zero_shot mode")
            return
        run_zero_shot_eval(args)


if __name__ == '__main__':
    main()

