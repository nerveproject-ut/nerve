#!/usr/bin/env python3
"""
Evaluation script for YOLOv8 with distance estimation support.
Similar to yoloX/tools/eval.py but for custom YOLOv8 experiments.

Usage:
    # Evaluate on validation set
    python eval_yolov8.py -f yolov8_exp/exp__dist__radar.py -c runs/detect/exp__dist__radar/weights/best.pt
    
    # Evaluate on test set
    python eval_yolov8.py -f yolov8_exp/exp__dist__radar.py -c runs/detect/exp__dist__radar/weights/best.pt --test
    
    # With custom confidence threshold
    python eval_yolov8.py -f yolov8_exp/exp__dist__radar.py -c runs/detect/exp__dist__radar/weights/best.pt --conf 0.25
"""

import argparse
import sys
import os
from pathlib import Path
import importlib.util

# Add current directory to path

from yolov8_distance_trainer import DistanceDetectionTrainer


def import_exp_from_file(exp_file):
    """
    Import experiment configuration from Python file.
    
    Args:
        exp_file: Path to experiment file
        
    Returns:
        Exp class from the file
    """
    exp_file = Path(exp_file)
    if not exp_file.exists():
        raise FileNotFoundError(f"Experiment file not found: {exp_file}")
    
    # Add the experiment file's directory to sys.path temporarily
    exp_dir = str(exp_file.parent.absolute())
    if exp_dir not in sys.path:
    
    # Load module from file
    spec = importlib.util.spec_from_file_location("exp_module", exp_file)
    exp_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exp_module)
    
    if not hasattr(exp_module, 'Exp'):
        raise AttributeError(f"Experiment file must contain 'Exp' class: {exp_file}")
    
    return exp_module.Exp


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='YOLOv8 Evaluation with Distance Estimation')
    
    # Experiment configuration
    parser.add_argument(
        '-f', '--exp-file',
        type=str,
        required=True,
        help='Path to experiment configuration file'
    )
    
    # Model checkpoint
    parser.add_argument(
        '-c', '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint (.pt file)'
    )
    
    # Evaluation parameters
    parser.add_argument(
        '--conf',
        type=float,
        default=0.001,
        help='Confidence threshold for predictions'
    )
    parser.add_argument(
        '--iou',
        type=float,
        default=0.6,
        help='IoU threshold for NMS'
    )
    parser.add_argument(
        '-b', '--batch-size',
        type=int,
        default=16,
        help='Batch size'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=None,
        help='Image size (overrides config)'
    )
    
    # Dataset split
    parser.add_argument(
        '--test',
        action='store_true',
        help='Evaluate on test set instead of validation set'
    )
    
    # Device
    parser.add_argument(
        '-d', '--device',
        type=str,
        default='',
        help='Device to use (e.g., 0, 1, cpu)'
    )
    
    # Workers
    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='Number of dataloader workers'
    )
    
    # Verbose
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    # Save results
    parser.add_argument(
        '--save-json',
        action='store_true',
        help='Save results to JSON file'
    )
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Import experiment configuration
    print(f"Loading experiment from: {args.exp_file}")
    Exp = import_exp_from_file(args.exp_file)
    exp = Exp()
    
    print("\n" + "="*70)
    print(exp)
    print("="*70 + "\n")
    
    # Get config as dict
    config = exp.to_dict()
    
    # Set evaluation mode
    config['mode'] = 'val'
    config['model'] = args.checkpoint
    
    # Override with command line arguments
    config['conf'] = args.conf
    config['iou'] = args.iou
    config['batch'] = args.batch_size
    config['workers'] = args.workers
    
    if args.imgsz is not None:
        config['imgsz'] = args.imgsz
    
    if args.device:
        config['device'] = args.device
    
    if args.test:
        config['split'] = 'test'
        print("Evaluating on TEST set")
    else:
        config['split'] = 'val'
        print("Evaluating on VALIDATION set")
    
    if args.verbose:
        config['verbose'] = True
    
    if args.save_json:
        config['save_json'] = True
    
    print(f"\nModel: {args.checkpoint}")
    print(f"Confidence threshold: {args.conf}")
    print(f"IoU threshold: {args.iou}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device if args.device else 'auto'}")
    print("\nStarting evaluation...")
    print("="*70)
    
    # Create trainer and run validation
    trainer = DistanceDetectionTrainer(overrides=config)
    
    try:
        results = trainer.val()
        
        print("\n" + "="*70)
        print("Evaluation completed successfully!")
        print("="*70)
        
        # Print results
        if hasattr(results, 'results_dict'):
            print("\nResults:")
            for key, value in results.results_dict.items():
                print(f"  {key}: {value}")
        
        return results
        
    except Exception as e:
        print("\n" + "="*70)
        print(f"Evaluation failed with error: {e}")
        print("="*70)
        raise


if __name__ == '__main__':
    main()

