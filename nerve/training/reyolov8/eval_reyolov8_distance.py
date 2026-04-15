"""
REYOLOv8 Distance Estimation Evaluation Script

Standalone evaluation for REYOLOv8 models with distance estimation on event camera
datasets. Loads a trained checkpoint, builds the validation dataloader, and runs
full detection + distance evaluation using EventVideoDistanceValidator.

Usage:
    # Evaluate with experiment config (validation set)
    python eval_reyolov8_distance.py -f ../experiments/templates/reyolov8_distance.py \
        -c runs/train/exp/weights/best.pt

    # Evaluate on test set
    python eval_reyolov8_distance.py -f ../experiments/templates/reyolov8_distance.py \
        -c runs/train/exp/weights/best.pt --split test

    # Custom thresholds and batch size
    python eval_reyolov8_distance.py -f ../experiments/templates/reyolov8_distance.py \
        -c runs/train/exp/weights/best.pt --conf 0.001 --iou 0.6 -b 32

    # Sequential mode (full hidden-state continuity across clips)
    python eval_reyolov8_distance.py -f ../experiments/templates/reyolov8_distance.py \
        -c runs/train/exp/weights/best.pt --sequential

    # Save predictions to JSON
    python eval_reyolov8_distance.py -f ../experiments/templates/reyolov8_distance.py \
        -c runs/train/exp/weights/best.pt --save-json
"""

import sys
import os
import argparse
from pathlib import Path
import importlib.util
import json

REYOLOV8_DIR = Path(__file__).parent
if str(REYOLOV8_DIR) not in sys.path:
    sys.path.insert(0, str(REYOLOV8_DIR))

import torch
import numpy as np

from ultralytics.yolo.utils import DEFAULT_CFG, LOGGER
from ultralytics.yolo.utils.files import increment_path
from ultralytics.yolo.cfg import get_cfg

from EventVideoDataloader import build_video_val_standalone_dataloader
import val_distance


def load_exp_config(exp_file):
    """
    Load experiment configuration from a Python file.

    The file must define an ``Exp`` class whose attributes (or ``to_dict()``
    return value) describe the experiment.
    """
    exp_file = Path(exp_file)
    if not exp_file.exists():
        raise FileNotFoundError(f"Experiment file not found: {exp_file}")

    spec = importlib.util.spec_from_file_location("exp_module", exp_file)
    exp_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exp_module)

    if not hasattr(exp_module, 'Exp'):
        raise AttributeError(f"Experiment file must define 'Exp' class: {exp_file}")

    exp = exp_module.Exp()
    LOGGER.info(f"Loaded experiment config: {exp.exp_name}")
    return exp


def load_model(checkpoint_path, config):
    """
    Load a REYOLOv8 model (optionally with distance head) from *checkpoint_path*.

    Returns the model on CPU — the caller is responsible for moving it to the
    target device.
    """
    from custom_reyolov8_distance import ReYOLOv8_WithDistance
    from ultralytics.nn.tasks import DetectionModel2

    LOGGER.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Build model architecture
    if config.get('process_distance') or config.get('distance'):
        model = ReYOLOv8_WithDistance.create_model(
            model_cfg=config['model'],
            nc=config.get('nc', 1),
            nbins=config.get('nbins', 100),
            min_dist=config.get('min_dist', 0.0),
            max_dist=config.get('max_dist', 10.0),
            channels=config.get('channels', 5),
        )
    else:
        model = DetectionModel2(
            cfg=config['model'],
            ch=config.get('channels', 5),
            nc=config.get('nc', 1),
        )

    # Extract state dict from various checkpoint layouts
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
    else:
        state_dict = checkpoint

    if hasattr(state_dict, 'float'):
        state_dict = state_dict.float().state_dict()
    elif hasattr(state_dict, 'state_dict'):
        state_dict = state_dict.state_dict()

    # Tolerant load: skip shape-mismatched keys so that, e.g., a model
    # trained with a different number of classes still loads what it can.
    model_sd = model.state_dict()
    filtered = {k: v for k, v in state_dict.items()
                if k in model_sd and v.shape == model_sd[k].shape}
    skipped = len(state_dict) - len(filtered)
    if skipped:
        LOGGER.info(f"Skipped {skipped} keys due to shape mismatch")

    model.load_state_dict(filtered, strict=False)
    LOGGER.info(f"Model loaded: {len(filtered)}/{len(model_sd)} parameters transferred")
    return model


def build_eval_args(exp_config, cli_args):
    """
    Merge the experiment config dict with CLI overrides into a Namespace that
    the Ultralytics machinery expects.
    """
    overrides = {}
    overrides['model'] = exp_config.get('model', 'yolov8n.yaml')
    overrides['data'] = exp_config.get('data_yaml', exp_config.get('data', ''))
    overrides['imgsz'] = cli_args.imgsz or exp_config.get('imgsz', 320)
    overrides['batch'] = cli_args.batch_size
    overrides['device'] = cli_args.device
    overrides['workers'] = cli_args.workers
    overrides['half'] = cli_args.half
    overrides['plots'] = cli_args.plots
    overrides['verbose'] = cli_args.verbose
    overrides['project'] = str(cli_args.project)
    overrides['name'] = cli_args.name
    overrides['exist_ok'] = cli_args.exist_ok
    overrides['split'] = cli_args.split
    overrides['save_json'] = cli_args.save_json

    if cli_args.conf is not None:
        overrides['conf'] = cli_args.conf
    if cli_args.iou is not None:
        overrides['iou'] = cli_args.iou
    if cli_args.max_det is not None:
        overrides['max_det'] = cli_args.max_det

    # Video / recurrent settings
    overrides['channels'] = exp_config.get('channels', 5)
    overrides['clip_length'] = exp_config.get('clip_length', 11)
    overrides['clip_stride'] = exp_config.get('clip_stride', 11)
    overrides['select_channels'] = exp_config.get('select_channels', None)

    # Distance settings
    overrides['distance'] = exp_config.get('process_distance', False) or exp_config.get('distance', False)
    overrides['nbins'] = exp_config.get('nbins', 100)
    overrides['min_dist'] = exp_config.get('min_dist', 0.0)
    overrides['max_dist'] = exp_config.get('max_dist', 10.0)

    args = get_cfg(DEFAULT_CFG, overrides)
    return args


def parse_args():
    parser = argparse.ArgumentParser(
        description='REYOLOv8 Distance Evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('-f', '--exp_file', type=str, required=True,
                        help='Experiment config file (.py with Exp class)')
    parser.add_argument('-c', '--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pt)')

    parser.add_argument('--conf', type=float, default=None,
                        help='Confidence threshold (default: use ultralytics default 0.001)')
    parser.add_argument('--iou', type=float, default=None,
                        help='IoU threshold for NMS')
    parser.add_argument('-b', '--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--imgsz', type=int, default=None,
                        help='Image size (overrides config)')
    parser.add_argument('--max_det', type=int, default=None,
                        help='Maximum detections per image')

    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'],
                        help='Dataset split to evaluate on')
    parser.add_argument('--sequential', action='store_true',
                        help='Use sequential mode (full hidden-state continuity, batch_size=1)')

    parser.add_argument('-d', '--device', type=str, default='0',
                        help='Device (e.g. 0, 0,1, or cpu)')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of dataloader workers')
    parser.add_argument('--half', action='store_true',
                        help='Use FP16 half-precision inference')

    parser.add_argument('--project', default=REYOLOV8_DIR / 'runs' / 'eval',
                        help='Save results under project/name')
    parser.add_argument('--name', default='exp', help='Experiment name')
    parser.add_argument('--exist_ok', action='store_true',
                        help='Allow overwriting existing project/name directory')

    parser.add_argument('--save-json', action='store_true',
                        help='Save predictions to JSON')
    parser.add_argument('--plots', action='store_true',
                        help='Generate validation plots')
    parser.add_argument('--verbose', action='store_true',
                        help='Per-class result table')

    return parser.parse_args()


def main():
    cli_args = parse_args()

    # ── Load experiment config ──────────────────────────────────────────
    exp = load_exp_config(cli_args.exp_file)
    exp_dict = exp.to_dict() if hasattr(exp, 'to_dict') else vars(exp)

    # ── Build ultralytics args ──────────────────────────────────────────
    args = build_eval_args(exp_dict, cli_args)

    # ── Print banner ────────────────────────────────────────────────────
    is_distance = getattr(args, 'distance', False)
    print("\n" + "=" * 70)
    print("REYOLOv8 Distance Evaluation" if is_distance else "REYOLOv8 Evaluation")
    print("=" * 70)
    print(f"  Experiment:       {exp.exp_name}")
    print(f"  Checkpoint:       {cli_args.checkpoint}")
    print(f"  Data config:      {args.data}")
    print(f"  Split:            {cli_args.split}")
    print(f"  Batch size:       {cli_args.batch_size}")
    print(f"  Image size:       {args.imgsz}")
    print(f"  Channels:         {args.channels}")
    print(f"  Clip length:      {args.clip_length}")
    print(f"  Device:           {args.device}")
    print(f"  Half precision:   {args.half}")
    if is_distance:
        print(f"  Distance range:   [{args.min_dist}, {args.max_dist}] m")
        print(f"  Distance bins:    {args.nbins}")
    print("=" * 70 + "\n")

    # ── Resolve save dir ────────────────────────────────────────────────
    save_dir = Path(
        increment_path(Path(args.project) / args.name,
                       exist_ok=args.exist_ok))
    save_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info(f"Results will be saved to: {save_dir}")

    # ── Load model ──────────────────────────────────────────────────────
    model = load_model(cli_args.checkpoint, exp_dict)

    # ── Build video config ──────────────────────────────────────────────
    video_config = {
        'clip_length': args.clip_length,
        'clip_stride': getattr(args, 'clip_stride', args.clip_length),
        'channels': args.channels,
    }

    # ── Build dataloader ────────────────────────────────────────────────
    from ultralytics.yolo.data.utils import check_det_dataset
    data = check_det_dataset(args.data)
    dataset_path = data.get(cli_args.split)
    if dataset_path is None:
        fallback = 'val' if cli_args.split == 'test' else 'test'
        dataset_path = data.get(fallback)
        if dataset_path is None:
            raise FileNotFoundError(
                f"Neither '{cli_args.split}' nor '{fallback}' split found in {args.data}")
        LOGGER.warning(f"Split '{cli_args.split}' not found, falling back to '{fallback}'")

    load_mode = 'sequential' if cli_args.sequential else 'batched'
    LOGGER.info(f"Building dataloader for '{cli_args.split}' split (mode={load_mode}) ...")

    select_channels = getattr(args, 'select_channels', None)
    dataloader = build_video_val_standalone_dataloader(
        args, video_config, cli_args.batch_size, dataset_path,
        rank=-1, mode=load_mode, select_channels=select_channels,
    )[0]
    LOGGER.info(f"Dataloader ready: {len(dataloader)} batches")

    # ── Create validator and run ────────────────────────────────────────
    if is_distance:
        validator = val_distance.EventVideoDistanceValidator(
            video_config=video_config,
            dataloader=dataloader,
            save_dir=save_dir,
            logger=LOGGER,
            args=args,
        )
    else:
        import val as val_base
        validator = val_base.EventVideoDetectionValidator(
            video_config=video_config,
            dataloader=dataloader,
            save_dir=save_dir,
            pbar=None,
            logger=LOGGER,
            args=args,
        )

    LOGGER.info("Starting evaluation ...")
    metrics = validator(model=cli_args.checkpoint)

    # ── Print results ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Evaluation Results")
    print("=" * 70)

    det_metrics = {k: v for k, v in metrics.items() if not k.startswith('distance/')}
    dist_metrics = {k: v for k, v in metrics.items() if k.startswith('distance/')}

    if det_metrics:
        print("\n  Detection Metrics:")
        for k, v in det_metrics.items():
            print(f"    {k:30s}  {v:.5f}" if isinstance(v, float) else f"    {k:30s}  {v}")

    if dist_metrics:
        print("\n  Distance Metrics:")
        for k, v in dist_metrics.items():
            if 'Acc' in k:
                print(f"    {k:30s}  {v * 100:.1f}%")
            elif isinstance(v, float):
                print(f"    {k:30s}  {v:.4f} m")
            else:
                print(f"    {k:30s}  {v}")

    print(f"\n  Results saved to: {save_dir}")
    print("=" * 70 + "\n")

    # ── Optionally save full metrics to JSON ────────────────────────────
    if cli_args.save_json:
        json_path = save_dir / 'eval_results.json'
        serialisable = {k: (float(v) if isinstance(v, (np.floating, float)) else
                            int(v) if isinstance(v, (np.integer, int)) else v)
                        for k, v in metrics.items()}
        with open(json_path, 'w') as f:
            json.dump(serialisable, f, indent=2)
        LOGGER.info(f"Full metrics saved to: {json_path}")

    return metrics


if __name__ == '__main__':
    main()
