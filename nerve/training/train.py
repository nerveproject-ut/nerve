#!/usr/bin/env python3
"""
Unified Training Runner for NERVE.

Single entry point for training all model types: YOLOX, YOLOv8, ReYOLOv8, and RVT.
Automatically detects the model type from the experiment configuration and
dispatches to the appropriate trainer.

Usage:
    # Train ReYOLOv8 with distance estimation
    python train.py -f experiments/templates/reyolov8_distance.py
    
    # Train YOLOv8 detection
    python train.py -f experiments/templates/yolov8_detection.py
    
    # Train YOLOX with custom batch size
    python train.py -f experiments/templates/yolox_detection.py -b 32
    
    # Resume training
    python train.py -f experiments/templates/reyolov8_distance.py --resume runs/reyolov8/exp/weights/last.pt
"""

import argparse
import sys
import os
import subprocess
from pathlib import Path
import importlib.util


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

    spec = importlib.util.spec_from_file_location("exp_module", exp_file)
    exp_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exp_module)

    if not hasattr(exp_module, 'Exp'):
        raise AttributeError(f"Experiment file must contain 'Exp' class: {exp_file}")

    return exp_module.Exp


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='NERVE Unified Training Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train ReYOLOv8 with distance estimation
    python train.py -f experiments/templates/reyolov8_distance.py
    
    # Train YOLOv8 with custom batch size and epochs
    python train.py -f experiments/templates/yolov8_detection.py -b 32 --epochs 50
    
    # Train YOLOX on specific GPU
    python train.py -f experiments/templates/yolox_detection.py --device 0
    
    # Resume training
    python train.py -f experiments/templates/reyolov8_distance.py --resume path/to/weights.pt
        """
    )
    
    # Experiment configuration (required)
    parser.add_argument(
        '-f', '--exp-file',
        type=str,
        required=True,
        help='Path to experiment configuration file'
    )
    
    # Training parameter overrides
    parser.add_argument('-b', '--batch-size', type=int, help='Batch size (overrides config)')
    parser.add_argument('--epochs', type=int, help='Number of epochs (overrides config)')
    parser.add_argument('--imgsz', type=int, help='Image size (overrides config)')
    parser.add_argument('--workers', type=int, help='Number of dataloader workers (overrides config)')
    parser.add_argument('--lr', type=float, help='Learning rate (overrides config)')
    
    # Device
    parser.add_argument('--device', type=str, default='', help='Device to use (e.g., 0, 1, cpu)')
    
    # Resume training
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume training from')
    
    # Experiment name override
    parser.add_argument('-expn', '--experiment-name', type=str, help='Experiment name (overrides config)')
    
    # Misc
    parser.add_argument('--cache', action='store_true', help='Cache images for faster training')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--fp16', action='store_true', help='Use FP16 mixed precision training')
    
    # Weights & Biases logging
    parser.add_argument('--wandb-project', type=str, help='W&B project name')
    parser.add_argument('--no-wandb', action='store_true', help='Disable W&B logging')
    
    return parser.parse_args()


def _find_yolox_dir():
    """Find the YOLOX framework directory."""
    candidates = [
        Path(__file__).parent / 'yoloX',
        Path(os.environ.get('YOLOX_DIR', '')) if os.environ.get('YOLOX_DIR') else None,
    ]
    for c in candidates:
        if c and c.is_dir() and (c / 'tools' / 'train.py').exists():
            return c

    try:
        import yolox
        pkg_dir = Path(yolox.__file__).parent.parent
        if (pkg_dir / 'tools' / 'train.py').exists():
            return pkg_dir
    except ImportError:
        pass

    return None


def train_yolox(config, args):
    """Train YOLOX model."""
    print("Dispatching to YOLOX trainer...")
    
    YOLOX_DIR = _find_yolox_dir()
    
    if YOLOX_DIR is None:
        print("\n" + "=" * 60)
        print("YOLOX FRAMEWORK NOT FOUND")
        print("=" * 60)
        print("\nYOLOX training requires the YOLOX framework to be available.")
        print("\nOption 1: Set the YOLOX_DIR environment variable:")
        print("  export YOLOX_DIR=/path/to/yoloX")
        print("\nOption 2: Place or symlink the YOLOX repo at:")
        print(f"  {Path(__file__).parent / 'yoloX'}")
        print("\nOption 3: Install YOLOX from source:")
        print("  git clone https://github.com/Megvii-BaseDetection/YOLOX.git yoloX")
        print("=" * 60 + "\n")
        raise FileNotFoundError("YOLOX framework not found. Set YOLOX_DIR or place it at nerve/training/yoloX/")
    
    print(f"Using YOLOX framework at: {YOLOX_DIR}")
    
    yolox_exp_file = config.get('yolox_exp_file')
    
    if not yolox_exp_file:
        bundled_dir = Path(__file__).parent / 'yolox_exps'
        if config.get('process_distance', False):
            yolox_exp_file = str(bundled_dir / 'yolox_distance.py')
            print(f"Auto-selected bundled YOLOX distance exp file")
        else:
            yolox_exp_file = str(bundled_dir / 'yolox_detection.py')
            print(f"Auto-selected bundled YOLOX detection exp file")
    
    # Build command line arguments for YOLOX training
    train_args = [
        sys.executable,
        str(YOLOX_DIR / 'tools' / 'train.py'),
        '-f', str(yolox_exp_file),
        '-b', str(config.get('batch_size', config.get('batch', 16))),
    ]
    
    # Add optional arguments
    if config.get('device'):
        train_args.extend(['-d', str(config['device'])])
    if config.get('fp16', False):
        train_args.append('--fp16')
    if config.get('occupy', False):
        train_args.append('-o')
    if config.get('resume'):
        train_args.extend(['-c', str(config['resume'])])
    if config.get('name') or config.get('exp_name'):
        train_args.extend(['-expn', config.get('name') or config.get('exp_name')])
    
    # Add config overrides via opts (YOLOX expects alternating key value pairs)
    # Format: key1 value1 key2 value2 ...
    opts = []
    if config.get('epochs'):
        opts.extend(['max_epoch', str(config['epochs'])])
    if config.get('eval_interval'):
        opts.extend(['eval_interval', str(config['eval_interval'])])
    if config.get('input_size'):
        input_size = config['input_size']
        if isinstance(input_size, (list, tuple)):
            # YOLOX expects tuple format as string
            opts.extend(['input_size', str(tuple(input_size))])
        else:
            opts.extend(['input_size', f"({input_size},{input_size})"])
    
    # Pass dataset path
    if config.get('dataset_path'):
        opts.extend(['data_dir', str(config['dataset_path'])])
    
    # Pass annotation file paths (important for custom annotation files like davis_radar.json)
    if config.get('train_ann'):
        opts.extend(['train_ann', str(config['train_ann'])])
    if config.get('val_ann'):
        opts.extend(['val_ann', str(config['val_ann'])])
    if config.get('test_ann'):
        opts.extend(['test_ann', str(config['test_ann'])])
    
    # Pass source folder (must match the image folder name, e.g., "davis" or "davis_radar")
    if config.get('source'):
        opts.extend(['source', str(config['source'])])
    
    # Pass training settings
    if 'warmup_epochs' in config:
        opts.extend(['warmup_epochs', str(config['warmup_epochs'])])
    if 'workers' in config:
        opts.extend(['data_num_workers', str(config['workers'])])
    
    # Pass augmentation settings
    if 'flip_prob' in config:
        opts.extend(['flip_prob', str(config['flip_prob'])])
    if 'no_aug_epochs' in config:
        opts.extend(['no_aug_epochs', str(config['no_aug_epochs'])])
    if 'mixup_prob' in config:
        opts.extend(['mixup_prob', str(config['mixup_prob'])])
    if 'hsv_prob' in config:
        opts.extend(['hsv_prob', str(config['hsv_prob'])])
    if 'use_mosaic' in config:
        opts.extend(['use_mosaic', str(config['use_mosaic'])])
    if 'mosaic_prob' in config:
        opts.extend(['mosaic_prob', str(config['mosaic_prob'])])
    if 'enable_mixup' in config:
        opts.extend(['enable_mixup', str(config['enable_mixup'])])
    
    # Pass distance-related settings
    if 'distance_loss_multiplier' in config:
        opts.extend(['distance_loss_multiplier', str(config['distance_loss_multiplier'])])
    if 'use_l1_loss' in config:
        opts.extend(['use_l1_loss', str(config['use_l1_loss'])])
    if 'include_radar' in config:
        opts.extend(['use_radar', str(config['include_radar'])])
    if 'process_distance' in config:
        opts.extend(['include_distance', str(config['process_distance'])])
    if 'distance_from_head' in config:
        opts.extend(['distance_from_head', str(config['distance_from_head'])])
    if 'min_dist' in config:
        opts.extend(['min_distance', str(config['min_dist'])])
    if 'max_dist' in config:
        opts.extend(['max_distance', str(config['max_dist'])])
    if 'nbins' in config:
        opts.extend(['nbins', str(config['nbins'])])
    
    # Pass evaluation settings (important for reducing false positives in visualizations)
    if 'test_conf' in config:
        opts.extend(['test_conf', str(config['test_conf'])])
    if 'nmsthre' in config:
        opts.extend(['nmsthre', str(config['nmsthre'])])
    
    # Pass additional YOLOX-specific settings
    if config.get('test_size'):
        test_size = config['test_size']
        if isinstance(test_size, (list, tuple)):
            opts.extend(['test_size', str(tuple(test_size))])
        else:
            opts.extend(['test_size', f"({test_size},{test_size})"])
    if 'max_labels' in config:
        opts.extend(['max_labels', str(config['max_labels'])])
    if config.get('mosaic_scale'):
        opts.extend(['mosaic_scale', str(tuple(config['mosaic_scale']))])
    if config.get('random_size'):
        opts.extend(['random_size', str(tuple(config['random_size']))])
    if 'depth' in config:
        opts.extend(['depth', str(config['depth'])])
    if 'width' in config:
        opts.extend(['width', str(config['width'])])
    if 'num_classes' in config:
        opts.extend(['num_classes', str(config['num_classes'])])
    
    # Append opts to train_args
    if opts:
        train_args.extend(opts)
    
    # Print full command for debugging
    print(f"\nExecuting YOLOX training with {len(opts)//2} config overrides:")
    for i in range(0, len(opts), 2):
        print(f"  {opts[i]} = {opts[i+1]}")
    print(f"\nCommand: {' '.join(train_args[:8])}...")
    
    # Add YOLOX to Python path and run
    env = os.environ.copy()
    pythonpath = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = f"{YOLOX_DIR}:{pythonpath}"
    
    import subprocess
    result = subprocess.run(train_args, cwd=str(YOLOX_DIR), env=env)
    
    if result.returncode != 0:
        raise RuntimeError(f"YOLOX training failed with return code {result.returncode}")
    
    # Get the actual output directory from the experiment file
    # Import the exp module to get the configured output_dir
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("exp_module", yolox_exp_file)
        exp_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(exp_module)
        exp = exp_module.Exp()
        exp_name = config.get('name') or config.get('exp_name') or exp.exp_name
        output_dir = Path(exp.output_dir) / exp_name
    except Exception:
        # Fallback to default if we can't load the exp
        output_dir = YOLOX_DIR.parent / 'runs' / config.get('name', 'yolox_exp')
    
    return output_dir


def train_yolov8(config, args):
    """Train YOLOv8 model."""
    print("Dispatching to YOLOv8 trainer...")
    
    from nerve.training.yolov8_distance_trainer import DistanceDetectionTrainer
    
    # Apply command line overrides
    if args.batch_size:
        config['batch'] = args.batch_size
    if args.epochs:
        config['epochs'] = args.epochs
    if args.imgsz:
        config['imgsz'] = args.imgsz
    if args.workers:
        config['workers'] = args.workers
    if args.lr:
        config['lr0'] = args.lr
    if args.device:
        config['device'] = args.device
    if args.experiment_name:
        config['name'] = args.experiment_name
    if args.cache:
        config['cache'] = True
    if args.fp16:
        config['amp'] = True
    if args.resume:
        config['resume'] = args.resume
    if args.verbose:
        config['verbose'] = True
    
    # Print config overrides being passed
    print(f"\nYOLOv8 training with config overrides:")
    important_keys = ['data', 'epochs', 'batch', 'imgsz', 'lr0', 'process_distance', 
                      'distance_loss_multiplier', 'include_radar', 'project', 'name']
    for key in important_keys:
        if key in config:
            print(f"  {key} = {config[key]}")
    
    # Create trainer and train
    trainer = DistanceDetectionTrainer(overrides=config)
    trainer.train()
    
    return trainer.save_dir


def train_reyolov8(config, args):
    """Train ReYOLOv8 model."""
    print("Dispatching to ReYOLOv8 trainer...")
    
    # Apply command line overrides
    if args.batch_size:
        config['batch'] = args.batch_size
    if args.epochs:
        config['epochs'] = args.epochs
    if args.imgsz:
        config['imgsz'] = args.imgsz
    if args.workers:
        config['workers'] = args.workers
    if args.lr:
        config['lr0'] = args.lr
    if args.device:
        config['device'] = args.device
    if args.experiment_name:
        config['name'] = args.experiment_name
    if args.resume:
        config['resume'] = args.resume
    if args.verbose:
        config['verbose'] = True
    
    # ReYOLOv8's train.py runs training at module level, so we use subprocess
    # This works for both distance and non-distance modes
    import subprocess
    from pathlib import Path
    
    REYOLOV8_DIR = Path(__file__).parent / 'reyolov8'
    
    # Build command line arguments - pass all config settings
    cmd_args = [
        sys.executable,  # Python interpreter
        str(REYOLOV8_DIR / 'train.py'),
        '--data', str(config.get('data_yaml', config.get('data', ''))),
        '--model', str(config.get('model', config.get('model_yaml', str(REYOLOV8_DIR / 'ultralytics/models/v8/Recurrent/ReYOLOv8n.yaml')))),
        '--epochs', str(config.get('epochs', 100)),
        '--batch', str(config.get('batch', 8)),
        '--imgsz', str(config.get('imgsz', 384)),
        '--channels', str(config.get('channels', 10)),
        '--clip_length', str(config.get('clip_length', 11)),
        '--clip_stride', str(config.get('clip_stride', 11)),
    ]
    
    # Pass pretrained weights if specified
    if config.get('pretrained'):
        cmd_args.extend(['--weights', str(config['pretrained'])])
    
    # Pass training parameters
    if config.get('seed') is not None:
        cmd_args.extend(['--seed', str(config['seed'])])
    if config.get('save_period'):
        cmd_args.extend(['--save_period', str(config['save_period'])])
    if config.get('nbs'):
        cmd_args.extend(['--nbs', str(config['nbs'])])
    if config.get('optimizer'):
        cmd_args.extend(['--optimizer', str(config['optimizer'])])
    
    # Check if distance estimation is enabled
    if config.get('process_distance', False):
        print("ReYOLOv8 with distance estimation...")
        cmd_args.extend([
            '--distance',  # Enable distance estimation
            '--nbins', str(config.get('nbins', 100)),
            '--min_dist', str(config.get('min_dist', 0.0)),
            '--max_dist', str(config.get('max_dist', 10.0)),
            '--dist_loss_mult', str(config.get('distance_loss_multiplier', 1.0)),
        ])
    else:
        print("ReYOLOv8 standard detection...")
    
    # Add optional arguments
    if config.get('name'):
        cmd_args.extend(['--name', config['name']])
    if config.get('project'):
        cmd_args.extend(['--project', str(config['project'])])
    if config.get('device'):
        cmd_args.extend(['--device', str(config['device'])])
    if config.get('workers'):
        cmd_args.extend(['--workers', str(config['workers'])])
    if config.get('resume'):
        cmd_args.extend(['--resume'])
    if config.get('val_epoch'):
        cmd_args.extend(['--val_epoch', str(config['val_epoch'])])
    
    # Event augmentation parameters - always pass them (use default 0 if not set)
    cmd_args.extend(['--flip', str(config.get('flip', 0.0))])
    cmd_args.extend(['--suppress', str(config.get('suppress', 0.0))])
    cmd_args.extend(['--invert', str(config.get('invert', 0.0))])
    cmd_args.extend(['--positive', str(config.get('positive', 0.0))])
    cmd_args.extend(['--zoom_out', str(config.get('zoom_out', 0.0))])
    
    # Channel selection (for ignoring radar channel)
    if config.get('select_channels') is not None:
        cmd_args.extend(['--select_channels', str(config['select_channels'])])
    
    # Cosine LR scheduler (boolean flag)
    if config.get('cos_lr', False):
        cmd_args.append('--cos_lr')
    
    # Learning rate parameters (pass only if set in config)
    if config.get('lr0') is not None:
        cmd_args.extend(['--lr0', str(config['lr0'])])
    if config.get('lrf') is not None:
        cmd_args.extend(['--lrf', str(config['lrf'])])
    if config.get('momentum') is not None:
        cmd_args.extend(['--momentum', str(config['momentum'])])
    if config.get('weight_decay') is not None:
        cmd_args.extend(['--weight_decay', str(config['weight_decay'])])
    if config.get('warmup_epochs') is not None:
        cmd_args.extend(['--warmup_epochs', str(config['warmup_epochs'])])
    if config.get('warmup_momentum') is not None:
        cmd_args.extend(['--warmup_momentum', str(config['warmup_momentum'])])
    if config.get('warmup_bias_lr') is not None:
        cmd_args.extend(['--warmup_bias_lr', str(config['warmup_bias_lr'])])
    
    # Loss gain parameters (pass only if set in config)
    if config.get('box') is not None:
        cmd_args.extend(['--box', str(config['box'])])
    if config.get('cls') is not None:
        cmd_args.extend(['--cls', str(config['cls'])])
    if config.get('dfl') is not None:
        cmd_args.extend(['--dfl', str(config['dfl'])])
    
    # Validation parameters (pass only if set in config)
    if config.get('conf') is not None:
        cmd_args.extend(['--conf', str(config['conf'])])
    if config.get('iou') is not None:
        cmd_args.extend(['--iou', str(config['iou'])])
    if config.get('max_det') is not None:
        cmd_args.extend(['--max_det', str(config['max_det'])])
    
    # Print config overrides being passed
    print(f"\nReYOLOv8 training with config overrides:")
    # Extract key=value pairs from cmd_args for display
    i = 2  # Skip python and script path
    while i < len(cmd_args):
        if cmd_args[i].startswith('--'):
            key = cmd_args[i][2:]
            if i + 1 < len(cmd_args) and not cmd_args[i + 1].startswith('--'):
                print(f"  {key} = {cmd_args[i + 1]}")
                i += 2
            else:
                print(f"  {key} = True")
                i += 1
        else:
            i += 1
    
    print(f"\nExecuting: {' '.join(cmd_args[:8])}...")
    
    # Execute the training script
    result = subprocess.run(cmd_args, cwd=str(REYOLOV8_DIR))
    
    if result.returncode != 0:
        raise RuntimeError(f"ReYOLOv8 training failed with return code {result.returncode}")
    
    # Return the results directory
    project = config.get('project', REYOLOV8_DIR / 'runs/train')
    name = config.get('name', 'exp')
    return Path(project) / name


def train_rvt(config, args):
    """Train RVT (Recurrent Vision Transformer) model."""
    print("Dispatching to RVT trainer...")
    
    RVT_DIR = Path(__file__).parent / 'rvt'
    
    # Apply command line overrides to config before generating Hydra config
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.epochs:
        config['max_epochs'] = args.epochs
        config['epochs'] = args.epochs
    if args.imgsz:
        # Update resolution
        config['resolution_hw'] = [args.imgsz, int(args.imgsz * 1.27)]  # Maintain aspect ratio
    if args.workers:
        config['workers'] = args.workers
    if args.lr:
        config['lr0'] = args.lr
    if args.device:
        config['gpus'] = int(args.device) if args.device.isdigit() else 0
    if args.experiment_name:
        config['name'] = args.experiment_name
    if args.verbose:
        config['verbose'] = True
    
    # RVT uses Hydra for configuration. We generate Hydra override strings
    # to pass experiment config values to RVT's native config system.
    # This approach uses RVT's existing config files as base and overrides specific values.
    
    # Build Hydra command-line overrides from config
    hydra_overrides = []
    
    # Training parameters
    if config.get('max_epochs') is not None:
        hydra_overrides.append(f'training.max_epochs={config["max_epochs"]}')
    if config.get('max_steps') is not None:
        hydra_overrides.append(f'training.max_steps={config["max_steps"]}')
    if config.get('lr0') is not None:
        hydra_overrides.append(f'training.learning_rate={config["lr0"]}')
    if config.get('weight_decay') is not None:
        hydra_overrides.append(f'training.weight_decay={config["weight_decay"]}')
    if config.get('precision') is not None:
        hydra_overrides.append(f'training.precision={config["precision"]}')
    if config.get('gradient_clip_val') is not None:
        hydra_overrides.append(f'training.gradient_clip_val={config["gradient_clip_val"]}')
    if config.get('limit_train_batches') is not None:
        hydra_overrides.append(f'training.limit_train_batches={config["limit_train_batches"]}')
    
    # LR scheduler
    if config.get('use_lr_scheduler') is not None:
        hydra_overrides.append(f'training.lr_scheduler.use={str(config["use_lr_scheduler"]).lower()}')
    if config.get('lr_scheduler_pct_start') is not None:
        hydra_overrides.append(f'training.lr_scheduler.pct_start={config["lr_scheduler_pct_start"]}')
    if config.get('lr_scheduler_div_factor') is not None:
        hydra_overrides.append(f'training.lr_scheduler.div_factor={config["lr_scheduler_div_factor"]}')
    if config.get('lr_scheduler_final_div_factor') is not None:
        hydra_overrides.append(f'training.lr_scheduler.final_div_factor={config["lr_scheduler_final_div_factor"]}')
    
    # Validation - RVT requires either val_check_interval OR check_val_every_n_epoch, not both
    if config.get('limit_val_batches') is not None:
        hydra_overrides.append(f'validation.limit_val_batches={config["limit_val_batches"]}')
    if config.get('val_check_interval') is not None:
        hydra_overrides.append(f'validation.val_check_interval={config["val_check_interval"]}')
        # Must disable epoch-based validation when using step-based
        hydra_overrides.append('validation.check_val_every_n_epoch=null')
    elif config.get('check_val_every_n_epoch') is not None:
        hydra_overrides.append(f'validation.check_val_every_n_epoch={config["check_val_every_n_epoch"]}')
        # Must disable step-based validation when using epoch-based
        hydra_overrides.append('validation.val_check_interval=null')
    
    # Batch size
    if config.get('batch_size') is not None:
        hydra_overrides.append(f'batch_size.train={config["batch_size"]}')
        hydra_overrides.append(f'batch_size.eval={config["batch_size"]}')
    
    # Hardware
    if config.get('workers') is not None:
        hydra_overrides.append(f'hardware.num_workers.train={config["workers"]}')
    if config.get('workers_eval') is not None:
        hydra_overrides.append(f'hardware.num_workers.eval={config["workers_eval"]}')
    if config.get('gpus') is not None:
        gpus = config['gpus']
        if isinstance(gpus, list):
            hydra_overrides.append(f'hardware.gpus=[{",".join(map(str, gpus))}]')
        else:
            hydra_overrides.append(f'hardware.gpus={gpus}')
    
    # Reproducibility
    if config.get('seed') is not None and config['seed'] != 0:
        hydra_overrides.append(f'reproduce.seed_everything={config["seed"]}')
    if config.get('deterministic') is not None:
        hydra_overrides.append(f'reproduce.deterministic_flag={str(config["deterministic"]).lower()}')
    
    # Logging
    if config.get('ckpt_every_n_epochs') is not None:
        hydra_overrides.append(f'logging.ckpt_every_n_epochs={config["ckpt_every_n_epochs"]}')
    if config.get('log_every_n_steps') is not None:
        hydra_overrides.append(f'logging.train.log_every_n_steps={config["log_every_n_steps"]}')
    if config.get('log_model_every_n_steps') is not None:
        hydra_overrides.append(f'logging.train.log_model_every_n_steps={config["log_model_every_n_steps"]}')
    
    # WandB
    if config.get('wandb_group') is not None:
        hydra_overrides.append(f'wandb.group_name={config["wandb_group"]}')
    if config.get('wandb_project') is not None:
        hydra_overrides.append(f'wandb.project_name={config["wandb_project"]}')
    
    # Dataset parameters
    if config.get('dataset_path') is not None:
        hydra_overrides.append(f'dataset.path={config["dataset_path"]}')
    if config.get('ev_repr_name') is not None:
        # Quote the value to handle special characters like '=' in event representation names
        hydra_overrides.append(f"dataset.ev_repr_name='{config['ev_repr_name']}'")
    if config.get('sequence_length') is not None:
        hydra_overrides.append(f'dataset.sequence_length={config["sequence_length"]}')
    if config.get('resolution_hw') is not None:
        res = config['resolution_hw']
        hydra_overrides.append(f'dataset.resolution_hw=[{res[0]},{res[1]}]')
    if config.get('downsample_by_factor_2') is not None:
        hydra_overrides.append(f'dataset.downsample_by_factor_2={str(config["downsample_by_factor_2"]).lower()}')
    
    # Channel selection (for ignoring radar channel from dataset)
    if config.get('select_channels') is not None:
        hydra_overrides.append(f'++dataset.select_channels={config["select_channels"]}')
    
    # Data augmentation
    if config.get('prob_hflip') is not None:
        hydra_overrides.append(f'dataset.data_augmentation.random.prob_hflip={config["prob_hflip"]}')
        hydra_overrides.append(f'dataset.data_augmentation.stream.prob_hflip={config["prob_hflip"]}')
    if config.get('rotate_prob') is not None:
        hydra_overrides.append(f'dataset.data_augmentation.random.rotate.prob={config["rotate_prob"]}')
        hydra_overrides.append(f'dataset.data_augmentation.stream.rotate.prob={config["rotate_prob"]}')
    if config.get('zoom_prob') is not None:
        hydra_overrides.append(f'dataset.data_augmentation.random.zoom.prob={config["zoom_prob"]}')
        # Stream zoom is typically lower
        stream_zoom = config['zoom_prob'] * 0.6
        hydra_overrides.append(f'dataset.data_augmentation.stream.zoom.prob={stream_zoom}')
    
    # Model parameters
    # Now that +model/maxvit_yolox=default loads the full config, we can override input_channels
    if config.get('backbone_in_channels') is not None:
        hydra_overrides.append(f'model.backbone.input_channels={config["backbone_in_channels"]}')
    
    # Model size parameters (for RVT-Tiny/Small/Base scaling)
    # embed_dim: 32=Tiny, 48=Small, 64=Base
    if config.get('backbone_dim') is not None:
        hydra_overrides.append(f'model.backbone.embed_dim={config["backbone_dim"]}')
    # FPN depth: 0.33=Tiny/Small, 0.67=Base (controls number of bottleneck blocks)
    if config.get('fpn_depth') is not None:
        hydra_overrides.append(f'model.fpn.depth={config["fpn_depth"]}')
    
    # NOTE: model.head.num_classes is handled by RVT's modifier.py dynamically for PEGMA dataset
    # Only override postprocess settings which exist in RVT's base config
    if config.get('confidence_threshold') is not None:
        hydra_overrides.append(f'model.postprocess.confidence_threshold={config["confidence_threshold"]}')
    if config.get('nms_threshold') is not None:
        hydra_overrides.append(f'model.postprocess.nms_threshold={config["nms_threshold"]}')
    
    # Distance estimation (PEGMA extension) - use ++ prefix for add-or-override
    if config.get('process_distance'):
        # Enable distance in model config
        hydra_overrides.append('++model.distance.enable=true')
        if config.get('nbins') is not None:
            hydra_overrides.append(f'++model.distance.nbins={config["nbins"]}')
        if config.get('min_dist') is not None:
            hydra_overrides.append(f'++model.distance.min_dist={config["min_dist"]}')
        if config.get('max_dist') is not None:
            hydra_overrides.append(f'++model.distance.max_dist={config["max_dist"]}')
        if config.get('distance_loss_multiplier') is not None:
            hydra_overrides.append(f'++model.distance.loss_multiplier={config["distance_loss_multiplier"]}')
        # Enable distance in dataset config (so distance labels are loaded)
        hydra_overrides.append('++dataset.has_distance=true')
    
    # Build command line arguments for RVT training
    # Use RVT's native config directory with Hydra overrides
    train_args = [
        sys.executable,
        str(RVT_DIR / 'train.py'),
        'dataset=pegma',  # Use the pegma dataset config we created
        '+model/maxvit_yolox=default',  # Append MaxViTRNN backbone config to defaults
    ]
    
    # Add all Hydra overrides
    train_args.extend(hydra_overrides)
    
    # Disable wandb if requested
    if args.no_wandb:
        # RVT doesn't have a direct wandb disable, but we can set env var
        os.environ['WANDB_MODE'] = 'disabled'
    
    # Print key RVT config settings being used
    print(f"\nRVT training with config settings:")
    print(f"  dataset_path = {config.get('dataset_path', 'N/A')}")
    print(f"  max_epochs = {config.get('max_epochs', 100)}")
    print(f"  batch_size = {config.get('batch_size', 8)}")
    print(f"  sequence_length = {config.get('sequence_length', 11)}")
    print(f"  resolution_hw = {config.get('resolution_hw', [240, 304])}")
    print(f"  learning_rate = {config.get('lr0', 0.0002)}")
    print(f"  process_distance = {config.get('process_distance', False)}")
    print(f"  distance_loss_multiplier = {config.get('distance_loss_multiplier', 1.0)}")
    print(f"  wandb_project = {config.get('wandb_project', 'RVT')}")
    
    print(f"\nExecuting RVT training...")
    print(f"Hydra overrides: {len(hydra_overrides)} settings")
    
    # Add RVT to Python path and run
    env = os.environ.copy()
    pythonpath = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = f"{RVT_DIR}:{pythonpath}"
    
    result = subprocess.run(train_args, cwd=str(RVT_DIR), env=env)
    
    if result.returncode != 0:
        raise RuntimeError(f"RVT training failed with return code {result.returncode}")
    
    # Note: RVT saves checkpoints to its own WandB-based directory structure:
    # <RVT_DIR>/<wandb_project>/<wandb_run_id>/checkpoints/
    # The actual paths are printed by RVT's train.py at the end of training.
    # Return None to avoid printing a misleading path.
    return None


def main():
    """Main training function."""
    args = parse_args()
    
    # Configure Weights & Biases
    if args.no_wandb:
        os.environ['WANDB_MODE'] = 'disabled'
        print("Weights & Biases logging disabled")
    elif args.wandb_project:
        os.environ['WANDB_PROJECT'] = args.wandb_project
    
    # Import experiment configuration
    print("=" * 70)
    print("NERVE Unified Training Runner")
    print("=" * 70)
    print(f"\nLoading experiment from: {args.exp_file}")
    
    Exp = import_exp_from_file(args.exp_file)
    exp = Exp()
    
    print("\n" + str(exp))
    
    # Get config as dict
    config = exp.to_dict()
    model_type = config.get('model_type', 'unknown')
    
    print(f"\nModel Type: {model_type.upper()}")
    print("=" * 70)
    
    # Dispatch to appropriate trainer
    try:
        if model_type == 'yolox':
            result_dir = train_yolox(config, args)
        elif model_type == 'yolov8':
            result_dir = train_yolov8(config, args)
        elif model_type == 'reyolov8':
            result_dir = train_reyolov8(config, args)
        elif model_type == 'rvt':
            result_dir = train_rvt(config, args)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        print("\n" + "=" * 70)
        print("Training completed successfully!")
        if result_dir:
            print(f"Results saved to: {result_dir}")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n" + "=" * 70)
        print("Training interrupted by user")
        print("=" * 70)
    except Exception as e:
        print("\n" + "=" * 70)
        print(f"Training failed with error: {e}")
        print("=" * 70)
        raise


if __name__ == '__main__':
    main()

