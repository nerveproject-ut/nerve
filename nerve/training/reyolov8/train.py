import sys 
import torch
import numpy as np 
import os
import random
import copy
import torch
import subprocess
import torch.nn as nn
import time
from collections import defaultdict
from copy import deepcopy
from copy import copy
import wandb
import torch.nn as nn
from datetime import datetime

from ultralytics.yolo.data.utils import check_det_dataset
from ultralytics.yolo.utils.checks import check_file, check_imgsz, print_args
from ultralytics.yolo.utils.dist import ddp_cleanup, generate_ddp_command
from ultralytics.yolo.utils.files import get_latest_run, increment_path
from EventVideoDataloader import build_video_dataloader, build_video_val_standalone_dataloader
from ultralytics.yolo.utils import LOGGER, colorstr
from ultralytics.yolo.data.utils import  PIN_MEMORY, RANK
from ultralytics.nn.tasks import DetectionModel2
from ultralytics.yolo import v8

from ultralytics.yolo.engine.trainer import BaseTrainer
from ultralytics.yolo.utils import DEFAULT_CFG, RANK, colorstr
from ultralytics.yolo.utils.loss import BboxLoss
from ultralytics.yolo.utils.ops import xywh2xyxy
from ultralytics.yolo.utils.plotting import plot_images, plot_results
from ultralytics.yolo.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors
from ultralytics.yolo.utils.torch_utils import de_parallel
import numpy as np
import val

# Import shared visualization module for standardized outputs
import sys
try:
    from visualization import (
        StandardizedCSVLogger,
        plot_training_batch,
        plot_validation_batch,
        plot_results as plot_unified_results,
        plot_pr_curve,
        plot_f1_curve,
        ConfusionMatrix as UnifiedConfusionMatrix
    )
    UNIFIED_VIZ_AVAILABLE = True
except ImportError:
    UNIFIED_VIZ_AVAILABLE = False
    print("Warning: Unified visualization module not available. Using default ultralytics plotting.")
from tqdm import tqdm
from ultralytics.yolo.utils import (DEFAULT_CFG, LOGGER, RANK, SETTINGS, TQDM_BAR_FORMAT, __version__, callbacks,
                                    colorstr, emojis, yaml_save)
import torch.distributed as dist
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import lr_scheduler
from tqdm import tqdm
from ultralytics.yolo.utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, init_seeds, one_cycle,
                                                select_device, strip_optimizer)
from ultralytics.nn.tasks import attempt_load_one_weight, attempt_load_weights
from ultralytics.yolo.cfg import get_cfg
import argparse 
import yaml


from pathlib import Path
import math
######################### ADDING THE ARG PARSE ##############################################
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'yolov8n.pt', help='initial weights path')
    parser.add_argument('--model', type=str, default=ROOT / 'ultralytics/models/v8/Recurrent/ReYOLOv8n.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--nbs', type=int, help='nominal batch size', default = 16)
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=320, help='train, val image size (pixels)',nargs='+')
    parser.add_argument('--seed',type=int, default=0, help='random seed for reproducibility')
    parser.add_argument('--save_period',type=int, default=-1, help='save checkpoint every x epochs, disabled if -1')
    parser.add_argument('--save',action='store_false', help='save train checkpoints and predict results')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--cos_lr', action='store_true', help='cosine LR scheduler')
    parser.add_argument('--half', action='store_true', help='use FP16 format')
    parser.add_argument('--plots', action='store_false', help='plot results')
    parser.add_argument('--pretrained', action='store_true', help='use pretrained model')
    # Hyperparameters 
    parser.add_argument('--hyp', type=str, default= ROOT / 'ultralytics/yolo/cfg/default.yaml', help='hyperparameters path')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    # Video Hyperparameters
    parser.add_argument('--clip_length', type=int, default=11)
    parser.add_argument('--clip_stride', type=int, default=11)
    parser.add_argument('--channels',  type=int, default=1)  
    parser.add_argument('--val_epoch',  type=int, default=1)
    parser.add_argument('--select_channels', type=int, default=None, help='Select first N channels from data (None=all)')  
    # Augmentation Hyperparameters
    parser.add_argument('--flip', type=float, default=0.0)
    parser.add_argument('--invert',  type=float, default=0.0)  
    parser.add_argument('--suppress',  type=float, default=0.0)  
    parser.add_argument('--positive',  type=float, default=0.0)  
    parser.add_argument('--zoom_out',  type=float, default=0.0)  
    parser.add_argument('--max_zoom_out_factor',  type=float, default=2.0)  
    parser.add_argument('--min_zoom_out_factor',  type=float, default=1.0)  
    # Distance estimation arguments
    parser.add_argument('--distance', action='store_true', help='Enable distance estimation')
    parser.add_argument('--nbins', type=int, default=100, help='Number of distance bins')
    parser.add_argument('--min_dist', type=float, default=0.0, help='Minimum distance (meters)')
    parser.add_argument('--max_dist', type=float, default=10.0, help='Maximum distance (meters)')
    parser.add_argument('--dist_loss_mult', type=float, default=1.0, help='Distance loss multiplier')
    
    # Learning rate arguments (to override hyp defaults)
    parser.add_argument('--lr0', type=float, default=None, help='Initial learning rate')
    parser.add_argument('--lrf', type=float, default=None, help='Final learning rate factor')
    parser.add_argument('--momentum', type=float, default=None, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=None, help='Weight decay')
    parser.add_argument('--warmup_epochs', type=float, default=None, help='Warmup epochs')
    parser.add_argument('--warmup_momentum', type=float, default=None, help='Warmup momentum')
    parser.add_argument('--warmup_bias_lr', type=float, default=None, help='Warmup bias learning rate')
    
    # Loss gain arguments
    parser.add_argument('--box', type=float, default=None, help='Box loss gain')
    parser.add_argument('--cls', type=float, default=None, help='Classification loss gain')
    parser.add_argument('--dfl', type=float, default=None, help='DFL loss gain')
    
    # Validation arguments
    parser.add_argument('--conf', type=float, default=None, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=None, help='IoU threshold for NMS')
    parser.add_argument('--max_det', type=int, default=None, help='Maximum detections per image')
    
    # Output directory behavior
    parser.add_argument('--exist_ok', action='store_true', help='Allow overwriting existing project/name directory')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def build_overrides_from_args(args):
    """Build overrides dictionary from parsed arguments."""
    # Open hyperparameter files
    overrides = yaml.safe_load(Path(args.hyp).read_text())
    
    # Append arg_parse items to the overrides dictionary
    overrides["save"] = args.save
    overrides["save_period"] = args.save_period
    overrides["model"] = args.model
    overrides["seed"] = args.seed
    
    # NOTE: Don't add 'weights' to overrides - YOLO config doesn't recognize it
    # The weights path is stored in PRETRAINED_WEIGHTS_PATH global variable instead
    overrides["data"] = args.data
    overrides["epochs"] = args.epochs
    overrides["batch"] = args.batch
    overrides["imgsz"] = args.imgsz
    overrides["rect"] = args.rect
    overrides["resume"] = args.resume
    overrides["cache"] = args.cache
    overrides["device"] = args.device
    overrides["workers"] = args.workers
    overrides["project"] = args.project
    overrides["name"] = args.name
    overrides["cos_lr"] = args.cos_lr
    overrides["half"] = args.half
    overrides["plots"] = args.plots
    overrides["pretrained"] = args.pretrained
    overrides["nbs"] = args.nbs
    overrides["optimizer"] = args.optimizer

    overrides["clip_length"] = args.clip_length
    overrides["clip_stride"] = args.clip_stride  # Fixed: was incorrectly using clip_length
    overrides["channels"] = args.channels
    # FIX: Pass select_channels through overrides so it propagates to DDP workers
    overrides["select_channels"] = args.select_channels

    overrides["val_epoch"] = args.val_epoch

    overrides["flip"] = args.flip
    overrides["invert"] = args.invert
    overrides["suppress"] = args.suppress
    overrides["positive"] = args.positive
    overrides["zoom_out"] = args.zoom_out
    overrides["max_zoom_out_factor"] = args.max_zoom_out_factor
    overrides["min_zoom_out_factor"] = args.min_zoom_out_factor

    # Distance estimation overrides
    overrides["distance"] = args.distance
    overrides["nbins"] = args.nbins
    overrides["min_dist"] = args.min_dist
    overrides["max_dist"] = args.max_dist
    overrides["dist_loss_mult"] = args.dist_loss_mult
    
    # Learning rate overrides (only if explicitly set)
    if args.lr0 is not None:
        overrides["lr0"] = args.lr0
    if args.lrf is not None:
        overrides["lrf"] = args.lrf
    if args.momentum is not None:
        overrides["momentum"] = args.momentum
    if args.weight_decay is not None:
        overrides["weight_decay"] = args.weight_decay
    if args.warmup_epochs is not None:
        overrides["warmup_epochs"] = args.warmup_epochs
    if args.warmup_momentum is not None:
        overrides["warmup_momentum"] = args.warmup_momentum
    if args.warmup_bias_lr is not None:
        overrides["warmup_bias_lr"] = args.warmup_bias_lr
    
    # Loss gain overrides (only if explicitly set)
    if args.box is not None:
        overrides["box"] = args.box
    if args.cls is not None:
        overrides["cls"] = args.cls
    if args.dfl is not None:
        overrides["dfl"] = args.dfl
    
    # Validation overrides (only if explicitly set)
    if args.conf is not None:
        overrides["conf"] = args.conf
    if args.iou is not None:
        overrides["iou"] = args.iou
    if args.max_det is not None:
        overrides["max_det"] = args.max_det
    
    # Output directory behavior
    overrides["exist_ok"] = args.exist_ok
    
    return overrides


# Global variables - only set when running as main script
args = None
overrides = None

# Global variable to store pretrained weights path (separate from YOLO config)
PRETRAINED_WEIGHTS_PATH = None

# Global variable for channel selection (not a standard YOLO arg)
SELECT_CHANNELS = None


# BaseTrainer python usage
class EventVideoYOLOv8DetectionTrainer(BaseTrainer):
    
    def __init__(self,cfg=DEFAULT_CFG, overrides=None):
        """
        Initializes the BaseTrainer class.

        Args:
            cfg (str, optional): Path to a configuration file. Defaults to DEFAULT_CFG.
            overrides (dict, optional): Configuration overrides. Defaults to None.
        """
        self.args = get_cfg(cfg, overrides)
        self.device = select_device(self.args.device, self.args.batch)
        self.check_resume()
        self.console = LOGGER
        self.validator = None
        self.model = None
        self.metrics = None  
        
        init_seeds(self.args.seed + 1 + RANK, deterministic=self.args.deterministic)
        
        # Get video config and augmentation params from self.args (works for both CLI and DDP modes)
        # FIX: Read select_channels from self.args instead of global variable
        # This ensures it propagates correctly to DDP worker processes
        _select_channels = getattr(self.args, 'select_channels', None)
        self.video_config = {
            "clip_length": getattr(self.args, 'clip_length', 11),
            "clip_stride": getattr(self.args, 'clip_stride', 11),
            "channels": getattr(self.args, 'channels', 10),
            "select_channels": _select_channels
        }
        self.aug_params = {
            "flip": getattr(self.args, 'flip', 0.0),
            "invert": getattr(self.args, 'invert', 0.0),
            "suppress": getattr(self.args, 'suppress', 0.0),
            "positive": getattr(self.args, 'positive', 0.0),
            "zoom_out": getattr(self.args, 'zoom_out', 0.0),
            "max_zoom_out_factor": getattr(self.args, 'max_zoom_out_factor', 2.0),
            "min_zoom_out_factor": getattr(self.args, 'min_zoom_out_factor', 1.0)
        }
        
        # Dirs
        project = self.args.project or Path(SETTINGS['runs_dir']) / self.args.task
        name = self.args.name or f'{self.args.mode}'
        if hasattr(self.args, 'save_dir'):
            self.save_dir = Path(self.args.save_dir)
        else:
            self.save_dir = Path(
                increment_path(Path(project) / name, exist_ok=self.args.exist_ok if RANK in {-1, 0} else True))
        self.wdir = self.save_dir / 'weights'  # weights dir
        if RANK in {-1, 0}:
            self.wdir.mkdir(parents=True, exist_ok=True)  # make dir
            self.args.save_dir = str(self.save_dir)
            yaml_save(self.save_dir / 'args.yaml', vars(self.args))  # save run args
        self.last, self.best = self.wdir / 'last.pt', self.wdir / 'best.pt'  # checkpoint paths
        self.save_period = self.args.save_period

        self.batch_size = self.args.batch
        self.epochs = self.args.epochs
        self.start_epoch = 0
        if RANK == -1:
            print_args(vars(self.args))

        # Device
        self.amp = self.device.type != 'cpu'
        self.scaler = amp.GradScaler(enabled=self.amp)
        if self.device.type == 'cpu':
            self.args.workers = 0  # faster CPU training as time dominated by inference, not dataloading

        # Model and Dataloaders.
        self.model = self.args.model
        try:
            if self.args.task == 'classify':
                self.data = check_cls_dataset(self.args.data)
            elif self.args.data.endswith('.yaml') or self.args.task in ('detect', 'segment'):
                self.data = check_det_dataset(self.args.data)
                if 'yaml_file' in self.data:
                    self.args.data = self.data['yaml_file']  # for validating 'yolo train data=url.zip' usage
        except Exception as e:
            raise FileNotFoundError(emojis(f"Dataset '{self.args.data}' error ❌ {e}")) from e

        self.trainset, self.testset = self.get_dataset(self.data)
        self.ema = None

        # Optimization utils init
        self.lf = None
        self.scheduler = None

        # Epoch level metrics
        self.best_fitness = None
        self.fitness = None
        self.loss = None
        self.tloss = None
        self.loss_names = ['Loss']
        self.csv = self.save_dir / 'results.csv'
        self.plot_idx = [0, 1, 2]
        
        # Unified visualization logger (initialized properly in _setup_train)
        self.unified_csv_logger = None

        # Callbacks
        self.callbacks = defaultdict(list, callbacks.default_callbacks)  # add callbacks
        if RANK in {0, -1}:
            callbacks.add_integration_callbacks(self)
            # Convert Path objects to strings and sanitize for wandb compatibility
            # wandb doesn't allow: /,\,#,?,%,:
            wandb_project = str(self.args.project).replace('/', '_').replace('\\', '_')
            wandb_name = str(self.args.name).replace('/', '_').replace('\\', '_')
            wandb.init(project=wandb_project, name=wandb_name, config=overrides)
        


    
    def get_dataloader(self, dataset_path, batch_size, aug_param, mode, rank=0, load = "batched"):
        # TODO: manage splits differently
        # calculate stride - check if model is initialized
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        
        # Get channel selection config
        select_channels = self.video_config.get("select_channels", None)

        if mode == "train":
         return build_video_dataloader(self.args, self.video_config, batch_size,dataset_path, aug_param = self.aug_params,rank=rank, mode = mode, load = load, select_channels=select_channels)[0]
        else: 

           return build_video_val_standalone_dataloader(self.args, self.video_config, batch_size,dataset_path,rank, mode = load, select_channels=select_channels)[0]

    def preprocess_batch(self, batch):
        #batch['img'] = batch['img'].to(self.device, non_blocking=True).float() / 255
        #batch = batch.to(self.device, non_blocking=True).float() / batch.max()
        #batch = (batch*127.5 + 127.5).to(self.device, non_blocking=True).float() / 255   
        batch = (batch).to(self.device, non_blocking=True).float()  

        #print("before interp", batch.shape)
        new_scale = [batch.shape[2] + (math.ceil(batch.shape[2]/32)*32 -  batch.shape[2]), batch.shape[3] + (math.ceil(batch.shape[3]/32)*32 -  batch.shape[3])]

        batch = nn.functional.interpolate(batch,scale_factor = (new_scale[0] / batch.shape[2], new_scale[1] / batch.shape[3]), mode = 'bilinear')

        return batch

    def set_model_attributes(self):
        # TO DO: IT SHOULD BE BETTER TO ADD MODEL CHANNELS HERE
        # nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)
        # self.args.box *= 3 / nl  # scale to layers
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # scale to classes and layers
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        self.model.nc = self.data['nc']  # attach number of classes to model
        self.model.names = self.data['names']  # attach class names to model
        self.model.args = self.args  # attach hyperparameters to model
        # TODO: self.model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc

    def setup_model(self):
        """
        Load/create model for ReYOLOv8 training.
        Overrides base to properly handle --weights argument for pretrained weights.
        """
        global PRETRAINED_WEIGHTS_PATH
        
        if isinstance(self.model, torch.nn.Module):  # if model is loaded beforehand
            return

        model, weights = self.model, None
        ckpt = None
        
        # Check if model is a .pt file (full checkpoint)
        if str(model).endswith('.pt'):
            weights, ckpt = attempt_load_one_weight(model)
            cfg = ckpt['model'].yaml
        else:
            cfg = model
            # Check for separate pretrained weights via global PRETRAINED_WEIGHTS_PATH
            if PRETRAINED_WEIGHTS_PATH:
                weights_path = str(PRETRAINED_WEIGHTS_PATH)
                if weights_path.endswith('.pt') and Path(weights_path).exists():
                    LOGGER.info(f"Loading pretrained weights from: {weights_path}")
                    weights = weights_path
        
        self.model = self.get_model(cfg=cfg, weights=weights, verbose=RANK == -1)
        return ckpt

    def get_model(self, cfg=None, weights=None, verbose=True):
        # Check if distance estimation is enabled
        if hasattr(self.args, 'distance') and self.args.distance:
            # Use distance-aware model
            from custom_reyolov8_distance import ReYOLOv8_WithDistance
            
            LOGGER.info(f"Creating REYOLOv8 model with distance estimation...")
            LOGGER.info(f"  Distance range: [{self.args.min_dist}, {self.args.max_dist}]m")
            LOGGER.info(f"  Distance bins: {self.args.nbins}")
            LOGGER.info(f"  Input channels: {self.video_config['channels']}")
            
            model = ReYOLOv8_WithDistance.create_model(
                model_cfg=cfg or self.args.model,
                nc=self.data['nc'],
                nbins=self.args.nbins,
                min_dist=self.args.min_dist,
                max_dist=self.args.max_dist,
                channels=self.video_config["channels"]
            )
            
            if weights:
                LOGGER.info(f"Loading pretrained weights from: {weights}")
                checkpoint = torch.load(weights, map_location='cpu', weights_only=False)
                state_dict = checkpoint.get('model', checkpoint)
                if hasattr(state_dict, 'state_dict'):
                    state_dict = state_dict.state_dict()
                elif hasattr(state_dict, 'float'):
                    state_dict = state_dict.float().state_dict()
                
                # Filter out keys with shape mismatches
                model_state_dict = model.state_dict()
                filtered_state_dict = {k: v for k, v in state_dict.items() 
                                       if k in model_state_dict and v.shape == model_state_dict[k].shape}
                skipped = len(state_dict) - len(filtered_state_dict)
                if skipped > 0:
                    LOGGER.info(f"Skipped {skipped} keys due to shape mismatch (will be randomly initialized)")
                
                model.load_state_dict(filtered_state_dict, strict=False)
                LOGGER.info(f"Pretrained weights loaded: {len(filtered_state_dict)}/{len(model_state_dict)} parameters transferred")
        else:
            # Standard REYOLOv8 model
            model = DetectionModel2(cfg, imgsz = self.args.imgsz,ch=self.video_config["channels"], nc=self.data['nc'], verbose=True)
            if weights:
                # Handle both path strings and loaded state dicts
                if isinstance(weights, str):
                    LOGGER.info(f"Loading pretrained weights from: {weights}")
                    checkpoint = torch.load(weights, map_location='cpu', weights_only=False)
                    # Handle different checkpoint formats
                    if isinstance(checkpoint, dict):
                        if 'model' in checkpoint:
                            state_dict = checkpoint['model']
                            # If model is stored as nn.Module, get its state_dict
                            if hasattr(state_dict, 'state_dict'):
                                state_dict = state_dict.state_dict()
                            elif hasattr(state_dict, 'float'):
                                state_dict = state_dict.float().state_dict()
                        else:
                            state_dict = checkpoint
                    else:
                        # checkpoint is the model itself
                        state_dict = checkpoint.state_dict() if hasattr(checkpoint, 'state_dict') else checkpoint
                    
                    # Filter out keys with shape mismatches (e.g., detection head when num_classes differs)
                    model_state_dict = model.state_dict()
                    filtered_state_dict = {}
                    skipped_keys = []
                    
                    for k, v in state_dict.items():
                        if k in model_state_dict:
                            if v.shape == model_state_dict[k].shape:
                                filtered_state_dict[k] = v
                            else:
                                skipped_keys.append(f"{k}: checkpoint {v.shape} vs model {model_state_dict[k].shape}")
                        else:
                            # Key doesn't exist in model - will be ignored
                            pass
                    
                    if skipped_keys:
                        LOGGER.info(f"Skipped {len(skipped_keys)} keys due to shape mismatch (will be randomly initialized):")
                        for sk in skipped_keys[:10]:  # Show first 10
                            LOGGER.info(f"  - {sk}")
                        if len(skipped_keys) > 10:
                            LOGGER.info(f"  ... and {len(skipped_keys) - 10} more")
                    
                    # Load the filtered state dict
                    missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
                    
                    loaded_count = len(filtered_state_dict)
                    total_model_params = len(model_state_dict)
                    LOGGER.info(f"Pretrained weights loaded: {loaded_count}/{total_model_params} parameters transferred")
                    if missing:
                        LOGGER.info(f"Missing keys (randomly initialized): {len(missing)} keys")
                else:
                    model.load(weights)

        return model
    
    def get_validator(self):
        # Use distance validator if distance estimation is enabled
        if hasattr(self.args, 'distance') and self.args.distance:
            import val_distance
            self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss', 'dist_loss'
            LOGGER.info("Using EventVideoDistanceValidator for validation with distance metrics")
            return val_distance.EventVideoDistanceValidator(
                self.video_config, 
                self.test_loader,
                save_dir=self.save_dir,
                logger=self.console,
                args=copy(self.args)
            )
        else:
            # Standard validator
            self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss'
            return val.EventVideoDetectionValidator(self.video_config, self.test_loader,save_dir=self.save_dir,logger=self.console,args=copy(self.args))

    def criterion(self, preds, batch, sequence_mask, cur_loss):
        if not hasattr(self, 'compute_loss'):
            self.compute_loss = LossVideo(de_parallel(self.model))
        return self.compute_loss(preds, batch, sequence_mask, cur_loss)

    def label_loss_items(self, loss_items=None, prefix='train'):
        """
        Returns a loss dict with labelled training loss items tensor
        """
        # Not needed for classification but necessary for segmentation & detection
        keys = [f'{prefix}/{x}' for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        return ('\n' + '%11s' *
                (4 + len(self.loss_names))) % ('Epoch', 'GPU_mem', *self.loss_names, 'Instances', 'Size')

    def plot_training_samples(self, batch, ni, T=None):
        """
        Plot training samples for event video data.
        
        For multi-channel event data, we visualize the temporal mean/sum
        or a specific timestep to create a meaningful 2D visualization.
        
        Args:
            batch: Batch data with 'img', 'bboxes', 'cls', 'batch_idx'
            ni: Batch index for filename
            T: Optional timestep to visualize (if None, uses last timestep)
        """
        try:
            images = batch['img']
            
            # Handle 5D event video tensor: (B, T, C, H, W) -> select one frame
            if images.dim() == 5:
                # Use last timestep if T not specified
                t_idx = T if T is not None else images.shape[1] - 1
                images = images[:, t_idx, :, :, :]  # (B, C, H, W)
            
            # Handle multi-channel (e.g., 11 channels) -> convert to 3-channel visualization
            if images.shape[1] > 3:
                # Option 1: Sum across channels and normalize
                vis_images = images.sum(dim=1, keepdim=True)  # (B, 1, H, W)
                vis_images = vis_images.repeat(1, 3, 1, 1)  # (B, 3, H, W)
                
                # Normalize per image for better visualization
                for i in range(vis_images.shape[0]):
                    img_min = vis_images[i].min()
                    img_max = vis_images[i].max()
                    if img_max > img_min:
                        vis_images[i] = (vis_images[i] - img_min) / (img_max - img_min)
            elif images.shape[1] == 1:
                # Single channel: repeat to make RGB
                vis_images = images.repeat(1, 3, 1, 1)
            else:
                vis_images = images
            
            # Call plot_images without paths (set to None or empty list)
            plot_images(
                images=vis_images,
                batch_idx=batch['batch_idx'],
                cls=batch['cls'].squeeze(-1),
                bboxes=batch['bboxes'],
                paths=None,  # No file paths for event video data
                fname=self.save_dir / f'train_batch{ni}.jpg'
            )
        except Exception as e:
            LOGGER.warning(f"Failed to plot training batch {ni}: {e}")

    def plot_metrics(self):
        """Plot training metrics from CSV files."""
        try:
            plot_results(file=self.csv)  # save results.png using ultralytics
        except Exception as e:
            LOGGER.warning(f"Warning: Plotting error for {self.csv}: {e}")
        
        # Also generate unified results if available
        if UNIFIED_VIZ_AVAILABLE and self.unified_csv_logger is not None:
            try:
                # Generate unified results.png from the standardized CSV
                unified_csv = self.save_dir / 'results_unified.csv'
                if unified_csv.exists():
                    plot_unified_results(str(unified_csv), str(self.save_dir))
            except Exception as e:
                LOGGER.warning(f"Failed to generate unified results plot: {e}")
        
        # Plot distance metrics if available
        distance_csv = self.save_dir / 'results_distance.csv'
        if distance_csv.exists():
            try:
                self._plot_distance_metrics(distance_csv)
            except Exception as e:
                LOGGER.warning(f"Failed to plot distance metrics: {e}")
    
    def _plot_distance_metrics(self, distance_csv):
        """Plot distance-specific metrics (MAE, accuracy) over epochs."""
        import matplotlib.pyplot as plt
        import pandas as pd
        
        try:
            df = pd.read_csv(distance_csv, skipinitialspace=True)
            if df.empty:
                return
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # Plot MAE/RMSE
            if 'distance/MAE' in df.columns:
                axes[0].plot(df['epoch'], df['distance/MAE'], label='MAE', marker='o')
            if 'distance/RMSE' in df.columns:
                axes[0].plot(df['epoch'], df['distance/RMSE'], label='RMSE', marker='s')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Error (m)')
            axes[0].set_title('Distance Estimation Error')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Plot accuracy at thresholds
            for col, label in [('distance/Acc@0.5m', '0.5m'), 
                               ('distance/Acc@1.0m', '1.0m'), 
                               ('distance/Acc@2.0m', '2.0m')]:
                if col in df.columns:
                    axes[1].plot(df['epoch'], df[col] * 100, label=f'Acc@{label}', marker='o')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy (%)')
            axes[1].set_title('Distance Accuracy at Thresholds')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.save_dir / 'results_distance.png', dpi=150)
            plt.close()
            LOGGER.info(f"Distance metrics plot saved to {self.save_dir / 'results_distance.png'}")
        except Exception as e:
            LOGGER.warning(f"Could not plot distance metrics: {e}")
    
    def save_metrics(self, metrics):
        """Save metrics to CSV files (both ultralytics and unified format).
        
        Handles distance metrics separately to maintain consistent CSV format.
        """
        # Separate distance metrics from detection metrics for consistent CSV format
        detection_metrics = {}
        distance_metrics = {}
        
        for k, v in metrics.items():
            if k.startswith('distance/'):
                distance_metrics[k] = v
            else:
                detection_metrics[k] = v
        
        # Log all metrics to wandb
        wandb.log(metrics)
        
        # Save detection metrics to main CSV (consistent columns)
        keys, vals = list(detection_metrics.keys()), list(detection_metrics.values())
        n = len(detection_metrics) + 1  # number of cols
        s = '' if self.csv.exists() else (('%23s,' * n % tuple(['epoch'] + keys)).rstrip(',') + '\n')  # header
        with open(self.csv, 'a') as f:
            f.write(s + ('%23.5g,' * n % tuple([self.epoch] + vals)).rstrip(',') + '\n')
        
        # Save distance metrics to separate CSV if any exist
        if distance_metrics:
            distance_csv = self.save_dir / 'results_distance.csv'
            dist_keys, dist_vals = list(distance_metrics.keys()), list(distance_metrics.values())
            dist_n = len(distance_metrics) + 1
            dist_s = '' if distance_csv.exists() else (('%23s,' * dist_n % tuple(['epoch'] + dist_keys)).rstrip(',') + '\n')
            with open(distance_csv, 'a') as f:
                f.write(dist_s + ('%23.5g,' * dist_n % tuple([self.epoch] + dist_vals)).rstrip(',') + '\n')
        
        # Also log to unified CSV logger for standardized format
        if UNIFIED_VIZ_AVAILABLE and self.unified_csv_logger is not None:
            try:
                # Map metrics to standardized format
                unified_metrics = {
                    'epoch': self.epoch,
                    'train/box_loss': detection_metrics.get('train/box_loss', 0),
                    'train/cls_loss': detection_metrics.get('train/cls_loss', 0),
                    'train/dfl_loss': detection_metrics.get('train/dfl_loss', 0),
                    'metrics/precision': detection_metrics.get('metrics/precision(B)', 0),
                    'metrics/recall': detection_metrics.get('metrics/recall(B)', 0),
                    'metrics/mAP50': detection_metrics.get('metrics/mAP50(B)', 0),
                    'metrics/mAP50-95': detection_metrics.get('metrics/mAP50-95(B)', 0),
                    'val/box_loss': detection_metrics.get('val/box_loss', 0),
                    'val/cls_loss': detection_metrics.get('val/cls_loss', 0),
                    'val/dfl_loss': detection_metrics.get('val/dfl_loss', 0),
                    'lr/pg0': detection_metrics.get('lr/pg0', 0),
                }
                self.unified_csv_logger.log(unified_metrics)
            except Exception as e:
                LOGGER.warning(f"Failed to log to unified CSV: {e}")

    def _setup_train(self, rank, world_size):
        """
        Builds dataloaders and optimizer on correct rank process.
        """
        # model
        self.run_callbacks('on_pretrain_routine_start')
        ckpt = self.setup_model()
        self.model = self.model.to(self.device)
        self.set_model_attributes()
        if world_size > 1:
            #self.model = DDP(self.model, device_ids=[rank], find_unused_parameters = True)
            self.model = DDP(self.model, device_ids=[rank], broadcast_buffers=False)
        # Check imgsz
        gs = max(int(self.model.stride.max() if hasattr(self.model, 'stride') else 32), 32)  # grid size (max stride)
        #self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)
        # Batch size
 
        # Optimizer
        self.accumulate = max(round((self.args.nbs / self.video_config["clip_length"] )/ self.batch_size), 1)  # accumulate loss before optimizing
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / (self.args.nbs / self.video_config["clip_length"] )# scale weight_decay
        self.optimizer = self.build_optimizer(model=self.model,
                                              name=self.args.optimizer,
                                              lr=self.args.lr0,
                                              momentum=self.args.momentum,
                                              decay=weight_decay)
        # Scheduler
        if self.args.cos_lr:
            self.lf = one_cycle(1, self.args.lrf, self.epochs)  # cosine 1->hyp['lrf']
        else:
            self.lf = lambda x: (1 - x / self.epochs) * (1.0 - self.args.lrf) + self.args.lrf  # linear
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False

        # dataloaders
        self.batch_size_ = self.batch_size // world_size if world_size > 1 else self.batch_size
        #get_dataloader(self, dataset_path, batch_size, img_x, img_y, mode='train', rank=0)

        self.train_loader = self.get_dataloader(self.trainset, batch_size=self.batch_size_, aug_param = self.aug_params, rank=rank, mode='train')

        if rank in {0, -1}:
               
            self.test_loader = self.get_dataloader(self.testset, batch_size=self.batch_size_, aug_param = self.aug_params,mode='val',rank=-1, load = "batched")
            self.validator = self.get_validator()
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix='val')
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))  # TODO: init metrics for plot_results()?
            self.ema = ModelEMA(self.model)
        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move
        self.run_callbacks('on_pretrain_routine_end')
        
        # Initialize unified CSV logger for standardized metric tracking
        if UNIFIED_VIZ_AVAILABLE and rank in {0, -1}:
            self.unified_csv_logger = StandardizedCSVLogger(self.save_dir)
        else:
            self.unified_csv_logger = None

    def get_test_dataset(self, data):
        """
        Get test path from data dict. Falls back to val if test doesn't exist.
        """
        test_path = data.get('test')
        if test_path is None:
            # Fall back to validation set if no test set exists
            test_path = data.get('val')
            if test_path:
                LOGGER.info("No 'test' split found in data.yaml, using 'val' split for final evaluation")
        return test_path


    def final_eval(self):
        del self.testset, self.trainset
        self.testset = self.get_test_dataset(self.data)
        
        # Skip final eval if no test/val dataset available
        if self.testset is None:
            LOGGER.warning("No test or val dataset found for final evaluation. Skipping final_eval.")
            return
        
        # For final evaluation after DDP training, use only a single GPU (GPU 0)
        # This avoids issues with multi-device selection when only RANK 0 runs final_eval
        original_device = self.args.device
        self.args.device = 0  # Use single GPU for final evaluation
        LOGGER.info(f"Final evaluation will run on device: {self.args.device}")
        
        LOGGER.info(f"Starting final evaluation on: {self.testset}")
        LOGGER.info("Loading data in batched mode (batch_size=32, clip-level LSTM memory)...")
        
        #self, dataset_path, batch_size, img_x, img_y, aug_param, mode, rank=0, load = "batched", mixed_load = False
        # Use batched mode with batch_size=32 for faster final evaluation
        final_eval_batch_size = 32
        self.test_loader = self.get_dataloader(self.testset, batch_size=final_eval_batch_size, aug_param=self.aug_params, mode='val', rank=-1, load="batched")
        
        LOGGER.info(f"Dataloader ready with {len(self.test_loader)} batches (batch_size={final_eval_batch_size})")
        self.validator = self.get_validator()

        for f in self.last, self.best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is self.best:
                    self.console.info(f'\nValidating {f}...')
                    self.metrics = self.validator(model=f)
                    self.metrics.pop('fitness', None)
                    self.run_callbacks('on_fit_epoch_end')
                    wandb.log(self.metrics)
        
        # Generate unified metric curves if available
        if UNIFIED_VIZ_AVAILABLE:
            try:
                # Get precision/recall data from validator's metrics
                if hasattr(self.validator, 'metrics') and hasattr(self.validator.metrics, 'results_dict'):
                    results = self.validator.metrics.results_dict
                    
                    # Try to get PR curve data - ultralytics stores this internally
                    # For now, we'll generate curves based on confusion matrix data
                    if hasattr(self.validator, 'confusion_matrix'):
                        # The confusion matrix is already plotted by ultralytics
                        # We just log a message about where to find it
                        LOGGER.info(f"Confusion matrix saved to {self.save_dir}")
                    
                    # Close the unified CSV logger
                    if self.unified_csv_logger is not None:
                        self.unified_csv_logger.close()
                        LOGGER.info(f"Unified metrics saved to {self.save_dir / 'results_unified.csv'}")
            except Exception as e:
                LOGGER.warning(f"Failed to generate unified metrics: {e}")
        
        # Restore original device setting
        self.args.device = original_device


    def _do_train(self, rank=-1, world_size=1):
        if world_size > 1:
            self._setup_ddp(rank, world_size)
    
        self._setup_train(rank, world_size)

        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        nb = len(self.train_loader)  # number of batches
        
        nw = max(round(self.args.warmup_epochs * nb), 100)  # number of warmup iterations
        last_opt_step = -1
        self.run_callbacks('on_train_start')
        self.log(f'Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n'
                 f'Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n'
                 f"Logging results to {colorstr('bold', self.save_dir)}\n"
                 f'Starting training for {self.epochs} epochs...')

        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            self.run_callbacks('on_train_epoch_start')
            self.model.train()
            

            if rank != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            
            if rank in {-1, 0}:
                self.console.info(self.progress_string())
                pbar = tqdm(enumerate(self.train_loader), total=nb, bar_format=TQDM_BAR_FORMAT)
            self.tloss = None

            for i, batch in pbar:        
             hidden_states = {"0": None, "1": None, "2": None, "3": None}
             #self.loss = torch.zeros([], device=self.device)
             self.optimizer.zero_grad()
             for T in range(self.video_config["clip_length"]):
                sequence_mask = batch['vid_pos'] == T
                self.run_callbacks('on_train_batch_start')
                # Warmup
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(1, np.interp(ni, xi, [1, (self.args.nbs / self.video_config["clip_length"])/ self.batch_size]).round())
                    for j, x in enumerate(self.optimizer.param_groups):
                        # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x['lr'] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x['initial_lr'] * self.lf(epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])


                with torch.cuda.amp.autocast(self.amp):

                    batch_ = self.preprocess_batch(batch['img'][:,T,:,:,:])

                    preds, hidden_states = self.model(batch_, hidden_states)

                    if T == 0:
                     self.loss, self.loss_items = self.criterion(preds, batch, sequence_mask, None)
                    else: 

                     self.loss, self.loss_items = self.criterion(preds, batch, sequence_mask, self.loss)

                    
                    if rank != -1:
                        self.loss *= world_size
                    self.tloss = (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None \
                        else self.loss_items


             self.scaler.scale(self.loss).backward()

             # Optimizer Step only at the end of sequence
             # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
             if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

             # Log
             mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
             loss_len = self.tloss.shape[0] if len(self.tloss.size()) else 1
             losses = self.tloss if loss_len > 1 else torch.unsqueeze(self.tloss, 0)
             if rank in {-1, 0}:
                    pbar.set_description(
                        ('%11s' * 2 + '%11.4g' * (2 + loss_len)) %
                        (f'{epoch + 1}/{self.epochs}', mem, *losses, batch['cls'].shape[0], batch['img'].shape[-1]))
                    self.run_callbacks('on_batch_end')
                    if self.args.plots and ni in self.plot_idx and T == self.video_config["clip_length"] - 1:
                        # Plot at last timestep of selected batches
                        self.plot_training_samples(batch, ni, T)

             self.run_callbacks('on_train_batch_end')

            self.lr = {f'lr/pg{ir}': x['lr'] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers

            self.scheduler.step()
            self.run_callbacks('on_train_epoch_end')

            # DDP: Synchronize all ranks after training batches complete
            # This ensures all GPU work is finished before rank-specific operations
            if rank != -1:
                if rank == 0:
                    LOGGER.info(f"[RANK {rank}] Epoch {epoch+1}: Training complete, starting CUDA sync...")
                torch.cuda.synchronize()  # Wait for all GPU operations to complete
                if rank == 0:
                    LOGGER.info(f"[RANK {rank}] Epoch {epoch+1}: CUDA sync done, entering first barrier...")
                dist.barrier()  # Ensure all ranks finished epoch training
                if rank == 0:
                    LOGGER.info(f"[RANK {rank}] Epoch {epoch+1}: First barrier passed")

            if rank in {-1, 0}:

                # Validation
                self.ema.update_attr(self.model, include=['yaml', 'nc', 'args', 'names', 'stride', 'class_weights'])
                final_epoch = (epoch + 1 == self.epochs) or self.stopper.possible_stop

                if (self.args.val and (epoch - 1) % self.args.val_epoch == 0  and epoch != 0):
                    self.metrics, self.fitness = self.validate()         



                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop = self.stopper(epoch + 1, self.fitness)

                # Save model
                if self.args.save or (epoch + 1 == self.epochs):
                    LOGGER.info(f"[RANK {rank}] Epoch {epoch+1}: Saving model...")
                    self.save_model()
                    self.run_callbacks('on_model_save')
                    LOGGER.info(f"[RANK {rank}] Epoch {epoch+1}: Model saved")

            # DDP: Second barrier to sync all ranks after rank-specific operations
            # This ensures RANK 0's validation/saving is complete before next epoch
            if rank != -1:
                if rank == 0:
                    LOGGER.info(f"[RANK {rank}] Epoch {epoch+1}: Entering second barrier...")
                dist.barrier()
                if rank == 0:
                    LOGGER.info(f"[RANK {rank}] Epoch {epoch+1}: Second barrier passed")

            tnow = time.time()
            self.epoch_time = tnow - self.epoch_time_start
            self.epoch_time_start = tnow
            self.run_callbacks('on_fit_epoch_end')

            # Early Stopping
            if RANK != -1:  # if DDP training
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                if RANK != 0:
                    self.stop = broadcast_list[0]
            if self.stop:
                break  # must break all DDP ranks

        if rank in {-1, 0}:
            # Do final val with best.pt
            self.log(f'\n{epoch - self.start_epoch + 1} epochs completed in '
                     f'{(time.time() - self.train_time_start) / 3600:.3f} hours.')
            self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.log(f"Results saved to {colorstr('bold', self.save_dir)}")
            self.run_callbacks('on_train_end')
        torch.cuda.empty_cache()
        self.run_callbacks('teardown')



# Criterion class for computing training losses
class LossVideo:

    def __init__(self, model):  # model must be de-paralleled

        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device
        self.model = model  # Store model reference for distance head access

        self.use_dfl = m.reg_max > 1
        roll_out_thr = h.min_memory if h.min_memory > 1 else 64 if h.min_memory else 0  # 64 is default

        self.assigner = TaskAlignedAssigner(topk=10,
                                            num_classes=self.nc,
                                            alpha=0.5,
                                            beta=6.0,
                                            roll_out_thr=roll_out_thr)
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)
        
        # Check if model has distance head
        self.has_distance_head = self._check_distance_head()
        if self.has_distance_head:
            LOGGER.info("LossVideo: Distance head detected, will compute distance loss")
    
    def _check_distance_head(self):
        """Check if model has distance estimation head."""
        for m in self.model.model:
            if hasattr(m, 'compute_distance_loss'):
                return True
        return False
    
    def _get_distance_head(self):
        """Get distance detection head from model."""
        for m in self.model.model:
            if hasattr(m, 'compute_distance_loss'):
                return m
        return None

    def preprocess(self, targets, batch_size, scale_tensor):

        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))

        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch, sequence_mask, cur_loss):
        # Check if model has distance head
        # Training mode returns: (feats, dist_outputs) where feats is a LIST
        # Inference mode returns: (y, x) where y is a TENSOR and x is a list
        has_distance = isinstance(preds, tuple) and len(preds) == 2 and self.has_distance_head
        
        if has_distance:
            first_elem, second_elem = preds
            # Detect if we're in training format or inference format
            if isinstance(first_elem, (list, tuple)) and all(isinstance(f, torch.Tensor) for f in first_elem):
                # Training format: (feats, dist_outputs)
                feats = first_elem
                dist_preds = second_elem
            elif isinstance(first_elem, torch.Tensor):
                # Inference format: (y, x) where y is predictions, x is feats
                # We need feats (x) for loss computation
                feats = second_elem  # x is the feature list
                dist_preds = None  # Distance is embedded in y, skip distance loss during validation
            else:
                # Fallback: treat as (det_preds, dist_preds)
                det_preds, dist_preds = preds
                feats = det_preds[1] if isinstance(det_preds, tuple) else det_preds
            
            # Always use 4 components when has_distance is True, to match validator's self.loss size
            loss = torch.zeros(4, device=self.device)  # box, cls, dfl, dist
        else:
            feats = preds[1] if isinstance(preds, tuple) else preds
            loss = torch.zeros(3, device=self.device)  # box, cls, dfl
            dist_preds = None


        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)

        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets

        targets = torch.cat((batch['batch_idx'][sequence_mask].view(-1, 1), batch['cls'][sequence_mask].view(-1, 1), batch['bboxes'][sequence_mask]), 1)

        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy

        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_bboxes /= stride_tensor
        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # bbox loss
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        
        # Compute distance loss if available
        if has_distance and dist_preds is not None and 'distances' in batch:
            dist_head = self._get_distance_head()
            
            if dist_head is not None:
                # Step 2: Preprocess distance targets to [batch_size, max_objects] format
                dist_targets_raw = batch['distances'][sequence_mask].to(self.device)
                
                if dist_targets_raw.shape[0] == 0:
                    gt_distances = torch.zeros(batch_size, 0, device=self.device)
                else:
                    i = batch['batch_idx'][sequence_mask].to(self.device)
                    _, counts = i.unique(return_counts=True)
                    gt_distances = torch.full((batch_size, counts.max()), -1.0, device=self.device)
                    for j in range(batch_size):
                        matches = i == j
                        n = matches.sum()
                        if n:
                            gt_distances[j, :n] = dist_targets_raw[matches]
                
                # Step 3: Gather distances for matched anchors using target_gt_idx
                # target_gt_idx shape: (batch, num_anchors) - contains GT index for each anchor
                # gt_distances shape: (batch, max_objects) - distance for each GT object
                # Result: (batch, num_anchors) - distance for each anchor
                if gt_distances.shape[1] > 0:
                    target_distances = torch.gather(
                        gt_distances,  # (batch, max_objects)
                        dim=1,
                        index=target_gt_idx.clamp(0, gt_distances.shape[1]-1)  # (batch, num_anchors)
                    )  # (batch, num_anchors)
                    
                    # Step 4: Filter foreground anchors and compute loss
                    # Create a full-size mask for valid distances
                    # Start with foreground mask, then filter for valid distances
                    full_valid_mask = fg_mask.clone()  # (batch, num_anchors)
                    
                    # Mark anchors with invalid distances as invalid
                    invalid_dist = target_distances < 0
                    full_valid_mask = full_valid_mask & ~invalid_dist
                    
                    # Flatten the mask
                    full_valid_mask_flat = full_valid_mask.view(-1)  # (batch*num_anchors,)
                    
                    if full_valid_mask_flat.any():
                        try:
                            # Select valid distance targets
                            target_distances_valid = target_distances[full_valid_mask]
                            
                            # Compute distance loss
                            dist_loss = dist_head.compute_distance_loss(
                                dist_preds,
                                target_distances_valid,  # Only valid distances
                                None,  # batch_idx not needed
                                full_valid_mask_flat  # Full-size mask for indexing predictions
                            )
                            
                            loss[3] = dist_loss
                            
                            # Apply distance loss multiplier
                            dist_mult = getattr(self.hyp, 'dist_loss_mult', 1.0)
                            loss[3] *= dist_mult
                        except Exception as e:
                            LOGGER.warning(f"Distance loss computation failed: {e}")
                            # Create zero loss connected to graph
                            loss[3] = dist_preds[0].sum() * 0.0
                    else:
                        # No valid distance targets in this batch
                        # Create zero loss connected to graph
                        loss[3] = dist_preds[0].sum() * 0.0
                else:
                    # No ground truth objects in this batch
                    # Create zero loss connected to graph
                    loss[3] = dist_preds[0].sum() * 0.0
            else:
                # Distance head not available
                # Create zero loss connected to graph
                loss[3] = dist_preds[0].sum() * 0.0 if dist_preds is not None else torch.zeros(1, device=self.device)[0]
        # else: Distance estimation not enabled - no need to set loss[3] since loss only has 3 components
        
        if cur_loss:
           return loss.sum() * batch_size + cur_loss, loss.detach()  # loss(box, cls, dfl, dist)
        else: 
           return loss.sum() * batch_size, loss.detach()

# Only run training when executed directly (not when imported for DDP)
if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_opt()
    
    # Set global pretrained weights path (separate from YOLO config)
    if args.weights and str(args.weights) != str(ROOT / 'yolov8n.pt'):
        PRETRAINED_WEIGHTS_PATH = str(args.weights)
        LOGGER.info(f"Pretrained weights path set: {PRETRAINED_WEIGHTS_PATH}")
    
    # Set global channel selection (not a standard YOLO arg)
    if args.select_channels is not None:
        SELECT_CHANNELS = args.select_channels
        LOGGER.info(f"Channel selection set: first {SELECT_CHANNELS} channels")
    
    # Build overrides from arguments
    overrides = build_overrides_from_args(args)
    # Create and run trainer
    trainer = EventVideoYOLOv8DetectionTrainer(overrides=overrides)
    trainer.train()
