# Ultralytics YOLO 🚀, GPL-3.0 license

import os
import shutil
import socket
import sys
import tempfile

from . import USER_CONFIG_DIR
from .torch_utils import TORCH_1_9


def find_free_network_port() -> int:
    """Finds a free port on localhost.

    It is useful in single-node training when we don't want to connect to a real main node but have to set the
    `MASTER_PORT` environment variable.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]  # port


def generate_ddp_file(trainer):
    """Generate a temporary file for DDP training.
    
    Handles both standard Ultralytics trainers and custom trainers like EventVideoYOLOv8DetectionTrainer.
    
    Returns:
        tuple: (file_path, is_custom_trainer)
    """
    trainer_class_name = trainer.__class__.__name__
    trainer_class_str = str(trainer.__class__)
    
    # Extract module path from class string
    # Format: <class 'module.path.ClassName'>
    import_path = '.'.join(trainer_class_str.split(".")[1:-1])
    
    # Build the import section (at module level, not inside if __main__)
    import_lines = []
    is_custom_trainer = False
    
    # Check if trainer is from ultralytics package or a custom module
    if 'ultralytics' in trainer_class_str:
        # Standard ultralytics trainer
        import_lines.append(f"from ultralytics.{import_path} import {trainer_class_name}")
    elif '__main__' in trainer_class_str or not import_path:
        # Custom trainer defined in __main__ (e.g., EventVideoYOLOv8DetectionTrainer in train.py)
        # For ReYOLOv8, we need to import from the train.py file directly
        is_custom_trainer = True
        
        # Get the path to the original training script
        import __main__
        main_file = getattr(__main__, '__file__', None)
        
        if main_file and os.path.exists(main_file):
            # Add the directory containing train.py to the Python path
            train_dir = os.path.dirname(os.path.abspath(main_file))
            import_lines.append("import sys")
            import_lines.append("import os")
            import_lines.append(f'sys.path.insert(0, "{train_dir}")')
            import_lines.append(f'os.chdir("{train_dir}")')  # Change to train dir for relative imports
            import_lines.append(f"from train import {trainer_class_name}")
        else:
            # Fallback: try to import from train module
            import_lines.append(f"from train import {trainer_class_name}")
    else:
        # Other custom trainer with a valid module path
        import_lines.append(f"from {import_path} import {trainer_class_name}")
    
    # Join import lines with newlines
    import_section = '\n'.join(import_lines)
   
    if not trainer.resume:
        shutil.rmtree(trainer.save_dir)  # remove the save_dir
    
    # Generate file content with imports at module level (proper Python style)
    content = f'''{import_section}

cfg = {vars(trainer.args)}

if __name__ == "__main__":
    trainer = {trainer_class_name}(cfg=cfg)
    trainer.train()
'''
    
    (USER_CONFIG_DIR / 'DDP').mkdir(exist_ok=True)
    
    with tempfile.NamedTemporaryFile(prefix="_temp_",
                                     suffix=f"{id(trainer)}.py",
                                     mode="w+",
                                     encoding='utf-8',
                                     dir=USER_CONFIG_DIR / 'DDP',
                                     delete=False) as file:
        file.write(content)
    return file.name, is_custom_trainer


def generate_ddp_command(world_size, trainer):
    import __main__  # noqa local import to avoid https://github.com/Lightning-AI/lightning/issues/15218

    file, is_custom_trainer = generate_ddp_file(trainer)

    # Build command
    torch_distributed_cmd = 'torch.distributed.run' if TORCH_1_9 else 'torch.distributed.launch'
    cmd = [
        sys.executable, '-m', torch_distributed_cmd, '--nproc_per_node', f'{world_size}', '--master_port',
        f'{find_free_network_port()}', file]
    
    # Only add command-line args for standard Ultralytics trainers
    # Custom trainers (like EventVideoYOLOv8DetectionTrainer) have the cfg embedded in the file
    # and use argparse which doesn't understand key=value format
    if not is_custom_trainer:
        exclude_args = ['save_dir']
        args = [f'{k}={v}' for k, v in vars(trainer.args).items() if k not in exclude_args]
        cmd += args
    
    return cmd, file


def ddp_cleanup(trainer, file):
    # delete temp file if created
    if f'{id(trainer)}.py' in file:  # if temp_file suffix in file
        os.remove(file)
