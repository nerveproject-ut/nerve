"""
PEGMA Experiment Configuration System.

Provides unified base classes for YOLOX, YOLOv8, ReYOLOv8, and RVT experiments.

Usage:
    from nerve.training.experiments import YOLOXBase, YOLOv8Base, ReYOLOv8Base, RVTBase
    
    class MyExperiment(ReYOLOv8Base):
        def __init__(self):
            super().__init__(data_yaml='/path/to/data.yaml')
            self.process_distance = True
            self.channels = 11  # 10 VTEI + 1 radar
    
    class MyRVTExperiment(RVTBase):
        def __init__(self):
            super().__init__(data_yaml='/path/to/data.yaml')
            self.process_distance = True
            self.sequence_length = 11
"""

from .base import BaseConfig
from .yolox_base import YOLOXBase
from .yolov8_base import YOLOv8Base
from .reyolov8_base import ReYOLOv8Base
from .rvt_base import RVTBase

__all__ = ['BaseConfig', 'YOLOXBase', 'YOLOv8Base', 'ReYOLOv8Base', 'RVTBase']


