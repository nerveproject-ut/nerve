
import abc
import sys
import os
import torch
from torch.nn.functional import avg_pool2d, pad
import numpy as np
import cv2
import h5py
import hdf5plugin

from nerve.processing.histogram import HistogramGenerator, SimpleConvFilter
from nerve.extraction.mapping.mapping_utils import mapPoints, GetDelay_ms, overlapImageToBackground
from nerve.extraction.utils.cameraParams import Mapping
from nerve.extraction.utils.HiddenPrints import HiddenPrints

try:
    from nerve.radar import get_backend as _get_radar_backend
    _RADAR_AVAILABLE = True
except ImportError:
    _RADAR_AVAILABLE = False

from nerve.extraction.custom_coco import CustomCOCO

# Import event representation functions for ReYOLOv8 compatibility
try:
    from nerve.processing.event_representations import process_events
except ImportError:
    # Fallback if not in the same directory
    from nerve.processing.event_representations import process_events



### This is a highly concrete code, usable from within dataset_creator.py.
# It aims to facilitate the handling and (eventual) pre-processing of data from various sensors which are going to be included into a custom dataset.
# So far, here it is supported just the case of DVS and TI radar data, but it can be improved with new modes.


class DataSource(abc.ABC):
    def __init__(self, settings:dict, transformation_function=None) -> None:

        assert 'data' in settings
        self.data = settings['data']

        if 'mapping' in settings and 'timings' in settings:
            self.mapping = Mapping.from_file(settings['mapping'])
            self.delay_ms = GetDelay_ms(self.mapping, settings['timings'])

        # since we need to perform temporal alignment, 
        # let's do it by defining a "target_time" --> where we would like to be
        # and a "actual_time" --> where we effectively are
        self.target_time_ms = 0
        self.actual_time_ms = 0
        self.transformation_func = transformation_function
    
    @abc.abstractmethod
    def GetFramePeriod_ms(self):
        pass
    
    def GetInitialDelay_ms(self):
        return self.delay_ms
    
    @abc.abstractmethod
    def __next__(self):
        """Note: what you are returning from here, is what later will be passed as `data` to method `StoreData`
        """
        pass

    def TotalNumberOfFrames(self):
        return len(self)

    @abc.abstractmethod
    def __len__(self):
        pass

    @abc.abstractmethod
    def Close(self):
        pass

    def Transform_annotation(self, annotation:dict)->dict:
        if self.transformation_func is None:
            return annotation
        else:
            return self.transformation_func(annotation, self)

    def Time_forward_ms(self, deltatime_ms:int, return_frame=True):
        # just MOVE FORWARD in time, not backwards.
        
        frame_period = self.GetFramePeriod_ms()
        if return_frame:
            assert deltatime_ms >= frame_period
        else:
            # we just aim to perform temporal alignment
            assert deltatime_ms >= 0
        
        self.target_time_ms += deltatime_ms
        last_data = None
        while self.actual_time_ms <= self.target_time_ms - frame_period:
            last_data = next(self)
            self.actual_time_ms += frame_period
        
        if (self.target_time_ms - self.actual_time_ms)/frame_period > 0.5:
            last_data = next(self)
            self.actual_time_ms += frame_period

        if return_frame:
            return last_data
    
    @abc.abstractmethod
    def StoreData(self, data, directory_path:str, index:int) -> str:
        """Store the previously extracted data on disk. The value od `data` depends by the source itself. It must the return the path where data were stored.

        Args:
            data : The data extracted from this source using method `next`
            directory_path (str): The directory where you can store data
            index (int): A unique value to identify this data on disk.

        Returns:
            str: the complete path were you stored data.
            height: int
            width: int
        """
        pass

class DVS_Source(DataSource):
    def __init__(self, settings:dict, transformation_function=None, verbose=False) -> None:

        super().__init__(settings, transformation_function)
        
        self.frame_period_ms = settings['frame_period_ms']
        
        # Processing parameters
        self.accumulation_ms = settings.get('accumulation_ms', settings['frame_period_ms'])
        self.max_events = settings.get('max_events', 15)
        self.avg_pool_kernel = settings.get('avg_pool_kernel', 1)
        self.pad_size = settings.get('pad_size', [0, 0])  # [width_pad, height_pad] for making dims multiple of 32
        
        self.use_filter = settings['use_filter'] if 'use_filter' in settings else False
        
        # Event representation options (works with both HDF5 and PNG output)
        self.event_representation = settings.get('event_representation', None)  # vtei, voxel_grid, shist, mdes, ev_temporal_volume
        self.store_as_hdf5 = settings.get('store_as_hdf5', False)
        self.representation_bins = settings.get('bins', 10)
        
        # AUTO-ENABLE raw events when event representation is specified
        if self.event_representation:
            self.use_raw_events = True
            output_format = "HDF5" if self.store_as_hdf5 else "PNG"
            if verbose:
                print(f"Auto-enabled raw events for {self.event_representation} representation ({self.representation_bins} bins, {output_format} output)")
        else:
            self.use_raw_events = settings.get('use_raw_events', False)
        
        data_path = str(settings['data_path'])
        assert data_path.endswith('.hdf5'), "DVS file must be a .hdf5 file"
        assert os.path.isfile(data_path), "Apparently, the DVS file located at {} doesn't exist..".format(data_path)

        # If using raw events for ReYOLOv8, load them directly
        if self.use_raw_events:
            from nerve.extraction.utils.event_readers import hdf5_FixedDurationEventReader
            
            if verbose:
                self.reader = hdf5_FixedDurationEventReader(
                    data_path, 
                    duration_ms=self.frame_period_ms
                )
            else:
                with HiddenPrints():
                    self.reader = hdf5_FixedDurationEventReader(
                        data_path, 
                        duration_ms=self.frame_period_ms
                    )
            
            self.native_width = self.reader.frame_width
            self.native_height = self.reader.frame_height
            self.raw_events_file = None
            
            if verbose:
                print(f"Using raw event reader: {self.native_width}x{self.native_height}, {self.frame_period_ms}ms windows")
        else:
            # Use existing HistogramGenerator for PNG output
            if verbose:
                self.reader = HistogramGenerator(HistogramGenerator.from_recording(data_path, self.frame_period_ms, self. accumulation_ms))
            else:
                with HiddenPrints():
                    self.reader = HistogramGenerator(HistogramGenerator.from_recording(data_path, self.frame_period_ms, self. accumulation_ms))
            self.native_width = self.reader.width
            self.native_height = self.reader.height
            self.raw_events_file = None
            
            if verbose:
                print(f"Using histogram generator: {self.native_width}x{self.native_height}, 2 channels")
        
        # Calculate actual output dimensions: native / pool_kernel + padding
        # NO resizing - just pooling and padding like the original implementation
        self.width = self.native_width // self.avg_pool_kernel + self.pad_size[0]
        self.height = self.native_height // self.avg_pool_kernel + self.pad_size[1]
        
        if verbose:
            print(f"Output dimensions: {self.width}x{self.height} (native: {self.native_width}x{self.native_height}, pool: {self.avg_pool_kernel}, pad: {self.pad_size})")

        

    def GetFramePeriod_ms(self):
        return self.frame_period_ms
    
    def __next__(self):
        data = next(self.reader)
        
        # If using raw event reader, convert format to structured array for event_representations.py
        if self.use_raw_events:
            # hdf5_FixedDurationEventReader returns unstructured array [['t', 'x', 'y', 'p']]
            # We need to convert it to structured array with proper field names
            if isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] == 4:
                # Convert from unstructured [[t,x,y,p], ...] to structured array
                structured_data = np.zeros(
                    data.shape[0],
                    dtype=[('t', 'f8'), ('x', 'i4'), ('y', 'i4'), ('p', 'i1')]
                )
                structured_data['t'] = data[:, 0]
                structured_data['x'] = data[:, 1].astype(np.int32)
                structured_data['y'] = data[:, 2].astype(np.int32)
                structured_data['p'] = data[:, 3].astype(np.int8)
                
                # Return with timestamp (use mean time of events)
                time_ms = structured_data['t'].mean() / 1000.0 if len(structured_data) > 0 else 0
                return (structured_data, time_ms)
        
        # Otherwise return as-is (histogram from HistogramGenerator)
        return data
    
    def __len__(self):
        return len(self.reader)
    
    def Close(self):
        if self.reader:
            # Close properly based on reader type
            if hasattr(self.reader, 'data_file'):
                # hdf5_FixedDurationEventReader has a data_file attribute
                if self.reader.data_file is not None:
                    self.reader.data_file.close()
            del self.reader
        if self.raw_events_file:
            self.raw_events_file.close()

    def StoreData(self, data, directory_path:str, index:int):
        """Store DVS data in either PNG (default) or HDF5 (ReYOLOv8) format."""
        
        # Check if we should use ReYOLOv8 HDF5 storage
        if self.store_as_hdf5 and self.event_representation:
            return self.StoreDataHDF5(data, directory_path, index)
        
        # If event representation is specified but storing as PNG, apply representation first
        if self.event_representation and self.use_raw_events:
            return self.StoreDataPNGWithRepresentation(data, directory_path, index)
        
        # Default PNG storage (basic histogram, existing behavior)
        (dvs_data, time_ms) = data

        if self.use_filter:
            dvs_data = SimpleConvFilter(dvs_data)
        dvs_data = torch.clamp(dvs_data, max=self.max_events) / self.max_events
        resize_dvs_data = avg_pool2d(dvs_data, kernel_size=self.avg_pool_kernel)
        padding = (0, self.pad_size[0], 0, self.pad_size[1])
        resize_dvs_data = pad(resize_dvs_data, padding, "constant", 0)
        
        # How to make the set the third channel? Here we are doing a simple sum, but it can be changed.
        additional_channel = (resize_dvs_data[0] + resize_dvs_data[1])/2
        resize_dvs_data = torch.cat((resize_dvs_data, additional_channel.unsqueeze(0)), dim=0) # shape ([3, H, W])
        resize_dvs_data = resize_dvs_data.cpu().numpy()

        back_frame = np.transpose((resize_dvs_data*255).astype(np.uint8), (1, 2, 0))
        frame_name = 'histogram_img__' + str(index).zfill(10) + '.png'
        frame_path = os.path.join(directory_path, frame_name)
        cv2.imwrite(frame_path, back_frame)

        return frame_path, resize_dvs_data.shape[1], resize_dvs_data.shape[2]

    def StoreDataHDF5(self, data, directory_path:str, index:int):
        """
        Store DVS data in HDF5 format for ReYOLOv8 compatibility.
        Uses event representations (vtei, voxel_grid, shist, mdes).
        """
        (dvs_data, time_ms) = data
        
        # If dvs_data is raw events (structured array), process them
        if isinstance(dvs_data, np.ndarray) and dvs_data.dtype.names:
            # Raw events with x, y, t, p fields
            # Process at native resolution
            representation = process_events(
                dvs_data, 
                self.event_representation, 
                self.representation_bins,
                self.native_height, 
                self.native_width
            )
            
            # Apply padding if specified (for making dimensions multiple of 32, etc.)
            if self.pad_size[0] > 0 or self.pad_size[1] > 0:
                # Pad format for numpy: ((before_1, after_1), (before_2, after_2), ...)
                # representation is (C, H, W), we pad H and W
                pad_width = ((0, 0), (0, self.pad_size[1]), (0, self.pad_size[0]))
                representation = np.pad(representation, pad_width, mode='constant', constant_values=0)
        else:
            # Already processed tensor data - convert to numpy
            if isinstance(dvs_data, torch.Tensor):
                representation = dvs_data.cpu().numpy()
            else:
                representation = dvs_data
        
        # Store as HDF5 file
        frame_name = 'event_frame__' + str(index).zfill(10) + '.h5'
        frame_path = os.path.join(directory_path, frame_name)
        
        with h5py.File(frame_path, 'w') as hf:
            hf.create_dataset(
                'events', 
                data=representation,
                compression='gzip',
                compression_opts=4
            )
        
        return frame_path, representation.shape[1], representation.shape[2]
    
    def StoreDataPNGWithRepresentation(self, data, directory_path:str, index:int):
        """
        Store DVS data as PNG after applying event representation.
        This allows direct PNG generation with VTEI, voxel_grid, etc. 
        without the two-step HDF5→PNG conversion process.
        
        Args:
            data: Tuple of (dvs_data, time_ms)
            directory_path: Output directory
            index: Frame index
            
        Returns:
            Tuple of (frame_path, height, width)
        """
        (dvs_data, time_ms) = data
        
        # Apply event representation to raw events
        if isinstance(dvs_data, np.ndarray) and dvs_data.dtype.names:
            # Raw events with x, y, t, p fields
            # Process at native resolution
            representation = process_events(
                dvs_data, 
                self.event_representation, 
                self.representation_bins,
                self.native_height, 
                self.native_width
            )
            
            # Apply padding if specified (for making dimensions multiple of 32, etc.)
            if self.pad_size[0] > 0 or self.pad_size[1] > 0:
                # Pad format for numpy: ((before_1, after_1), (before_2, after_2), ...)
                # representation is (C, H, W), we pad H and W
                pad_width = ((0, 0), (0, self.pad_size[1]), (0, self.pad_size[0]))
                representation = np.pad(representation, pad_width, mode='constant', constant_values=0)
        else:
            # Already processed tensor data
            if isinstance(dvs_data, torch.Tensor):
                representation = dvs_data.cpu().numpy()
            else:
                representation = dvs_data
        
        # representation is now (C, H, W) where C could be 10 (VTEI), 20 (voxel_grid), etc.
        channels, height, width = representation.shape
        
        # For multi-channel data (>3 channels), store as NPY to preserve all channels
        # This allows YOLOX/YOLOv8 with custom dataloaders to use full event representation info
        if channels > 3:
            frame_name = 'event_rep__' + str(index).zfill(10) + '.npy'
            frame_path = os.path.join(directory_path, frame_name)
            
            # Normalize to [0, 1] for consistency with training
            normalized = np.zeros_like(representation, dtype=np.float32)
            for c in range(channels):
                channel_data = representation[c]
                min_val = channel_data.min()
                max_val = channel_data.max()
                if max_val > min_val:
                    normalized[c] = (channel_data - min_val) / (max_val - min_val)
                else:
                    normalized[c] = channel_data
            
            np.save(frame_path, normalized)
            return frame_path, height, width
        
        # For 1-3 channels, store as PNG (standard image format)
        # Normalize for visualization (same logic as hdf5_to_png.py)
        normalized = np.zeros_like(representation, dtype=np.float32)
        for c in range(channels):
            channel_data = representation[c]
            min_val = channel_data.min()
            max_val = channel_data.max()
            
            if max_val > min_val:
                normalized[c] = (channel_data - min_val) / (max_val - min_val)
            else:
                normalized[c] = channel_data
        
        # Convert to uint8
        normalized = (normalized * 255).astype(np.uint8)
        
        # Create PNG image based on number of channels
        if channels == 1:
            # Grayscale
            img_data = normalized[0]
        elif channels == 2:
            # 2-channel (histogram) - convert to RGB for visualization
            # Channel 0 = positive events (red), Channel 1 = negative events (blue)
            img_data = np.zeros((height, width, 3), dtype=np.uint8)
            img_data[:, :, 2] = normalized[0]  # Red
            img_data[:, :, 0] = normalized[1]  # Blue
        else:  # channels == 3
            # RGB - transpose to (H, W, C)
            img_data = np.transpose(normalized, (1, 2, 0))
        
        # Save as PNG
        frame_name = 'event_frame__' + str(index).zfill(10) + '.png'
        frame_path = os.path.join(directory_path, frame_name)
        cv2.imwrite(frame_path, img_data)
        
        return frame_path, height, width
            
    
class Prophesee_source(DVS_Source):
    def __init__(self, settings: dict, transformation_function=None, verbose=False) -> None:
        assert 'data' in settings and settings['data'] == 'prophesee'
        super().__init__(settings, transformation_function, verbose)

class DAVIS_source(DVS_Source):
    def __init__(self, settings: dict, transformation_function=None, verbose=False) -> None:
        assert 'data' in settings and settings['data'] == 'davis'
        super().__init__(settings, transformation_function, verbose)


class Radar_source(DataSource):
    def __init__(self, settings: dict, transformation_function=None) -> None:
        super().__init__(settings, transformation_function)
        self.space_shape = (self.mapping.dst.width, self.mapping.dst.height) # W x H
        self.output_shape = settings['output_shape']    # W x H
        self.max_distance = settings["max_dist"]

        try:
            self.store_fft = settings['store_fft']
        except:
            self.store_fft = True

        try:
            self.project_poc = settings['project_poc']
        except:
            self.project_poc = True

        # ReYOLOv8 radar fusion mode: 'separate', 'fused', or 'both'
        self.fusion_mode = settings.get('fusion_mode', 'separate')
        self.dvs_fusion_channel = settings.get('dvs_fusion_channel', 2)  # Which DVS channel to replace (default: R channel = 2 in BGR)

        self.radar_dilation = settings.get('radar_dilation', 0)

        self.resize_height_ratio = self.output_shape[1] / self.mapping.dst.height
        self.resize_width_ratio = self.output_shape[0] / self.mapping.dst.width

        if not _RADAR_AVAILABLE:
            raise ImportError(
                "No radar backend is available. Install a backend package "
                "or subclass nerve.radar.RadarBackend and call "
                "nerve.radar.register_backend()."
            )

        assert os.path.isdir(settings['data_path']), "Apparently, the radar directory located at {} doesn't exist..".format(settings['data_path'])

        BackendCls = _get_radar_backend()
        self._backend = BackendCls.from_recording(settings['data_path'])
        self._num_frames = self._backend.get_num_frames()
        self._frame_idx = 0

        if 'frame_period_ms' in settings:
            self.frame_period_ms = settings['frame_period_ms']
        elif 'frame_period_s' in settings:
            self.frame_period_ms = settings['frame_period_s'] * 1e3
        else:
            self.frame_period_ms = self._backend.get_frame_period() * 1e3

    def GetFramePeriod_ms(self):
        return self.frame_period_ms
    
    def __next__(self):
        if self._frame_idx >= self._num_frames:
            raise StopIteration
        idx = self._frame_idx
        points = self._backend.get_point_cloud(idx)
        fft = self._backend.get_range_doppler(idx)
        velocities = points[:, 3:] if points.shape[1] > 3 else np.empty((len(points), 0))
        coords = points[:, :3] if points.shape[1] >= 3 else points
        self._frame_idx += 1
        return (idx, 0, coords, velocities, fft)

    def __len__(self):
        return self._num_frames

    def Close(self):
        self._backend.close()
        del self._backend


    def create_frame_from_points(self, points):

        point_cloud_frame = np.zeros((self.output_shape[1], self.output_shape[0]))
        if len(points) == 0:
            return point_cloud_frame
        
        mapping = self.mapping
        pixels, distances = mapPoints(points, mapping)

        visible_mask = np.where((pixels[:, 0] >= 0) & (pixels[:, 0] < self.space_shape[0]) & (pixels[:, 1] >= 0) & (pixels[:, 1] < self.space_shape[1]))
        pixels = pixels[visible_mask]
        distances = distances[visible_mask]

        point_cloud_frame = np.zeros((self.output_shape[1], self.output_shape[0]))
        pos_list = {}
        for i, p in enumerate(pixels):
            pos = (int(p[0] * self.resize_width_ratio), int(p[1] * self.resize_height_ratio))
            if pos in pos_list:
                if distances[i] < pos_list[pos]:
                    pos_list[pos] = distances[i]

            else:
                pos_list[pos] = distances[i]
    
        for pos in pos_list.keys():
            point_cloud_frame[pos[1],pos[0]] = pos_list[pos]


        return point_cloud_frame

    def StoreData(self, data, directory_path:str, index:int):
        (idx, ts, points, vels, fft) = data
        pc_frame = self.create_frame_from_points(points)

        if self.project_poc:
            back_frame = (pc_frame * 255/self.max_distance).clip(0, 255).astype(np.uint8)

            if self.radar_dilation > 0 and np.count_nonzero(back_frame) > 0:
                from nerve.generation.creator import dilate_sparse_radar
                back_frame = dilate_sparse_radar(back_frame, kernel_size=self.radar_dilation)

            frame_name = 'point_cloud_img__' + str(index).zfill(10) + '.png'
            frame_path = os.path.join(directory_path, frame_name)
            cv2.imwrite(frame_path, back_frame)
    

        if self.store_fft:
            fft_name = 'fft__' + str(index).zfill(10) + '.npy'
            fft_path = os.path.join(directory_path, fft_name)
            np.save(fft_path, fft)

        # in this case we stored two different data, but that's not a problem. Just return the most significant.
        return frame_path, pc_frame.shape[0], pc_frame.shape[1]
    

class RGB_annotations_source(DataSource):
    def __init__(self, settings: dict, verbose=False) -> None:
        assert not 'mapping' in settings # annotations don't need any (other) mapping, we will use the ones provided by the other streams.
        super().__init__(settings)
        self.delay_ms = 0 # rgb annotations starts at time 0

        annotations_path = str(settings['data_path'])
        assert annotations_path.endswith('.json'), "Annotation file mus be a .json file"
        assert os.path.isfile(annotations_path), "Apparently, the annotation file located at {} doesn't exist..".format(annotations_path)
        
        self.classes =  settings['only_classes'] if 'only_classes' in settings else []

        if verbose:
            self.annotations = CustomCOCO(annotations_path)
        else:
            with HiddenPrints():
                self.annotations = CustomCOCO(annotations_path)
        self.period_ms = 1e3 / self.annotations.fps
        self.current_index = 0

    def GetFramePeriod_ms(self):
        return self.period_ms
    
    def __len__(self):
        return self.annotations.totalImages
    
    def __next__(self):
        annIds = self.annotations.getAnnIds(imgIds=[self.current_index], catIds=self.classes)
        if len(annIds) == 0:
            return None
        
        # let's load the annotations for this frame.
        anns = self.annotations.loadAnns(annIds)
        self.current_index += 1
        return anns
    
    def Close(self):
        del self.annotations

    def StoreData(self, data, directory_path:str, index:int):
        # Labels are stored in a different way, since they have to be mapped toward each of the output streams, and store in the respective annotation file.
        # So, this method doesn't make sense for annotations handling, and it must not be invoked.
        raise NotImplementedError