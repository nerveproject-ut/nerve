import numpy as np
import torch
import sys
import os
from tqdm import tqdm

from dataclasses import dataclass
from torch import nn

from nerve.extraction.utils.event_readers import hdf5_FixedDurationEventReader

def CreateHistogram(height:int, width:int, data_slice:torch.Tensor) -> torch.Tensor:
    """Creates a histogram of the events contained in the `data_slice` time-slice.

    Args:
        height (int): the output frame height.
        width (int): the output frame width.
        data_slice (torch.Tensor): a sequence of events, where the inner dimension is ['t', 'x', 'y', 'p']. Polarities are 0 when negative, 1 when positive.

    Returns:
        torch.Tensor: A tensor with shape (2,H,W) where:
        - [0,:,:] contains negative polarities
        - [1,:,:] contains positive polarities
    """
    device = data_slice.device
    
    #negative pols:
    h_neg = torch.zeros((height, width), dtype=torch.int64, device=device).flatten()
    negative_idx = data_slice[:, 3] < 0.5
    xs = data_slice[negative_idx][:,1]
    ys = data_slice[negative_idx][:,2]
    h_neg.index_add_(dim=0, source=torch.ones_like(xs), index=(xs + ys * width))
    h_neg = h_neg.view(height, width).unsqueeze(0)

    #positive pols:
    h_pos = torch.zeros((height, width), dtype=torch.int64, device=device).flatten()
    positive_idx = data_slice[:, 3] > 0.5
    xs = data_slice[positive_idx][:,1]
    ys = data_slice[positive_idx][:,2]
    h_pos.index_add_(dim=0, source=torch.ones_like(xs), index=(xs + ys * width))
    h_pos = h_pos.view(height, width).unsqueeze(0)
    
    return torch.cat((h_neg, h_pos), 0)

@dataclass
class EventSource:
    iterator : object
    frame_height : int
    frame_width : int
    temporal_window_uS: int
    accumulation_time_uS : int
    number_of_windows:int = -1 #recordings have this information, while realtime application don't.

class HistogramGenerator:

    @classmethod
    def from_recording(cls, data_path:str, temporal_window_mS:float=50.0,  accumulation_time_mS:float=None):
        """ Load a HDF5 file from the filesystem.

        Args:
            data_path (str): path of the HDF5 file.
            
            temporal_window_mS (float, optional): Processing period, in milliseconds. Defaults to 50.0.

            accumulation_time_mS (float, optional): Accumulation time, in milliseconds. The value can be both greater or smaller than `temporal_window_mS`. 
            If left to None, it will be assumed equal to the `temporal_window_mS`. Defaults to None.
        """
        it = hdf5_FixedDurationEventReader(data_path, temporal_window_mS)
        temporal_window_uS = round(temporal_window_mS*1000)
        accumulation_time_uS = temporal_window_uS if accumulation_time_mS is None else round(accumulation_time_mS*1000)
        
        return EventSource(it, it.frame_height, it.frame_width, temporal_window_uS, accumulation_time_uS, len(it))


    def __init__(self, events_source: EventSource) -> None:
        
        self.temporal_window_uS = events_source.temporal_window_uS
        self.accumulation_time_uS = events_source.accumulation_time_uS
        self.height = events_source.frame_height
        self.width = events_source.frame_width
        self.number_of_windows = events_source.number_of_windows
        self.current_time_uS = None
        self.last_events_previous_window = None #used only in case accumulation_time > temporal_window

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ## NOTE this iterator returns events over a period which is the maxium value between
        # accumulation time and temporal window.
        self.data = events_source.iterator
        
    def __len__(self):
        if self.number_of_windows < 0:
            return super().__len__()
        return self.number_of_windows

    def __iter__(self):
        return self

    def __next__(self) -> torch.Tensor:

        with torch.no_grad():
            data_slice = next(self.data) ## NOTE: output data is a sequence [['t', 'x', 'y', 'p']]
            data_slice = torch.from_numpy(data_slice).to(self.device)

            if self.current_time_uS is None:
                self.current_time_uS = data_slice[0,0]
            self.current_time_uS += self.temporal_window_uS

            if self.accumulation_time_uS < self.temporal_window_uS:
                data_slice = data_slice[data_slice[:, 0] >= self.current_time_uS - self.accumulation_time_uS]
            
            if self.accumulation_time_uS > self.temporal_window_uS:
                if self.last_events_previous_window is not None:
                    data_slice = torch.cat((self.last_events_previous_window, data_slice), dim=0)
                self.last_events_previous_window = data_slice[data_slice[:, 0] >= self.current_time_uS - (self.accumulation_time_uS - self.temporal_window_uS)]

            return CreateHistogram(self.height, self.width, data_slice), self.current_time_uS/1e3


def SimpleConvFilter(input_histogram:torch.Tensor, kernel_size=25, threshold=0.05) -> torch.Tensor:
    """Try to remove noise in the given histogram encoding. 
    To do it, for each pixel we sum up the number of events in a squared neighborhood with width equals to `kernel_size`, and we divide the result by the square of `kernel_size`.
    If the result is greater than the given `threshold`, we keep that pixel, otherwise we nullify it.

    Args:
        input_histogram (torch.Tensor): histogram with noise.
        kernel_size (int, optional): width of the neighborhood, in pixels. Defaults to 25.
        threshold (float, optional): the normalized threshold which decides if a pixel should be kept or not. The smaller, the more noise will be kept. Defaults to 0.05.

    Returns:
        torch.Tensor: histogram with less noise, with the same shape of the input tensor
    """
    assert input_histogram.ndim == 3 and input_histogram.shape[0] == 2
    with torch.no_grad():   
        weights = torch.ones((2, 1, kernel_size, kernel_size), device=input_histogram.device, requires_grad=False)
        conv = nn.Conv2d(2, 2, kernel_size, bias=False, padding_mode='reflect', padding=kernel_size//2, groups=2)
        conv.weight = nn.Parameter(weights, requires_grad=False)
        output = conv(input_histogram.to(torch.float))

        mask = output.to(torch.long)
        goods = mask >= threshold * kernel_size * kernel_size
        mask[goods] = 1
        mask[~goods] = 0
        return input_histogram * mask

if __name__ == '__main__':

    import cv2
    #h_gen = HistogramGenerator(HistogramGenerator.from_recording('/home/pietro/sessions/2023-12-15_15-02-22/davis_labels/davis_events.hdf5', 50, None))
    h_gen = HistogramGenerator(HistogramGenerator.from_recording('/home/pietro/sessions/2023-12-15_15-02-22/evk4_events.hdf5', 1e3/60, None))
        
    for data, time_ms in tqdm(h_gen):
        continue
        print(data[0,:,:].count_nonzero(), " VS ", data[0,:,:].count_nonzero())
        vis = (data * 255).cpu().numpy().clip(0,255).astype(np.uint8) # just for sake of visualization
        cv2.imshow("negative", vis[0,:,:])
        cv2.imshow("positive", vis[1,:,:])

        filtered = SimpleConvFilter(data)
        print(filtered.shape)
        cv2.imshow("filtered negative", (filtered[0,:,:] * 255).cpu().numpy().clip(0,255).astype(np.uint8))
        cv2.imshow("filtered positive", (filtered[1,:,:] * 255).cpu().numpy().clip(0,255).astype(np.uint8))
        cv2.waitKey(0)
        break

    """
    w = 3 
    h = 4
    # ['t', 'x', 'y', 'p']. Polarities are 0 when negative, 1 when positive.
    samples = [[3, 2, 0, 1], [4, 1, 0, 0], [4, 0, 3, 1], [6, 0, 3, 1], [7, 1, 0, 0], [8, 2, 0, 0], [8, 0, 3, 0]]
    input = torch.asarray(samples)

    print("input:")
    print(input)

    output = CreateHistogram(h, w, input)
    print(output.shape)

    print("negative pols:")
    print(output[0,:,:])

    print("positive pols:")
    print(output[1,:,:])
    """