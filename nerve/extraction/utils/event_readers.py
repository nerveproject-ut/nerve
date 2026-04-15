import pandas as pd
import zipfile
from os.path import splitext
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured

import h5py
import math

import sys
import os


class hdf5_PropertiesReader:
    """
    Just a retriever of properties stored in this .hdf5 file.
    """
    def __init__(self, path_to_event_file, close_file_after_reading=True):

        file_extension = splitext(path_to_event_file)[1]
        assert (file_extension in ['.hdf5', '.h5'])
        assert os.path.isfile(path_to_event_file)

        self.data_file = h5py.File(path_to_event_file, 'r')
        self.data = self.data_file['events']
        self.max_idx = len(self.data)
        self.total_time_us = float(self.data.attrs['total_time_uS'])
        self.frame_width = int(self.data.attrs['width'])
        self.frame_height = int(self.data.attrs['height'])
        self.total_events = int(self.data.attrs['total_events'])
        self.name = str(self.data.attrs['name'])

        if close_file_after_reading:
            del self.data
            del self.data_file



class hdf5_FixedSizeReader(hdf5_PropertiesReader):
    """
    Reads events from a '.hdf5' file, and packages the events into
    non-overlapping event windows, each containing a fixed number of events.
    """

    def __init__(self, path_to_event_file, num_events=10000, start_index=0):
        print('Will use fixed size event windows with {} events'.format(num_events))
        print('Output frame rate: variable')

        super().__init__(path_to_event_file, False)

        self.num_events = num_events
        self.cur_idx = start_index

    def __len__(self):
        return math.ceil(self.total_events / self.num_events)

    def __iter__(self):
        return self

    def __next__(self):
        remaining_events = self.total_events - self.cur_idx
        if remaining_events <= 0:
            raise StopIteration
        
        max_events_readable = self.num_events if self.num_events < remaining_events else remaining_events

        d = self.data[self.cur_idx:self.cur_idx +
                        max_events_readable][['t', 'x', 'y', 'p']]
        dd = structured_to_unstructured(d, dtype=np.int64)
        self.cur_idx += max_events_readable
        return dd



class hdf5_FixedDurationEventReader(hdf5_PropertiesReader):
    """
    Reads events from a '.hdf5' file, and packages the events into
    non-overlapping event windows, each of a fixed duration.
    """

    def __init__(self, path_to_event_file, duration_ms=50.0, start_index=0, event_type="all"):
        print('Will use fixed duration event windows of size {:.2f} ms'.format(
            duration_ms))
        print('Output frame rate: {:.1f} Hz'.format(1000.0 / duration_ms))

        super().__init__(path_to_event_file, False)

        self.first_ts = int(self.data[0]['t'])
        self.last_stamp = self.first_ts

        try:
            # Let's load it completely into memory. they should be just some tens of MB.
            support_indexes = self.data_file['support_indexes']
            self.support_available = True
            self.support_indexes = np.array(support_indexes)
            self.support_timestep = int(support_indexes.attrs['timestep_uS'])
            
            print("Found support indexing system, with timestep={} uS.".format(self.support_timestep))

        except KeyError:
            self.support_available = False

        self.duration_us = int(duration_ms * 1000.0)
        self.cur_idx = start_index

        self.only_positive_events = event_type == 'on'
        self.only_negative_events = event_type == 'off'

    def __iter__(self):
        return self
    
    def __len__(self):
        return math.ceil(self.total_time_us / self.duration_us)


    def __next__(self):
            
        if self.cur_idx >= self.max_idx:
            raise StopIteration
        
        if not self.support_available:
            # slow way..
            event_list = []
            while True:
                pol = self.data[self.cur_idx]
                self.cur_idx += 1
                x, y, pol, t = pol['x'], pol['y'], pol['p'], pol['t']
                
                max_ts = self.last_stamp + self.duration_us

                if t > max_ts or self.cur_idx == self.max_idx :
                    self.last_stamp = max_ts
                    event_window = np.array(event_list)

                    if self.only_positive_events:
                        event_window = event_window[event_window[2]>0.5]
                    if self.only_negative_events:
                        event_window = event_window[event_window[2]<0.5]
                    return event_window
                
                else:
                    event_list.append([t, x, y, pol])
        
        else:
            # quick way! but you can do only if data support it.

            max_ts = self.last_stamp + self.duration_us
            i = (max_ts - self.first_ts) // self.support_timestep
            if i >= len(self.support_indexes):
                i = len(self.support_indexes) - 1
            
            min_idx = int(self.support_indexes[i])

            idx = min_idx
            while True:
                if self.data[idx]['t'] > max_ts:
                    break
                idx += 1
                if idx == self.total_events:
                    break

            self.last_stamp = max_ts
            
            #Let's extract this temporal window of events:
            pols = self.data[self.cur_idx : idx]
            self.cur_idx = idx

            events = pols[['t', 'x', 'y', 'p']]
            if self.only_positive_events:
                events = events[events['p']>0.5]
            if self.only_negative_events:
                events = events[events['p']<0.5]
            d = structured_to_unstructured(events, dtype=np.int64)
            return d