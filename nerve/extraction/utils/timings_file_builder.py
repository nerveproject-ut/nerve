import cv2
import numpy as np
import os
import glob
import sys
from tqdm import tqdm
import argparse
import shutil
import json
import h5py

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__))+'/../')
from nerve.extraction.access.syncFileReader import getDataArrivalTime


metasync_support = {
    '4mic'          : '4mic_audio.sync',
    'rgb'           : 'L515_rgb.sync',
    'depth'         : 'L515_depth.sync',
    'prophesee'     : 'prophesee/evk4_events.sync',
    'davis'         : 'davis346_events.sync'
}

timing_file_name = 'timings.json'

def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-dir", "-i", type=str, required=True)

    parser.add_argument('--root-directory', '-r', action='store_true',
                        help="Set it if the given directory is a root dir for multiple sessions.")

    return parser.parse_args()



def get_ti_radar_ts(path:str):
    if not path.endswith('/'):
        path += '/'
    
    data_file = path + "ti_radar/captured_data/set000/data.h5"
    if not os.path.isfile(data_file):
        data_file = path + "data.h5"
        if not os.path.isfile(data_file):
            return -1
    try:
        data_file = h5py.File(data_file, 'r')
    except:
        print("Something wrong happened while trying to open {}.. Apparently, file is corrupted :(".format(data_file))
        return -1
    ts_dataset = data_file['radar/dataset_1/timestamps']
    first_ts = ts_dataset[0]
    return first_ts


def get_infineon_radar_ts(path:str):
    if not path.endswith('/'):
        path += '/'

    data_file = path + "infineon_radar/captured_data/set000/data.h5"
    if not os.path.isfile(data_file):
        return -1
    try:
        f = h5py.File(data_file, 'r')
    except:
        print("Something wrong happened while trying to open {}..".format(data_file))
        return -1
    ts_dataset = f['radar/dataset_1/timestamps']
    first_ts = ts_dataset[0]
    f.close()
    return first_ts


def work_on_dir(path:str):
    if not path.endswith('/'):
        path += '/'
    
    timings = {}

    for sensor, metasync_file in metasync_support.items():

        file_path = path + metasync_file
        try:
            start_time = getDataArrivalTime(file_path)    # Let's save it in uSeconds.
        except:
            start_time = -1
        timings[sensor] = start_time
    
    # Other specific sensors:

    # 1) ti radar
    ti_radar_ts = get_ti_radar_ts(path)
    timings['ti_radar'] = ti_radar_ts

    # 2) infineon radar (timestamps from radardb HDF5, same format as TI)
    infineon_radar_ts = get_infineon_radar_ts(path)
    timings['infineon_radar'] = infineon_radar_ts

    # Serializing json
    json_object = json.dumps(timings, indent=4)
    # Writing to sample.json
    with open(path + timing_file_name, "w") as outfile:
        outfile.write(json_object)   

    print("Created timing support for dir {}.".format(path))

    return

def main():
    args = get_arguments()
    input_dir = str(args.input_dir)
    hierarchy = bool(args.root_directory)

    assert os.path.isdir(input_dir)

    if not hierarchy:
        work_on_dir(input_dir)
    else:
        subdirs = [os.path.join(input_dir, o) for o in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir ,o))]
        subdirs.sort()
        for s in subdirs:
            work_on_dir(s)




if __name__ == "__main__":
    main()