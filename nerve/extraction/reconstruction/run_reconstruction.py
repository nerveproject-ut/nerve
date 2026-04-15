import torch
from utils_dvs.loading_utils import load_model, get_device
import numpy as np
import argparse
import pandas as pd
from tqdm import tqdm

from utils_dvs.inference_utils import events_to_voxel_grid, events_to_voxel_grid_pytorch
import time
from image_reconstructor import ImageReconstructor
from utils_dvs.inference_options import set_inference_options
import os
import sys

from nerve.extraction.utils.timers import Timer
from nerve.extraction.utils.event_readers import *

def main():
    default_weights = os.path.dirname(os.path.realpath(__file__))+"/weights/E2VID_lightweight.pth.tar"
    parser = argparse.ArgumentParser(
        description='Evaluating a trained network')
    parser.add_argument('-c', '--path_to_model', default=default_weights, type=str,
                        help='path to model weights')
    parser.add_argument('-i', '--input_file', required=True, type=str)
    parser.add_argument('--fixed_events_number', dest='fixed_duration', action='store_false')
    parser.set_defaults(fixed_duration=True)
    parser.add_argument('-N', '--window_size', default=None, type=int,
                        help="Size of each event window, in number of events. Ignored if --fixed_duration=True")
    parser.add_argument('--fps', default=60.0, type=float,
                        help="How often to crate a frame? Ignored if --fixed_duration=False")
    parser.add_argument('--num_events_per_pixel', default=0.35, type=float,
                        help='in case N (window size) is not specified, it will be \
                              automatically computed as N = width * height * num_events_per_pixel')
    parser.add_argument('--skipevents', default=0, type=int)
    parser.add_argument('--suboffset', default=0, type=int)
    
    parser.add_argument('--output_type', '-t', type=str, default="lossy-video",
                        help="Do you want as output images, lossy video or lossless video? \
                        Choose between 'img', 'lossy-video' or 'lossless-video'")

    parser.add_argument('--compute_voxel_grid_on_cpu', dest='compute_voxel_grid_on_cpu', action='store_true')
    parser.set_defaults(compute_voxel_grid_on_cpu=False)

    #set to "all", "on" or "off" --> if you want to process just a type of events
    parser.add_argument('--event_type', default="all", type=str)

    set_inference_options(parser)

    args = parser.parse_args()


    # Read sensor size from the first first line of the event file
    path_to_events = str(args.input_file)

    if path_to_events.endswith(".rad"):
        #need to convert it first.
        output_dir = str(args.output_folder)
        result_path = os.path.join(output_dir, "events.hdf5")

        if os.path.isfile(result_path):
            print("Found conversion of {} in {} .. using it.".format(path_to_events, result_path))
        else:
            convert_script = os.path.dirname(os.path.realpath(__file__))+'/../data_access/radar-dvs/rad_polarities_to_hdf5.py'
            assert os.path.isfile(convert_script), "Impossible to find conversion file.."
            print("Performing conversion from {} to {}".format(path_to_events, result_path))
            os.system("python {} -i {} -o {}".format(convert_script, path_to_events, result_path))

        path_to_events = result_path

    file_properties = hdf5_PropertiesReader(path_to_events)
    width = file_properties.frame_width
    height = file_properties.frame_height

    print('Sensor size: {} x {}'.format(width, height))

    # Load model
    model = load_model(args.path_to_model)
    device = get_device(args.use_gpu)

    model = model.to(device)
    model.eval()

    reconstructor = ImageReconstructor(model, height, width, args)

    """ Read chunks of events using Pandas """

    # Loop through the events and reconstruct images
    N = args.window_size
    if not args.fixed_duration:
        if N is None:
            N = int(width * height * args.num_events_per_pixel)
            print('Will use {} events per tensor (automatically estimated with num_events_per_pixel={:0.2f}).'.format(
                N, args.num_events_per_pixel))
        else:
            print('Will use {} events per tensor (user-specified)'.format(N))
            mean_num_events_per_pixel = float(N) / float(width * height)
            if mean_num_events_per_pixel < 0.1:
                print('!!Warning!! the number of events used ({}) seems to be low compared to the sensor size. \
                    The reconstruction results might be suboptimal.'.format(N))
            elif mean_num_events_per_pixel > 1.5:
                print('!!Warning!! the number of events used ({}) seems to be high compared to the sensor size. \
                    The reconstruction results might be suboptimal.'.format(N))

    initial_offset = args.skipevents
    sub_offset = args.suboffset
    start_index = initial_offset + sub_offset

    if args.compute_voxel_grid_on_cpu:
        print('Will compute voxel grid on CPU.')

    if args.fixed_duration:
        durations_ms = 1000.0 / args.fps 
        event_window_iterator = hdf5_FixedDurationEventReader(path_to_events,
                                                        duration_ms=durations_ms,
                                                        start_index=start_index, event_type=args.event_type)
    else:
        event_window_iterator = hdf5_FixedSizeReader(path_to_events, num_events=N, start_index=start_index)


    for event_window in tqdm(event_window_iterator):

        with Timer('Building event tensor'):
            if args.compute_voxel_grid_on_cpu:
                event_tensor = events_to_voxel_grid(event_window,
                                                    num_bins=model.num_bins,
                                                    width=width,
                                                    height=height)
                event_tensor = torch.from_numpy(event_tensor)
            else:
                event_tensor = events_to_voxel_grid_pytorch(event_window,
                                                            num_bins=model.num_bins,
                                                            width=width,
                                                            height=height,
                                                            device=device)
        
        reconstructor.update_reconstruction(event_tensor)


if __name__ == "__main__":
    main()