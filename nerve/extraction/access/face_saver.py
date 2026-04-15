import os
import sys
import h5py
import argparse
import numpy as np
from tqdm import tqdm
import math
import shutil
import torch
import cv2
import matplotlib.pyplot as plt



from nerve.extraction.utils.cameraParams import CameraParams
from nerve.extraction.utils.timers import Timer
from nerve.extraction.utils.event_readers import hdf5_FixedDurationEventReader

from nerve.extraction.custom_coco import CustomCOCO

def get_arguments():
    parser = argparse.ArgumentParser(
        description="Rendering of labels extracted from RGB video.")

    parser.add_argument("--input","-i", type=str, required=True,
                        help="The .hdf5 file containing events data")

    parser.add_argument("--output","-o", type=str, required=True,
                        help="The path of resulting .hdf5 output file.")
    
    parser.add_argument("--annotations","-a", type=str, required=True,
                        help="The path of the .json file containing labels.")

    parser.add_argument("--camera","-c", type=str, required=True,
                        help="The path of the .json file containing camera paramters.")
    
    parser.add_argument("--events-delay", "-ed", type=float, default=0.0,
                        help="Eventually, a time offset for the events, in milliseconds. Can be also negative")

   
    return parser.parse_args()


def create_circular_mask(w:int, h:int, center:torch.Tensor, params:torch.Tensor, 
                         perceived_distance:float) -> torch.Tensor:
    
    """
    params should be float tensor containing (conf, threshold, k), with the same dimensionality of center tensor.
    
    For example, center could contains 3 couples of coordinates (one for nose, one for left eye and one for right eye).
    Each of those coords must have its own specific params

    Args:

        center: tensor with shape [..., 2] containg xy coordinates of centers
        params: tensor with shape [..., 3] containing (confidence, threshold, K), where :
            
            if confidence < threshold, then no masking is applied
            
            otherwise, radius circular mask depends by the confidence level, the distance and is multiplied by the given K.
    
    --------

    Returns:
        a tensor with shape [h, w, ...]
    """
    assert center is not None and params is not None
    assert center.shape[-1] == 2
    assert params.shape[-1] == 3
    assert params.shape[:-1] == center.shape[:-1]
    assert params.device == center.device
    
    previous_dims = center[..., 0].shape
    under_threshold = params[..., 0] < params[..., 1]
    if under_threshold.count_nonzero() == center.numel()//2:
        # no masking for any data, let's avoid unnecessary calculations.
        return torch.ones(((h,w)+ previous_dims),dtype=torch.bool, device=params.device)

    # r = conf * K / rel_dist
    radius = params[..., 0] * params[..., 2] / perceived_distance
    # if conf < threshold, then radius = 0 (no masking)
    radius[under_threshold] = 0

    xs = torch.linspace(0, w, steps=w, dtype=torch.int16)
    ys = torch.linspace(0, h, steps=h, dtype=torch.int16)
    X, Y = torch.meshgrid(xs, ys, indexing='xy')
    size =  (h, w) + previous_dims + (2,)
    maps = torch.empty((size), dtype=torch.float)

    tmp = maps.view((h,w, -1, 2))
    hidden_dim = tmp.shape[2]
    tmp[..., : , 0] = X.unsqueeze(-1).expand(-1,-1, hidden_dim)
    tmp[..., : , 1] = Y.unsqueeze(-1).expand(-1,-1, hidden_dim)

    #print("maps: {}; centers: {}".format(maps.shape, centers.shape))
    dist_from_center = torch.sqrt((maps[..., 0] - center[..., 0])**2 + (maps[..., 1] - center[..., 1])**2)
    return dist_from_center > radius



def get_indexes_of_points_out_circular_mask(w:int, h:int, center:torch.Tensor, 
                                            params:torch.Tensor, coordinates:torch.Tensor) -> torch.Tensor:
    
    """
    params should be float tensor containing (conf, threshold, k, relative_distance), with the same dimensionality of center tensor.
    
    For example, center could contains 3 couples of coordinates (one for nose, one for left eye and one for right eye).
    Each of those coords must have its own specific params

    Args:

        center: tensor with shape [..., 2] containg xy coordinates of centers
        params: tensor with shape [..., 4] containing (confidence, threshold, K, relative_distance), where :
            
            if confidence < threshold, then no masking is applied
            otherwise, radius circular mask depends by the confidence level, the distance and is multiplied by the given K.
        
            events: a vector of coordinates (x,y)
    
    --------

    Returns:
        Boolean mask of `coordinates` which are outside the circular masks.
    """
    assert center is not None and params is not None
    assert center.shape[-1] == 2
    assert params.shape[-1] == 4
    assert params.shape[:-1] == center.shape[:-1]
    assert params.device == center.device == coordinates.device
    assert coordinates.ndim ==2 and coordinates.shape[-1] == 2
    
    center = center.view((-1, 2))
    params = params.view((-1, 4))

    result = torch.ones((coordinates.shape[0],), dtype=torch.bool, device=coordinates.device)

    under_threshold = params[:, 0] < params[:, 1]
    if under_threshold.count_nonzero() == center.numel()//2:
        # no masking for any data, let's avoid unnecessary calculations.
        return result

    # r = conf * K / rel_dist
    radius = params[:, 0] * params[:, 2] / params[:, 3]
    # if conf < threshold, then radius = 0 (no masking)
    radius[under_threshold] = 0

    # not the best of efficiency, but these masks are supposed to be a few (2 to 5), so it should not be terrible.
    for i in range(radius.shape[0]):
        dist = torch.sqrt((coordinates[:, 0] - center[i, 0])**2 + (coordinates[:, 1] - center[i, 1])**2)
        result = torch.min(result, dist > radius[i])

    return result




def main():
    args = get_arguments()
    input_path = str(args.input)
    output_path = str(args.output)
    annotations_path = str(args.annotations)
    camera_path = str(args.camera)
    events_delay=float(args.events_delay)

    assert input_path.endswith(".hdf5")
    assert output_path.endswith(".hdf5")
    assert annotations_path.endswith(".json")
    assert camera_path.endswith(".json")

    assert os.path.isfile(input_path)
    assert os.path.isfile(annotations_path)
    assert os.path.isfile(camera_path)

    if torch.cuda.is_available():
        device=torch.device("cuda:0")
    else:
        device=torch.device('cpu')

    with Timer("loading-input-files"):
        dataset = CustomCOCO(annotations_path)
        time_window_size_ms = 1000.0 / dataset.fps
        camera = CameraParams.from_file(camera_path)
        event_window_iterator = hdf5_FixedDurationEventReader(input_path, duration_ms=time_window_size_ms)

    output_dir = os.path.dirname(output_path)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # since we don't know how many events there will be in the final file, but we know that at most they will be
    # the same amount of input file, first we use a file with the same shape of input file.
    # Then, after storing only the events we are interested into it, we will knoe how many they are,
    # and we'll store them properly in a file with the most suitable shape.
    # This procedure is a little slow and redudant, but looks like the only option available.
    
    input_events_number = event_window_iterator.total_events
    input_events_duration_us = event_window_iterator.duration_us

    with Timer("copying-events"):
        out_file = h5py.File(output_path, 'w')
        output_events_dataset = out_file.create_dataset('events', (input_events_number,), dtype=event_window_iterator.data.dtype,
                                                        compression="gzip", compression_opts=4,
                                                        chunks=True)
        output_events_dataset.attrs['name'] = event_window_iterator.name
        output_events_dataset.attrs['width'] = event_window_iterator.frame_width
        output_events_dataset.attrs['height'] = event_window_iterator.frame_height

        support_idx_ds = out_file.create_dataset('support_indexes', (len(event_window_iterator.support_indexes),), dtype=np.uint64, 
                                                 compression="gzip", compression_opts=4, 
                                                 chunks=True)
        support_timestep = event_window_iterator.support_timestep
        support_idx_ds.attrs['timestep_uS'] = support_timestep 
        support_idx_ds[0] = 0
        next_support_idx = 1

    dataset_frames, dataset_fps = dataset.totalImages, dataset.fps
    w,h = dataset.frame_width, dataset.frame_height
    focal_lenght = (camera.intrinsics.fx + camera.intrinsics.fy)/2
    focal_lenght = focal_lenght/1000 # focal lenght was expressed in mm, but we need it in meters.


    # hyper params
    K_nose, K_eye = 30, 40
    conf_th_nose, conf_th_eye = 0.6, 0.6


    first_ts = -1
    last_ts = -1
    labels_period_ms = 1e3 / dataset_fps
    starting_idx = 0
    if events_delay < 0:
        starting_idx = round(-events_delay / labels_period_ms)
    else:
        events_idx = round(events_delay / time_window_size_ms)
        for i in range(events_idx):
            _ = next(event_window_iterator)

    output_events = 0
    for idx in tqdm(range(starting_idx, dataset_frames)):
        with Timer("data-update"):

            #events structured as [[t, x, y, p]] -> int64
            try:
                events_window = next(event_window_iterator)
            except StopIteration:
                break
            if events_window is None:
                break
            if events_window.size == 0:
                continue

            events_window = torch.from_numpy(events_window).to(device)
            #print("source events: {}".format(len(events_window)))

            annIds = dataset.getAnnIds(imgIds=[idx])
            if len(annIds) == 0:
                continue
            # let's load the annotations for this frame of th original dataset.
            anns = dataset.loadAnns(annIds)

            centers = []
            params = []

            for ann in anns:
                # we are interested only in humans.
                if ann['category_id'] != 1:
                    continue
                if not 'keypoints' in ann:
                    continue
                 
                # NOTE: keep in mind that keypoints coords could be out of the frame.

                # faces keypoints:
                # 0 --> nose
                # 1 --> left eye
                # 2 --> right eye
                # 3 --> left ear
                # 4 --> right ear 
                keypoints = ann['keypoints']
                distance = ann['avg_distance']

                # now we need to remove events close to eyes and wherever you want,
                # keeping track of how many events you leave.
                rel_dist = distance/focal_lenght

                if len(keypoints) == 0:
                    continue

                
                ann_centers = [[keypoints[0], keypoints[1]],   # nose
                               [keypoints[3], keypoints[4]],   # left eye
                               [keypoints[6], keypoints[7]]]   # right eye

                
                ann_params = [[keypoints[2], conf_th_nose, K_nose, rel_dist],
                              [keypoints[5], conf_th_eye, K_eye, rel_dist],
                              [keypoints[8], conf_th_eye, K_eye, rel_dist]]
                
                centers = centers + ann_centers
                params = params + ann_params



            # will return a tensor shaped [h, w, 3]
            coords = events_window[:, 1:3]
            centers_torch = torch.tensor(centers, device=device)
            params_torch = torch.tensor(params, device=device)
            if centers_torch.numel() > 0 and params_torch.numel() > 0:
                masks = get_indexes_of_points_out_circular_mask(w,h, centers_torch, params_torch, coords)
                cleaned_events = events_window[masks]
            else:
                cleaned_events = events_window
            
            size = cleaned_events.shape[0]

            if output_events == 0:
                first_ts = cleaned_events[0, 0]
            last_ts = cleaned_events[-1, 0]

            events_np = cleaned_events.cpu().numpy()
            data = np.empty((size,), dtype=output_events_dataset.dtype)
            data['t'] = events_np[:,0].astype(np.int64)
            data['x'] = events_np[:,1].astype(np.uint16)
            data['y'] = events_np[:,2].astype(np.uint16)
            data['p'] = events_np[:,3].astype(np.int8)

            output_events_dataset[output_events:output_events+size] = data
            
            next_ts_threshold = first_ts + next_support_idx * support_timestep
            max_ts = cleaned_events[-1, 0]
            while max_ts >= next_ts_threshold:
                # if we are here, the next event which we aim to index is in this batch. Let's find it.
                batch_idx = (cleaned_events[:,0]>=next_ts_threshold).nonzero()[0]

                support_idx_ds[next_support_idx] = output_events + int(batch_idx)
                next_support_idx += 1
                next_ts_threshold = first_ts + next_support_idx * support_timestep
            
            
            output_events += size


    output_events_dataset.resize(output_events, 0)
    support_idx_ds.resize(next_support_idx, 0)
    output_events_dataset.attrs['total_events'] = output_events
    deltaTime = float(last_ts - first_ts)
    output_events_dataset.attrs['total_time_uS'] = deltaTime


    print("total output events: {} ({:.2f} %%  resp. input) --> time: {}".format(output_events, (100* output_events/input_events_number), deltaTime))
    







    print("all done.")

    return


# python data-access/face_saver.py -i  ~/other_sessions/calibr_20231019_test/davis_labels/events.hdf5 -o ~/other_sessions/calibr_20231019_test/davis_labels/cleaned.hdf5 -c ./data-mapping/davis.json -a ~/other_sessions/calibr_20231019_test/davis_labels/annotations/annotations.json 

if __name__ == "__main__":
    main()

    """input_events_number
    points = torch.tensor([[802, 604],[510, 415],[200, 4]])

    idx = get_indexes_of_points_out_circular_mask(w,h,centers,params, points,  rel_dist)

    print("output tensor: {}".format(points[idx]))
    #plt.imshow(mask2[...,0, 0].numpy(), cmap='gray')
    #plt.show()
    """