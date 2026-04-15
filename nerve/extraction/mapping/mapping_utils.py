import cv2
import numpy as np
import os
import sys
import json
from typing import Union
import torch
import copy

from nerve.extraction.utils.cameraParams import CameraParams, Mapping
from nerve.extraction.utils.depth_utils import retrieve_distance



def empiric_delay_ms(sensor:str) -> int:
    """It has been saw that some streams have some constant temporal shift respect others.
    This function gets an empiric temporal offset, which is meaningful only when compared with the temporal offste of another stream.

    Args:
        sensor (str): the name of the sensor (same of names in mapping files)

    Returns:
        int: A temporal offset, in mS, to compensate constant tim shifts.
    """
    # actually, the only real shift involves only DAVIS data. All other streams do not show siginificant constant time shift respect each other (as seen so far.)
    if sensor == 'davis':
        return -500
    
    return 0


def GetDelay_ms(mapping:Mapping, timing_file:str):
    """
    Join information between mapping data (--> name of the sensors involved) and timing .json file, to retrieve the estimated time delay between those two sensor streams.
    """

    assert timing_file.endswith('.json')
    assert os.path.isfile(timing_file)
     # Opening JSON file
    with open(timing_file, 'r') as openfile:
        # Reading from json file
        timings = json.load(openfile)
    try:
        src_ts = float(timings[mapping.src.name])
    except:
        print("Impossible to retrieve timestamp of sensor {} from file {} ..".format(mapping.src.name, timing_file))
        raise Exception
    try:
        dst_ts = float(timings[mapping.dst.name])
    except:
        print("Impossible to retrieve timestamp of sensor {} from file {} ..".format(mapping.dst.name, timing_file))
        raise Exception
    
    time_diff_ms = round((dst_ts - src_ts)*1000)
    empiric_time_shift = empiric_delay_ms(mapping.dst.name) - empiric_delay_ms(mapping.src.name)
    return time_diff_ms + empiric_time_shift



def get_distance_of_pixels(points: np.ndarray, samples:torch.Tensor, avg_dist:float):
    if samples is None or len(samples) < 4:
        return avg_dist
    points = torch.tensor(points.transpose(), device=samples.device)
    return retrieve_distance(points, samples).cpu().numpy()


def MapLabels(anns:list, mapping:Mapping, dst_dataset:list, device:torch.device) -> list:
    """
    Given the listed annotations, this function maps them into the destination space pointed by the mapping.
    Results are stored in the `dst_dataset` output list.

    Args:
        anns (list): A list of annotations from the source space of the mapping
        mapping (Mapping): The mapping with which you intend to transform the given annotations
        dst_image_id (int): Override the annotations' `image_id` filed with this one.
        dst_dataset (list): If different from None, resulting annotations will be appended here.
        device (torch.device): Since some calculations are done using pytorch, you can accellerate the process pointing here a specific GPU.

    Returns:
        list: The re-mapped annotations.
    """
    if dst_dataset is None:
        dst_dataset = []

    for ann in anns:
        dst_ann= copy.deepcopy(ann)
        src_avg_dist = float(dst_ann['avg_distance'])

        if mapping is None:
            # in this case, it means that user wants a mapping toward the RGB device itself,
            # aka the identity mapping. This means we just aim to clone the annotation in the new dataset,
            # eventually changing the image_id. 
            dst_dataset.append(dst_ann)
            continue

        if 'bbox' in dst_ann:
            dst_ann['bbox'] = dst_ann['bbox']
            bb = np.ndarray((2,2), dtype=np.float32)
            bb[0,0], bb[1,0] = dst_ann['bbox'][0], dst_ann['bbox'][1]
            bb[0,1], bb[1,1] = bb[0,0] + dst_ann['bbox'][2], bb[1,0] + dst_ann['bbox'][3]
            
            #print("old: {} -> {}".format(dst_ann['bbox'], bb))
            dst_bb, dst_avg_dist, visible = mapPixels(bb, mapping, src_avg_dist)
            
            if not visible:
                continue
            
            dst_ann['bbox'][0] = float(dst_bb[0,0])
            dst_ann['bbox'][1] = float(dst_bb[1,0])
            new_w = float(dst_bb[0,1] - dst_bb[0,0])
            new_h = float(dst_bb[1,1] - dst_bb[1,0])
            dst_ann['bbox'][2] = new_w
            dst_ann['bbox'][3] = new_h

            #print("new: {} -> {}".format(dst_ann['bbox'], dst_bb))
            dst_ann['area'] = new_h * new_w
            dst_ann['avg_distance'] = dst_avg_dist


        if 'distance_points' in dst_ann:
            samples = torch.tensor(dst_ann['distance_points'], device=device)
            distance_points = samples.view((len(samples)//3, 3))
            # More robust check for empty distance_points - check both numel and shape
            if distance_points.numel() == 0 or distance_points.shape[0] == 0:
                del dst_ann['distance_points']
                distance_points = None
            else:
                # let's remap also these points:
                new_coords, new_dist, _ = mapPixels(distance_points[:, :2].T.cpu().numpy(), mapping, distance_points[:, -1].cpu().numpy(), return_specific_distances=True)
                new_coords = torch.tensor(new_coords).T.round()
                new_coords = torch.cat((new_coords, torch.tensor(new_dist).unsqueeze(1)), dim=-1)
                dst_ann['distance_points'] = new_coords.flatten().tolist()

        else:
            distance_points = None


        if 'segmentation' in dst_ann:
            seg = dst_ann['segmentation']
            for s in seg:
                n_points = int(len(s)/2)
                coords = np.ndarray((2, n_points), dtype=np.float32)
                for i in range(n_points):
                    coords[0, i] = s[2*i]
                    coords[1, i] = s[2*i+1]
                
                distances = get_distance_of_pixels(coords, distance_points, src_avg_dist)
                #print(coords[:,:4])
                new_coords, _, _ = mapPixels(coords, mapping, distances)
                for i in range(new_coords.shape[1]):
                    s[2*i] = round(new_coords[0,i])
                    s[2*i +1] = round(new_coords[1,i])
                #print(new_coords[:,:4])
        
        #other things to be done in case of human label
        if dst_ann['category_id'] == 1:
            if 'keypoints' in dst_ann:
                keypoints = dst_ann['keypoints']
                n_points = int(len(keypoints)/3)
                if n_points >0:
                    coords = np.ndarray((2, n_points), dtype=np.float32)
                    for i in range(n_points):
                        coords[0, i] = keypoints[3*i]
                        coords[1, i] = keypoints[3*i+1]

                    distances = get_distance_of_pixels(coords, distance_points, src_avg_dist)
                    new_coords, _, _ = mapPixels(coords, mapping, distances, clamp_to_dest_POV=False)
                    for i in range(new_coords.shape[1]):
                        keypoints[3*i] = float(new_coords[0,i])
                        keypoints[3*i +1] = float(new_coords[1,i])

            if 'parts' in dst_ann:
                for part_type in dst_ann['parts']:
                    for s in part_type:
                        n_points = int(len(s)/2)
                        coords = np.ndarray((2, n_points), dtype=np.float32)
                        for i in range(n_points):
                            coords[0, i] = s[2*i]
                            coords[1, i] = s[2*i+1]
                        
                        #print(coords[:,:4])
                        distances = get_distance_of_pixels(coords, distance_points, src_avg_dist)
                        new_coords, _, _ = mapPixels(coords, mapping, distances)
                        for i in range(new_coords.shape[1]):
                            s[2*i] = round(new_coords[0,i])
                            s[2*i +1] = round(new_coords[1,i])
                        #print(new_coords[:,:4])
        dst_dataset.append(dst_ann)
    return dst_dataset
        


def overlapImageToBackground(foregroung_img:np.ndarray, opacity_map:np.ndarray, background_img:np.ndarray):
    #print("foreground: {}, background: {}, opacity: {}".format(foregroung_img.shape, background_img.shape, opacity_map.shape))
    opacity_map = opacity_map.astype(np.float32) / 255.0
    if foregroung_img.ndim == 2 and background_img.ndim == 3:
        foregroung_img = np.dstack([foregroung_img] * 3)
    if foregroung_img.ndim == 2:
        foregroung_img = np.expand_dims(foregroung_img, axis=-1)
    if background_img.ndim == 2:
        background_img = np.expand_dims(background_img, axis=-1)
    opacity_map = np.expand_dims(opacity_map, axis=-1)
    result = background_img * (1.0 - opacity_map) + foregroung_img * opacity_map
    return result.astype(np.uint8)


def mapPixels(pixels:np.ndarray, target_mapping:Mapping, 
              distance:Union[float, np.ndarray], clamp_to_dest_POV:bool=True,
              return_specific_distances=False) -> (np.ndarray, Union[float, np.ndarray], bool):
    """ Given a vector of pixels as seen from src camera, returns coordinates as the dst camera sees those pixels.
        As second arg, it returns the updated distance respect the dst sensor.
        As third arg, returns if the dst camera sees at least a portion of the same pixels.
        If return_specific_distances==True, then instead of the average distance scalar, a tensor is returned,
        containing distance for each input point.
    
    Returns:
        A vector containing coords as seen from the dst camera, and if dst camera sees at least one of those pixels.
    """

    assert pixels.ndim == 2
    assert pixels.shape[0] == 2
    assert target_mapping is not None

    # Handle empty pixel arrays gracefully - return early to avoid OpenCV error
    if pixels.shape[1] == 0 or pixels.size == 0:
        empty_distance = distance if isinstance(distance, np.ndarray) else np.array([])
        return pixels, empty_distance, False
    
    # Ensure pixels are in the correct format for OpenCV (contiguous float32 or float64)
    if not pixels.flags['C_CONTIGUOUS']:
        pixels = np.ascontiguousarray(pixels)
    if pixels.dtype not in [np.float32, np.float64]:
        pixels = pixels.astype(np.float32)

    if isinstance(distance, np.ndarray):
        assert distance.ndim == 1 and distance.size == pixels.shape[-1]
        distance = np.expand_dims(distance, -1)

    # -- phase 1: unproject data from pixels to 3d space

    if target_mapping.src.is_fisheye:
        # not tested this scenario, but it should work. At worst you need to reshape something.
        rays = cv2.fisheye.undistortPoints(np.transpose(pixels), target_mapping.src.intrinsics.matrix, 
                        target_mapping.src.distortions.dist_vector)
    else:
        rays = cv2.undistortPoints(np.transpose(pixels), target_mapping.src.intrinsics.matrix, 
                            target_mapping.src.distortions.dist_vector)
        
    #extracting omogeneous coords for these points
    pts = cv2.convertPointsToHomogeneous(rays).reshape(-1, 3).astype(np.float32) * distance


     # -- phase 2: reproject 3d points to destination POV (to pixels)

    trans = target_mapping.extrinsics.trans_vector
    rot, _ = cv2.Rodrigues(target_mapping.extrinsics.rot_matrix)

    if not return_specific_distances:
        # Let's re-calculate (as estimation) the avg distance between the 3d surface and the new destination POV
        new_dist = np.linalg.norm(np.average(pts - trans, axis=0))
    else:
        new_dist = np.linalg.norm(pts - trans, axis=1)

    if target_mapping.dst.is_fisheye:
        #print(pts.dtype, pts.shape)
        pts = np.expand_dims(pts, -2)
        #print("post: ", pts.shape, pts)
        dst_pixels, _ = cv2.fisheye.projectPoints(pts, rot, trans,
                        target_mapping.dst.intrinsics.matrix, 
                        target_mapping.dst.distortions.dist_vector)
    else:
        dst_pixels, _ = cv2.projectPoints(pts, rot, trans,
                        target_mapping.dst.intrinsics.matrix, 
                        target_mapping.dst.distortions.dist_vector)
        
    #print("distance: {} -->\tnew pix cords: {}".format(distance, dst_pixels))
    
    #print(dst_pixels.shape, dst_pixels)
    x = dst_pixels[:,0,0]
    y = dst_pixels[:,0,1]

    w = target_mapping.dst.width
    h = target_mapping.dst.height

    is_visible = np.count_nonzero(x[x >= 0]) > 0
    if is_visible:
        is_visible = np.count_nonzero(x[x < w]) > 0
    if is_visible:
        is_visible = np.count_nonzero(y[y >= 0]) > 0
    if is_visible:
        is_visible = np.count_nonzero(y[y < h]) > 0

    if clamp_to_dest_POV:
        x = x.clip(0, w-1)
        y = y.clip(0, h-1)

    res = np.stack((x,y), axis=0)
    return res, new_dist, is_visible


def mapPoints(points_3d:np.ndarray, target_mapping:Mapping) -> (np.ndarray):
    """ Given a vector of 3d points, returns coordinates as the dst camera sees those pixels.
    
    Returns:
        A vector containing pixel coords as seen from the dst camera, and the distance for each of this pixel
    """

    assert points_3d.ndim == 2
    assert points_3d.shape[1] == 3
    assert target_mapping is not None

    assert len(points_3d) > 0, "Apparently you didn't pass any point.. You should avoid this."

    pts = points_3d.astype(np.float32)

    # reproject 3d points to destination POV (to pixels)
    trans = target_mapping.extrinsics.trans_vector
    rot, _ = cv2.Rodrigues(target_mapping.extrinsics.rot_matrix)

    # Let's re-calculate (as estimation) the avg distance between the 3d surface and the new destination POV
    distances = np.linalg.norm(pts - trans, axis=1)

    if target_mapping.dst.is_fisheye:
        #print(pts.dtype, pts.shape)
        pts = np.expand_dims(pts, -2)
        #print("post: ", pts.shape, pts)
        dst_pixels, _ = cv2.fisheye.projectPoints(pts, rot, trans,
                        target_mapping.dst.intrinsics.matrix, 
                        target_mapping.dst.distortions.dist_vector)
    else:
        try:
            dst_pixels, _ = cv2.projectPoints(pts, rot, trans,
                            target_mapping.dst.intrinsics.matrix, 
                            target_mapping.dst.distortions.dist_vector)
        except:
            print("ERROR. points: {}:\n{}".format(points_3d.shape, points_3d))
            raise RuntimeError
        
    #print("distance: {} -->\tnew pix cords: {}".format(distance, dst_pixels))
    
    #print(dst_pixels.shape, dst_pixels)
    x = dst_pixels[:,0,0]
    y = dst_pixels[:,0,1]

    res = np.stack((x,y), axis=0).transpose()
    return res, distances

