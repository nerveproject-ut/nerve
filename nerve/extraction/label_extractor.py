"""
@Author  :   Pietro Martinello
@Contact :   martin66@imec.be / pietromartinello.dev@gmail.com
"""

import sys

import os
from ultralytics import YOLO
from kornia.morphology import erosion
import cv2 as cv
import numpy as np
import torch
import argparse
import json
from tqdm import tqdm
from math import ceil, floor

from nerve.extraction.utils.dataset_utils import GetCategories, GetYolo2COCO_CategoryMapping
from nerve.extraction.utils.depth_utils import extract_depth_points
from nerve.extraction.utils.ffmpegReaders import VideoReader_x264

BodySegmenter = None  # lazy-loaded when --perform-human-seg is non-empty


def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="RGB video label extractor")

    parser.add_argument("--seg-model", type=str, default='yolov8n-seg.pt',
                        help="the YOLOv8 model you intend to use to perform segmentation")
    parser.add_argument("--pose-model", type=str, default='yolov8n-pose.pt',
                        help="the YOLOv8 model you intend to use to perform pose estimation")
    parser.add_argument("--perform-human-seg", type=str, default='pascal',
                        help="leave blank to avoid it, or use one of the possibilities: 'pascal', 'atr', 'lip'")
    parser.add_argument("--human-seg-weights", type=str, default='',
                        help="weights of model which should extract masks of human parts")
    parser.add_argument("--device", type=str, default='0',
                        help="choose gpu/cpu device.")
    parser.add_argument("--target-video", type=str, default='',
                        help="the RGB video from which you want to extract labels")
    
    # Eventually, extract depth information:
    parser.add_argument("--depth", type=str, default='',
                        help="Facultative. Path of video file containing distances data. It must be a .mp4 file.")
    parser.add_argument("--unit-depth", type=float, default=0.000250,
                        help="Value to be multiplied by depth pixel values to get the distance value in meters.")

    parser.add_argument("--depth-confidence", type=str, default='',
                        help="Facultative. Path of video file containing depth confidences data. It must be a .mp4 file.")
    parser.add_argument("--max-conf", type=float, default=255.0,
                        help="In case of confidence map, this is the max value of pixels (which means total confidence).")


    parser.add_argument("--output-dir", type=str, default='.',
                        help="path of output labels folder.")

    return parser.parse_args()


def extract_countours(binary_img:np.ndarray, frame_left_pad:int=0, frame_top_pad:int=0, min_contour_size:int=5):
    """
    So, unfortunately YOLOv8 result.masks.xy returns at most one single contour. In case the detected object is not a contiguous shape,
    then the models returns just a single contour.
    Nevertheless, the binary mask provided is okay.
    In order to retrieve all the contours, we need to add this bottleneck, passing via cv2 and numpy arrays..
    """

    # Finding Contours
    # Use a copy of the image e.g. edged.copy()
    # since findContours alters the image
    contours, hierarchy = cv.findContours(binary_img,
                                            cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    vertices = []
    for i in range(len(contours)):
        if len(contours[i]) <= min_contour_size:
            continue
        shape_vertices = []
        for c in contours[i]:
            x = float(c[0][0]) - frame_left_pad
            y = float(c[0][1]) - frame_top_pad
            shape_vertices.append(x)
            shape_vertices.append(y)
        vertices.append(shape_vertices)
    return vertices




def main():
    args = get_arguments()
    if args.device == 'cpu':
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = torch.device("cpu")
    else:
        gpus = [int(i) for i in args.device.split(',')]
        assert len(gpus) == 1
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        device = torch.cuda.current_device()

    # setup YOLO models
    segmenter = YOLO(args.seg_model)    # Load a segment model
    poser = YOLO(args.pose_model)       # Load a pose estimator model

    perform_human_segmentation = (args.perform_human_seg != '')
    print(args.human_seg_weights)
    if perform_human_segmentation:
        from nerve.extraction.segmentation.BodySegmenter import BodySegmenter
        human_segmenter = BodySegmenter(
            args.human_seg_weights, args.perform_human_seg)

    vidcap = cv.VideoCapture(args.target_video)
    width = int(vidcap.get(3))
    height = int(vidcap.get(4))
    fps = vidcap.get(cv.CAP_PROP_FPS)
    totalFrames = int(vidcap.get(cv.CAP_PROP_FRAME_COUNT))
    print("fps: {}; totalFrames: {}".format(fps, totalFrames))


    depth_available = args.depth != ""
    if depth_available:
        depth_file = str(args.depth)
        assert depth_file.endswith(".mp4")
        assert os.path.exists(depth_file)
        depth_conversion_unit = float(args.unit_depth)

        depth_video = VideoReader_x264(depth_file, in_pix_fmt='gray16le', out_pix_fmt=np.int16, channels=1, frames_buffer=30 * 100)
        assert depth_video.width == width and depth_video.height == height
        depth_fps = depth_video.fps
        current_depth_idx = 0

    depth_conf_available = args.depth_confidence != ""
    if depth_conf_available:
        confidence_file=str(args.depth_confidence)
        assert confidence_file.endswith(".mp4")
        assert os.path.exists(confidence_file)
        depth_conf_max_val = float(args.max_conf)

        conf_video = VideoReader_x264(confidence_file, in_pix_fmt='gray', out_pix_fmt=np.uint8, channels=1, frames_buffer=30 * 100)
        assert conf_video.width == width and conf_video.height == height
        conf_fps = conf_video.fps
        current_conf_idx = 0

    json_file = {
        "info": {
            "description": "NERVE multi-sensor office dataset",
            "url": "https://data.4tu.nl/",
            "version": "1.0",
            "fps": fps,
            "frame_height": height,
            "frame_width": width,
            "year": 2023,
            "contributor": "COCO Consortium",
            "date_created": "2023/10/01"},
        "licenses": [
            {
                "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
                "id": 1,
                "name": "Attribution-NonCommercial-ShareAlike License"
            },
            {
                "url": "http://creativecommons.org/licenses/by-nc/2.0/",
                "id": 2,
                "name": "Attribution-NonCommercial License"
            }],
        "images": [],
        "annotations": [],
        "categories": []
    }
    coco_categories = GetCategories()

    # Apparently, YOLO class order is misaligned and in a different order respect the one provided by COCO.
    # In order to keep the supercategories (available in COCO categories, not available in YOLO ones),
    # let's code a mapping from YOLO categories to COCO ones.
    category_mapping = GetYolo2COCO_CategoryMapping()

    if (not os.path.exists(args.output_dir)):
        os.mkdir(args.output_dir)

    # Let's save a black image also
    empty_image_name = "0.jpg"
    empty_image_path = os.path.join(args.output_dir, "images")
    if (not os.path.exists(empty_image_path)):
        os.mkdir(empty_image_path)
    empty_image_path = os.path.join(empty_image_path, empty_image_name)
    cv.imwrite(empty_image_path, np.zeros((height, width, 3), dtype=np.uint8))

    # it is required that the image to be processed has spatial dimensions divisible by 32, so let's pad it
    frame_v_pad = (32 - height % 32) % 32
    frame_h_pad = (32 - width % 32) % 32
    frame_top_pad = frame_v_pad//2
    frame_left_pad = frame_h_pad//2
    frame_pad = ((frame_top_pad, frame_v_pad-frame_top_pad),
                 (frame_left_pad, frame_h_pad - frame_left_pad), (0, 0))
    single_channel_frame_pad = ((frame_top_pad, frame_v_pad-frame_top_pad),
                 (frame_left_pad, frame_h_pad - frame_left_pad))
    print("original dimensions: ", height, "x", width,
          "; padded dimensions: ", height+frame_v_pad, "x", width+frame_h_pad)

    
    annotations = []
    images = []
    annotation_id = 1
    
    # if you set it to 1, it will analyze one frame every two
    # if you set it to 2, it will analyze one frame every three
    # if you set it to 0, it will analyze every frame
    skip_frames = 0
    for frame_count in tqdm(range(0, totalFrames, 1 + skip_frames)):
        success, org_image = vidcap.read()
        #eventually, skip frames
        for i in range(skip_frames):
            vidcap.read()
        if not success:
            break

        frame_time = frame_count / fps
        images.append(
            {
                "id": frame_count,
                "time_ms": frame_time * 1000,
                "license": 1,
                "file_name": empty_image_name,
                "height": height,
                "width": width,
            }
        )

        to_be_stopped = False
        if depth_available:
            # Let's get the depth frame which is (in time) as close as possible
            while(abs(frame_time - (current_depth_idx / depth_fps)) > abs(frame_time - ((current_depth_idx+1) / depth_fps))):
                current_depth_idx +=1
                if current_depth_idx >= depth_video.totalFrames:
                    print("depth video terminated. Read {} frames".format(current_depth_idx+1))
                    to_be_stopped = True
                    break
            if to_be_stopped:
                break
            depth_frame = depth_video.GetFrameFromIndex(current_depth_idx)[0]

            if depth_conf_available:
                # Eventually, let's get the confidence frame which is (in time) as close as possible
                while(abs(frame_time - (current_conf_idx / conf_fps)) > abs(frame_time - ((current_conf_idx+1) / conf_fps))):
                    current_conf_idx +=1
                    if current_conf_idx >= conf_video.totalFrames:
                        print("confidence video terminated. Read {} frames".format(current_conf_idx+1))
                        to_be_stopped = True
                        break
                if to_be_stopped:
                    break
                conf_frame = conf_video.GetFrameFromIndex(current_conf_idx)[0]



        #cv.imshow("frame", org_image)
        #cv.waitKey(0)

        padded_img = np.pad(org_image, frame_pad,
                            'constant', constant_values=0)
        if depth_available:
            depth_frame = np.pad(depth_frame, single_channel_frame_pad,'constant', constant_values=0)
            depth_frame = torch.from_numpy(depth_frame).to(device, torch.float32) * depth_conversion_unit
            if depth_conf_available:
                conf_frame = np.pad(conf_frame, single_channel_frame_pad,'constant', constant_values=0)
                conf_frame = torch.from_numpy(conf_frame).to(device, torch.float32)


        results = segmenter.track(source=padded_img, persist=True, conf=0.25, device=device,
                                  iou=0.20, save=False, show=False, imgsz=(height+frame_v_pad, width+frame_h_pad), verbose=False)

        frame_tensor = torch.tensor(padded_img, device=device)

        frame_annotations = []

        for r in results:
            boxes = r.boxes.xywh.cpu()
            if len(boxes) < 1:
                continue
            confidences = r.boxes.conf.cpu()
            classes = r.boxes.cls.cpu()
            
            # for some unknown reason, it could happen that BB are given, but there is not an ID for it.
            # In these cases, let's use -1 as tracked_id
            if r.boxes.id is not None:
                track_ids = r.boxes.id.cpu().tolist()
            else:
                track_ids = [-1 for i in range(len(boxes))]
            
            masks = r.masks.data
            
            for box, id, mask, cls, conf in zip(boxes, track_ids, masks, classes, confidences):
                tracked_id = int(id)
                new_annotation = {}
                x, y, w, h = box
                x = float(x-w/2)
                y = float(y-h/2)
                w = float(w)
                h = float(h)
                cls_idx = int(cls)
                conf = float(conf)

                # the class of the detected entity
                annotation_class = category_mapping[cls_idx]
                new_annotation["category_id"] = annotation_class
                # the unique id of this annotation
                new_annotation["id"] = annotation_id
                annotation_id += 1
                # a given ID which should be coherent for the same antity across different frames
                new_annotation["track_id"] = tracked_id
                # the ID of the image from which this annotation is extracted
                new_annotation["image_id"] = frame_count
                new_annotation["bbox"] = [
                    x - frame_left_pad, y - frame_top_pad, w, h]
                new_annotation["area"] = round(h * w)
                new_annotation["conf"] = conf
                new_annotation["iscrowd"] = 0

                np_mask = mask.cpu().numpy().astype(np.uint8) * 255
                contours_vertex = extract_countours(np_mask, frame_left_pad, frame_top_pad)
                new_annotation["segmentation"] = contours_vertex


                min_y, max_y = floor(y), ceil(y+h)
                min_x, max_x = floor(x), ceil(x+w)


                avg_depth = -1.0
                if depth_available:
                    assert depth_conf_available, "We want also this information, otherwise depth estimation would be unreliable."
                     # now, lets calculate the average distance of the segmentation map
                    target_indexes = mask
                    
                    #print("pre: ", target_indexes.sum())
                    inner_indexes = erosion(target_indexes.unsqueeze(0).unsqueeze(0), torch.ones(25, 25).to(device))[0,0]
                    #print("post: ", target_indexes.sum())
                    inner_indexes = inner_indexes.bool()
                    confs_int = conf_frame[inner_indexes]
                    
                    if confs_int.sum() == 0:
                        # depth is completely unreliable for this annotation.
                        avg_depth = -1.0
                    else:
                        confs_norm = confs_int.to(torch.float32) / depth_conf_max_val

                        weighted_average = (depth_frame[inner_indexes]@confs_norm)/confs_norm.sum()
                        avg_depth = float(weighted_average)

                    new_annotation["avg_distance"] = avg_depth


                    # let's store also valid samples containing depth information.

                    norm_conf_frame = conf_frame.to(torch.float32)/depth_conf_max_val
                    conf_threshold = 0.75 # hyperparameter
                    pix_window = 40 # hyperparameter

                    cropped_mask = inner_indexes[min_y:max_y+1, min_x:max_x+1]
                    cropped_norm_conf_frame = norm_conf_frame[min_y:max_y+1, min_x:max_x+1]
                    cropped_depth_frame = depth_frame[min_y:max_y+1, min_x:max_x+1]

                    grid = ( max(1, cropped_norm_conf_frame.shape[0] // pix_window), max(1, cropped_norm_conf_frame.shape[1] // pix_window))

                    distance_points = extract_depth_points(grid, cropped_mask, cropped_norm_conf_frame, cropped_depth_frame, False)
                    distance_points = distance_points[distance_points[:,2] >= conf_threshold]
                    distance_points[:, 0] = distance_points[:, 0] + min_x - frame_left_pad
                    distance_points[:, 1] = distance_points[:, 1] + min_y - frame_top_pad
                    
                    #let's remove the conf value now, we won't store it.
                    distance_points = torch.cat((distance_points[:, :2], distance_points[:, 3:]), dim=-1)
                    new_annotation["distance_points"] = distance_points.flatten().tolist()
                    

                if cls_idx == 0:
                    # This is a person


                    # Solution A: let's extract the pose using the original image masked by the BB
                    # --> in case of people close to each other, results can be problematic.
                    # --> in case of occlusion, it works better tho
                    """
                    target_mask = torch.zeros(
                        frame_tensor.shape, dtype=torch.uint8, device=device)
                       
                    target_mask[min_y:max_y, min_x:max_x] = 1
                    target_img = frame_tensor * target_mask
                    """


                    # Solution B: let's extract the pose using the original image masked by the segmentation mask
                    # --> in case of people close to each other, results are better.
                    # --> in case of occlusion, it could work worst tho

                    # after some tests, solution B looks definitely a better compromise
                    target_img = frame_tensor * mask.bool().unsqueeze(-1)

                    pose_results = poser.predict(
                        target_img.cpu().numpy(), device=device, conf=0.25, save=False, verbose=False)
                    pose_labels = []
                    for r in pose_results:
                        for rr in r:
                            keypoints = rr.keypoints
                            pose_labels = []
                            for i in range(len(keypoints.xy[0])):
                                xy = keypoints.xy[0, i]
                                conf = float(keypoints.conf[0, i])
                                pose_labels.append(float(xy[0]) - frame_left_pad)
                                pose_labels.append(float(xy[1]) - frame_top_pad)
                                pose_labels.append(conf)
                    new_annotation["keypoints"] = pose_labels

                    # human segmentation subparts:
                    if perform_human_segmentation:
                        bb = target_img[min_y:max_y+1, min_x:max_x+1]
                        msk = mask[min_y:max_y+1, min_x:max_x+1].bool()
                        extended_msk = msk.unsqueeze(-1)
                        green_background = torch.zeros_like(bb, device=device)
                        green_background[:, :,1] = 255
                        final = bb * extended_msk + green_background * ~extended_msk
                        
                        #cv.imshow("frame1", final.cpu().numpy())
                        #cv.waitKey(0)
                        vertices = human_segmenter(final.cpu().numpy(), msk.cpu().numpy(), 
                                                   min_x- frame_left_pad, min_y- frame_top_pad)
                        
                        new_annotation["parts"] = vertices
                        
                frame_annotations.append(new_annotation)
        
        if len(frame_annotations) > 0:
            annotations = annotations + frame_annotations

        #break #tmp
        
        #if frame_count > 60:
        #    break

    print("Created {} annotations.".format(annotation_id-1))
    
    json_file["annotations"] = annotations
    json_file["images"] = images
    json_file["categories"] = coco_categories

    res_dir = os.path.join(args.output_dir, "annotations")
    if (not os.path.exists(res_dir)):
        os.mkdir(res_dir)

    out_file = os.path.join(res_dir, "annotations.json")
    if os.path.isfile(out_file):
        os.remove(out_file)

    with open(out_file, "w") as write_file:
        json.dump(json_file, write_file)


if __name__ == "__main__":
    main()
