import os
import cv2 as cv
import numpy as np
import argparse
import sys
from tqdm import tqdm
from math import ceil, floor
import ffmpeg
import torch

from mapping_utils import GetDelay_ms

from nerve.extraction.custom_coco import CustomCOCO
from nerve.extraction.utils.timers import Timer
from nerve.extraction.utils.cameraParams import *
from nerve.extraction.utils.depth_utils import retrieve_distance
from mapping_utils import *

def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Script to move COCO labels from the point of view of a camera to another. Note that you need a depth estimation of pixels in source space.")
    
    parser.add_argument("--timings", "-T", type=str, default="", help="Path of mapping data between the two sensors (must be .json).")
    parser.add_argument("--from-frame", "-f", type=int, default=-1, help="Eventually, start remampping from this image_id")
    parser.add_argument("--to-frame", "-t", type=int, default=-1, help="Eventually, stop remapping at this image_id")
    parser.add_argument("--add-delay", "-d", type=int, default=0, help="Eventually, add a temporal offset [in mS]. Can be both positive and negative.")

    requiredNamed = parser.add_argument_group('required named arguments')
    
    requiredNamed.add_argument("--input-labels", "-i", type=str, required=True, help="Path of labels file (must be .json).")
    requiredNamed.add_argument("--output", "-o", type=str, required=True, help="Path of output directory.")
    requiredNamed.add_argument("--mapping", "-M", type=str, required=True, help="Path of mapping data between the two sensors (must be .json).")


    return parser.parse_args()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = get_arguments()
    input_file = str(args.input_labels)
    mapping = str(args.mapping)
    output_dir = str(args.output)

    assert input_file.endswith(".json")
    assert mapping.endswith(".json")

    m = Mapping.from_file(mapping)

    timings_file = str(args.timings)
    delay_ms = GetDelay_ms(m, timings_file) if timings_file != '' else 0

    #adding the user defined delay (still in mS)
    delay_ms += int(args.add_delay)

    print("Total delay: {} mS".format(delay_ms))

    from_frame = int(args.from_frame)
    to_frame = int(args.to_frame)

    if not os.path.exists(output_dir):
        print("creating {} directory..".format(output_dir))
        os.mkdir(output_dir)




    with Timer("loading-JSON-file"):
        src_dataset = CustomCOCO(input_file)
    dst_dataset = CustomCOCO()

    dst_dataset.dataset['info'] = src_dataset.dataset['info']
    dst_dataset.dataset['licenses'] = src_dataset.dataset['licenses']
    dst_dataset.dataset['categories'] = src_dataset.dataset['categories']

    h,w = m.dst.height, m.dst.width
    dst_dataset.dataset["info"]["frame_height"] = h
    dst_dataset.dataset["info"]["frame_width"] = w
    dst_dataset.dataset["info"]["description"] = "{} POV".format(m.dst.name)        
        
    #dataset_numFrames, dataset_fps = src_dataset.totalImages, src_dataset.fps
    
    img_ids = list(src_dataset.imgs.keys())
    if from_frame >= 0:
        img_ids = img_ids[img_ids.index(from_frame):]
    if to_frame >= 0:
        img_ids = img_ids[:img_ids.index(to_frame)+1]

    for idx in tqdm(img_ids):
        with Timer("data-update"):
            img = src_dataset.loadImgs(idx)[0]

            dst_image_id = idx

            img['id'] = dst_image_id
            img['width'] = w
            img['height'] = h
            ann_time = img['time_ms'] + delay_ms

            if ann_time <= 0:
                continue

            img['time_ms'] = ann_time
            dst_dataset.dataset['images'].append(img)

            annIds = src_dataset.getAnnIds(imgIds=[idx])
            if len(annIds) == 0:
                continue
            # let's load the annotations for this frame of the original dataset.
            anns = src_dataset.loadAnns(annIds)

            for a in anns:
                a['image_id'] = dst_image_id # correcting the image_id with the new one

            MapLabels(anns, m, dst_dataset.dataset["annotations"], device)

            #if(idx > 10):
            #    break


    print("total annotations: ", len(dst_dataset.dataset["annotations"]))

    with Timer("writing-annotations"):
        res_dir = os.path.join(output_dir, "annotations")
        if (not os.path.exists(res_dir)):
            os.mkdir(res_dir)

        dst_file = os.path.join(res_dir, "annotations.json")
        with open(dst_file, "w") as write_file:
            json.dump(dst_dataset.dataset, write_file)

        # Let's save a black image also
        empty_image_name = "0.jpg"
        empty_image_path = os.path.join(output_dir, "images")
        if (not os.path.exists(empty_image_path)):
            os.mkdir(empty_image_path)
        empty_image_path = os.path.join(empty_image_path, empty_image_name)
        cv.imwrite(empty_image_path, np.zeros((h, w, 3), dtype=np.uint8))
    
    #Let's show resulting dataset info:
    #dataset = CustomCOCO(dst_file)

    return


if __name__ == "__main__":
    main()