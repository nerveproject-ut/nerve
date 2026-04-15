import cv2
import numpy as np
import os
import glob
import sys
from tqdm import tqdm
import argparse
import shutil


def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-dir", "-i", type=str, required=True)
    parser.add_argument("--src-name", type=str, required=True)
    parser.add_argument("--dst-name", type=str, required=True)

    return parser.parse_args()


def main():
    args = get_arguments()
    input_dir = str(args.input_dir)
    src_name = str(args.src_name)
    dst_name = str(args.dst_name)

    assert os.path.isdir(input_dir)

    sessions = [os.path.join(input_dir, o) for o in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir ,o))]
    for s in sessions:
        subdirs = [os.path.join(s, o) for o in os.listdir(s) if os.path.isdir(os.path.join(s ,o))]
        for dir in subdirs:
            base_name = os.path.basename(dir)
            if base_name == src_name:
                dst = os.path.join(os.path.dirname(dir), dst_name)
                shutil.move(dir, dst)
                print('moved {} in {}.'.format(dir, dst))




if __name__ == "__main__":
    main()