import cv2
import numpy as np
import os
import glob
import sys
from tqdm import tqdm
import argparse
import shutil



parent_dirs = {
    'recording.xml'     : 'ti_radar/',
    'scenario.xml'      : 'ti_radar/meta_data/',
    'data.h5'           : 'ti_radar/captured_data/set000/',
    'ti_radar.log'      : 'ti_radar/captured_data/set000/',
    'TI_xWR14xx.xml'    : 'ti_radar/captured_data/set000/'
}


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

def organize_dir(path:str):
    if not path.endswith('/'):
        path += '/'
    
    organized = False
    for f, p in parent_dirs.items():

        file_path = path + p + f
        if os.path.isfile(file_path):
            #this file is already here (ok)
            continue
        else:
            src_path = path + f
            assert os.path.isfile(src_path), "File {} doesn't exists..".format(src_path)

            parent_dir = path
            dirs_on_path = p.split('/')
            for d in dirs_on_path:
                parent_dir = parent_dir + d + '/' 
                if not os.path.isdir(parent_dir):
                    os.mkdir(parent_dir)

            shutil.move(src_path, file_path)
            organized = True

    if organized:
        print("directory {} correctly organized.".format(path))
    else:
        print("directory {} was already organized.".format(path))

    return

def main():
    args = get_arguments()
    input_dir = str(args.input_dir)
    hierarchy = bool(args.root_directory)

    assert os.path.isdir(input_dir)

    if not hierarchy:
        organize_dir(input_dir)
    else:
        subdirs = [os.path.join(input_dir, o) for o in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir ,o))]
        subdirs.sort()
        for s in subdirs:
            organize_dir(s)




if __name__ == "__main__":
    main()