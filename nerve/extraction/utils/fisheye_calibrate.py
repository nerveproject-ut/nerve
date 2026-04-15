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
    parser = argparse.ArgumentParser(description="Script to move COCO labels from the point of view of a camera to another. Note that you need a depth estimation of pixels in source space.")

    parser.add_argument("--image-dir", "-i", type=str, required=True,
                        help="Path of directory containing -png images")
    parser.add_argument("--rows", type=int, default=8,
                        help="Number of rows of the pattern checkboard")
    parser.add_argument("--cols", type=int, default=11,
                        help="Number of columns of the pattern checkboard")
    parser.add_argument('--store-goods', action='store_true',
                        help="Save images where corners have been detected in a subfolder?")

    return parser.parse_args()

def main():
    args = get_arguments()
    input_dir = str(args.image_dir)

    save_good_imgs = bool(args.store_goods)

    assert os.path.isdir(input_dir)
    if input_dir[-1] != '/':
        input_dir += '/'
    
    CHECKERBOARD = (args.rows -1 ,args.cols -1)
    subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

    #calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_FIX_SKEW


    objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * 25
    _img_shape = None
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    
    images = glob.glob('{}*.png'.format(input_dir))

    if save_good_imgs:
        good_imgs_dir = input_dir + "good_images/"
        if not os.path.exists(good_imgs_dir):
            os.mkdir(good_imgs_dir)

    print("Extracting vertex from images..")
    for fname in tqdm(sorted(images)):
        img = cv2.imread(fname)
        if _img_shape == None:
            _img_shape = img.shape[:2]
        else:
            assert _img_shape == img.shape[:2], "All images must share the same size."
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
            #print("{} OK".format(fname))
            if save_good_imgs:
                rel_name = os.path.basename(fname)
                shutil.copyfile("{}{}".format(input_dir,rel_name), "{}{}".format(good_imgs_dir,rel_name))
            imgpoints.append(corners)

    print("Elaborating data..")
    N_OK = len(objpoints)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    rms, _, _, _, _ = \
        cv2.fisheye.calibrate(
            objpoints,
            imgpoints,
            gray.shape[::-1],
            K,
            D,
            rvecs,
            tvecs,
            calibration_flags,
            (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )
    print("Found " + str(N_OK) + " valid images for calibration")
    print("DIM=" + str(_img_shape[::-1]))
    print("K=np.array(" + str(K.tolist()) + ")")
    print("D=np.array(" + str(D.tolist()) + ")")



if __name__ == "__main__":
    main()