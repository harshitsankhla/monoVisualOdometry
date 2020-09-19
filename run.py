#! /usr/bin/env python3

import os
import cv2
import argparse

import vision
import helpers

# basic arguments and parsing
parser = argparse.ArgumentParser()
parser.add_argument('-k', required=True, type=str, help='Camera Intrinsic Matrix .txt file')
parser.add_argument('-i', required=True, type=str, help='Images directory for VO Estimation')
parser.add_argument('--feature_type', type=str, default='FAST', help='Feature descriptor format (FAST, SIFT, SURF)')
parser.add_argument('--image_format', type=str, default='.png', help='Images format (.png, .jpg, .jpeg, etc)')
args = parser.parse_args()

# Read in the Intrinsic Matrix from path
K = helpers.get_K_from_txt(args.k)

# Read image folder and get image sequence
try:
    print("Listing images from -> %s" % args.i)
    images = sorted(os.listdir(args.i))
    if not all([args.image_format in x for x in images]):
        raise ValueError("Image directory has format inconsistency")
    print("Sequence contains %d Images !" % len(images))
except Exception as e:
    raise ValueError("Image listing failed")

# File to save poses to
file = open('estimated_poses.txt', 'w')

# Start Visual Odometry Estimation
vo = vision.VisualOdometry(args.feature_type, K)

# Initialize VO
image1 = cv2.imread(os.path.join(args.i, images[0]))
image2 = cv2.imread(os.path.join(args.i, images[1]))
vo.initialize(image1, image2)
helpers.write_pose_to_file(file, vo.R, vo.T)

# Loop through the remaining images
for i in range(2, len(images)):
    print("Processing frame %d" % i)
    image = cv2.imread(os.path.join(args.i, images[i]))
    vo.nextFrame(image)
    helpers.write_pose_to_file(file, vo.R, vo.T)

print("Done !")



