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
parser.add_argument('-scale', action='store_true', help='Enable Scale Consistent Odometry (Need to Provide GT Poses)')
parser.add_argument('--feature_type', type=str, default='FAST', help='Feature descriptor format (FAST, ORB, SIFT, SURF)')
parser.add_argument('--image_format', type=str, default='png', help='Images format (png, jpg, jpeg, etc)')
parser.add_argument('--result_file', type=str, default='estimated_poses.txt', help='Estimated poses txt file name')
parser.add_argument('--poses_file', type=str, default=None, help='Ground Truth Poses txt (for scale and visualization)')
parser.add_argument('--video', type=str, default=None, help='Video file to parse for VO images')
args = parser.parse_args()

# Load camera intrinsic matrix
K = helpers.get_K_from_txt(args.k)

# If video file is provided, parse it to images
if args.video:
    if os.path.exists(args.i):
        if len(os.listdir(args.i)) != 0:
            raise Exception('Non-empty directory exists, maybe video already processed ?')
    else:
        os.makedirs(args.i)
    helpers.get_images_from_video(args.video, args.i)

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
file = open(args.result_file, 'w')

# Start Visual Odometry Estimation
vo = vision.VisualOdometry(args.feature_type, K, args.scale, args.poses_file)

# Initialize VO with first 2 images
image1 = cv2.imread(os.path.join(args.i, images[0]), 0)
image2 = cv2.imread(os.path.join(args.i, images[1]), 0)
vo.initialize(image1, image2)
helpers.write_pose_to_file(file, vo.R, vo.T)

# Loop through the remaining images
for i in range(2, len(images)):
    print("Processing frame %d" % i)

    image = cv2.imread(os.path.join(args.i, images[i]), 0)
    vo.nextFrame(image)
    helpers.write_pose_to_file(file, vo.R, vo.T)

print("Done ! Poses saved to %s" % args.result_file)
