import sys
import cv2
import os
# import glob
import numpy as np
import vision
import helpers
import matplotlib.pyplot as plt

kitti = True
dataset = 'images'
npzFile_K = 'calibration_data.npz'

if kitti:
    K = np.array([[718.8560, 0.0, 607.1928], [0.0, 718.8560, 185.2157], [0.0, 0.0, 1.0]])
else:
    K = getKfromNPZ(npzFile_K)

detector = vision.featureDetector('FAST')
tracker  = vision.featureTracker()
pose     = np.hstack((np.eye(3, 3), np.zeros((3, 1))))

position_figure = plt.figure()
position_axes = position_figure.add_subplot(1, 1, 1)
position_axes.set_aspect('equal', adjustable='box')

# imageList = sorted(glob.glob(dataset+"/*.*"))
imageList = sorted((os.listdir(dataset)))
imageList = imageList[:100]

prev_img = cv2.imread(os.path.join(dataset, imageList[0]), cv2.IMREAD_GRAYSCALE)
prev_kp  = detector.getFeatures(prev_img)
for i in range(1,len(imageList),1):
    cur_img = cv2.imread(os.path.join(dataset, imageList[i]), cv2.IMREAD_GRAYSCALE)
    prev_kp, cur_kp  = tracker.trackPoints(prev_img, cur_img, prev_kp)
    E, mask = cv2.findEssentialMat(cur_kp, prev_kp, focal=K[0][0], pp = (K[0][2], K[1][2]),
                                    method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, mask = cv2.recoverPose(E, cur_kp, prev_kp, focal=K[0][0], pp = (K[0][2], K[1][2]))
    pose[:,0:3] = np.dot(R, pose[:,0:3])
    pose[:,3] = pose[:,3] + np.dot(t.reshape(1,3).flatten(),pose[:,0:3])
    position_axes.scatter(pose[:,3][0], pose[:,3][2])
    plt.pause(.01)
    # print(pose)
    prev_kp  = detector.getFeatures(cur_img)
