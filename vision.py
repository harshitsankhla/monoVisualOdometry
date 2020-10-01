import cv2
import numpy as np


class FeatureDetector:
    def __init__(self, detector):
        self.type = detector
        if self.type == 'SIFT':
            self.detector = cv2.xfeatures2d.SIFT_create(nfeatures=1000)
            self.tag = 'SIFTed'
        elif self.type == 'SURF':
            self.detector = cv2.xfeatures2d.SURF_create()
            self.tag = 'SURFed'
        elif self.type == 'FAST':
            self.detector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
            self.tag = 'FASTed'
        elif self.type == 'ORB':
            self.detector = cv2.ORB_create(nfeatures=1000)
            self.tag = 'ORBed'
        else:
            raise ValueError("Invalid Feature Type !")

    def getKeypoints(self, img):
        """Use a feature detector to get keypoints in an image

        :param img: image to detect keypoints in
        :return: detected keypoints as array of points
        """
        points2f = self.detector.detect(img, None)
        keypoints = cv2.KeyPoint_convert(points2f)
        return keypoints


class FeatureTracker:
    def __init__(self):
        self.lk_params = dict(winSize=(21, 21), criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    def trackPoints(self, prevImage, curImage, prevPoints):
        """Track keypoints in previous to current image

        :param prevImage: previous image (track from)
        :param curImage: current image (track to)
        :param prevPoints: keypoints to track in previous image
        :return: refined points that are tracked between both images
        """
        curPoints, status, error = cv2.calcOpticalFlowPyrLK(prevImage, curImage, prevPoints, None, **self.lk_params)
        status = status.reshape(status.shape[0])
        prevPoints = prevPoints[status == 1]
        curPoints = curPoints[status == 1]
        return prevPoints, curPoints


class VisualOdometry:
    def __init__(self, featureType, K, scale=None, gt_poses_file=None):
        self.R = 0
        self.T = 0
        self.K = K
        self.scale = scale
        self.lastImage = 0
        self.lastPoints = 0
        self.frameNumber = 0
        self.numKeypoints = 0
        self.featureDetector = FeatureDetector(featureType)
        self.featureTracker = FeatureTracker()

        if scale:
            assert gt_poses_file is not None, "Must provide Ground Truth Poses if True Scale VO Required"
            with open(gt_poses_file) as file:
                self.gt_poses = file.readlines()

    def getScale(self, frameNumber):
        """calculate scale from ground truth poses provided

        :param frameNumber: current frame we are processing
        :return:
        """
        ss = self.gt_poses[frameNumber - 1].strip().split()
        x_prev = float(ss[3])
        y_prev = float(ss[7])
        z_prev = float(ss[11])
        ss = self.gt_poses[frameNumber].strip().split()
        x = float(ss[3])
        y = float(ss[7])
        z = float(ss[11])
        return np.sqrt((x - x_prev)**2 + (y - y_prev)**2 + (z - z_prev)**2)

    def initialize(self, image1, image2):
        """Initialize VO System with first 2 frames

        For initialization, get keypoints from first frame and track them in the second
        then calculate Essential Matrix then pose between the images and set our initial
        otation and translation to be the relative pose from image 2 to 1

        :param image1: first image of the sequence
        :param image2: second image of the sequence
        """
        kp1 = self.featureDetector.getKeypoints(image1)
        self.lastPoints, curPoints = self.featureTracker.trackPoints(image1, image2, kp1)
        E, mask = cv2.findEssentialMat(curPoints, self.lastPoints, self.K, method=cv2.RANSAC, prob=0.99, threshold=0.999)
        _, self.R, self.T, mask = cv2.recoverPose(E, curPoints, self.lastPoints, self.K)

        self.frameNumber = 2
        self.lastImage = image2
        self.lastPoints = curPoints
        self.numKeypoints = curPoints.shape[0]

        print("Successfully initialized VO with first 2 frames !")

    def nextFrame(self, image):
        """Process next image in sequence

        Run our visual odometry pipeline on the image sequence by
        first tracking points in the new image then recovering the pose

        :param image: next image in VO sequence
        """
        # if we have less than 500 keypoints left after the last processed frame, then find new ones
        if self.numKeypoints < 1500:
            print('Re-detecting Keypoints')
            self.lastPoints = self.featureDetector.getKeypoints(self.lastImage)

        self.lastPoints, curPoints = self.featureTracker.trackPoints(self.lastImage, image, self.lastPoints)
        E, mask = cv2.findEssentialMat(curPoints, self.lastPoints, self.K, method=cv2.RANSAC, prob=0.99, threshold=0.999)
        _, R, T, mask = cv2.recoverPose(E, curPoints, self.lastPoints, self.K)

        # pose update
        if self.scale:
            self.T = self.T + (self.getScale(self.frameNumber) * self.R.dot(T))
        else:
            self.T = self.T + self.R.dot(T)
        self.R = R.dot(self.R)

        self.lastImage = image
        self.lastPoints = curPoints
        self.numKeypoints = curPoints.shape[0]
        self.frameNumber = self.frameNumber + 1
