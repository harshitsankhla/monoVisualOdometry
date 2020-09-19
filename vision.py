import cv2


class FeatureDetector:
    def __init__(self, detector):
        self.type = detector
        if self.type == 'SIFT':
            self.detector = cv2.xfeatures2d.SIFT_create(4000)
            self.tag = 'SIFTed'
        elif self.type == 'SURF':
            self.detector = cv2.xfeatures2d.SURF_create(4000)
            self.tag = 'SURFed'
        elif self.type == 'FAST':
            self.detector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
            self.tag = 'FASTed'
        else:
            raise ValueError("Invalid Feature Type !")

    def getFeatures(self, img):
        points2f = self.detector.detect(img, None)
        keypoints = cv2.KeyPoint_convert(points2f)
        return keypoints


class FeatureTracker:
    def __init__(self):
        self.lk_params = dict(winSize=(21, 21), criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    def trackPoints(self, prevImage, curImage, prevPoints):
        curPoints, status, error = cv2.calcOpticalFlowPyrLK(prevImage, curImage, prevPoints, None, **self.lk_params)
        status = status.reshape(status.shape[0])
        prevPoints = prevPoints[status == 1]
        curPoints = curPoints[status == 1]
        return prevPoints, curPoints


class VisualOdometry:
    def __init__(self, featureType, K):
        self.R = 0
        self.T = 0
        self.K = K
        self.lastImage = 0
        self.lastPoints = 0
        self.numFeatures = 0
        self.featureDetector = FeatureDetector(featureType)
        self.featureTracker = FeatureTracker()

    def initialize(self, image1, image2):
        """Initialize VO System with first 2 frames

        For initialization, get keypoints from first frame and track them in the second
        then calculate Essential Matrix then pose between the images and set our initial
        otation and translation to be the relative pose between image 2 and 1

        :param image1: first image of the sequence
        :param image2: second image of the sequence
        """
        kp1 = self.featureDetector.getFeatures(image1)
        self.lastPoints, curPoints = self.featureTracker.trackPoints(image1, image2, kp1)
        E, mask = cv2.findEssentialMat(self.lastPoints, curPoints, self.K, method=cv2.RANSAC, threshold=0.999)
        _, self.R, self.T, mask = cv2.recoverPose(E, self.lastPoints, curPoints, self.K)

        self.lastImage = image2
        self.lastPoints = curPoints
        self.numFeatures = curPoints.shape[0]

        print("Successfully initialized VO with first 2 frames !")

    def nextFrame(self, image):
        """Process next image in sequence

        Run our visual odometry pipeline on the image sequence by
        first tracking points in the new image then recovering the pose

        :param image: next image in VO sequence
        """
        # if we have less than 2000 features left after last processed frame, then find new features
        if self.numFeatures < 2000:
            self.lastPoints = self.featureDetector.getFeatures(self.lastImage)

        self.lastPoints, curPoints = self.featureTracker.trackPoints(self.lastImage, image, self.lastPoints)
        E, mask = cv2.findEssentialMat(self.lastPoints, curPoints, self.K, method=cv2.RANSAC, threshold=0.999)
        _, R, T, mask = cv2.recoverPose(E, self.lastPoints, curPoints, self.K)

        # pose update
        self.T = self.T + self.R.dot(T)
        self.R = R.dot(self.R)

        self.lastImage = image
        self.lastPoints = curPoints
        self.numFeatures = curPoints.shape[0]
