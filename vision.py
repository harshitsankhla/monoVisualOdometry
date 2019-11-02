import cv2

class featureDetector:
    def __init__(self, detector):
        self.type = detector
        if self.type == 'SIFT':
            self.detector = cv2.xfeatures2d.SIFT_create(00)
            self.tag = 'SIFTed'
        elif self.type == 'SURF':
            self.detector = cv2.xfeatures2d.SURF_create(200)
            self.tag = 'SURFed'
        elif self.type == 'FAST':
            self.detector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
            self.tag = 'FASTed'

    def getFeatures(self,img):
        points2f = self.detector.detect(img, None)
        # points2f, descriptors = detector.detectAndCompute(img, None)
        # cv2.imshow(self.tag, cv2.drawKeypoints(img, points2f, None))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        keypoints = cv2.KeyPoint_convert(points2f)
        return keypoints


class featureTracker:
    def __init__(self):
        self.lk_params = dict(
                              winSize  = (21, 21),
                              criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
                             )

    def trackPoints(self, prevImage, curImage, prevPoints):
        curPoints, status, error = cv2.calcOpticalFlowPyrLK(prevImage, curImage,
                                                            prevPoints, None, **self.lk_params)
        status = status.reshape(status.shape[0])
        prevPoints = prevPoints[status==1]
        curPoints  = curPoints[status==1]
        return prevPoints, curPoints
