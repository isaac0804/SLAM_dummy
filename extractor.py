import cv2
import numpy as np

np.set_printoptions(suppress=True)
from skimage.measure import ransac
from skimage.transform import EssentialMatrixTransform, FundamentalMatrixTransform


# turn x=[u,v] to x=[u,v,1]
def add_ones(x):
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)


class Extractor(object):
    def __init__(self, k):
        self.orb = cv2.ORB_create(1000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.last = None
        self.k = k
        self.kinv = np.linalg.inv(self.k)

    def normalize(self, pts):
        return np.dot(self.kinv, add_ones(pts).T).T[:, 0:2]

    def denormalize(self, pt):
        ret = np.dot(self.k, [pt[0], pt[1], 1.0])
        ret /= ret[2]
        # print(ret)
        return int(round(ret[0])), int(round(ret[1]))

    def extract(self, image):
        # detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(image=gray, maxCorners=2000, qualityLevel=0.01, minDistance=3)

        # extraction
        kps = [cv2.KeyPoint(x=corner[0][0], y=corner[0][1], _size=20) for corner in corners]
        kps, des = self.orb.compute(image=image, keypoints=kps)

        # matching
        ret = []
        if self.last is not None:
            # knnMatch returns a list of k(=2) best matches
            matches = self.bf.knnMatch(des, self.last['des'], k=2)
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    kp1 = kps[m.queryIdx].pt
                    kp2 = self.last['kps'][m.trainIdx].pt
                    ret.append((kp1, kp2))

        # filtering
        if len(ret) > 0:
            ret = np.array(ret)

            # normalize
            ret[:, 0, :] = self.normalize(ret[:, 0, :])
            ret[:, 1, :] = self.normalize(ret[:, 1, :])

            model, inliers = ransac((ret[:, 0], ret[:, 1]), EssentialMatrixTransform, min_samples=8,
                                    residual_threshold=0.005, max_trials=100)

            # s, v, d = np.linalg.svd(model.params)
            # print(v)
            R1, R2, t = cv2.decomposeEssentialMat(model.params)
            if np.sum(R1.diagonal()) > 0:
                R = R1
            else:
                R = R2
            print(f"Rotation: \n{R}")
            print(f"Translation: {t.T}")
            pose = np.concatenate([R, t], axis=1)
            print(f"Pose: \n{pose}")
            ret = ret[inliers]

        # return
        self.last = {"kps": kps, "des": des}
        return ret, pose


'''
# Harris Corner Detection
def hc(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    img[dst > 0.03 * dst.max()] = [0, 0, 255]
    cv2.imshow('frame', img)


# Shi-Tomasi Corner Detection
def stc(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, 1500, 0.01, 3)
    corners = np.int0(corners)
    for i in corners:
        x, y = i.ravel()
        img[y - 1:y + 1, x - 1:x + 1] = [0, 0, 255]
    cv2.imshow('frame', img)


# ORB (Oriented FAST and Rotated BRIEF)
def orb_method(img):
    orb = cv2.ORB_create()
    kp = orb.detect(img, None)
    kp, des = orb.compute(img, kp)
    img = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)
    cv2.imshow('frame', img)
'''
