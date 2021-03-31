import cv2
import numpy as np

np.set_printoptions(suppress=True)

from skimage.transform import EssentialMatrixTransform
from skimage.transform import FundamentalMatrixTransform
from skimage.measure import ransac

IRt = np.eye(4)  # initial pose?

# turn x=[u,v] to x=[u,v,1]
def add_ones(x):
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)


def normalize(pts, kinv):
    return np.dot(kinv, add_ones(pts).T).T[:, 0:2]


def denormalize(pt, k):
    # Use Intrinsic matrix(k) to turn 3d points coordinates into 2d pixel coordinates
    ret = np.dot(k, [pt[0], pt[1], 1.0])
    ret /= ret[2]
    # print(ret)
    return int(round(ret[0])), int(round(ret[1]))


def poseRt(R, t):
    ret = np.eye(4)
    ret[:3, :3] = R
    ret[:3, 3] = t
    return ret


def extractRt(F):
    # extract transformation matrix from given matrix
    W = np.mat([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    U, d, Vt = np.linalg.svd(F)
    if np.linalg.det(U) < 0:
        U *= -1.0
    if np.linalg.det(Vt) < 0:
        Vt *= -1.0
    R = np.dot(np.dot(U, W), Vt)
    if np.sum(R.diagonal()) < 0:
        R = np.dot(np.dot(U, W.T), Vt)
    t = U[:, 2]
    return poseRt(R, t)


def extract(image):
    orb = cv2.ORB_create()
    # detection
    corners = cv2.goodFeaturesToTrack(image=image, maxCorners=3000, qualityLevel=0.01, minDistance=5)
    # extraction
    kps = [cv2.KeyPoint(x=corner[0][0], y=corner[0][1], _size=20) for corner in corners]
    kps, des = orb.compute(image=image, keypoints=kps)
    return np.array([(kp.pt[0], kp.pt[1]) for kp in kps]), des


def match_frames(f1, f2):
    # match similar points between two frames f1(current) and f2(last frame) then filter 
    # return paired points and transformation matrix
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    ret = []  # used for ransac
    idx1, idx2 = [], []
    # knnMatch returns a list of k(=2) best matches
    matches = bf.knnMatch(f1.des, f2.des, k=2)
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            p1 = f1.kps[m.queryIdx]
            p2 = f2.kps[m.trainIdx]
            # keep the points with travel distance < 10% diagonal and be within orb distance 32
            if np.linalg.norm((p1-p2)) < 0.1*np.linalg.norm([f1.w, f1.h]) and m.distance < 32:
                # TODO: refactor this to not be O(N^2)
                if m.queryIdx not in idx1 and m.trainIdx not in idx2:
                    # Keep the indices
                    idx1.append(m.queryIdx)
                    idx2.append(m.trainIdx)
                    ret.append((p1, p2))
    # no duplicates
    assert (len(set(idx1)) == len(idx1))
    assert (len(set(idx2)) == len(idx2))
    assert len(ret) >= 8
    ret = np.array(ret)
    idx1 = np.array(idx1)
    idx2 = np.array(idx2)
    model, inliers = ransac((ret[:, 0], ret[:, 1]),
                            # EssentialMatrixTransform,
                            FundamentalMatrixTransform,
                            min_samples=8,
                            residual_threshold=0.005,
                            max_trials=100)
    print("Matches: %d -> %d -> %d -> %d"% (len(f1.des), len(matches), len(inliers), sum(inliers)))
    # print(model.params)
    Rt = extractRt(model.params)
    return idx1[inliers], idx2[inliers], Rt


class Frame(object):
    def __init__(self, mapp, image, k):
        self.k = k
        self.kinv = np.linalg.inv(self.k)
        self.h, self.w = image.shape[0:2]
        self.pose = np.eye(4)
        self.kpus, self.des = extract(image)
        self.kps = normalize(self.kpus, self.kinv)
        self.pts = [None]*len(self.kps)
        self.id = len(mapp.frames)
        mapp.frames.append(self)
